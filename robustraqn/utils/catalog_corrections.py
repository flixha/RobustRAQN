
# %%
import logging
Logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(name)40s:%(lineno)s\t%(funcName)20s()\t%(levelname)s\t%(message)s")
"[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
# Logger.info('Start module import')

from importlib import reload
import glob
import numpy as np
from collections import defaultdict
from orderedset import OrderedSet
from urllib.error import HTTPError
from re import A
from math import sqrt
from obspy import read_events, Catalog

from obspy.core.event import (
    Event, Origin, Magnitude, StationMagnitude, Catalog, EventDescription,
    CreationInfo, OriginQuality, OriginUncertainty, Pick, WaveformStreamID,
    Arrival, Amplitude, FocalMechanism, MomentTensor, NodalPlane, NodalPlanes,
    QuantityError, Tensor, ResourceIdentifier, Comment)
from obspy.core.event.magnitude import Amplitude
from obspy.core.utcdatetime import UTCDateTime
from obspy.io.nordic.core import write_select, _write_nordic, read_nordic
from obspy.io.nordic.utils import (
    _km_to_deg_lat, _km_to_deg_lon, _str_conv, _float_conv, _is_iasp_ampl_phase)
from obspy.io.nordic.ellipse import Ellipse
from obspy.geodetics import kilometers2degrees, degrees2kilometers, gps2dist_azimuth
from obspy.io.iaspei.core import _read_ims10_bulletin
from obsplus import events_to_df
from obsplus.events.validate import attach_all_resource_ids
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from timeit import default_timer
import pickle
# from obspy.clients.fdsn.client import Client
# client = Client('http://nnsn.geo.uib.no/intaros_eqcat/fdsnws/event')
# Client.get_events()

from joblib import Parallel, delayed, parallel_backend
from robustraqn.core.load_events import (
    get_all_relevant_stations, read_seisan_database)
from robustraqn.core.seismic_array import (
    _check_extra_info, SEISARRAY_REF_EQUIVALENT_STATIONS)
from robustraqn.utils.bayesloc import read_bayesloc_events

INV_SEISARRAY_REF_EQUIVALENT_STATIONS = {
    val: key for key, val in SEISARRAY_REF_EQUIVALENT_STATIONS.items()}
INV_SEISARRAY_REF_EQUIVALENT_STATIONS = defaultdict(
    str, INV_SEISARRAY_REF_EQUIVALENT_STATIONS.items())
SEISARRAY_REF_EQUIVALENT_STATIONS = (
    SEISARRAY_REF_EQUIVALENT_STATIONS | INV_SEISARRAY_REF_EQUIVALENT_STATIONS)


def get_events_from_url(
        starttime, endtime, minlatitude, maxlatitude, minlongitude,
        maxlongitude, write=False, nordic_format=True,
        base_url='http://nnsn.geo.uib.no/intaros_eqcat/fdsnws/event/1/query?'):
    """
    """
    cat_base_url = (
        base_url +
        'starttime={starttime}&endtime={endtime}'
        '&minlatitude={minlatitude}&maxlatitude={maxlatitude}'
        '&minlongitude={minlongitude}&maxlongitude={maxlongitude}'
        '&phases=true&focalmechanism=true')
    get_url = cat_base_url.format(
        starttime=starttime, endtime=endtime,
        minlatitude=minlatitude, maxlatitude=maxlatitude,
        minlongitude=minlongitude, maxlongitude=maxlongitude)
    if 'nnsn' in base_url and nordic_format:
        get_url = get_url + '&format=nordic'
    try:
        cat = read_events(get_url, nordic_format='NEW')
    except Exception as e:
        Logger.error('Could not download catalog for %s- %s', starttime,
                     endtime)
        Logger.error(e)
        return Catalog(), get_url
    cat.events = sorted(
        cat.events, key=lambda d: d.preferred_origin().time)
    # Remove magnitudes without an actual magnitude
    for event in cat:
        for magnitude in event.magnitudes.copy():
            if magnitude.mag is None:
                event.magnitudes.remove(magnitude)
    if write:
        cat.write(str(starttime)[0:7] + '.qml', format='QUAKEML')
    return cat, get_url


def _read_events(cat_file, format='QUAKEML'):
    """
    Read_events-wrapper with a bit of logging.
    """
    Logger.info('Reading event-file %s', cat_file)
    outtic = default_timer()
    cat = read_events(cat_file, format=format)
    outtoc = default_timer()
    Logger.info('Reading event-file %s took %s s', cat_file,
                str(outtoc - outtic))
    return cat


def check_resource_id_linking(cats, cat_files):
    for j, cat in enumerate(cats):
        ref_orig = cat[0].preferred_origin()
        if ref_orig is None:
            Logger.warning(cat_files[j] + ': cannot find referred origin-'
                           'object for first event')


def _merge_picks(event, pick, other_picks):
    """
    merge information in picks into one pick. first pick's information has
    precedence
    """
    new_pick = Pick()
    for key, value in zip(list(pick.keys()), list(pick.values())):
        if key in ['resource_id', 'extra']:
            continue
        new_pick[key] = value

    for opick in other_picks:
        for key, value in zip(list(opick.keys()), list(opick.values())):
            if key in ['resource_id']:
                continue
            if hasattr(new_pick, key):
                if (new_pick[key] is None or new_pick[key] == '' and
                        value is not None):
                    new_pick[key] = value
            else:
                new_pick[key] = value
    
    # Update links in:
    # Amplitudes
    old_pick_ids = [p.resource_id for p in other_picks]
    old_pick_ids.append(pick.resource_id)
    for amplitude in event.amplitudes:
        if amplitude.pick_id in old_pick_ids:
            amplitude.pick_id = new_pick.resource_id
    # Arrivals
    for origin in event.origins:
        for arrival in origin.arrivals:
            if arrival.pick_id in old_pick_ids:
                arrival.pick_id = new_pick.resource_id
    # Remove old picks and add new pick    
    event.picks.remove(pick)
    for p in other_picks:
        event.picks.remove(p)
    event.picks.append(new_pick)
    
    return event, new_pick


def _floor_decimal(number, decimal):
    """
    Floor-rounding where decimal precision can be selected, as it can be with
    "round".
    """
    return np.floor(number * pow(10, decimal))/pow(10, decimal)


def _pick_within_precision(pick1, pick2):
    """
    Check whether picks have the same time within 1 and 2 decimals, and
    consider that the decimals may have just been truncated.
    """
    is_within_precision = (
        pick1.time.ns / 1e9 == round(pick2.time.ns / 1e9, 0) or
        pick1.time.ns / 1e9 == round(pick2.time.ns / 1e9, 1) or
        pick1.time.ns / 1e9 == round(pick2.time.ns / 1e9, 2) or
        pick1.time.ns / 1e9 == _floor_decimal(pick2.time.ns / 1e9, 0) or
        pick1.time.ns / 1e9 == _floor_decimal(pick2.time.ns / 1e9, 1) or
        pick1.time.ns / 1e9 == _floor_decimal(pick2.time.ns / 1e9, 2))
    return is_within_precision


def _check_and_merge_picks(event, sfile_name='',
                           check_resource_id_linking=False):
    """
    check potentially similar picks and merge them if required
    """
    if not isinstance(event, Event):
        raise TypeError('event is not of type obspy.core.Event')
    if not isinstance(event.picks, list):
        raise TypeError('event.picks is not a list')
    if check_resource_id_linking:
        attach_all_resource_ids(event)
    # TODO: Check whether there is the same phase with "E" / "I" / "" in front:
    removed_picks = []
    for pick in event.picks.copy():
        if pick in removed_picks:
            continue
        if _is_iasp_ampl_phase(pick.phase_hint):
            continue
        similar_picks = [
            p for p in event.picks
            if p != pick and p.waveform_id == pick.waveform_id and
            p.time == pick.time and p.phase_hint == pick.phase_hint]
        if len(similar_picks) > 0:
            Logger.info(
                'Duplicate picks - '
                'Merging %s %s %s with %s similar picks (sfile %s)',
                pick.waveform_id.id, pick.time, pick.phase_hint,
                len(similar_picks), sfile_name)
            event, merged_pick = _merge_picks(event, pick, similar_picks)
            removed_picks.append(pick)
            removed_picks += similar_picks
        
    # TODO: Check whether there is P / Pn / Pg at the same time:
    removed_picks = []
    for pick in event.picks.copy():
        if pick in removed_picks:
            continue
        if _is_iasp_ampl_phase(pick.phase_hint):
            continue
        if not pick.phase_hint:
            continue
        if len(pick.phase_hint) < 2:
            continue
        similar_picks = [
            p for p in event.picks
            if (p != pick and p.waveform_id == pick.waveform_id and
                p.phase_hint == pick.phase_hint[0:-1] and
                p.time == pick.time)]
        if len(similar_picks) > 0:
            Logger.info(
                'Similar picks (but missing b/n/g) - '
                'Merging %s %s %s with %s similar picks (sfile %s)',
                pick.waveform_id.id, pick.time, pick.phase_hint,
                len(similar_picks), sfile_name)
            event, merged_pick = _merge_picks(event, pick, similar_picks)
            removed_picks.append(pick)
            removed_picks += similar_picks

    # TODO: check whether there is the same phase with higher precision pick
    #       (e.g., 28.120 instead of 28.100)
    removed_picks = []
    for pick in event.picks.copy():
        if not pick.phase_hint:
            continue
        if pick in removed_picks:
            continue
        if _is_iasp_ampl_phase(pick.phase_hint):
            continue
        similar_picks = [
            p for p in event.picks
            if (p != pick and p.waveform_id == pick.waveform_id and
                p.phase_hint == pick.phase_hint and
                _pick_within_precision(p, pick))]
        if len(similar_picks) > 0:
            Logger.info(
                'Similar picks within rounding precision - '
                'Merging %s %s %s with %s similar picks (sfile %s)',
                pick.waveform_id.id, pick.time, pick.phase_hint,
                len(similar_picks), sfile_name)
            event, merged_pick = _merge_picks(event, pick, similar_picks)
            removed_picks.append(pick)
            removed_picks += similar_picks

    # Check whether there is a pick for the same station with extra info on
    # location or network code (and allow different precision and phase name)
    removed_picks = []
    for pick in event.picks.copy():
        if not pick.phase_hint:
            continue
        if pick in removed_picks:
            continue
        if _is_iasp_ampl_phase(pick.phase_hint):
            continue
        if (pick.waveform_id.network_code == 'ZZ'
                and pick.waveform_id.location_code == 'ZZ'):
                    continue
        similar_picks = [
            p for p in event.picks
            if (p != pick
                and p.waveform_id.station_code == pick.waveform_id.station_code
                and p.waveform_id.location_code != 'ZZ'
                and p.waveform_id.network_code != 'ZZ'
                and p.phase_hint == pick.phase_hint
                and p.phase_hint == pick.phase_hint[0:-1]
                and _pick_within_precision(p, pick))]
        if len(similar_picks) > 0:
            Logger.info(
                'Similar picks for same station, similar phase hint and '
                'within rounding precision - '
                'Merging %s %s %s with %s similar picks (sfile %s)',
                pick.waveform_id.id, pick.time, pick.phase_hint,
                len(similar_picks), sfile_name)
            event, merged_pick = _merge_picks(event, pick, similar_picks)
            removed_picks.append(pick)
            removed_picks += similar_picks
    return event


def update_baz_appvel(event):
    """
    update picks for events from UiB web service such that a phase and its
    associated baz-pick are merged into one pick with backazimuth
    """
    remove_picks_list = list()
    for pick in event.picks:
        if not pick.phase_hint:
            continue
        if pick.phase_hint.startswith('BAZ-'):
            assoc_phase_hint = pick.phase_hint.removeprefix('BAZ-')
            assoc_phase_picks = [
                p for p in event.picks
                if (p.time == pick.time and
                    p.phase_hint == assoc_phase_hint and
                    p.waveform_id == pick.waveform_id)]
            # should only be one at most really
            for assoc_phase_pick in assoc_phase_picks:
                if assoc_phase_pick.backazimuth is None:
                    assoc_phase_pick.backazimuth = pick.backazimuth
                if assoc_phase_pick.horizontal_slowness is None:
                    assoc_phase_pick.horizontal_slowness = pick.horizontal_slowness
                remove_picks_list.append(pick)
    for rpick in remove_picks_list:
        event.picks.remove(rpick)

    return event



def add_NNSN_catalog_info(
        event, recursion_level=0, max_recursion_level=7,
        skip_event_with_existing_NNSN_info=False, add_extra_uib_picks=False,
        default_min_latitude=55, default_max_latitude=90,
        default_min_longitude=-25, default_max_longitude=45):
    """
    request similar event from Uib-webservice to add waveformfile-information
    """
    Logger.info('Starting on event %s.', event.short_str())
    problematic_picks = []
    if skip_event_with_existing_NNSN_info:
        problematic_picks = [pick for pick in event.picks
                             if (pick.phase_hint and
                                 '*' in pick.phase_hint and
                                 (pick.phase_hint != 'P*' or
                                  pick.phase_hint != 'S*'))]
        if len(problematic_picks) == 0:
            Logger.info('Skipping event %s - there are no problems.',
                        event.short_str())
            return event

    # Save time on events that were directly merged from NNSN catalog
    nnsn_comments = [comment for comment in event.comments
                     if comment and comment.text and
                     comment.text.startswith('Merged from NNSN catalog')]
    if len(nnsn_comments) > 0:
        return event

    if recursion_level > max_recursion_level:
        return event
# 1996-08-20T04:59:26
    if recursion_level == 0:
        Logger.info('Working on event %s - there are problems with picks.',
                    event.short_str())
    # OR read_nordic
    rlfactor = (recursion_level + 1)
    ev_min_starttime = min([orig.time for orig in event.origins
                            if orig.time is not None]) - 3 * rlfactor
    ev_max_starttime = max([orig.time for orig in event.origins
                            if orig.time is not None]) + 3 * rlfactor
    try:
        ev_minlatitude = min([orig.latitude for orig in event.origins
                            if orig.latitude is not None]) - 0.5 * rlfactor
        ev_maxlatitude = max([orig.latitude for orig in event.origins
                            if orig.latitude is not None]) + 0.5 * rlfactor
    except ValueError:  # In case event has not latitude / no hypocenter
        ev_minlatitude = default_min_latitude
        ev_maxlatitude = default_max_latitude
    try:
        ev_minlongitude = min([orig.longitude for orig in event.origins
                            if orig.longitude is not None]) - 2 * rlfactor
        ev_maxlongitude = max([orig.longitude for orig in event.origins
                            if orig.longitude is not None]) + 2 * rlfactor
    except ValueError:    # In case event has not longitude / no hypocenter
        ev_minlongitude = default_min_longitude
        ev_maxlongitude = default_max_longitude

    namespace_url = 'https://seis.geus.net/software/seisan/node239.html'
    wavfiles = []
    Logger.info('Requesting UiB event web-service')
    uib_cat, request_url = get_events_from_url(
        ev_min_starttime, ev_max_starttime, ev_minlatitude, ev_maxlatitude,
        ev_minlongitude, ev_maxlongitude, write=False,
        base_url='http://rick.geo.uib.no/nnsn_eqcat/fdsnws/event/1/query?')
    if len(uib_cat) == 0:
        Logger.warning(
            'Could not find any event in the UiB-database that matches request'
            ' %s, trying without lat/lon-limits', request_url)
        uib_cat, request_url = get_events_from_url(
            ev_min_starttime, ev_max_starttime, -99, 99, -999, 999,
            write=False,
            base_url='http://rick.geo.uib.no/nnsn_eqcat/fdsnws/event/1/query?')
    if len(uib_cat) == 0:
        Logger.warning(
            'Could not find any event in the UiB-database that matches request'
            ' %s, doing nothing.', request_url)
    elif len(uib_cat) > 1:
        Logger.warning(
            'Found more than one event in the UiB-database that matches '
            'request %s, doing nothing', request_url)
    elif len(uib_cat) == 1:
        # maybe do some check that they match?
        Logger.info('Found one matching event in the UiB-database for request')
        uib_event = uib_cat[0]
        
        # Merge pick+baz-pick into one pick when needed
        uib_event = update_baz_appvel(uib_event)
       
        # TODO HOW to handle that Norsar may have renamed picks after sending to NNSN?
        # E.g. Pn in INT, but Pg in NNSN, with same time
        # --> Ignore, trust the Intaros/Eurarc/ISC catalog
        
        # Check if there's extra picks in UiB catalog
        # ev_picks_tuples = [
        #     (pick.phase_hint.upper(), pick.time, pick.waveform_id.station_code)
        #     for pick in event.picks if pick.phase_hint]
        ev_picks_tuples = list()
        for pick in event.picks:
            if not pick.phase_hint:
                continue
            if pick.phase_hint:
                ev_picks_tuples.append((pick.phase_hint[0].upper(), pick.time,
                                        pick.waveform_id.station_code))
                array_equi_station = SEISARRAY_REF_EQUIVALENT_STATIONS[
                    pick.waveform_id.station_code]
                if array_equi_station:
                    ev_picks_tuples.append((pick.phase_hint[0].upper(),
                                            pick.time, array_equi_station))
                
        uib_picks_tuples = [(pick.phase_hint[0].upper(),
                             pick.time, pick.waveform_id.station_code)
            for pick in uib_event.picks if pick.phase_hint]
        for upick in uib_event.picks:
            if not upick.phase_hint:
                continue
            upick_tuple = (upick.phase_hint[0].upper(), upick.time,
                            upick.waveform_id.station_code)
            if upick_tuple in ev_picks_tuples:
                # Make sure to update missing information, e.g. on:
                # polarities
                matching_picks = [
                    p for p in event.picks
                    if (p.phase_hint and
                        p.phase_hint[0] == upick.phase_hint[0] and
                        p.time == upick.time and
                        p.waveform_id.station_code == upick.waveform_id.station_code)]
                for matching_pick in matching_picks:
                    if not matching_pick.polarity and upick.polarity:
                        matching_pick.polarity = upick.polarity
                    if not matching_pick.onset and upick.onset:
                        matching_pick.onset = upick.onset
            elif add_extra_uib_picks:  # (upick_tuple not in ev_picks_tuples):
                # Additional checks: see if there's 
                #   - P for Pn / Pg  or S for Sn / Sg
                #   - Pn for PN, Pg for PG etc.
                # How to handle small time differences??
                # E.g., ISC catalog may have cut 2nd decimal in pick time.
                # for older events
                # Check whether there's any events with same phase hint within
                # 1.5 s. If there is one pick that fulfills criteria, use
                # that to correct the pick. If there's more than 1, don't.
                if upick.phase_hint.startswith('BAZ-'):
                    baz_phase = (
                        upick.phase_hint.removeprefix('BAZ-')[0].upper())
                    upbaz_tuple = (baz_phase, upick.time,
                                    upick.waveform_id.station_code)
                    if upbaz_tuple in ev_picks_tuples:
                        matchin_baz_picks = [
                            p for p in event.picks
                            if (p.phase_hint and
                                p.phase_hint[0] == baz_phase and
                                p.time == upick.time and
                                p.waveform_id.station_code == upick.waveform_id.station_code)]
                        for mbaz_pick in matchin_baz_picks:
                            if mbaz_pick.backazimuth is None:
                                mbaz_pick.backazimuth = upick.backazimuth
                            if mbaz_pick.horizontal_slowness is None:
                                mbaz_pick.horizontal_slowness = upick.horizontal_slowness
                        continue
                    equi_station = SEISARRAY_REF_EQUIVALENT_STATIONS[
                        upick.waveform_id.station_code]
                    upbaz_tuple = (baz_phase, upick.time, equi_station)
                    if upbaz_tuple in ev_picks_tuples:
                        matchin_baz_picks = [
                            p for p in event.picks
                            if (p.phase_hint and
                                p.phase_hint[0] == baz_phase and
                                p.time == upick.time and
                                p.waveform_id.station_code == equi_station)]
                        for mbaz_pick in matchin_baz_picks:
                            if mbaz_pick.backazimuth is None:
                                mbaz_pick.backazimuth = upick.backazimuth
                            if mbaz_pick.horizontal_slowness is None:
                                mbaz_pick.horizontal_slowness = upick.horizontal_slowness
                        continue
                closely_matching_picks = [
                    p for p in event.picks
                    if (p.phase_hint and upick.phase_hint
                        and p.phase_hint[0].upper() == upick.phase_hint[0].upper()
                        and p.waveform_id.station_code == upick.waveform_id.station_code
                        and abs(p.time - upick.time) < 1.5)]
                if len(closely_matching_picks) == 1:
                    # just correct the pick in the event
                    # TODO ARE YOU SURE?
                    pick = closely_matching_picks[0]
                    pick.time = upick.time
                    pick.phase_hint = upick.phase_hint
                    continue
                elif len(closely_matching_picks) == 0:
                    array_equi_station = SEISARRAY_REF_EQUIVALENT_STATIONS[
                        upick.waveform_id.station_code]
                    closely_matching_picks = [
                        p for p in event.picks
                        if (p.phase_hint and upick.phase_hint
                            and p.phase_hint[0].upper() == upick.phase_hint[0].upper()
                            and p.waveform_id.station_code == array_equi_station
                            and abs(p.time - upick.time) < 1.5)]
                else:
                    continue
                if len(closely_matching_picks) == 0:
                    # add NNSN-pick to event
                    event.picks.append(upick)
        amp_tuples = [
            (amp.type, amp.waveform_id.station_code, amp.generic_amplitude, amp.period)
            for amp in event.amplitudes]
        for uamp in uib_event.amplitudes:
            uamp_tuple = (uamp.type, uamp.waveform_id.station_code,
                          uamp.generic_amplitude, uamp.period)
            if uamp_tuple not in amp_tuples:
                # TODO: need to find the relevant pick for the amplitude
                ref_pick = uamp.pick_id.get_referred_object()
                new_ref_picks = [
                    p for p in event.picks
                    if (p.waveform_id.station_code == ref_pick.waveform_id.station_code
                        and p.phase_hint and ref_pick.phase_hint
                        and p.time == ref_pick.time
                        and p.phase_hint[0] == ref_pick.phase_hint[0])]
                if len(new_ref_picks) > 0:
                    Logger.info(
                        'Adding amplitude from UiB database for station %s, '
                        'found pick to reference amplitude to (event %s).',
                        ref_pick.waveform_id.station_code, event.short_str())
                    new_ref_pick = new_ref_picks[0]
                    new_amp = uamp.copy()
                    new_amp.pick_id = new_ref_pick.resource_id
                    event.amplitudes.append(new_amp)

        # CHECK if there's asterisk in phase hint:
        for pick in event.picks:
            if pick.phase_hint is not None and '*' in pick.phase_hint:
                # Previously I did this corretion incorrectly! - I just removed
                # the asterisk from the phase name. But:
                # P* is alternative for Pb: http://www.isc.ac.uk/standards/phases/
                # find and replace phase hint from nnsn catalog
                pick.phase_hint.replace('*', 'b')
                # if '*' in pick.phase_hint:
                #     shortened_phase_hint = pick.phase_hint.strip('*')
                # phase_hint_len = len(shortened_phase_hint)
                # matching_uib_picks = [
                #     p for p in uib_event.picks
                #     if pick.time == p.time and p.phase_hint and
                #     pick.waveform_id.station_code == p.waveform_id.station_code
                #     and shortened_phase_hint == p.phase_hint[0:phase_hint_len]]
                # if len(matching_uib_picks) == 1:
                #     matching_uib_pick = matching_uib_picks[0]
                #     pick.phase_hint = matching_uib_pick.phase_hint
                #     pick = _check_extra_info(pick, matching_uib_pick)
        
        # CHeck if there's parentheses in phase hint 
        for pick in event.picks:
            if not pick.phase_hint:
                continue
            if '(' in pick.phase_hint or ')' in pick.phase_hint:
                # find and replace phase hint from nnsn catalog
                shortened_phase_hint = pick.phase_hint.strip('()')
                phase_hint_len = len(shortened_phase_hint)
                matching_uib_picks = [
                    p for p in uib_event.picks
                    if pick.time == p.time and p.phase_hint and
                    pick.waveform_id.station_code == p.waveform_id.station_code
                    and shortened_phase_hint == p.phase_hint[0:phase_hint_len]]
                if len(matching_uib_picks) == 1:
                    matching_uib_pick = matching_uib_picks[0]
                    pick.phase_hint = matching_uib_pick.phase_hint
                    pick = _check_extra_info(pick, matching_uib_pick)
        # Check if UiB-event contains any extra info on Coda length
        coda_picks = [p for p in uib_event.picks if p.phase_hint == 'END']
        if coda_picks:
            for coda_pick in coda_picks:
                matching_cat_pick = [
                    p for p in event.picks
                    if coda_pick.time == p.time and
                    coda_pick.waveform_id.station_code == p.waveform_id.station_code
                    and coda_pick.phase_hint == p.phase_hint]
                if not matching_cat_pick:
                    Logger.info('Adding coda-pick for trace %s.',
                                coda_pick.waveform_id.id)
                    event.picks.append(coda_pick)

    previous_wavfiles = list()
    for comment in event.comments:
        if not comment or not comment.text:
            continue
        if comment.text[-2:] == ' 6':
            previous_wavfiles.append(comment.text.strip('   6').strip())
        if comment.text.startswith('Waveform-filename: '):
            previous_wavfiles.append(
                comment.text.removeprefix('Waveform-filename: '))

    for uevent in uib_cat:
        for comment in uevent.comments:
            if not comment or not comment.text:
                continue
            if comment.text[-2:] == ' 6':
                wavfiles.append(comment.text.strip('   6').strip())
            if comment.text[-2:] == ' 3' and 'NSN' in comment.text:
                wavfiles.append(comment.text.strip('   3').strip())

    # Try to find any extra waveform files nearby in time
    if (len(previous_wavfiles) < 2 and len(wavfiles) < 3 and 
            ev_min_starttime < UTCDateTime(2017,1,1,0,0,0) and
            (len(uib_cat) >= 1 or recursion_level == max_recursion_level)):
        wav_cat, wav_request_url = get_events_from_url(
            ev_min_starttime - 60, ev_max_starttime + 60, -90, 90, -180, 180,
            write=False,
            base_url='http://rick.geo.uib.no/nnsn_eqcat/fdsnws/event/1/query?')
        for wevent in wav_cat:
            for comment in wevent.comments:
                if comment.text[-2:] == ' 6':
                    wavfiles.append(comment.text.strip('   6').strip())
                if comment.text[-2:] == ' 3' and 'NSN' in comment.text:
                    wavfiles.append(comment.text.strip('   3').strip())

    for wavfile in wavfiles:
        if wavfile not in previous_wavfiles:
            new_comment = Comment(text='Waveform-filename: ' + wavfile)
            if new_comment not in event.comments:
                event.comments.append(new_comment)
                Logger.info('Added link to waveform file %s to event %s',
                            wavfile, event.short_str())
    # If there's no wavfiles, request events with less strict criteria
    if len(uib_cat) == 0:
        event = add_NNSN_catalog_info(
            event, recursion_level=recursion_level+1,
            add_extra_uib_picks=add_extra_uib_picks)
    return event



def check_ps_nbg_phase_hints(
        cat, stations_df, pn_group_velocity=(6.5, 8.2),
        sn_group_velocity=(4.0, 4.6), local_distance_cutoff=100,
        teleseismic_distance_cutoff=2000,
        check_s_minus_p_time=True, extract_len=None):
    # Make sure that picks are properly named P / Pn / Pb /  Pg / S / Sn / Sb / Sg
    # for each pick:
    for event in cat:
        remove_pick_list = []
        origin = event.preferred_origin() or event.origins[0]
        # pick_ttimes = {pick.id: pick.time - orig.time for pick in event.picks}
        # pick_distances = {pick: pick.distance for pick in event.picks}
        for pick in event.picks:
            # Only rename picks that do not have a specification yet
            if pick.phase_hint not in ['P', 'S']:
                continue
            stacode = pick.waveform_id.station_code
            sta_series = stations_df[stations_df.station == stacode].iloc[0]
            if origin.latitude is None or origin.longitude is None:
                continue
            pick_distance = gps2dist_azimuth(
                origin.latitude, origin.longitude,
                sta_series.latitude, sta_series.longitude)[0] / 1000.
            # Treat everything as Pg / Sg if it is closer than 100 km
            if pick_distance < local_distance_cutoff:
                pick.phase_hint = pick.phase_hint + 'g'
                continue
            # Do not change the phase hint if it is teleseismic
            if pick_distance > teleseismic_distance_cutoff:
                continue  # Do not change teleseismic picks
            traveltime = pick.time - origin.time
            if traveltime <= 0 or np.isnan(traveltime):
                continue
            app_vel = pick_distance / traveltime
            if pick.phase_hint == 'P':
                if (app_vel >= pn_group_velocity[0] and
                        app_vel <= pn_group_velocity[1]):
                    pick.phase_hint = 'Pn'
                elif app_vel < pn_group_velocity[0]:
                    pick.phase_hint = 'Pg'
            if pick.phase_hint == 'S':
                if (app_vel >= sn_group_velocity[0] and
                        app_vel <= sn_group_velocity[1]):
                    pick.phase_hint = 'Sn'
                elif app_vel < pn_group_velocity[0]:
                    pick.phase_hint = 'Sg'
            # Check S-P time vs window length:
            # - if the window is longer than the S-P time, then throw out all
            #   P-picks so that we don't compute P-differential time picks on
            #   S-waveforms.
            if check_s_minus_p_time:
                if pick.phase_hint.startswith('P'):
                    if origin.depth is None:
                        o_depth = origin.depth / 1000.
                    else:
                        o_depth = 10.
                    total_dist = np.sqrt(pick_distance ** 2 + o_depth ** 2)
                    theo_SmP_time = (total_dist / sn_group_velocity[0] -
                                            total_dist / pn_group_velocity[1])
                    s_picks = [pk for pk in event.picks
                            if pk.phase_hint.startswith('S') and
                            pk.waveform_id.station_code == stacode]
                    picked_SmP_time = np.nan
                    if len(s_picks) >= 0:
                        # Select earliest S-pick:
                        s_pick = s_picks[np.argmin([pk.time for pk in s_picks])]
                        picked_SmP_time = s_pick.time - pick.time
                    if ((picked_SmP_time is None and theo_SmP_time < extract_len)
                        or (picked_SmP_time is not None and
                            picked_SmP_time < extract_len)):
                        remove_pick_list += [pick.id]
        for rpick in remove_pick_list:
            event.picks.remove(event.picks[rpick])


def correct_asterisk_picks_from_intaros(
    event, int_df, max_euarc_error_km=100, s_diff=5):
    """
    """
    non_Pb_stations = ['JNE', 'JNW', 'JMI', 'JMIC', 'JMIM', 'HOPEN', 'BJO1']

    sfile_name = ''
    try:
        sfile_name = event.extra['sfile_name']['value']
    except AttributeError:
        pass
    
    int_event = None
    orig = event.preferred_origin()
    int_df['tdiff'] = abs(int_df.time - orig.time._get_datetime())
    min_idx = int_df['tdiff'].idxmin()
    int_ev_df = int_df.iloc[min_idx]
    try:
        hor_diff_m, _, _ = gps2dist_azimuth(
            int_ev_df.latitude, int_ev_df.longitude,
            orig.latitude, orig.longitude)
        hor_diff_km = hor_diff_m / 1000
    except (ValueError, TypeError):  # if lat / lon are None 
        hor_diff_km = max_euarc_error_km - 1
    if (int_ev_df.tdiff < np.timedelta64(s_diff, 's') and
            hor_diff_km < max_euarc_error_km):
        int_event = int_ev_df.events

    if int_event:
        Logger.info('Found matching event %s in INTAROS, checking picks.',
                    int_event.short_str())
        asterisk_picks = [
            pick for pick in int_event.picks if '*' in pick.phase_hint and
            pick.waveform_id.station_code not in non_Pb_stations]
        if len(asterisk_picks) == 0:
            return event
        for intpick in asterisk_picks:
            closely_matching_picks = [
                p for p in event.picks
                if (p.phase_hint and intpick.phase_hint
                    and p.phase_hint[0].upper() == intpick.phase_hint[0].upper()
                    and p.waveform_id.station_code == intpick.waveform_id.station_code
                    and p.time == intpick.time)]
            if len(closely_matching_picks) == 1:
                pick = closely_matching_picks[0]
                old_phase_hint = pick.phase_hint
                new_phase_hint = intpick.phase_hint.replace('*', 'b')
                if pick.phase_hint != new_phase_hint:
                    pick.phase_hint = new_phase_hint
                    Logger.info(
                        'Asterisk pick matched: Sfile %s: Changed pick for %5s '
                        'from %s to %s (event %s)', sfile_name,
                        pick.waveform_id.station_code,
                        old_phase_hint, new_phase_hint, event.short_str())
    return event


def merge_new_events_into_catalog(cat, new_cat, max_error_km=200, s_diff=5):
    """
    """
    cat_df = events_to_df(cat)
    cat_df['events'] = cat.events
    new_cat_df = events_to_df(new_cat)
    new_cat_df['events'] = new_cat.events
    
    for new_event in new_cat:
        orig = new_event.preferred_origin()
        existing_event = None
        cat_df['tdiff'] = abs(cat_df.time - orig.time._get_datetime())
        min_idx = cat_df['tdiff'].idxmin()
        existing_ev_df = cat_df.iloc[min_idx]
        try:
            hor_diff_m, _, _ = gps2dist_azimuth(
                existing_ev_df.latitude, existing_ev_df.longitude,
                orig.latitude, orig.longitude)
            hor_diff_km = hor_diff_m / 1000
        except (ValueError, TypeError):  # if lat / lon are None 
            hor_diff_km = max_error_km - 1
        if (existing_ev_df.tdiff < np.timedelta64(s_diff, 's') and
                hor_diff_km < max_error_km):
            existing_event = existing_ev_df.events

        if existing_event is None:
            Logger.info(
                'There was no event similar to %s in ECEAS yet, adding event.',
                new_event.short_str())
            cat.append(new_event)
        else:
            Logger.info(
                'There was already an event similar to %s in ECEAS: %s',
                new_event.short_str(), existing_event.short_str())
    return cat


def _check_event(event):
    """
    """
    attach_all_resource_ids(event)
    Logger.info('Checking event %s', event.short_str())
    sfile_name = ''
    try:
        sfile_name = event.extra['sfile_name']['value']
    except AttributeError:
        pass

    eu_event = None
    # select best suited origin
    origin = event.preferred_origin().copy()
    origin_agency_priorities = ['BER', 'ISC', 'INT', 'NAO', 'NEI', 'LDG', 'EID']
    mean_standard_error = np.mean(
            [orig.quality.standard_error for orig in event.origins
                if orig.quality is not None and
                orig.quality.standard_error is not None])
    for orig_prio in origin_agency_priorities:
        if (origin and origin.quality and origin.quality.standard_error
                and mean_standard_error):
            if origin.quality.standard_error > mean_standard_error:
                origin = None
            else:
                break
        if origin is None:
            origins = [orig for orig in event.origins
                        if orig.creation_info.agency_id == orig_prio]
            if len(origins) > 0:
                origin = origins[0]
    
    if not origin:
        origin = event.preferred_origin().copy()
    orig = origin
    lower_dtime = (orig.time - s_diff)._get_datetime()
    upper_dtime = (orig.time + s_diff)._get_datetime()
    # Fix slowness values to s/deg and check that picks are on right day
    # CHECK VALUES FOR SLOWNESS AND FIND OUT WHETHER THEY NEED CORRECTION
    # XXXXX
    ev_picks_tuples = list()
    for pick in event.picks:
        if not pick.phase_hint:
            continue
        if len(pick.phase_hint) > 1:
            ev_picks_tuples.append(
                (pick.phase_hint[0].upper(), pick.time,
                    pick.waveform_id.station_code))
        else:
            ev_picks_tuples.append(
                (pick.phase_hint, pick.time,
                    pick.waveform_id.station_code))
    # ev_picks_tuples = [
    #     (pick.phase_hint[0].upper(), pick.time, pick.waveform_id.station_code)
    #     for pick in event.picks]

    if check_asterisk_picks_from_intaros:
        correct_asterisk_picks_from_intaros(
            event, int_df=int_df, max_euarc_error_km=max_euarc_error_km,
            s_diff=s_diff)

    if check_pick_logic:
        Logger.info('Checking pick logic for sfile %s, event %s',
                    sfile_name, event.short_str())
        # Figure out picks that are really problematic:
        #   - S arriving before P,
        #   - picks with different names at same time
        #   -
        pick_tuples = [(p.waveform_id.station_code, p.phase_hint, p.time)
                    for p in event.picks]
        for pick in event.picks.copy():
            if not pick.phase_hint:
                continue
            pick_sta = pick.waveform_id.station_code
            if len(pick.phase_hint) > 0 and pick.phase_hint[0].upper() == 'S':
                # If there's the equivalent phase for P and S comes before P,
                # then remove the S-phase.
                equi_p_picks = [
                    p for p in event.picks
                    if p.waveform_id.station_code == pick_sta
                    and p.phase_hint == pick.phase_hint.replace('S', 'P').replace('s', 'p')
                    and p.time >= pick.time]
                for equi_p_pick in equi_p_picks:
                    Logger.warning(
                        'S-pick %s on %s arrives before P-pick %s for sfile %s',
                        pick.phase_hint, pick.waveform_id.id,
                        equi_p_pick.phase_hint, sfile_name)
                if len(equi_p_picks) > 0 and pick in event.picks:
                        event.picks.remove(pick)
                # Check whether any other S-phases come before P
                if pick.phase_hint in ['S', 'Sg', 'Sb', 'Sn', 'SS']:
                    later_p_picks = [
                        p for p in event.picks
                        if p.waveform_id.station_code == pick_sta
                        and p.phase_hint in ['P', 'Pg', 'Pb', 'Pn', 'PP']
                        and p.time >= pick.time]
                    for late_p_pick in later_p_picks:
                        Logger.warning(
                        'S-pick %s on %s arrives before P-pick %s for sfile %s',
                        pick.phase_hint, pick.waveform_id.id,
                        late_p_pick.phase_hint, sfile_name)
                    if len(later_p_picks) > 0 and pick in event.picks:
                        event.picks.remove(pick)

    # find minimum time difference
    if check_eurarc_cat:
        Logger.info('Checking EURARC catalog for sfile %s, event %s',
                    sfile_name, event.short_str())
        eu_df['tdiff'] = abs(eu_df.time - orig.time._get_datetime())
        min_idx = eu_df['tdiff'].idxmin()
        eu_ev_df = eu_df.iloc[min_idx]
        try:
            hor_diff_m, _, _ = gps2dist_azimuth(
                eu_ev_df.latitude, eu_ev_df.longitude,
                orig.latitude, orig.longitude)
            hor_diff_km = hor_diff_m / 1000
        except (ValueError, TypeError):  # if lat / lon are None 
            hor_diff_km = max_euarc_error_km - 1
        if (eu_ev_df.tdiff < np.timedelta64(s_diff, 's') and
                hor_diff_km < max_euarc_error_km):
            eu_event = eu_ev_df.events
    # TODO: check that I'm adding events from EURARC ca where I have not matches in main catalog
    if eu_event:
        Logger.info('Found matching event %s in EURARC, checking picks.',
                    eu_event.short_str())
        for eupick in eu_event.picks:
            if not eupick.comments:
                continue
            if not eupick.phase_hint:
                continue
            ctxt = eupick.comments[0].text
            slowness_str = 'observed slowness (seconds/degree): "'
            slowness_pos = int(ctxt.find(slowness_str) + len(slowness_str))
            slowness = _float_conv(ctxt[slowness_pos:slowness_pos + 6])

            # azimuth_str = 'observed azimuth (degrees): "'
            # azimuth_pos = int(ctxt.find(azimuth_str) + len(azimuth_str))
            # azimuth = _float_conv(ctxt[azimuth_pos:azimuth_pos + 5])
            # Checked the values and this is actually the baz in the comment,
            # not the azimuth. Read directly as BAZ from ISF file.
            baz_str = 'observed azimuth (degrees): "'
            baz_pos = int(ctxt.find(baz_str) + len(baz_str))
            baz = _float_conv(ctxt[baz_pos:baz_pos + 5])

            if slowness is None and baz is None:
                continue

            # Find ECEAS-pick and replace Slowness and BAZ
            if (eupick.phase_hint is None or len(eupick.phase_hint) == 0):
                continue
            eu_pick_tuple = (eupick.phase_hint[0].upper(), eupick.time,
                                eupick.waveform_id.station_code)
            if eu_pick_tuple in ev_picks_tuples:
                ev_pick_index = ev_picks_tuples.index(eu_pick_tuple)
                if slowness is not None:
                    Logger.info('Replacing slowness %s with %s', str(
                        event.picks[ev_pick_index].horizontal_slowness),
                        str(slowness))
                    event.picks[ev_pick_index].horizontal_slowness = (
                        slowness)
                if baz is not None:
                    # baz = (azimuth + 180) % 360 # it's actuall baz already
                    Logger.info('Replacing backazimuth %s with %s', str(
                        event.picks[ev_pick_index].backazimuth), str(baz))
                    event.picks[ev_pick_index].backazimuth = baz
            else:
                # First check for closely matching picks
                closely_matching_picks = [
                    p for p in event.picks
                    if (p.phase_hint and eupick.phase_hint
                        and p.phase_hint[0].upper() == eupick.phase_hint[0].upper()
                        and p.waveform_id.station_code == eupick.waveform_id.station_code
                        and abs(p.time - eupick.time) < 0.1)]
                if len(closely_matching_picks) == 1:
                    # just correct the pick in the event
                    # TODO ARE YOU SURE?
                    pick = closely_matching_picks[0]
                    pick.time = eupick.time
                    pick.phase_hint = eupick.phase_hint
                    if slowness is not None:
                        pick.horizontal_slowness = slowness
                    if baz is not None:
                        pick.backazimuth = baz
                # If pick not in event, then add it after all
                elif len(closely_matching_picks) == 0:
                    if slowness is not None:
                        eupick.horizontal_slowness = slowness
                    if baz is not None:
                        # baz = (azimuth + 180) % 360 # it's actuall baz already
                        eupick.backazimuth = baz
                    event.picks.append(eupick)
        # if pick.horizontal_slowness is not None:
        #     # Slowness probably in s/km instead of s/deg:
        #     if pick.horizontal_slowness > 100:
        #         pick.horizontal_slowness = kilometers2degrees(
        #             pick.horizontal_slowness)

    bayes_g_event = None
    if check_Gibbons_ridge_cat:
        Logger.info('Checking Gibbons/Bayes catalog for sfile %s, event %s',
                    sfile_name, event.short_str())
        bayes_g_df['tdiff'] = abs(bayes_g_df.time - orig.time._get_datetime())
        min_idx = bayes_g_df['tdiff'].idxmin()
        bayes_g_ev_df = bayes_g_df.iloc[min_idx]
        try:
            hor_diff_m, _, _ = gps2dist_azimuth(
                bayes_g_ev_df.latitude, bayes_g_ev_df.longitude,
                orig.latitude, orig.longitude)
            hor_diff_km = hor_diff_m / 1000
        except (ValueError, TypeError):  # if lat / lon are None 
            hor_diff_km = max_euarc_error_km - 1
        if (bayes_g_ev_df.tdiff < np.timedelta64(s_diff, 's') and
                hor_diff_km < max_euarc_error_km):
            bayes_g_event = bayes_g_ev_df.events
    if bayes_g_event:
        pick_tuples = [
            (p.waveform_id.station_code, p.phase_hint[0], p.time)
            for p in event.picks if p.phase_hint and len(p.phase_hint) > 0]
        for pick in bayes_g_event.picks:
            # translate picks back from e.g. P1 -> P
            if pick.phase_hint.endswith('1'):
                pick.phase_hint = pick.phase_hint[0:-1]
            # pick_tuple = (pick.waveform_id.station_code, pick.time,
            #               pick.phase_hint[0])
            # if pick_tuple not in pick_tuples:
            #     Logger.info(
            #         "SG pick %s %s %s not in event %s (sfile %s).",
            #         pick.waveform_id.station_code, pick.phase_hint,
            #         pick.time, event.short_str(), sfile_name)
            if pick.phase_hint is None or len(pick.phase_hint) == 0:
                continue
            similar_picks = [
                p for p in event.picks
                if (p != pick and len(p.phase_hint) > 0 and
                    p.waveform_id.station_code == pick.waveform_id.station_code and
                    p.phase_hint[0] == pick.phase_hint[0] and
                    (_pick_within_precision(p, pick) or
                        abs(p.time - pick.time) < max_gibbons_pick_difference))]
            if len(similar_picks) == 0:
                Logger.info(
                    "SG pick %s %s %s not in event %s (sfile %s).",
                    pick.waveform_id.station_code, pick.phase_hint,
                    pick.time, event.short_str(), sfile_name)
                event.picks.append(pick)


    # find minimum time difference
    if check_loki_2017_cat:
        Logger.info('Checking Lokis castle 2017-2018 catalog for sfile %s,'
                    'event %s', sfile_name, event.short_str())
        loki_df['tdiff'] = abs(loki_df.time - orig.time._get_datetime())
        min_idx = loki_df['tdiff'].idxmin()
        loki_ev_df = loki_df.iloc[min_idx]
        loki_event = None
        try:
            hor_diff_m, _, _ = gps2dist_azimuth(
                loki_ev_df.latitude, loki_ev_df.longitude,
                orig.latitude, orig.longitude)
            hor_diff_km = hor_diff_m / 1000
        except (ValueError, TypeError):  # if lat / lon are None 
            hor_diff_km = max_loki_error_km - 1
        if (loki_ev_df.tdiff < np.timedelta64(max_loki_s_diff, 's') and
                hor_diff_km < max_loki_error_km):
            loki_event = loki_ev_df.events
        if loki_event:
            Logger.info(
                'Found matching event %s in LOKIS castle 2017-2018 catalog, '
                ' adding all picks / amplitudes / magnitudes.',
                loki_event.short_str())
            for lpick in loki_event.picks:
                event.picks.append(lpick)
            for lamp in loki_event.amplitudes:
                event.amplitudes.append(lamp)
            for lmag in loki_event.magnitudes:
                event.magnitudes.append(lmag)
            for orig in loki_event.origins:
                if orig.creation_info and orig.creation_info.agency_id:
                    orig.creation_info.agency_id == 'LOK'
                else:
                    orig.creation_info = CreationInfo(agency_id='LOK')
                event.origins.append(orig)

    if check_loki_2019_cat:
        Logger.info('Checking Lokis castle 2019-2020 catalog for sfile %s,'
                    'event %s', sfile_name, event.short_str())
        loki2_df['tdiff'] = abs(loki2_df.time - orig.time._get_datetime())
        min_idx = loki2_df['tdiff'].idxmin()
        loki2_ev_df = loki2_df.iloc[min_idx]
        loki2_event = None
        try:
            hor_diff_m, _, _ = gps2dist_azimuth(
                loki2_ev_df.latitude, loki2_ev_df.longitude,
                orig.latitude, orig.longitude)
            hor_diff_km = hor_diff_m / 1000
        except (ValueError, TypeError):  # if lat / lon are None 
            hor_diff_km = max_loki_error_km - 1
        if (loki2_ev_df.tdiff < np.timedelta64(max_loki_s_diff, 's') and
                hor_diff_km < max_loki_error_km):
            loki2_event = loki2_ev_df.events
        if loki2_event:
            Logger.info(
                'Found matching event %s in LOKIS castle 2019-2020 catalog, '
                ' adding all picks / amplitudes / magnitudes.',
                loki2_event.short_str())
            for lpick in loki2_event.picks:
                event.picks.append(lpick)
            for lamp in loki2_event.amplitudes:
                event.amplitudes.append(lamp)
            for lmag in loki2_event.magnitudes:
                event.magnitudes.append(lmag)
            for orig in loki2_event.origins:
                if orig.creation_info and orig.creation_info.agency_id:
                    orig.creation_info.agency_id == 'LOK'
                else:
                    orig.creation_info = CreationInfo(agency_id='LOK')
                event.origins.append(orig)
    
    if check_storfjord_2019:
        Logger.info('Checking Storfjorden 2019-2020 catalog for sfile %s,'
                    'event %s', sfile_name, event.short_str())
        stor_df['tdiff'] = abs(stor_df.time - orig.time._get_datetime())
        min_idx = stor_df['tdiff'].idxmin()
        stor_ev_df = stor_df.iloc[min_idx]
        stor_event = None
        try:
            hor_diff_m, _, _ = gps2dist_azimuth(
                stor_ev_df.latitude, stor_ev_df.longitude,
                orig.latitude, orig.longitude)
            hor_diff_km = hor_diff_m / 1000
        except (ValueError, TypeError):  # if lat / lon are None 
            hor_diff_km = max_loki_error_km - 1
        if (stor_ev_df.tdiff < np.timedelta64(max_loki_s_diff, 's') and
                hor_diff_km < max_loki_error_km):
            stor_event = stor_ev_df.events
        if stor_event:
            Logger.info(
                'Found matching event %s in Storfjorden catalog, '
                ' adding all picks / amplitudes / magnitudes.',
                stor_event.short_str())
            for lpick in stor_event.picks:
                Logger.info('Check pick %s %s', lpick.waveform_id.id,
                            lpick.phase_hint)
                if lpick.waveform_id.station_code.startswith('STOR'):
                    event.picks.append(lpick)
            for lamp in stor_event.amplitudes:
                if lamp.waveform_id.station_code.startswith('STOR'):
                    event.amplitudes.append(lamp)


    if check_pick_syntax:
        Logger.info('Checking phase syntax for sfile %s, event %s',
                    sfile_name, event.short_str())
        pref_orig = event.preferred_origin() or event.origins[0]
        for pick in event.picks:
            if not pick.phase_hint:
                continue
            # CHECK FOR PARENTHESES IN PHASE NAMES
            if '(' in pick.phase_hint or ')' in pick.phase_hint:
                pick.phase_hint = pick.phase_hint.strip('()')
                nordic_weight = None
                if 'extra' in pick.keys():
                    try:
                        nordic_weight = pick.extra.nordic_pick_weight.value
                    except (KeyError or AttributeError):
                        pass
                if nordic_weight is not None:
                    namespace_url = (
                        'https://seis.geus.net/software/seisan/node239.html')
                    pick.extra = {'nordic_pick_weight': {
                        'value': '3', 'namespace': namespace_url}}

    if check_pick_duplication:
        Logger.info('Checking pick duplication for sfile %s, event %s',
                    sfile_name, event.short_str())
        event = _check_and_merge_picks(event, sfile_name=sfile_name)

    if check_station_specifics:
        # Check station names for OBS stations - 2007/2008 IPY stations 
        # should be renmaed (I don't have locations right now and the name
        # conflicts with 2018-2019 deployment)
        for pick in event.picks:
            if (pick.time.year >= 2007 and pick.time.year <= 2008 and
                    pick.waveform_id.station_code in [
                    'OBS01', 'OBS02', 'OBS05', 'OBS06', 'OBS07', 'OBS08',
                    'OBS09', 'OBS11']):
                pick.waveform_id.station_code = (
                    pick.waveform_id.station_code[0:2] + 'Y' +
                    pick.waveform_id.station_code[3:5])
                Logger.info(
                    'Renamed OBS-station from IPY deployment to %s',
                    pick.waveform_id.station_code)

    if check_amplitudes:
        Logger.info('Checking amplitudes for sfile %s, event %s',
                    sfile_name, event.short_str())
        # merge A and IAML with empty amplitude value into one IAML
        # for OBIN? -stations; there can be IAMLHF and IAML amplitudes. IAML (or the other) may have lost its 
        #       amplitude phase name and is just called A, but has different time than IAMLHF
        for amp in event.amplitudes:
            if not amp.pick_id:
                Logger.info('Amplitude got no pick-id, cannot find pick')
                continue
            pick = amp.pick_id.get_referred_object()
            if pick is None or not pick.phase_hint:
                continue
            if (amp.type == 'A'
                and amp.type != pick.phase_hint
                and _is_iasp_ampl_phase(pick.phase_hint)):
                # Correct amplitude type name
                Logger.info('Found amplitude that needs fixing')
                amp.type = pick.phase_hint

        non_A_amp_picks = [
            pick for pick in event.picks
            if _is_iasp_ampl_phase(pick.phase_hint) and pick.phase_hint != 'A']
        picks_with_amp = [
            amp.pick_id.get_referred_object() for amp in event.amplitudes
            if amp.pick_id]
        non_A_amp_picks_without_amp = [
            pick for pick in non_A_amp_picks if not pick in picks_with_amp]

        for pick in non_A_amp_picks_without_amp:
            # search for A-pick with matching time and waveform id
            matching_A_picks = [
                p for p in event.picks
                if p.phase_hint == 'A' and p.time == pick.time and
                p.waveform_id == pick.waveform_id]
            if not len(matching_A_picks) == 1:
                continue
            matching_A_pick = matching_A_picks[0]
            matching_amps = [amp for amp in event.amplitudes
                            if amp.pick_id == matching_A_pick.resource_id]
            if not len(matching_amps) == 1:
                continue
            matching_amp = matching_amps[0]
            Logger.info('Found amplitude that needs fixing')
            # Now merge e.g. IAML and A picks
            # matching_A_pick
            new_amplitude = Amplitude(
                generic_amplitude=matching_amp.generic_amplitude,
                generic_amplitude_errors=matching_amp,
                type=pick.phase_hint,  # e.g., IAML
                category=matching_amp.category,
                unit=matching_amp.unit,
                period=matching_amp.period,
                snr=matching_amp.snr,
                time_window=matching_amp.time_window,
                pick_id=pick.resource_id,  # refer to existing IAML pick
                waveform_id=matching_amp.waveform_id,
                filter_id=matching_amp.filter_id,
                scaling_time=matching_amp.scaling_time,
                scaling_time_errors=matching_amp.scaling_time_errors,
                magnitude_hint=matching_amp.magnitude_hint,
                evaluation_mode=matching_amp.evaluation_mode,
                evaluation_status=matching_amp.evaluation_status,
                comments=matching_amp.comments,
                creation_info=matching_amp.creation_info)
            # Remove old pick/amps for 'A' and add new amp
            event.picks.remove(matching_A_pick)
            event.amplitudes.remove(matching_amp)
            event.amplitudes.append(new_amplitude)
            for station_magnitude in event.station_magnitudes:
                if station_magnitude.amplitude_id == matching_amp.resource_id:
                    station_magnitude.amplitude_id = new_amplitude.resource_id
            
        # Fix OBIN IAML
        A_amp_picks_OBIN = [
            pick for pick in event.picks
            if pick.waveform_id.station_code.startswith('OBIN') and
            pick.phase_hint == 'A']
        for pick in A_amp_picks_OBIN:
            matching_amps = [amp for amp in event.amplitudes
                            if amp.pick_id == pick.resource_id]
            if not len(matching_amps) == 1:
                continue
            matching_amp = matching_amps[0]
            pick.phase_hint = 'IAML'
            matching_amp.type = 'IAML'
            
    if check_pick_order:
        Logger.info('Checking pick order for sfile %s, event %s',
                    sfile_name, event.short_str())
        pick_order = [
            'P', 'Pn', 'PnPn', 'Pb', 'Pg', 'S', 'Sn', 'SnSn', 'Sb', 'Sg']
        event_stations = list(set([
            pick.waveform_id.station_code for pick in event.picks]))
        for station in event_stations:
            station_picks = [pick for pick in event.picks
                                if pick.waveform_id.station_code == station
                                and pick.phase_hint is not None
                                and pick.phase_hint in pick_order]
            station_picks = sorted(station_picks, key=lambda p: p.time)
            station_picks_hints_order = [
                pick.phase_hint for pick in station_picks]
            # Check if any pick of event is not according to pick order:
            supposed_picks_hints_order = [
                phint for phasehint in pick_order
                for phint in [ph for ph in station_picks_hints_order
                                if ph == phasehint]]
                # if ph in station_picks_hints_order]
            # TODO: better deal with P and S, e.g., Pn, P
            if station_picks_hints_order != supposed_picks_hints_order:
                # The order could still be ok if P / S appear as the
                # second unique phase-hint for all P-picks / all S-picks
                # at the station
                order_ok = False
                if 'P' in station_picks_hints_order:
                    uniqe_phase_hints_ordered_P = list(OrderedSet([
                        ph for ph in station_picks_hints_order
                        if ph[0] == 'P']))
                    if (len(uniqe_phase_hints_ordered_P) == 1 or
                            uniqe_phase_hints_ordered_P[0] == 'P' or
                            uniqe_phase_hints_ordered_P[1] == 'P'):
                        order_ok = True
                    else:
                        order_ok = False
                if 'S' in station_picks_hints_order:
                    uniqe_phase_hints_ordered_S = list(OrderedSet([
                        ph for ph in station_picks_hints_order
                        if ph[0] == 'S']))
                    if (len(uniqe_phase_hints_ordered_S) == 1 or
                            uniqe_phase_hints_ordered_S[0] == 'S' or
                            uniqe_phase_hints_ordered_S[1] == 'S'):
                        order_ok = True
                    else:
                        order_ok = False
                if order_ok:
                    continue
                Logger.warning(
                    'Picks for sfile %s, station %s, not in '
                    'expected order,  %-60s (event %s)', sfile_name, station,
                    ', '.join(station_picks_hints_order), event.short_str())
    return event


# %%

if __name__ == "__main__":
    cores = 20  #6
    filter_catalog = False # False
    check_eurarc_cat = False #False 
    check_nnsn_cat = False  #True # False  #  !True
    add_extra_uib_picks = False  # e.g., to only try to add amplitudes
    check_nnsn_cat_after_2018 = False  #True # False  #  !True
    merge_isc_after_2018 = False #False  #True # False  #  !True
    merge_nnsn_after_2018 = False # False  #True # False  #  !True
    check_Gibbons_ridge_cat = False
    check_loki_2017_cat = False # True
    check_loki_2019_cat = False
    check_storfjord_2019 = False
    max_gibbons_pick_difference = 0.06  # 0.5
    s_diff = 6
    max_euarc_error_km = 200
    max_loki_error_km = 100
    max_loki_s_diff = 8
    check_pick_logic = True  #  !True
    check_pick_syntax = True # False  #  !True
    check_station_specifics = False
    check_pick_duplication = True  #True # False  #  !True  # can be slow
    check_amplitudes = False  # True # False  #  !True
    check_pick_order = False # True # False  #  !True
    check_asterisk_picks_from_intaros = False
    parallel = False

    lats_vect = [67, 65, 69, 72, 75, 90, 83, 81, 78, 75, 71]
    lons_vect = [-23, 0, 8, 18, 40, 40, -20, -8, -11, -15, -20]
    lons_lats_vect = np.column_stack((lons_vect, lats_vect)) # Reshape coordinates
    polygon = Polygon(lons_lats_vect) # create polygon

    # full_cat = pickle.load(open('ECEAS_ISC_NNSN_after2018_merged_plusNNSNamps_03.pickle', "rb"))

    # full_cat = read_nordic(
    #     '/home/felix/Documents2/Ridge/EQCatalogs/Zeinabs_ArcticCatalog/' +
    #     'Convert_to_Nordic/Obspy/NewDownload_missing_events/INTAROS_missing_events_01.out',
    #     nordic_format='NEW')

    full_cat = read_seisan_database(
        # '~/Documents2/Ridge/Seisan/Backups/INTEU_backup_20220401', cores=cores, nordic_format='NEW',
        '../../Seisan/INTEU/', 
        # '../../Seisan/Backups/INTEU_backup_20220524_before_pick_merge',
        # cores=cores, nordic_format='NEW',
        cores=cores, nordic_format='NEW',
        # '~/Documents2/Ridge/Seisan/INTEU/', cores=cores, nordic_format='NEW',
        # starttime=UTCDateTime(1990,1,3,0,0,0), endtime=UTCDateTime(1990,1,4,0,0,0),
        # starttime=UTCDateTime(1999,12,25,0,0,0), endtime=UTCDateTime(1999,12,26,0,0,0),
        # starttime=UTCDateTime(1999,10,1,1,0,0), endtime=UTCDateTime(1999,12,31,23,0,0),
        # starttime=UTCDateTime(1990,1,1,0,0,0), endtime=UTCDateTime(2010,5,31,0,0,0),
        # starttime=UTCDateTime(1985,12,6,8,0,0), endtime=UTCDateTime(1985,12,23,23,0,0),
        # starttime=UTCDateTime(1999,1,1,0,10,0), endtime=UTCDateTime(1999,6,1,0,0,0),
        # starttime=UTCDateTime(1960,1,1,0,0,0), endtime=UTCDateTime(2020,1,1,0,0,0),
        # starttime=UTCDateTime(1998,12,24,0,0,0), endtime=UTCDateTime(2015,10,13,0,0,0),
        # starttime=UTCDateTime(2007,1,1,0,0,0), endtime=UTCDateTime(2007,4,1,0,0,0),
        # starttime=UTCDateTime(2015,10,1,0,0,0), endtime=UTCDateTime(2015,10,13,0,0,0),
        # starttime=UTCDateTime(2019,8,1,0,0,0), endtime=UTCDateTime(2020,8,31,23,59,59),
        # starttime=UTCDateTime(2019,8,23,0,0,0), endtime=UTCDateTime(2019,8,23,23,59,59),
        check_resource_ids=True)
    

    if check_asterisk_picks_from_intaros:
        Logger.info('Reading Intaros catalog')
        int_cat = read_nordic(
            '/home/felix/Documents2/Ridge/EQCatalogs/Zeinabs_ArcticCatalog/' +
            # 'Convert_to_Nordic/Obspy/intaros_cat_1999-01_1999-12.out')
            'Convert_to_Nordic/Obspy/intaros_cat_1960-01_2019-08_cleaned.out')
        int_df = events_to_df(int_cat)
        int_df['events'] = int_cat.events
        Logger.info('Data read in.')

    # full_cat = pickle.load(open('ECEAS_phasehint_fixes_plus_picks_slowness_baz_wav_03.pickle', "rb"))
    if check_Gibbons_ridge_cat:
        bayes_g_cat = read_bayesloc_events(
            '/home/felix/Software/Bayesloc/Example_Ridge_Gibbons_2017/output')
        bayes_g_df = events_to_df(bayes_g_cat)
        bayes_g_df['events'] = bayes_g_cat.events

    if check_eurarc_cat:
        Logger.info('Reading IMS EURARC bulleting')
        eu_cat = _read_ims10_bulletin(
            # '/home/felix/Documents2/Ridge/EQCatalogs/Johannes_24yr_Arctic_catalog/EURARC/test_isf_cat.out')
            '/home/felix/Documents2/Ridge/EQCatalogs/Johannes_24yr_Arctic_catalog/EURARC/EURARC_bulletin_Headers')
        eu_df = events_to_df(eu_cat)
        eu_df['events'] = eu_cat.events
        Logger.info('Data read in.')

    if merge_isc_after_2018:
        # new_isc_files = sorted(glob.glob('../ISC_after2018/isc_20??.quakeml'))
        # new_isc_files = glob.glob('../ISC_after2018/isc_2022.quakeml')
        # new_isc_cat = Catalog()
        # for new_isc_file in new_isc_files:
        #     Logger.info('Reading new ISC catalog file %s.', new_isc_file)
        #     new_isc_cat += read_events(new_isc_file)
        
        # new_isc_cat = pickle.load(open(
        #     '../ISC_after2018/isc_cat_201806-202206.pickle', 'rb'))
        new_isc_cat = pickle.load(open(
            '../ISC_2015-2018_missingEvents/isc_cat_2018_03-06.pickle', 'rb'))
        new_isc_cat = Catalog(
            [event for event in new_isc_cat
             if any([Point(orig.longitude, orig.latitude).within(polygon)
                     for orig in event.origins
                     if orig.longitude and orig.latitude])])
        new_isc_cat.events = sorted(
            new_isc_cat.events, key=lambda d: d.origins[0].time)

        if check_nnsn_cat_after_2018:
            Logger.info(
                'Updating new ISC catalog of %s events from NNSN catalog.',
                str(len(new_isc_cat)))
            events = Parallel(n_jobs=cores, prefer='threads')(
                delayed(add_NNSN_catalog_info)(
                    event, default_min_latitude=65, default_max_latitude=90,
                    default_min_longitude=-25, default_max_longitude=44,
                    skip_event_with_existing_NNSN_info=False,
                    add_extra_uib_picks=add_extra_uib_picks)
                for event in new_isc_cat)
            new_isc_cat = Catalog(events)
            new_isc_cat.events = sorted(
                new_isc_cat.events, key=lambda d: d.origins[0].time)
        for event in new_isc_cat:
            for pick in event.picks:
                if pick.waveform_id.network_code == 'IR':
                    pick.waveform_id.network_code == ''
                if pick.waveform_id.location_code == '--':
                    pick.waveform_id.location_code == ''
            attach_all_resource_ids(event)

    if merge_nnsn_after_2018:
        Logger.info('Reading new NNSN catalog file.')
        new_nnsn_cat = read_nordic(
            # '../NNSN_after2018/select_square_noExpl_201806-202206.out',
            '../NNSN_after2018/NNSN_after_2018_polygon_filterede_noExpl.out',
            nordic_format='NEW')
        # new_nnsn_cat = read_nordic(
        #    '../NNSN_2015-2018_missingEvents/select_square_noExpl_201803-201806.out',
        #    nordic_format='NEW')
        new_nnsn_cat = Catalog(
            [event for event in new_nnsn_cat
             if any([Point(orig.longitude, orig.latitude).within(polygon)
                     for orig in event.origins
                     if orig.longitude and orig.latitude])])
        new_nnsn_cat.events = sorted(
                new_nnsn_cat.events, key=lambda d: d.origins[0].time)
        for event in new_nnsn_cat:
            event.comments.append(Comment(text='Merged from NNSN catalog'))

    if check_loki_2017_cat:
        Logger.info('Reading Lokis castle catalog 2017-2018')
        loki_cat = read_seisan_database(
            '/home/seismo/WOR/felix/R/KGJ_2017-2018/LocalEvents/MOHNE_with_picks',
            # '/home/seismo/WOR/felix/R/KGJ_2017-2018/LocalEvents/MOHNE',
            cores=cores, check_resource_ids=True, nordic_format='OLD')
        loki_df = events_to_df(loki_cat)
        loki_df['events'] = loki_cat.events
        Logger.info('Data read in.')

    if check_loki_2019_cat:
        Logger.info('Reading Lokis castle catalog 2018-2019')
        loki2_cat = read_nordic(
            '/home/felix/Documents2/Ridge/EQCatalogs/Loki_2019-20/LOKP2_minmag25.nor',
            # '/home/seismo/WOR/felix/R/KGJ_2017-2018/LocalEvents/MOHNE_with_picks',
            # '/home/seismo/WOR/felix/R/KGJ_2017-2018/LocalEvents/MOHNE',
            # cores=cores, check_resource_ids=True,
            nordic_format='OLD')
        loki2_df = events_to_df(loki2_cat)
        loki2_df['events'] = loki2_cat.events
        Logger.info('Data read in.')

    if check_storfjord_2019:
        Logger.info('Reading Storfjorden 2019-2021 dataset')
        # stor_cat = read_seisan_database('STOR3', nordic_format='OLD')
        stor_cat = read_nordic('STOR3_cat_onlyEventsWithSTORpicks.out',
                               nordic_format='OLD')
        Logger.info('Data read in.')
        stor_df = events_to_df(stor_cat)
        stor_df['events'] = stor_cat.events

    # PRE-select catalog:
    cat = Catalog([event for event in full_cat
                   if 'explosion' not in event.event_type])

    # Define polygon (more points / interpolation along great circle arcs would be
    # better because the algorithm doesn't take that into account)
    # -- ca, 2.4 M square kilometers area
    if filter_catalog:
        cat = Catalog([event for event in cat if
                    any([Point(orig.longitude, orig.latitude).within(polygon)
                    for orig in event.origins
                    if orig.longitude and orig.latitude])])


# %%


    # Merge new catalogs
    if merge_isc_after_2018:
        cat = merge_new_events_into_catalog(cat, new_isc_cat)
    if merge_nnsn_after_2018:
        cat = merge_new_events_into_catalog(cat, new_nnsn_cat)

    # FIX STUFF in the catalog
    if parallel:
        events = Parallel(n_jobs=cores)(delayed( # , prefer='threads'
            _check_event)(event) for event in cat)
        cat = Catalog(events)
        cat.events = sorted(cat.events, key=lambda d: d.origins[0].time)
        for event in cat:
            attach_all_resource_ids(event)
    else:
        for event in cat:
            _check_event(event)


# %%


# add_NNSN_catalog_info(
#     cat[0], default_min_latitude=65, default_max_latitude=90,
#     default_min_longitude=-15, default_max_longitude=30,
#     skip_event_with_existing_NNSN_info=False)


# %%
    # fix phases and check with NNSN catalog
    if check_nnsn_cat:
        Logger.info('Updating catalog of %s events from NNSN catalog.',
                    str(len(cat)))
        events = Parallel(n_jobs=cores, prefer='threads')(
            delayed(add_NNSN_catalog_info)(
                event, default_min_latitude=65, default_max_latitude=90,
                default_min_longitude=-25, default_max_longitude=44,
                skip_event_with_existing_NNSN_info=False,
                add_extra_uib_picks=False)
            for event in cat)

            # event = add_NNSN_catalog_info(event)
        # Fix resource-IDs again
        cat = Catalog(events)
    for event in cat:
        attach_all_resource_ids(event)


    # %%
    cat.events = sorted(cat.events, key=lambda d: d.origins[0].time)

    # pickle.dump(cat, open('ECEAS_ISC_NNSN_after2018_merged_plusNNSNamps_plusLoki_02.pickle', "wb"), protocol=4)
    # write_select(
    #     cat, 'ECEAS_ISC_NNSN_after2018_merged_plusNNSNamps_plusLoki_02.out', userid='OBSP', evtype='R',
    #     wavefiles=None, high_accuracy=False, nordic_format='NEW')
    # cat.write('ECEAS_ISC_NNSN_after2018_merged_plusNNSNamps_plusLoki_02.qml', format='QUAKEML')


    pickle.dump(cat, open('ECEAS_merge_picks_again_01.pickle', "wb"), protocol=4)
    write_select(
        cat, 'ECEAS_merge_picks_again_01.out', userid='OBSP', evtype='R',
        wavefiles=None, high_accuracy=False, nordic_format='NEW')
    cat.write('ECEAS_merge_picks_again_01.qml', format='QUAKEML')
    
    # cat = pickle.load(open('intaros_cat.pickle', "rb"))
    # cat = pickle.load(open('intaros_cat_1960-01_2019-08.pickle', "rb"))

    # %%
    # write_select(cat, 'ECEAS_phasehint_fixes.out', nordic_format='NEW')


#%%
# from obspy.io.iaspei.core import _read_ims10_bulletin
# cat = _read_ims10_bulletin('/home/felix/Documents2/Ridge/EQCatalogs/Johannes_24yr_Arctic_catalog/EURARC/test_isf_cat.out')
# cat = _read_ims10_bulletin(
#     '/home/felix/Documents2/Ridge/EQCatalogs/Johannes_24yr_Arctic_catalog/EURARC/test_isf_event_read.dat')
# cat = _read_ims10_bulletin('/home/felix/Documents2/Ridge/EQCatalogs/Johannes_24yr_Arctic_catalog/EURARC/EURARC_bulletin_Headers')
# %%


# %%
# write_select(catalog=cc, filename='test_nordic_out.out', userid='fh', evtype='R', nordic_format='NEW')


# %%
# test 'A' and 'IAML' merging:
# cat = read_nordic('../../Seisan/INTEU/2019/03/27-2347-26R.S201903')
# 
# cat = read_nordic('../../Seisan/INTEU/2008/11/29-2130-05R.S200811')
# cat = read_nordic('../../Seisan/INTEU/1960/09/09-1619-36R.S196009')

# event = cat[0]
# _write_nordic(event, 'amp_test.sfile', nordic_format='NEW')


# %%
# ncat = read_events('/home/felix/Documents2/Ridge/Seisan/INTEU/2000/07/07-1835-47R.S200007', nordic_format='NEW')


