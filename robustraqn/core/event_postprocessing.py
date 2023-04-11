import os
import getpass
# import matplotlib

from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool
from joblib import Parallel, delayed
from xml.dom.minidom import Attr

import numpy as np
import difflib
import pandas as pd
from pathlib import Path

# from obspy.core.stream import Stream
from obspy.core.event import (Event, Catalog, Origin, Comment, CreationInfo,
                              WaveformStreamID)
from obspy.io.nordic.core import read_nordic, write_select, _write_nordic
from obspy import read as obspyread
from obspy import UTCDateTime
from obspy.clients.fdsn import RoutingClient
from eqcorrscan.core.match_filter.tribe import Tribe
from eqcorrscan.core.match_filter.party import Party

from robustraqn.obspy.clients.filesystem.sds import Client
from robustraqn.obspy.core import Stream, Trace
from robustraqn.utils.quality_metrics import (get_waveforms_bulk, read_ispaq_stats)
from robustraqn.core.seismic_array import get_station_sites
from robustraqn.utils.relative_magnitude import compute_relative_event_magnitude

from robustraqn.core.load_events import (get_all_relevant_stations)

import logging
Logger = logging.getLogger(__name__)


def add_origins_to_detected_events(
        catalog, party, tribe, overwrite_origins=False,
        origin_latitude=None, origin_longitude=None, origin_depth=None):
    """
    Add preliminary origins to all detected events based on the template events
    """
    catalog = catalog.copy()  # keep input safe
    for event in catalog:
        # Select detection
        detection = None
        found_det_fam = False
        for fam in party:
            family = fam
            for det in fam:
                if det.id == event.resource_id:
                    detection = det
                    found_det_fam = True
                    break
            if found_det_fam:
                break
        # Alternative: (not working now)
        # detection = party[event.resource_id.id[0:28]][0]
        # SHOULD BE: ????
        # detection = party[event.resource_id.id][0]
        if not detection:
            continue

        # Find hypocenter of template event
        # detection.template_name
        template_event = Event()
        template_orig = Origin()
        try:
            template_event = tribe[detection.template_name].event
            template_orig = (
                template_event.preferred_origin() or template_event.origins[0])
        except (AttributeError, IndexError):
            Logger.error(
                'Could not find template %s for related detection, did you '
                'provide the correct tribe?', detection.template_name)
            template_names = [templ.name for templ in tribe]
            template_name_match = difflib.get_close_matches(
                detection.template_name, template_names)
            if len(template_name_match) >= 1:
                template_name_match = template_name_match[0]
                template_event = tribe[template_name_match].event
                template_orig = (template_event.preferred_origin() or
                                 template_event.origins[0])
                Logger.warning(
                    'Found template with name %s, using instead of %s',
                    template_name_match, detection.template_name)
            else:
                Logger.warning('Cannot add origin for detection with template '
                               + '%s', detection.template_name)
                continue
        orig = Origin()
        approx_time = None
        if event.picks:
            # If possible, origin time should be set based on template origin:
            if template_event.picks:
                pick_times = [pick.time for pick in event.picks]
                earliest_pick_time = min(pick_times)
                earliest_pick_index = np.argmin(pick_times)
                earliest_pick = event.picks[earliest_pick_index]
                ep_sta = earliest_pick.waveform_id.station_code
                ep_component = earliest_pick.waveform_id.channel_code[-1]
                matching_template_pick = None
                matching_template_pick_list = [
                    pick for pick in template_event.picks
                    if pick.phase_hint == earliest_pick.phase_hint and
                    pick.waveform_id.station_code == ep_sta and
                    pick.waveform_id.channel_code[-1] == ep_component]
                # Sometimes there is no initial match when phase hints are not
                # strictly the same, when do_not_rename_refracted_phases=True.
                # Then relax comparison:
                if not matching_template_pick_list:
                    matching_template_pick_list = [
                        pick for pick in template_event.picks
                        if pick.phase_hint[0] == earliest_pick.phase_hint[0]
                        and pick.waveform_id.station_code == ep_sta]
                    mpick_times = [pick.time
                                   for pick in matching_template_pick_list]
                    if mpick_times:
                        earliest_mpick_index = np.argmin(mpick_times)
                        matching_template_pick = matching_template_pick_list[
                            earliest_mpick_index]
                    else:
                        matching_template_pick = None
                else:  # Select only match in list
                    matching_template_pick = matching_template_pick_list[0]
                if matching_template_pick:
                    shortest_traveltime = (
                        matching_template_pick.time - template_orig.time)
                    approx_time = earliest_pick_time - shortest_traveltime
            # Alternative: based on picks alone
            if not approx_time:
                approx_time = min([p.time for p in event.picks])
        if not approx_time:
            approx_time = detection.detect_time
        orig.time = approx_time
        orig.longitude = template_orig.longitude or origin_longitude
        orig.latitude = template_orig.latitude or origin_latitude
        orig.depth = template_orig.depth or origin_depth
        # day_st.slice(dt, dt + 5)
        if overwrite_origins:
            event.origins = [orig]
        else:
            event.origins.append(orig)
        event.preferred_origin_id = orig.resource_id
    return catalog


def postprocess_picked_events(
        picked_catalog, party, tribe, original_stats_stream,
        det_tribe=Tribe(), day_st=Stream(), pre_processed=False,
        write_sfiles=False, sfile_path='Sfiles',
        write_to_year_month_folders=False,
        operator='feha', all_chans_for_stations=[], extract_len=300,
        min_pick_stations=4, min_n_station_sites=4,
        min_picks_on_detection_stations=6, write_waveforms=False,
        archives=[], archive_types=[], request_fdsn=False, template_path=None,
        sta_translation_file=None,
        origin_longitude=None, origin_latitude=None, origin_depth=None,
        evtype='L', high_accuracy=False, do_not_rename_refracted_phases=False,
        compute_relative_magnitudes=False, remove_response=False,
        output='DISP', min_mag_cc=0.2, mag_min_cc_from_mean_cc_factor=None,
        mag_min_cc_from_median_cc_factor=None,
        absolute_values=False, parallel=False, cores=1, n_threads=1, **kwargs):
    """
    :type picked_catalog: :class:`obspy.core.Catalog`
    :param picked_catalog: Catalog of events picked, e.g., with lag_calc.
    :type party: :class:`eqcorrscan.core.match_filter.party.Party`
    :param party:
        Party containing all :class:`eqcorrscan.core.match_filter.Detection`
        that were used for picking events.
    :type tribe: :class:`eqcorrscan.core.match_filter.Tribe`
    :param tribe:
        Tribe which contains the template that was used to pick the
        picked_catalog.
    :type original_stats_stream: :class:`obspy.core.Stream`
    :param original_stats_stream:
    :type write_sfiles: bool
    :param write_sfiles: Whether to output Nordic S-files
    :type sfile_path: str
    :param sfile_path: Path to folder where to save S-files.
    :type operator: str
    :param operator: initials of the operating user (max 4 chars)
    :type all_chans_for_stations: bool
    :param all_chans_for_stations:
        Whether to save seismograms from all available channels for the
        picked event.
    :type extract_len: float
    :param extract_len:
        Length (in seconds) of seismograms to extract and save to file.
    :type min_pick_stations int
    :param min_pick_stations:
        Minimum number of stations that have to return a pick from lag_calc
        above lag_calc's correlation-threshold.
    :type min_picks_on_detection_stations: int
    :param min_picks_on_detection_stations:
        Minimum thershold for number of picks that have to be obtained from
        lag-calc for function to return an event from detections. This count
        involves only stations that were also used to make a detection, and not
        just pick with lag-calc.
    :type write_waveforms: bool
    :param write_waveforms: Whether to write the seismograms to a file.
    :type archives: list
    :param archives:
        List containing paths to the Seiscomp archives from which seismograms
        shall be requested.
    :type request_fdsn: bool
    :param request_fdsn:
        Whether to download waveform data for the requested stations from FDSN
        web services.
    :type template_path: str
    :param template_path: Path to folder where template-waveforms are saved.
    :type origin_longitude: float
    :param origin_longitude:
        Longitude value that shall be written to event as an initial guess.
    :type origin_latitude: float
    :param origin_latitude:
        Latitude value that shall be written to event as an initial guess.
    :type origin_depth: float
    :param origin_depth:
        Depth value that shall be written to event as an initial guess.

    :type return: :class:`obspy.core.Catalog`
    :param return: Catalog with the exported events.
    """
    export_catalog = Catalog()
    retained_detection_catalog = Catalog()
    for event in picked_catalog:
        # Save unaltered event
        detection_event = event.copy()
        # Correct Picks by the pre-Template time
        num_p_picks = 0
        num_s_picks = 0
        num_p_picks_on_det_sta = 0
        num_s_picks_on_det_sta = 0
        pick_stations = []
        s_pick_stations = []
        pick_and_detect_stations = []

        # Select detection
        detection = None
        found_det_fam = False
        for fam in party:
            family = fam
            for det in fam:
                if det.id == event.resource_id:
                    detection = det
                    found_det_fam = True
                    break
            if found_det_fam:
                break
        # Alternative: (not working now)
        # detection = party[event.resource_id.id[0:28]][0]
        # SHOULD BE: ????
        # detection = party[event.resource_id.id][0]
        if not detection:
            continue
        # Count number of unique P- and S-picks (i.e., on different stations)
        # Also differentiates picks based on whether the station was also used
        # for detection or only for picking.
        for pick in event.picks:
            pick_net = pick.waveform_id.network_code
            pick_station = pick.waveform_id.station_code
            pick_chan = pick.waveform_id.channel_code
            if pick_station not in pick_stations:
                pick_stations.append(pick_station)
            # Pick-correction not required any more!
            # May need to adjust pick's phase hint for Pg / Pn / Sg / Sn
            matching_picks = [
                p for p in family.template.event.picks
                if (p.waveform_id.network_code == pick_net and
                    p.waveform_id.station_code == pick_station and
                    p.waveform_id.channel_code[0:2] == pick_chan[0:2] and
                    p.phase_hint[0] == pick.phase_hint[0])]
            # TODO: once EQcorrscan supports multiple picks per trace, need to
            #       fix lines below!
            matching_picks = sorted(matching_picks, key=lambda p: p.time)
            if matching_picks: # and pick == matching_picks[0]:
                if len(matching_picks[0].phase_hint) > 1:
                    # option to allow first arriving P/S remain as P/S instead
                    # of Pn/Sn; otherwise some location programs may struggle.
                    if (do_not_rename_refracted_phases and
                            matching_picks[0].phase_hint == 'Pn'):
                        pass
                    else:
                        pick.phase_hint = matching_picks[0].phase_hint #[0]

            # pick.time = pick.time + 0.2
            if pick.phase_hint[0] == 'P':
                num_p_picks += 1
            elif pick.phase_hint[0] == 'S':
                sPick_Station = pick.waveform_id.station_code
                if sPick_Station not in s_pick_stations:
                    num_s_picks += 1
                    s_pick_stations.append(sPick_Station)
            # Count the number of picks on stations used
            # during match-filter detection

            if (pick_station, pick_chan) in detection.chans:
                if pick_station not in pick_and_detect_stations:
                    pick_and_detect_stations.append(pick_station)
                if pick.phase_hint[0] == 'P':
                    num_p_picks_on_det_sta += 1
                elif pick.phase_hint[0] == 'S':
                    num_s_picks_on_det_sta += 1
            # Put the original (non-normalized) channel information back into
            # the pick's stats.
            # Make sure to check for [1,2,etc]-channels if the data were
            # rotated to ZNE.
            pick_chan_comp = pick.waveform_id.channel_code[2]
            if pick_chan_comp == 'N':
                pick_chan_comp = '[' + pick_chan_comp + '1]'
            elif pick_chan_comp == 'E':
                pick_chan_comp = '[' + pick_chan_comp + '2]'
            req_chan = '??' + pick_chan_comp
            req_stream = original_stats_stream.select(station=pick_station,
                                                      channel=req_chan)
            if len(req_stream) == 0:
                continue
            pick.waveform_id = WaveformStreamID(seed_string=req_stream[0].id)
        if len(event.picks) == 0:
            continue

        # Find hypocenter of template and use it as preliminary origin for the
        # detection (happens again here so that origin is corrected according
        # to all corrected picks)
        event = add_origins_to_detected_events(
            Catalog([event]), party, tribe, overwrite_origins=True)[0]
        if event.preferred_origin() is None and len(event.origins) == 0:
            Logger.warning('Aborting picking for detection with template %s',
                           detection.template_name)
            continue
        # try:
        #     template_orig = tribe[
        #         detection.template_name].event.preferred_origin()
        # except AttributeError:
        #     Logger.error(
        #         'Could not find template %s for related detection, did you '
        #         'provide the correct tribe?', detection.template_name)
        #     template_names = [templ.name for templ in tribe]
        #     template_name_match = difflib.get_close_matches(
        #         detection.template_name, template_names)
        #     if len(template_name_match) >= 1:
        #         template_name_match = template_name_match[0]
        #         template_orig = tribe[
        #             template_name_match].event.preferred_origin()
        #         Logger.warning(
        #             'Found template with name %s, using instead of %s',
        #             template_name_match, detection.template_name)
        #     else:
        #         Logger.warning('Aborting picking for detection with template '
        #                        + '%s', detection.template_name)
        #         continue
        # orig = Origin()
        # orig.time = min([p.time for p in event.picks])
        # orig.longitude = origin_longitude or template_orig.longitude
        # orig.latitude = origin_latitude or template_orig.latitude
        # orig.depth = origin_depth or template_orig.depth
        # # day_st.slice(dt, dt + 5)
        # event.origins.append(orig)
        # event.preferred_origin_id = orig.resource_id
        Logger.info(
            'PickCheck: %s P_d: %s S_d: %s, P: %s, S: %s',
            str(event.origins[0].time), num_p_picks_on_det_sta,
            num_s_picks_on_det_sta, num_p_picks, num_s_picks)

        # Add comment to event to save detection-id
        event.comments.append(Comment(
            text='EQC_detection_id: ' + detection.id,
            creation_info=CreationInfo(agency='robustraqn',
                                       author=getpass.getuser())))
        # And threshold / exceedance stats
        thresh_text = ('EQC_threshold: type: {0:12s}, in: {1:7.2f}, '
                       + 'set: {2:7.2f}, exc: {3:7.2f}').format(
                           detection.threshold_type, detection.threshold_input,
                           detection.threshold,
                           detection.detect_val / detection.threshold)
        event.comments.append(Comment(
            text=thresh_text,
            creation_info=CreationInfo(agency='robustraqn',
                                       author=getpass.getuser())))
        event.event_type = family.template.event.event_type
        event.event_type_certainty = family.template.event.event_type_certainty
        # TODO: nordic write function may not prop. translate type uncertainty
        event.event_descriptions = family.template.event.event_descriptions
        # if (len(pick_stations) >= 3\
        #     and num_p_picks_on_det_sta >= 2\
        #     and num_p_picks_on_det_sta + num_s_picks_on_det_sta >= 6)\
        #     or (len(pick_stations) >= 4\
        #     and num_s_picks_on_det_sta >= 5):
        unique_stations = list(dict.fromkeys([
            p.waveform_id.station_code for p in event.picks]))
        n_station_sites = len(list(set(get_station_sites(unique_stations))))
        if ((len(pick_stations) >= min_pick_stations) and 
                n_station_sites >= min_n_station_sites and (
                num_p_picks_on_det_sta + num_s_picks_on_det_sta >=
                min_picks_on_detection_stations)):
            export_catalog += event
            retained_detection_catalog += detection_event

    # Sort catalog so that it's in correct order for output
    export_catalog.events = sorted(
        export_catalog.events,
        key=lambda d: (d.preferred_origin() or d.origins[0]).time)
    retained_detection_catalog.events = sorted(
        retained_detection_catalog.events,
        key=lambda d: (d.preferred_origin() or d.origins[0]).time)
    #                                        d.origins[0].time
    # Output
    if export_catalog.count() == 0:
        Logger.info(
            'No picked events saved (no detections fulfill pick-criteria).')
        return None
    # get waveforms for events
    wavefiles = None
    if write_waveforms or compute_relative_magnitudes:
        wavefiles, detection_list = extract_stream_for_picked_events(
            export_catalog, party, template_path, archives, archive_types,
            original_stats_stream=original_stats_stream,
            request_fdsn=request_fdsn, wav_out_dir=sfile_path,
            extract_len=extract_len, det_tribe=det_tribe,
            all_chans_for_stations=all_chans_for_stations,
            write_waveforms=write_waveforms,
            write_to_year_month_folders=write_to_year_month_folders,
            sta_translation_file=sta_translation_file,
            parallel=parallel, cores=cores)

    # Compute relative magnitudes for events
    if compute_relative_magnitudes:
        for j_event, (event, detection_event) in enumerate(zip(
                export_catalog, retained_detection_catalog)):
            # Select detection
            detection = None
            found_det_fam = False
            for fam in party:
                family = fam
                for det in fam:
                    if det.id == detection_event.resource_id:
                        detection = det
                        found_det_fam = True
                        break
                if found_det_fam:
                    break
            # Alternative: (not working now)
            # detection = party[event.resource_id.id[0:28]][0]
            # SHOULD BE: ????
            # detection = party[event.resource_id.id][0]
            if not detection:
                continue
            detection_event, pre_processed = compute_relative_event_magnitude(
                detection=detection, detected_event=detection_event,
                j_ev=j_event, day_st=day_st, party=party, tribe=tribe,
                templ2=None, detection_template_names=[], write_events=False,
                accepted_magnitude_types=['ML', 'Mw', 'MW'],
                accepted_magnitude_agencies=['BER', 'NAO'],
                mag_out_dir=None, min_cc=min_mag_cc,
                absolute_values=absolute_values,
                min_cc_from_mean_cc_factor=mag_min_cc_from_mean_cc_factor,
                min_cc_from_median_cc_factor=mag_min_cc_from_median_cc_factor,
                # min_snr=1.1, min_n_relative_amplitudes=3,
                # noise_window_start=-40, noise_window_end=-29.5,
                # signal_window_start=-0.5, signal_window_end=10,
                # use_s_picks=True, correlations=None, shift=0.35,
                return_correlations=True, correct_mag_bias=False,
                remove_response=remove_response, output=output,
                pre_processed=pre_processed,
                parallel=parallel, cores=cores, n_threads=n_threads, **kwargs)
            # Copy over magnitudes from detection-event to export-event
            event.magnitudes += detection_event.magnitudes
            event.station_magnitudes += detection_event.station_magnitudes
            event.amplitudes += detection_event.amplitudes
            for amplitude in event.amplitudes:
                if amplitude.pick_id is None:  # Cannot resolve seed-id
                    continue
                # May need to update pick resource id in amplitude
                ref_pick = amplitude.pick_id.get_referred_object()
                if ref_pick not in event.picks:
                    pick_ids = [p.resource_id for p in event.picks
                                if p.waveform_id == ref_pick.waveform_id]
                    for pick_id in pick_ids:
                        amplitude.pick_id = pick_id
            for comment in detection_event.comments:
                if comment not in event.comments:
                    event.comments.append(comment)

    # Create Seisan-style Sfiles for the whole day
    # sfile_path = os.path.join(sfile_path, + str(orig.time.year)\
    #   + str(orig.time.month).zfill(2) + str(orig.time.day).zfill(2) + ".out")
    # write_select(export_catalog, sfile_path, userid=operator, evtype='L',
    #              wavefiles=wavefiles, high_accuracy=False)
    # Create Seisan-style Sfiles for each event
    # e.g.: 01-0024-07L.S202001
    if write_sfiles:
        for j, event in enumerate(export_catalog):
            # Check if filename exists, otherwise try with one more second
            #  added to the filename's time.
            filename_exists = True
            orig_time = event.origins[0].time
            sfile_rea_path = os.path.join(sfile_path,
                                          '{:04}'.format(orig_time.year),
                                          '{:02}'.format(orig_time.month))
            while filename_exists:
                sfile_name = orig_time.strftime(
                    '%d-%H%M-%S' + evtype + '.S%Y%m')
                if write_to_year_month_folders:
                    # Make Year/month structure in folder if it does not exist
                    Path(sfile_rea_path).mkdir(parents=True, exist_ok=True)
                    sfile_out = os.path.join(sfile_rea_path, sfile_name)
                else:
                    sfile_out = os.path.join(sfile_path, sfile_name)
                filename_exists = os.path.exists(sfile_out)
                orig_time = orig_time + 1
            _write_nordic(event, sfile_out, userid=operator, evtype=evtype,
                          wavefiles=wavefiles[j], high_accuracy=high_accuracy,
                          nordic_format='NEW')
    # export_catalog.write(sfile_path, format="NORDIC", userid=operator,
    #                     evtype="L")

    return export_catalog


def extract_stream_for_picked_events(
        catalog, party, template_path, archives, archive_types,
        original_stats_stream=Stream(), det_tribe=Tribe(),
        request_fdsn=False, wav_out_dir='.', write_waveforms=False,
        write_to_year_month_folders=False, extract_len=300,
        all_chans_for_stations=[], sta_translation_file=None,
        only_relevant_stations=True, parallel=False, cores=1, **kwargs):
    """
    Extracts a stream object with all channels from the SDS-archive.
    Allows the input of multiple archives as a list
    """
    detection_list = list()
    templ_list = []
    for event in catalog:
        found_detection = False
        for family in party:
            for detection in family:
                if detection.id == event.resource_id:
                    detection_list.append(detection)
                    found_detection = True
                    break
            if found_detection:
                break

        # Find stream of detection template - can be loaded from tribe or files
        if len(det_tribe) > 0:
            try:
                templ_tuple = [
                    (family.template, det_tribe.select(family.template.name).st)]
                templ_list += templ_tuple
            except (AttributeError, IndexError):
                Logger.error('Could not find template %s for related detection',
                            family.template.name)
                template_names = [templ.name for templ in det_tribe]
                template_name_match = difflib.get_close_matches(
                    family.template.name, template_names)
                if len(template_name_match) >= 1:
                    template_name_match = template_name_match[0]
                Logger.warning(
                    'Found template with name %s, using instead of %s',
                    template_name_match, family.template.name)
                templ_tuple = [
                    (family.template, det_tribe.select(template_name_match).st)]
                templ_list += templ_tuple
        else:
            try:
                # templ_tuple = [(family.template, obspyread(os.path.join(
                #     template_path, family.template.name + '.mseed')))]
                templ_tuple = [(family.template, family.template.st)]
                templ_list += templ_tuple
            except FileNotFoundError:
                Logger.error(
                    'Cannot access stream for detection template with name %s, ' +
                    'during picking', family.template.name)
                return

    additional_stachans = list()
    for sta in all_chans_for_stations:
        additional_stachans.append((sta, '???'))
    additional_stations = all_chans_for_stations

    list_of_stream_lists = list()
    stream_list = extract_detections(
        detection_list, templ_list, archives, archive_types=archive_types,
        day_st=original_stats_stream, request_fdsn=request_fdsn,
        extract_len=extract_len, outdir=None,
        only_relevant_stations=only_relevant_stations,
        sta_translation_file=sta_translation_file,
        additional_stations=additional_stations,
        cores=cores, parallel=parallel)
    list_of_stream_lists.append(stream_list)

    stream_list = list()
    # put enough empty stream in stream_list
    stream_list = [Stream() for st in list_of_stream_lists[0]]
    # for st in list_of_stream_lists[0]:
    #     stream_list.append(Stream())
    # Now append the streams from each archive to the corresponding stream in
    # stream-list.
    for sl in list_of_stream_lists:
        for n_st, st in enumerate(sl):
            if isinstance(st, Stream):
                stream_list[n_st] += st

    wavefiles = list()
    for event, detection, stream in zip(catalog, detection_list, stream_list):
        # Attach stream to detection
        detection.st = stream
        # 2019_09_24t13_38_14
        # 2019-12-15-0323-41S.NNSN__008
        if not write_waveforms:
            continue
        utc_str = str(stream[0].stats.starttime)
        utc_str = utc_str.lower().replace(':', '-').replace('t', '-')
        # Define waveform filename based on starttime of stream
        w_name = (utc_str[0:13] + utc_str[14:19] +
                  'M.EQCS__' + str(len(stream)).zfill(3))
        try:
            orig_time = event.origins[0].time # detection.event.origins[0].time
        except (KeyError, AttributeError, IndexError):
            orig_time = detection.detect_time
            # Origin time would be about 1/4 to 1/3 into stream
            # orig_time = (
            #     stream[0].stats.starttime +
            #     (stream[0].stats.endtime - stream[0].stats.starttime) * 1/4)
        # Convert to 32 bit int data so that Seisan can read waveforms
        for trace in stream:
            trace.data = trace.data.astype("int32")
            try:
                trace.stats.mseed.encoding = 'INT32'
            except (KeyError, AttributeError):
                pass
        if write_to_year_month_folders:
            wav_folder_path = os.path.join(wav_out_dir,
                                           '{:04}'.format(orig_time.year),
                                           '{:02}'.format(orig_time.month))
            # Make Year/month structure in folder if it does not exist
            Path(wav_folder_path).mkdir(parents=True, exist_ok=True)
            waveform_path_filename = os.path.join(wav_folder_path, w_name)
            try:  # Try to write with full compression
                stream.write(waveform_path_filename, format='MSEED',
                             encoding='STEIM2')
            except Exception as e:
                Logger.info(e)
                Logger.info(
                    'First attempt to write mseed file %s with compression'
                    ' failed, trying with original trace encoding.',
                    waveform_path_filename)
                stream.write(waveform_path_filename, format='MSEED')
            wavefiles.append(waveform_path_filename)
        else:
            waveform_filename = os.path.join(wav_out_dir, w_name)
            wavefiles.append(waveform_filename)
            try:  # Try to write with full compression
                stream.write(waveform_filename, format='MSEED',
                             encoding='STEIM2')
            except Exception as e:
                Logger.info(e)
                Logger.info(
                    'First attempt to write mseed file %s with compression'
                    ' failed, trying with original trace encoding.',
                    waveform_filename)
                stream.write(waveform_filename, format='MSEED')

    return wavefiles, detection_list


def replace_templates_for_picking(party, tribe, set_sample_rate=100.0):
    """
    replace the old templates in the detection-families with those for
    picking (these contain more channels)

    :param party: _description_
    :type party: _type_
    :param tribe: _description_
    :type tribe: _type_
    :param set_sample_rate: _description_, defaults to 100.0
    :type set_sample_rate: float, optional
    :return: _description_
    :rtype: _type_
    """
    

    for family in party.families:
        family.template.samp_rate = set_sample_rate
        for newtemplate in tribe:
            if family.template.name == newtemplate.name:
                family.template = newtemplate
                break
    templateStreams = list()
    templateNames = list()
    for template in tribe:
        templateStreams += (template.st)
        templateNames += template.name

    return party


def _check_duplicate_template_channels(
    template, all_vert=False, all_horiz=False, vertical_chans=['Z'],
    horizontal_chans=['E', 'N', '1', '2']):
    """
    Check template for duplicate channels (happens when there are P- and
        S-picks on the same channel, or Pn/Pg and Sn/Sg). Then throw away the
        later one for now.
    """
    # Keep only the earliest trace for traces with same ID
    temp_st_new = Stream()
    unique_trace_ids = list(dict.fromkeys([tr.id for tr in template.st]))
    for trace_id in unique_trace_ids:
        trace_check_id = trace_id
        if (all_vert and len(vertical_chans) > 1
                and trace_id[-1] in vertical_chans):
            # Formulate wildcards to select existing traces
            trace_check_id = (
                trace_id[0:-1] + '[' + ''.join(vertical_chans) + ']')
        elif (all_horiz and len(horizontal_chans) > 1
                and trace_id[-1] in horizontal_chans):
            trace_check_id = (
                trace_id[0:-1] + '[' + ''.join(horizontal_chans) + ']')
        # same_id_st = template.st.select(id=trace_check_id)
        same_id_st = template.st.select(id=trace_check_id)
        if len(same_id_st) == 1 and same_id_st[0] not in temp_st_new:
            temp_st_new += same_id_st
        elif len(same_id_st) > 1:  # keep only earliest traces
            starttimes = [tr.stats.starttime for tr in same_id_st]
            earliest_starttime = min(starttimes)
            for tr in same_id_st:
                if (tr.stats.starttime == earliest_starttime
                        and tr not in temp_st_new):
                    temp_st_new += tr
    template.st = temp_st_new  # replace previous template stream

    # Also throw away the later pick from the template's event
    new_pick_list = list()
    pick_tr_id_list = list()
    uniq_pick_trace_ids = list(dict.fromkeys(
        [pick.waveform_id.id for pick in template.event.picks]))
    for pick_tr_id in uniq_pick_trace_ids:
        pick_tr_check_id = [pick_tr_id]
        if (all_vert and len(vertical_chans) > 1
                and pick_tr_id[-1] in vertical_chans):
            for vert_chan in vertical_chans:
                if vert_chan != pick_tr_id[-1]:
                    pick_tr_check_id.append(pick_tr_id[0:-1] + vert_chan)
        elif (all_horiz and len(horizontal_chans) > 1
                and pick_tr_id[-1] in horizontal_chans):
            for hor_chan in horizontal_chans:
                if hor_chan != pick_tr_id[-1]:
                    pick_tr_check_id.append(pick_tr_id[0:-1] + hor_chan)
        same_id_picks = [pick for pick in template.event.picks
                            if pick.waveform_id.id in pick_tr_check_id]
        if (len(same_id_picks) == 1
                and same_id_picks[0] not in new_pick_list):
            new_pick_list.append(same_id_picks[0])
        elif len(same_id_picks) > 1:  # keep only earliest picks per trace
            pick_times = [pick.time for pick in same_id_picks]
            earliest_pick_time = min(pick_times)
            for pick in same_id_picks:
                # Every trace can only have one pick for lag-calc. If there
                # are two picks at the same time on same trace (e.g.,
                # one manual and one for array), then just keep the first
                # one from the list.
                if (pick.time == earliest_pick_time
                        and pick not in new_pick_list
                        and pick_tr_id not in pick_tr_id_list):
                    new_pick_list.append(pick)
                    pick_tr_id_list.append(pick_tr_id)
    template.event.picks = new_pick_list
    return template


def check_duplicate_template_channels(
        tribe, all_vert=False, all_horiz=False, vertical_chans=['Z'],
        horizontal_chans=['E', 'N', '1', '2'], parallel=False, cores=None):
    """
    Check templates for duplicate channels (happens when there are P- and
        S-picks on the same channel, or Pn/Pg and Sn/Sg). Then throw away the
        later one for now.
    """
    Logger.info('Checking templates in %s for duplicate channels', tribe)
    if not parallel:
        for template in tribe:
            template = _check_duplicate_template_channels(
                template, all_vert=all_vert, all_horiz=all_horiz,
                vertical_chans=vertical_chans,
                horizontal_chans=horizontal_chans)
    else:
        if cores is None:
            cores = cpu_count()
        cores = min(cores, len(tribe))
        out_lists = Parallel(n_jobs=cores)(
            delayed(_check_duplicate_template_channels)(
                template, all_vert=all_vert, all_horiz=all_horiz,
                vertical_chans=vertical_chans,
                horizontal_chans=horizontal_chans)
            for template in tribe)
        tribe = Tribe([out for out in out_lists
                       if out is not None and len(out.st) > 0])
    return tribe

    # for template in tribe:
    #     nt = 0
    #     # loop through stream and keep only earliest traces for each trace-ID.
    #     channel_ids = list()
    #     stream_copy = template.st.copy()
    #     for nt, trace in enumerate(stream_copy):
    #         if trace.id in channel_ids:
    #             for j in range(0, nt):
    #                 test_same_id_trace = stream_copy[j]
    #                 similar_trace_in_stream = False
    #                 if trace.id == test_same_id_trace.id:
    #                     similar_trace_in_stream = True
    #                 elif all_vert and len(vertical_chans) > 1:
    #                     # Check for equivalent vertical channels
    #                     for vert_chan in vertical_chans:
    #                         alt_trace_id = trace.id[0:-1] + vert_chan
    #                         if alt_trace_id == test_same_id_trace.id:
    #                             similar_trace_in_stream = True
    #                             break
    #                 elif all_horiz and len(horizontal_chans) > 1:
    #                     # Check for equivalent horizontal channels
    #                     for hor_chan in horizontal_chans:
    #                         alt_trace_id = trace.id[0:-1] + hor_chan
    #                         if alt_trace_id == test_same_id_trace.id:
    #                             similar_trace_in_stream = True
    #                             break
    #                 if similar_trace_in_stream:
    #                     if trace.stats.starttime >= test_same_id_trace.stats.\
    #                             starttime:
    #                         if trace in template.st:
    #                             # TODO: check that all traces 
    #                             template.st.remove(trace)
    #                     else:
    #                         if test_same_id_trace in template.st:
    #                             template.st.remove(test_same_id_trace)
    #                     continue
    #         else:
    #             channel_ids.append(trace.id)

        # Also throw away the later pick from the template's event
        # np = 0
        # pick_ids = list()
        # picks_copy = template.event.picks.copy()
        # for np, pick in enumerate(picks_copy):
        #     if pick.waveform_id.id in pick_ids:
        #         for j in range(0, np):
        #             test_same_id_pick = picks_copy[j]
        #             # if pick.waveform_id.id == test_same_id_pick.waveform_id.id:
        #             similar_pick_in_picks = False
        #             if pick.waveform_id.id == test_same_id_pick.waveform_id.id:
        #                 similar_pick_in_picks = True
        #             elif all_vert and len(vertical_chans) > 1:
        #                 # Check for equivalent vertical channels
        #                 for vert_chan in vertical_chans:
        #                     alt_pick_id = pick.waveform_id.id[0:-1] + vert_chan
        #                     if alt_pick_id == test_same_id_pick.waveform_id.id:
        #                         similar_pick_in_picks = True
        #                         break
        #             elif all_horiz and len(horizontal_chans) > 1:
        #                 # Check for equivalent horizontal channels
        #                 for hor_chan in horizontal_chans:
        #                     alt_pick_id = pick.waveform_id.id[0:-1] + hor_chan
        #                     if alt_pick_id == test_same_id_pick.waveform_id.id:
        #                         similar_pick_in_picks = True
        #                         break
        #             if similar_pick_in_picks:
        #                 if pick.time >= test_same_id_pick.time:
        #                     if pick in template.event.picks:
        #                         template.event.picks.remove(pick)
        #                 else:
        #                     if test_same_id_pick in template.event.picks:
        #                         template.event.picks.remove(test_same_id_pick)
        #                 continue
        #     else:
        #         pick_ids.append(pick.waveform_id.id)


def extract_detections(detections, templates, archives, archive_types,
                       day_st=Stream(), request_fdsn=False, extract_len=90.0,
                       outdir=None, extract_Z=True, additional_stations=[],
                       only_relevant_stations=True, sta_translation_file=None,
                       parallel=False, cores=None):
    """
    Extract waveforms associated with detections

    Takes a list of detections for the template, template.  Waveforms will be
    returned as a list of :class:`obspy.core.stream.Stream` containing
    segments of extract_len.  They will also be saved if outdir is set.
    The default is unset.  The  default extract_len is 90 seconds per channel.

    :type detections: list
    :param detections: List of :class:`eqcorrscan.core.match_filter.Detection`.
    :type templates: list
    :param templates:
        A list of tuples of the template name and the template Stream used
        to detect detections.
    :type archive: str
    :param archive:
        Either name of archive or path to continuous data, see
        :func:`eqcorrscan.utils.archive_read` for details
    :type arc_type: str
    :param arc_type: Type of archive, either seishub, FDSN, day_vols
    :type extract_len: float
    :param extract_len:
        Length to extract around the detection (will be equally cut around
        the detection time) in seconds.  Default is 90.0.
    :type outdir: str
    :param outdir:
        Default is None, with None set, no files will be saved,
        if set each detection will be saved into this directory with files
        named according to the detection time, NOT than the waveform
        start time. Detections will be saved into template subdirectories.
        Files written will be multiplexed miniseed files, the encoding will
        be chosen automatically and will likely be float.
    :type extract_Z: bool
    :param extract_Z:
        Set to True to also extract Z channels for detections delays will be
        the same as horizontal channels, only applies if only horizontal
        channels were used in the template.
    :type additional_stations: list
    :param additional_stations:
        List of tuples of (station, channel) to also extract data
        for using an average delay.

    :returns: list of :class:`obspy.core.streams.Stream`
    :rtype: list

    .. rubric: Example

    >>> from eqcorrscan.utils.clustering import extract_detections
    >>> from eqcorrscan.core.match_filter import Detection
    >>> from obspy import read, UTCDateTime
    >>> # Get the path to the test data
    >>> import eqcorrscan
    >>> import os
    >>> TEST_PATH = os.path.dirname(eqcorrscan.__file__) + '../tests/test_data'
    >>> # Use some dummy detections, you would use real one
    >>> detections = [Detection(
    ...     template_name='temp1', detect_time=UTCDateTime(2012, 3, 26, 9, 15),
    ...     no_chans=2, chans=['WHYM', 'EORO'], detect_val=2, threshold=1.2,
    ...     typeofdet='corr', threshold_type='MAD', threshold_input=8.0),
    ...               Detection(
    ...     template_name='temp2', detect_time=UTCDateTime(2012, 3, 26, 18, 5),
    ...     no_chans=2, chans=['WHYM', 'EORO'], detect_val=2, threshold=1.2,
    ...     typeofdet='corr', threshold_type='MAD', threshold_input=8.0)]
    >>> archive = os.path.join(TEST_PATH, 'day_vols')
    >>> template_files = [os.path.join(TEST_PATH, 'temp1.ms'),
    ...                   os.path.join(TEST_PATH, 'temp2.ms')]
    >>> templates = [('temp' + str(i), read(filename))
    ...              for i, filename in enumerate(template_files)]
    >>> extracted = extract_detections(detections, templates,
    ...                                archive=archive, arc_type='day_vols')
    >>> print(extracted[0].sort())
    2 Trace(s) in Stream:
    AF.EORO..SHZ | 2012-03-26T09:14:15.000000Z - 2012-03-26T09:15:45.000000Z |\
 1.0 Hz, 91 samples
    AF.WHYM..SHZ | 2012-03-26T09:14:15.000000Z - 2012-03-26T09:15:45.000000Z |\
 1.0 Hz, 91 samples
    >>> print(extracted[1].sort())
    2 Trace(s) in Stream:
    AF.EORO..SHZ | 2012-03-26T18:04:15.000000Z - 2012-03-26T18:05:45.000000Z |\
 1.0 Hz, 91 samples
    AF.WHYM..SHZ | 2012-03-26T18:04:15.000000Z - 2012-03-26T18:05:45.000000Z |\
 1.0 Hz, 91 samples
    >>> # Extract from stations not included in the detections
    >>> extracted = extract_detections(
    ...    detections, templates, archive=archive, arc_type='day_vols',
    ...    additional_stations=[('GOVA', 'SHZ')])
    >>> print(extracted[0].sort())
    3 Trace(s) in Stream:
    AF.EORO..SHZ | 2012-03-26T09:14:15.000000Z - 2012-03-26T09:15:45.000000Z |\
 1.0 Hz, 91 samples
    AF.GOVA..SHZ | 2012-03-26T09:14:15.000000Z - 2012-03-26T09:15:45.000000Z |\
 1.0 Hz, 91 samples
    AF.WHYM..SHZ | 2012-03-26T09:14:15.000000Z - 2012-03-26T09:15:45.000000Z |\
 1.0 Hz, 91 samples
    >>> # The detections can be saved to a file:
    >>> extract_detections(detections, templates, archive=archive,
    ...                    arc_type='day_vols',
    ...                    additional_stations=[('GOVA', 'SHZ')], outdir='.')
    """
    # Sort the template according to start-times, needed so that stachan[i]
    # corresponds to delays[i]
    all_delays = []  # List of tuples of template name, delays
    all_stachans = []
    for template in templates:
        templatestream = template[1].sort(['starttime'])
        stations = [tr.stats.station for tr in templatestream]
        stachans = [(tr.stats.station, tr.stats.channel)
                    for tr in templatestream]
        mintime = templatestream[0].stats.starttime
        delays = [tr.stats.starttime - mintime for tr in templatestream]
        all_delays.append((template[0], delays))
        all_stachans.append((template[0], stachans))
    # Sort the detections and group by day
    detections.sort(key=lambda d: d.detect_time)
    detection_days = [detection.detect_time.date
                      for detection in detections]
    detection_days = list(set(detection_days))
    detection_days.sort()
    detection_days = [UTCDateTime(d) for d in detection_days]

    # Initialize output list
    detection_wavefiles = []

    all_stations = sorted(list(set(stations + additional_stations)))

    # Define client
    clients = []
    for archive, arc_type in zip(archives, archive_types):
        if arc_type == 'SDS':
            from obspy.clients.filesystem.sds import Client
            clients.append(Client(archive))

    if request_fdsn:
        routing_client = RoutingClient('iris-federator')

    # Loop through the days
    for detection_day in detection_days:
        Logger.info('Working on detections for day: ' + str(detection_day))
        # List of all unique stachans - read in all data
        # st = read_data(archive=archive, arc_type=arc_type, day=detection_day,
        #                stachans=stachans)
        # st.merge(fill_value='interpolate')
        day_detections = [detection for detection in detections
                          if UTCDateTime(detection.detect_time.date) ==
                          detection_day]

        # Reuqest the whole day's stream plus 15 minutes before / after
        starttime = UTCDateTime(detection_day.date) - 15*60
        endtime = starttime + 24.5 * 60 * 60
        bulk = [('*', sta, '*', '*', starttime, endtime)
                for sta in all_stations]
        # day_st = get_waveforms_bulk(
        #         client, bulk, parallel=parallel, cores=cores)
        for client in clients:  # Append to existing day_st (can be empty)
            day_st += client.get_waveforms_bulk(
                bulk, parallel=parallel, cores=cores)
        n_chans_day_str = len(day_st)
        for detection in day_detections:
            Logger.info(
                'Cutting for detections at: ' +
                detection.detect_time.strftime('%Y/%m/%d %H:%M:%S'))
                # 2018/11/05 08:51:58
            t1 = UTCDateTime(detection.detect_time) - extract_len * 0.2
            t2 = UTCDateTime(detection.detect_time) + extract_len * 0.8
            # Slice instead of trim allows stream to first cut, then copied - 
            # otherwise will run out of memory for multiple cuts.
            st = day_st.slice(starttime=t1, endtime=t2) # .copy()

            # Reduce the stream to all traces relevant to the picks and traces
            # in detection and template:
            if only_relevant_stations:
                relevant_stations = [pick.waveform_id.station_code
                                     for pick in detection.event.picks]
                relevant_stations += [chan[0] for chan in detection.chans]
                template = [templ for templ in templates
                            if templ[0].name == detection.template_name]
                if len(template) >= 1:  # Select one template alone
                    template_st = template[0][1]
                    template = template[0][0]
                else:
                    Logger.warning(
                        'Could not find template for detection %s, cannot '
                        'resolve which channels were part of template.',
                        detection)
                # Template picks contain array picks even if some channels are
                # missing in template stream
                relevant_stations = [pick.waveform_id.station_code
                                     for pick in template.event.picks]
                relevant_stations += [trace.stats.station
                                     for trace in template_st if template]
                # Add any additional stations that are requested for each event
                relevant_stations += additional_stations
                relevant_stations = list(set(relevant_stations))
                # Add alternative station codes for requested stations
                if sta_translation_file is not None:
                    relevant_stations = get_all_relevant_stations(
                        relevant_stations,
                        sta_translation_file=sta_translation_file)

                new_st = Stream()
                for station in relevant_stations:
                    new_st += st.select(station=station)
                st = new_st
                Logger.debug('Reduced full stream (%s traces) to %s traces for'
                             ' detection %s', n_chans_day_str, len(st),
                             detection.id)

            # Request extra channels from fdsn webservices if required, but
            # only request the exact data segments required to spare services.
            if request_fdsn:
                existing_stations = list(dict.fromkeys(
                    [tr.stats.station for tr in st]))
                other_stations = list(dict.fromkeys(
                    all_stations).difference(existing_stations))
                if len(other_stations) > 0:
                    Logger.info('Requesting %s additional stations from FDSN-'
                                'routing client.', len(other_stations))
                    bulk = [
                        ('*', sta, '*', '*', t1, t2) for sta in other_stations]
                    add_st = routing_client.get_waveforms_bulk(bulk)
                    if add_st:
                        add_st = add_st.select(channel='[NDSEHBM]??').select(
                            starttime=t1, endtime=t2)
                        st += add_st
                    else:
                        Logger.info('FDSN-request did not return data.')

            if outdir:
                if not os.path.isdir(os.path.join(outdir,
                                                  detection.template_name)):
                    os.makedirs(os.path.join(outdir, detection.template_name))
                for trace in st:
                    trace.data = trace.data.astype("int32")
                    try:
                        trace.stats.mseed.encoding = 'INT32'
                    except (KeyError, AttributeError):
                        pass
                st_filename = os.path.join(
                    outdir, detection.template_name,
                    detection.detect_time.strftime(
                        '%Y-%m-%d_%H-%M-%S') + '.ms')
                try:  # Try to write with full compression
                    st.write(st_filename, format='MSEED', encoding='STEIM2')
                except Exception as e:
                    Logger.info(e)
                    Logger.info(
                        'First attempt to write mseed file %s with compression'
                        ' failed, trying with original trace encoding.',
                        st_filename)
                    st.write(st_filename, format='MSEED')
                Logger.info(
                    'Written file: %s' % '/'.join(
                        [outdir, detection.template_name,
                         detection.detect_time.strftime('%Y-%m-%d_%H-%M-%S')
                         + '.ms']))
            if not outdir:
                detection_wavefiles.append(st)
            del st
        if outdir:
            detection_wavefiles = []
    if not outdir:
        return detection_wavefiles
    else:
        return
