import os
import getpass
# import matplotlib

from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool


import numpy as np
import difflib
import pandas as pd
# import numexpr as ne

# from obspy import read_events, read_inventory
# from obspy.core.event import Catalog
# import obspy
from obspy.core.stream import Stream
# from obspy.core.util.base import TypeError
from obspy.core.event import (Event, Catalog, Origin, Comment, CreationInfo,
                              WaveformStreamID)
from obspy.io.nordic.core import read_nordic, write_select, _write_nordic
from obspy import read as obspyread
from obspy import UTCDateTime
from obspy.io.mseed import InternalMSEEDError
from obspy.clients.fdsn import RoutingClient
from eqcorrscan.core.match_filter.tribe import Tribe
from eqcorrscan.utils.correlate import pool_boy
from eqcorrscan.core.match_filter.party import Party
# from eqcorrscan.utils.clustering import extract_detections
# from eqcorrscan.utils.despike import median_filter
# from obspy import read_nordic
# import obspy

from robustraqn.quality_metrics import (
    create_bulk_request, get_waveforms_bulk, read_ispaq_stats)

import logging
Logger = logging.getLogger(__name__)



# TODO: write function to just add origins to template
def add_origins_to_detected_events(catalog, party, tribe,
                                   overwrite_origins=False):
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
        try:
            template_orig = tribe[
                detection.template_name].event.preferred_origin()
        except AttributeError:
            Logger.error(
                'Could not find template %s for related detection, did you '
                'provide the correct tribe?', detection.template_name)
            template_names = [templ.name for templ in tribe]
            template_name_match = difflib.get_close_matches(
                detection.template_name, template_names)
            if len(template_name_match) >= 1:
                template_name_match = template_name_match[0]
                template_orig = tribe[
                    template_name_match].event.preferred_origin()
                Logger.warning(
                    'Found template with name %s, using instead of %s',
                    template_name_match, detection.template_name)
            else:
                Logger.warning('Cannot add origin for detection with template '
                               + '%s', detection.template_name)
                continue
        orig = Origin()
        orig.time = min([p.time for p in event.picks])
        orig.longitude = origin_longitude or template_orig.longitude
        orig.latitude = origin_latitude or template_orig.latitude
        orig.depth = origin_depth or template_orig.depth
        # day_st.slice(dt, dt + 5)
        if overwrite_origins:
            event.origins = [orig]
        else:
            event.origins.append(orig)
        event.preferred_origin_id = orig.resource_id
    return catalog


def postprocess_picked_events(
        picked_catalog, party, tribe, original_stats_stream,
        det_tribe=Tribe(), write_sfiles=False, sfile_path='Sfiles',
        operator='feha', all_channels_for_stations=[], extract_len=300,
        min_pick_stations=4, min_picks_on_detection_stations=6,
        write_waveforms=False, archives=list(), request_fdsn=False,
        template_path=None, origin_longitude=None, origin_latitude=None,
        origin_depth=None):
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
    :type all_channels_for_stations: bool
    :param all_channels_for_stations:
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
    for event in picked_catalog:
        # Correct Picks by the pre-Template time
        num_pPicks = 0
        num_sPicks = 0
        num_pPicks_onDetSta = 0
        num_sPicks_onDetSta = 0
        pick_stations = []
        sPick_Stations = []
        pick_and_detect_Stations = []

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
            matching_picks = sorted(matching_picks, key=lambda p: p.time)
            if matching_picks:
                if len(matching_picks[0].phase_hint) > 1:
                    pick.phase_hint = matching_picks[0].phase_hint

            # pick.time = pick.time + 0.2
            if pick.phase_hint[0] == 'P':
                num_pPicks += 1
            elif pick.phase_hint[0] == 'S':
                sPick_Station = pick.waveform_id.station_code
                if sPick_Station not in sPick_Stations:
                    num_sPicks += 1
                    sPick_Stations.append(sPick_Station)
            # Count the number of picks on stations used
            # during match-filter detection

            if (pick_station, pick_chan) in detection.chans:
                if pick_station not in pick_and_detect_Stations:
                    pick_and_detect_Stations.append(pick_station)
                if pick.phase_hint[0] == 'P':
                    num_pPicks_onDetSta += 1
                elif pick.phase_hint[0] == 'S':
                    num_sPicks_onDetSta += 1
            # Put the original (non-normalized) channel information back into
            # the pick's stats.
            # Make sure to check for [1,2,etc]-channels if the data were
            # rotated to ZNE.
            pick_chan_comp = pick.waveform_id.channel_code[2]
            if pick_chan_comp == 'N':
                pick_chan_comp = '[' + pick_chan_comp + '1]'
            elif pick_chan_comp == 'E':
                pick_chan_comp = '[' + pick_chan_comp + '2]'
            reqChan = '??' + pick_chan_comp
            reqStream = original_stats_stream.select(station=pick_station,
                                                     channel=reqChan)
            if len(reqStream) == 0:
                continue
            pick.waveform_id = WaveformStreamID(seed_string=reqStream[0].id)
        if len(event.picks) == 0:
            continue

        # Find hypocenter of template and use it as preliminary origin for the
        # detection (happens again here so that origin is corrected according
        # to all corrected picks)
        event = add_origins_to_detected_events(
            Catalog(event), dayparty, tribe, overwrite_origins=True)[0]
        if event.preferred_origin() is None:
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
            'PickCheck: ' + str(event.origins[0].time) + ' P_d: ' +
            str(num_pPicks_onDetSta) + ' S_d ' + str(num_sPicks_onDetSta) +
            ' P: ' + str(num_pPicks) + ' S: ' + str(num_sPicks))

        # Add comment to event to save detection-id
        event.comments.append(Comment(
            text='EQC_detection_id: ' + detection.id,
            creation_info=CreationInfo(agency='eqcorrscan',
                                       author=getpass.getuser())))
        # if (len(pick_stations) >= 3\
        #     and num_pPicks_onDetSta >= 2\
        #     and num_pPicks_onDetSta + num_sPicks_onDetSta >= 6)\
        #     or (len(pick_stations) >= 4\
        #     and num_sPicks_onDetSta >= 5):
        if ((len(pick_stations) >= min_pick_stations) and (
                num_pPicks_onDetSta + num_sPicks_onDetSta >=
                min_picks_on_detection_stations)):
            export_catalog += event

    # Sort catalog so that it's in correct order for output
    export_catalog.events = sorted(
        export_catalog.events, key=lambda d: d.preferred_origin().time)
    #                                        d.origins[0].time
    # Output
    if export_catalog.count() == 0:
        Logger.info(
            'No picked events saved (no detections fulfill pick-criteria).')
        return None
    # get waveforms for events
    wavefiles = None
    if write_waveforms:
        wavefiles = extract_stream_for_picked_events(
            export_catalog, party, template_path, archives,
            request_fdsn=request_fdsn, wav_out_dir=sfile_path,
            extract_len=extract_len, det_tribe=det_tribe,
            all_chans_for_stations=all_channels_for_stations)

    # Create Seisan-style Sfiles for the whole day
    # catalogFile = os.path.join(sfile_path, + str(orig.time.year)\
    #   + str(orig.time.month).zfill(2) + str(orig.time.day).zfill(2) + ".out")
    # write_select(export_catalog, catalogFile, userid=operator, evtype='L',
    #              wavefiles=wavefiles, high_accuracy=False)
    # Create Seisan-style Sfiles for each event
    # e.g.: 01-0024-07L.S202001
    if write_sfiles:
        for j, event in enumerate(export_catalog):
            # Check if filename exists, otherwise try with one more second
            #  added to the filename's time.
            filename_exists = True
            orig_time = event.origins[0].time
            while filename_exists:
                sfile_name = orig_time.strftime('%d-%H%M-%SL.S%Y%m')
                catalogFile = os.path.join(sfile_path, sfile_name)
                filename_exists = os.path.exists(catalogFile)
                orig_time = orig_time + 1
            _write_nordic(event, catalogFile, userid=operator, evtype='L',
                          wavefiles=wavefiles[j], high_accuracy=False,
                          nordic_format='NEW')
    # export_catalog.write(catalogFile, format="NORDIC", userid=operator,
    #                     evtype="L")

    return export_catalog


def extract_stream_for_picked_events(
        catalog, party, template_path, archives, det_tribe=Tribe(),
        request_fdsn=False, wav_out_dir='.', extract_len=300,
        all_chans_for_stations=[]):
    """
    Extracts a stream object with all channels from the SDS-archive.
    Allows the input of multiple archives as a list
    """
    detection_list = list()
    for event in catalog:
        for family in party:
            for detection in family:
                if detection.id == event.resource_id:
                    detection_list.append(detection)

    # Find stream of detection template - can be loaded from tribe or files
    if len(det_tribe) > 0:
        try:
            templ_tuple = [
                (family.template, det_tribe.select(family.template.name).st)]
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
    else:
        try:
            templ_tuple = [(family.template, obspyread(os.path.join(
                template_path, family.template.name + '.mseed')))]
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
    for archive in archives:
        stream_list = extract_detections(
            detection_list, templ_tuple, archive, "SDS",
            request_fdsn=request_fdsn, extract_len=extract_len,
            outdir=None, additional_stations=additional_stations)
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
            stream_list[n_st] += st

    wavefiles = list()
    for stream in stream_list:
        # 2019_09_24t13_38_14
        # 2019-12-15-0323-41S.NNSN__008
        utc_str = str(stream[0].stats.starttime)
        utc_str = utc_str.lower().replace(':', '-').replace('t', '-')
        waveform_filename = wav_out_dir + '/' + utc_str[0:13] + utc_str[14:19]\
            + 'M.EQCS__' + str(len(stream)).zfill(3)
        wavefiles.append(waveform_filename)
        stream.write(waveform_filename, format='MSEED')

    return wavefiles


def replace_templates_for_picking(party, tribe, set_sample_rate=100.0):
    """"
    replace the old templates in the detection-families with those for
    picking (these contain more channels)
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


def check_duplicate_template_channels(tribe):
    """
    Check templates for duplicate channels (happens when there are  P- and
        S-picks on the same channel). Then throw away the S-trace (the later
        one) for now.
    """
    Logger.info('Checking templates in %s for duplicate channels', tribe)
    for template in tribe:
        nt = 0
        channel_ids = list()
        stream_copy = template.st.copy()
        for nt, trace in enumerate(stream_copy):
            if trace.id in channel_ids:
                for j in range(0, nt):
                    test_same_id_trace = stream_copy[j]
                    if trace.id == test_same_id_trace.id:
                        if trace.stats.starttime >= test_same_id_trace.stats.\
                                starttime:
                            if trace in template.st:
                                template.st.remove(trace)
                        else:
                            if test_same_id_trace in template.st:
                                template.st.remove(test_same_id_trace)
                        continue
            else:
                channel_ids.append(trace.id)

        # Also throw away the later pick from the template's event
        np = 0
        pick_ids = list()
        picks_copy = template.event.picks.copy()
        for np, pick in enumerate(picks_copy):
            if pick.waveform_id.id in pick_ids:
                for j in range(0, np):
                    test_same_id_pick = picks_copy[j]
                    if pick.waveform_id.id == test_same_id_pick.waveform_id.id:
                        if pick.time >= test_same_id_pick.time:
                            if pick in template.event.picks:
                                template.event.picks.remove(pick)
                        else:
                            if test_same_id_pick in template.event.picks:
                                template.event.picks.remove(test_same_id_pick)
                        continue
            else:
                pick_ids.append(pick.waveform_id.id)

    return tribe


def extract_detections(detections, templates, archive, arc_type,
                       request_fdsn=False, extract_len=90.0, outdir=None,
                       extract_Z=True, additional_stations=[],
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
    >>> TEST_PATH = os.path.dirname(eqcorrscan.__file__) + '/tests/test_data'
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
    if arc_type == 'SDS':
        from obspy.clients.filesystem.sds import Client
        client = Client(archive)

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
        del delays

        # Reuqest the whole day's stream plus 15 minutes before / after
        starttime = UTCDateTime(detection_day.date) - 15*60
        endtime = starttime + 24.5 * 60 * 60
        bulk = [('*', sta, '*', '*', starttime, endtime)
                for sta in all_stations]
        day_st = get_waveforms_bulk(
                client, bulk, parallel=parallel, cores=cores)
        for detection in day_detections:
            Logger.info(
                'Cutting for detections at: ' +
                detection.detect_time.strftime('%Y/%m/%d %H:%M:%S'))
            t1 = UTCDateTime(detection.detect_time) - extract_len * 1 / 3
            t2 = UTCDateTime(detection.detect_time) + extract_len * 2 / 3
            st = day_st.copy().trim(starttime=t1, endtime=t2)
            if request_fdsn:
                existing_stations = list(set([tr.stats.station for tr in st]))
                other_stations = list(
                    set(all_stations).difference(existing_stations))
                if len(other_stations) > 0:
                    Logger.info('Requesting %s additional stations from FDSN-'
                                'routing client.', len(other_stations))
                    bulk = [
                        ('*', sta, '*', '*', t1, t2) for sta in other_stations]
                    add_st = routing_client.get_waveforms_bulk(bulk)
                    if add_st:
                        add_st = add_st.select(
                            channel='[NDSEHBM]??').trim(t1, t2)
                        st += add_st
                    else:
                        Logger.info('FDSN-request did not return data.')

            if outdir:
                if not os.path.isdir(os.path.join(outdir,
                                                  detection.template_name)):
                    os.makedirs(os.path.join(outdir, detection.template_name))
                st.write(os.path.join(
                    outdir, detection.template_name,
                    detection.detect_time.strftime(
                        '%Y-%m-%d_%H-%M-%S') + '.ms'), format='MSEED')
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
