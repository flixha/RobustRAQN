import os
import glob
import fnmatch
import wcmatch
import subprocess
from pathlib import Path
import pandas as pd
# import matplotlib
from threadpoolctl import threadpool_limits
# import pickle
# import dill

from multiprocessing import Pool, cpu_count, current_process, get_context
# from multiprocessing.pool import ThreadPool
from joblib import Parallel, delayed, parallel_backend
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from itertools import chain, repeat
from collections import Counter
import copy

from obspy.core.event import (Catalog, Pick, Arrival, WaveformStreamID,
                              CreationInfo)
# from obspy.core.stream import Stream
from obspy.core.event import Comment
from obspy.core.inventory.inventory import Inventory
from obspy.io.nordic.core import read_nordic
from obspy import read as obspyread
from obspy import UTCDateTime
from obspy.geodetics.base import degrees2kilometers, locations2degrees
from obspy.io.mseed import InternalMSEEDError, InternalMSEEDWarning
from obspy.io.segy.segy import SEGYTraceReadingError
from obspy.core.util.attribdict import AttribDict

import warnings
warnings.filterwarnings("ignore", category=InternalMSEEDWarning)

from eqcorrscan.core.match_filter import Tribe
from eqcorrscan.core.match_filter.party import Party
from eqcorrscan.core.match_filter.family import Family
from eqcorrscan.utils.correlate import pool_boy
from eqcorrscan.utils.despike import median_filter
from eqcorrscan.utils.mag_calc import _max_p2t
from eqcorrscan.utils.plotting import detection_multiplot
from eqcorrscan.utils.catalog_utils import filter_picks
from eqcorrscan.utils.pre_processing import dayproc, shortproc

from obsplus.events.validate import attach_all_resource_ids
from obsplus.stations.pd import stations_to_df

# import obustraqn.spectral_tools
from robustraqn.obspy.core.stream import Stream
import robustraqn.utils.spectral_tools  # absolute import to avoid circular import
from robustraqn.utils.quality_metrics import get_parallel_waveform_client
from robustraqn.core import seismic_array
# get_station_sites, get_station_sites_dict, mask_array_trace_offsets)
from robustraqn.obspy.clients.filesystem.sds import Client
from timeit import default_timer
import logging
Logger = logging.getLogger(__name__)
logging.basicConfig(
   level=logging.INFO,
   format="%(asctime)s\t%(name)40s:%(lineno)s\t%(funcName)20s()\t%(levelname)s\t%(message)s")


MY_ENV = os.environ.copy()
MY_ENV["SEISAN_TOP"] = '/home/felix/Software/SEISANrick'


def _read_nordic(sfile, unused_kwargs=True, **kwargs):
    """
    Internal function to log while reading Nordic file. 
    
    :type sfile: str
    :param sfile: Path to Nordic file
    :type unused_kwargs: bool
    :param unused_kwargs:
        Parameter to pass to some functions in obspy (which could otherwise
        raise an error).
    :type kwargs: dict
    :param kwargs: Additional kwargs to pass to read_nordic.

    :returns: obspy Catalog object
    :rtype: :class:`obspy.core.event.Catalog`
    """
    Logger.info('Reading sfile %s', sfile)
    select = read_nordic(sfile, unused_kwargs=unused_kwargs, **kwargs)
    return select


def read_seisan_database(database_path, cores=1, nordic_format='UKN',
                         starttime=None, endtime=None,
                         check_resource_ids=True):
    """
    Reads all S-files in Seisan database, which can be either a folder
    containing all Sfiles or a YYYY/MM/Sfiles-structure.
    Outputs a catalog of events, with

    :type database_path: str
    :param database_path: path to Seisan-database
    :type cores: int
    :param cores: number of cores to use for reading multiple S-files.
    :type nordic_format: str
    :param nordic_format:
        'UKN', 'NEW', or 'OLD; specifies the version of the Nordic S-files.
    :type starttime: obspy.UTCDatetime or None
    :param starttime:
        Sets a lower limit for files to be read in. If None, reads all files
        older than endtime.
    :type endtime: obspy.UTCDatetime or None
    :param endtime:
        Sets an upper limit for files to be read in. If None, reads all files
        younger than starttime.

    :returns: Catalog of events
    :rtype: :class:`obspy.core.event.Catalog`
    """
    database_path = os.path.expanduser(database_path)
    gsfiles = glob.glob(os.path.join(database_path, '*.S??????'))
    gsfiles += glob.glob(os.path.join(database_path, '????', '??', '*.S??????'))
    gsfiles.sort(key=lambda x: x[-6:])
    sfiles = []
    if starttime or endtime:
        for sfile in gsfiles:
            sfile_name = os.path.normpath(sfile).split(os.path.sep)[-1]
            # Check if seconds are out of range for UTCDatetime / strftime
            # functions
            if sfile_name[8:10] == '60':
                sfile_name = sfile_name[0:8] + '59' + sfile_name[10:]
            sfile_time = UTCDateTime(sfile_name[13:] + sfile_name[0:10],
                                     strict=False)
            if ((not starttime or sfile_time >= starttime) and 
                    (not endtime or sfile_time <= endtime)):
                sfiles.append(sfile)
    else:
        sfiles = gsfiles

    cats = Parallel(n_jobs=cores)(delayed(_read_nordic)(
        sfile, nordic_format=nordic_format) for sfile in sfiles)
    cat = Catalog([cat[0] for cat in cats if cat])
    for event, sfile in zip(cat, sfiles):
        event.comments.append(Comment(text='Sfile-name: ' + sfile))
        if not hasattr(event, 'extra'):
            event.extra = AttribDict()
        event.extra.update(
            {'sfile_name': {'value': sfile, 'namespace': 'Seisan'}})

        if check_resource_ids:
            attach_all_resource_ids(event)
    # attach_all_resource_ids(cat)
    # validate_catalog(cat)
    cat = Catalog(sorted(cat, key=lambda x: (
        x.preferred_origin() or x.origins[0]).time))

    return cat


def load_event_stream(
        event, sfile='', seisan_wav_path=None, selected_stations=[],
        clients=[], st=Stream(), min_samp_rate=np.nan, pre_event_time=30,
        allowed_band_codes="ESBHNMCFDX", forbidden_instrument_codes="NGAL",
        allowed_component_codes="ZNE0123ABCXYRTH", channel_priorities="HBSEN*",
        template_length=300, search_only_month_folders=True, bulk_rejected=[],
        wavetool_path='/home/felix/Software/SEISANrick/PRO/linux64/wavetool',
        wav_suffixes=['', '.gz'], unused_kwargs=True, cores=1, **kwargs):
    """
    Load the waveforms for an event file (here: Nordic file) while performing
    some checks for duplicates, incompleteness, etc.

    :type event: :class:`obspy.core.event.Event`
    :param event: Event for which to load the stream from attached filename
    :type sfile: str
    :param sfile:
        Path to Nordic event file from which to load event / waveform path.
    :type seisan_wav_path
    
    :type selected_stations
    
    :type clients: list of :class:`obspy.clients`
    :param clients:
    :type st: :class:`obspy.core.stream.Stream`
    :param st:
    :type min_samp_rate: float
    :param min_samp_rate: minimum sampling rate to retain streams
    :type pre_event_time: float
    :param pre_event_time: time (in s) to trim waveforms before event
    :type allowed_band_codes: str
    :param allowed_band_codes: allowed SEED band codes (1st char of channel ID)
    :type forbidden_instrument_codes: str
    :param forbidden_instrument_codes: forbidden SEED instrument codes (2nd)
    :type allowed_component_codes: str
    :param allowed_component_codes: allowed SEED component codes (3rd char)
    :type channel_priorities: str
    :param channel_priorities:
        Band codes in order of priority - when multiple traces with different
        band codes are available for station, will keep the highest priority
        one.
    :type template_length: float
    :param template_length: Length to trim stream to in seconds.
    :type search_only_month_folders: bool
    :param search_only_month_folders:
        When using Seisan REA / WAV database, limit search for waveform file to
        relevant year/month-folder.
    :type bulk_rejected: list
    :param bulk_rejected:
        Trace-IDs that were rejected in bulk request due to data quality
        concerns.
    :type wavetool_path: str
    :param wavetool_path:
        Path to Seisan's wavetool program that can handle conversion of old
        (version < 6) Seisan waveform files that Obspy does not understand.
    :type wav_suffixes: list
    :param wav_suffixes:
        List of file possible file suffixes, in case waveform files were
        compressed and a file suffix added to original filename.
    :type unused_kwargs: bool
    :param unused_kwargs:
        Whether some called Obspy function should accept unknown kwargs.
    :type cores: int
    :param cores: number of cores to use when reading in parallel
    """
    origin = event.preferred_origin() or event.origins[0]
    # Stop event processing if there are no waveform files
    wavfilenames = []
    if isinstance(sfile, str) and len(sfile) > 0:
        select, wavname = read_nordic(sfile, return_wavnames=True,
                                      unused_kwargs=unused_kwargs, **kwargs)
        wavfilenames = wavname[0]

    # # Read extra wavefile names from comments
    # if search_only_month_folders:
    #     seisan_wav_path = os.path.join(seisan_wav_path, str(origin.time.year),
    #                                    "{:02d}".format(origin.time.month))
    # for comment in event.comments.copy():
    #     if 'Waveform-filename:' in comment.text:
    #         wav_file = comment.text.removeprefix('Waveform-filename: ')
    #         if 'ARC _' in wav_file:
    #             continue
    #         out_lines = subprocess.check_output(
    #                 "find {} -iname '{}'".format(seisan_wav_path, wav_file),
    #                 shell=True).splitlines()
    #         if len(out_lines) == 0:
    #             continue
    #         if type(out_lines[0]) == bytes:
    #             wav_file_paths = [line.decode('UTF-8') for line in out_lines]
    #         else:
    #             wav_file_paths = [line[2:] for line in out_lines]
    #         if wav_file_paths:
    #             wavfilenames.append(wav_file_paths[0])
    #             # Should I remove the old waveform-filelinks ?
    #             # event.comments.remove(comment)

    for comment in event.comments.copy():
        if 'Waveform-filename:' in comment.text:
            wav_file = comment.text.removeprefix('Waveform-filename: ')
            if wav_file not in wavfilenames:
                wavfilenames.append(wav_file)

    if not wavfilenames:
        Logger.warning('Event %s: no waveform file links in sfile', sfile)
    # If a path is provided, read waveform files from filenames. If waveform
    # filepaths are absolute, then set seisan_wav_path to ''
    if seisan_wav_path is not None:
        for wav_file in wavfilenames:
            # Check that there are proper mseed/SEISAN7.0 waveform files
            if (wav_file[0:3] == 'ARC' or wav_file[0:4] == 'WAVE' or
                    wav_file[0:6] == 'ACTION' or wav_file[0:6] == 'OLDACT'):
                continue

            # Check if the wav-file is in the main folder or Seisan year-month
            # subfolders
            wav_file_found = False
            for wav_suffix in wav_suffixes:
                full_wav_file = os.path.join(
                    seisan_wav_path, wav_file + wav_suffix)
                if not os.path.isfile(full_wav_file):
                    full_wav_file = os.path.join(
                        seisan_wav_path, str(origin.time.year),
                        "{:02d}".format(origin.time.month),
                        wav_file + wav_suffix)
                    # Check for station's subdirectory just above WAV-path
                    # check e.g. file 2005-07-09-1833-27S.BER___003 in folder
                    # BER__/2005/07
                    if not os.path.isfile(full_wav_file):
                        if len(wav_file.split('.')) > 1:
                            full_wav_file = os.path.join(
                                os.path.dirname(seisan_wav_path),
                                wav_file.split('.')[1][0:5],
                                str(origin.time.year),
                                "{:02d}".format(origin.time.month),
                                wav_file + wav_suffix)
                        if os.path.isfile(full_wav_file):
                            wav_file_found = True
                            break
                    else:
                        wav_file_found = True
                        break
                else:
                    wav_file_found = True
                    break
            if not wav_file_found:
                Logger.warning('Could not find waveform file %s', wav_file)
                continue
            else:
                Logger.info('Found wav-file %s', full_wav_file)

            try:
                st += obspyread(full_wav_file)
            except FileNotFoundError as e:
                Logger.warning(
                    'Waveform file %s does not exist', full_wav_file)
                Logger.warning(e)
            except PermissionError as e:
                Logger.warning(
                    'Could not read waveform file %s', full_wav_file)
                Logger.warning(e)
            except (TypeError, ValueError, AssertionError,
                    SEGYTraceReadingError, InternalMSEEDError,
                    NotImplementedError) as e:
                Logger.warning(
                    'Could not read waveform file %s', full_wav_file)
                Logger.warning(e)
                Path(os.path.join(os.getcwd(), 'TMP')).mkdir(
                    parents=True, exist_ok=True)
                wav_file_name = os.path.normpath(
                    wav_file).split(os.path.sep)[-1]
                new_wav_file_name = os.path.join(
                    'TMP', wav_file_name + '.mseed')
                # If Obspy cannot read file, try to convert it with Seisan's
                # wavetool:
                Logger.info('Trying to use wavetool to convert %s:', wav_file)
                subprocess.run(
                    [wavetool_path + " " +
                    " -wav_in_file {}".format(full_wav_file) +
                    " -wav_out_file {}".format(new_wav_file_name) +
                    " -format MSEED"], shell=True, env=MY_ENV)
                try:
                    st += obspyread(new_wav_file_name)
                except FileNotFoundError:
                    Logger.error('Could not read converted file %s, skipping.',
                                 new_wav_file_name)
        Logger.info(
            'Event %s (sfile %s): read %s traces from event-based waveform'
            ' files', event.short_str(), sfile, str(len(st)))

    # Request waveforms from client
    try:
        latest_pick = max([p.time for p in event.picks])
    except ValueError:
        latest_pick = origin.time + 2 * template_length
    t1 = origin.time - pre_event_time
    t2 = (latest_pick + template_length + 10) or (
        origin.time + template_length * 2)
    if clients:
        bulk_request = [("??", s, "*", "?H?", t1, t2)
                        for s in selected_stations]
    for client in clients:
        client = get_parallel_waveform_client(client)
        Logger.info('Requesting waveforms from client %s', client)
        outtic = default_timer()
        # add_st = client.get_waveforms_bulk_parallel(
        #     bulk_request, parallel=False, cores=cores)
        add_st = client.get_waveforms_bulk(
            bulk_request, parallel=False, cores=cores)
        outtoc = default_timer()
        Logger.info(
            'Received %s traces from client for one event / file, which took:'
            ' {0:.4f}s'.format(outtoc - outtic), len(add_st))
        st += add_st
    if len(st) == 0:
        Logger.warning('Did not find any waveforms for sfile %s', sfile)
        return None
    # If requested, make sure to remove traces that would not have passed the
    # quality-metrics check:
    for trace_reject in bulk_rejected:
        tr_id = '.'.join(trace_reject[0:4])
        st_tr = st.select(tr_id)
        for tr in st_tr:
            Logger.info(
                'Removed trace %s for event %s because its metrics are not '
                'within selected quality thresholds', tr.id, event.short_str())
            st.remove(tr)

    # remove_st = Stream()
    # for tr in st:
    #     for trace_reject in bulk_rejected:
    #         if '.'.join(trace_reject[0:3]) == tr.id:
    #             remove_st += tr
    # for tr in remove_st:
    #     Logger.info('Removed trace %s for event %s because its metrics are '
    #                 'selected thresholds', tr.id, event.short_str())
    #     st.remove(tr)

    n_tr_before = len(st)
    # Trim so that any function-supplied streams are the right length as well
    st.trim(starttime=t1, endtime=t2, pad=False, nearest_sample=True)

    # REMOVE UNUSABLE CHANNELS
    st_copy = st.copy()
    for tr in st_copy:
        # channels without signal
        # if sum(tr.copy().detrend().data)==0 and tr in st:
        n_samples_nonzero = np.count_nonzero(tr.copy().detrend().data)
        if n_samples_nonzero == 0 and tr in st:
            st = st.remove(tr)
            continue
        # channels with empty channel code:
        if len(tr.stats.channel) == 0 and tr in st:
            st = st.remove(tr)
            continue
        # channels with undecipherable channel names
        # potential names: MIC, SLZ, S Z, M, ALE, MSF, TC
        if tr.stats.channel[0] not in allowed_band_codes and tr in st:
            st = st.remove(tr)
            continue
        if (tr in st and len(tr.stats.channel) == 3 and
                tr.stats.channel[-1] not in allowed_component_codes):
            st = st.remove(tr)
            continue
        # channels from accelerometers/ gravimeters/ tiltmeter/low-gain seism.
        if len(tr.stats.channel) == 3 and tr in st:
            if tr.stats.channel[1] in forbidden_instrument_codes:
                st = st.remove(tr)
                continue
        # channels whose sampling rate is lower than the one chosen for the
        # templates
        if tr.stats.sampling_rate < 0.99 * min_samp_rate:
            st = st.remove(tr)
            continue

    # Adjust some strange channel / location names
    for tr in st:
        # Adjust messed up channel/location codes from some files, e.g.:
        # .BER.Z.HH
        if (len(tr.stats.channel) == 2 and len(tr.stats.location) == 1 and
                tr.stats.location[0] in 'ZNE'):
           tr.stats.channel = tr.stats.channel + tr.stats.location
           tr.stats.location = ''
        
        # ADJUST unsupported, but decipherable codes
        # Adjust single-letter location codes:
        if len(tr.stats.location) == 1:
            tr.stats.location = tr.stats.location + '0'
        # Adjust empty-letter channel codes: ## DO NOT ADJUST YET - IT MAY
        # WORK LIKE THIS WITH PROPER INVENTORY
        # if tr.stats.channel[1] == ' ' and len(tr.stats.channel) == 3:
        #    tr.stats.channel = tr.stats.channel[0] + 'H' + tr.stats.channel[2]

        # try:  # Convert special encodings to something obspy can write
        # what about: , 'SRO', 'GEOSCOPE', 'CDSN'
        # if tr.stats.mseed.encoding in ['DWWSSN']:

    # Add channels to template, but check if similar components are already
    # present. first, expand input to wildcarded list:
    channel_priorities = [chan_code + "*" for chan_code in channel_priorities]
    wave_at_sel_stations = Stream()
    for station in selected_stations:
        for channel_priority in channel_priorities:
            wave_alrdy_at_sel_station = wave_at_sel_stations.select(
                station=station)
            if not wave_alrdy_at_sel_station:
                add_wavs = st.select(station=station, channel=channel_priority)
                wave_at_sel_stations += add_wavs
    # If there are more than one traces for the same station-component-
    # combination, then choose the "best" trace
    wave_at_sel_stations_copy = wave_at_sel_stations.copy()
    for tr in wave_at_sel_stations_copy:
        same_sta_chan_st = wave_at_sel_stations.select(
            station=tr.stats.station, channel='*'+tr.stats.channel[-1])
        remove_tr_st = Stream()
        keep_tr_st = Stream()
        n_same_sta_chan_wav = len(same_sta_chan_st)
        if n_same_sta_chan_wav > 1:
            # 1. best trace: highest sample rate
            samp_rates = [t.stats.sampling_rate for t in same_sta_chan_st]
            keep_tr_st = same_sta_chan_st.select(sampling_rate=max(samp_rates))
            # 2. best trace: longest trace
            trace_lengths = [t.stats.npts for t in same_sta_chan_st]
            keep_tr_st = same_sta_chan_st.select(npts=max(trace_lengths))
            # 3. best trace: more complete metadata - location code
            if len(keep_tr_st) == 0 or len(keep_tr_st) > 1:
                loccodes = [t.stats.location for t in same_sta_chan_st]
                if any(locc == '' for locc in loccodes) and\
                   any(locc == '??' for locc in loccodes):
                    remove_tr_st += same_sta_chan_st.select(location="")
                    keep_tr_st += same_sta_chan_st.select(
                        sampling_rate=max(samp_rates), location="?*")
            # 4 best trace: more complete metadata - network code
            if len(keep_tr_st) == 0 or len(keep_tr_st) > 1:
                netcodes = [t.stats.network for t in same_sta_chan_st]
                if (any(n == '' for n in netcodes)
                        and any(n == '??' for n in netcodes)):
                    remove_tr_st += same_sta_chan_st.select(network="")
                    keep_tr_st += same_sta_chan_st.select(
                        sampling_rate=max(samp_rates), location="?*",
                        network="??")
                if len(keep_tr_st) > 1:
                    keep_tr_st = Stream() + keep_tr_st[0]
            for tt in same_sta_chan_st:
                wave_at_sel_stations.remove(tt)
            wave_at_sel_stations += keep_tr_st

    # Double-check to remove duplicate channels
    # wave_at_sel_stations.merge(method=0, fill_value=0, interpolation_samples=0)
    # 2021-01-22: changed merge method to below one to fix error with
    #             incomplete day.
    wave_at_sel_stations.merge(method=1, fill_value=0,
                               interpolation_samples=-1)
    k = 0
    channel_ids = list()
    for trace in wave_at_sel_stations:
        if trace.id in channel_ids:
            for j in range(0, k):
                test_same_id_trace = wave_at_sel_stations[j]
                if trace.id == test_same_id_trace.id:
                    if (trace.stats.starttime >=
                            test_same_id_trace.stats.starttime):
                        wave_at_sel_stations.remove(trace)
                    else:
                        wave_at_sel_stations.remove(test_same_id_trace)
        else:
            channel_ids.append(trace.id)
            k += 1
    st = wave_at_sel_stations
    # Preprocessing
    # cut around origin plus some

    # Don't trim the stream if that means you are padding with zeros
    starttime = t1
    endtime = t2
    st.trim(starttime=starttime, endtime=endtime, pad=False,
            nearest_sample=True)

    # don't use the waveform if more than 5% is zero
    non_zero_wave = Stream()
    for tr in st:
        n_nonzero = np.count_nonzero(tr.copy().detrend().data)
        # if (sum(tr.copy().detrend().data==0) < tr.data.size*0.05 and not\
        if (n_nonzero > tr.data.size * 0.95 and not any(np.isnan(tr.data))):
            non_zero_wave.append(tr)
    st = non_zero_wave
    n_tr_after = len(st)
    Logger.info('Event %s (sfile %s): %s out of %s traces remaining after '
                'initial selection.', event.short_str(), sfile,
                str(n_tr_after), str(n_tr_before))

    return st


def prepare_detection_stream(
        st, tribe, parallel=False, cores=None, ispaq=pd.DataFrame(),
        try_despike=False, downsampled_max_rate=None,
        accepted_band_codes='HBSENMCFDX', forbidden_instrument_codes='NGAL',
        accepted_component_codes='ZNE0123ABCDH'):
    """
    Prepare the streams for being used in EQcorrscan. The following criteria
    are being considered:
     - channel code is not empty
     - sampling rate is not much lower than the sampling rate of the tribes
     - band code is in list of allowed characters
     - instrument code is not in list of forbidden characters (a list of
       forbidden chars is used because in Old Seisan files, this is often "0",
       rather than indicating a real instrument)    
     - component code is in list of allowed characters
     - despiking can be done when ispaq-stats indicate there are spikes

    :type st: :class:`obspy.core.stream.Stream`
    :param st: the day's streams
    :type tribe: :class:`eqcorrscan.core.match_filter.Tribe`
    :param tribe: Tribe of templates
    :type parallel: boolean
    :param parallel: Whether to run some processing steps in parallel
    :type cores: int
    :param cores: number of cores
    :type ispaq: pandas.DataFrame()
    :param ispaq:
        ispaq/Mustang-generated data quality metrics, including 'num_spikes'.
    :type try_despike: bool
    :param try_despike:
        Whether to try to despike the data. Not always successful.
    :type accepted_band_codes: str
    :param accepted_band_codes: string of the acceptable band codes
    :type forbidden_instrument_codes: str
    :param forbidden_instrument_codes: string of the forbidden instrument codes
    :type accepted_component_codes: str
    :param accepted_component_codes: string of the acceptable component codes

    :returns: :class:`obspy.core.stream.Stream`
    """
    tr_rates = [tr.stats.sampling_rate for templ in tribe for tr in templ.st]
    if len(tr_rates) == 0:
        return st
    min_samp_rate = min(list(set(tr_rates)))
    # REMOVE UNUSABLE CHANNELS
    # st_copy = st.copy()
    st_of_tr_to_be_removed = Stream()
    # for tr in st_copy:
    for tr in st:
        # channels with empty channel code:
        if len(tr.stats.channel) == 0:
            st_of_tr_to_be_removed += tr
            Logger.info('Removing trace %s because the channel code is empty',
                        tr.id)
            continue
        # channels with undecipherable channel names
        # potential names: MIC, SLZ, S Z, M, ALE, MSF, TC
        if tr.stats.channel[0] not in accepted_band_codes:
            # st = st.remove(tr)
            st_of_tr_to_be_removed += tr
            Logger.info('Removing trace %s because the band code is not in %s',
                        tr.id, accepted_band_codes)
            continue
        if tr.stats.channel[-1] not in accepted_component_codes:
            # st = st.remove(tr)
            st_of_tr_to_be_removed += tr
            Logger.info('Removing trace %s because the component code is not '
                        'in %s', tr.id, accepted_band_codes)
            continue
        # channels from accelerometers/ gravimeters/ tiltmeter/low-gain seism.
        if len(tr.stats.channel) == 3:
            if tr.stats.channel[1] in forbidden_instrument_codes:
                # st = st.remove(tr)
                st_of_tr_to_be_removed += tr
                Logger.info('Removing trace %s because the instrument code is '
                            'in %s', tr.id, forbidden_instrument_codes)
                continue
        # Here we still need to allow somewhat incorrect sampling rates
        if tr.stats.sampling_rate < min_samp_rate * 0.95:
            st_of_tr_to_be_removed += tr
            Logger.info('Removing trace %s because the sampling rate is too '
                        'low: %s', tr.id, tr.stats.sampling_rate)
            continue
        # ADJUST unsupported, but decipherable codes
        # Adjust single-letter location codes:
        if len(tr.stats.location) == 1:
            tr.stats.location = tr.stats.location + '0'

    # Remove unsuitable traces from stream
    for tr in st_of_tr_to_be_removed:
        if tr in st:
            st.remove(tr)
    # For NORSAR data, replace Zeros with None

    # Do resampling if requested (but only on channels with higher rate):
    # if downsampled_max_rate is not None:
    #    for tr in st:
    #        if tr.stats.sampling_rate > downsampled_max_rate:
    #            tr.resample(sampling_rate=downsampled_max_rate)

    # Detrend and merge
    # st = parallel_detrend(st, parallel=True, cores=None, type='simple')
    # st.detrend(type = 'linear')

    # If I merge the traces before removing the response, then the masked
    # arrays / None-values removal will mess up the response-corrected trace
    # and make it all None.
    # st.merge(method=0, fill_value=0, interpolation_samples=0)
    # If using Zero as a fill_value, then EQcorrscan will not be able to
    # automatically recognize these times and exclude the correlation-
    # value on those traces from the stacked value.
    # st.merge(method=0, fill_value=0, interpolation_samples=0)

    # TODO write despiking smartly
    if try_despike:
        # st_dspike = st.copy()
        for tr in st:
            # starttime = tr.stats.starttime
            # endtime = tr.stats.endtime
            # mid_t = starttime + (endtime - starttime)/2
            # # The reqtimes are those that are listed in the ispaq-stats per
            # # day
            # # (starts on 00:00:00 and ends on 23:59:59).
            # reqtime1 = UTCDateTime(mid_t.year, mid_t.month, mid_t.day,0,0,0)
            # reqtime2 = UTCDateTime(mid_t.year,mid_t.month,mid_t.day,23,59,59)

            # day_stats = stats[stats['start'].str.contains(
            #     str(reqtime1)[0:19])]
            chn_stats = ispaq[ispaq['target'].str.contains(tr.id)]
            # target = availability.iloc[0]['target']
            # num_spikes = ispaq[(ispaq['target']==target) &
            # (day_stats['metricName']=='num_spikes')]
            num_spikes = chn_stats[chn_stats['metricName'] == 'num_spikes']
            if len(num_spikes) > 0:
                if num_spikes.iloc[0]['value'] == 0:
                    continue
            Logger.warning('%s: %s spikes, attempting to despike', str(tr),
                           num_spikes.iloc[0]['value'])
            try:
                tr = median_filter(tr, multiplier=10, windowlength=0.5,
                                   interp_len=0.1)
                Logger.warning('Successfully despiked, %s', str(tr))
                # testStream = Stream()
                # testStream += testtrace
                # _spike_test(testStream, percent=0.98, multiplier=5e4)
            except Exception as e:
                Logger.warning('Failed to despike %s: %s', str(tr))
                Logger.warning(e)

                # tracelength = (testtrace.stats.npts /
                #                testtrace.stats.sampling_rate)
                #         print('Despiking ' + contFile )
                # testtrace = median_filter(testtrace, multiplier=10,
                #         windowlength=tracelength, interp_len=0.1, debug=0)
                # test for step functions which mess up a lot
            # try:
            #     diftrace = tr.copy()
            #     diftrace.differentiate()
            #     difStream = Stream()
            #     difStream += diftrace
            #     _spike_test(difStream, percent=0.9999, multiplier=1e1)
            # except MatchFilterError:
            #     tracelength = (diftrace.stats.npts /
            #                    diftrace.stats.sampling_rate)
            #     print('Despiking 1st differential of ' + contFile)
            #     testtrace = median_filter(diftrace, multiplier=10,
            #             windowlength=tracelength, interp_len=0.1, debug=0)
            #     testtrace.integrate()
    return st


def parallel_detrend(st, parallel=True, cores=None, type='simple'):
    """
    """
    # Detrend in parallel across traces
    if not parallel:
        st.detrend(type=type)
    else:
        if cores is None:
            cores = min(len(st), cpu_count())
        # with pool_boy(Pool=Pool, traces=len(st), cores=cores) as pool:
        #     results = [pool.apply_async(tr.detrend, {type})
        #                for tr in st]
        # traces = [res.get() for res in results]
        traces = Parallel(n_jobs=cores)(delayed(tr.detrend)(type) for tr in st)
        st = Stream(traces=traces)
    return st


def parallel_merge(st, method=0, fill_value=0, interpolation_samples=0,
                   cores=1):
    seed_id_list = [tr.id for tr in st]
    unique_seed_id_list = list(dict.fromkeys(seed_id_list))

    stream_list = [st.select(id=seed_id) for seed_id in unique_seed_id_list]
    # with pool_boy(
    #         Pool=Pool, traces=len(unique_seed_id_list), cores=cores) as pool:
    #     results = [pool.apply_async(
    #         trace_st.merge, {method, fill_value, interpolation_samples})
    #                 for trace_st in stream_list]
    # streams = [res.get() for res in results]
    # st = Stream()
    # for trace_st in streams:
    #     for tr in trace_st:
    #         st.append(tr)

    streams = Parallel(n_jobs=cores)(
        delayed(trace_st.merge)(method, fill_value, interpolation_samples)
        for trace_st in stream_list)
    st = Stream()
    for trace_st in streams:
        for tr in trace_st:
            st.append(tr)

    # st.merge(method=0, fill_value=0, interpolation_samples=0)
    return st


def robust_rotate(stream, inventory, method="->ZNE"):
    """
    Rotation with error catching for parallel execution.
    """
    if len(stream) > 0:
        return stream
    try:
        stream = stream.rotate(method, inventory=inventory)
    except Exception as e:
        try:
            st_id = ' '
            st_time = ' '
            if len(stream) > 0:
                st_id = stream[0].id
                st_time = str(stream[0].stats.starttime)[0:19]
            Logger.warning('Cannot rotate traces for station %s on %s: %s',
                           st_id, st_time, e)
        except IndexError as e2:
            Logger.warning('Cannot rotate traces: %s --- %s', e, e2)
    return stream


def parallel_rotate(st, inv, parallel=True, cores=None,
                    thread_parallel=False, n_threads=1, method="->ZNE"):
    """
    wrapper function to rotate 3-component seismograms in a stream in parallel.
    """
    net_sta_loc = [(tr.stats.network, tr.stats.station, tr.stats.location)
                   for tr in st]
    # Sort unique-ID list by most common, so that 3-component stations
    # appear first and are processed first in parallel loop (for better load-
    # balancing).
    net_sta_loc = list(chain.from_iterable(repeat(item, count)
                       for item, count in Counter(net_sta_loc).most_common()))
    unique_net_sta_loc_list = list(dict.fromkeys(net_sta_loc))
    # Need to sort list by original order after set()
    # sorted(set(net_sta_loc), key=lambda x: net_sta_loc.index(x))
    # stream_list = [st.select(id=seed_id) for seed_id in unique_seed_id_list]
    if cores is None:
        cores = min(len(unique_net_sta_loc_list), cpu_count())

    if thread_parallel and not parallel:
        with parallel_backend('threading', n_jobs=cores):
            streams = Parallel(n_jobs=n_threads, prefer='threads')(delayed(
                st.select(
                    network=nsl[0], station=nsl[1], location=nsl[2]
                    ).robust_rotate)(inv.select(
                        network=nsl[0], station=nsl[1], location=nsl[2]),
                                     method=method)
                    for nsl in unique_net_sta_loc_list)
    else:
        streams = Parallel(n_jobs=cores)(delayed(
            st.select(network=nsl[0], station=nsl[1], location=nsl[2]
                      ).robust_rotate)(
                          inv.select(network=nsl[0], station=nsl[1],
                                     location=nsl[2]), method=method)
                      for nsl in unique_net_sta_loc_list)
    st = Stream([tr for trace_st in streams for tr in trace_st])
    # for trace_st in streams:
    #     for tr in trace_st:
    #         st.append(tr)

    # st.merge(method=0, fill_value=0, interpolation_samples=0)
    return st


def daily_plot(st, year, month, day, data_unit='counts', suffix=''):
    """
    """
    for tr in st:
        out_plot_file = os.path.join('DebugPlots', str(year)
                                   + str(month).zfill(2) + str(day).zfill(2)
                                   + '_' + tr.stats.station
                                   + '_' + tr.stats.channel + suffix + '.png')
        tr.plot(type='dayplot', size=(1900, 1080), outfile=out_plot_file,
                data_unit=data_unit)


def get_matching_trace_for_pick(pick, stream):
    """
    find the trace-id that matches a pick to a suitable trace in stream

    pick for which to find the suitable trace
    stream of relevant traces (i.e., for the same station as the pick is)
    """
    k = None
    avail_tr_ids = [tr.id for tr in
                    stream.select(station=pick.waveform_id.station_code)]
    avail_nets = [tr_id.split('.')[0] for tr_id in avail_tr_ids]
    avail_locs = [tr_id.split('.')[2] for tr_id in avail_tr_ids]
    avail_chans = [tr_id.split('.')[3] for tr_id in avail_tr_ids]
    pick_id = pick.waveform_id.id
    pick_net_chan = pick_id.split('.')[0] + '.' + pick_id.split('.')[3]
    pick_loc_chan = '.'.join(pick_id.split('.')[2:])
    avail_nets_chans = [
        id.split('.')[0] + '.' + id.split('.')[3] for id in avail_tr_ids]
    avail_locs_chans = ['.'.join(id.split('.')[2:]) for id in avail_tr_ids]
    # for k, code in enumerate(avail_tr_ids):
    # Check available traces for suitable network/location/channel
    if pick.waveform_id.id in avail_tr_ids:
        k = avail_tr_ids.index(pick.waveform_id.id)
    # 1. only location code doesn't match
    elif pick_net_chan in avail_nets_chans:
        k = avail_nets_chans.index(pick_net_chan)
    # 2. network code doesn't match
    elif pick_loc_chan in avail_locs_chans:
        k = avail_locs_chans.index(pick_loc_chan)
    # 3 if not found, check if the channel code is missing a letter
    elif (' ' in pick.waveform_id.channel_code
            or len(pick.waveform_id.channel_code) <= 2):
        if len(pick.waveform_id.channel_code) == 2:
            pick.waveform_id.channel_code = (
                pick.waveform_id.channel_code[0] + ' '
                + pick.waveform_id.channel_code[1])
        elif len(pick.waveform_id.channel_code) == 1:
            pick.waveform_id.channel_code = (
                '  ' + pick.waveform_id.channel_code[1])
        pick_id = pick.waveform_id.id
        pick_chan_wildcarded = pick_id.split('.')[3].replace(' ', '?')
        pick_id_wildcarded = '.'.join(
            pick_id.split('.')[0:3] + [pick_chan_wildcarded])
        matches = [(i, id) for i, id in enumerate(avail_tr_ids)
                   if fnmatch.fnmatch(id, pick_id_wildcarded)]
        if matches:
            k = matches[0][0]
        else:
            # 4 if not found, allow space in channel and wrong network
            pick_loc_chan_wildcarded = (pick_id.split('.')[1] + '.'
                                        + pick_chan_wildcarded)
            matches = [(i, id) for i, id in enumerate(avail_locs_chans)
                       if fnmatch.fnmatch(id, pick_loc_chan_wildcarded)]
            if matches:
                k = matches[0][0]
            # 6 if not found, allow space in channel and wrong location
            if k is None:
                pick_net_chan_wildcarded = (pick_id.split('.')[0]
                                            + pick_chan_wildcarded)
                matches = [
                    (i, id) for i, id in enumerate(avail_nets_chans)
                    if fnmatch.fnmatch(id, pick_net_chan_wildcarded)]
                if matches:
                    k = matches[0][0]
            # 7 if not found, allow space in channel, wrong network,
            #   and wrong location
            if k is None:
                matches = [
                    (i, id) for i, id in enumerate(avail_chans)
                    if fnmatch.fnmatch(id, pick_chan_wildcarded)]
                if matches:
                    k = matches[0][0]
    if k is None:
        Logger.debug('Could not find matching trace for pick on %s',
                     pick.waveform_id.id)
        return None, None
    chosen_tr_id = '.'.join([avail_nets[k], pick.waveform_id.station_code,
                             avail_locs[k], avail_chans[k]])
    pick.waveform_id.network_code = avail_nets[k]
    pick.waveform_id.location_code = avail_locs[k]
    pick.waveform_id.channel_code = avail_chans[k]

    return chosen_tr_id, k


def _make_append_new_pick_and_arrival(
        origin, st, tr, phase, new_picks, new_arrivals, app_vel, dist_deg,
        min_snr=1, pre_pick=0, winlen=2, *args, **kwargs):
    """
    creates a new pick from an apparent velocity and adds it to the picks- and
    arrivals-list.
    """
    tr = tr.copy()
    if tr.stats.station in [
        p.waveform_id.station_code for p in new_picks
        if (p.phase_hint == phase or p.phase_hint == phase[0])]:
        return new_picks, new_arrivals
    dist_km = degrees2kilometers(dist_deg)
    hor_chans = st.select(id=tr.id[0:-1] + '[NERT12XY]')
    if (phase[0] == 'S' and
            (tr.stats.channel[-1] in 'NERT12XY' or len(hor_chans) == 0)):
        chans = hor_chans or st.select(id=tr.id[0:-1] + 'Z')
    elif phase[0] == 'P':
        chans = st.select(id=tr.id[0:-1] + 'Z')
    else:
        return new_picks, new_arrivals
    for chan in chans:
        new_pick = Pick(
            phase_hint=phase, force_resource_id=True,
            time=origin.time + dist_km / app_vel,
            waveform_id=WaveformStreamID(seed_string=chan.id),
            evaluation_mode='automatic', onset='emergent',
            creation_info=CreationInfo(agency_id='RR', author=''))
        new_arrival = Arrival(phase=phase, pick_id=new_pick.resource_id,
                            force_resource_id=True, distance=dist_deg)
        new_picks.append(new_pick)
        new_arrivals.append(new_arrival)

    return new_picks, new_arrivals


def compute_picks_from_app_velocity(
        event, origin, stream=Stream(), pick_calculation=None,
        crossover_distance_km=100, app_vel_Pg=7.2, app_vel_Sg=4.1,
        app_vel_Pn=8.1, app_vel_Sn=4.6, *args, **kwargs):
    """
    Computes theoretical arrival-picks based on apparent velocities of Pg/Pn/
    Sg/Sn for all traces associated with an event. This is useful when one
    wants to correlate all available traces for cross-correlation based
    relocation, and when there's plenty more traces than picks available that
    may be useful. Make sure to exclude low-CCC observations at some later
    point in the earthquake relocation process.

    :type pick_calculation: string
    :param:
        one of 'filter_direct', 'only_direct', 'filter_refracted',
        'only_refracted'
    """
    Logger.info('Computing theoretical picks for %s arrivals',
                pick_calculation)
    # For stations closer than XX km: assume that first arrival is Pg / Sg
    # For stations farther than XX km: Check if there is a picked Pg / Sg,
    # otherwise get approximate times of the direct arrivals based on group
    # velocities.
    new_picks = []
    new_arrivals = []
    # FIRST, deal with existing Picks
    for pick in event.picks:
        arrivals = [arr for arr in origin.arrivals
                    if arr.pick_id == pick.resource_id]
        if pick is None or len(arrivals) == 0:
            continue
        arr = arrivals[0]
        dist_km = degrees2kilometers(arr.distance)
        add_pick = False
        if pick_calculation == 'only_direct':
            if dist_km is not None and dist_km >= crossover_distance_km:
                if pick.phase_hint in ['Pg', 'Sg', 'Pb', 'Sb']:
                    add_pick = True
                elif pick.phase_hint in ['P', 'Pn']:
                    pick.time = origin.time + dist_km / app_vel_Pg
                    pick.phase_hint = 'Pg'
                    add_pick = True
                elif pick.phase_hint in ['S', 'Sn']:
                    pick.time = origin.time + dist_km / app_vel_Sg
                    pick.phase_hint = 'Sg'
                    add_pick = True
            elif pick.phase_hint in ['P', 'S', 'Pg', 'Sg']:
                add_pick = True
        elif pick_calculation == 'only_refracted':
            if dist_km is not None and dist_km >= crossover_distance_km:
                if pick.phase_hint in ['Pg', 'Sg', 'Pb', 'Sb']:
                    add_pick = True
                elif pick.phase_hint in ['P', 'Pg']:
                    pick.time = origin.time + dist_km / app_vel_Pn
                    pick.phase_hint = 'Pg'
                    add_pick = True
                elif pick.phase_hint in ['S', 'Sg']:
                    pick.time = origin.time + dist_km / app_vel_Sn
                    pick.phase_hint = 'Sg'
                    add_pick = True
            elif pick.phase_hint in ['P', 'S', 'Pn', 'Sn']:
                add_pick = True
        if add_pick:
            # if pick was moved, make clear that residual and takeoff
            # aren't known any more
            if arr.phase != pick.phase_hint:
                arr.time_residual = None
                arr.takeoff_angle = None
                arr.phase = pick.phase_hint
            arr.pick_id = pick.resource_id
            new_arrivals.append(arr)
            new_picks.append(pick)

    # SECOND deal with traces where there's no picks:
    st = stream
    for tr in stream:
        # found_matching_resp, tr, sel_inv = try_find_matching_response(tr,inv)
        # if not found_matching_resp:
        #     continue
        # chan = inv.network[0].stations[0].channels[0]
        if (not tr.stats.coordinates.latitude or
                not tr.stats.coordinates.longitude):
            continue
        dist_deg = locations2degrees(
            origin.latitude, origin.longitude,
            tr.stats.coordinates.latitude, tr.stats.coordinates.longitude)
        dist_km = degrees2kilometers(dist_deg)
        if not dist_deg or np.isnan(dist_deg) or np.isnan(dist_km):
            continue
        if pick_calculation == 'only_direct':
            # then get theoretical p -pick
            new_picks, new_arrivals = _make_append_new_pick_and_arrival(
                origin=origin, st=st, tr=tr, phase='Pg', new_picks=new_picks,
                new_arrivals=new_arrivals, app_vel=app_vel_Pg,
                dist_deg=dist_deg, *args, **kwargs)
            new_picks, new_arrivals = _make_append_new_pick_and_arrival(
                origin=origin, st=st, tr=tr, phase='Sg', new_picks=new_picks,
                new_arrivals=new_arrivals, app_vel=app_vel_Sg,
                dist_deg=dist_deg, *args, **kwargs)
        elif (pick_calculation == 'only_refracted'
                and dist_km > crossover_distance_km):
            new_picks, new_arrivals = _make_append_new_pick_and_arrival(
                origin=origin, st=st, tr=tr, phase='Pn', new_picks=new_picks,
                new_arrivals=new_arrivals, app_vel=app_vel_Pn,
                dist_deg=dist_deg, *args, **kwargs)
            new_picks, new_arrivals = _make_append_new_pick_and_arrival(
                origin=origin, st=st, tr=tr, phase='Sn', new_picks=new_picks,
                new_arrivals=new_arrivals, app_vel=app_vel_Sn,
                dist_deg=dist_deg, *args, **kwargs)
    event.picks = new_picks
    origin.arrivals = new_arrivals
    return event, origin


def fix_phasehint_capitalization(event):
    # Check that the caps-sensitive picks are properly named (b, n, g should be
    # lowercase)
    for pick in event.picks:
        if pick.phase_hint in ['PN', 'PG', 'PB', 'SN', 'SG', 'SB']:
            pick.phase_hint = pick.phase_hint[0] + pick.phase_hint[1].lower()
    return event


def prepare_picks(
        event, stream, inv=Inventory(), normalize_NSLC=True,
        std_network_code='NS', std_location_code='00', std_channel_prefix='BH',
        sta_translation_file="station_code_translation.txt",
        vertical_chans=['Z', 'H'], horizontal_chans=['E', 'N', '1', '2', '3'],
        allowed_phase_types='PST',
        allowed_phase_hints=['Pn', 'Pg', 'P', 'Sn', 'Sg', 'S', 'Lg', 'sP',
                             'pP', 'Pb', 'PP', 'Sb', 'SS'],
        forbidden_phase_hints=['pmax', 'PMZ', 'PFAKE'], *args, **kwargs):
    """
    Prepare the picks for being used in EQcorrscan. The following criteria are
    being considered:
     - remove all amplitude-picks
     - remove picks without phase_hint
     - remove picks for stations that have timing problems (indicated by any
       picks with pick.extra.nordic_pick_weight.value == 9)
     - compare picks to the available waveform-traces
     - normalize network/station/channel codes
     - if the channel is not available, the pick will be switched to a suitable
       alternative channel
     - put P-pick on the vertical-channel(s)
     - put S-picks on horizontal channels if available
    TODO: only allow phases Pn, Pg, P, Sn, Sg, S, Lg, sP, pP, Pb, PP, Sb, SS?

    :type template: obspy.core.stream.Stream
    :param template: Template stream to plot
    :type background: obspy.core.stream.stream
    :param background: Stream to plot the template within.
    :type picks: list
    :param picks: List of :class:`obspy.core.event.origin.Pick` picks.
    {plotting_kwargs}
    :type pick_calculation: string
    :param:
        one of 'filter_direct', 'only_direct', 'filter_refracted',
        'only_refracted'

    :returns: :class:`obspy.event`

    """
    # correct PG / SG picks to P / S
    # for pick in event.picks:
    #    if pick.phase_hint == 'PG':
    #        pick.phase_hint = 'P'
    #    elif pick.phase_hint == 'SG':
    #        pick.phase_hint = 'S'

    # catalog.plot(projection='local', resolution='h')
    sta_fortransl_dict, sta_backtrans_dict = load_station_translation_dict(
        file=sta_translation_file)

    stations_w_timing_issue = []
    for pick in event.picks:
        try:
            # if pick.extra.nordic_pick_weight.value == 9:
            if pick.extra['nordic_pick_weight']['value'] == '9':
                stations_w_timing_issue.append(pick.waveform_id.station_code)
        except AttributeError:
            pass
    stations_w_timing_issue = list(dict.fromkeys(stations_w_timing_issue))

    new_event = event.copy()
    new_event.picks = list()
    # Remove all picks for amplitudes etc: keep only P and S
    for j, pick in enumerate(event.picks):
        # Don't allow picks with weight 4:
        try:
            if pick.extra['nordic_pick_weight']['value'] == '4':
                continue
        except (AttributeError, KeyError):
            pass
        # Don't allow picks without phase_hint
        if len(pick.phase_hint) == 0:
            continue
        # Manually sort out picks for phase-hints that are not really arrival
        # picks, but that start with 'P' or 'S' (or 'p' or 's)
        if pick.phase_hint in forbidden_phase_hints:
            continue
        # Skip picks that are not allowed or explicitly forbidden:
        if (allowed_phase_types and
                pick.phase_hint.upper()[0] not in allowed_phase_types):
            continue
        if allowed_phase_hints and pick.phase_hint not in allowed_phase_hints:
            continue

        request_station = pick.waveform_id.station_code
        original_station_code = request_station
        if normalize_NSLC:
            if request_station in sta_fortransl_dict:
                request_station = sta_fortransl_dict.get(request_station)
        if (request_station in stations_w_timing_issue or
                original_station_code in stations_w_timing_issue):
            continue
        # Check which channels are available for pick's station in stream
        avail_comps = [tr.stats.channel[-1] for tr in
                       stream.select(station=request_station)]
        if not avail_comps:
            continue
        # Sort available channels so that Z is the last item
        avail_comps.sort()
        # for no normalization; just do any required channel-switching
        if not normalize_NSLC:
            matching_tr_id, k = get_matching_trace_for_pick(
                pick, stream.select(station=pick.waveform_id.station_code))
            if matching_tr_id is None:
                continue
            std_network_code = matching_tr_id.split('.')[0]
            std_location_code = matching_tr_id.split('.')[2]
            std_channel_prefix = matching_tr_id.split('.')[3][0:2]
        # Check wether pick is P or S-phase, and normalize network/station/
        # location and channel codes
        previous_chan_id = pick.waveform_id.channel_code
        pick.waveform_id.network_code = std_network_code
        pick.waveform_id.location_code = std_location_code
        # Check station code for alternative code, change to the standard
        if normalize_NSLC:
            if pick.waveform_id.station_code in sta_fortransl_dict:
                pick.waveform_id.station_code = sta_fortransl_dict.get(
                    pick.waveform_id.station_code)
        # Check channel codes
        # 1. If pick has no channel information, then put it a preferred
        #    channel (Z>N>E>2>1) for now; otherwise just normalize channel-
        #    prefix.
        if len(pick.waveform_id.channel_code) == 0:
            pick.waveform_id.channel_code = (
                std_channel_prefix + avail_comps[-1])
        elif len(pick.waveform_id.channel_code) <= 2:
            pick.waveform_id.channel_code = (
                std_channel_prefix + pick.waveform_id.channel_code[-1])
        # 2. Check that channel is available - otherwise switch to suitable
        #    other channel.
        if pick.waveform_id.channel_code[-1] not in avail_comps:
            pick.waveform_id.channel_code = (
                std_channel_prefix + avail_comps[-1])
        # 3. If P-pick is not on vertical channel and there exists a 'Z'-
        #    channel, then switch P-pick to Z.
        # TODO: allow different/multiple vertical chans (e.g., Z and H)
        # TODO: add option to avoid switching picks when data quality bad on
        #       one channel - picks may have been set on other channel
        #       deliberately
        if (pick.phase_hint.upper()[0] == 'P' and
                pick.waveform_id.channel_code[-1] not in vertical_chans):
            for vertical_chan in vertical_chans:
                if vertical_chan in avail_comps:
                    pick.waveform_id.channel_code = (
                        std_channel_prefix + vertical_chan)
                    break
            # Event if no available vertical channel was found, change
            # component to Z so that lag-calc does not pick wrong phase_hint
            pick.waveform_id.channel_code = std_channel_prefix + 'Z'
        # 4. If S-pick is on vertical channel and there exist horizontal
        #    channels, then switch S_pick to the first horizontal.
        elif (pick.phase_hint.upper()[0] == 'S' and
                pick.waveform_id.channel_code[-1] in vertical_chans):
            horizontal_traces = stream.select(
                station=pick.waveform_id.station_code,
                channel='{0}[{1}]'.format(
                    std_channel_prefix, ''.join(horizontal_chans)))
            if horizontal_traces:
                pick.waveform_id.channel_code = (
                    horizontal_traces[0].stats.channel)
            # 4b. If S-pick is on vertical and there is no P-pick, then
            # remove S-pick.
            else:
                P_picks = [
                    p for p in event.picks if len(p.phase_hint) > 0
                    if p.phase_hint.upper()[0] == 'P' and
                    p.waveform_id.station_code ==
                    pick.waveform_id.station_code]
                if len(P_picks) == 0:
                    continue
        # If not fixed yet, make sure to always fix codes
        if pick.waveform_id.channel_code == previous_chan_id:
            pick.waveform_id.channel_code = (
                std_channel_prefix + pick.waveform_id.channel_code[-1])
        new_event.picks.append(pick)
        # else:
        #    new_event.picks.append(pick)

    event = new_event
    # Select a subset of picks based on user choice
    origin = event.preferred_origin() or event.origins[0]
    pick_calculation = kwargs.get('pick_calculation')
    if pick_calculation is not None:
        Logger.info('Picks will be calculated for %s', str(pick_calculation))
        event, origin = compute_picks_from_app_velocity(
            event, origin, stream=stream, *args, **kwargs)

    event = fix_phasehint_capitalization(event)

    # Check for duplicate picks. Remove the later one when they are
    # on the same channel
    pickIDs = list()
    new_event = event.copy()
    new_event.picks = list()
    # TODO check whether Pn / Pg are correctly handled here
    for pick in event.picks:
        # uncomment to change back to retaining Pn and Pg
        # pickIDTuple = (pick.waveform_id, pick.phase_hint)
        pickIDTuple = (pick.waveform_id.station_code,
                       pick.phase_hint.upper())
        if pickIDTuple not in pickIDs:
            # pickIDs.append((pick.waveform_id, pick.phase_hint))
            pickIDs.append((pick.waveform_id.station_code,
                            pick.phase_hint.upper()))
            new_event.picks.append(pick)
        else:
            # check which is the earlier pick; remove the old and
            # append the new one . This also takes care of only retaining the
            # earlier pick of Pg, Pn and Sg, Sn.
            for pick_old in new_event.picks:
                if (pick_old.waveform_id.station_code
                        == pick.waveform_id.station_code
                        and pick_old.phase_hint.upper()
                        == pick.phase_hint.upper()):
                    if pick.time < pick_old.time:
                        new_event.picks.remove(pick_old)
                        new_event.picks.append(pick)
    event = new_event
    return event


def try_apply_agc(st, tribe, agc_window_sec=5, pre_processed=False,
                  starttime=None, cores=None, parallel=False, n_threads=1,
                  **kwargs):
    """
    Wrapper to apply agc to a day-long stream for a particular tribe of
    templates.

    :type st: :class:`obspy.core.stream.Stream`
    :param st: Stream object containing the day-long data.
    :type tribe: :class:`eqcorrscan.core.match_filter.Tribe`
    :param tribe: Tribe of templates to be used for detection.
    :type agc_window_sec: float
    :param agc_window_sec: Length of the window to use for AGC in seconds.
    :type pre_processed: bool
    :param pre_processed: If the data are pre-processed, this will be True.
    :type starttime: :class:`obspy.core.utcdatetime.UTCDateTime`
    :param starttime: Start time of the data in the stream.
    :type cores: int
    :param cores: Number of cores to use for multiprocessing.
    :type parallel: bool
    :param parallel: If True, will use multiprocessing to apply the AGC.
    :type n_threads: int
    :param n_threads: Number of threads to use for multiprocessing.
    :type kwargs: dict
    :param kwargs: Additional arguments to pass to

    :rtype: :class:`obspy.core.stream.Stream`
    :return: Stream object with the AGC applied.
    """
    lowcuts = list(set([tp.lowcut for tp in tribe]))
    highcuts = list(set([tp.highcut for tp in tribe]))
    filt_orders = list(set([tp.filt_order for tp in tribe]))
    samp_rates = list(set([tp.samp_rate for tp in tribe]))
    if (len(lowcuts) == 1 and len(highcuts) == 1 and
            len(filt_orders) == 1 and len(samp_rates) == 1):
        Logger.info(
            'All templates have the same trace-processing parameters. '
            'Preprocessing data once for AGC application.'
            'length: %s samples, lowcut: %s Hz, highcut: %s Hz, samp_rate: %s',
            tribe[0].st[0].stats.npts, lowcuts[0], highcuts[0], samp_rates[0])
        st = shortproc(
            st, lowcut=lowcuts[0], highcut=highcuts[0],
            filt_order=filt_orders[0], samp_rate=samp_rates[0],
            starttime=starttime, parallel=parallel, num_cores=cores,
            ignore_length=False, seisan_chan_names=False, fill_gaps=True,
            ignore_bad_data=False, fft_threads=n_threads)

        # TODO: fix error eqcorrscan.core.match_filter.matched_filter:315
        # _group_process() ERROR Data must be process_length or longer, not computing
        # when applying agc
        pre_processed = True
        Logger.info('Applying AGC to preprocessed stream.')
        outtic = default_timer()
        if parallel:
            # This parallel exection cannot use Loky backend because Loky
            # reimports Trace and Stream without the monkey-patch for agc.
            # TODO: figure out how to monkey-patch such that Loky works.
            # with parallel_backend('multiprocessing', n_jobs=cores):
            with parallel_backend('threading', n_jobs=cores):
                traces = Parallel(n_jobs=cores, prefer='threads')(
                    delayed(tr.agc)(agc_window_sec=agc_window_sec, **kwargs)
                    for tr in st)
            st = Stream(traces)
        else:
            st = st.agc(agc_window_sec=agc_window_sec, **kwargs)

        outtoc = default_timer()
        Logger.info('Applying AGC took: {0:.4f}s'.format(outtoc - outtic))
    else:
        msg = ('Templates do not have the same trace-processing ' +
                'parameters, cannot apply AGC.')
        raise NotImplementedError(msg)
    return st, pre_processed


def _merge_streams(streams):
    """
    Helper function to merge many processed streams into one stream, but only
    if traces with same ID are not part of the aggregated stream yet.

    This fixes rare case where data at the same site were recorded on two 
    different instruments / sampling rates and with different station codes,
    so they cannot be merged.
    
    :param streams: List of streams to merge.
    :type streams: list
    :return: Merged stream.
    :rtype: :class:`~obspy.core.stream.Stream`
    """
    stream = Stream()
    for trace_st in streams:
        new_trace_st = Stream()
        st_ids = [etr.id for etr in stream]
        for tr in trace_st:
            if tr.id in st_ids:
                Logger.info(
                    'Cannot add trace %s after initial processing, trace %s '
                    'with same ID is already in stream.',
                    tr, stream.select(id=tr.id)[0])
            else:
                new_trace_st += tr
        stream += new_trace_st
    return stream


def taper_trace_segments(stream, min_length_s=2.0, max_percentage=0.1,
                         max_length=1.0, **kwargs):
    """
    Taper all segments / traces after masking problematic values (e.g., spikes,
    zeros).

    :param stream: input stream with traces
    :type stream: :class:`obspy.core.stream.Stream`
    :param min_length_s:
        minimum segment length in s to keep trace, defaults to 10.0
    :type min_length_s: float, optional
    :param max_percentage:
        maximum length of taper in decimal percent, defaults to 0.1
    :type max_percentage: float, optional
    :param max_length: maximum taper lenght in s, defaults to 1.0
    :type max_length: float, optional

    :return: stream with tapered / remaining traces
    :rtype: :class:`obspy.core.stream.Stream`
    """
    stream = stream.split()
    n_tr_before_sel = len(stream)
    if min_length_s is not None:
        stream = Stream([
            tr for tr in stream
            if tr.stats.npts >= tr.stats.sampling_rate * min_length_s])
        n_tr_after_sel = len(stream)
        if n_tr_after_sel != n_tr_before_sel:
            Logger.info(
                '%s: Removed %s (of %s) traces shorter than %s s',
                '.'.join(stream[0].id.split('.')[0:3]),
                n_tr_before_sel  - n_tr_after_sel,
                n_tr_before_sel, min_length_s)
    stream = stream.taper(max_percentage=max_percentage, max_length=max_length)
    return stream


def get_all_relevant_stations(
        selected_stations, sta_translation_file="station_code_translation.txt",
        **kwargs):
    """
    Return list of relevant stations for a given list of selected stations,
    considering all alternative station names.

    :param selected_stations: list of selected stations
    :type selected_stations: list
    :param sta_translation_file: file with station translation list
    :type sta_translation_file: str

    :return: list of relevant stations
    :rtype: list
    """
    relevant_stations = selected_stations
    sta_fortransl_dict, sta_backtrans_dict = load_station_translation_dict(
        file=sta_translation_file)

    for sta in selected_stations:
        if sta in sta_backtrans_dict:
            relevant_stations.append(sta_backtrans_dict.get(sta))
    relevant_stations = sorted(set(relevant_stations))
    return relevant_stations


def load_station_translation_dict(file="station_code_translation.txt",
                                  **kwargs):
    """
    reads a list of stations with their alternative names from a file
    returns a dictionary of key:alternative name, value: standard name

    :param file: file with station translation list
    :type file: str
    :return: dictionary of alternative station names and standard station names
    :rtype: dict
    """
    station_forw_translation_dict = dict()
    if file == '' or file is None:
        return station_forw_translation_dict, station_forw_translation_dict
    try:
        f = open(file, "r+")
    except Exception as e:
        Logger.warning('Cannot load station translation file %s', file)
        Logger.warning(e)
        return station_forw_translation_dict, station_forw_translation_dict

    for line in f.readlines()[1:]:
        # station_translation_list.append(tuple(line.strip().split()))
        standard_sta_code, alternative_sta_code = line.strip().split()
        station_forw_translation_dict[alternative_sta_code] = standard_sta_code
    station_backw_translation_dict = {y: x for x, y in
                                      station_forw_translation_dict.items()}
    f.close()
    return station_forw_translation_dict, station_backw_translation_dict


def load_forbidden_chan_file(file="forbidden_chans.txt", **kwargs):
    """
    reads a list of channels that are to be removed from all EQcorrscan-data,
    e.g., because of some critical naming conflict (e.g. station NRS as part of
    the DK network and as Norsar array beam code)

    :param file: file with list of forbidden channels
    :type file: str
    :return: list of forbidden channels
    :rtype: list
    """
    forbidden_chans = []
    try:
        f = open(file, "r+")
    except Exception as e:
        Logger.warning('Cannot load forbidden channel file %s', file)
        Logger.warning(e)
        return forbidden_chans
    for line in f.readlines():
        forbidden_chans.append(line.strip())
    f.close()

    return forbidden_chans


def check_template(st, template_length, remove_nan_strict=True,
                   max_perc_zeros=5, allow_channel_duplication=True, **kwargs):
    """
    Function to check that templates do not contain NaNs or zeros, do not
    contain duplicate channels, and that all traces are the same length.
    
    :type st: :class:`obspy.core.stream.Stream`
    :param st: Stream of templates to check.
    :type template_length: float
    :param template_length: Length of templates in seconds.
    :type remove_nan_strict: bool
    :param remove_nan_strict: If True, will remove traces that contain NaNs
    :type max_perc_zeros: float
    :param max_perc_zeros: Maximum percentage of zeros allowed in a trace.
    :type allow_channel_duplication: bool
    :param allow_channel_duplication:
        If True, will allow duplicate channels, otherwise it will remove the
        later duplicated channel.

    :return: Stream of templates with NaNs removed.
    :rtype: :class:`obspy.core.stream.Stream`
    """
    # Now check the templates
    # Check that all traces are the same length:
    t_lengths = [len(tr.data) for tr in st]
    t_length_max = max(t_lengths)
    if any([t_item == t_length_max for t_item in t_lengths]):
        Logger.info('Template stream: %s has traces with unequal lengths.',
                    st[0].stats.starttime)
    # Check each trace
    k = 0
    channel_ids = list()
    st_copy = st.copy()
    for tr in st_copy:
        # Check templates for duplicate channels (happens when there are
        # P- and S-picks on the same channel). Then throw away the
        # S-trace (the later one) for now.
        if tr.id in channel_ids:
            for j in range(0, k):
                test_same_id_trace = st[j]
                if tr.id == test_same_id_trace.id:
                    # remove if the duplicate traces have the same start-time
                    if (tr.stats.starttime == test_same_id_trace.stats.starttime
                            and tr in st):
                        st.remove(tr)
                        continue
                    # if channel-duplication is forbidden, then throw away the
                    # later trace (i.e., S-trace)
                    elif not allow_channel_duplication:
                        st.remove(test_same_id_trace)
                        continue
        else:
            channel_ids.append(tr.id)
            k += 1

    st_copy = st.copy()
    for tr in st_copy:
        # Check that the trace is long enough
        if tr.stats.npts < template_length*tr.stats.sampling_rate and tr in st:
            st.remove(tr)
            Logger.info(
                'Trace %s %s is too short (%s s), removing from template.',
                tr.id, tr.stats.starttime, 
                str(tr.stats.npts / tr.stats.sampling_rate))
        # Check that the trace has no NaNs
        if remove_nan_strict and any(np.isnan(tr.data)) and tr in st:
            st.remove(tr)
            Logger.info('Trace %s contains NaNs, removing from template.',
                        tr.id)
        # Check that not more than 5 % of the trace is zero:
        n_nonzero = np.count_nonzero(tr.copy().detrend().data)
        # if sum(tr.copy().detrend().data==0) > tr.data.size*max_perc_zeros\
        if (n_nonzero < tr.data.size * (1-max_perc_zeros) and tr in st):
            st.remove(tr)
            Logger.info('Trace %s contains more than %s %% zeros, removing '
                        'from template.', tr.id, str(max_perc_zeros*100))
        # Correct messed-up location/channel in case of LRW, MOL
        # if len(tr.stats.channel)<=2 and len(tr.stats.location)==1:
        #    tr.stats.channel = tr.stats.channel + tr.stats.location
        # chn = tr.stats.channel
        # if chn[1]=='L' or chn[1]=='H' or chn[1]==' ':
        #    tr.stats.channel = 'HH' + chn[2]
        #    tr.stats.location = '00'
        # tr.stats.network = 'XX'
    return st


def print_error_plots(st, path='ErrorPlots', time_str=''):
    """
    Prints a daylong-plot of every trace in stream to specified folder.

    :type st: :class:`obspy.core.stream.Stream`
    :param st: Stream of traces to plot.
    :type path: str
    :param path: Path to folder where plots should be saved.
    :type time_str: str
    :param time_str: String to add to plot name.

    :return: None
    :rtype: None
    """
    mid_time = st[0].stats.starttime + (
        st[0].stats.starttime - st[0].stats.endtime) / 2
    current_day_str = str(mid_time)[0:10]
    try:
        if not os.path.isdir(path):
            os.mkdir(path)
        for trace in st:
            png_name = time_str + '_' + trace.stats.station +\
                '_' + trace.stats.channel + '.png'
            out_plot_file = os.path.join(path, png_name)
            trace.plot(type='dayplot', size=(1900, 1080),
                       outfile=out_plot_file, data_unit='nm')
    except Exception as e:
        Logger.error('Got an exception when trying to plot Error figures'
                     'for %s', current_day_str)
        Logger.error(e)


def multiplot_detection(
        party, tribe, st, out_folder='DetectionPlots', **kwargs):
    """
    Create a plot of a detection including the background stream.

    :type party: :class:`eqcorrscan.core.match_filter.party.Party`
    :param party: Party containing the detection to plot.
    :type tribe: :class:`eqcorrscan.core.match_filter.tribe.Tribe`
    :param tribe: Tribe containing the template used for detection.
    :type st: :class:`obspy.core.stream.Stream`
    :param st: Stream containing the background data.
    :type out_folder: str
    :param out_folder: Folder to save the plot to.
    :type kwargs: dict
    :param kwargs:
        Additional keyword arguments to pass to
        :func:`eqcorrscan.utils.plotting.detection_multiplot`.

    :return: None
    """
    if len(party) == 0:
        return
    for family in party:
        if len(family) == 0:
            continue
        for detection in family:
            # times = [d.detect_time for d in family]
            template = [templ for templ in tribe
                        if templ.name == detection.template_name][0]
            templ_starttimes = [tr.stats.starttime for tr in template.st]
            detection_templ_starttimes = [
                tr.stats.starttime for tr in template.st
                if (tr.stats.station, tr.stats.channel) in detection.chans]
            first_trace_offset =\
                min(detection_templ_starttimes) - min(templ_starttimes)
            times = [detection.detect_time + first_trace_offset]

            stt = times[0]
            dst = st.copy().trim(starttime=stt-120, endtime=stt+400)
            dst = dst.split()
            # remove empty trces from stream
            for tr in dst:
                if tr.stats.npts <= 2:
                    dst.remove(tr)
            dst = dst.detrend().taper(0.1).filter(
                'bandpass', freqmin=tribe[0].lowcut, freqmax=tribe[0].highcut,
                zerophase=True, corners=tribe[0].filt_order
                ).resample(tribe[0].samp_rate, no_filter=True, window='hann')
            dst = dst.trim(starttime=stt-30, endtime=stt+240)
            filename = str(detection.detect_time)[0:19] + '_'\
                + family.template.name + '.png'
            filename = os.path.join(out_folder, filename.replace('/', ''))
            try:
                fig = detection_multiplot(
                    stream=dst, template=family.template.st, times=times,
                    save=True, savefile=filename, size=(20, 30), show=False)
            except Exception as e:
                Logger.error(
                    'Could not create multi-plot for detection %s', detection)
                Logger.error(e)


def reevaluate_detections(
        party, short_tribe, stream, threshold_type='MAD', threshold=9,
        re_eval_thresh_factor=0.6, trig_int=40.0, overlap='calculate',
        plot=False, multiplot=False, plotdir='DetectionPlots',
        daylong=False, fill_gaps=False, ignore_bad_data=False,
        ignore_length=True, pre_processed=False,
        parallel_process=False, cores=None, xcorr_func='fftw',
        concurrency=None, arch='precise', group_size=1, full_peaks=False,
        save_progress=False, process_cores=None, spike_test=False, min_chans=4,
        time_difference_threshold=3, detect_value_allowed_reduction=2.5,
        return_party_with_short_templates=False, min_n_station_sites=4,
        use_weights=False, copy_data=True, **kwargs):
    """
    This function takes a set of detections and reruns the match-filter
    detection with a set of templates that are shortened to XX length. Only if
    the detections are also significant (e.g., better detect_value) with the
    shorter templates, then they are retained. Other detections that do not
    pass this test are considered misdetections, which can often happen when
    seismic arrays are involved in detection and there is a seismic event near
    one of the arrays.

    :type party: :class:`eqcorrscan.core.match_filter.party.Party`
    :param party: Party containing the detections to reevaluate.
    :param short_tribe:
        Tribe (shortened compared to detection tribe), containing the templates
        to use for reevaluation of detections.
    :type stream: :class:`obspy.core.stream.Stream`
    :param stream: Stream containing the background data.
    :type threshold_type: str
    :param threshold_type: Threshold type to use for detection
    :type threshold: float
    :param threshold: Threshold to use for detection
    :type re_eval_thresh_factor: float
    :param re_eval_thresh_factor:
        Factor to multiply the original threshold by for match_filter detection
        with short template.
    :type trig_int: float
    :param trig_int: Trigger interval in seconds to use for detection
    :type overlap: float or str
    :param overlap:
    :type plot: bool
    :param plot: Whether to plot the detections
    :type multiplot: bool
    :param multiplot:
        Whether to plot the detections in a nice multi-channel plot.
    :type plotdir: str
    :param plotdir: Directory to save the plots to.
    :type daylong: bool
    :param daylong: Whether the data are daylong or not.
    :type fill_gaps: bool
    :param fill_gaps: Whether to fill gaps in the data or not.
    :type ignore_bad_data: bool
    :param ignore_bad_data: Whether to ignore bad data in EQcorrscan or not.
    :type ignore_length: bool
    :param ignore_length: Whether to ignore trace length or not.
    :type pre_processed: bool
    :param pre_processed: Whether the data are pre-processed or not.
    :type parallel_process: bool
    :param parallel_process:
    :type cores: int
    :param cores:
    :type xcorr_func: str
    :param xcorr_func: Cross-correlation function to use.
    :type concurrency: str
    :param concurrency:
        Concurrency to use for multiprocessing, can be one of 'concurrent',
        'multiprocess', 'multithread'. For more details see
        :func:`eqcorrscan.utils.correlate.get_stream_xcorr`.
    :type arch: str
    :param arch: Architecture of fmf / fmf2 to use, can be 'GPU' or 'CPU'.
    :type group_size: int
    :param group_size: Size of template group to process at once.
    :type full_peaks: bool
    :param full_peaks: Whether to use full peaks or not.
    :type save_progress: bool
    :param save_progress: Whether to save progress or not.
    :type process_cores: int
    :param process_cores: Number of cores to use for processing.
    :type spike_test: bool
    :param spike_test: Whether to use spike test or not.
    :type min_chans: int
    :param min_chans:
        Minimum number of channels to accept a detection as significant.
    :type time_difference_threshold: float
    :param time_difference_threshold:
        Time difference threshold in seconds between detection from long and
        short templates.
    :type detect_value_allowed_reduction: float
    :param detect_value_allowed_reduction:
        Allowed reduction in detect_value between detections from long and
        short templates.
    :type return_party_with_short_templates: bool
    :param return_party_with_short_templates:
        Whether to return the party with short templates or with long templates
        attached to the detections.
    :type min_n_station_sites: int
    :param min_n_station_sites:
        Minimum number of station sites to accept a detection. This is to avoid
        spurious detections that are only due to one array (i.e., one site.)
    :type use_weights: bool
    :param use_weights: Whether to use weights or not.
    :type copy_data: bool
    :param copy_data: Whether to copy the data at the start of EQcorrscan.
    :type kwargs: dict
    :param kwargs: Additional keyword arguments to pass to match_filter.

    :return: Party with detections that have been reevaluated.
    :rtype: :class:`eqcorrscan.core.match_filter.party.Party`
    """
    # Maybe do some checks to see if tribe and short_tribe have somewhat of the
    # same templates?

    # Check there's enough individual station sites for detection - otherwise
    # don't bother with the detection. This should avoid spurious picks that
    # are only due to one array.
    # TODO: this check should be executed before declustering in EQcorrscan
    # TODO: function should always return both the party for the long templates
    #       that have a short-template detection, and the party for the short
    #       templates.
    Logger.info('Start reevaluation of detections.')
    n_families_in = len(party.families)
    n_detections_in = len(party)
    # Get list of unique station names in party for station-site dict lookup
    unique_stations = list(set(
        [chan[0] for fam in party for det in fam for chan in det.chans]))
    station_sites_dict = seismic_array.get_station_sites_dict(
        unique_stations)

    if min_n_station_sites > 1:
        checked_party = Party()
        for family in party:
            checked_family = Family(template=family.template, detections=[],
                                    catalog=None)
            for detection in family:
                # TODO: is there a way to speed up the checks on number of 
                #       station sites?
                unique_det_stations = list(set([chan[0]
                                            for chan in detection.chans]))
                # Get the number of station sites
                n_station_sites = len(list(set(
                    station_sites_dict[uniq_station]
                    for uniq_station in unique_det_stations)))
                if n_station_sites >= min_n_station_sites:
                    checked_family.detections.append(detection)
            if len(family.detections) > 0:
                checked_party.families.append(checked_family)
        Logger.info(
            'Checked party, %s detections fulfill minimum sites criterion.',
            len([det for fam in checked_party for det in fam]))
    else:
        checked_party = party
    long_party = checked_party
    n_detections_ok = len(long_party)

    # Need to scale factor slightly for fftw vs time-domain
    # (based on empirical observation)
    if xcorr_func == 'fftw':
        re_eval_thresh_factor = re_eval_thresh_factor * 1.1
    threshold = threshold * re_eval_thresh_factor

    # Select only the relevant templates
    det_templ_names = list(dict.fromkeys(
        [d.template_name for f in long_party for d in f]))
    short_tribe = Tribe(
        [short_tribe.select(templ_name) for templ_name in det_templ_names])
    # Find the relevant parts of the stream so as not to rerun the whole day:
    det_times = [d.detect_time for f in long_party for d in f]
    Logger.info(
        'Re-evaluating party to sort out misdetections, checking %s'
        + ' detections.', len(det_times))
    if len(det_times) == 0:
        return long_party, long_party
    earliest_det_time = min(det_times)
    latest_det_time = max(det_times)
    # if detections on significan part of the day:
    det_st = stream

    # Rerun detection, but with shorter templates and lower thresholds.
    short_party = short_tribe.detect(
        stream=det_st, threshold=threshold, trig_int=trig_int/10,
        threshold_type=threshold_type, overlap=overlap, plot=plot,
        plotdir=plotdir, daylong=daylong, pre_processed=pre_processed,
        fill_gaps=fill_gaps, ignore_bad_data=ignore_bad_data,
        ignore_length=ignore_length,
        parallel_process=parallel_process, cores=cores,
        concurrency=concurrency, xcorr_func=xcorr_func, arch=arch,
        group_size=group_size, output_event=False,
        full_peaks=full_peaks, save_progress=save_progress,
        process_cores=process_cores, spike_test=spike_test,
        use_weights=use_weights, copy_data=copy_data, **kwargs)
    # TODO: Sanity check: if there are ca. 1000 times or more detections
    #       for each long template, then rerun with higher threshold for speed.

    # Check detections from short templates again the original set of
    # detections. If there is no short-detection for an original detection,
    # or if the detect_value is a lot worse, then remove original detection
    # from party.
    Logger.info(
        'Compare %s detections for short templates against %s existing detect'
        'ions', len([d for fam in short_party for d in fam]), n_detections_ok)
    return_party = Party()
    long_return_party = Party()
    short_return_party = Party()
    short_party_templ_names = [
        f.template.name for f in short_party if f is not None]
    for long_fam in long_party:
        if long_fam.template.name not in short_party_templ_names:
            continue  # do not retain the whole family
        # select matching family
        short_fam = short_party.select(long_fam.template.name)
        if len(short_fam) == 0:
            Logger.debug('Re-evaluation obtained no detections for %s.',
                          long_fam)
            continue
        short_det_times_np = np.array(
            [np.datetime64(d.detect_time.ns, 'ns') for d in short_fam])
        # Adjust by trace-offset when checking with an offset short tribe
        # (e.g., to check whether there is still significant correlation 
        # outside the time window for the first short tribe/templates)
        if hasattr(short_fam.template, 'trace_offset'):
            Logger.debug(
                'Template %s: Adjusting detection times with trace offset',
                short_fam.template.name)
            short_det_times_np += - np.timedelta64(
                int(short_fam.template.trace_offset * 1e9), 'ns')

        # Allow to return either partys with the original templates or the
        # short templates
        # if return_party_with_short_templates:
        #     return_family = Family(short_fam.template)
        # else:
        #     return_family = Family(long_fam.template)
        long_family = Family(long_fam.template)
        short_family = Family(short_fam.template)

        # Check detections for whether they fulfill the reevaluation-criteria.
        for det in long_fam:
            time_diffs = abs(
                short_det_times_np - np.datetime64(det.detect_time.ns, 'ns'))
            time_diff_thresh = np.timedelta64(
                int(time_difference_threshold * 1E9), 'ns')
            # If there is a short-detection close enough in time to the
            # original detection, then check detection values:
            # TODO: here I should pick the detection with the best detection-
            #       value within the time_difference_threshold, so I don't 
            #       pick a spurious value right next to it.
            if not any(time_diffs <= time_diff_thresh):
                Logger.debug('No detections within time-threshold found during'
                             ' re-evaluation of %s at %s', det.template_name,
                             det.detect_time)
            else:
                # Filter short-detections within time error threshold:
                t_diff_ind = np.arange(0, len(time_diffs))  # make index array
                # candidate_time_diffs = time_diffs[
                #     time_diffs <= time_diff_thresh]
                # Filter index array for detections within time threshold
                candidate_indices = t_diff_ind[time_diffs <= time_diff_thresh]
                # Find best detection within time threshold
                detection_values = [
                    detec.detect_val for id, detec in enumerate(short_fam)
                    if id in candidate_indices]
                cand_index = np.argmax(abs(np.array(detection_values)))
                sdi = candidate_indices[cand_index]
                # get the matching short-detection
                # sdi = np.argmin(time_diffs)
                short_det = short_fam[sdi]
                # If detection-value is now better or at least not a lot worse
                # within allowed error, only then keep the original detection.
                det_value_deviation_limit = abs(
                    det.detect_val / detect_value_allowed_reduction)
                # Compare MAD exceedance: If channels in templates have changed
                # between detection and picking, then detect_val may have
                # changed more than allowed, but as long as MAD is just a bit
                # the short-template detection should also be accepted.
                if threshold_type == 'MAD':
                    long_det_mad_exc = abs(det.detect_val / det.threshold)
                    mad_det_value_deviation_limit = (
                        long_det_mad_exc / detect_value_allowed_reduction)
                    short_det_mad_exc = abs(
                        short_det.detect_val / short_det.threshold)
                # Compare detection value for short vs long template
                if (abs(short_det.detect_val) >= det_value_deviation_limit):
                    # if return_party_with_short_templates:
                    #     return_family += short_det
                    # else:
                    #     return_family += det
                    long_family += det
                    short_family += short_det
                elif (threshold_type == 'MAD' and
                      short_det_mad_exc >= mad_det_value_deviation_limit):
                    long_family += det
                    short_family += short_det
                else:
                    Logger.info(
                        'Re-evaluation detections did not meet detection-value'
                        ' criterion for %s at %s (orig det. value: %s, new '
                        'det. value: %s, limit: %s', det.template_name,
                        det.detect_time, det.detect_val,
                        abs(short_det.detect_val), det_value_deviation_limit)
                    if threshold_type == 'MAD':
                        Logger.info(
                            'MAD values change for %s at %s (orig MAD exceedan'
                            'ce %s, new MAD exc. %s, limit: %s',
                            det.template_name, det.detect_time,
                            long_det_mad_exc, short_det_mad_exc,
                            mad_det_value_deviation_limit)
        if len(long_family) >= 0:
            # Quicker with append, to avoid checks between all templates:
            # long_return_party += long_family
            long_return_party.families.append(long_family)
            short_return_party.families.append(short_family)

    if len(long_return_party) == 0:
        n_detections = 0
        n_families = 0
    else:
        n_detections = len(long_return_party)
        n_families = len(long_return_party.families)
    Logger.info(
        'Re-evaluation of %s detections (%s families) finished, remaining are'
        ' %s detections (%s families).', n_detections_in, n_families_in,
        n_detections, n_families)
    if multiplot:
        multiplot_detection(long_return_party, short_tribe, det_st, **kwargs)

    return long_return_party, short_return_party
