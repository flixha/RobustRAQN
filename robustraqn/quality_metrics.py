"""
Utilities module whose functions are designed to handle request for seismic
data informed by data quality metrics prodcued by the IRIS Mustang / ISPAQ
system. Also implements a simple parallel reading for a client (mostly suited
for local data; beware of overloading servers with requests).

:copyright:
    Felix Halpaap 2021
"""

import os
import glob
import re
from signal import signal, SIGSEGV
import numpy as np
import datetime

from multiprocessing import Pool, cpu_count, get_context
from multiprocessing.pool import ThreadPool
from joblib import Parallel, delayed


import pandas as pd
import fnmatch
import itertools
from timeit import default_timer

from obspy.core.stream import Stream
from obspy import UTCDateTime
from obspy.io.mseed import InternalMSEEDError

from eqcorrscan.utils.correlate import pool_boy

import logging
Logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(name)40s:%(lineno)s\t%(funcName)20s()\t" +
    "%(levelname)s\t%(message)s")


def _get_waveforms_bulk(client, bulk):
    """
    """
    st = Stream()
    for arg in bulk:
        try:
            st += client.get_waveforms(*arg)
        except Exception as e:
            document_client_read_error(e)
            continue
    return st


def get_waveforms_bulk(client, bulk, parallel=False, cores=16):
    """
    Perform a bulk-waveform request in parallel. Return one stream.
    There seems to be a negative effect on speed if there's too many read-
    threads - for now set default to 16.

    :type client: obspy.obspy.Client
    :param name: Client to request the data from.
    :type bulk: list of tuples
    :param bulk: Information about the requested data.
    :type parallel: book
    :param parallel: Whether to run reading of waveform files in parallel.
    :type cores: int
    :param bulk: Number of parallel processors to use.

    :rtype: obspy.core.Stream
    """
    # signal(SIGSEGV, sigsegv_handler)
    Logger.info('Start bulk-read')
    outtic = default_timer()
    st = Stream()
    if parallel:
        if cores is None:
            cores = min(len(bulk), cpu_count())

        # Switch to Process Pool for now. Threadpool is quicker, but it is not
        # stable against errors that need to be caught in Mseed-library. Then
        # as segmentation fault can crash the whole program. ProcessPool needs
        # to be spawned, otherwise this can cause a deadlock.

        # Logger.info('Start bulk-read paralle pool')
        # Process / spawn handles segmentation fault better?
        #with pool_boy(Pool=ThreadPool, traces=len(bulk), cores=cores) as pool:
        # with pool_boy(Pool=Pool, traces=len(bulk), cores=cores) as pool:

        # with pool_boy(Pool=get_context("spawn").Pool, traces=len(bulk),
        #               cores=cores) as pool:
        #     results = [pool.apply_async(
        #         _get_waveforms_bulk, args=(client, [arg])) for arg in bulk]

        # Use joblib with loky-pools; this is the most stable
        results = Parallel(n_jobs=cores)(
            delayed(_get_waveforms_bulk)(client, [arg]) for arg in bulk)
        # Concatenate all NSLC-streams into one stream
        st = Stream()
        for res in results:
            # st += res.get()
            st += res
        # pool.close()
        # pool.join()
        # pool.terminate()
    else:
        st = _get_waveforms_bulk(client, bulk)

    outtoc = default_timer()
    Logger.info('Bulk-reading of waveforms took: {0:.4f}s'.format(
        outtoc - outtic))
    return st


# def get_waveforms_bulk_old(client, bulk, parallel=False, cores=None):
#     """
#     Perform a bulk-waveform request in parallel. Return one stream.

#     :type client: obspy.obspy.Client
#     :param name: Client to request the data from.
#     :type bulk: list of tuples
#     :param bulk: Information about the requested data.
#     :type parallel: book
#     :param parallel: Whether to run reading of waveform files in parallel.
#     :type cores: int
#     :param bulk: Number of parallel processors to use.

#     :rtype: obspy.core.Stream
#     """
#     # signal(SIGSEGV, sigsegv_handler)
#     Logger.info('Start bulk-read')
#     outtic = default_timer()
#     st = Stream()
#     if parallel:
#         if cores is None:
#             cores = min(len(bulk), cpu_count())
#         # There seems to be a negative effect on speed if there's too many
#         # read-threads - For now set limit to 16
#         cores = min(cores, 16)

#         # Logger.info('Start bulk-read paralle pool')
#         # Process / spawn handles segmentation fault better?
#         #with pool_boy(Pool=get_context("spawn").Pool, traces=len(bulk),
#         #              cores=cores) as pool:
#         #with pool_boy(Pool=ThreadPool, traces=len(bulk), cores=cores) as pool:
#         with pool_boy(Pool=Pool, traces=len(bulk), cores=cores) as pool:
#             results = [pool.apply_async(
#                 client.get_waveforms,
#                 args=arg,
#                 error_callback=document_client_read_error)
#                        for arg in bulk]

#         # ORRRR
#         # with pool_boy(Pool=Pool, traces=len(bulk), cores=cores) as pool:
#         #     results = [pool.apply_async(
#         #         get_waveforms_bulk,
#         #         args=(client, [arg]),
#         #         kwds=dict(parallel=False, cores=None))
#         #                for arg in bulk]
#         # Need to handle possible read-errors in each request when getting each
#         # request-result.
#         # Logger.info('Get bulk-read pool results')
#         st_list = list()
#         for res in results:
#             try:
#                 st_list.append(res.get())
#             # InternalMSEEDError
#             except Exception as e:
#                 Logger.error(e)
#                 pass
#         # Concatenate all NSLC-streams into one stream
#         for st_part in st_list:
#             st += st_part
#     else:
#         for arg in bulk:
#             try:
#                 st += client.get_waveforms(*arg)
#             except Exception as e:
#                 document_client_read_error(e)
#                 continue

#     outtoc = default_timer()
#     Logger.info('Bulk-reading of waveforms took: {0:.4f}s'.format(
#         outtoc - outtic))
#     return st


def document_client_read_error(s):
    """
    Function to be called when an exception occurs within one worker in
        `get_waveforms_bulk_parallel`.

    :type s: str
    :param s: Error message
    """
    Logger.error("Error reading waveform file - skipping this file.")
    Logger.error(s, exc_info=True)


# def sigsegv_handler(sigNum, frame):
#     """
#     Segmentation fault handler (can occur in MSEED-read)
#     """
#     print("handle signal", sigNum)


def get_parallel_waveform_client(waveform_client):
    """
    Bind a `get_waveforms_bulk` method to waveform_client if it doesn't already
    have one.
    """
    def _get_waveforms_bulk_parallel_naive(self, bulk, parallel=True,
                                           cores=None):
        """
        parallel implementation of get_waveforms_bulk.
        """
        # signal.signal(signal.SIGSEGV, sigsegv_handler)

        st = Stream()
        if parallel:
            if cores is None:
                cores = min(len(bulk), cpu_count())
            # There seems to be a negative effect on speed if there's too many
            # read-threads - For now set limit to 16
            cores = min(cores, 16)

            # Logger.info('Start bulk-read paralle pool')
            with pool_boy(
                    Pool=ThreadPool, traces=len(bulk), cores=cores) as pool:
                results = [pool.apply_async(
                    self.get_waveforms,
                    args=arg,
                    error_callback=document_client_read_error)
                        for arg in bulk]
            # Need to handle possible read-errors in each request when getting
            # each request-result.
            for res in results:
                try:
                    st += res.get()
                # InternalMSEEDError
                except Exception as e:
                    Logger.error(e)
                    pass
        else:
            for arg in bulk:
                try:
                    st += self.get_waveforms(*arg)
                except Exception as e:
                    document_client_read_error(e)
                    continue
        return st

    # add waveform_bulk method dynamically if it doesn't exist already
    if not hasattr(waveform_client, "get_waveforms_bulk_parallel"):
        bound_method = _get_waveforms_bulk_parallel_naive.__get__(
            waveform_client)
        setattr(waveform_client, "get_waveforms_bulk_parallel", bound_method)

    return waveform_client


# def get_waveform_client(waveform_client):
#     """
#     Bind a `get_waveforms_bulk` method to waveform_client if it doesn't
#     already have one.
#     Copyright Calum Chamberlain, 2020
#     """
#     def _get_waveforms_bulk_naive(self, bulk_arg):
#         """
#         a naive implementation of get_waveforms_bulk that uses iteration.
#         """
#         st = Stream()
#         for arg in bulk_arg:
#             st += self.get_waveforms(*arg)
#         return st

#     # add waveform_bulk method dynamically if it doesn't exist already
#     if not hasattr(waveform_client, "get_waveforms_bulk"):
#         bound_method = _get_waveforms_bulk_naive.__get__(waveform_client)
#         setattr(waveform_client, "get_waveforms_bulk", bound_method)

#     return waveform_client

def check_request_for_wildcards(stats, pattern_list, pattern_position):
    """
    Checks and translates wildcard patterns in station/location/channel-names
    to regexes that work with pandas
    """
    if stats.empty:
        return []
    if pattern_position[0] not in [0, 1, 2]:
        raise(IndexError, 'first pattern_position can only be 0, 1, or 2')
    if len(pattern_position) > 1:
        if pattern_position[1] not in [0, 1, 2]:
            raise(IndexError, 'second pattern_position can only be 0, 1, or 2')
    # Check for wildcards in station names
    full_pattern_list = list()
    for pattern in pattern_list:
        if '?' in pattern or '*' in pattern:
            # If there are wildcards like ? or * in pattern, replace with regex
            pattern = pattern.replace(
                '.', '\.').replace('*', '.*').replace('?', '.')
            # Split short_target and match only by one multi-index column:
            # pattern_position describes which part of the SLC-target should be
            # matched: 0: station, 1: location, 2: channel
            sub_targets = stats.index.str.split(
                '.', expand=True).get_level_values(pattern_position[0])
            # pattern_position can be [2, 0] to match only the bandcode of SCL
            if len(pattern_position) == 2:
                sub_targets = stats.index.str[pattern_position[1]]
            matching_patterns = sub_targets[
                sub_targets.str.contains(pattern, flags=re.I)]
            # targets = stats[stats.index.str.startswith(station + '\.')]
            # pattern_position describes which part of the SLC-target should be
            # matched: 0: station, 1: location, 2: channel
            matching_patterns = list(dict.fromkeys(matching_patterns))
            if len(matching_patterns) > 0:
                for mpattern in matching_patterns:
                    full_pattern_list.append(mpattern)
        else:
            full_pattern_list.append(pattern)
    if full_pattern_list:
        # Get unqiue patterns in original order
        pattern_list = list(dict.fromkeys(full_pattern_list))
    return pattern_list


def create_bulk_request(
        inventory, starttime, endtime, stats=pd.DataFrame(), stations=['*'],
        location_priority=['*'], band_priority=['B', 'H', 'S', 'E', 'N'],
        instrument_priority=['H'], components=['Z', 'N', 'E', '1', '2'],
        minimum_sample_rate=20, parallel=False, cores=1, **kwargs):
    """
    stats = read_ispaq_stats(folder, starttime,endtime, networks=['??'],
                             stations=['*'],)
    Assuming that each station name is unique (even if part of diff. networks)
    1. Loop through stations
    2. check availability
    3. append to bulk those location+channels that best fulfill priorities at
       that station

    :type starttime: obspy.UTCDateTime
    :param starttime: Starttime of the waveforms to be requested
    :type endtime: obspy.UTCDateTime
    :param endtime: Endtime of the waveforms to be requested
    :type stats: pd.DataFrame
    :param stats:
        ispaq-based Mustang-style pandas-dataframe with data quality metrics.
    :type parallel: bool
    :param parallel: Indicate whether to create the request in parallel.
    :type cores: int
    :param cores: Number of parallel processes to use.
    :type stations: list
    :param stations: List of stations for which data shall be requested
    :type location_priority: list
    :param location_priority:
        List of prioritized location codes for which data shall be requested.
        Data will be requested for the first location code for which the
        quality metrics fulfill the criteria.
    :type band_priority: list
    :param band_priority:
        List of letters indicating the priorities for the channel's band code
        that indicates the instrument's bandwidth / sampling rate (e.g.,
        B, H, E, S, etc.).
    :type instrument_priority: list
    :param instrument_priority:
        List of letters indicating the priorities for the channel's instrument
        code that indicates the type of instrument at the station (e.g.,
        H, N etc.).
    :type components: list
    :param components:
        List of letters for the component code to be requested. If more than
        3 letters are listed, then by default the function tries to find
        3 channels that belong together to make up a 3-component system (e.g.,
        will try to get either ZNE or Z12 if the list contains
        ['Z','N','E','1','2'] ).
    :param kwargs:
        Additional arguments that get passed on to
        `quality_metrics.get_station_bulk_request`; mainly the thresholds for
        different quality metrics.

    :rtype: list of tuples
    """
    Logger.info('Creating bulk request for %s - %s.', str(starttime)[0:19],
                str(endtime)[0:19])
    if len(stations) == 0:
        Logger.warning('Creating bulk request for Zero stations, are you sure '
                       'about that?')
    # Do no yet allow requests that are much longer than one day (some overlap
    # at day-boundary can be requested).
    if endtime - starttime > 86400 * 1.5:
        raise ValueError("start- and endtime for one bulk-request cannot be"
                         + "more than 36 hours apart.")
    mid_t = starttime + (endtime - starttime)/2
    # The reqtimes are those that are listed in the ispaq-stats per day (starts
    # on 00:00:00 and ends on 23:59:59).
    reqtime1 = UTCDateTime(mid_t.year, mid_t.month, mid_t.day, 0, 0, 0)
    reqtime2 = UTCDateTime(mid_t.year, mid_t.month, mid_t.day, 23, 59, 59)
    bulk = list()
    rejected_bulk = list()

    # If there's no stats at all, then make a bulk-request for everything:
    if stats is None or stats.empty:
        bulk = [('*', sta, loc, band + inst + comp, reqtime1, reqtime2)
                for sta in stations
                for loc in location_priority
                for band in band_priority
                for inst in instrument_priority
                for comp in components]
        return bulk, rejected_bulk, stats

    # Smartly set the index column of dataframes to speed up list selection:
    # Check if the index-column is called "startday", otherwise make it such.
    if stats.index.name != 'startday':
        try:
            stats['startday'] = stats['start'].str[0:10]
        except KeyError:
            Logger.error('No data quality metrics available for %s - %s.',
                        str(starttime)[0:19], str(endtime)[0:19])
            return None, None, None
        stats = stats.set_index(['startday'])
    if 'short_target' not in stats.columns:
        stats['short_target'] = stats['target'].str[3:-2]
    # Now that "startday" is set as index:
    try:
        day_stats = stats.loc[str(reqtime1)[0:10]]
    except KeyError:
        Logger.warning('No data quality metrics for %s',
                        str(reqtime1)[0:10])
        return None, None, None
    # Now set "short_target" as index-column to speed up the selection in
    # the loop across stations below.
    day_stats = day_stats.set_index(['short_target'])
    # Check for wildcards in station names
    stations = check_request_for_wildcards(
        day_stats, stations, pattern_position=[0])

    station_requested = False
    if parallel:
        if cores is None:
            cores = min(len(stations), cpu_count())
        out_lists = Parallel(n_jobs=cores)(
            delayed(get_station_bulk_request)(
                inventory.select(station=station, time=reqtime1),
                station, location_priority, band_priority, instrument_priority,
                components, # send only stats relevant to station to worker
                day_stats[day_stats.index.str.startswith(station + '.')],
                minimum_sample_rate, reqtime1, starttime, endtime, **kwargs)
            for station in stations)
        bulk_lists = [out_l[0] for out_l in out_lists]
        rejected_bulk_lists = [out_l[1] for out_l in out_lists]
    else:
        bulk_lists = list()
        rejected_bulk_lists = list()
        for station in stations:
            bulk_new, rejected_bulk = get_station_bulk_request(
                inventory.select(station=station, time=reqtime1),
                station, location_priority, band_priority, instrument_priority,
                components, day_stats, minimum_sample_rate, reqtime1,
                starttime, endtime, **kwargs)
            bulk_lists.append(bulk_new)
            rejected_bulk_lists.append(rejected_bulk)
    Logger.debug('Bulk request is: %s', bulk_lists)

    # Merge the lists containted in bulk_lists
    bulk = list(itertools.chain.from_iterable(b for b in bulk_lists))
    rejected_bulk = list(itertools.chain.from_iterable(
        b for b in rejected_bulk_lists))

    return bulk, rejected_bulk, day_stats


def get_station_bulk_request(
        inventory, station, location_priority, band_priority,
        instrument_priority, components, day_stats, minimum_sample_rate,
        request_time, starttime, endtime, **kwargs):
    """
    Inner function to create a bulk-request for one specific day.

    :type day_stats: pd.DataFrame
    :param day_stats:
        ispaq-based Mustang-style pandas-dataframe with data quality metrics
        for the relevant day.
    :type parallel: bool
    :param parallel: Indicate whether to create the request in parallel.
    :type cores: int
    :param cores: Number of parallel processes to use.
    :type station: str
    :param station: Station for which data shall be requested
    :type location_priority: list
    :param location_priority:
        List of prioritized location codes for which data shall be requested.
        Data will be requested for the first location code for which the
        quality metrics fulfill the criteria.
    :type band_priority: list
    :param band_priority:
        List of letters indicating the priorities for the channel's band code
        that indicates the instrument's bandwidth / sampling rate (e.g.,
        B, H, E, S, etc.).
    :type instrument_priority: list
    :param instrument_priority:
        List of letters indicating the priorities for the channel's instrument
        code that indicates the type of instrument at the station (e.g., H, N
        etc.).
    :type components: list
    :param components:
        List of letters for the component code to be requested. If more than
        3 letters are listed, then by default the function tries to find
        3 channels that belong together to make up a 3-component system (e.g.,
        will try to get either ZNE or Z12 if the list contains
        ['Z','N','E','1','2'] ).
    :param kwargs:
        Additional arguments that get passed on to
        `quality_metrics.get_station_bulk_request`; mainly the thresholds for
        different quality metrics.

    :rtype: list of tuples
    """
    # Check for wildcards in location names
    location_priority = check_request_for_wildcards(
        day_stats, location_priority, pattern_position=[1])
    # Check for wildcards in band codes
    band_priority = check_request_for_wildcards(
        day_stats, band_priority, pattern_position=[2, 0])
    # Check for wildcards in band codes
    instrument_priority = check_request_for_wildcards(
        day_stats, instrument_priority, pattern_position=[2, 1])
    # Check for wildcards in component code
    components = check_request_for_wildcards(
        day_stats, components, pattern_position=[2, 2])
    bulk = list()
    bulk_rejected = list()
    # Now magically find the prioritized channels that best fulfill all
    # criteria like availability, num_spikes, etc. etc...
    for location in location_priority:
        if all_channels_requested(bulk, station, components):
            break
        for band in band_priority:
            if all_channels_requested(bulk, station, components):
                break
            for instrument in instrument_priority:
                if all_channels_requested(bulk, station, components):
                    break
                for component in components:
                    channel = band + instrument + component
                    if all_channels_requested(bulk, station, components):
                        break
                    if same_comp_requested(bulk, station, component):
                        continue
                    if not _sample_rate_ok(
                            inventory, request_time, minimum_sample_rate,
                            station, location, channel):
                        Logger.info(
                            'Sample rate for %s.%s.%s%s%s too low on %s, '
                            'looking for alternative location / channel.',
                            station, location, band, instrument, component,
                            request_time)
                        continue
                    # Add target if it passes some metrics-checks
                    add_target_request = True
                    short_scnl = station + "." + location + '.' + channel
                    # With "short_target" as index-column:
                    if day_stats.index.name == "short_target":
                        try:
                            chn_stats = day_stats.loc[short_scnl]
                        except KeyError:
                            continue
                    elif day_stats.index.name == "startday":
                        try:
                            chn_stats = day_stats[
                                day_stats['target'].str[3:-2] == short_scnl]
                        except KeyError:
                            continue
                    else:
                        msg = ('Data quality metrics dataframe is missing '
                            + 'expected column headers short_target or ' +
                            ' startday')
                        raise(KeyError, msg)

                    try:
                        availability = chn_stats[
                            chn_stats['metricName'] == "percent_availability"]
                    except Exception as e:
                        Logger.error(
                            'Cannot find availability for %s.%s.%s%s%s on %s',
                            station, location, band, instrument, component,
                            request_time)
                        # Logger.error(e, exc_info=True)
                        continue

                    target_rejected = False
                    # Availability-metric has to exist
                    if len(availability) == 0:
                        add_target_request = False
                    else:
                        # Now check all metrics against their thresholds:
                        add_target_request, target = check_metrics(
                            day_stats, request_time, availability, **kwargs)
                        if not add_target_request:
                            target_rejected = True
                    # Add specific channel-request to bulk-request
                    if add_target_request:
                        # Split "target" into net, station, loc, channel
                        nscl = target.split('.')
                        bulk.append(tuple(nscl[0:4]) + (starttime, endtime))
                        # bulk.append((nscl[0], nscl[1], nscl[2], nscl[3],
                        #              starttime, endtime))
                    if target_rejected:
                        nscl = target.split('.')
                        bulk_rejected.append(tuple(nscl[0:4]) + (
                            starttime, endtime))
                        # bulk_rejected.append((nscl[0], nscl[1], nscl[2],
                        #                       nscl[3], starttime, endtime))
    return bulk, bulk_rejected


def check_metrics(day_stats, request_time, availability, min_availability=0.8,
                  max_spikes=1000, max_glitches=1000, max_num_gaps=500,
                  max_num_overlaps=1000, max_max_overlap=86400,
                  min_sample_unique=150, max_abs_sample_mean=1e7,
                  min_sample_rms=2, max_sample_rms=1e8,
                  max_sample_median=1e6, min_abs_sample_average=(1, 1e-9),
                  require_clock_lock=False, max_suspect_time_tag=86400,
                  max_dead_channel_lin=3, require_alive_channel_gsn=False,
                  max_pct_below_nlnm=50, max_pct_above_nhnm=50,
                  max_cross_talk=0.999, **kwargs):
    """
    Function to check all data quality metrics for one specific day against the
    set thresholds, and return True or False depending on whether the data ful-
    fill all thresholds.

    :type day_stats: pd.DataFrame
    :param day_stats:
        ispaq-based Mustang-style pandas-dataframe with data quality metrics
        for the relevant day.
    :type request_time: obspy.UTCDateTime
    :param request_time:
        Time for which the nearest data quality metrics shall be requested.
    :type availability: pandas.DataFrame
    :param availability:
        ispaq Mustang-style Dataframe listing the availability information.
    :type min_availability: float
    :param min_availability: Lower threshold for availability
    :type max_spikes: float
    :param max_spikes: Upper threshold for number of spikes.
    :type max_glitches: float
    :param max_glitches: Upper threshold for number of glitches.
    :type max_num_gaps: float
    :param max_num_gaps: Upper threshold for number of gaps.
    :type max_num_overlaps: float
    :param max_num_overlaps: Upper threshold for number of overlaps.
    :type max_max_overlap: float
    :param max_max_overlap:
        Upper threshold for the largest overlap (in seconds).
    :type min_sample_unique: float
    :param min_sample_unique:
        Lower threshold for the number of unique sample values.
    :type max_abs_sample_mean: float
    :param max_abs_sample_mean:
        Upper threshold for the absolute value of the sample mean.
    :type min_sample_rms: float
    :param min_sample_rms: Lower threshold for the value of the sample RMS.
    :type max_sample_rms: float
    :param max_sample_rms: Upper threshold for the value of the sample RMS.
    :type max_sample_median: float
    :param max_sample_median:
        Upper threshold for the value of the sample median.
    :type min_abs_sample_average: float
    :param min_abs_sample_average:
        Lower threshold for the absolute value of the sample mean.
    :type require_clock_lock: bool
    :param require_clock_lock: Whether a clock locking shall be required.
    :type max_suspect_time_tag: float
    :param max_suspect_time_tag:
        Upper threshold for the number of "suspect-time"-tag set in waveform
        file.
    :type max_dead_channel_lin: float
    :param max_dead_channel_lin: Upper threshold for dead_channel_lin metric.
    :type require_alive_channel_gsn: bool
    :param require_alive_channel_gsn:
        Whether to require the alive_channel_gsn-metric.
    :type max_pct_below_nlnm: float
    :param max_pct_below_nlnm:
        Upper threshold for the percentage of the PSD below the New Low Noise
        Model.
    :type max_pct_above_nhnm: float
    :param max_pct_above_nhnm:
        Upper threshold for the percentage of the PSD above the New High Noise
        Model.
    :type max_cross_talk: float
    :param max_cross_talk Upper threshold for the amount of cross-talk.

    :rtype: tuple of (bool, tuple)
    :rparam: Whether metrics passed, and a tuple containing the target to
             request.
    """

    add_target_request = True
    req_time_str = str(request_time)[0:10]

    # Check whether there is more than 1 row in metrics -
    # that would imply multiple networks offer the data.
    if len(availability) >= 2:
        max_index = availability['value'].argmax()
        availability = availability.iloc[max_index]
    else:
        availability = availability.iloc[0]

    target = availability['target']

    # Now go through all implemented data quality metrics

    # Check for minimum-availability - watch out for availability in PERCENT
    if (availability['value']/100 < min_availability):
        Logger.info('%s, %s: less than %s %% data available, not using.',
                    target, req_time_str, min_availability*100)
        return False, target
    # General formulation #############
    # check_stat(day_stats, metricName='num_spikes',
    #            min_value, max_value)

    # Check whether channel below maximum number of spikes
    # max_spikes < 100 ? 1000? 10000?
    num_spikes = day_stats[(day_stats['target'] == target)
                           & (day_stats['metricName'] == 'num_spikes')]
    if len(num_spikes) > 0:
        if num_spikes.iloc[0]['value'] > max_spikes:
            Logger.info('%s, %s: more than %s spikes, not using', target,
                        req_time_str, max_spikes)
            return False, target

    # Check whether there is no GPS timing
    # clock_locked == 0
    clock_locked = day_stats[(day_stats['target'] == target)
                             & (day_stats['metricName'] == 'clock_locked')]
    if len(clock_locked) > 0 and require_clock_lock:
        if clock_locked.iloc[0]['value'] == 0:
            Logger.info('%s, %s: clock not locked, not using', target,
                        req_time_str)
            return False, target

    # Check whether masses are pegged
    # sample_mean > 1e7 !!! NEEDS check for instruments
    sample_mean = day_stats[(day_stats['target'] == target)
                            & (day_stats['metricName'] == 'sample_mean')]
    if len(sample_mean) > 0:
        if abs(sample_mean.iloc[0]['value']) > max_abs_sample_mean:
            Logger.info('%s, %s: sample_mean too large, not using', target,
                        req_time_str)
            return False, target

    # Check whether channel is dead
    dead_channel_lin = day_stats[
        (day_stats['target'] == target)
        & (day_stats['metricName'] == 'dead_channel_lin')]
    if len(dead_channel_lin) > 0:
        if dead_channel_lin.iloc[0]['value'] < max_dead_channel_lin:
            Logger.info('%s, %s: channel is dead, not using', target,
                        req_time_str)
            return False, target

    # dead_channel_gsn == 1 and
    dead_channel_gsn = day_stats[
        (day_stats['target'] == target)
        & (day_stats['metricName'] == 'dead_channel_gsn')]
    if len(dead_channel_gsn) > 0 and require_alive_channel_gsn:
        if dead_channel_gsn.iloc[0]['value'] == 1:
            Logger.info('%s, %s: channel is dead, not using', target,
                        req_time_str)
            return False, target

    # pct_below_nlnm > 20
    pct_below_nlnm = day_stats[(day_stats['target'] == target)
                               & (day_stats['metricName'] == 'pct_below_nlnm')]
    if len(pct_below_nlnm) > 0:
        if pct_below_nlnm.iloc[0]['value'] > max_pct_below_nlnm:
            Logger.info('%s, %s: More than %s %% of noise spectrum below NLNM,'
                        + ' not using.', target, req_time_str,
                        str(max_pct_below_nlnm))
            return False, target

    # Check whether high noise on channel
    # dead_channel_exp / _lin / _gsn < 0.3 and
    # pct_above_nhnm > 20
    pct_above_nhnm = day_stats[(day_stats['target'] == target)
                               & (day_stats['metricName'] == 'pct_above_nhnm')]
    if len(pct_above_nhnm) > 0:
        if pct_above_nhnm.iloc[0]['value'] > max_pct_above_nhnm:
            Logger.info('%s, %s: More than %s %% of noise spectrum above NHNM,'
                        + ' not using.', target, req_time_str,
                        str(max_pct_above_nhnm))
            return False, target

    # Check number of unique samples
    sample_unique = day_stats[(day_stats['target'] == target)
                              & (day_stats['metricName'] == 'sample_unique')]
    if len(sample_unique) > 0:
        if sample_unique.iloc[0]['value'] < min_sample_unique:
            Logger.info('%s, %s: Less than %s unique samples, not using.',
                        target, req_time_str, str(min_sample_unique))
            return False, target

    # Check whether high-amplitude on channel
    # sample_rms > 50000 !!! NEEDS check for instruments
    sample_rms = day_stats[(day_stats['target'] == target)
                           & (day_stats['metricName'] == 'sample_rms')]
    if len(sample_rms) > 0:
        if sample_rms.iloc[0]['value'] < min_sample_rms:
            Logger.info('%s, %s: sample_rms too small, not using. ', target,
                        req_time_str)
            return False, target
        elif sample_rms.iloc[0]['value'] > max_sample_rms:
            Logger.info('%s, %s: sample_rms too large, not using. ', target,
                        req_time_str)
            return False, target

    # Maximum number of overlaps
    num_overlaps = day_stats[(day_stats['target'] == target)
                             & (day_stats['metricName'] == 'num_overlaps')]
    if len(num_overlaps) > 0:
        if num_overlaps.iloc[0]['value'] > max_num_overlaps:
            Logger.info('%s, %s: too many overlaps, not using. ', target,
                        req_time_str)
            return False, target

    # Maximum length of overlap
    max_overlap = day_stats[(day_stats['target'] == target)
                            & (day_stats['metricName'] == 'max_overlap')]
    if len(max_overlap) > 0:
        if max_overlap.iloc[0]['value'] > max_max_overlap:
            Logger.info('%s, %s: max_overlap too large, not using.',
                        target, req_time_str)
            return False, target

    # Sample median
    sample_median = day_stats[(day_stats['target'] == target)
                              & (day_stats['metricName'] == 'sample_median')]
    if len(sample_median) > 0:
        if sample_median.iloc[0]['value'] > max_sample_median:
            Logger.info('%s, %s: sample median greater than %s, not using.',
                        target, req_time_str, max_sample_median)
            return False, target

    # Maximum number of suspect_time_tag-flags
    suspect_time_tag = day_stats[
        (day_stats['target'] == target)
        & (day_stats['metricName'] == 'suspect_time_tag')]
    if len(suspect_time_tag) > 0:
        if suspect_time_tag.iloc[0]['value'] > max_suspect_time_tag:
            Logger.info('%s, %s: Too many suspect-time tags, not using.',
                        target, req_time_str)
            return False, target

    # Maximum cross talk
    cross_talk = day_stats[(day_stats['target'] == target)
                           & (day_stats['metricName'] == 'cross_talk')]
    if len(cross_talk) > 0:
        if cross_talk.iloc[0]['value'] > max_cross_talk:
            Logger.info('%s, %s: Cross talk too large, not using.',
                        target, req_time_str)
            return False, target

    # Check whether channel shows too little amplitude
    # (i.e., there are too many Zeros)
    if len(sample_mean) > 0 and len(sample_median) > 0:
        s_abs_mean = abs(sample_mean.iloc[0]['value'])
        s_abs_median = abs(sample_median.iloc[0]['value'])
        _min_abs_sample_mean = min_abs_sample_average[0]
        _min_abs_sample_median = min_abs_sample_average[1]
        if (s_abs_mean > _min_abs_sample_mean
                and s_abs_median < _min_abs_sample_median):
            Logger.info(
                '%s, %s: Sample mean (%s) above limit (%s), but sample median '
                + '(%s) below limit (%s), indicating too many zeros in trace, '
                + 'not using', target, req_time_str, str(s_abs_mean),
                str(_min_abs_sample_mean), str(s_abs_median),
                str(_min_abs_sample_median))
            return False, target

    return add_target_request, target


def _sample_rate_ok(inventory, time, minimum_sample_rate, station_code,
                    location_code, channel_code):
    """
    Check sampling rate of channel response in inventory; return False
    if it does not fulfill minimum criterion.
    """
    sample_rate_ok = True
    sel_inv = inventory.select(
        station=station_code, location=location_code, channel=channel_code,
        time=time)
    if len(sel_inv.networks) > 0:
        channels = [channel for net in sel_inv.networks
                     for station in net for channel in station]
        if len(channels) > 1:
            Logger.warning(
                'Found more than one matching response for %s on %s, comparing'
                ' sampling rate only against first match.',
                station_code + '.' + location_code + '.' + channel_code, time)
        s_rate = channels[0].sample_rate
        if s_rate < minimum_sample_rate * 0.99:
            sample_rate_ok = False
    # If there is no response then we shouldn't reject the request - return OK
    return sample_rate_ok


def all_channels_requested(bulk_request, station, requested_components):
    """
    :type bulk_request: list of lists
    :param bulk_request:
        list of lists, with each list item containing network, station,
        location, channel, starttime, endtime
    :type station: str
    :param station: string describing station name
    :type requested_components: list
    :param requested_components: list of strings, with each string being a
                                 1-character component code.

    :rtype: bool
    """
    self = False
    n_chan_required = len(requested_components)
    # Channels are understood as requested if either N,E or 1,2 are in bulk-
    # request (in case the same station delivers ZNE to one network code and
    # Z12 to another network code, e.g., station ESK).
    if ('N' in requested_components and 'E' in requested_components
            and '1' in requested_components and '2' in requested_components):
        n_chan_required = n_chan_required - 2
    n_channels_requested = 0
    for component in requested_components:
        nc = len(component)
        for request in bulk_request:
            # component needs to match the end of the requested channel-code
            if request[1] == station and request[3][-nc:] == component:
                n_channels_requested += 1
    if n_channels_requested == n_chan_required:
        self = True
    return self


def same_comp_requested(bulk, station, component):
    """
    Check if the same station-component-pair, but a different network or
    bandwith code, has already been requested.

    :type bulk_request: list
    :param bulk_request: listcontaining network, station, location, channel,
                         starttime, endtime
    :type station: str
    :param station: string describing station name
    :type component: str
    :param component: Component code to check against lines that were
                      previously added to bulk-request.

    :rtype: bool
    """
    for request in bulk:
        if (station == request[1]
                and fnmatch.fnmatch(request[3], '??' + component)):
            return True
    return False


def read_ispaq_stats(folder, networks=['??'], stations=['*'],
                     ispaq_prefixes=['all'], ispaq_suffixes=['simpleMetrics'],
                     file_type='csv', startyear=1970, endyear=2030,
                     starttime=None, endtime=None, max_ispaq_files_read=300):
    """
    function to read in Mustang-style data metrics from an ispaq-"csv"-output
    folder.
    TODO: should support ispaq 3.0 MySQL-database

    :type folder: str
    :param folder: Path to a folder that contains ISPAQ's csv-output files.
    :type networks: list
    :param networks:
        list of networks for which to read quality metrics  (accepts wildcards)
    :type stations: list
    :param stations:
        list of stations for which to read quality metrics (accepts wildcards)
    :type ispaq_prefixes: list
    :param ispaq_prefixes:
        list of prefixes (i.e., the names of a set of metrics in ispaq, e.g.,
        "all") in the filenames for which to load metrics
    :type ispaq_suffixes:
        list of suffixes (i.e., the names of a subset of metrics in ispaq,
        e.g., "simpleMetrics") in the filenames for which to load metrics
    :param ispaq_suffixes:
    :type file_type: str
    :param file_type: can be 'csv' or "parquet"
    :type startyear: int
    :param startyear: earliest year for which to load metrics
    :type endyear: int
    :param endyear: latest year for which to load metrics
    """
    Logger.info('Reading Mustang metrics for %s stations from ISPAQ for '
                + '%s - %s', str(len(stations)), startyear, endyear)
    # Check input:
    if not isinstance(networks, list):
        raise TypeError("networks should be a list")
    if not isinstance(stations, list):
        raise TypeError("stations should be a list")
    if not isinstance(ispaq_prefixes, list):
        raise TypeError("ispaq_prefixes should be a list")
    if not isinstance(ispaq_suffixes, list):
        raise TypeError("ispaq_suffixes should be a list")

    ispaq = pd.DataFrame()
    # Alternative if there's many stations and years: read a merged csv-metrics
    # file that contains all metrics from the databse (saves concat-time).
    load_all_files = True
    if (file_type == 'parquet' and
            len(stations) * (endyear + 1 - startyear) > max_ispaq_files_read):
        merged_metrics_file = glob.glob(os.path.join(os.path.split(
            os.path.expanduser(folder))[0], 'all_csv_metrics_merged.parquet'))
        merged_file_age_days = np.nan
        if merged_metrics_file:
            merged_file_age_days = (
                datetime.datetime.now() - datetime.datetime.fromtimestamp(
                    os.stat(merged_metrics_file[0]).st_mtime)).days
            if merged_file_age_days < 2:
                Logger.info('Reading aggregated metrics file %s',
                            merged_metrics_file[0])
                ispaq = pd.read_parquet(merged_metrics_file[0])
                load_all_files = False
                # Reduce size of ispaq according to requested years:
                # ispaq['startyear'] = pd.to_numeric(ispaq.start.str[0:4])
                # ispaq = ispaq[(ispaq.startyear >= startyear) &
                #               (ispaq.startyear <= endyear)]
                # ispaq.drop(columns=['startyear'])
                # Quicker with datetime rather than numbers:
                # Make starttime and endtime columns
                ispaq['starttime'] = pd.to_datetime(ispaq.start)
                # Select only relevant years / times
                ispaq = ispaq[
                    (ispaq.starttime >= datetime.datetime(startyear, 1, 1)) &
                    (ispaq.starttime <= datetime.datetime(endyear, 12, 31))]
                ispaq.drop(columns=['starttime'])

    # Find all files that match criteria
    if load_all_files:
        df_list = list()
        relevant_files = []
        # check if folder exists
        for network in networks:
            for station in stations:
                for year in range(startyear, endyear+1):
                    for ispaq_prefix in ispaq_prefixes:
                        for ispaq_suffix in ispaq_suffixes:
                            filename = (ispaq_prefix + '_' + network + '.'
                                        + station + '.x.x_'
                                        + '????-??-??_????-??-??_'
                                        + ispaq_suffix + '.' + file_type)
                            files = glob.glob(os.path.join(os.path.expanduser(
                                folder), filename))
                            # Allow filenames to cover several years - check
                            # those years are within requested range.
                            for file in files:
                                file_startyear = int(file.split('.')[-2].split(
                                    '_')[1].split('-')[0])
                                file_endyear = int(file.split('.')[-2].split(
                                    '_')[2].split('-')[0])
                                overlap = range(max(file_startyear, startyear),
                                                min(file_endyear, endyear)+1)
                                if overlap:
                                    relevant_files.append(file)

        for file in relevant_files:
            Logger.debug('Reading metrics file %s', file)
            if file_type == 'csv':
                in_df = pd.read_csv(file)
            elif file_type == 'parquet':
                in_df = pd.read_parquet(file)
            else:
                Logger.error('file_type %s not supported', file_type)
                return ispaq
            df_list.append(in_df)
        try:
            ispaq = pd.concat(df_list, axis=0)
            ispaq = ispaq.drop_duplicates(keep='last')
            ispaq.sort_values(by=['target', 'start'])
        except (ValueError, KeyError):
            Logger.error('No data quality metrics available for years %s - %s',
                        startyear, endyear)
            return ispaq

    if starttime is not None and endtime is not None:
        if isinstance(starttime, UTCDateTime):
            starttime = starttime.datetime
        if isinstance(endtime, UTCDateTime):
            endtime = endtime.datetime
        ispaq['starttime'] = pd.to_datetime(ispaq.start)
        ispaq['endtime'] = pd.to_datetime(ispaq.end)
        # Select only relevant years / times
        ispaq = ispaq[(ispaq.starttime >= starttime) &
                        (ispaq.starttime <= endtime)]
        ispaq.drop(columns=['starttime', 'endtime'])

    # Set an extra "startday"-column to use as index
    if 'startday' not in ispaq.columns:
        ispaq['startday'] = ispaq['start'].str[0:10]
        ispaq = ispaq.set_index(['startday'])
    if 'short_target' not in ispaq.columns:
        ispaq['short_target'] = ispaq['target'].str[3:-2]
    Logger.info('Successfully read in %s quality metrics values.', len(ispaq))
    return ispaq
