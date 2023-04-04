

import os

import numpy as np
import pandas as pd
import wcmatch

from itertools import chain, repeat
from collections import Counter
from timeit import default_timer

# import dill
# performance improvement for instance method pickling
# dill is required here so that the monkey-patched instance methods can be
# pickled (otherwise multiprocessing fails). Speed penalty for big streams
# appears to be around 5%, but this would be a lot worse with events or
# inventories (x10).
# dill.settings['byref'] = True
# dill.settings['protocol'] = dill.HIGHEST_PROTOCOL
from joblib import Parallel, delayed, parallel_backend
# from joblib import wrap_non_picklable_objects


from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor
from threadpoolctl import threadpool_limits

from obspy.core import Stream
from obspy.core.inventory import Inventory

# import robustraqn
from robustraqn.obspy.core.trace import Trace
# from robustraqn.obspy.core.threecomp_stream import Stream
import robustraqn.seismic_array_tools as seismic_array_tools
import robustraqn.load_events_for_detection as load_events_for_detection
import robustraqn.spectral_tools as spectral_tools

import logging
Logger = logging.getLogger(__name__)


# class Stream(object):

def init_processing(self, starttime, endtime, remove_response=False,
                    output='DISP', inv=Inventory(), pre_filt=None,
                    gain_traces=True, water_level=10,
                    min_segment_length_s=10, max_sample_rate_diff=1,
                    skip_check_sampling_rates=[20, 40, 50, 66, 75, 100, 500],
                    skip_interp_sample_rate_smaller=1e-7,
                    interpolation_method='lanczos', taper_fraction=0.005,
                    detrend_type='simple', downsampled_max_rate=None,
                    noise_balancing=False, balance_power_coefficient=2,
                    suppress_arraywide_steps=True,
                    parallel=False, cores=None, **kwargs):
    """
    Does an initial processing of the day's stream, including removing the
    response, detrending, and resampling. 
    
    :param self: Stream of traces for the day.
    :type self: :class:`obspy.core.stream.Stream`
    :param starttime: Starttime of the day.
    :type starttime: :class:`obspy.core.utcdatetime.UTCDateTime`
    :param endtime: Endtime of the day.
    :type endtime: :class:`obspy.core.utcdatetime.UTCDateTime`
    :param remove_response: Whether to remove the response or not.
    :type remove_response: bool
    :param output: Output units for the response removal, can be 'DISP',
    :type output: str
    :param inv: Inventory for the response removal.
    :type inv: :class:`obspy.core.inventory.inventory.Inventory`
    :param pre_filt: Pre-filter for the response removal.
    :type pre_filt: list
    :param min_segment_length_s: Minimum segment length for the response
    :type min_segment_length_s: float
    :param max_sample_rate_diff:
        Maximum difference in sample rates between trace and metadata.
    :type max_sample_rate_diff: float
    :param skip_check_sampling_rates:
        List of sampling rates to skip in the checks (just assume they are
        correct).
    :type skip_check_sampling_rates: list
    :param skip_interp_sample_rate_smaller:
        Skip interpolation if the sample rate differs less than this from the
        metadata.
    :type skip_interp_sample_rate_smaller: float
    :param interpolation_method: Interpolation method for the response removal.
    :type interpolation_method: str
    :param taper_fraction: Fraction of the trace to taper at the start and end
    :type taper_fraction: float
    :param detrend_type: Type of detrending to do.
    :type detrend_type: str
    :param downsampled_max_rate: Maximum sample rate to downsample to.
    :type downsampled_max_rate: float
    :param noise_balancing: Whether to balance the noise or not.
    :type noise_balancing: bool
    :param balance_power_coefficient:
        Power coefficient for the noise balancing.
    :type balance_power_coefficient: float
    :param suppress_arraywide_steps:
        Whether to suppress arraywide steps or not.
    :type suppress_arraywide_steps: bool
    :param parallel: Whether to use parallel processing or not.
    :type parallel: bool
    :param cores: Number of cores to use for parallel processing.
    :type cores: int
    :param kwargs: Additional keyword arguments to pass to the processing
    
    :returns: :class:`obspy.core.stream.Stream`
    """
    # If I merge the traces before removing the response, then the masked
    # arrays / None-values removal will mess up the response-corrected trace
    # and make it all None.
    # st.merge(method=0, fill_value=0, interpolation_samples=0)
    # If using Zero as a fill_value, then EQcorrscan will not be able to
    # automatically recognize these times and exclude the correlation-
    # value on those traces from the stacked value.

    Logger.info('Starting initial processing for %s - %s.',
                str(starttime)[0:19], str(endtime)[0:19])
    outtic = default_timer()

    seed_id_list = [tr.id for tr in self]
    unique_seed_id_list = list(dict.fromkeys(seed_id_list))

    # First check if there are consecutive Zeros in the data (happens for
    # example in NO array data; this should really be gaps rather than Zeros)
    self = self.mask_consecutive_zeros(
        min_run_length=5, starttime=starttime, endtime=endtime)
    # Second check for array-wide steps in the data
    if suppress_arraywide_steps:
        self = seismic_array_tools.mask_array_trace_offsets(
            self, split_taper_stream=True, **kwargs)

    streams = []
    if not parallel:
        for id in unique_seed_id_list:
            Logger.debug('Starting initial processing of %s for %s - %s.',
                         id, str(starttime)[0:19], str(endtime)[0:19])
            streams.append(self.select(id=id)._init_processing_per_channel(
                starttime, endtime,
                remove_response=remove_response, output=output,
                inv=inv.select(station=id.split('.')[1]), pre_filt=pre_filt,
                water_level=water_level, gain_traces=gain_traces,
                min_segment_length_s=min_segment_length_s,
                max_sample_rate_diff=max_sample_rate_diff,
                skip_check_sampling_rates=skip_check_sampling_rates,
                skip_interp_sample_rate_smaller=
                skip_interp_sample_rate_smaller,
                interpolation_method=interpolation_method,
                detrend_type=detrend_type,
                downsampled_max_rate=downsampled_max_rate,
                taper_fraction=taper_fraction,
                noise_balancing=noise_balancing,
                balance_power_coefficient=balance_power_coefficient))
        # Make a copy of the day-stream to find the values that need to be
        # masked.
        # masked_st = self.copy()
        # masked_st.merge(method=0, fill_value=0, interpolation_samples=0)
        # masked_st.trim(starttime=starttime, endtime=endtime, pad=True,
        #             nearest_sample=True, fill_value=0)

        # # Merge daystream without masking
        # self.merge(method=0, fill_value=0, interpolation_samples=0)
        # # Correct response (taper should be outside of the main day!)
        # self = try_remove_responses(self, inv, taper_fraction=0.005,
        #                             parallel=parallel, cores=cores)
        # # Trim to full day and detrend again
        # self.trim(starttime=starttime, endtime=endtime, pad=True,
        #             nearest_sample=True, fill_value=0)
        # self = self.parallel_detrend(parallel=True, cores=cores,
        #                              type='simple')

        # # Put masked array into response-corrected stream self:
        # for j, tr in enumerate(self):
        #     if isinstance(masked_st[j].data, np.ma.MaskedArray):
        #         tr.data = np.ma.masked_array(
        #     tr.data, mask=masked_st[j].data.mask)
    else:
        if cores is None:
            cores = min(len(self), cpu_count())

        with threadpool_limits(limits=1, user_api='blas'):
            streams = Parallel(n_jobs=cores)(
                # delayed(self.select(id=id)._init_processing_per_channel)
                delayed(_init_processing_per_channel)
                (self.select(id=id), starttime=starttime, endtime=endtime,
                 remove_response=remove_response, output=output,
                 inv=inv.select(station=id.split('.')[1], starttime=starttime,
                                endtime=endtime), pre_filt=pre_filt,
                 water_level=water_level, gain_traces=gain_traces,
                 min_segment_length_s=min_segment_length_s,
                 max_sample_rate_diff=max_sample_rate_diff,
                 skip_check_sampling_rates=skip_check_sampling_rates,
                 skip_interp_sample_rate_smaller=
                 skip_interp_sample_rate_smaller,
                 interpolation_method=interpolation_method,
                 detrend_type=detrend_type,
                 taper_fraction=taper_fraction,
                 downsampled_max_rate=downsampled_max_rate,
                 noise_balancing=noise_balancing,
                 balance_power_coefficient=balance_power_coefficient)
                for id in unique_seed_id_list)
            # st = Stream([tr for trace_st in streams for tr in trace_st])
    self = load_events_for_detection._merge_streams(streams)

    outtoc = default_timer()
    Logger.info(
        'Initial processing of %s traces in stream took: {0:.4f}s'.format(
            outtoc - outtic), str(len(self)))

    return self


def init_processing_w_rotation(
        self, starttime, endtime, remove_response=False, output='DISP',
        inv=Inventory(), pre_filt=None, gain_traces=True, water_level=10,
        sta_translation_file='',
        parallel=False, cores=None, n_threads=1, suppress_arraywide_steps=True,
        min_segment_length_s=10, max_sample_rate_diff=1,
        skip_check_sampling_rates=[20, 40, 50, 66, 75, 100, 500],
        skip_interp_sample_rate_smaller=1e-7, interpolation_method='lanczos',
        taper_fraction=0.005, detrend_type='simple', downsampled_max_rate=None,
        std_network_code="NS", std_location_code="00", std_channel_prefix="BH",
        noise_balancing=False, balance_power_coefficient=2, **kwargs):
    """Copilot, please write the docstring.
    :type self: :class:`obspy.core.stream.Stream`
    :param self: Stream of data to process
    :type starttime: :class:`obspy.core.utcdatetime.UTCDateTime`
    :param starttime: Start time of the data to process
    :type endtime: :class:`obspy.core.utcdatetime.UTCDateTime`
    :param endtime: End time of the data to process
    :type remove_response: bool
    :param remove_response: Whether to remove the response or not
    :type output: str
    :param output: Output units of the data, can be DISP, VEL, ACC
    :type inv: :class:`obspy.core.inventory.inventory.Inventory`
    :param inv: Inventory to use for response removal
    :type pre_filt: list
    :param pre_filt: Pre-filter to use for response removal
    :type sta_translation_file: str
    :param sta_translation_file: Path to station translation file
    :type parallel: bool
    :param parallel: Whether to use parallel processing or not
    :type cores: int
    :param cores: Number of cores to use for parallel processing
    :type n_threads: int
    :param n_threads: Number of threads to use for parallel processing
    :type suppress_arraywide_steps: bool
    :param suppress_arraywide_steps: Whether to suppress array-wide steps
    :type min_segment_length_s: float
    :param min_segment_length_s: Minimum segment length in seconds
    :type max_sample_rate_diff: float
    :param max_sample_rate_diff:
        Maximum difference in sample rates between traces and metadata.
    :type skip_check_sampling_rates: list
    :param skip_check_sampling_rates: Sample rates to skip checking
    :type skip_interp_sample_rate_smaller: float
    :param skip_interp_sample_rate_smaller:
        Maximum sample rate difference between traces and metadata to skip
        interpolation.
    :type interpolation_method: str
    :param interpolation_method: Interpolation method to use
    :type taper_fraction: float
    :param taper_fraction: Fraction of trace to taper
    :type detrend_type: str
    :param detrend_type: Type of detrending to do
    :type downsampled_max_rate: float
    :param downsampled_max_rate: Maximum sample rate to downsample to
    :type std_network_code: str
    :param std_network_code: Standard network code to use
    :type std_location_code: str
    :param std_location_code: Standard location code to use
    :type std_channel_prefix: str
    :param std_channel_prefix: Standard channel prefix to use
    :type noise_balancing: bool
    :param noise_balancing: Whether to balance the noise or not
    :type balance_power_coefficient: float
    :param balance_power_coefficient: Power coefficient for noise balancing
    
    :return: Stream of processed data
    :rtype: :class:´obspy.core.stream.Stream´
    """
    # If I merge the traces before removing the response, then the masked
    # arrays / None-values removal will mess up the response-corrected trace
    # and make it all None.
    # If using Zero as a fill_value, then EQcorrscan will not be able to
    # automatically recognize these times and exclude the correlation-
    # value on those traces from the stacked value.

    Logger.info('Starting initial processing for %s - %s.',
                str(starttime)[0:19], str(endtime)[0:19])
    outtic = default_timer()

    net_sta_loc = [(tr.stats.network, tr.stats.station, tr.stats.location)
                   for tr in self]
    if len(net_sta_loc) == 0:
        Logger.error('There are no traces to do initial processing on %s',
                     str(starttime))
        return self

    # First check if there are consecutive Zeros in the data (happens for
    # example in NO array data; this should really be gaps rather than Zeros)
    self = self.mask_consecutive_zeros(
        min_run_length=5, starttime=starttime, endtime=endtime)
    # Second check for array-wide steps in the data
    if suppress_arraywide_steps:
        self = seismic_array_tools.mask_array_trace_offsets(
            self, split_taper_stream=True, **kwargs)

    # Sort unique-ID list by most common, so that 3-component stations
    # appear first and are processed first in parallel loop (for better load-
    # balancing)
    # net_sta_loc = list(chain.from_iterable(repeat(i, c)
    #                    for i,c in Counter(net_sta_loc).most_common()))
    # Need to sort list by original order after set() ##

    # Better: Sort by: whether needs rotation; npts per 3-comp stream
    unique_net_sta_loc_list = list(dict.fromkeys(net_sta_loc))
    three_comp_strs = [
        self.select(network=nsl[0], station=nsl[1], location=nsl[2])
        for nsl in unique_net_sta_loc_list]
    sum_npts = [sum([tr.stats.npts for tr in s]) for s in three_comp_strs]
    needs_rotation = [
        '1' in [tr.stats.channel[-1] for tr in s] for s in three_comp_strs]
    unique_net_sta_loc_list = [x for x, _, _ in sorted(
        zip(unique_net_sta_loc_list, needs_rotation, sum_npts),
        key=lambda y: (y[1], y[2]), reverse=True)]

    if not parallel:
        streams = []
        for nsl in unique_net_sta_loc_list:
            Logger.info(
                'Starting initial processing of %s for %s - %s.',
                '.'.join(nsl), str(starttime)[0:19], str(endtime)[0:19])
            with threadpool_limits(limits=n_threads, user_api='blas'):
                Logger.info(
                    'Starting initial 3-component processing with 1 process '
                    'with up to %s threads.', str(n_threads))
                streams.append(self.select(
                    network=nsl[0], station=nsl[1], location=nsl[2]
                    )._init_processing_per_channel_w_rotation(
                        starttime=starttime, endtime=endtime,
                        remove_response=remove_response,
                        output=output, pre_filt=pre_filt,
                        gain_traces=gain_traces, water_level=water_level,
                        inv=inv.select(station=nsl[1], starttime=starttime,
                                    endtime=endtime),
                        min_segment_length_s=min_segment_length_s,
                        max_sample_rate_diff=max_sample_rate_diff,
                        skip_check_sampling_rates=skip_check_sampling_rates,
                        skip_interp_sample_rate_smaller=
                        skip_interp_sample_rate_smaller,
                        interpolation_method=interpolation_method,
                        sta_translation_file=sta_translation_file,
                        std_network_code=std_network_code,
                        std_location_code=std_location_code,
                        std_channel_prefix=std_channel_prefix,
                        detrend_type=detrend_type,
                        downsampled_max_rate=downsampled_max_rate,
                        taper_fraction=taper_fraction,
                        noise_balancing=noise_balancing,
                        balance_power_coefficient=balance_power_coefficient,
                        **kwargs))
    # elif thread_parallel and n_threads:

    else:
        if cores is None:
            cores = min(len(self), cpu_count())
        # Check if I can allow multithreading in each of the parallelized
        # subprocesses:
        # thread_parallel = False
        n_threads = 1
        if cores > 2 * len(self):
            # thread_parallel = True
            n_threads = int(cores / len(self))
        Logger.info('Starting initial 3-component processing with %s parallel '
                    'processes with up to %s threads each.', str(cores),
                    str(n_threads))

        with threadpool_limits(limits=n_threads, user_api='blas'):
            streams = Parallel(n_jobs=cores)(
                delayed(_init_processing_per_channel_w_rotation)
                (self.select(network=nsl[0], station=nsl[1], location=nsl[2]),
                 starttime=starttime, endtime=endtime,
                 remove_response=remove_response,
                 output=output, inv=inv.select(
                     station=nsl[1], starttime=starttime, endtime=endtime),
                 pre_filt=pre_filt, gain_traces=gain_traces,
                 water_level=water_level,
                 sta_translation_file=sta_translation_file,
                 min_segment_length_s=min_segment_length_s,
                 max_sample_rate_diff=max_sample_rate_diff,
                 skip_check_sampling_rates=skip_check_sampling_rates,
                 skip_interp_sample_rate_smaller=
                 skip_interp_sample_rate_smaller,
                 interpolation_method=interpolation_method,
                 std_network_code=std_network_code,
                 std_location_code=std_location_code,
                 std_channel_prefix=std_channel_prefix,
                 detrend_type=detrend_type,
                 taper_fraction=taper_fraction,
                 downsampled_max_rate=downsampled_max_rate,
                 noise_balancing=noise_balancing,
                 balance_power_coefficient=balance_power_coefficient,
                 parallel=False, cores=None, **kwargs)
                for nsl in unique_net_sta_loc_list)
    self = load_events_for_detection._merge_streams(streams)
    # st = Stream([tr for trace_st in streams for tr in trace_st])

    outtoc = default_timer()
    Logger.info('Initial processing of streams took: {0:.4f}s'.format(
        outtoc - outtic))
    return self


def _init_processing_per_channel_w_rotation(
        self, starttime, endtime, remove_response=False, output='DISP',
        pre_filt=None, inv=Inventory(), gain_traces=True, water_level=10,
        sta_translation_file='',
        min_segment_length_s=10, max_sample_rate_diff=1,
        skip_check_sampling_rates=[20, 40, 50, 66, 75, 100, 500],
        skip_interp_sample_rate_smaller=1e-7, interpolation_method='lanczos',
        std_network_code="NS", std_location_code="00", std_channel_prefix="BH",
        detrend_type='simple', taper_fraction=0.005, downsampled_max_rate=25,
        noise_balancing=False, balance_power_coefficient=2, apply_agc=False,
        agc_window_sec=5, agc_method='gismo',
        parallel=False, cores=1, thread_parallel=False, n_threads=1, **kwargs):
    """
    Inner loop over which the initial processing can be parallelized
    
    :param st: Stream object with three component traces from one station
    :type st: :class:`~obspy.core.stream.Stream`
    :param starttime: Starttime of the day to be processed
    :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param endtime: Endtime of the day to be processed
    :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param remove_response: Remove instrument response
    :type remove_response: bool
    :param output:
        Output units of instrument response removal, can be one of 'DISP',
        'VEL', 'ACC'.
    :type output: str
    :param pre_filt: Pre-filter for instrument response removal
    :type pre_filt: list
    :param inv: Inventory object with instrument response information
    :type inv: :class:`~obspy.core.inventory.inventory.Inventory`
    :param sta_translation_file: Path to station translation file
    :type sta_translation_file: str
    :param min_segment_length_s: Minimum length of trace segments in seconds
    :type min_segment_length_s: float
    :param max_sample_rate_diff:
        Maximum difference between sampling rates of traces and inventory
        metadata to allow for trace resampling.
    :type max_sample_rate_diff: float
    :param skip_check_sampling_rates:
        Sampling rates that are not checked for match between traces and
        inventory metadata.
    :type skip_check_sampling_rates: list
    :param skip_interp_sample_rate_smaller:
        When difference in sampling rate between traces and inventory metadata
        is smaller than this value, no resampling is performed.
    :type skip_interp_sample_rate_smaller: float
    :param interpolation_method: Interpolation method for resampling
    :type interpolation_method: str
    :param std_network_code: Standard network code
    :type std_network_code: str
    :param std_location_code: Standard location code
    :type std_location_code: str
    :param std_channel_prefix: Standard channel prefix
    :type std_channel_prefix: str
    :param detrend_type: Type of detrending
    :type detrend_type: str
    :param taper_fraction: Fraction of trace to taper
    :type taper_fraction: float
    :param downsampled_max_rate: Maximum sampling rate of downsampled data
    :type downsampled_max_rate: float
    :param noise_balancing: Balance noise levels
    :type noise_balancing: bool
    :param balance_power_coefficient: Power coefficient for noise balancing
    :type balance_power_coefficient: float
    :param apply_agc: Apply automatic gain control
    :type apply_agc: bool
    :param agc_window_sec: Length of automatic gain control window in seconds
    :type agc_window_sec: float
    :param agc_method:
        Method for automatic gain control, can be 'gismo' or 'obspy'
    :type agc_method: str
    :param parallel: Parallelize processing
    :type parallel: bool
    :param cores: Number of cores to use for parallel processing
    :type cores: int
    :param thread_parallel: Parallelize processing with threads
    :type thread_parallel: bool
    :param n_threads: Number of threads to use for parallel processing
    :type n_threads: int
    :param kwargs: Additional keyword arguments
    :type kwargs: dict

    :return: Stream object with initially processed traces
    :rtype: :class:`~obspy.core.stream.Stream`
    """
    outtic = default_timer()

    # # First check if there are consecutive Zeros in the data (happens for
    # # example in Norsar data; this should be gaps rather than Zeros)
    # self = self.mask_consecutive_zeros(min_run_length=5)
    # if len(self) == 0:
    #     return self
    # # Taper all the segments after inserting nans
    # self = self.taper_trace_segments()

    # Second, check trace segments for strange sampling rates and segments that
    # are too short:
    self, st_normalized = self.check_normalize_sampling_rate(
        inv, min_segment_length_s=min_segment_length_s,
        max_sample_rate_diff=max_sample_rate_diff,
        skip_check_sampling_rates=skip_check_sampling_rates,
        skip_interp_sample_rate_smaller=skip_interp_sample_rate_smaller,
        interpolation_method=interpolation_method)

    # If numpy data arrays are read-only (not writeable), need to re-create the
    # arrays:
    for tr in self:
        if not tr.data.flags.writeable:
            Logger.debug('Array for trace %s: %s', tr, str(tr.data.flags))
            if isinstance(tr.data, np.ma.MaskedArray):
                tr.data = np.ma.MaskedArray(tr.data)
            else:
                tr.data = np.array(tr.data)
    # Detrend
    self = self.detrend(type=detrend_type)
    # Merge, but keep "copy" of the masked array for filling back
    # Make a copy of the day-stream to find the values that need to be masked.
    masked_st = self.copy()
    masked_st.merge(method=0, fill_value=0, interpolation_samples=0)
    masked_st.trim(starttime=starttime, endtime=endtime, pad=True,
                   nearest_sample=True, fill_value=0)
    masked_st_tr_dict = dict()
    for tr in masked_st:
        masked_st_tr_dict[tr.id] = tr

    # Merge daystream without masking
    # self.merge(method=0, fill_value=0, interpolation_samples=0)
    # 2021-01-22: changed merge method to below one to fix error with
    #             incomplete day.
    self = self.merge(method=1, fill_value=0, interpolation_samples=-1)
    # Correct response (taper should be outside of the main day!)
    if remove_response:
        self = self.try_remove_responses(
            inv.select(starttime=starttime, endtime=endtime),
            taper_fraction=0.005, output=output, pre_filt=pre_filt,
            gain_traces=gain_traces, water_level=water_level,
            parallel=parallel, cores=cores, n_threads=n_threads)
    # Trim to full day and detrend again
    self.trim(starttime=starttime, endtime=endtime, pad=True,
            nearest_sample=True, fill_value=0)
    self.detrend(type=detrend_type)

    # normalize NSLC codes, including rotation
    self, trace_id_change_dict = self.normalize_nslc_codes(
        inv, sta_translation_file=sta_translation_file,
        std_network_code=std_network_code, std_location_code=std_location_code,
        std_channel_prefix=std_channel_prefix, parallel=False, cores=1,
        thread_parallel=thread_parallel, n_threads=n_threads, **kwargs)

    # Do noise-balancing by the station's PSDPDF average
    if noise_balancing:
        Logger.debug('Applying noise balancing to continuous data.')
        # if not hasattr(st, "balance_noise"):
        #     bound_method = st_balance_noise.__get__(st)
        #     st.balance_noise = bound_method
        self = self.filter('highpass', freq=0.1, zerophase=True).detrend()
        self = spectral_tools.st_balance_noise(
            self, inv, balance_power_coefficient=balance_power_coefficient,
            sta_translation_file=sta_translation_file)
        self = self.taper(0.005, type='hann', max_length=None, side='both')

    # Put masked array into response-corrected stream st:
    for tr in self:
        # Find the mask that fits to the trace (which may have changed id)
        inv_trace_id_change_dict = {
            v: k for k, v in trace_id_change_dict.items()}
        old_tr_id = inv_trace_id_change_dict[tr.id]
        masked_st_tr = masked_st_tr_dict[old_tr_id]
        if isinstance(masked_st_tr.data, np.ma.MaskedArray):
            tr.data = np.ma.masked_array(tr.data,
                                         mask=masked_st_tr.data.mask)

    # Downsample if necessary
    if downsampled_max_rate is not None:
        for tr in self:
            if tr.stats.sampling_rate > downsampled_max_rate:
                tr.resample(sampling_rate=downsampled_max_rate,
                            no_filter=False, window='hann')

    # Check that at least 80 (x) % of data are not masked:
    # TODO: Don't do consecutive Zero checks here, just the other checks 
    self = self.mask_consecutive_zeros(
        min_run_length=None, starttime=starttime, endtime=endtime)
    if len(self) == 0:
        return self

    outtoc = default_timer()
    try:
        Logger.debug(
            'Initial processing of %s traces in stream %s took: '
            '{0:.4f}s'.format(outtoc - outtic), str(len(self)), self[0].id)
    except Exception as e:
        Logger.warning(e)

    return self


def _init_processing_per_channel(
        self, starttime, endtime, remove_response=False, output='DISP',
        inv=Inventory(), gain_traces=True, min_segment_length_s=10,
        max_sample_rate_diff=1,
        skip_check_sampling_rates=[20, 40, 50, 66, 75, 100, 500],
        skip_interp_sample_rate_smaller=1e-7, interpolation_method='lanczos',
        detrend_type='simple', taper_fraction=0.005, pre_filt=None,
        water_level=10,
        downsampled_max_rate=None, noise_balancing=False,
        balance_power_coefficient=2, sta_translation_file='',
        normalize_all_station_channels=False, exclude_component_codes=['H'],
        n_threads=1, **kwargs):
    """
    Inner loop over which the initial processing can be parallelized for
    individual channels (rather than sets of three-component channels).

    :param self: input stream with traces
    :type self: :class:`obspy.core.stream.Stream`
    :param starttime: start time of data to process
    :type starttime: :class:`obspy.core.utcdatetime.UTCDateTime`
    :param endtime: end time of data to process
    :type endtime: :class:`obspy.core.utcdatetime.UTCDateTime`
    :param remove_response: remove instrument response, defaults to False
    :type remove_response: bool, optional
    :param output: output units, defaults to 'DISP'
    :type output: str, optional
    :param inv: inventory, defaults to Inventory()
    :type inv: :class:`obspy.core.inventory.inventory.Inventory`, optional
    :param min_segment_length_s: minimum segment length in s to keep trace,
    :type min_segment_length_s: float, optional
    :param max_sample_rate_diff:
        Maximum difference in sampling rate between traces and metadata in
        inventory for which trace will be resampled to match metadata.
    :type max_sample_rate_diff: float, optional
    :param skip_check_sampling_rates: sampling rates to skip check for
    :type skip_check_sampling_rates: list, optional
    :param skip_interp_sample_rate_smaller: skip interpolation if sampling rate
    :type skip_interp_sample_rate_smaller: float, optional
    :param interpolation_method: interpolation method, defaults to 'lanczos'
    :type interpolation_method: str, optional
    :param detrend_type: detrend type, defaults to 'simple'
    :type detrend_type: str, optional
    :param taper_fraction: taper fraction, defaults to 0.005
    :type taper_fraction: float, optional
    :param pre_filt: pre-filter, defaults to None
    :type pre_filt: list, optional
    :param downsampled_max_rate: maximum sampling rate after downsampling,
    :type downsampled_max_rate: float, optional
    :param noise_balancing: noise balancing, defaults to False
    :type noise_balancing: bool, optional
    :param balance_power_coefficient: balance power coefficient, defaults to 2
    :type balance_power_coefficient: int, optional
    :param sta_translation_file: station translation file, defaults to ''
    :type sta_translation_file: str, optional
    :param normalize_all_station_channels: 
        Specify whether channels belonging to the same station should be
        normalized to the same (most common) sampling rate).
    :type normalize_all_station_channels: bool, optional
    :param exclude_component_codes: exclude component codes, defaults to ['H']
    :type exclude_component_codes: list, optional
    :param n_threads: number of threads, defaults to 1
    :type n_threads: int, optional
    :return: stream with processed traces
    :rtype: :class:`obspy.core.stream.Stream`
    :param kwargs: additional keyword arguments
    :type kwargs: dict

    :return: stream with processed traces
    :rtype: :class:`obspy.core.stream.Stream`
    """
    # # First check if there are consecutive Zeros in the data (happens for
    # # example in NO array data; this should really be gaps rather than Zeros)
    # self = self.mask_consecutive_zeros(
    #     min_run_length=5, starttime=starttime, endtime=endtime)
    # # Taper all the segments
    # self = self.taper_trace_segments()
    
    if len(self) == 0:
        return self

    # Second check trace segments for strange sampling rates and segments that
    # are too short:
    self, st_normalized = self.check_normalize_sampling_rate(
        inv, min_segment_length_s=min_segment_length_s,
        max_sample_rate_diff=max_sample_rate_diff,
        skip_check_sampling_rates=skip_check_sampling_rates,
        skip_interp_sample_rate_smaller=skip_interp_sample_rate_smaller,
        interpolation_method=interpolation_method)
    if len(self) == 0:
        return self

    # Check whether all the traces at the same station have the same samling
    # rate:
    if normalize_all_station_channels:
        self = self.check_normalize_station_sample_rates(
            exclude_component_codes=exclude_component_codes, **kwargs)
    
    # Detrend
    self.detrend(type=detrend_type)
    # Merge, but keep "copy" of the masked array for filling back
    # Make a copy of the day-stream to find the values that need to be masked.
    masked_st = self.copy()
    masked_st.merge(method=0, fill_value=0, interpolation_samples=0)
    masked_st.trim(starttime=starttime, endtime=endtime, pad=True,
                   nearest_sample=True, fill_value=0)
    masked_st_tr_dict = dict()
    for tr in masked_st:
        masked_st_tr_dict[tr.id] = tr

    # Merge daystream without masking
    # self.merge(method=0, fill_value=0, interpolation_samples=0)
    # 2021-01-22: changed merge method to below one to fix error with
    #             incomplete day.
    self.merge(method=1, fill_value=0, interpolation_samples=-1)
    # Correct response (taper should be outside of the main day!)
    if remove_response:
        self = self.try_remove_responses(
            inv.select(starttime=starttime, endtime=endtime),
            taper_fraction=taper_fraction, output=output, pre_filt=pre_filt,
            gain_traces=gain_traces, water_level=water_level,
            parallel=False, cores=1, n_threads=n_threads)
    # Detrend now?
    self = self.detrend(type='simple')
    # st = st.detrend(type='linear')

    if noise_balancing:
        # Need to do some prefiltering to avoid phase-shift effects when very
        # low frequencies are boosted
        self = self.filter('highpass', freq=0.1, zerophase=True)  # detrend()
        self = spectral_tools.st_balance_noise(
            self, inv, balance_power_coefficient=balance_power_coefficient,
            sta_translation_file=sta_translation_file)
        self = self.taper(0.005, type='hann', max_length=None, side='both'
                      ).detrend(type='linear')
        # self = self.detrend(type='linear').taper(
        #    0.005, type='hann', max_length=None, side='both')

    # Trim to full day and detrend again
    self = self.trim(starttime=starttime, endtime=endtime, pad=True,
                 nearest_sample=True, fill_value=0)
    self = self.detrend(type=detrend_type)

    # Put masked array into response-corrected stream st:
    # masked_st
    for tr in self:
        masked_st_tr = masked_st_tr_dict[tr.id]
        if isinstance(masked_st_tr.data, np.ma.MaskedArray):
            tr.data = np.ma.masked_array(tr.data,
                                         mask=masked_st_tr.data.mask)
    # for j, tr in enumerate(st):
    #     if isinstance(masked_st.traces[j].data, np.ma.MaskedArray):
    #         tr.data = np.ma.masked_array(tr.data,
    #                                      mask=masked_st.traces[j].data.mask)
    
    # Check that at least 80 (x) % of data are not masked:
    # TODO: Don't do consecutive Zero checks here, just the other checks 
    self = self.mask_consecutive_zeros(
        min_run_length=None, starttime=starttime, endtime=endtime)
    if len(self) == 0:
        return self

    # Downsample if necessary
    if downsampled_max_rate is not None:
        for tr in self:
            if tr.stats.sampling_rate > downsampled_max_rate:
                tr.resample(sampling_rate=downsampled_max_rate,
                            no_filter=False, window='hann')

    return self


def _mask_consecutive(data, value_to_mask=0, min_run_length=5, axis=-1):
    """
    from user https://stackoverflow.com/users/2988730/mad-physicist posted at
    https://stackoverflow.com/questions/63741396/how-to-build-a-mask-true-or-
    false-for-consecutive-zeroes-in-a-row-in-a-pandas
    - posted under license CC-BY-SA 4.0 (compatible with GPLv3 used in
      RobustRAQN, see license: https://creativecommons.org/licenses/by-sa/4.0/)
    - variable names modified
    
    :param data: 1D or 2D array of data to mask
    :type data: :class:`numpy.ndarray`
    :param value_to_mask: value to mask
    :type value_to_mask: int
    :param min_run_length:
        Minimum number of consecutive values that equal value_to_mask to mask.
    :type min_run_length: int
    :param axis: axis along which to mask
    :type axis: int

    :return: Masked array
    :rtype: :class:`numpy.ndarray`
    """
    shape = list(data.shape)
    shape[axis] = 1;
    z = np.zeros(shape, dtype=bool)
    mask = np.concatenate((z, data == value_to_mask, z), axis=axis)
    locs = np.argwhere(np.diff(mask, axis=axis))
    run_lengths = locs[1::2, axis] - locs[::2, axis]
    valid_runs = np.flatnonzero(run_lengths >= min_run_length)
    result = np.zeros(data.shape, dtype=np.int8)
    v = 2 * valid_runs
    result[tuple(locs[v, :].T)] = 1
    v += 1
    v = v[locs[v, axis] < result.shape[axis]]
    result[tuple(locs[v, :].T)] = -1
    return np.cumsum(result, axis=axis, out=result).view(bool)


def mask_consecutive_zeros(self, min_run_length=5, min_data_percentage=80,
                           starttime=None, endtime=None, cores=None):
    """Mask consecutive Zeros in trace

    :param st: input stream
    :type st: :class:`obspy.core.stream.Stream`
    :param min_run_length:
        minmum number of consecutive zero-samples to be masked, defaults to 5.
    :type min_run_length: int, optional
    :param min_data_percentage:
        minimum percentage of actual data in trace to retain, defaults to 80
    :type min_data_percentage: float, optional
    :param starttime: starttime of data vailability check, defaults to None
    :type starttime: :class:`obspy.core.event.UTCDateTime`, optional
    :param endtime: endtime of data availability check, defaults to None
    :type endtime: :class:`obspy.core.event.UTCDateTime`, optional
    :param cores: number of cores to use in checks, defaults to None
    :type cores: int, optional

    :return: stream with consecutive zero-samples masked.
    :rtype: :class:`obspy.core.stream.Stream`
    """
    if starttime is None:
        starttime = min([tr.stats.starttime for tr in self])
    if endtime is None:
        endtime = min([tr.stats.endtime for tr in self])
    if min_run_length is not None:
        if cores is None:
            cores = min(len(self), cpu_count())
        with ThreadPoolExecutor(max_workers=cores) as executor:
            # Because numpy releases GIL threading can use multiple cores
            consecutive_zeros_masks = executor.map(
                _mask_consecutive, [tr.data for tr in self.traces])
        for tr, consecutive_zeros_mask in zip(self, consecutive_zeros_masks):
            # consecutive_zeros_mask = _mask_consecutive(
            #     tr.data, value_to_mask=0, min_run_length=min_run_length,
            #     axis=-1)
            if np.any(consecutive_zeros_mask):
                # Convert trace data to masked array if required
                if isinstance(tr.data, np.ma.MaskedArray):
                    # Combine the previous mask and the new nonzero mask:
                    mask = tr.data.mask
                    mask[np.where(consecutive_zeros_mask == 1)] = 1
                else:
                    mask = consecutive_zeros_mask
                Logger.info(
                    'Trace %s contain more than %s consecutive zeros, '
                    'masking Zero data', tr, min_run_length)
                tr.data = np.ma.MaskedArray(
                    data=tr.data, mask=mask, fill_value=0)
        self = self.split()  # After splitting there should be no masks
    removal_st = Stream()
    min_data_fraction = min_data_percentage / 100
    streams_are_masked = False
    for tr in self:
        if hasattr(tr.data, 'mask'):
            streams_are_masked = True
            # Masked values indicate Zeros or missing data
            n_masked_values = np.sum(tr.data.mask)
            # Maximum 20 % should be masked
            if n_masked_values / tr.stats.npts > (1 - min_data_fraction):
                removal_st += tr
    # Once traces are split they should not have masks any more. So need to
    # check length of data in trace again.
    if not streams_are_masked:
        uniq_tr_ids = list(dict.fromkeys([tr.id for tr in self]))
        for uniq_tr_id in uniq_tr_ids:
            trace_len_in_seconds = np.sum(
                [tr.stats.npts / tr.stats.sampling_rate for tr in self])
            if trace_len_in_seconds < (endtime - starttime) * min_data_fraction:
                for tr in self:
                    if tr not in removal_st and tr.id == uniq_tr_id:
                        removal_st += tr
    # Remove trace if not enough real data (not masked, not Zero)
    for tr in removal_st:
        Logger.info(
            'After masking consecutive zeros, there is less than %s %% data '
            'for trace %s - removing.', min_data_percentage, tr.id)
        self.remove(tr)
    return self


def taper_trace_segments(self, min_length_s=2.0, max_percentage=0.1,
                         max_length=1.0, **kwargs):
    """
    Taper all segments / traces after masking problematic values (e.g., spikes,
    zeros).

    :param self: input stream with traces
    :type self: :class:`obspy.core.stream.Stream`
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
    self = self.split()
    n_tr_before_sel = len(self)
    if min_length_s is not None:
        self = Stream([
            tr for tr in self
            if tr.stats.npts >= tr.stats.sampling_rate * min_length_s])
        n_tr_after_sel = len(self)
        if n_tr_after_sel != n_tr_before_sel:
            Logger.info(
                '%s: Removed %s (of %s) traces shorter than %s s',
                '.'.join(self[0].id.split('.')[0:3]),
                n_tr_before_sel  - n_tr_after_sel,
                n_tr_before_sel, min_length_s)
    self = self.taper(max_percentage=max_percentage, max_length=max_length)
    return self


def check_normalize_sampling_rate(
        self, inventory, min_segment_length_s=10, max_sample_rate_diff=1,
        skip_check_sampling_rates=[20, 40, 50, 66, 75, 100, 250, 500],
        skip_interp_sample_rate_smaller=1e-7, interpolation_method='lanczos',
        **kwargs):
    """
    Function to check the sampling rates of traces against the response-
        provided sampling rate and against all other segments from the same
        NSLC-object in the stream. Traces with mismatching sample-rates are
        excluded. If the sample rate is only slightly off, then the trace is
        interpolated onto the response-defined sample rate.
        If there is no matching response, then the traces are checked against
        other traces with the same NSLC-identified id.
        The sample rate can be adjusted without interpolation if it is just
        very slightly off due to floating-point issues (e.g., 99.9999998 vs
        100 Hz). A value of 1e-7 implies that for a 100 Hz-sampled channel, the
        sample-wise offset across 24 hours is less than 1 sample.

    :type self: obspy.core.Stream
    :param self: Stream that will be checked and corrected.
    :type inventory: obspy.core.Inventory
    :param inventory:
        Inventory of stations that is required to check for what the sampling
        rate of the channels should be.
    :type min_segment_length_s: float
    :param min_segment_length_s:
        Minimum length of segments that should be kept in stream - shorter
        segments are most likley signs of some data problems that cannot easily
        be mended.
    :type max_sample_rate_diff: float
    :param max_sample_rate_diff:
        Upper limit for the difference in sampling rate between the intended 
        sampling rate (from channel response) and the actual sampling rate of
        the trace. If the difference is larger than max_sample_rate_diff, the
        trace will be removed.
    :type skip_check_sampling_rates: list of float
    :param skip_check_sampling_rates:
        List of typical sampling rates for which not interpolation-checks
        are performed (when sampling rates match exactly).
    :type skip_interp_sample_rate_smaller:
    :param skip_interp_sample_rate_smaller:
        Limit under which no interpolation of the trace is required; instead
        the time stamps can just be adjusted to stretch the trace slightly if
        the sampling rate is just slighly off (e.g, one sample per day). Can
        save time.
    :type interpolation_method: string
    :param interpolation_method:
        Type of method for interpolation used by
        `obspy.core.trace.Trace.interpolate`

    returns:
        stream with corrected traces, True/False if any traces were corrected
        or not.
    :rtype: tuple of (:class:`obspy.core.event.Catalog`, boolean)
    """

    # First, throw out trace bits that are too short, and hence, likely
    # problematic (e.g., shorter than 10 s):
    keep_st = Stream()
    # st_copy = self.copy()
    st_tr_to_be_removed = Stream()
    for tr in self:
        if tr.stats.npts < min_segment_length_s * tr.stats.sampling_rate:
            # st_copy.remove(tr)
            st_tr_to_be_removed += tr
    for tr in st_tr_to_be_removed:
        if tr in self:
            self.remove(tr)
    # self = st_copy

    # Create list of unique Seed-identifiers (NLSC-objects)
    seed_id_list = [tr.id for tr in self]
    unique_seed_id_list = list(dict.fromkeys(seed_id_list))

    # Do a quick check: if all sampling rates are the same, and all values are
    # one of the allowed default values, then skip all the other tests.
    channel_rates = [tr.stats.sampling_rate for tr in self]
    if (all([cr in skip_check_sampling_rates for cr in channel_rates])
            and len(set(channel_rates)) <= 1):
        return self, False

    # Now do a thorough check of the sampling rates; exclude traces if needed;
    # interpolate if needed.
    new_st = Stream()
    # Check what the trace's sampling rate should be according to the inventory
    for seed_id in unique_seed_id_list:
        # Need to check for the inventory-information only once per seed_id
        # (Hopefully - I'd rather not do this check for every segment in a
        # mseed-file; in case there are many segments.)
        check_st = self.select(id=seed_id).copy()
        tr = check_st[0]

        check_inv = inventory.select(
            network=tr.stats.network, station=tr.stats.station,
            location=tr.stats.location,
            channel=tr.stats.channel, time=tr.stats.starttime)
        if len(check_inv) > 0:
            found_matching_resp = True
        else:
            found_matching_resp, tr, check_inv = (
                tr.try_find_matching_response(inventory))

        def_channel_sample_rate = None
        if found_matching_resp:
            if (len(check_inv.networks) > 1
                    or len(check_inv.networks[0].stations) > 1
                    or len(check_inv.networks[0].stations[0].channels) > 1):
                Logger.warning('Found more than one matching response for '
                               + 'trace %s, using first.', seed_id)

            matching_st = Stream()
            def_channel_sample_rate =\
                check_inv.networks[0].stations[0].channels[0].sample_rate
            for tr in check_st:
                sample_rate_diff =\
                    tr.stats.sampling_rate - def_channel_sample_rate
                if abs(sample_rate_diff) > max_sample_rate_diff:
                    Logger.warning(
                        'Big sampling rate mismatch between trace segment %s '
                        '(%s Hz) starting on %s and channel response '
                        'information (%s Hz), removing offending traces.',
                        tr.id, tr.stats.sampling_rate,
                        str(tr.stats.starttime)[0:19], def_channel_sample_rate)
                else:
                    matching_st += tr
        else:
            # If there is no matching response information, keep all traces for
            # a bit.
            matching_st = check_st

        # Now check to keep only traces with the most common sampling rate for
        # each channel (or with sampling rates that deviate only slightly).
        channel_rates = [tr.stats.sampling_rate for tr in matching_st]
        if not found_matching_resp:
            # TODO: weight the channel_rates by the length of each segment
            c = Counter(channel_rates)
            common_channel_sampling_rate = c.most_common(1)[0][0]
            # Make sure the definitive channel-sampling rate is an integer
            def_channel_sample_rate = round(common_channel_sampling_rate)

        # If there remain any differences in samping rates, we need to keep
        # only those within a reasonable difference to the definitive rate.
        keep_st = Stream()
        if any(cr != def_channel_sample_rate for cr in channel_rates):
            for tr in matching_st:
                sample_rate_diff =\
                    tr.stats.sampling_rate - def_channel_sample_rate
                if abs(sample_rate_diff) > max_sample_rate_diff:
                    Logger.warning(
                        'There are traces with differing sampling rates for '
                        '%s on %s. There is no response that defines the '
                        'correct sampling rate for this channel, so I am '
                        'keeping only those traces with the most common '
                        'sampling rate for this channel (%s Hz).', tr.id,
                        str(tr.stats.starttime)[0:19], def_channel_sample_rate)
                else:
                    keep_st += tr
        else:
            keep_st = matching_st

        # And then we need to correct the offending trace's sampling rates
        # through interpolation.
        raw_datatype = self[0].data.dtype
        channel_rates = [tr.stats.sampling_rate for tr in keep_st]
        if any(cr != def_channel_sample_rate for cr in channel_rates):
            for tr in keep_st:
                sample_rate_diff = (
                    tr.stats.sampling_rate - def_channel_sample_rate)
                if 0 < abs(sample_rate_diff) <= max_sample_rate_diff:
                    # If difference is very small, then only adjust sample-rate
                    # description.
                    if abs(sample_rate_diff) < skip_interp_sample_rate_smaller:
                        Logger.info(
                            'Correcting sampling rate (%s Hz) of %s on %s by '
                            'adjusting trace stats.', tr.stats.sampling_rate,
                            tr.id, str(tr.stats.starttime))
                        tr.stats.sampling_rate = def_channel_sample_rate
                    else:
                        Logger.info(
                            'Correcting sampling rate (%s Hz) of %s on %s with'
                            ' interpolation.', tr.stats.sampling_rate, tr.id,
                            str(tr.stats.starttime))
                        # TODO: can this be done quicker with resample?
                        # tr.interpolate(sampling_rate=def_channel_sample_rate,
                        #               method=interpolation_method, a=25)
                        tr.resample(
                            sampling_rate=def_channel_sample_rate,
                            window='hann', no_filter=True, strict_length=False)
                        # Make sure data have the same datatype as before
                        if tr.data.dtype is not raw_datatype:
                            tr.data = tr.data.astype(raw_datatype)
        new_st += keep_st
    return new_st, True


def check_normalize_station_sample_rates(
        self, exclude_component_codes=['H'], **kwargs):
    """
    Function to compare sampling rates across channels for same station, and
    normalize trace's sampling rate to the most common one for all traces.

    :type self: obspy.core.Stream
    :param self: Stream that will be checked and corrected.
    :type exclude_component_codes: list
    :param exclude_component_codes:
        Specify component codes for which the comparison for different
        sampling rates at one station should be skipped (e.g., hydrophone vs.
        3-comp seismometer).

    returns:
        stream with corrected traces, True/False if any traces were corrected
        or not.
    :rtype: tuple of (:class:`obspy.core.event.Catalog`, boolean)
        # Check that sampling rates are the same for all channels on one station
    """
    resampled_st = Stream()
    uniq_stations = list(set([tr.stats.station_code for tr in self]))
    for station in uniq_stations:
        match_st = Stream([
            tr for tr in self if tr.station.code == station
            and tr.stats.channel[-1] not in exclude_component_codes])
        channel_rates = [tr.stats.sampling_rate for tr in match_st]
        if len(channel_rates) > 1:
            c = Counter(channel_rates)
            common_channel_sampling_rate = c.most_common(1)[0][0]
            Logger.info(
                'For station %s, there are traces with %s different '
                'sampling rates: %s; normalizing to %s Hz.',
                station, len(channel_rates), str(set(channel_rates)),
                common_channel_sampling_rate)
            for tr in match_st:
                tr.resample(
                    sampling_rate=common_channel_sampling_rate,
                    window='hann', no_filter=True, strict_length=False)
        resampled_st += match_st
        # Now add also the traces that were excluded from comparison due to
        # specific component codes (e.g., hydrophone)
        excl_match_st = Stream([
            tr for tr in self if tr.stats.station == station
            and tr.stats.channel[-1] in exclude_component_codes])
        resampled_st += excl_match_st
    return resampled_st


def try_remove_responses(
        self, inventory, taper_fraction=0.05, pre_filt=None, water_level=10,
        parallel=False, cores=None, thread_parallel=False, n_threads=1,
        output='DISP', gain_traces=False, **kwargs):
    """
    Wrapper function to try to remove response from all traces in parallel,
    taking care of a few common issues with response defintions (e.g.,
    incorrect location code).

    :param self: stream containing traces for response-removal
    :type self: :class:`obspy.core.stream.Stream`
    :param inventory: inventory containing all available responses
    :type inventory: :class:`obspy.core.inventory.Inventory`
    :param taper_fraction: fraction of trace to be tapered, defaults to 0.05
    :type taper_fraction: float, optional
    :param pre_filt:
        list of optional pre-filtering parameters, defaults to None
    :type pre_filt: list of float, optional
    :param parallel: remove responses in parallel, defaults to False
    :type parallel: bool, optional
    :param cores: maximum number of cores, defaults to None
    :type cores: int, optional
    :param thread_parallel:
        option to use thread-parallelism when run from within a subprocess
        (better use parallel=True for proper process-parallelism),
        defaults to False
    :type thread_parallel: bool, optional
    :param n_threads: maximum number of threads, defaults to 1
    :type n_threads: int, optional
    :param output:
        physical output magnitude after response removal (displacement,
        velocity or acceleration), defaults to 'DISP'
    :type output: 'DISP', 'VEL', or 'ACC', optional
    :param gain_traces:
        whether to multiply traces with a gain-factor so that corrected trace-
        values fit into float16 / float32 variables (else: float16-error),
        defaults to True
    :type gain_traces: bool, optional

    :return:
        stream with response of each trace removed (if no response found, trace is
        kept, potentially with dummy filter)
    :rtype: :class:`obspy.core.stream.Stream`
    """
    if len(self) == 0:
        Logger.warning('Stream is empty')
        return self

    # remove response
    if not parallel and not thread_parallel:
        with threadpool_limits(limits=n_threads, user_api='blas'):
            for tr in self:
                tr = tr.try_remove_response(
                    inventory, taper_fraction=taper_fraction,
                    pre_filt=pre_filt, output=output, gain_traces=gain_traces,
                    water_level=water_level)
    elif thread_parallel and not parallel:
        with threadpool_limits(limits=n_threads, user_api='blas'):
            with parallel_backend('threading', n_jobs=n_threads):
                streams = Parallel(n_jobs=cores, prefer='threads')(
                    delayed(tr.try_remove_response)
                    (inventory.select(station=tr.stats.station),
                    taper_fraction, pre_filt, output, gain_traces=gain_traces,
                    water_level=water_level)
                    for tr in self)
        # st = Stream([tr for trace_st in streams for tr in trace_st])
        self = Stream([tr for tr in streams])
    else:
        if cores is None:
            cores = min(len(self), cpu_count())
        with threadpool_limits(limits=n_threads, user_api='blas'):
            streams = Parallel(n_jobs=cores)(
                delayed(tr.try_remove_response)
                (inventory.select(station=tr.stats.station),
                 taper_fraction, pre_filt, output, gain_traces=gain_traces,
                 water_level=water_level)
                for tr in self)
        # st = Stream([tr for trace_st in streams for tr in trace_st])
        self = Stream([tr for tr in streams])
    return self


def normalize_nslc_codes(self, inv, std_network_code="NS",
                         std_location_code="00", std_channel_prefix="BH",
                         parallel=False, cores=None,
                         sta_translation_file="station_code_translation.txt",
                         forbidden_chan_file="", rotate=True,
                         thread_parallel=False, n_threads=1, **kwargs):
    """
    Normalize Network/station/location/channel codes to a standard so that
    data from different time periods can be directly compared / correlated.

    That usually requires the following steps:
        1. Correct non-FDSN-standard-complicant channel codes
        2. Rotate to proper ZNE, and hence change codes from [Z12] to [ZNE]
        3. Translate old station code
        4. Set all network codes to NS.
        5. Set all location codes to 00.

    :param self: _description_
    :type self: :class:`obspy.core.stream.Stream`
    :param inv: _description_
    :type inv: :class:`obspy.core.inventory.Inventory`
    :param std_network_code: default network code, defaults to "NS"
    :type std_network_code: str, optional
    :param std_location_code: default location code, defaults to "00"
    :type std_location_code: str, optional
    :param std_channel_prefix:
        default channel prefix (band + instrument code), defaults to "BH"
    :type std_channel_prefix: str, optional
    :param parallel: run function in parallel, defaults to False
    :type parallel: bool, optional
    :param cores: maximum number of cores to use, defaults to None
    :type cores: int, optional
    :param sta_translation_file:
        file which contains 2 columns with station names - 1st col: default
        station name, 2nd col: any alternative station names for the same
        station, defaults to "station_code_translation.txt"
    :type sta_translation_file: str, optional
    :param forbidden_chan_file:
        file that contains a list of explicitly disallowed channel names,
        defaults to ""
    :type forbidden_chan_file: str, optional
    :param rotate:
        Whether to rotate all 3-component channel sets to Z-N-E,
        defaults to True
    :type rotate: bool, optional
    :param thread_parallel:
        option to use thread-parallelism when run from within a subprocess
        (better use parallel=True for proper process-parallelism),
        defaults to False
    :type thread_parallel: bool, optional
    :param n_threads: maximum number of threads, defaults to 1
    :type n_threads: int, optional

    :return: stream with normalized traces
    :rtype: :class:`obspy.core.stream.Stream`
    """
    # 0. remove forbidden channels that cause particular problems
    if forbidden_chan_file != "":
        forbidden_chans = load_events_for_detection.load_forbidden_chan_file(
            file=forbidden_chan_file)
        st_copy = self.copy()
        for tr in st_copy:
            if wcmatch.fnmatch.fnmatch(tr.id, forbidden_chans):
                self.remove(tr)
                Logger.info('Removed trace %s because it is a forbidden trace',
                            tr.id)

    original_trace_ids = [tr.id for tr in self]
    trace_id_change_dict_1 = dict()
    # 1.
    for tr in self:
        old_tr_id = tr.id
        # Check the channel names and correct if required
        if len(tr.stats.channel) <= 2 and len(tr.stats.location) == 1:
            tr.stats.channel = tr.stats.channel + tr.stats.location
        chn = tr.stats.channel
        if std_channel_prefix is not None:
            if len(chn) < 3:
                target_channel = std_channel_prefix + chn[-1]
            elif chn[1] in 'LH10V ':
                target_channel = std_channel_prefix + chn[2]
            # Check if there are traces for the target name that would be
            # incompatible
            existing_sta_chans = self.select(
                station=tr.stats.station, channel=target_channel)
            existing_sta_chans = Stream(
                [etr for etr in existing_sta_chans
                 if etr.stats.sampling_rate != tr.stats.sampling_rate])
            try:  # Remove the trace itself from comparison
                existing_sta_chans.remove(tr)
            except ValueError:  # When trace itself not in existing stream
                pass
            if len(existing_sta_chans) == 0:
                tr.stats.channel = target_channel
            else:
                Logger.info(
                    'Cannot rename channel of trace %s to %s because there is '
                    'already a trace with id %s', tr.id, target_channel,
                    existing_sta_chans[0])
        trace_id_change_dict_1[old_tr_id] = tr.id
    if rotate:
        if not inv:
            Logger.error(
                'No inventory information available, cannot rotate channels')
        else:
            if parallel:
                self = self.parallel_rotate(
                    inv, parallel=parallel, cores=cores,
                    thread_parallel=False, n_threads=1, method="->ZNE")
            elif thread_parallel:
                self = self.parallel_rotate(
                    inv, parallel=parallel, cores=cores,
                    thread_parallel=True, n_threads=n_threads, method="->ZNE")
            else:
                # self.rotate(method="->ZNE", inventory=inv)
                # Use parallel-function to initiate error-catching rotation
                self = self.parallel_rotate(
                    inv, parallel=parallel, cores=1,
                    thread_parallel=False, n_threads=1, method="->ZNE")

    # Need to merge again here, because rotate may split merged traces if there
    # are masked arrays (i.e., values filled with None). The merge here will
    # recreate the masked arrays (after they disappeared during rotate).
    self = self.merge(method=1, fill_value=0, interpolation_samples=-1)

    # 3. +4 +5 Translate station codes, Set network and location codes
    # load list of tuples for station-code translation
    sta_fortransl_dict, sta_backtrans_dict = (
        load_events_for_detection.load_station_translation_dict(
            file=sta_translation_file))
    intermed_trace_ids = [tr.id for tr in self]
    updated_trace_ids = []
    for tr in self:
        if std_network_code is not None:
            tr.stats.network = std_network_code
        if std_location_code is not None:
            tr.stats.location = std_location_code
        if tr.stats.station in sta_fortransl_dict:
            # Make sure not to normalize station/channel-code to a combination
            # that already exists in stream
            existing_sta_chans = self.select(station=sta_fortransl_dict.get(
                tr.stats.station), channel=tr.stats.channel)
            if len(existing_sta_chans) == 0:
                tr.stats.station = sta_fortransl_dict.get(tr.stats.station)
        updated_trace_ids.append(tr.id)

    # here I need to figure out in which order the original traces are now with
    # updated IDs in the stream.
    # original_trace_ids <--> intermed_trace_ids: not in same order
    # intermed_trace_ids <--> updated_trace_ids: are in same order
    # Reverse dictionary of changes from the first round of id-changes
    inv_trace_id_change_dict_1 = {
        v: k for k, v in trace_id_change_dict_1.items()}
    original_trace_ids_new_order = []
    for tr_id in intermed_trace_ids:
        old_tr_id = inv_trace_id_change_dict_1[tr_id]
        original_trace_ids_new_order.append(old_tr_id)
    # Create the dict that contains the original IDs and the corresponding
    # updated IDs.
    trace_id_change_dict = dict()
    for tr_id_old, tr_id_new in zip(original_trace_ids_new_order,
                                    updated_trace_ids):
        trace_id_change_dict[tr_id_old] = tr_id_new

    return self, trace_id_change_dict


def print_error_plots(self, path='ErrorPlots', time_str=''):
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
    mid_time = self[0].stats.starttime + (
        self[0].stats.starttime - self[0].stats.endtime) / 2
    current_day_str = str(mid_time)[0:10]
    try:
        if not os.path.isdir(path):
            os.mkdir(path)
        for trace in self:
            png_name = time_str + '_' + trace.stats.station +\
                '_' + trace.stats.channel + '.png'
            out_plot_file = os.path.join(path, png_name)
            trace.plot(type='dayplot', size=(1900, 1080),
                       outfile=out_plot_file, data_unit='nm')
    except Exception as e:
        Logger.error('Got an exception when trying to plot Error figures'
                     'for %s', current_day_str)
        Logger.error(e)


def automatic_gain_control(self, agc_window_sec, agc_method='gismo',
                           method_exec='new', **kwargs):
    """
    Apply automatic gain correction (AGC) to traces in an ObsPy Stream object.
    This function is copied and modified (sped up and monkey-pathced) from
    package  https://github.com/uafgeotools/rtm which is originally published
    under MIT license:

    MIT License

    Copyright (c) 2019-2022 The University of Alaska Fairbanks

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

    Args:
        st (:class:`~obspy.core.stream.Stream`): Stream containing waveforms to
            be processed
        agc_window_sec (int or float): AGC window [s]. A shorter time window results
            in a more aggressive AGC effect (i.e., increased gain for quieter
            signals)
        agc_method (str): One of `'gismo'` or `'walker'` (default: `'gismo'`)

            * `'gismo'` A Python implementation of ``agc.m`` from the GISMO
              suite:

              https://github.com/geoscience-community-codes/GISMO/blob/master/core/%40correlation/agc.m

              It preserves the relative amplitudes of traces (i.e. doesn't
              normalize) but is limited in how much in can boost quiet sections
              of waveform.

            * `'walker'` An implementation of the AGC algorithm described in
              Walker *et al.* (2010), paragraph 22:

              https://doi.org/10.1029/2010JB007863

              (The code is adopted from Richard Sanderson's version.) This
              method scales the amplitudes of the resulting traces between
              :math:`[-1, 1]` (or :math:`[0, 1]` for envelopes) so inter-trace
              amplitudes are not preserved. However, the method produces a
              stronger AGC effect which may be desirable depending upon the
              context.

    Returns:
        :class:`~obspy.core.stream.Stream`: Copy of input Stream with AGC
        applied
    """
    for tr in self:
        tr.agc(agc_window_sec=agc_window_sec, agc_method=agc_method,
               method_exec=method_exec)
    return self


# TODO: maybe monkey patch these functions onto Stream ?
# def extract_array_stream(st, seisarray_prefixes=SEISARRAY_PREFIXES):



def prepare_for_detection(  # old: prepare_detection_stream(
        self, tribe, parallel=False, cores=None, ispaq=pd.DataFrame(),
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
    :return:
        the day's streams with traces excluded or renamed according to 
        the input criteria.
    """
    tr_rates = [tr.stats.sampling_rate for templ in tribe for tr in templ.st]
    if len(tr_rates) == 0:
        return self
    min_samp_rate = min(list(set(tr_rates)))
    # REMOVE UNUSABLE CHANNELS
    st_of_tr_to_be_removed = Stream()
    for tr in self:
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
        if tr in self:
            self.remove(tr)

    # Do resampling if requested (but only on channels with higher rate):
    # if downsampled_max_rate is not None:
    #    for tr in st:
    #        if tr.stats.sampling_rate > downsampled_max_rate:
    #            tr.resample(sampling_rate=downsampled_max_rate)

    # Detrend and merge
    # st = self.parallel_detrend(parallel=True, cores=None, type='simple')
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
        self.smart_despike(ispaq=ispaq)
    return self


def smart_despike(self, ispaq=pd.DataFrame()):
    """
    Try to despike the traces in the stream smartly and efficiently.

    :type self: :class:`obspy.core.stream.Stream`
    :param self: the day's streams
    :type ispaq: pandas.DataFrame()
    :param ispaq:
        Dataframe of ispaq/Mustang-generated data quality metrics,
        including spike-metrics

    :rtype: :class:`obspy.core.stream.Stream`
    :returns: stream with despiked traces
    """
    import eqcorrscan.utils.despike as despike

    # st_dspike = st.copy()
    for tr in self:
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
            tr = despike.median_filter(tr, multiplier=10, windowlength=0.5,
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
    return self


def parallel_detrend(self, parallel=True, cores=None, type='simple'):
    """
    Detrend traces in parallel.

    :type self: :class:`~obspy.core.stream.Stream`
    :param self: Stream to detrend.
    :type parallel: bool
    :param parallel: If True, detrend in parallel.
    :type cores: int
    :param cores: Number of cores to use for parallel processing.
    :type type: str
    :param type:
        Type of detrending to perform, see :meth:`obspy.core.trace.detrend`

    :rtype: :class:`~obspy.core.stream.Stream`
    :return: Detrended stream.
    """
    # Detrend in parallel across traces
    if not parallel:
        self.detrend(type=type)
    else:
        if cores is None:
            cores = min(len(self), cpu_count())
        traces = Parallel(n_jobs=cores)(delayed(tr.detrend)(type)
                                        for tr in self)
        self = Stream(traces=traces)
    return self


def parallel_merge(self, method=0, fill_value=0, interpolation_samples=0,
                   cores=1):
    """Merge traces in parallel.

    :type st: :class:`~obspy.core.stream.Stream`
    :param st: Stream to merge.
    :type method: int
    :param method:
        Method to use for merging, see :meth:`obspy.core.stream.merge`
    :type fill_value: int
    :param fill_value:
        Value to use for filling gaps, see :meth:`obspy.core.merge`
    :type interpolation_samples: int
    :param interpolation_samples: Number of samples to use for interpolation,
                                  see :meth:`obspy.core.merge`
    :type cores: int
    :param cores: Number of cores to use for parallel processing.
    :rtype: :class:`~obspy.core.stream.Stream`
    :return: Merged stream.
    """
    seed_id_list = [tr.id for tr in st]
    unique_seed_id_list = list(dict.fromkeys(seed_id_list))

    stream_list = [self.select(id=seed_id) for seed_id in unique_seed_id_list]
    streams = Parallel(n_jobs=cores)(
        delayed(trace_st.merge)(method, fill_value, interpolation_samples)
        for trace_st in stream_list)
    st = Stream()
    for trace_st in streams:
        for tr in trace_st:
            st.append(tr)
    self = st
    return self



def robust_rotate(self, inventory, method="->ZNE"):
    """
    Rotation with error catching for parallel execution.

    :type self: :class:`~obspy.core.stream.Stream`
    :param self: Stream to rotate.
    :type inventory: :class:`~obspy.core.inventory.inventory.Inventory`
    :param inventory: Inventory to use for rotation.
    :type method: str
    :param method:
        Rotation method. See :meth:`~obspy.core.stream.Stream.rotate`
    """
    if len(self) > 0:
        return self
    try:
        self = self.rotate(method, inventory=inventory)
    except Exception as e:
        try:
            st_id = ' '
            st_time = ' '
            if len(self) > 0:
                st_id = self[0].id
                st_time = str(self[0].stats.starttime)[0:19]
            Logger.warning('Cannot rotate traces for station %s on %s: %s',
                           st_id, st_time, e)
        except IndexError as e2:
            Logger.warning('Cannot rotate traces: %s --- %s', e, e2)
    return self


def parallel_rotate(self, inv, parallel=True, cores=None,
                    thread_parallel=False, n_threads=1, method="->ZNE"):
    """
    Wrapper function to rotate 3-component seismograms in a stream in parallel.

    :type self: :class:`~obspy.core.stream.Stream`
    :param self: Stream containing 3-component seismograms to be rotated.
    :type inv: :class:`~obspy.core.inventory.inventory.Inventory`
    :param inv: Inventory containing station metadata.
    :type parallel: bool
    :param parallel: If True, rotate in parallel across 3-component trace sets.
    :type cores: int
    :param cores: Number of cores to use for parallel processing.
    :type thread_parallel: bool
    :param thread_parallel: If True, rotate in parallel across 3-component
    """
    net_sta_loc = [(tr.stats.network, tr.stats.station, tr.stats.location)
                   for tr in self]
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
                robust_rotate)(self.select(
                    network=nsl[0], station=nsl[1], location=nsl[2]),
                               inventory=inv.select(
                                   network=nsl[0], station=nsl[1],
                                   location=nsl[2]),
                               method=method)
                for nsl in unique_net_sta_loc_list)
    else:
        streams = Parallel(n_jobs=cores)(delayed(
            self.select(
                network=nsl[0], station=nsl[1], location=nsl[2]
                ).robust_rotate)(inventory=inv.select(
                    network=nsl[0], station=nsl[1], location=nsl[2]),
                                 method=method)
                for nsl in unique_net_sta_loc_list)
    self = Stream([tr for trace_st in streams for tr in trace_st])

    # st.merge(method=0, fill_value=0, interpolation_samples=0)
    return self


def daily_plot(self, year, month, day, data_unit='counts', suffix=''):
    """
    Make a daily plot of each trace in the stream and save it to a file.

    :type self: :class:`~obspy.core.stream.Stream`
    :param self: Stream to plot.
    :type year: int
    :param year: Year of the day to plot.
    :type month: int
    :param month: Month of the day to plot.
    :type day: int
    :param day: Day of the day to plot.
    :type data_unit: str
    :param data_unit: Unit of the data to plot.
    :type suffix: str
    :param suffix: Suffix to append to the output file name.

    :rtype: None
    """
    for tr in self:
        out_plot_file = os.path.join('DebugPlots', str(year)
                                   + str(month).zfill(2) + str(day).zfill(2)
                                   + '_' + tr.stats.station
                                   + '_' + tr.stats.channel + suffix + '.png')
        tr.plot(type='dayplot', size=(1900, 1080), outfile=out_plot_file,
                data_unit=data_unit)



Stream.automatic_gain_control = automatic_gain_control
Stream.agc = automatic_gain_control
Stream.init_processing = init_processing
Stream.init_processing_w_rotation = init_processing_w_rotation
Stream._init_processing_per_channel_w_rotation = (
    _init_processing_per_channel_w_rotation)
Stream.mask_consecutive_zeros = mask_consecutive_zeros
Stream.taper_trace_segments = taper_trace_segments
Stream.check_normalize_sampling_rate = check_normalize_sampling_rate
Stream.check_normalize_station_sample_rates = (
    check_normalize_station_sample_rates)
Stream.try_remove_responses = try_remove_responses
Stream.normalize_nslc_codes = normalize_nslc_codes
Stream.print_error_plots = print_error_plots
Stream.prepare_for_detection = prepare_for_detection
Stream.smart_despike = smart_despike
Stream.parallel_detrend = parallel_detrend
Stream.parallel_merge = parallel_merge
Stream.robust_rotate = robust_rotate
Stream.parallel_rotate = parallel_rotate
Stream.daily_plot = daily_plot