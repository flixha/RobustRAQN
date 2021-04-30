import os
import glob
import fnmatch
import wcmatch
from eqcorrscan.utils.plotting import detection_multiplot
import pandas as pd
#import matplotlib
from threadpoolctl import threadpool_limits

from multiprocessing import Pool, cpu_count, current_process, get_context
from multiprocessing.pool import ThreadPool

import numpy as np
from itertools import chain, repeat
from collections import Counter
# import numexpr as ne

#from obspy import read_events, read_inventory
# from obspy.core.event import Catalog
# import obspy
from obspy.core.stream import Stream
from obspy.core.inventory.inventory import Inventory
#from obspy.core.util.base import TypeError
# from obspy.core.event import Event
from obspy.io.nordic.core import read_nordic
from obspy import read as obspyread
from obspy import UTCDateTime

from eqcorrscan.utils.correlate import pool_boy
from eqcorrscan.utils.despike import median_filter
from eqcorrscan.core.match_filter import Tribe
from eqcorrscan.core.match_filter.party import Party
from eqcorrscan.core.match_filter.family import Family

# import obustraqn.spectral_tools
from robustraqn.spectral_tools import st_balance_noise, Noise_model

from timeit import default_timer
import logging
Logger = logging.getLogger(__name__)

#from eqcorrscan.utils.catalog_utils import filter_picks


def load_event_stream(event, sfile, seisanWAVpath, selectedStations,
                      min_samp_rate=np.nan):
    """
    Load the waveforms for an event file (here: Nordic file) while performing
    some checks for duplicates, incompleteness, etc.
    """
    origin = event.preferred_origin()
    # Stop event processing if there are no waveform files
    select, wavname = read_nordic(sfile, return_wavnames=True)
    wavfilenames = wavname[0]
    if not wavfilenames:
        Logger.warning('Event ' + sfile + ': no waveform files found')
        return None
    st = Stream()
    for wavefile in wavfilenames:
        # Check that there are proper mseed/SEISAN7.0 waveform files
        if wavefile[0:3] == 'ARC' or wavefile[0:4] == 'WAVE':
            continue
        fullWaveFile = os.path.join(seisanWAVpath, wavefile)
        try:
            st += obspyread(fullWaveFile)
        except TypeError:
            Logger.error('Type Error: Could not read waveform file ' +
                         fullWaveFile)
        except FileNotFoundError:
            Logger.error('FileNotFoundError: Could not read waveform file '
                  + fullWaveFile)
        except ValueError:
            Logger.error('ValueError: Could not read waveform file' +
                         fullWaveFile)
        except AssertionError:
            Logger.error('AsertionError: Could not read waveform file ' +
                         fullWaveFile)
            
    # REMOVE UNUSABLE CHANNELS
    st_copy = st.copy()
    for tr in st_copy:
        # channels without signal
        # if sum(tr.copy().detrend().data)==0 and tr in st:
        n_samples_nonzero = np.count_nonzero(tr.copy().detrend().data)
        if n_samples_nonzero==0 and tr in st:
            st = st.remove(tr)
            continue
        # channels with empty channel code:
        if len(tr.stats.channel) == 0 and tr in st:
            st = st.remove(tr)
            continue
        # channels with undecipherable channel names
        # potential names: MIC, SLZ, S Z, M, ALE, MSF, TC
        if tr.stats.channel[0] not in 'HBSENM' and tr in st:
            st = st.remove(tr)
            continue
        if tr.stats.channel[-1] not in 'ZNE0123ABC' and tr in st:
            st = st.remove(tr)
            continue
        # channels from accelerometers/ gravimeters/ tiltmeter/low-gain seism.
        if len(tr.stats.channel) == 3 and tr in st:
            if tr.stats.channel[1] in 'NGAL':
                st = st.remove(tr)
                continue
        # channels whose sampling rate is lower than the one chosen for the
        # templates
        if tr.stats.sampling_rate < min_samp_rate:
            st = st.remove(tr)
            continue

        #print(tr.id)
        # ADJUST unsupported, but decipherable codes
        # Adjust single-letter location codes:
        if len(tr.stats.location) == 1:
            tr.stats.location = tr.stats.location + '0'
        # Adjust empty-letter channel codes: ## DO NOT ADJUST YET - IT MAY 
        # WORK LIKE THIS WITH PROPER INVENTORY
        #if tr.stats.channel[1] == ' ' and len(tr.stats.channel) == 3:
        #    tr.stats.channel = tr.stats.channel[0] + 'H' + tr.stats.channel[2]
            
    #Add channels to template, but check if similar components are already present
    channel_priorities = ["H*","B*","S*","E*","N*","*"]
    waveAtSelStations = Stream()
    for station in selectedStations:
        for channel_priority in channel_priorities:
            waveAlreadyAtSelStation = waveAtSelStations.select(station=station)
            if not waveAlreadyAtSelStation:
                addWaves = st.select(station=station,
                                        channel=channel_priority)
                waveAtSelStations += addWaves
    # If there are more than one traces for the same station-component-
    # combination, then choose the "best" trace
    waveAtSelStations_copy = waveAtSelStations.copy()
    for tr in waveAtSelStations_copy:
        sameStaChanSt = waveAtSelStations.select(
            station=tr.stats.station, channel='*'+tr.stats.channel[-1])
        removeTrSt = Stream()
        keepTrSt = Stream()
        nSameStaChanW = len(sameStaChanSt)
        if nSameStaChanW > 1:
            # 1. best trace: highest sample rate
            samp_rates = [t.stats.sampling_rate for t in sameStaChanSt]
            keepTrSt = sameStaChanSt.select(sampling_rate=max(samp_rates))
            # 2. best trace: longest trace
            trace_lengths = [t.stats.npts for t in sameStaChanSt]
            keepTrSt = sameStaChanSt.select(npts=max(trace_lengths))
            # 3. best trace: more complete metadata - location code
            if len(keepTrSt) == 0 or len(keepTrSt) > 1:
                loccodes = [t.stats.location for t in sameStaChanSt]
                if any(l=='' for l in loccodes) and\
                   any(l=='??' for l in loccodes):
                    removeTrSt += sameStaChanSt.select(location="")
                    keepTrSt += sameStaChanSt.select(
                        sampling_rate= max(samp_rates), location="?*")
            # 4 best trace: more complete metadata - network code
            if len(keepTrSt) == 0 or len(keepTrSt) > 1:
                netcodes = [t.stats.network for t in sameStaChanSt]
                if (any(n=='' for n in netcodes)\
                   and any(n=='??' for n in netcodes)):
                    removeTrSt += sameStaChanSt.select(network="")
                    keepTrSt += sameStaChanSt.select(sampling_rate=
                        max(samp_rates), location="?*", network="??")
                if len(keepTrSt) > 1:
                    keepTrSt = Stream() + keepTrSt[0]
            for tt in sameStaChanSt:
                waveAtSelStations.remove(tt)
            waveAtSelStations += keepTrSt

    # Double-check to remove duplicate channels
    # waveAtSelStations.merge(method=0, fill_value=0, interpolation_samples=0)
    # 2021-01-22: changed merge method to below one to fix error with
    #             incomplete day.
    waveAtSelStations.merge(method=1, fill_value=0, interpolation_samples=-1)
    k = 0
    channelIDs = list()
    for trace in waveAtSelStations:
        if trace.id in channelIDs:
            for j in range(0,k):
                testSameIDtrace = waveAtSelStations[j]
                if trace.id == testSameIDtrace.id:
                    if trace.stats.starttime >= testSameIDtrace.stats.starttime:
                        waveAtSelStations.remove(trace)
                    else:
                        waveAtSelStations.remove(testSameIDtrace)
        else:
            channelIDs.append(trace.id)
            k += 1
    st = waveAtSelStations
    # Preprocessing
    # cut around origin plus some
    
    # Don't trim the stream if that means you are padding with zeros
    starttime = origin.time - 30
    endtime = starttime + 360
    st.trim(starttime=starttime, endtime=endtime, pad=False,
            nearest_sample=True)
    
    #don't use the waveform if more than 5% is zero
    nonZeroWave = Stream()
    for tr in st:
        n_nonzero = np.count_nonzero(tr.copy().detrend().data)
        # if (sum(tr.copy().detrend().data==0) < tr.data.size*0.05 and not\
        if (n_nonzero > tr.data.size * 0.95 and not any(np.isnan(tr.data))):
            nonZeroWave.append(tr)
    st = nonZeroWave
    return st


# def prepare_detection_stream(st, parallel=False, cores=None,
#                              ispaq=pd.DataFrame(), try_despike=False,
#                              downsampled_max_rate=None):
    

def prepare_detection_stream(st, tribe, parallel=False, cores=None,
                             ispaq=pd.DataFrame(), try_despike=False,
                             downsampled_max_rate=None):
    """
    """
    min_samp_rate = min(list(set([tr.stats.sampling_rate
                                  for templ in tribe for tr in templ.st])))
    # REMOVE UNUSABLE CHANNELS
    # st_copy = st.copy()
    st_of_tr_to_be_removed = Stream()
    # for tr in st_copy:
    for tr in st:
        # channels with empty channel code:
        if len(tr.stats.channel) == 0:
            #st = st.remove(tr)
            st_of_tr_to_be_removed += tr
            continue
        # channels with undecipherable channel names
        # potential names: MIC, SLZ, S Z, M, ALE, MSF, TC
        if tr.stats.channel[0] not in 'HBSENM':
            # st = st.remove(tr)
            st_of_tr_to_be_removed += tr
            continue
        if tr.stats.channel[-1] not in 'ZNE0123ABC':
            # st = st.remove(tr)
            st_of_tr_to_be_removed += tr
            continue
        # channels from accelerometers/ gravimeters/ tiltmeter/low-gain seism.
        if len(tr.stats.channel) == 3:
            if tr.stats.channel[1] in 'NGAL':
                # st = st.remove(tr)
                st_of_tr_to_be_removed += tr
                continue
        if tr.stats.sampling_rate < min_samp_rate:
            st_of_tr_to_be_removed += tr
        #print(tr.id)
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
    # st.merge(method=0, fill_value=None, interpolation_samples=0)
    # If using Zero as a fill_value, then EQcorrscan will not be able to
    # automatically recognize these times and exclude the correlation-
    # value on those traces from the stacked value.
    #st.merge(method=0, fill_value=0, interpolation_samples=0)

    # TODO write despiking smartly
    if try_despike:
        #st_despike = st.copy()
        for tr in st:
            # starttime = tr.stats.starttime
            # endtime = tr.stats.endtime
            # mid_t = starttime + (endtime - starttime)/2
            # # The reqtimes are those that are listed in the ispaq-stats per day
            # # (starts on 00:00:00 and ends on 23:59:59).
            # reqtime1 = UTCDateTime(mid_t.year, mid_t.month, mid_t.day, 0, 0, 0)
            # reqtime2 = UTCDateTime(mid_t.year, mid_t.month, mid_t.day,23,59,59)
    
            # day_stats = stats[stats['start'].str.contains(str(reqtime1)[0:19])]
            nslc_target = tr.stats.network + "." + tr.stats.station + "."\
                + tr.stats.location + '.' + tr.stats.channel
            chn_stats = ispaq[ispaq['target'].str.contains(nslc_target)]
            #target = availability.iloc[0]['target']
            # num_spikes = ispaq[(ispaq['target']==target) & 
            # (day_stats['metricName']=='num_spikes')]
            num_spikes = chn_stats[chn_stats['metricName']=='num_spikes']
            if len(num_spikes) > 0:
                if num_spikes.iloc[0]['value'] == 0:
                    continue
            Logger.warning('%s: %s spikes, attempting to despike', str(tr),
                           num_spikes.iloc[0]['value'])
            try:
                tr = median_filter(tr, multiplier=10, windowlength=0.5,
                                   interp_len=0.1)
                Logger.warning('Successfully despiked, %s', str(tr))
                #testStream = Stream()
                #testStream += testtrace
                #_spike_test(testStream, percent=0.98, multiplier=5e4)
            except Exception as e:
                Logger.warning('Failed to despike %s: %s', str(tr))
                Logger.warning(e)
        
                # tracelength = testtrace.stats.npts / testtrace.stats.sampling_rate
                #         print('Despiking ' + contFile )
                # testtrace = median_filter(testtrace, multiplier=10,
                #         windowlength=tracelength, interp_len=0.1, debug=0)
                #test for step functions which mess up a lot
            # try:
            #     diftrace = tr.copy()
            #     diftrace.differentiate()
            #     difStream = Stream()
            #     difStream += diftrace
            #     _spike_test(difStream, percent=0.9999, multiplier=1e1)
            # except MatchFilterError:
            #     tracelength = diftrace.stats.npts / diftrace.stats.sampling_rate
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
        with pool_boy(Pool=Pool, traces=len(st), cores=cores) as pool:
            results = [pool.apply_async(tr.detrend, {type})
                       for tr in st]
        traces = [res.get() for res in results]
        st = Stream(traces=traces)
        
    return st


def parallel_merge(st, method=0, fill_value=None, interpolation_samples=0,
                   cores=1):
    seed_id_list = [tr.id for tr in st]
    unique_seed_id_list = set(seed_id_list)
    
    stream_list = [st.select(id=seed_id) for seed_id in unique_seed_id_list]
    
    with pool_boy(
        Pool=Pool, traces=len(unique_seed_id_list), cores=cores) as pool:
        results = [pool.apply_async(
            trace_st.merge, {method, fill_value, interpolation_samples})
                    for trace_st in stream_list]
            
    streams = [res.get() for res in results]
    for trace_st in streams:
        for tr in trace_st:
            st.append(tr)

    #st.merge(method=0, fill_value=None, interpolation_samples=0)
    return st



def parallel_rotate(st, inv, cores=None, method="->ZNE"):
    """
    wrapper function to rotate 3-component seismograms in a stream in parallel.
    """
    net_sta_loc = [(tr.stats.network, tr.stats.station, tr.stats.location)
                   for tr in st]
    # Sort unique-ID list by most common, so that 3-component stations
    # appear first and are processed first in parallel loop (for better load-
    # balancing).
    net_sta_loc = list(chain.from_iterable(repeat(item, count)
                       for item,count in Counter(net_sta_loc).most_common()))
    # Need to sort list by original order after set()
    unique_net_sta_loc_list = sorted(set(net_sta_loc),
                                     key=lambda x: net_sta_loc.index(x))
    
    # stream_list = [st.select(id=seed_id) for seed_id in unique_seed_id_list]
    
    if cores is None:
        cores = min(len(unique_net_sta_loc_list), cpu_count())
    with pool_boy(
        Pool=Pool, traces=len(unique_net_sta_loc_list), cores=cores) as pool:
        results = [pool.apply_async(
            st.select(network=nsl[0], station=nsl[1], location=nsl[2]).rotate,
            args=(method,),
            kwds=dict(inventory=inv.select(network=nsl[0], station=nsl[1],
                                           location=nsl[2])))
                   for nsl in unique_net_sta_loc_list]
            
    streams = [res.get() for res in results]
    for trace_st in streams:
        for tr in trace_st:
            st.append(tr)
       
    #st.merge(method=0, fill_value=None, interpolation_samples=0)
    return st



def daily_plot(st, year, month, day, data_unit='counts', suffix=''):
    """
    """
    for tr in st:
        outPlotFile = os.path.join('DebugPlots', str(year)
                                   + str(month).zfill(2) + str(day).zfill(2)
                                   + '_' + tr.stats.station
                                   + '_' + tr.stats.channel + suffix + '.png')
        tr.plot(type='dayplot', size=(1900, 1080), outfile = outPlotFile,
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
    #for k, code in enumerate(avail_tr_ids):
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


def prepare_picks(event, stream, normalize_NSLC=True, std_network_code='NS',
                  std_location_code='00', std_channel_prefix='BH',
                  sta_translation_file="station_code_translation.txt"):
    """
    Prepare the picks for being used in EQcorrscan. The following criteria are
    being considered:
     - remove all amplitude-picks
     - remove picks without phase_hint
     - compare picks to the available waveform-traces
     - normalize network/station/channel codes
     - if the channel is not available, the pick will be switched to a suitable 
       alternative channel
     - put P-pick on Z-channel
     - put S-picks on horizontal channels if available

    :type template: obspy.core.stream.Stream
    :param template: Template stream to plot
    :type background: obspy.core.stream.stream
    :param background: Stream to plot the template within.
    :type picks: list
    :param picks: List of :class:`obspy.core.event.origin.Pick` picks.
    {plotting_kwargs}

    :returns: :class:`obspy.event`

    """
    #correct PG / SG picks to P / S
    #for pick in event.picks:
    #    if pick.phase_hint == 'PG':
    #        pick.phase_hint = 'P'
    #    elif pick.phase_hint == 'SG':
    #        pick.phase_hint = 'S'
    
    ## catalog.plot(projection='local', resolution='h')
    sta_fortransl_dict, sta_backtrans_dict = load_station_translation_dict(
        file=sta_translation_file)
   
    newEvent = event.copy()
    newEvent.picks = list()
    #remove all picks for amplitudes etc: keep only P and S
    for j, pick in enumerate(event.picks):
        # Don't allow picks without phase_hint
        if len(pick.phase_hint) == 0:
            continue
        # Check which channels are available for pick's station in stream
        avail_comps = [tr.stats.channel[-1] for tr in
                          stream.select(station=pick.waveform_id.station_code)]
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
        if pick.phase_hint.upper()[0]=='P' or pick.phase_hint.upper()[0]=='S':
            pick.waveform_id.network_code = std_network_code
            pick.waveform_id.location_code = std_location_code
            # Check station code for altnerative code, change to the standard
            if pick.waveform_id.station_code in sta_fortransl_dict:
                pick.waveform_id.station_code = sta_fortransl_dict.get(
                    pick.waveform_id.station_code)
            # Check channel codes
            # 1. If pick has no channel information, then put it a preferred
            #    channel (Z>N>E>2>1) for now; otherwise just normalize channel-
            #    prefix.
            if len(pick.waveform_id.channel_code) == 0:
                pick.waveform_id.channel_code = std_channel_prefix +\
                    avail_comps[-1]
            elif len(pick.waveform_id.channel_code) <= 2:
                pick.waveform_id.channel_code =\
                    std_channel_prefix + pick.waveform_id.channel_code[-1]
            # 2. Check that channel is available - otherwise switch to suitable
            #    other channel.
            if pick.waveform_id.channel_code[-1] not in avail_comps:
                pick.waveform_id.channel_code =\
                    std_channel_prefix + avail_comps[-1]
            # 3. If P-pick is not on vertical channel and there exists a 'Z'-
            #    channel, then switch P-pick to Z.
            if pick.phase_hint.upper()[0] == 'P' and\
                    pick.waveform_id.channel_code[-1] != 'Z':
                if 'Z' in avail_comps:
                    pick.waveform_id.channel_code = std_channel_prefix +'Z'
            # 4. If S-pick is on vertical channel and there exist horizontal
            #    channels, then switch S_pick to the first horizontal.         
            elif pick.phase_hint.upper()[0] == 'S' and\
                pick.waveform_id.channel_code[-1] == 'Z':
                horizontalTraces = stream.select(station=pick.waveform_id.
                        station_code, channel=std_channel_prefix+'[EN123]')
                if horizontalTraces:
                    pick.waveform_id.channel_code =\
                        horizontalTraces[0].stats.channel
                # 4b. If S-pick is on vertical and there is no P-pick, then
                # remove S-pick.
                else:
                    P_picks = [p for p in event.picks
                               if len(p.phase_hint) > 0
                                    if p.phase_hint.upper()[0] == 'P' and
                                    p.waveform_id.station_code ==
                                    pick.waveform_id.station_code]
                    if len(P_picks) == 0:
                        continue
            newEvent.picks.append(pick)
            #else:
            #    newEvent.picks.append(pick)
    event = newEvent
    # Check for duplicate picks. Remove the later one when they are
    # on the same channel
    pickIDs = list()
    newEvent = event.copy()
    newEvent.picks = list()
    # TODO check whether Pn / Pg are correctly handled here
    for pick in event.picks:
        #uncomment to change back to retaining Pn and Pg
        #pickIDTuple = (pick.waveform_id, pick.phase_hint)
        pickIDTuple = (pick.waveform_id.station_code,
                       pick.phase_hint.upper())
        if not pickIDTuple in pickIDs:
            #pickIDs.append((pick.waveform_id, pick.phase_hint))
            pickIDs.append((pick.waveform_id.station_code,
                            pick.phase_hint.upper()))
            newEvent.picks.append(pick)
        else:
            # check which is the earlier pick; remove the old and
            # append the new one . This also takes care of only retaining the
            # earlier pick of Pg, Pn and Sg, Sn.
            for pick_old in newEvent.picks:
                if (pick_old.waveform_id.station_code
                        == pick.waveform_id.station_code
                        and pick_old.phase_hint.upper()
                        == pick.phase_hint.upper()):
                    if pick.time < pick_old.time:
                        newEvent.picks.remove(pick_old)
                        newEvent.picks.append(pick)
    event = newEvent
    return event



def init_processing(day_st, starttime, endtime, remove_response=False,
                    inv=Inventory(), min_segment_length_s=10,
                    max_sample_rate_diff=1,
                    skip_check_sampling_rates=[20, 40, 50, 66, 75, 100, 500],
                    skip_interp_sample_rate_smaller=1e-7,
                    interpolation_method='lanczos', taper_fraction=0.005,
                    detrend_type='simple', downsampled_max_rate=None,
                    noise_balancing=False, balance_power_coefficient=2,
                    parallel=False, cores=None):
    """
    Does an initial processing of the day's stream,
    """
    # If I merge the traces before removing the response, then the masked
    # arrays / None-values removal will mess up the response-corrected trace
    # and make it all None.
    # st.merge(method=0, fill_value=None, interpolation_samples=0)
    # If using Zero as a fill_value, then EQcorrscan will not be able to
    # automatically recognize these times and exclude the correlation-
    # value on those traces from the stacked value.

    Logger.info('Starting initial processing for %s - %s.',
                str(starttime)[0:19], str(endtime)[0:19])
    outtic = default_timer()
                
    seed_id_list = [tr.id for tr in day_st]
    unique_seed_id_list = sorted(set(seed_id_list))

    if not parallel:
        st = Stream()
        for id in unique_seed_id_list:
            Logger.debug('Starting initial processing of %s for %s - %s.',
                         id(), str(starttime)[0:19], str(endtime)[0:19])
            st += _init_processing_per_channel(
                day_st.select(id=id), starttime, endtime,
                remove_response=remove_response,
                inv=inv.select(station=id.split('.')[1]),
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
                balance_power_coefficient=balance_power_coefficient)

        # # Make a copy of the day-stream to find the values that need to be masked.
        # masked_st = day_st.copy()
        # masked_st.merge(method=0, fill_value=None, interpolation_samples=0)
        # masked_st.trim(starttime=starttime, endtime=endtime, pad=True,
        #             nearest_sample=True, fill_value=0)
        
        # # Merge daystream without masking
        # day_st.merge(method=0, fill_value=0, interpolation_samples=0)
        # # Correct response (taper should be outside of the main day!)
        # day_st = try_remove_responses(day_st, inv, taper_fraction=0.005,
        #                             parallel=parallel, cores=cores)
        # # Trim to full day and detrend again
        # day_st.trim(starttime=starttime, endtime=endtime, pad=True,
        #             nearest_sample=True, fill_value=0)
        # day_st = parallel_detrend(day_st, parallel=True, cores=cores,
        #                         type='simple')
        
        # # Put masked array into response-corrected stream day_st:
        # for j, tr in enumerate(day_st):
        #     if isinstance(masked_st[j].data, np.ma.MaskedArray):
        #         tr.data = np.ma.masked_array(tr.data, mask=masked_st[j].data.mask)
    else:
        if cores is None:
            cores = min(len(day_st), cpu_count())
        
        with threadpool_limits(limits=1, user_api='blas'):
            with pool_boy(Pool=get_context("spawn").Pool, traces
                          =len(unique_seed_id_list),cores=cores) as pool:
                #params = ((tr, inv, taper_fraction, parallel, cores)
                #          for tr in st)
                results =\
                    [pool.apply_async(
                        _init_processing_per_channel, 
                        ((day_st.select(id=id), starttime, endtime)),
                        dict(remove_response=remove_response,
                            inv=inv.select(station=id.split('.')[1],
                                           starttime=starttime,
                                           endtime=endtime),
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
                            balance_power_coefficient=balance_power_coefficient))
                    for id in unique_seed_id_list]
        #args = (st.select(id=id), inv.select(station=id.split('.')[1]),
        #        detrend_type=detrend_type, taper_fraction=taper_fraction)
        streams = [res.get() for res in results]
        st = Stream()
        for trace_st in streams:
            for tr in trace_st:
                st.append(tr)
                
    outtoc = default_timer()
    Logger.info(
        'Initial processing of %s traces in stream took: {0:.4f}s'.format(
            outtoc - outtic), str(len(st)))
                
    return st



def init_processing_wRotation(
    day_st, starttime, endtime, remove_response=False, inv=Inventory(),
    sta_translation_file='', parallel=False, cores=None,
    min_segment_length_s=10, max_sample_rate_diff=1,
    skip_check_sampling_rates=[20, 40, 50, 66, 75, 100, 500],
    skip_interp_sample_rate_smaller=1e-7, interpolation_method='lanczos',
    taper_fraction=0.005, detrend_type='simple', downsampled_max_rate=None,
    std_network_code="NS", std_location_code="00", std_channel_prefix="BH",
    noise_balancing=False, balance_power_coefficient=2):
    """
    Does an initial processing of the day's stream,
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
                   for tr in day_st]
    if len(net_sta_loc) == 0:
        Logger.error('There are no traces to do initial processing on %s',
                     str(starttime))
        return day_st
    # Sort unique-ID list by most common, so that 3-component stations
    # appear first and are processed first in parallel loop (for better load-
    # balancing)
    # net_sta_loc = list(chain.from_iterable(repeat(i, c)
    #                    for i,c in Counter(net_sta_loc).most_common()))
    ### Need to sort list by original order after set()

    # Better: Sort by: whether needs rotation; npts per 3-comp stream
    unique_net_sta_loc_list = set(net_sta_loc)
    three_comp_strs = [
        day_st.select(network=nsl[0], station=nsl[1], location=nsl[2])
        for nsl in unique_net_sta_loc_list]
    sum_npts = [sum([tr.stats.npts for tr in s]) for s in three_comp_strs]
    needs_rotation = [
        '1' in [tr.stats.channel[-1] for tr in s] for s in three_comp_strs]
    unique_net_sta_loc_list = [x for x,_,_ in sorted(
        zip(unique_net_sta_loc_list, needs_rotation, sum_npts),
        key=lambda y: (y[1], y[2]), reverse=True)]

    if not parallel:
        st = Stream()
        for nsl in unique_net_sta_loc_list:
            Logger.info('Starting initial processing of %s for %s - %s.',
                         str(nsl), str(starttime)[0:19], str(endtime)[0:19])
            st += _init_processing_per_channel_wRotation(
                day_st.select(network=nsl[0], station=nsl[1], location=nsl[2]),
                starttime, endtime, remove_response=remove_response,
                inv=inv.select(station=nsl[1], starttime=starttime,
                               endtime=endtime),
                min_segment_length_s=min_segment_length_s,
                max_sample_rate_diff=max_sample_rate_diff,
                skip_check_sampling_rates=skip_check_sampling_rates,
                skip_interp_sample_rate_smaller=
                skip_interp_sample_rate_smaller,
                interpolation_method=interpolation_method,
                sta_translation_file=sta_translation_file,
                std_network_code=std_network_code, std_location_code=
                std_location_code, std_channel_prefix=std_channel_prefix,
                detrend_type=detrend_type, downsampled_max_rate=
                downsampled_max_rate, taper_fraction=taper_fraction,
                noise_balancing=noise_balancing,
                balance_power_coefficient=balance_power_coefficient)

    else:
        if cores is None:
            cores = min(len(day_st), cpu_count())
        
        # Check if I can allow multithreading in each of the parallelized
        # subprocesses:
        thread_parallel = False
        n_threads = 1        
        # if cores > 2 * len(day_st):
        #     thread_parallel = True
        #     n_threads = int(cores / len(day_st))
        # Logger.info('Starting initial 3-component processing with %s parallel '
        #             + 'processes with up to %s threads each.', str(cores),
        #             str(n_threads))
        with threadpool_limits(limits=1, user_api='blas'):
            with pool_boy(Pool=get_context("spawn").Pool, traces
                          =len(unique_net_sta_loc_list), cores=cores) as pool:
                results =\
                    [pool.apply_async(
                        _init_processing_per_channel_wRotation,
                        args=(day_st.select(network=nsl[0], station=nsl[1],
                                            location=nsl[2]),
                            starttime, endtime),
                        kwds=dict(
                            remove_response=remove_response,
                            inv=inv.select(station=nsl[1],
                                           starttime=starttime,
                                           endtime=endtime),
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
                            parallel=thread_parallel, cores=n_threads))
                    for nsl in unique_net_sta_loc_list]
        st = Stream()
        if len(results) > 0:
            streams = [res.get() for res in results]    
            for trace_st in streams:
                for tr in trace_st:
                    st.append(tr)
  
    outtoc = default_timer()
    Logger.info('Initial processing of streams took: {0:.4f}s'.format(
        outtoc - outtic))
    return st



def _init_processing_per_channel(
    st, starttime, endtime, remove_response=False, inv=Inventory(),
    min_segment_length_s=10, max_sample_rate_diff=1,
    skip_check_sampling_rates=[20, 40, 50, 66, 75, 100, 500],
    skip_interp_sample_rate_smaller=1e-7, interpolation_method='lanczos',
    detrend_type='simple', taper_fraction=0.005, pre_filt=None,
    downsampled_max_rate=None, noise_balancing=False,
    balance_power_coefficient=2):
    """
    Inner loop over which the initial processing can be parallelized
    """

    # First check trace segments for strange sampling rates and segments that
    # are too short:
    st, st_normalized = check_normalize_sampling_rate(
        st, inv,
        min_segment_length_s=min_segment_length_s,
        max_sample_rate_diff=max_sample_rate_diff,
        skip_check_sampling_rates=skip_check_sampling_rates,
        skip_interp_sample_rate_smaller=skip_interp_sample_rate_smaller,
        interpolation_method=interpolation_method)
    if len(st) == 0:
        return st
    
    # Detrend
    st.detrend(type=detrend_type)
    
    # Merge, but keep "copy" of the masked array for filling back
    # Make a copy of the day-stream to find the values that need to be masked.
    masked_st = st.copy()
    masked_st.merge(method=0, fill_value=None, interpolation_samples=0)
    masked_st.trim(starttime=starttime, endtime=endtime, pad=True,
                   nearest_sample=True, fill_value=None)
    
    # Merge daystream without masking
    # st.merge(method=0, fill_value=0, interpolation_samples=0)
    # 2021-01-22: changed merge method to below one to fix error with
    #             incomplete day.
    st.merge(method=1, fill_value=0, interpolation_samples=-1)
    # Correct response (taper should be outside of the main day!)
    if remove_response:
        st = try_remove_responses(
            st, inv.select(starttime=starttime, endtime=endtime),
            taper_fraction=taper_fraction, pre_filt=pre_filt,
            parallel=False, cores=1)
    # Detrend now?
    st = st.detrend(type='simple')
    # st = st.detrend(type='linear')

    if noise_balancing:
        # Need to do some prefiltering to avoid phase-shift effects when very 
        # low frequencies are boosted
        st = st.filter('highpass', freq=0.1, zerophase=True) #.detrend()
        st = st_balance_noise(st, inv, balance_power_coefficient=
                              balance_power_coefficient)
        st = st.detrend(type='linear').taper(
            0.005, type='hann', max_length=None, side='both')
        
    # Trim to full day and detrend again
    st = st.trim(starttime=starttime, endtime=endtime, pad=True,
                 nearest_sample=True, fill_value=0)
    st = st.detrend(type=detrend_type)
    
    # Put masked array into response-corrected stream st:
    for j, tr in enumerate(st):
        if isinstance(masked_st[j].data, np.ma.MaskedArray):
            tr.data = np.ma.masked_array(tr.data, mask=masked_st[j].data.mask)

    # Downsample if necessary
    if downsampled_max_rate is not None:
        for tr in st:
            if tr.stats.sampling_rate > downsampled_max_rate:
                tr.resample(sampling_rate=downsampled_max_rate,
                            no_filter=False)

    return st



def _init_processing_per_channel_wRotation(
    st, starttime, endtime, remove_response=False, inv=Inventory(),
    sta_translation_file='', min_segment_length_s=10, max_sample_rate_diff=1,
    skip_check_sampling_rates=[20, 40, 50, 66, 75, 100, 500],
    skip_interp_sample_rate_smaller=1e-7, interpolation_method='lanczos',
    std_network_code="NS", std_location_code="00", std_channel_prefix="BH",
    detrend_type='simple', taper_fraction=0.005, downsampled_max_rate=25,
    noise_balancing=False, balance_power_coefficient=2, parallel=False,
    cores=1):
    """
    Inner loop over which the initial processing can be parallelized
    """
   
    outtic = default_timer()
    # First check trace segments for strange sampling rates and segments that
    # are too short:
    st, st_normalized = check_normalize_sampling_rate(
        st, inv, min_segment_length_s=min_segment_length_s,
        max_sample_rate_diff=max_sample_rate_diff,
        skip_check_sampling_rates=skip_check_sampling_rates,
        skip_interp_sample_rate_smaller=skip_interp_sample_rate_smaller,
        interpolation_method=interpolation_method)
    
    # Detrend
    st.detrend(type=detrend_type)
    # Merge, but keep "copy" of the masked array for filling back
    # Make a copy of the day-stream to find the values that need to be masked.
    masked_st = st.copy()
    masked_st.merge(method=0, fill_value=None, interpolation_samples=0)
    masked_st.trim(starttime=starttime, endtime=endtime, pad=True,
                   nearest_sample=True, fill_value=None)
    
    # Merge daystream without masking
    # st.merge(method=0, fill_value=0, interpolation_samples=0)
    # 2021-01-22: changed merge method to below one to fix error with
    #             incomplete day.
    st = st.merge(method=1, fill_value=0, interpolation_samples=-1)
    # Correct response (taper should be outside of the main day!)
    if remove_response:
        st = try_remove_responses(
            st, inv.select(starttime=starttime, endtime=endtime),
            taper_fraction=0.005, parallel=parallel, cores=cores)
    # Trim to full day and detrend again
    st.trim(starttime=starttime, endtime=endtime, pad=True,
            nearest_sample=True, fill_value=0)
    st.detrend(type=detrend_type)

    # normalize NSLC codes, including rotation
    st = normalize_NSLC_codes(
        st, inv, sta_translation_file=sta_translation_file,
        std_network_code=std_network_code, std_location_code=std_location_code,
        std_channel_prefix=std_channel_prefix, parallel=False, cores=1)
    
    # Do noise-balancing by the station's PSDPDF average
    if noise_balancing:
        # if not hasattr(st, "balance_noise"):
        #     bound_method = st_balance_noise.__get__(st)
        #     st.balance_noise = bound_method
        st = st.filter('highpass', freq=0.1, zerophase=True).detrend()
        st = st_balance_noise(st, inv, balance_power_coefficient=
                              balance_power_coefficient)
        st = st.taper(0.005, type='hann', max_length=None, side='both')

    # Put masked array into response-corrected stream st:
    for j, tr in enumerate(st):
        if isinstance(masked_st[j].data, np.ma.MaskedArray):
            try:
                tr.data = np.ma.masked_array(
                    tr.data, mask=masked_st[j].data.mask)
            except np.ma.MaskError as e:
                Logger.error(e)
                Logger.error(
                    'Numpy Mask error - this is a problematic exception '
                    + 'because it does not appear to be reproducible. I shall '
                    + 'hence try to process this trace again.')
                
                
    # Downsample if necessary
    if downsampled_max_rate is not None:
        for tr in st:
            if tr.stats.sampling_rate > downsampled_max_rate:
                tr.resample(sampling_rate=downsampled_max_rate,
                            no_filter=False)

    outtoc = default_timer()
    try:
        Logger.debug(
            'Initial processing of %s traces in stream %s took: {0:.4f}s'.format(
                outtoc - outtic), str(len(st)), st[0].id)
    except Exception as e:
        Logger.warning(e)

    return st



def check_normalize_sampling_rate(
    st, inv, min_segment_length_s=10, max_sample_rate_diff=1,
    skip_check_sampling_rates=[20, 40, 50, 66, 75, 100, 500],
    skip_interp_sample_rate_smaller=1e-7, interpolation_method='lanczos'):
    """
    Function to check the sampling rates of traces against the response-
        provided sampling rate and the against all other segments from the same
        NSLC-object in the stream. Traces with mismatching sample-rates are 
        excluded. If the sample rate is only slightly off, then the trace is
        interpolated onto the response-defined sample rate.
        If there is no matching response, then the traces are checked against
        other traces with the same NSLC-identified id.
        The sample rate can be adjusted without interpolation if it is just
        very slightly off due to floating-point issues (e.g., 99.9999998 vs 
        100 Hz). A value of 1e-7 implies that for a 100 Hz-sampled channel, the
        sample-wise offset across 24 hours is less than 1 sample.
        
    returns: stream, bool
    """
    
    # First, throw out trace bits that are too short, and hence, likely
    # problematic (e.g., shorter than 10 s):
    keep_st = Stream()
    #st_copy = st.copy()
    st_tr_to_be_removed = Stream()
    for tr in st:
        if tr.stats.npts < min_segment_length_s * tr.stats.sampling_rate:
            #st_copy.remove(tr)
            st_tr_to_be_removed += tr
    for tr in st_tr_to_be_removed:
        if tr in st:
            st.remove(tr)
    #st = st_copy
    
    # Create list of unique Seed-identifiers (NLSC-objects)
    seed_id_list = [tr.id for tr in st]
    unique_seed_id_list = set(seed_id_list)
    
    # Do a quick check: if all sampling rates are the same, and all values are
    # one of the allowed default values, then skip all the other tests.
    channel_rates = [tr.stats.sampling_rate for tr in st]
    if all([cr in skip_check_sampling_rates for cr in channel_rates])\
            and len(set(channel_rates)) <= 1:
        return st, False
    
    # Now do a thorough check of the sampling rates; exclude traces if needed;
    # interpolate if needed.
    new_st = Stream()
    # Check what the trace's sampling rate should be according to the inventory
    for seed_id in unique_seed_id_list:
        # Need to check for the inventory-information only once per seed_id
        # (Hopefully - I'd rather not do this check for every segment in a 
        # mseed-file; in case there are many segments.)
        check_st = st.select(id=seed_id).copy()
        tr = check_st[0]
        
        check_inv = inv.select(
            network=tr.stats.network, station=tr.stats.station,
            location=tr.stats.location,
            channel=tr.stats.channel, time=tr.stats.starttime)
        if len(check_inv) > 0:
            found_matching_resp = True
        else:
            found_matching_resp, tr, check_inv =\
                try_find_matching_response(tr, inv)

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
                        + '(%s Hz) starting on %s and channel response '
                        + 'information (%s Hz), removing offending traces.',
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
                        'There are traces with differing sampling rates '
                        + 'for %s on %s. There is no response that defines'
                        + ' the correct sampling rate for this channel, '
                        + 'so I am keeping only those traces with the most '
                        + 'common sampling rate for this channel (%s Hz).',
                        tr.id, str(tr.stats.starttime)[0:19],
                        def_channel_sample_rate)
                else:
                    keep_st += tr
        else:
            keep_st = matching_st

        # And then we need to correct the offending trace's sampling rates
        # through interpolation.
        raw_datatype = st[0].data.dtype
        channel_rates = [tr.stats.sampling_rate for tr in keep_st]
        if any(cr != def_channel_sample_rate for cr in channel_rates):
            for tr in keep_st:
                sample_rate_diff =\
                    tr.stats.sampling_rate - def_channel_sample_rate
                if 0 < abs(sample_rate_diff) <= max_sample_rate_diff:
                    # If difference is very small, then only adjust sample-rate
                    # description.
                    if abs(sample_rate_diff) < skip_interp_sample_rate_smaller:
                        Logger.info('Correcting sampling rate of %s on %s '
                                    + 'by adjusting trace stats.', tr.id,
                                    str(tr.stats.starttime))                        
                        tr.stats.sampling_rate = def_channel_sample_rate
                    else:
                        Logger.info('Correcting sampling rate of %s on %s '
                                    + 'with interpolation.', tr.id,
                                    str(tr.stats.starttime))
                        tr.interpolate(sampling_rate=def_channel_sample_rate,
                                       method=interpolation_method, a=25)
                        # Make sure data have the same datatype as before
                        if tr.data.dtype is not raw_datatype:
                            tr.data = tr.data.astype(raw_datatype)
        new_st += keep_st
        
    return new_st, True


def try_remove_responses(st, inv, taper_fraction=0.05, pre_filt=None,
                         parallel=False, cores=None, output='DISP'):
    """
    """
    # remove response
    if not parallel:
        for tr in st:
            tr = _try_remove_responses(tr, inv, taper_fraction=taper_fraction,
                                       pre_filt=pre_filt, output=output)
    else:
        if cores is None:
            cores = min(len(st), cpu_count())
            
        # If this function is called from a subprocess and asked for parallel
        # execution, then open a thread-pool to distribute work further.
        if current_process().name == 'MainProcess':
            my_pool = get_context("spawn").Pool
        else:
            my_pool = ThreadPool
        
        with threadpool_limits(limits=1, user_api='blas'):
            with pool_boy(Pool=my_pool, traces=len(st), cores=cores) as pool:
                #params = ((tr, inv, taper_fraction, parallel, cores)
                #          for tr in st)
                results = [pool.apply_async(
                    _try_remove_responses,
                    args=(tr, inv.select(station=tr.stats.station),
                          taper_fraction, pre_filt, output)
                    ) for tr in st]
        traces = [res.get() for res in results]
        traces = [tr for tr in traces if tr is not None]
        st = Stream(traces=traces)
        
        # my_pool().close()
        # my_pool().join()
        # my_pool().terminate()
        
        # params = ((sub_arr, arr_thresh, trig_int, full_peaks)
        #             for sub_arr, arr_thresh in zip(arr, thresh))
                
        # with pool_boy(Pool, len(stream_dict), cores=max_workers) as pool:
        #     func = partial(
        #         _meta_filter_stream, stream_dict=stream_dict,
        #         lowcut=lowcut, highcut=highcut)
        #     results = [pool.apply_async(func, key)
        #                 for key in stream_dict.keys()]
        # for result in results:
        #     processed_stream_dict.update(result.get())
    
    return st



def _try_remove_responses(tr, inv, taper_fraction=0.05, pre_filt=None,
                          output='DISP'):
    """
    Internal function that tries to remove the response from a trace
    """
    # remove response
    outtic = default_timer()
    try:
        tr.remove_response(inventory=inv, output=output, water_level=60,
                           pre_filt=pre_filt, zero_mean=True, taper=True,
                           taper_fraction=taper_fraction)
        sel_inv = inv.select(
            network=tr.stats.network, station=tr.stats.station,
            location=tr.stats.location, channel=tr.stats.channel,
            time=tr.stats.starttime)
    except Exception as e:
        # Try to find the matching response
        found_matching_resp, tr, sel_inv = try_find_matching_response(
            tr, inv.copy())
        if not found_matching_resp:
            Logger.warning('Finally cannot remove reponse for ' + str(tr) +
                           ' - no match found')
            # Logger.warning(e)
        else:
            # TODO: what if trace's location code is empty, and there are 
            # multiple instruments at one station that both match the trace in
            # a channel code?
            try: 
                tr.remove_response(inventory=sel_inv, output=output,
                                   water_level=60, pre_filt=pre_filt,
                                   zero_mean=True, taper=True,
                                   taper_fraction=taper_fraction)
            except Exception as e:
                found_matching_resp = False
                Logger.warning('Finally cannot remove reponse for ' + str(tr) +
                               ' - no match found')
                # Logger.warning(e)
        # IF reponse isn't found, then adjust amplitude to something 
        # similar to the properly corrected traces
        if not found_matching_resp:
            tr.data = tr.data / 1e7

    # Set station coordinates 
    # initialize
    tr.stats["coordinates"] = {}
    tr.stats["coordinates"]["latitude"] = np.NaN
    tr.stats["coordinates"]["longitude"] = np.NaN
    tr.stats["coordinates"]["elevation"] = 0
    tr.stats["coordinates"]["depth"] = 0
    tr.stats['distance'] = np.NaN
    # try to set coordinates from channel-info; but using station-info is
    # also ok
    stachan_info = None
    try:
        stachan_info = sel_inv.networks[0].stations[0]
        stachan_info = sel_inv.networks[0].stations[0].channels[0]
    except Exception as e:
        Logger.warning('Cannot find metadata for trace %s', tr.id)
        pass
    try:
        tr.stats["coordinates"]["latitude"] = stachan_info.latitude
        tr.stats["coordinates"]["longitude"] = stachan_info.longitude
        tr.stats["coordinates"]["elevation"] = stachan_info.elevation 
        tr.stats["coordinates"]["depth"] = stachan_info.depth
    except Exception as e:
         Logger.warning('Could not set all station coordinates for %s', tr.id)
    # Gain all traces to avoid a float16-zero error
    # basically converts from m to nm (for displacement)
    tr.data = tr.data * 1e6
    # Now convert back to 32bit-double to save memory ! (?)
    #if np.dtype(tr.data[0]) == 'float64':
    tr.data = np.float32(tr.data)
    # before response removed: tr.data is in int32
    # after response removed, tr.data is in float64)
    
    outtoc = default_timer()
    if (outtoc - outtic) > 3:
        Logger.debug(
            'Response-removal of trace %s took: {0:.4f}s'.format(
                outtoc - outtic), tr.id)
    
    # Check that data are not NaN:
    if np.isnan(tr.data).any():
        Logger.warning('Data for trace %s contain NaN after response-removal,'
                       + 'will discard this trace.', str(tr))
        return None
    
    return tr


def try_find_matching_response(tr, inv):
    """
    If code doesn't find the response, then assume that the trace's
    metadata lack network or location code. Look for reponse in inv-
    entory that has the same station code, and check start/endtimes
    of channel - correct trace stats if there's a match.
    :returns: bool, trace, inventory
    
    Logic:
    1. only location code is empty:
    2. neither location nor network codes are empty, but there is a response
       for an empty location code.
    2. network code is empty
    3. if not found, try again and allow any location code
    4 if not found, check if the channel code may contain a space in
       the middle
       5 if not found, allow space in channel and empty network
       6 if not found, allow space in channel and empty location
       7 if not found, allow space in channel, empty network, and 
         empty location
    """
    found = False
    # 1. only location code is empty:
    if (tr.stats.location == '' or tr.stats.location == '--')\
        and not tr.stats.network == '':
        tempInv = inv.select(network=tr.stats.network,
                             station=tr.stats.station,
                             channel=tr.stats.channel)
        found = False
        for network in tempInv.networks:
            for station in network.stations:
                for channel in station.channels:
                    if response_stats_match(tr, channel):
                        tr.stats.location = channel.location_code
                        inv = return_matching_response(tr, inv, network,
                                                       station, channel)
                        return True, tr, inv
    # 2. neither location nor network codes are empty, but there is a response
    #    for an empty location code.
    if not (tr.stats.location == '' or tr.stats.location == '--') and\
      not tr.stats.network == '':
        tempInv = inv.select(network=tr.stats.network,
                             station=tr.stats.station,
                             channel=tr.stats.channel)
        found = False
        for network in tempInv.networks:
            for station in network.stations:
                for channel in station.channels:
                    if response_stats_match(tr, channel):
                        tr.stats.location = channel.location_code
                        inv = return_matching_response(tr, inv, network,
                                                       station, channel)
                        return True, tr, inv
    # 2. network code is empty
    if tr.stats.network == '':
        tempInv = inv.select(station=tr.stats.station,
                             location=tr.stats.location,
                             channel=tr.stats.channel)
        found = False
        for network in tempInv.networks:
            for station in network.stations:
                #chan_codes = [c.code for c in station.channels]
                for channel in station.channels:
                    if response_stats_match(tr, channel):
                        tr.stats.network = network.code
                        inv = return_matching_response(tr, inv, network,
                                                       station, channel)
                        return True, tr, inv
    # 3. if not found, try again and allow any location code
    if tr.stats.network == '':
        tempInv = inv.select(station=tr.stats.station,
                                channel=tr.stats.channel)
        found = False
        for network in tempInv.networks:
            for station in network.stations:
                for channel in station.channels:
                    if response_stats_match(tr, channel):
                        tr.stats.network = network.code
                        tr.stats.location = channel.location_code
                        inv = return_matching_response(tr, inv, network,
                                                       station, channel)
                        return True, tr, inv
    # 4 if not found, check if the channel code may contain a space in
    #   the middle
    if tr.stats.channel[1] == ' ':
        tempInv = inv.select(network = tr.stats.network,
                                station=tr.stats.station,
                                location=tr.stats.location,
                                channel=tr.stats.channel[0] + '?' +
                                tr.stats.channel[2])
        found = False
        for network in tempInv.networks:
            for station in network.stations:
                for channel in station.channels:
                    if response_stats_match(tr, channel):
                        tr.stats.channel = channel.code
                        inv = return_matching_response(tr, inv, network,
                                                       station, channel)
                        return True, tr, inv
        # 5 if not found, allow space in channel and empty network
        tempInv = inv.select(station=tr.stats.station,
                                location=tr.stats.location,
                                channel=tr.stats.channel[0] + '?' +
                                tr.stats.channel[2])
        found = False
        for network in tempInv.networks:
            for station in network.stations:
                for channel in station.channels:
                    if response_stats_match(tr, channel):
                        tr.stats.network = network.code
                        tr.stats.channel = channel.code
                        inv = return_matching_response(tr, inv, network,
                                                       station, channel)
                        return True, tr, inv
        # 6 if not found, allow space in channel and empty location
        tempInv = inv.select(network=tr.stats.network,
                                station=tr.stats.station,
                                channel=tr.stats.channel[0] + '?' +
                                tr.stats.channel[2])
        found = False
        for network in tempInv.networks:
            for station in network.stations:
                for channel in station.channels:
                    if response_stats_match(tr, channel):
                        tr.stats.location = channel.location_code
                        tr.stats.channel = channel.code
                        inv = return_matching_response(tr, inv, network,
                                                       station, channel)
                        return True, tr, inv
        # 7 if not found, allow space in channel, empty network, and 
        #   empty location
        tempInv = inv.select(station=tr.stats.station,
                                channel=tr.stats.channel[0] + '?' +
                                tr.stats.channel[2])
        found = False
        for network in tempInv.networks:
            for station in network.stations:
                for channel in station.channels:
                    if response_stats_match(tr, channel):
                        tr.stats.network = network.code
                        tr.stats.location = channel.location_code
                        tr.stats.channel = channel.code
                        inv = return_matching_response(tr, inv, network,
                                                       station, channel)
                        return True, tr, inv

    return found, tr, Inventory()



def response_stats_match(tr, channel):
    #"""
    #check whether some criteria of the inventory-response and the trace match
    #tr: obspy.trace
    #channel: inv.networks.channel
    #"""
    sample_rate_diff = abs(channel.sample_rate - tr.stats.sampling_rate)
    if (channel.start_date <= tr.stats.starttime 
            and (channel.end_date >= tr.stats.endtime
                 or channel.end_date is None)
            and sample_rate_diff < 1):
        return True
    else:
        return False



def return_matching_response(tr, inv, network, station, channel):
    """
    """
    inv = inv.select(
        network=network.code, station=station.code, channel=channel.code,
        location=channel.location_code, starttime=tr.stats.starttime,
        endtime=tr.stats.endtime)
    if len(inv.networks) > 1 or len(inv.networks[0].stations) > 1\
            or len(inv.networks[0].stations[0].channels) > 1:
        Logger.debug('Found more than one matching response for trace, '
                     + 'returning all.')
            
    return inv



def normalize_NSLC_codes(st, inv, std_network_code="NS",
                         std_location_code="00", std_channel_prefix="BH",
                         parallel=False, cores=None,
                         sta_translation_file="station_code_translation.txt",
                         forbidden_chan_file="forbidden_chans.txt"):
    """
    1. Correct non-FDSN-standard-complicant channel codes
    2. Rotate to proper ZNE, and hence change codes from [Z12] to [ZNE]
    3. Translate old station code
    4. Set all network codes to NS.
    5. Set all location codes to 00.
    """
    import numpy as np
    # 0. remove forbidden channels that cause particular problems
    forbidden_chans = load_forbidden_chan_file(file=forbidden_chan_file)
    st_copy = st.copy()
    for tr in st_copy:
        if wcmatch.fnmatch.fnmatch(tr.id, forbidden_chans):
            st.remove(tr)
            Logger.info('Removed trace %s because it is a forbidden trace',
                        tr.id)

    # 1.
    for tr in st:
        # Check the channel names and correct if required
        if len(tr.stats.channel)<=2 and len(tr.stats.location)==1:
            tr.stats.channel = tr.stats.channel + tr.stats.location
        chn = tr.stats.channel
        if chn[1] in 'LH10V ':
        #=='L' or chn[1]=='H' or chn[1]==' ' or\  chn[1]=='1' or chn[1]=='0' or chn[1]=='V':
            tr.stats.channel = std_channel_prefix + chn[2]
            #tr.stats.location = '00'
    # 2. Rotate to proper ZNE and rename channels to ZNE
    # for tr in st:
    #     if tr.stats.channel[-1] in '12':
    #         rot_inv = inv.select(network=tr.stats.network,
    #                              station=tr.stats.station,
    #                              location=tr.stats.location,
    #                              channel=tr.stats.channel,
    #                              time=tr.stats.starttime)
    #         chn_inv = rot_inv.networks[0].stations[0].channels
    #         if len(chn_inv) > 1:
    #             Logger.info('ZNE rotation: Found more than one matching '
    #                         + 'responses for %s, %s, returning first response',
    #                         chn_inv, str(tr.stats.starttime)[0:10])
    #             chn_inv = chn_inv[0]
    #         if len(chn_inv) == 0:
    #             continue
    #         if chn_inv.code[-1] == '1' and\
    #                 abs(chn_inv.azimuth) <= rotation_threshold_degrees:
    #             tr.stats.channel[-1] = 'N'
    #         if chn_inv.code[-1] == '2' and\
    #                 90 - abs(chn_inv.azimuth) <= rotation_threshold_degrees:
    #             tr.stats.channel[-1] = 'E'
    if parallel:
        st = parallel_rotate(st, inv, cores=cores, method="->ZNE")    
    else:
        st.rotate(method="->ZNE", inventory=inv)

    # Need to merge again here, because rotate may split merged traces if there
    # are masked arrays (i.e., values filled with None). The merge here will
    # recreate the masked arrays (after they disappeared during rotate).
    st = st.merge(method=1, fill_value=None, interpolation_samples=-1)

    # 3. +4 +5 Translate station codes, Set network and location codes
    # load list of tuples for station-code translation
    sta_fortransl_dict, sta_backtrans_dict = load_station_translation_dict(
        file=sta_translation_file)
    for tr in st:
        tr.stats.network = std_network_code
        tr.stats.location = std_location_code
        if tr.stats.station in sta_fortransl_dict:
            # Make sure not to normalize station/channel-code to a combination
            # that already exists in stream
            existing_sta_chans = st.select(station=sta_fortransl_dict.get(
                tr.stats.station), channel=tr.stats.channel)
            if len(existing_sta_chans) == 0:
                tr.stats.station = sta_fortransl_dict.get(tr.stats.station)

    return st


def get_all_relevant_stations(selectedStations, sta_translation_file=
                              "station_code_translation.txt"):
    """
    return list of relevant stations
    """
    relevantStations = selectedStations
    sta_fortransl_dict, sta_backtrans_dict = load_station_translation_dict(
        file=sta_translation_file)

    for sta in selectedStations:
        if sta in sta_backtrans_dict:
            relevantStations.append(sta_backtrans_dict.get(sta))
    return relevantStations


def load_station_translation_dict(file="station_code_translation.txt"):
    """
    reads a list of stations with their alternative names from a file
    returns a dictionary of key:alternative name, value: standard name
    """
    station_forw_translation_dict = dict()
    try:
        f = open(file, "r+")
    except Exception as e:
        Logger.error('Cannot load station translation file %s', file)
        Logger.error(e)
        return station_forw_translation_dict, station_forw_translation_dict

    for line in f.readlines()[1:]:
        #station_translation_list.append(tuple(line.strip().split()))
        standard_sta_code, alternative_sta_code = line.strip().split()
        station_forw_translation_dict[alternative_sta_code] = standard_sta_code
    station_backw_translation_dict = {y:x for x,y in 
                                      station_forw_translation_dict.items()}
    return station_forw_translation_dict, station_backw_translation_dict


def load_forbidden_chan_file(file="forbidden_chans.txt"):
    """
    reads a list of channels that are to be removed from all EQcorrscan-data,
    e.g., because of some critical naming conflict (e.g. station NRS as part of
    the DK network and as Norsar array beam code)
    """
    forbidden_chans = []
    try:
        f = open(file, "r+")
    except Exception as e:
        Logger.error('Cannot load forbidden channel file %s', file)
        Logger.error(e)
        return forbidden_chans
    for line in f.readlines():
        forbidden_chans.append(line.strip())

    return forbidden_chans


def check_template(st, template_length, remove_nan_strict=True,
                   max_perc_zeros=5, allow_channel_duplication=True):
    """
    """
    # Now check the templates
    # Check that all traces are the same length:
    t_lengths = [len(tr.data) for tr in st]
    t_length_max = max(t_lengths)
    if any([t_item == t_length_max for t_item in t_lengths]):
        Logger.info('Template stream: ' + str(st[0].stats.starttime)[0:10]
                    + ' has traces with unequal lengths.')
    # Check each trace
    k = 0
    channelIDs = list()
    st_copy = st.copy()
    for tr in st_copy:
        # Check templates for duplicate channels (happens when there are
        # P- and S-picks on the same channel). Then throw away the
        # S-trace (the later one) for now.
        if tr.id in channelIDs:
            for j in range(0, k):
                testSameIDtrace = st[j]
                if tr.id == testSameIDtrace.id:
                    # remove if the duplicate traces have the same start-time
                    if tr.stats.starttime == testSameIDtrace.stats.starttime\
                            and tr in st:
                        st.remove(tr)
                        continue
                    # if channel-duplication is forbidden, then throw away the
                    # later trace (i.e., S-trace)
                    elif not allow_channel_duplication:
                        st.remove(testSameIDtrace)
                        continue
        else:
            channelIDs.append(tr.id)
            k += 1

    st_copy = st.copy()
    for tr in st_copy:
        #Check that the trace is long enough
        if tr.stats.npts < template_length*tr.stats.sampling_rate and tr in st:
            st.remove(tr)
            Logger.info('Trace ' + tr.stats.network + '.' + tr.stats.station 
                        + '.' + tr.stats.location + '.' + tr.stats.channel
                        + ' is too short, removing from template.')
        # Check that the trace has no NaNs
        if remove_nan_strict and any(np.isnan(tr.data)) and tr in st:
            st.remove(tr)
            Logger.info('Trace ' + tr.stats.network + tr.stats.station + 
                        tr.stats.location + tr.stats.channel
                        + ' contains NaNs, removing from template.')
        # Check that not more than 5 % of the trace is zero:
        
        n_nonzero = np.count_nonzero(tr.copy().detrend().data)
        #if sum(tr.copy().detrend().data==0) > tr.data.size*max_perc_zeros\
        if (n_nonzero < tr.data.size * (1-max_perc_zeros) and tr in st):
            st.remove(tr)
            Logger.info('Trace ' + tr.stats.network + tr.stats.station + 
                        tr.stats.location + tr.stats.channel
                        + ' contains more than ' + str(max_perc_zeros*100)
                        + ' %% zeros, removing from template.')
        # Correct messed-up location/channel in case of LRW, MOL
        #if len(tr.stats.channel)<=2 and len(tr.stats.location)==1:
        #    tr.stats.channel = tr.stats.channel + tr.stats.location
        #chn = tr.stats.channel
        #if chn[1]=='L' or chn[1]=='H' or chn[1]==' ':
        #    tr.stats.channel = 'HH' + chn[2]
        #    tr.stats.location = '00'
        #tr.stats.network = 'XX'
    return st


def print_error_plots(st, path='ErrorPlots', time_str=''):
    """
    Prints a daylong-plot of every trace in stream to specified folder.
    """
    mid_time = st[0].stats.starttime + (
        st[0].stats.starttime - st[0].stats.endtime) / 2
    current_day_str = str(mid_time)[0:10]
    try:
        if not os.path.isdir(path):
            os.path.mkdir(path)
        for trace in st:
            png_name = time_str + '_' + trace.stats.station +\
                '_' + trace.stats.channel + '.png'
            outPlotFile = os.path.join(path, png_name)
            trace.plot(type='dayplot', size=(1900, 1080), outfile=outPlotFile,
                       data_unit='nm')
    except Exception as e:
        Logger.error('Got an exception when trying to plot '
                        + 'Error figures for %s', current_day_str)
        Logger.error(e)


def multiplot_detection(party, tribe, st, out_folder='DetectionPlots'):
    """
    Create a plot of a detection including the background stream.
    """
    if len(party) == 0:
        return
    for family in party:
        if len(family) == 0:
            continue
        for detection in family:
            #times = [d.detect_time for d in family]
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
                ).resample(tribe[0].samp_rate, no_filter=True)
            dst = dst.trim(starttime=stt-30, endtime=stt+240)
            filename = str(detection.detect_time)[0:19] + '_'\
                + family.template.name + '.png'
            filename = os.path.join(out_folder, filename.replace('/', ''))
            try:
                fig = detection_multiplot(
                    stream=dst, template=family.template.st, times=times,
                    save=True, savefile=filename, size=(20, 30), show=False)
            except Exception as e:
                Logger.error('Could not create multi-plot for detection %s',
                             detection)
                Logger.error(e)


def reevaluate_detections(
    party, short_tribe, stream, threshold_type='MAD', threshold=9,
    trig_int=40.0, overlap='calculate', plot=False, plotDir='DetectionPlots',
    daylong=False, fill_gaps=False, ignore_bad_data=False, ignore_length=True,
    parallel_process=False, cores=None, concurrency='multithread',
    xcorr_func='fftw', group_size=1, full_peaks=False,
    save_progress=False, process_cores=None, spike_test=False, min_chans=4,
    time_difference_threshold=2, detect_value_allowed_error=60,
    return_party_with_short_templates=False):
    """
    This function takes a set of detections and reruns the match-filter
    detection with a set of templates that are shortened to XX length. Only if
    the detections are also significant (e.g., better detect_value) with the
    shorter templates, then they are retained. Other detections that do not
    pass this test are considered misdetections, which can often happen when
    seismic arrays are involved in detection and there is a seismic event near
    one of the arrays.
    """
    # Maybe do some checks to see if tribe and short_tribe have somewhat of the
    # same templates?

    # Select only the relevant templates
    det_templ_names = set([d.template_name for f in party for d in f])
    short_tribe = Tribe(
        [short_tribe.select(templ_name) for templ_name in det_templ_names])
    # Find the relevant parts of the stream so as not to rerun the whole day:
    det_times = [d.detect_time for f in party for d in f]
    Logger.info('Re-evaluating party to sort out misdetections, checking %s'
                + ' detections.', len(det_times))
    if len(det_times) == 0:
        return party
    earliest_det_time = min(det_times)
    latest_det_time = max(det_times)
    # if detections on significan part of the day:
    det_st = stream
    # if (latest_det_time - earliest_det_time) > 86400 * 0.5:
    #     Logger.info('Using full day for detection re-evaluation.')
    #     # then use whole day
    #     det_st = stream
    # else:
    #     #cut around half an hour before earliest and half an hour after latest 
    #     # detection
    #     tr_start_times = [tr.stats.starttime for tr in stream.traces]
    #     tr_end_times = [tr.stats.endtime for tr in stream.traces]
    #     earliest_st_time = min(tr_start_times)
    #     latest_st_time = max(tr_end_times)
    #     starttime = earliest_det_time - (10 * 60)
    #     if starttime < earliest_st_time:
    #         starttime = earliest_st_time
    #     endtime = latest_det_time + (10 * 60)
    #     if endtime > latest_st_time:
    #         endtime = latest_st_time
    #     det_st = stream.trim(starttime=starttime, endtime=endtime)
    #     daylong = False
        # for temp in short_tribe:
        #     temp.process_len = endtime - starttime

    # rerun detection
    # TODO: if threshold is MAD, then I would have to set the threshold lower
    # than before. Or use other threshold here.
    short_party = short_tribe.detect(
        stream=det_st, threshold=threshold, trig_int=3.0,
        threshold_type=threshold_type, overlap=overlap, plot=plot,
        plotDir=plotDir, daylong=daylong, fill_gaps=fill_gaps,
        ignore_bad_data=ignore_bad_data, ignore_length=ignore_length,
        parallel_process=parallel_process, cores=cores,
        concurrency=concurrency, xcorr_func=xcorr_func, group_size=group_size,
        full_peaks=full_peaks, save_progress=save_progress,
        process_cores=process_cores, spike_test=spike_test)

    # Check detections from short templates again the original set of
    # detections. If there is no short-detection for an original detection,
    # or if the detect_value is a lot worse, then remove original detection
    # from party.
    return_party = Party()
    short_party_templ_names = [
        f.template.name for f in short_party if f is not None]
    for fam in party:
        if fam.template.name not in short_party_templ_names:
            continue # do not retain the whole family
        # select matching family
        short_fam = short_party.select(fam.template.name)
        if len(short_fam) == 0:
            Logger.info('Re-evaluation obtained no detections for %s.', fam)
            continue
        short_det_times_np = np.array(
            [np.datetime64(d.detect_time.ns, 'ns') for d in short_fam])
        
        # Allow to return either partys with the original templates or the
        # short templates
        if return_party_with_short_templates:
            return_family = Family(short_fam.template)
        else:
            return_family = Family(fam.template)

        # Check detections for whether they fulfill the reevaluation-criteria.
        for det in fam:
            time_diffs = abs(
                short_det_times_np - np.datetime64(det.detect_time.ns, 'ns'))
            time_diff_thresh = np.timedelta64(
                int(time_difference_threshold * 1E9), 'ns')
            # If there is a short-detection close enough in time to the
            # original detection, then check detection values:
            if not any(time_diffs <= time_diff_thresh):
                Logger.info('No detections within time-threshold found during '
                            + 're-evaluation of %s at %s', det.template_name,
                            det.detect_time)
            else:
                # get the matching short-detection
                sdi = np.argmin(time_diffs)
                short_det = short_fam[sdi]
                # If detection-value is now better within error, only then keep
                # the original detection.
                if (abs(short_det.detect_val) >= abs(
                  det.detect_val * (1 - detect_value_allowed_error/100))):
                    if return_party_with_short_templates:
                        return_family += short_det
                    else:
                        return_family += det
                else:
                    Logger.info('Re-evaluation detections did not meet '
                                + 'detection-value criterion for %s at %s',
                                det.template_name, det.detect_time)
        if len(return_family) >= 0:
            return_party += return_family

    if len(return_party) == 0:
        n_det = 0
    else:
        return_party = return_party.decluster(
            trig_int=trig_int, timing='detect', metric='thresh_exc',
            min_chans=min_chans, absolute_values=True)
        n_det = len([d for f in return_party for d in f])
    Logger.info('Re-evaluation finished, remaining are %s detections.',
                str(n_det))

    return return_party
