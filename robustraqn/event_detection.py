#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 17:23:45 2017

@author: felix
"""

# %%


def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


import os, glob, gc, math, calendar, matplotlib, platform, sys
from numpy.core.numeric import True_
# sys.path.insert(1, os.path.expanduser(
#   "~/Documents2/NorthSea/Elgin/Detection"))
from os import times
import pandas as pd
if not run_from_ipython:
    matplotlib.use('Agg')  # to plot figures directly for print to file
from importlib import reload
import numpy as np
import pickle
import hashlib
from joblib import Parallel, delayed, parallel_backend

from timeit import default_timer
import logging


from obspy import read_inventory
#from obspy.core.event import Event, Origin, Catalog
# from obspy.core.stream import Stream
from obspy.core.inventory.inventory import Inventory
#from obspy import read as obspyread
from obspy import UTCDateTime
# from obspy.io.mseed import InternalMSEEDError
# from obspy.clients.filesystem.sds import Client
from robustraqn.obspy.clients.filesystem.sds import Client

from robustraqn.processify import processify
from robustraqn.fancy_processify import fancy_processify

from eqcorrscan.utils import pre_processing
# from eqcorrscan.core import match_filter, lag_calc
# from eqcorrscan.utils.correlate import CorrelationError
from eqcorrscan.core.match_filter import Template, Tribe, MatchFilterError
from eqcorrscan.core.match_filter.party import Party, Family
# from eqcorrscan.utils.clustering import extract_detections
from eqcorrscan.utils.plotting import detection_multiplot
# from eqcorrscan.utils.despike import median_filter
# from eqcorrscan.core.match_filter import _spike_test
from eqcorrscan.utils.pre_processing import dayproc, shortproc
from eqcorrscan.core.template_gen import _rms

# import quality_metrics, spectral_tools, load_events_for_detection
# reload(quality_metrics)
# reload(load_events_for_detection)
from robustraqn.obspy.core import Stream, Trace
from robustraqn.quality_metrics import (
    create_bulk_request, get_waveforms_bulk, read_ispaq_stats)
    #get_parallel_waveform_client)
from robustraqn.load_events_for_detection import (
    prepare_detection_stream, init_processing, init_processing_wRotation,
    print_error_plots, get_all_relevant_stations, reevaluate_detections,
    multiplot_detection, try_apply_agc)
from robustraqn.spectral_tools import (Noise_model,
                                       get_updated_inventory_with_noise_models)
from robustraqn.templates_creation import create_template_objects

Logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")


def read_bulk_test(client, bulk, parallel=False, cores=None):
    # Read in continuous data and prepare for processing
    st = get_waveforms_bulk(client, bulk, parallel=parallel, cores=cores)
    return st


def append_list_completed_days(file, date, hash):
    """
    """
    # setting_hash = hashlib.md5()
    # setting_hash.update(kwargs)
    if file is None:
        return
    with open(file, "a+") as list_completed_days:
        list_completed_days.write(str(date) + ',' + str(hash) + '\n')


def prepare_day_overlap(
        tribe, short_tribe, stream, starttime_req, endtime_req,
        overlap_length=600, **kwargs):
    """
    set processing parameters to take care of 10 minutes overlap between days
    """
    # TODO: implement variable overlap
    starttime_overlap = starttime_req + 5 * 60
    endtime_overlap = endtime_req - 5 * 60
    process_length = endtime_overlap - starttime_overlap
    for templ in tribe:
        templ.process_length = process_length
    for templ in short_tribe:
        templ.process_length = process_length
    stream.trim(starttime=starttime_overlap, endtime=endtime_overlap)
    return tribe, short_tribe, stream


def get_multi_obj_hash(hash_object_list):
    """
    """
    hash_list = []
    for obj in hash_object_list:
        hash = None
        # # Some objects have __hash__ as a method that returns a string
        # try:
        #     hash = obj.__hash__()
        # except (AttributeError, TypeError):
        #     pass
        # # Some objects have __hash__ as an attribute
        # if hash is None:
        #     try:
        #         hash = obj.__hash__
        #     except AttributeError:
        #         pass
        # # Some objects have no __hash__, so hash their string-representation
        if hash is None:
            try:
                hash = hashlib.md5(obj.__str__(extended=True).encode('utf-8'))
            except TypeError:
                pass
        if hash is None:
            try:
                hash = hashlib.md5(str(obj).encode('utf-8'))
            except ValueError:
                pass
        if hash is not None:
            try:
                hash = hash.hexdigest()
            except AttributeError:
                pass
            hash_list.append(hash)
    settings_hash = hashlib.md5(str(hash_list).encode('utf-8')).hexdigest()
    return settings_hash


# @processify
def run_day_detection(
        clients, tribe, date, ispaq, selected_stations,
        parallel=False, cores=1, io_cores=1, remove_response=False,
        inv=Inventory(), noise_balancing=False, let_days_overlap=True,
        balance_power_coefficient=2, apply_agc=False, agc_window_sec=5,
        n_templates_per_run=20, xcorr_func='fftw',
        concurrency=None, arch='precise', trig_int=0, threshold=10,
        threshold_type='MAD', re_eval_thresh_factor=0.6, min_chans=10,
        decluster_metric='thresh_exc', hypocentral_separation=200,
        absolute_values=True, minimum_sample_rate=20,
        time_difference_threshold=3, detect_value_allowed_error=60,
        multiplot=False, day_st=Stream(), check_array_misdetections=False,
        min_n_station_sites=4, short_tribe=Tribe(), write_party=False,
        detection_path='Detections', redetection_path=None, copy_data=True,
        return_stream=False, dump_stream_to_disk=False, day_hash_file=None,
        use_weights=False, sta_translation_file=os.path.expanduser(
            "~/Documents2/ArrayWork/Inventory/station_code_translation.txt"),
        **kwargs):
    """
    Function to run reading, initial processing, detection etc. on one day.
    """
    Logger.info('Starting detection run for day %s', str(date)[0:10])
    # Keep user's data safe
    #  - - - - Probably not needed if it is also done in EQcorrscan, else COPY!
    if not copy_data:
        tribe = tribe.copy()
        short_tribe = short_tribe.copy()
    # Set the path to the folders with continuous data:
    # archive_path2 = '/data/seismo-wav/EIDA/archive'
    # client2 = Client(archive_path2)

    n_templates = len(tribe)
    if n_templates == 0:
        msg = 'Cannot detect events with an empty tribe!'
        raise ValueError(msg)
    n_runs = math.ceil(n_templates / n_templates_per_run)

    starttime = UTCDateTime(pd.to_datetime(date))
    starttime_req = starttime - 15 * 60
    endtime = starttime + 60 * 60 * 24
    endtime_req = endtime + 15 * 60
    current_day_str = date.strftime('%Y-%m-%d')

    if not os.path.exists(detection_path):
        os.mkdir(detection_path)
    if redetection_path is None:
        redetection_path = 'Re' + detection_path
    if not os.path.exists(redetection_path):
        os.mkdir(redetection_path)

    if day_hash_file is not None:
        # Check if this date has already been processed with the same settings
        # i.e., current date and a settings-based hash exist already in file
        Logger.info('Checking if a run with the same parameters has been '
                    'performed before...')
        settings_hash = get_multi_obj_hash(
            [tribe.templates, selected_stations, remove_response, inv, ispaq,
            noise_balancing, balance_power_coefficient, xcorr_func, arch,
            trig_int, threshold, re_eval_thresh_factor, min_chans, multiplot,
            check_array_misdetections, short_tribe, write_party,
            detection_path, redetection_path, time_difference_threshold,
            minimum_sample_rate, min_n_station_sites, apply_agc,
            agc_window_sec, use_weights])
        # Check hash against existing list
        try:
            day_hash_df = pd.read_csv(day_hash_file, names=["date", "hash"])
            if ((day_hash_df['date'] == current_day_str) &
                    (day_hash_df['hash'] == settings_hash)).any():
                Logger.info(
                    'Day %s already processed: Date and hash match entry in '
                    'date-hash list, skipping this day.', current_day_str)
                if not return_stream and dump_stream_to_disk:
                    return
                else:
                    return [Party(), Stream()]
        except FileNotFoundError:
            pass

    # keep input safe:
    day_st = day_st.copy()
    if len(day_st) == 0:
        # # Differentiate between Echo and Delta (deadlocks here otherwise)
        # req_parallel = False
        # if int(platform.release()[0]) >= 4 and parallel:
        #     req_parallel = parallel
        # Create a smart request, i.e.: request only recordings that match
        # the quality metrics criteria and that best match the priorities.
        bulk_request, bulk_rejected, day_stats = create_bulk_request(
            inv.select(starttime=starttime_req, endtime=endtime_req),
            starttime_req, endtime_req, stats=ispaq,
            parallel=parallel, cores=cores, stations=selected_stations,
            minimum_sample_rate=minimum_sample_rate, **kwargs)
        if not bulk_request:
            Logger.warning('No waveforms requested for %s - %s',
                           str(starttime)[0:19], str(endtime)[0:19])
            if not return_stream and dump_stream_to_disk:
                return
            else:
                return [Party(), Stream()]

        # Read in continuous data and prepare for processing
        day_st = Stream()
        for client in clients:
            # day_st += get_waveforms_bulk(client, bulk_request, parallel=parallel,
            #                              cores=io_cores)
            Logger.info('Requesting waveforms from client %s', client)
            # client = get_parallel_waveform_client(client)
            # day_st += client.get_waveforms_bulk_parallel(
            #     bulk_request, parallel=parallel, cores=io_cores)
            # Or with joblib:
            day_st += client.get_waveforms_bulk(
                bulk_request, parallel=parallel, cores=io_cores)

        Logger.info(
            'Successfully read in %s traces for bulk request of %s NSLC-'
            'objects for %s - %s.', len(day_st), len(bulk_request),
            str(starttime)[0:19], str(endtime)[0:19])
        day_st = prepare_detection_stream(
            day_st, tribe, parallel=parallel, cores=cores, try_despike=False,
            ispaq=day_stats)
        # daily_plot(day_st, year, month, day, data_unit="counts", suffix='')

        # Do initial processing (rotation, stats normalization, merging)
        # by parallelization across three-component seismogram sets.
        nyquist_f = minimum_sample_rate / 2
        day_st = init_processing_wRotation(
            day_st, starttime=starttime_req, endtime=endtime_req,
            # day_st, starttime=starttime, endtime=endtime,
            remove_response=remove_response, inv=inv,
            pre_filt=[0.1, 0.2, 0.9 * nyquist_f, 0.95 * nyquist_f],
            parallel=parallel, cores=cores,
            sta_translation_file=sta_translation_file,
            noise_balancing=noise_balancing,
            balance_power_coefficient=balance_power_coefficient, **kwargs)

        # # Alternatively, do parallel processing across each trace, but then
        # # the NSLC-normalization with rotation has to happen independently.
        # day_st = init_processing(
        #    day_st, inv, starttime=starttime, endtime=endtime,
        #    parallel=parallel, cores=cores, min_segment_length_s=10,
        #    max_sample_rate_diff=1, skip_interp_sample_rate_smaller=1e-7,
        #    interpolation_method='lanczos',
        #    skip_check_sampling_rates=[20, 40, 50, 66, 75, 100, 500],
        #    taper_fraction=0.005, downsampled_max_rate=None,
        #    noise_balancing=noise_balancing)

        # # # #Normalize NSLC codes
        # day_st = normalize_NSLC_codes(
        #     day_st, inv, parallel=False, cores=cores,
        #     std_network_code="NS", std_location_code="00",
        #     std_channel_prefix="BH",
        #     sta_translation_file="station_code_translation.txt")

    # daily_plot(day_st, year, month, day, data_unit="counts",
    #           suffix='resp_removed')
    # If there is no data for the day, then continue on next day.
    if not day_st.traces:
        Logger.warning('No data for detection on %s, continuing' +
                       ' with next day.', current_day_str)
        if not return_stream and dump_stream_to_disk:
            return
        else:
            return [Party(), Stream()]

    # Figure out weights to take difference in template trace SNR and channel
    # noise into account:
    if use_weights:
        for templ in tribe:
            for tr in templ.st:
                # Get trace snr from ispaq-stats  - need to calc noise-amp in
                # relevant frequenc band
                if day_stats is not None:
                    # day_stats[tr.id]
                    # TODO get noise level in a smarter way
                    det_day_noise_level = 1
                try:
                    station_weight_factor = (
                        tr.stats.extra.station_weight_factor)
                except AttributeError as e:
                    Logger.warning(e)
                    station_weight_factor = 1
                # look up noise on this day /trace
                # weight = trace_snr * trace_noise_level
                tr.stats.extra.weight = (
                    # TODO: use cube root??? - difference may be very small,
                    #       but should be tested on Snorre events
                    # tr.stats.extra.rms_snr ** (1/3) *
                    np.sqrt(tr.stats.extra.rms_snr) *
                    station_weight_factor *
                    np.sqrt(
                        tr.stats.extra.day_noise_level / det_day_noise_level))

    # tmp adjust process_lenght parameters
    daylong = True
    if let_days_overlap:
        daylong = False
        tribe, short_tribe, day_st = prepare_day_overlap(
            tribe, short_tribe, day_st, starttime_req, endtime_req)

    pre_processed = False
    if apply_agc and agc_window_sec:
        day_st, pre_processed = try_apply_agc(
            day_st, tribe, agc_window_sec=agc_window_sec,
            pre_processed=pre_processed, starttime=None,
            cores=cores, parallel=parallel, **kwargs)

    # Start the detection algorithm on all traces that are available.
    detections = []
    Logger.info('Start match_filter detection on %s with up to %s cores.',
                current_day_str, str(cores))
    try:
        party = tribe.detect(
            stream=day_st, threshold=threshold, trig_int=trig_int / 10,
            threshold_type=threshold_type, overlap='calculate', plot=False,
            plotDir='DetectionPlots', daylong=daylong,
            fill_gaps=True, ignore_bad_data=False, ignore_length=True, 
            #apply_agc=apply_agc, agc_window_sec=agc_window_sec,
            pre_processed=pre_processed,
            parallel_process=parallel, cores=cores,
            # concurrency='multiprocess', xcorr_func='time_domain',
            xcorr_func=xcorr_func, concurrency=concurrency, arch=arch,
            # xcorr_func='fftw', concurrency='multithread',
            # parallel_process=False, #concurrency=None,
            group_size=n_templates_per_run, full_peaks=False,
            save_progress=False, process_cores=cores, spike_test=False,
            use_weights=use_weights, copy_data=copy_data, **kwargs)
        # xcorr_func='fftw', concurrency=None, cores=1,
        # xcorr_func=None, concurrency=None, cores=1,
        # xcorr_func=None, concurrency='multiprocess', cores=cores,
        # party = Party().read()
    except Exception as e:
        Logger.error('Exception on %s', current_day_str)
        Logger.error(e, exc_info=True)
        print_error_plots(day_st, path='ErrorPlots', time_str=current_day_str)
        return_st = Stream()
        if return_stream:
            return_st = day_st
        Logger.info('Size of return_st is: %s', len(pickle.dumps(return_st)))
        if not return_stream and dump_stream_to_disk:
            return
        else:
            return [Party(), return_st]
    n_families = len([f for f in party if len(f) > 0])
    n_detections = len([d for f in party for d in f])
    Logger.info('Got a party of %s families with %s detections!',
                 str(n_families), str(n_detections))

    # check that detection occurred on request day, not during overlap time
    if let_days_overlap:
        families = [Family(
            template=family.template, detections=[
                detection for detection in family
                if detection.detect_time >= starttime and
                detection.detect_time <= endtime])
                    for family in party]
        party = Party(
            families=[family for family in families if len(family) > 0])

    if not party:
        Logger.warning('Party of families of detections is empty')
        return_st = Stream()
        append_list_completed_days(
            file=day_hash_file, date=current_day_str, hash=settings_hash)

        if return_stream:
            return_st = day_st
            Logger.debug(
                'Size of party is: %s', len(pickle.dumps(party)))
            Logger.debug(
                'Size of return_st is: %s', len(pickle.dumps(return_st)))
        if not return_stream and dump_stream_to_disk:
            return
        else:
            return [party, return_st]

    # Check for erroneous detections of real signals (mostly caused by smaller
    # seismic events near one of the arrays). Solution: check whether templates
    # with shorter length increase detection value - if not; it's not a
    # desriable detection.
    if check_array_misdetections:
        if len(short_tribe) < len(tribe):
            Logger.error('Missing short templates for detection-reevaluation.')
        else:
            party, short_party = reevaluate_detections(
                party, short_tribe, stream=day_st, threshold=threshold,
                re_eval_thresh_factor=re_eval_thresh_factor,
                trig_int=trig_int, threshold_type=threshold_type,
                overlap='calculate', plot=False, plotDir='ReDetectionPlots',
                fill_gaps=True, ignore_bad_data=False, daylong=daylong,
                pre_processed=pre_processed,
                ignore_length=True, parallel_process=parallel, cores=cores,
                xcorr_func=xcorr_func, concurrency=concurrency, arch=arch,
                #xcorr_func='time_domain', concurrency='multiprocess', 
                # min_chans=min_chans,
                group_size=n_templates_per_run, process_cores=cores,
                time_difference_threshold=time_difference_threshold,
                detect_value_allowed_error=detect_value_allowed_error,
                return_party_with_short_templates=True,
                min_n_station_sites=min_n_station_sites,
                use_weights=use_weights, copy_data=copy_data, **kwargs)

            append_list_completed_days(
                file=day_hash_file, date=current_day_str, hash=settings_hash)

            if not party:
                Logger.warning('Party of families of detections is empty')
                return_st = Stream()
                if return_stream:
                    return_st = day_st
                if not return_stream and dump_stream_to_disk:
                    return
                else:
                    return [party, return_st]
            
            for family in short_party:
                for detection in family:
                    _ = detection._calculate_event(
                        template=family.template, template_st=None,
                         estimate_origin=True, correct_prepick=True)
                
            short_party = short_party.decluster(
                trig_int=trig_int, timing='detect', metric=decluster_metric,
                hypocentral_separation=hypocentral_separation,
                min_chans=min_chans, absolute_values=absolute_values)
            # TODO: maybe the order should be:
            # check array-misdet - decluster party - compare short-party vs. party
            if write_party:
                detection_file_name = os.path.join(
                    redetection_path, 'UniqueDet_short_' + current_day_str)
                short_party.write(detection_file_name, format='tar', overwrite=True)
                short_party.write(detection_file_name + '.csv', format='csv',
                            overwrite=True)

    # Add origins to detections
    for family in party:
        for detection in family:
            _ = detection._calculate_event(
                template=family.template, template_st=None,
                    estimate_origin=True, correct_prepick=True)
    # Decluster detection and save them to filesf
    # metric='avg_cor' isn't optimal when one detection may only be good on
    # very few channels - i.e., allowing higher CC than any detection made on
    # many channels
    party = party.decluster(trig_int=trig_int, timing='detect',
                            metric=decluster_metric, min_chans=min_chans,
                            hypocentral_separation=hypocentral_separation,
                            absolute_values=absolute_values)

    if write_party:
        detection_file_name = os.path.join(detection_path,
                                           'UniqueDet' + current_day_str)
        party.write(detection_file_name, format='tar', overwrite=True)
        party.write(detection_file_name + '.csv', format='csv',
                    overwrite=True)

    if multiplot:
        multiplot_detection(party, tribe, day_st, out_folder='DetectionPlots')
    if dump_stream_to_disk:
        pickle.dump(day_st, open('tmp_st.pickle', "wb"), protocol=-1)
        pickle.dump(party, open('tmp_party.pickle', "wb"), protocol=-1)
    gc.collect()
    # n = gc.collect()
    # Logger.info("Number of unreachable objects collected by GC: ", n)
    # Logger.info("Uncollectable garbage: ", gc.garbage)
    
    return_st = Stream()
    if return_stream:
        return_st = day_st
    Logger.debug('Size of party is: %s', len(pickle.dumps(party)))
    Logger.debug('Size of return_st is: %s', len(pickle.dumps(return_st)))
    if not return_stream and dump_stream_to_disk:
        return
    else:
        return [party, return_st]


# %% 

if __name__ == "__main__":
    # add bulk_request method to SDS-client
    # client = get_waveform_client(client)

    #client = get_parallel_waveform_client(client)
    # contPath = '/data/seismo-wav/EIDA'

    #selected_stations=['BER','ASK','KMY','HYA','FOO','BLS5','STAV','SNART',
    #                  'MOL','LRW','BIGH','ESK']
    selected_stations = ['ASK','BER','BLS5','DOMB','EKO1','FOO','HOMB','HYA','KMY',
                        'MOL','ODD1','SKAR','SNART','STAV','SUE','KONO',
                        'BIGH','DRUM','EDI','EDMD','ESK','GAL1','GDLE','HPK',
                        'INVG','KESW','KPL','LMK','LRW','PGB1',
                        'EKB','EKB1','EKB2','EKB3','EKB4','EKB5','EKB6','EKB7',
                        'EKB8','EKB9','EKB10','EKR1','EKR2','EKR3','EKR4','EKR5',
                        'EKR6','EKR7','EKR8','EKR9','EKR10',
                        'NAO00','NAO01','NAO02','NAO03','NAO04','NAO05',
                        'NB200','NB201','NB202','NB203','NB204','NB205',
                        'NBO00','NBO01','NBO02','NBO03','NBO04','NBO05',
                        'NC200','NC201','NC202','NC203','NC204','NC205',
                        'NC300','NC301','NC302','NC303','NC304','NC305',
                        'NC400','NC401','NC402','NC403','NC404','NC405',
                        'NC600','NC601','NC602','NC603','NC604','NC605',
                'EKO2','EKO3','EKO4','EKO5','EKO6','EKO7','EKO8','EKO9'
                'EKO10','EKO11','EKO12','EKO13','EKO14','EKO15','EKO16',
                'EKO17','EKO18','EKO19','EKO20','EKO21','EKO22','EKO23',
                'EKO24','EKO25','EKO26','EKO27','EKO28',
                'GRA01','GRA02','GRA03','GRA04','GRA05','GRA06','GRA07',
                'GRA08','GRA09','GRA10', 'OSE01','OSE02','OSE03','OSE04',
                'OSE05','OSE06','OSE07','OSE08','OSE09','OSE10']
    # selected_stations  = ['ASK', 'BLS5', 'KMY', 'ODD1', 'NAO01', 'ESK', 'EDI', 'KPL']
    # selected_stations  = ['ASK', 'BER', 'BLS5', 'STAV', 'NAO001', 'ESK', 'EKB2']
    selected_stations  = ['EKB10', 'EKR1', 'EKB']
                        # need   need    need   need   need
    # without-KESW, 2018-event will not detect itself
                        # 'EKB1','EKB2','EKB3','EKB4','EKB5','EKB6','EKB7',
                        # 'EKB8','EKB9','EKB10','EKR1','EKR2','EKR3','EKR4','EKR5',
                        # 'EKR6','EKR7','EKR8','EKR9','EKR10',]
    # 'SOFL','OSL', 'MUD',
    relevant_stations = get_all_relevant_stations(
        selected_stations, sta_translation_file="station_code_translation.txt")
    
    seisan_rea_path = '../SeisanEvents/'
    seisan_wav_path = '../SeisanEvents/'
    
    sfiles = glob.glob(os.path.join(seisan_rea_path, '*L.S??????'))
    sfiles.sort(key = lambda x: x[-6:])
    # sfiles = [sfiles[8]]
    # sfiles = sfiles[5:7]
    # sfiles = sfiles[0:5] + sfiles[6:8]
    

    startday = UTCDateTime(2017,1,1,0,0,0)
    endday = UTCDateTime(2019,12,31,0,0,0)

    # startday = UTCDateTime(2007,6,4,0,0,0)
    # startday = UTCDateTime(2007,7,24,0,0,0)
    # startday = UTCDateTime(2008,6,30,0,0,0)
    # startday = UTCDateTime(2011,7,20,0,0,0)
    # startday = UTCDateTime(2013,10,28,0,0,0)
    # startday = UTCDateTime(2015,9,4,0,0,0)    
    # startday = UTCDateTime(2018,10,25,0,0,0)
    # startday = UTCDateTime(2019,7,10,0,0,0)
    startday = UTCDateTime(2019,9,24,0,0,0)
    #endday = UTCDateTime(2019,9,30,0,0,0)
    # startday = UTCDateTime(2010,9,1,0,0,0)

    # startday = UTCDateTime(2011,7,20,0,0,0) 
    # endday = UTCDateTime(2011,7,20,0,0,0)
    #endday = UTCDateTime(2013,3,16,0,0,0)
    #startday = UTCDateTime(2015,9,7,0,0,0)
    #endday = UTCDateTime(2016,1,1,0,0,0)

    startday = UTCDateTime(2021,1,1,0,0,0) 
    endday = UTCDateTime(2021,1,1,0,0,0)
    
    
    # TODO: debug segmentation fault on 2018-03-30 # this may be fixed through
    #       correct error-handling of mseed-read-errors
    # TODO: debug segmentation fault on 2012-09-25 # this may be fixed through
    #       correct error-handling of mseed-read-errors
    # TODO: dEBUG MASK ERROR on 2009-07-01
    # TODO: BrokenPipeError on 2015-11-30
    # startday = UTCDateTime(2018,3,30,0,0,0)
    # endday = UTCDateTime(2018,3,31,0,0,0)

    inv_file = '~/Documents2/ArrayWork/Inventory/NorSea_inventory.xml'
    # inv_file = '~/Documents2/ArrayWork/Inventory/NorSea_inventory.dataless_seed'
    # inv = read_inventory(os.path.expanduser(inv_file))
    inv = get_updated_inventory_with_noise_models(
        os.path.expanduser(inv_file),
        pdf_dir='~/repos/ispaq/WrapperScripts/PDFs/',
        outfile='inv.pickle', check_existing=True)

    templateFolder='Templates'
    parallel = True
    noise_balancing = True
    make_templates = False
    check_array_misdetections = True
    cores = 52
    balance_power_coefficient = 2

    # Read templates from file
    if make_templates:
        Logger.info('Creating new templates')
        tribe = create_template_objects(
            sfiles, relevant_stations, template_length=120,
            lowcut=0.2, highcut=9.9, min_snr=3, prepick=0.5, samp_rate=20,
            min_n_traces=8, seisan_wav_path=seisan_wav_path, inv=inv,
            remove_response=True, noise_balancing=noise_balancing, 
            balance_power_coefficient=balance_power_coefficient,
            parallel=parallel, cores=cores, write_out=True,
            make_pretty_plot=False)
        Logger.info('Created new set of templates.')
        
        short_tribe = Tribe()
        if check_array_misdetections:
            short_tribe = create_template_objects(
                sfiles, relevant_stations, template_length=10,
                lowcut=0.2, highcut=9.9, min_snr=3, prepick=0.5, samp_rate=20,
                min_n_traces=8, seisan_wav_path=seisan_wav_path, inv=inv,
                remove_response=True, noise_balancing=noise_balancing,
                balance_power_coefficient=balance_power_coefficient,
                parallel=parallel, cores=cores, make_pretty_plot=False,
                write_out=True, prefix='short_')
            Logger.info('Created new set of short templates.')
    else:    
        Logger.info('Starting template reading')
        # tribe = Tribe().read('TemplateObjects/Templates_min8tr_8.tgz')
        tribe = Tribe().read('TemplateObjects/Templates_min8tr_balNoise_9.tgz')
        # tribe = Tribe().read('TemplateObjects/Templates_min3tr_noNoiseBal_1.tgz')
        # tribe = Tribe().read('TemplateObjects/Templates_min3tr_balNoise_1.tgz')
        # tribe = Tribe().read('TemplateObjects/Templates_7tr_noNoiseBal_3.tgz')
        # tribe = Tribe().read('TemplateObjects/Templates_7tr_wiNoiseBal_3.tgz')
        Logger.info('Tribe archive readily read in')
        if check_array_misdetections:
            short_tribe = Tribe().read(
                'TemplateObjects/short_Templates_min8tr_balNoise_9.tgz')
            Logger.info('Short-tribe archive readily read in')

    # Load Mustang-metrics from ISPAQ for the whole time period
    # ispaq = read_ispaq_stats(folder='/home/felix/repos/ispaq/WrapperScripts/csv',
    #                          stations=relevant_stations, startyear=startday.year,
    #                          endyear=endday.year, ispaq_prefixes=['all'],
    #                          ispaq_suffixes=['simpleMetrics','PSDMetrics'])
    #availability = stats[stats['metricName'].str.contains("percent_availability")]

    # %% ### LOOP OVER DAYS ########
    day_st = Stream()
    dates = pd.date_range(startday.datetime, endday.datetime, freq='1D')
    # For each day, read in data and run detection from templates
    current_year = None
    for date in dates:
        # Load in Mustang-like ISPAQ stats for the whole year
        if not date.year == current_year:
            current_year = date.year
            ispaq = read_ispaq_stats(folder=
                '/home/felix/repos/ispaq/WrapperScripts/Parquet_database/csv_parquet',
                stations=relevant_stations, startyear=current_year,
                endyear=current_year, ispaq_prefixes=['all'],
                ispaq_suffixes=['simpleMetrics','PSDMetrics'],
                file_type = 'parquet')

        [party, day_st] = run_day_detection(
            tribe=tribe, date=date, ispaq=ispaq,
            selected_stations=relevant_stations, remove_response=True, inv=inv,
            parallel=parallel, cores=cores, noise_balancing=noise_balancing,
            balance_power_coefficient=balance_power_coefficient,
            threshold=12, check_array_misdetections=check_array_misdetections,
            trig_int=40, short_tribe=short_tribe, multiplot=True,
            write_party=True, min_chans=1)


# TODO: option to use C|C| instead of C for correlation stats
# - in line 476 multi_corr.c         # fftw
# - in line 51, 85 in time_corr.c    # time domain
# - in line 134 in matched_filter.cu # fmf-gpu
# - in line 109 in matched_filter.c  # fmf-cpu
