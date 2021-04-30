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
#sys.path.insert(1, os.path.expanduser("~/Documents2/NorthSea/Elgin/Detection"))

from os import times
import pandas as pd
if not run_from_ipython:
    matplotlib.use('Agg') # to plot figures directly for print to file
from importlib import reload
import numpy as np
import pickle

from timeit import default_timer
import logging
Logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")

from obspy import read_inventory
#from obspy.core.event import Event, Origin, Catalog
from obspy.core.stream import Stream
from obspy.core.inventory.inventory import Inventory
#from obspy import read as obspyread
from obspy import UTCDateTime
#from obspy.io.mseed import InternalMSEEDError
from obspy.clients.filesystem.sds import Client

from robustraqn.processify import processify
from robustraqn.fancy_processify import fancy_processify

# from eqcorrscan.utils import pre_processing
# from eqcorrscan.core import match_filter, lag_calc
# from eqcorrscan.utils.correlate import CorrelationError
from eqcorrscan.core.match_filter import Template, Tribe, MatchFilterError
from eqcorrscan.core.match_filter.party import Party
# from eqcorrscan.utils.clustering import extract_detections
from eqcorrscan.utils.plotting import detection_multiplot
# from eqcorrscan.utils.despike import median_filter
# from eqcorrscan.core.match_filter import _spike_test

# import quality_metrics, spectral_tools, load_events_for_detection
# reload(quality_metrics)
# reload(load_events_for_detection)
from robustraqn.quality_metrics import (create_bulk_request, get_waveforms_bulk,
                             read_ispaq_stats)
from robustraqn.load_events_for_detection import (
    prepare_detection_stream, init_processing, init_processing_wRotation,
    print_error_plots, get_all_relevant_stations, reevaluate_detections,
    multiplot_detection)
from robustraqn.spectral_tools import (Noise_model,
                                       get_updated_inventory_with_noise_models)
from robustraqn.templates_creation import create_template_objects


#@processify
def run_day_detection(tribe, date, ispaq, selectedStations,
                      parallel=False, cores=1,
                      remove_response=False, inv=Inventory(),
                      noise_balancing=False, balance_power_coefficient=2,
                      trig_int=0, threshold=10, min_chans=10, multiplot=False,
                      day_st=Stream(), check_array_misdetections=False,
                      short_tribe=Tribe(), write_party=False,
                      detection_path='Detections',
                      redetection_path='ReDetections',
                      return_stream=True, dump_stream_to_disk=False):
    """
    Function to run reading, initial processing, detection etc. on one day.
    """
    # Keep user's data safe
    tribe = tribe.copy()
    short_tribe = short_tribe.copy()
    #Set the path to the folders with continuous data:
    archive_path = '/data/seismo-wav/SLARCHIVE'
    # archive_path2 = '/data/seismo-wav/EIDA/archive'
    client = Client(archive_path)
    # client2 = Client(archive_path2)
    n_templates_per_run = 20
    n_templates = len(tribe)
    n_runs = math.ceil(n_templates / n_templates_per_run)


    starttime = UTCDateTime(pd.to_datetime(date))
    starttime_req = starttime - 15*60
    endtime = starttime + 60*60*24
    endtime_req = endtime + 15*60
    current_day_str = date.strftime('%Y-%m-%d')
    
    # keep input safe:
    day_st = day_st.copy()
    if len(day_st) == 0:
        # # Differentiate between Echo and Delta (deadlocks here otherwise)
        # req_parallel = False
        # if int(platform.release()[0]) >= 4 and parallel:
        #     req_parallel = parallel
        # Create a smart request, i.e.: request only recordings that match
        # the quality metrics criteria and that best match the priorities.
        bulk, day_stats = create_bulk_request(
            starttime_req, endtime_req, stats=ispaq,
            parallel=parallel, cores=cores,
            stations=selectedStations, location_priority=['00','10',''],
            band_priority=['B','H','S','E','N'], instrument_priority=['H'],
            components=['Z','N','E','1','2'],
            min_availability=0.8, max_cross_talk=1,
            max_spikes=1000, max_glitches=1000, max_num_gaps=500,
            max_num_overlaps=1000, max_max_overlap=86400,
            max_dead_channel_lin=3, require_alive_channel_gsn=True,
            max_pct_below_nlnm=50, max_pct_above_nhnm=50,
            min_sample_unique=150, max_abs_sample_mean=1e7,
            min_sample_rms=1e-6, max_sample_rms=1e8,
            max_sample_median=1e6, min_abs_sample_average=(1, 1e-9),
            require_clock_lock=False, max_suspect_time_tag=86400)
        if not bulk:
            Logger.warning('No waveforms requested for %s - %s',
                            str(starttime)[0:19], str(endtime)[0:19])
            if not return_stream and dump_stream_to_disk:
                return
            else:
                return [Party(), Stream()]

        # Read in continuous data and prepare for processing
        day_st = get_waveforms_bulk(client, bulk, parallel=parallel,
                                    cores=cores)
            
        Logger.info('Successfully read in %s traces for bulk request of %s'
                    + ' NSLC-objects for %s - %s.', len(day_st), len(bulk),
                    str(starttime)[0:19], str(endtime)[0:19])
        day_st = prepare_detection_stream(
            day_st, tribe, parallel=parallel, cores=cores, try_despike=False,
            ispaq=day_stats)
        # daily_plot(day_st, year, month, day, data_unit="counts", suffix='')
        
        # Do initial processing (rotation, stats normalization, merging)
        # by parallelization across three-component seismogram sets.
        day_st = init_processing_wRotation(
            day_st, starttime=starttime, endtime=endtime,
            remove_response=remove_response, inv=inv,
            parallel=parallel, cores=cores, min_segment_length_s=10,
            max_sample_rate_diff=1, skip_interp_sample_rate_smaller=1e-7,
            interpolation_method='lanczos', skip_check_sampling_rates=
            [20, 40, 50, 66, 75, 100, 500], sta_translation_file=
            "station_code_translation.txt", taper_fraction=0.005,
            detrend_type='simple', downsampled_max_rate=None,
            std_network_code="NS", std_location_code="00",
            std_channel_prefix="BH", noise_balancing=noise_balancing,
            balance_power_coefficient=balance_power_coefficient)
        
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
    
    #daily_plot(day_st, year, month, day, data_unit="counts",
    #           suffix='resp_removed')            
    # If there is no data for the day, then continue on next day.
    if not day_st.traces:
        Logger.warning('No data for detection on %s, continuing' +
                        ' with next day.', current_day_str)
        if not return_stream and dump_stream_to_disk:
            return
        else:
            return [Party(), Stream()]

    # Start the detection algorithm on all traces that are available.
    detections = []
    Logger.info('Start match_filter detection on %s with up to %s cores.',
                current_day_str, str(cores))
    try:
        party = tribe.detect(
            stream=day_st, threshold=threshold, trig_int=3.0,
            threshold_type='MAD', overlap='calculate', plot=False,
            plotDir='DetectionPlots', daylong=True, fill_gaps=True,
            ignore_bad_data=False, ignore_length=True, 
            parallel_process=parallel, cores=cores,
            # concurrency='multiprocess', xcorr_func='time_domain',
            xcorr_func='fftw', concurrency='multithread',
            #parallel_process=False, #concurrency=None,
            group_size=n_templates_per_run, full_peaks=False,
            save_progress=False, process_cores=cores, spike_test=False)
            # xcorr_func='fftw', concurrency=None, cores=1,
            # xcorr_func=None, concurrency=None, cores=1,
            # xcorr_func=None, concurrency='multiprocess', cores=cores,
        #party = Party().read()
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
    Logger.info('Got a party of families of detections!')
    
    # Decluster detection and save them to files
    # metric='avg_cor' isn't optimal when one detection may only be good on
    # very few channels - i.e., allowing higher CC than any detection made on
    # many channels
    party = party.decluster(trig_int=trig_int, timing='detect',
                            metric='thresh_exc', min_chans=min_chans,
                            absolute_values=True)
    
    if not party:
        Logger.warning('Party of families of detections is empty')
        return_st = Stream()
        if return_stream:
            return_st = day_st
        Logger.info('Size of party is: %s', len(pickle.dumps(party)))
        Logger.info('Size of return_st is: %s', len(pickle.dumps(return_st)))
        if not return_stream and dump_stream_to_disk:
            return
        else:
            return [party, return_st]

    if write_party:
        detection_file_name = os.path.join(detection_path,
                                           'UniqueDet' + current_day_str)
        party.write(detection_file_name, format='tar', overwrite=True)
        party.write(detection_file_name + '.csv', format='csv',
                    overwrite=True)

    # Check for erroneous detections of real signals (mostly caused by smaller
    # seismic events near one of the arrays). Solution: check whether templates
    # with shorter length increase detection value - if not; it's not a
    # desriable detection.
    if check_array_misdetections:
        if len(short_tribe) < len(tribe):
            Logger.error('Missing short templates for detection-reevaluation.')
        else:    
            party = reevaluate_detections(
                party, short_tribe, stream=day_st, threshold=threshold * 0.6,
                trig_int=40.0, threshold_type='MAD', overlap='calculate',
                plot=False, plotDir='ReDetectionPlots', fill_gaps=True,
                ignore_bad_data=False, daylong=True, ignore_length=True,
                concurrency='multiprocess', parallel_process=parallel,
                cores=cores, xcorr_func='time_domain', min_chans=min_chans,
                group_size=n_templates_per_run, process_cores=cores,
                time_difference_threshold=12, detect_value_allowed_error=60,
                return_party_with_short_templates=True)
            if not party:
                Logger.warning('Party of families of detections is empty')
                return_st = Stream()
                if return_stream:
                    return_st = day_st
                if not return_stream and dump_stream_to_disk:
                    return
                else:
                    return [party, return_st]
            if write_party:
                detection_file_name = os.path.join(
                    redetection_path, 'UniqueDet_short_' + current_day_str)
                party.write(detection_file_name, format='tar', overwrite=True)
                party.write(detection_file_name + '.csv', format='csv',
                            overwrite=True)

    if multiplot:
        multiplot_detection(party, tribe, day_st, out_folder='DetectionPlots')
    if dump_stream_to_disk:
        pickle.dump(day_st, open('tmp_st.pickle', "wb" ) , protocol=-1)
        pickle.dump(party, open('tmp_party.pickle', "wb" ) , protocol=-1)
    gc.collect()
    # n = gc.collect()
    # Logger.info("Number of unreachable objects collected by GC: ", n)
    # Logger.info("Uncollectable garbage: ", gc.garbage)
    
    return_st = Stream()
    if return_stream:
        return_st = day_st
    Logger.info('Size of party is: %s', len(pickle.dumps(party)))
    Logger.info('Size of return_st is: %s', len(pickle.dumps(return_st)))
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

    #selectedStations=['BER','ASK','KMY','HYA','FOO','BLS5','STAV','SNART',
    #                  'MOL','LRW','BIGH','ESK']
    selectedStations = ['ASK','BER','BLS5','DOMB','EKO1','FOO','HOMB','HYA','KMY',
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
    # selectedStations  = ['ASK', 'BLS5', 'KMY', 'ODD1', 'NAO01', 'ESK', 'EDI', 'KPL']
    # selectedStations  = ['ASK', 'BER', 'BLS5', 'STAV', 'NAO001', 'ESK', 'EKB2']
    selectedStations  = ['EKB10', 'EKR1', 'EKB']
                        # need   need    need   need   need
    # without-KESW, 2018-event will not detect itself
                        # 'EKB1','EKB2','EKB3','EKB4','EKB5','EKB6','EKB7',
                        # 'EKB8','EKB9','EKB10','EKR1','EKR2','EKR3','EKR4','EKR5',
                        # 'EKR6','EKR7','EKR8','EKR9','EKR10',]
    # 'SOFL','OSL', 'MUD',
    relevantStations = get_all_relevant_stations(
        selectedStations, sta_translation_file="station_code_translation.txt")
    
    seisanREApath = '../SeisanEvents/'
    seisanWAVpath = '../SeisanEvents/'
    
    sfiles = glob.glob(os.path.join(seisanREApath, '*L.S??????'))
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

    invFile = '~/Documents2/ArrayWork/Inventory/NorSea_inventory.xml'
    # invFile = '~/Documents2/ArrayWork/Inventory/NorSea_inventory.dataless_seed'
    # inv = read_inventory(os.path.expanduser(invFile))
    inv = get_updated_inventory_with_noise_models(
        os.path.expanduser(invFile),
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
            sfiles, relevantStations, template_length=120,
            lowcut=0.2, highcut=9.9, min_snr=3, prepick=0.5, samp_rate=20,
            min_n_traces=8, seisanWAVpath=seisanWAVpath, inv=inv,
            remove_response=True, noise_balancing=noise_balancing, 
            balance_power_coefficient=balance_power_coefficient,
            parallel=parallel, cores=cores, write_out=True,
            make_pretty_plot=False)
        Logger.info('Created new set of templates.')
        
        short_tribe = Tribe()
        if check_array_misdetections:
            short_tribe = create_template_objects(
                sfiles, relevantStations, template_length=10,
                lowcut=0.2, highcut=9.9, min_snr=3, prepick=0.5, samp_rate=20,
                min_n_traces=8, seisanWAVpath=seisanWAVpath, inv=inv,
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
    #                          stations=relevantStations, startyear=startday.year,
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
                stations=relevantStations, startyear=current_year,
                endyear=current_year, ispaq_prefixes=['all'],
                ispaq_suffixes=['simpleMetrics','PSDMetrics'],
                file_type = 'parquet')

        [party, day_st] = run_day_detection(
            tribe=tribe, date=date, ispaq=ispaq,
            selectedStations=relevantStations, remove_response=True, inv=inv,
            parallel=parallel, cores=cores, noise_balancing=noise_balancing,
            balance_power_coefficient=balance_power_coefficient,
            threshold=12, check_array_misdetections=check_array_misdetections,
            trig_int=40, short_tribe=short_tribe, multiplot=True,
            write_party=True, min_chans=1)

# %%
# [party, day_st] = run_day_detection(
#     tribe=tribe, date=date, ispaq=ispaq, selectedStations=relevantStations,
#     remove_response=True, inv=inv, parallel=False, cores=cores,
#     noise_balancing=noise_balancing, trig_int=40, threshold=10,
#     balance_power_coefficient=balance_power_coefficient,
#     check_array_misdetections=check_array_misdetections,
#     short_tribe=short_tribe, multiplot=True, write_party=True)

# party = Party().read('Detections_MAD9/UniqueDet2007-06-21.tgz')
    # if check_array_misdetections:
    # if len(short_tribe) < len(tribe):
    #     Logger.error('Missing short templates for detection-reevaluation.')
    # else:    
    #         party = reevaluate_detections(
    #             party, short_tribe, stream=day_st, threshold_type='MAD',
    #             threshold=threshold-2, trig_int=3.0, overlap='calculate',
    #             plot=False, plotDir='ReDetectionPlots', fill_gaps=True,
    #             ignore_bad_data=False, daylong=True, ignore_length=True,
    #             concurrency='multiprocess', parallel_process=parallel,
    #             cores=cores, xcorr_func='time_domain', min_chans=min_chans,
    #             group_size=n_templates_per_run, process_cores=cores,
    #             time_difference_threshold=8, detect_value_allowed_error=30,
    #             return_party_with_short_templates=True)


# %%
