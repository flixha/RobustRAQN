#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Felix Halpaap
"""

# %%
def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


import os, glob, gc, math, matplotlib
import warnings

from os import times
import pandas as pd
if not run_from_ipython:
    matplotlib.use('Agg')  # to plot figures directly for print to file
from importlib import reload
import numpy as np
import pickle
import hashlib
from collections import Counter
from joblib import Parallel, delayed, parallel_backend

from timeit import default_timer
import logging


from obspy import read_inventory
#from obspy.core.event import Event, Origin, Catalog
# from obspy.core.stream import Stream
from obspy.core.inventory.inventory import Inventory
from obspy import UTCDateTime
from obspy.core.util.attribdict import AttribDict
from obspy.core.util.deprecation_helpers import ObsPyDeprecationWarning

from eqcorrscan.core.match_filter import Template, Tribe
from eqcorrscan.core.match_filter.party import Party, Family

# import quality_metrics, spectral_tools, load_events
# reload(quality_metrics)
# reload(load_events)
from robustraqn.core.event_detection import run_day_detection
from robustraqn.utils.quality_metrics import (
    create_bulk_request, get_waveforms_bulk, read_ispaq_stats)
from robustraqn.core.load_events import get_all_relevant_stations
from robustraqn.core.templates_creation import create_template_objects
from robustraqn.utils.spectral_tools import (
    Noise_model, get_updated_inventory_with_noise_models)
from robustraqn.utils.obspy import _quick_copy_stream
from robustraqn.utils.processify import processify
from robustraqn.utils.fancy_processify import fancy_processify
from robustraqn.obspy.clients.filesystem.sds import Client
from robustraqn.obspy.core import Stream, Trace

Logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
# Set up some warnings-filtering
warnings.filterwarnings("ignore", category=ObsPyDeprecationWarning)

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