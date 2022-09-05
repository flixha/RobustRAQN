# !/usr/bin/env python3
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

import logging
Logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(name)40s:%(lineno)s\t%(funcName)20s()\t%(levelname)s\t%(message)s")
"[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
Logger.info('Start module import')
import faulthandler; faulthandler.enable()

import os, glob, gc, math, calendar, matplotlib, platform, sys
try:
    SLURM_CPUS = (int(os.environ['SLURM_CPUS_PER_TASK']) *
                int(os.environ['SLURM_JOB_NUM_NODES']))
    os.environ["OMP_NUM_THREADS"] = str(SLURM_CPUS) # export OMP_NUM_THREADS=1
except KeyError as e:
    Logger.error('Could not retrieve number of SLURM CPUS per task, %s', e)
    SLURM_CPUS = None

from numpy.core.numeric import True_

from os import times
import pandas as pd
if not run_from_ipython:
    matplotlib.use('Agg') # to plot figures directly for print to file
from importlib import reload
import numpy as np
import pickle
from joblib import parallel_backend
from timeit import default_timer
import GPUtil

from obspy import read_inventory
from obspy.core.stream import Stream
from obspy.core.inventory.inventory import Inventory
from obspy import UTCDateTime
from datetime import datetime
from obspy.io.mseed import InternalMSEEDWarning
from robustraqn.obspy.clients.filesystem.sds import Client

from obsplus.events.validate import attach_all_resource_ids

import warnings
warnings.filterwarnings("ignore", category=InternalMSEEDWarning)

from eqcorrscan.core.match_filter import Template, Tribe, MatchFilterError
from eqcorrscan.core.match_filter.party import Party
from eqcorrscan.utils.plotting import detection_multiplot

from robustraqn.quality_metrics import (
    create_bulk_request, get_waveforms_bulk, read_ispaq_stats,
    get_parallel_waveform_client)
from robustraqn.load_events_for_detection import (
    prepare_detection_stream, init_processing, init_processing_wRotation,
    print_error_plots, get_all_relevant_stations, reevaluate_detections,
    multiplot_detection, read_seisan_database)
from robustraqn.spectral_tools import (Noise_model, attach_noise_models,
                                       get_updated_inventory_with_noise_models)
from robustraqn.templates_creation import (
    create_template_objects, _shorten_tribe_streams)
from robustraqn.event_detection import run_day_detection
from robustraqn.bayesloc_utils import update_cat_from_bayesloc
Logger.info('Module import done')


# %%

if __name__ == "__main__":
    parallel_backend('loky')

    templateFolder = 'Templates'
    parallel = True
    cores = SLURM_CPUS or 40
    remove_response = True
    noise_balancing = True
    make_templates = False
    add_array_picks = True
    add_large_aperture_array_picks = True
    check_array_misdetections = True
    balance_power_coefficient = 2
    samp_rate = 20
    lowcut = 1.0
    highcut = 9.9
    prepick = 1.0
    template_length_long = 90.0
    template_length_short = 10.0
    min_n_traces = 13
    min_snr = 3
    apply_agc = True
    use_weights = True
    weight_current_noise_level = False
    bayesloc_path = [
        '../Relocation/Bayesloc/Ridge_INTEU_06e_wRegionalEvents_min12stations',
        '../Relocation/Bayesloc/Ridge_INTEU_09b_continental',
        '../Relocation/Bayesloc/Ridge_INTEU_09a_oceanic',
        '../Relocation/Bayesloc/Ridge_INTEU_09a_oceanic_01b',
        '../Relocation/Bayesloc/Ridge_INTEU_09a_oceanic_02',
        '../Relocation/Bayesloc/Ridge_INTEU_09a_oceanic_03',
        '../Relocation/Bayesloc/Ridge_INTEU_09a_oceanic_04',
        '../Relocation/Bayesloc/Ridge_INTEU_09a_oceanic_05',
        '../Relocation/Bayesloc/Ridge_INTEU_09a_oceanic_06',
        '../Relocation/Bayesloc/Ridge_INTEU_09a_oceanic_07',
        '../Relocation/Bayesloc/Ridge_INTEU_09a_oceanic_08b',
        '../Relocation/Bayesloc/Ridge_INTEU_09a_oceanic_09',
        '../Relocation/Bayesloc/Ridge_INTEU_09a_oceanic_10b',
        '../Relocation/Bayesloc/Ridge_INTEU_09a_oceanic_11',
        '../Relocation/Bayesloc/Ridge_INTEU_09a_oceanic_12d',
        ]


    sta_translation_file = os.path.expanduser(
        "~/Documents2/ArrayWork/Inventory/station_code_translation.txt")
    ispaq_folder = '~/repos/ispaq/WrapperScripts/Parquet_database/csv_parquet'

    inv_file = '~/Documents2/ArrayWork/Inventory/NorSea_inventory.xml'
    inv = get_updated_inventory_with_noise_models(
        inv_file=os.path.expanduser(inv_file),
        pdf_dir=os.path.expanduser('~/repos/ispaq/WrapperScripts/PDFs/'),
        check_existing=True,
        outfile=os.path.expanduser(
            '~/Documents2/ArrayWork/Inventory/inv.pickle'))

    working_on_cluster = True
    xcorr_func = 'fmf'
    if GPUtil.getAvailable():
        xcorr_func = 'fmf'
    seisan_rea_path = '../Seisan/INTEU'
    archive_path = '/cluster/shared/NNSN/SLARCHIVE'
    if not os.path.exists(archive_path):
        archive_path = '/data/seismo-wav/SLARCHIVE'
        working_on_cluster = False
        os.environ["NUMEXPR_MAX_THREADS"] = str(4)
        client2 = Client('/data/seismo-wav/EIDA/NO/archive')
        seisan_wav_path = '/data/seismo-wav/NNSN_'
        wavetool_path = '/home/felix/Software/SEISANrick/PRO/linux64/wavetool'
    else:
        seisan_wav_path = '/cluster/shared/NNSN/SEI/WAV/NNSN_'
        wavetool_path = (
            '/cluster/home/fha053/repos/SEI/LIN64/PRO/linux64/wavetool')
    client = Client(archive_path)
    clients = [client]
    if not working_on_cluster:
        clients.append(client2)

    # Read in detection stations
    det_sta_f = open('stations_selection.dat', "r+")
    selected_stations = [line.strip() for line in det_sta_f.readlines()]
    det_sta_f.close()
    Logger.info('Reading in metadata and data')
    relevant_stations = get_all_relevant_stations(
        selected_stations, sta_translation_file=sta_translation_file)

    # FULL:
    sfiles = glob.glob(os.path.join(seisan_rea_path, '198[7-9]/??/*.S??????'))
    sfiles += glob.glob(os.path.join(seisan_rea_path, '199?/??/*.S??????'))
    sfiles += glob.glob(os.path.join(seisan_rea_path, '20??/??/*.S??????'))

    sfiles.sort(key = lambda x: x[-6:] + x[-19:-9], reverse=True)

    startday = UTCDateTime(2018,1,1,0,0,0)
    endday = UTCDateTime(2019,12,31,0,0,0)

    catalog = None
    if bayesloc_event_solutions:
        sfiles = ''
        catalog = read_seisan_database(
            seisan_rea_path, cores=cores, nordic_format='NEW',
            starttime=UTCDateTime(1988,1,1,0,0,0),
            )
        Logger.info('Updating catalog from bayesloc solutions')
        catalog = update_cat_from_bayesloc(
            catalog, bayesloc_event_solutions,
            custom_epoch=UTCDateTime(1960, 1, 1, 0, 0, 0))

    # Define time range based on parallel execution in Slurm array job
    try:
        SLURM_ARRAY_TASK_ID = int(os.environ['SLURM_ARRAY_TASK_ID'])
        SLURM_ARRAY_TASK_COUNT = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
        # set starttime depending on ID of slurm array task
        # first task is 1
        month = startday.month + 1 * SLURM_ARRAY_TASK_ID
        # increment year if required
        year = startday.year + month // 12
        # change month if required
        month = month % 12 + (month // 12) * 1
        startday = UTCDateTime(year, month, 1, 0, 0, 0)
        last_day_of_month = calendar.monthrange(year, month)[1]
        new_endday = UTCDateTime(year, month, last_day_of_month, 23, 59, 59.99)
        if SLURM_ARRAY_TASK_ID == SLURM_ARRAY_TASK_COUNT:
            if endday > new_endday:
                Logger.warning('Not enough SLURM tasks to complete time range')
        endday = new_endday
        Logger.info('This is SLURM array task %s (task count: %s) for the time'
                    + 'period %s - %s', str(SLURM_ARRAY_TASK_ID),
                    str(SLURM_ARRAY_TASK_COUNT), str(startday), str(endday))
    except Exception as e:
        Logger.info('This is not a SLURM array task.')
        pass


    # %%
    # Read templates from file
    ispaq_full = None
    if make_templates:
        if bayesloc_event_solutions:
            templates_startday = catalog[0].origins[0].time
            templates_endday = catalog[-1].origins[0].time
        else:
            templates_startday = UTCDateTime(
                sfiles[0][-6:] + os.path.split(sfiles[0])[-1][0:2])
            templates_endday = UTCDateTime(
                sfiles[-1][-6:] + os.path.split(sfiles[-1])[-1][0:2])
        startyear = min(startday.year, templates_startday.year)
        endyear = max(endday.year, templates_endday.year)
        ispaq_full = read_ispaq_stats(folder=ispaq_folder,
            stations=relevant_stations, startyear=startyear,
            endyear=endyear, ispaq_prefixes=['all'],
            ispaq_suffixes=['simpleMetrics','PSDMetrics'],
            file_type='parquet')
        Logger.info('Creating new templates')
        tribe, _ = create_template_objects(
            sfiles=sfiles, catalog=catalog, selected_stations=relevant_stations,
            template_length=template_length_long,
            lowcut=lowcut, highcut=highcut, min_snr=min_snr, prepick=prepick,
            samp_rate=samp_rate, min_n_traces=min_n_traces,
            seisan_wav_path=seisan_wav_path, inv=inv, clients=clients,
            remove_response=remove_response, noise_balancing=noise_balancing,
            balance_power_coefficient=balance_power_coefficient,
            apply_agc=apply_agc, make_pretty_plot=False, normalize_NSLC=True,
            parallel=parallel, cores=cores, write_out=True,
            max_events_per_file=200, forbidden_phase_hints=['pmax'],
            add_array_picks=add_array_picks, min_array_distance_factor=10,
            add_large_aperture_array_picks=add_large_aperture_array_picks,
            sta_translation_file=sta_translation_file,
            max_horizontal_error_km=700, max_depth_error_km=200,
            max_time_error_s=20, nordic_format='NEW', unused_kwargs=True,
            ispaq=ispaq_full, min_availability=0.005, max_pct_below_nlnm=50,
            max_pct_above_nhnm=70, location_priority=['00', '10', ''],
            band_priority=['B', 'H', 'S', 'E', 'N'], instrument_priority=['H'],
            components=['Z', 'N', 'E', '1', '2'], require_clock_lock=False,
            bayesloc_event_solutions=bayesloc_path,
            wavetool_path=wavetool_path)
        Logger.info('Created new set of templates.')
    else:
        Logger.info('Starting template reading')
        tribe = Tribe().read(
            'TemplateObjects/Templates_min13tr_balNoise_agc_14472.tgz',
            cores=cores)
        # May need to fix all resource IDs when read in in parallel
        # assert rid.get_referred_object() is rid_to_object[rid.id]
        # for template in tribe:
        #    attach_all_resource_ids(template.event)
        Logger.info('Tribe archive readily read in')

    short_tribe = Tribe()
    if check_array_misdetections:
        short_tribe = _shorten_tribe_streams(
            tribe, tribe_len_pct=0.2, max_tribe_len=None,
            min_n_traces=min_n_traces, write_out=False,
            make_pretty_plot=False, prefix='short_',
            noise_balancing=noise_balancing, apply_agc=apply_agc)
        Logger.info('Short-tribe archive readily read in')


    # %% ### LOOP OVER DAYS ########
    day_st = Stream()
    dates = pd.date_range(startday.datetime, endday.datetime, freq='1D')
    # For each day, read in data and run detection from templates
    current_year = None
    for date in dates:
        # Load in Mustang-like ISPAQ stats for the whole year
        if not date.year == current_year:
            if ispaq_full is not None:  # No need to reload if already used for templates
                ispaq = ispaq_full
            else:
                current_year = date.year
                ispaq = read_ispaq_stats(
                    folder=ispaq_folder,
                    stations=relevant_stations, startyear=current_year,
                    endyear=current_year, ispaq_prefixes=['all'],
                    ispaq_suffixes=['simpleMetrics','PSDMetrics'],
                    file_type='parquet', starttime=startday, endtime=endday)

        # n_templates_per_run = 1000: 6 /16 GB GPU RAm used
        [party, day_st] = run_day_detection(
            clients=clients, tribe=tribe, date=date, ispaq=ispaq, 
            selected_stations=relevant_stations,
            remove_response=remove_response,
            inv=inv, parallel=parallel, cores=cores, io_cores=cores,
            xcorr_func=xcorr_func, concurrency='concurrent', arch='GPU',
            n_templates_per_run=2500, noise_balancing=noise_balancing,
            balance_power_coefficient=balance_power_coefficient,
            apply_agc=apply_agc,
            check_array_misdetections=check_array_misdetections,
            threshold=10, re_eval_thresh_factor=0.65, min_chans=3,
            decluster_metric='thresh_exc', hypocentral_separation=200,
            absolute_values=True, trig_int=30, short_tribe=short_tribe,
            write_party=True, detection_path='Detections_MAD11_01',
            redetection_path='ReDetections_MAD11_01', multiplot=False,
            day_hash_file='dates_hash_list.csv', use_weights=use_weights,
            weight_current_noise_level=weight_current_noise_level,
            copy_data=False, sta_translation_file=sta_translation_file,
            location_priority=['*'])


