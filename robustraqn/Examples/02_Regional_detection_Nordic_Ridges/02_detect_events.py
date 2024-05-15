# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 17:23:45 2017
Main detection program to detect events with template matching in a large
dataset. Loops across available days. Takes data quality metrics, seismic
arrays, and station noise models into account. Supports three main correlation
backend packages: EQcorrscan (best: fftw-based), fast_matched_filter (best:
time-domain with CUDA on Nvidia GPUs), and fmf2 (best: time-domain on CPUs with
AVX2/AVX512 optimizations, td on Nvidia / AMD GPUs with hipSYCL compiler.)

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

import os, glob, matplotlib
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

# import quality_metrics, spectral_tools, load_events_for_detection
# reload(quality_metrics)
# reload(load_events_for_detection)
from robustraqn.core.templates_creation import (
    create_template_objects, _shorten_tribe_streams)
from robustraqn.core.event_detection import run_day_detection
from robustraqn.utils.quality_metrics import read_ispaq_stats
from robustraqn.core.load_events import (
    get_all_relevant_stations, read_seisan_database)
from robustraqn.utils.spectral_tools import (
    Noise_model, get_updated_inventory_with_noise_models)
from robustraqn.utils.bayesloc import update_cat_from_bayesloc
Logger.info('Module import done')


# %%

if __name__ == "__main__":
    parallel_backend('loky')

    templateFolder = 'Templates'
    parallel = True
    cores = SLURM_CPUS or 40
    remove_response = True
    output = 'VEL'
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
    bayesloc_event_solutions = None
    bayesloc_path = [
        '../Relocation/Bayesloc/Ridge_INTEU_09b_continental',
        '../Relocation/Bayesloc/Ridge_INTEU_09a_oceanic_04',
        ]
    custom_epoch = UTCDateTime(1960, 1, 1, 0, 0, 0)

    sta_translation_file = "station_code_translation.txt"
    ispaq_folder = '~/repos/ispaq/WrapperScripts/Parquet_database/csv_parquet'

    inv_file = '~/Documents2/ArrayWork/Inventory/NorSea_inventory.xml'
    inv = get_updated_inventory_with_noise_models(
        inv_file=os.path.expanduser(inv_file),
        pdf_dir=os.path.expanduser('~/repos/ispaq/WrapperScripts/PDFs/'),
        check_existing=True,
        outfile=os.path.expanduser(
            '~/Documents2/ArrayWork/Inventory/inv.pickle'))

    # Path definitions for servers and clusters
    working_on_cluster = True
    xcorr_func = 'fmf2'
    arch = 'precise'
    if GPUtil.getAvailable():
        xcorr_func = 'fmf'
        arch = 'GPU'
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
            catalog, bayesloc_event_solutions, custom_epoch=custom_epoch)

    # Define time range based on parallel execution in Slurm array job
    try:
        SLURM_ARRAY_TASK_ID = int(os.environ['SLURM_ARRAY_TASK_ID'])
        SLURM_ARRAY_TASK_COUNT = max([
            int(os.environ['SLURM_ARRAY_TASK_COUNT']),
            int(os.environ['SLURM_ARRAY_TASK_MAX'])])
        # split into equal batches
        date_ranges = pd.date_range(startday.datetime, endday.datetime,
                                    periods=SLURM_ARRAY_TASK_COUNT+1)
        start_dstamp = date_ranges[SLURM_ARRAY_TASK_ID]
        startday = UTCDateTime(
            start_dstamp.year, start_dstamp.month, start_dstamp.day)
        end_dstamp = date_ranges[SLURM_ARRAY_TASK_ID+1]
        endday = UTCDateTime(end_dstamp.year, end_dstamp.month, end_dstamp.day)
        Logger.info('This is SLURM array task %s (task count: %s) for the time'
                    ' period %s - %s', SLURM_ARRAY_TASK_ID,
                    SLURM_ARRAY_TASK_COUNT, str(startday), str(endday))
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
            add_array_picks=add_array_picks,
            add_large_aperture_array_picks=add_large_aperture_array_picks,
            min_array_distance_factor=min_array_distance_factor,
            sta_translation_file=sta_translation_file,
            max_horizontal_error_km=700, max_depth_error_km=200,
            max_time_error_s=20, nordic_format='NEW', unused_kwargs=True,
            ispaq=ispaq_full, min_availability=0.005, max_pct_below_nlnm=50,
            max_pct_above_nhnm=70, location_priority=['00', '10', ''],
            band_priority=['B', 'H', 'S', 'E', 'N'], instrument_priority=['H'],
            components=['Z', 'N', 'E', '1', '2'], require_clock_lock=False,
            bayesloc_event_solutions=bayesloc_path, custom_epoch=custom_epoch,
            wavetool_path=wavetool_path)
        Logger.info('Created new set of templates.')
    else:
        Logger.info('Starting template reading')
        tribe = Tribe().read(
            # 'TemplateObjects/Templates_min13tr_balNoise_agc_14472.tgz',
            # 'mohns_cluster_tribe_17_displacement_balanced.tgz',
            'mohns_cluster_tribe.tgz',
            cores=cores)
        Logger.info('Tribe archive readily read in')

    short_tribe = Tribe()
    short_tribe2 = Tribe()
    if check_array_misdetections:
        Logger.info('Creating tribe of %s shortened templates', len(tribe))
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
            short_tribe=short_tribe, short_tribe2=short_tribe2,
            selected_stations=relevant_stations,
            remove_response=remove_response, output=output,
            inv=inv, parallel=parallel, cores=cores, io_cores=cores,
            # xcorr_func='fftw', concurrency='concurrent',
            # xcorr_func='time_domain', concurrency='multiprocess',
            xcorr_func=xcorr_func, concurrency='concurrent', arch=arch,
            # fftw: "fmf2: 1570
            n_templates_per_run=1570, noise_balancing=noise_balancing,
            balance_power_coefficient=balance_power_coefficient,
            apply_agc=apply_agc, suppress_arraywide_steps=True,
            check_array_misdetections=check_array_misdetections,
            detect_value_allowed_reduction=4, time_difference_threshold=4,
            threshold=11, re_eval_thresh_factor=0.6, min_chans=3, # 11
            decluster_metric='thresh_exc', hypocentral_separation=250,
            absolute_values=True, trig_int=30,
            write_party=True, detection_path='Detections_MAD11_01',
            redetection_path='ReDetections_MAD11_01', multiplot=False,
            day_hash_file='dates_hash_list.csv', use_weights=use_weights,
            weight_current_noise_level=weight_current_noise_level,
            copy_data=True, sta_translation_file=sta_translation_file,
            location_priority=['*'])
    Logger.info('Job completed successfully.')


