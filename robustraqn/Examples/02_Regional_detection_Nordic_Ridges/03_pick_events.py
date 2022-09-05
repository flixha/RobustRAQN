#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2021

@author: felix
"""

# %%

def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

import matplotlib
matplotlib.use('TKAgg')

import logging
Logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s\t%(name)40s:%(lineno)s\t%(funcName)20s()\t%(levelname)s\t%(message)s")
"[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
Logger.info('Start module import')

EQCS_logger = logging.getLogger('EQcorrscan')
EQCS_logger.setLevel(logging.ERROR)

import sys
sys.settrace
import os, glob, math, calendar, platform

try:
    SLURM_CPUS = (int(os.environ['SLURM_CPUS_PER_TASK']) *
                int(os.environ['SLURM_JOB_NUM_NODES']))
    os.environ["OMP_NUM_THREADS"] = str(SLURM_CPUS) # export OMP_NUM_THREADS=1
except KeyError as e:
    Logger.error('Could not retrieve number of SLURM CPUS per task, %s', e)
    SLURM_CPUS = None

import numpy as np
from joblib import parallel_backend
import pandas as pd
from importlib import reload
import statistics as stats
import GPUtil

from obspy import read_events
from obspy.core.event import Catalog, Event, Origin
from obspy.core.stream import Stream
from obspy.core.utcdatetime import UTCDateTime
from obspy.io.mseed import InternalMSEEDError
from obspy import read as obspyread
from obspy import read_inventory, Inventory
from obspy.clients.filesystem.sds import Client

from eqcorrscan.utils import pre_processing
from obspy.io.nordic.core import read_nordic, write_select, _write_nordic
from eqcorrscan.core import lag_calc
from eqcorrscan.core.lag_calc import LagCalcError
from eqcorrscan.core.match_filter import (read_detections, MatchFilterError,
    _spike_test, Template, Tribe, Party, Family)
from eqcorrscan.utils.catalog_utils import filter_picks
from eqcorrscan.utils.despike import median_filter
from eqcorrscan.utils.correlate import CorrelationError
from eqcorrscan.utils.clustering import extract_detections

sys.path.insert(1, os.path.expanduser("~/Documents2/NorthSea/Elgin/Detection"))
from robustraqn.quality_metrics import (
    create_bulk_request, get_waveforms_bulk, read_ispaq_stats)
from robustraqn.load_events_for_detection import (
    prepare_detection_stream, init_processing, init_processing_wRotation,
    print_error_plots, get_all_relevant_stations, normalize_NSLC_codes,
    reevaluate_detections, read_seisan_database)
from robustraqn.spectral_tools import (Noise_model,
                                       get_updated_inventory_with_noise_models)
from robustraqn.templates_creation import (
    create_template_objects, _shorten_tribe_streams)
from robustraqn.lag_calc_postprocessing import (
    check_duplicate_template_channels, postprocess_picked_events)
from robustraqn.detection_picking import pick_events_for_day
from robustraqn.processify import processify
from robustraqn.fancy_processify import fancy_processify
from robustraqn.seismic_array_tools import get_updated_stations_df
from robustraqn.bayesloc_utils import update_cat_from_bayesloc


# %% Now run the day-loop

if __name__ == "__main__":
    parallel_backend('loky')

    # Read in detection stations
    det_sta_f = open('stations_selection.dat', "r+")
    selected_stations = [line.strip() for line in det_sta_f.readlines()]
    det_sta_f.close()

    inv_file = '~/Documents2/ArrayWork/Inventory/NorSea_inventory.xml'
    inv_file = os.path.expanduser(inv_file)
    inv = get_updated_inventory_with_noise_models(
        os.path.expanduser(inv_file),  check_existing=True,
        pdf_dir='~/repos/ispaq/WrapperScripts/PDFs/',
        outfile=os.path.expanduser(
            '~/Documents2/ArrayWork/Inventory/inv.pickle'))
    stations_df = get_updated_stations_df(inv)

    working_on_cluster = True
    arch = 'CPU'
    xcorr_func = 'fmf'
    if GPUtil.getAvailable():
        arch = 'GPU'
    seisan_rea_path = '../Seisan/INTEU'
    archive_path = '/cluster/shared/NNSN/SLARCHIVE'
    if not os.path.exists(archive_path):
        archive_path = '/data/seismo-wav/SLARCHIVE'
        working_on_cluster = False
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

    template_path ='Templates'
    #template_path ='LagCalcTemplates'
    parallel = True
    cores = SLURM_CPUS or 40
    # det_folder = 'Detections_MAD10_01'
    det_folder = 'Detections_MAD11_01'

    make_templates = False
    add_array_picks = True
    add_large_aperture_array_picks = True
    remove_response = True
    noise_balancing = True
    balance_power_coefficient = 2
    check_array_misdetections = False
    apply_array_lag_calc = True
    write_party = True
    new_threshold = 10
    min_cc = 0.4
    min_cc_from_mean_cc_factor = 0.9 # 0.9
    n_templates_per_run = 2500
    min_snr = 3
    min_n_traces = 13
    min_det_chans = 15
    samp_rate = 20
    lowcut = 1.0
    highcut = 9.9
    prepick = 1.0
    template_length_long = 90.0
    template_length_short = 10.0
    only_request_detection_stations = True
    all_horiz = True
    all_vert = True
    apply_agc = True
    bayesloc_event_solutions = None
    bayesloc_path = (
        '../Relocation/Bayesloc/Ridge_INTEU_06e_wRegionalEvents_min12stations')

    sta_translation_file = os.path.expanduser(
        "~/Documents2/ArrayWork/Inventory/station_code_translation.txt")
    ispaq_folder = '~/repos/ispaq/WrapperScripts/Parquet_database/csv_parquet'

    relevant_stations = get_all_relevant_stations(
        selected_stations, sta_translation_file=sta_translation_file)

    seisan_rea_path = '../Seisan/INTEU'
    seisan_wav_path = '/data/seismo-wav/NNSN_'

    sfiles = glob.glob(os.path.join(seisan_rea_path, '198[7-9]/??/*.S??????'))
    sfiles += glob.glob(os.path.join(seisan_rea_path, '199?/??/*.S??????'))
    sfiles += glob.glob(os.path.join(seisan_rea_path, '20??/??/*.S??????'))

    sfiles = glob.glob(os.path.join(seisan_rea_path, '2018/05/07-1846-58R.S201805'))
    sfiles.sort(key = lambda x: x[-6:] + x[-19:-9])

    startday = UTCDateTime(2018,11,1,0,0,0)
    endday = UTCDateTime(2020,1,1,0,0,0)

    startday = UTCDateTime(2018,5,7,0,0,0)
    endday = UTCDateTime(2018,5,7,0,0)

    # Define time range based on parallel execution in Slurm array job
    try:
        SLURM_ARRAY_TASK_ID = int(os.environ['SLURM_ARRAY_TASK_ID'])
        SLURM_ARRAY_TASK_COUNT = max([
            int(os.environ['SLURM_ARRAY_TASK_COUNT']),
            int(os.environ['SLURM_ARRAY_TASK_MAX'])])
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
            max_events_per_file=200,
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

        short_tribe = Tribe()
        if check_array_misdetections:
            short_tribe = _shorten_tribe_streams(
                tribe, tribe_len_pct=0.2, max_tribe_len=template_length_short,
                min_n_traces=min_n_traces, write_out=False,
                make_pretty_plot=False, prefix='short_',
                noise_balancing=noise_balancing, apply_agc=apply_agc)
            Logger.info('Created new set of short templates.')
    else:
        Logger.info('Starting template reading')
        tribe = Tribe().read(
            'TemplateObjects/Templates_min13tr_balNoise_agc_14472.tgz',
            cores=cores)

        Logger.info('Tribe archive readily read in')
        short_tribe = Tribe()
        if check_array_misdetections:
            short_tribe = _shorten_tribe_streams(
                tribe, tribe_len_pct=0.2, max_tribe_len=None,
                min_n_traces=min_n_traces, write_out=False,
                make_pretty_plot=False, prefix='short_',
                noise_balancing=noise_balancing, apply_agc=apply_agc)
                #cores=4)
            Logger.info('Short-tribe archive readily read in')

    # Check templates for duplicate channels
    tribe = check_duplicate_template_channels(
        tribe, all_vert=all_vert, all_horiz=all_horiz)
    short_tribe = check_duplicate_template_channels(
        short_tribe, all_vert=all_vert, all_horiz=all_horiz)

    # Read in and process the daylong data
    dates = pd.date_range(startday.datetime, endday.datetime, freq='1D')
    # For each day, read in data and run detection from templates
    current_year = None


# %%

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
                file_type='parquet')
    export_catalog = pick_events_for_day(
        tribe=tribe, short_tribe=short_tribe, det_tribe=tribe,
        date=date, det_folder=det_folder, template_path=template_path,
        ispaq=ispaq, clients=clients, relevant_stations=relevant_stations,
        only_request_detection_stations=only_request_detection_stations,
        sta_translation_file=sta_translation_file, apply_agc=apply_agc,
        noise_balancing=noise_balancing, remove_response=remove_response,
        balance_power_coefficient=balance_power_coefficient,
        apply_array_lag_calc=apply_array_lag_calc, min_array_distance_factor=10,
        inv=inv, stations_df=stations_df, parallel=parallel, cores=cores,
        io_cores=min(cores, 16), ignore_cccsum_comparison=True,
        min_cc=min_cc, min_cc_from_mean_cc_factor=min_cc_from_mean_cc_factor,
        interpolate=False, use_new_resamp_method=True,
        do_not_rename_refracted_phases=True,
        all_horiz=all_horiz, all_vert=all_vert,
        check_array_misdetections=check_array_misdetections,
        xcorr_func=xcorr_func, arch=arch,
        write_party=write_party, new_threshold=new_threshold,
        n_templates_per_run=n_templates_per_run, extract_len=600,
        archives=[archive_path], request_fdsn=False,
        day_hash_file='dates_hash_list_picking.csv',
        min_det_chans=min_det_chans, sfile_path='Sfiles_01',
        operator='feha', evtype='R')


