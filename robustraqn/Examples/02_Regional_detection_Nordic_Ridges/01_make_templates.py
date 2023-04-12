
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
SLURM_MEM = 1
try:
    SLURM_CPUS = (int(os.environ['SLURM_CPUS_PER_TASK']) *
                int(os.environ['SLURM_JOB_NUM_NODES']))
    os.environ["OMP_NUM_THREADS"] = str(SLURM_CPUS) # export OMP_NUM_THREADS=1
except KeyError as e:
    Logger.error('Could not retrieve number of SLURM CPUS per task, %s', e)
    SLURM_CPUS = None

if not run_from_ipython:
    matplotlib.use('Agg') # to plot figures directly for print to file
from importlib import reload
import numpy as np
from joblib import parallel_backend

from obspy import UTCDateTime
from obspy.io.mseed import InternalMSEEDWarning
from robustraqn.obspy.clients.filesystem.sds import Client

import warnings
warnings.filterwarnings("ignore", category=InternalMSEEDWarning)

from robustraqn.core.load_events import (
    get_all_relevant_stations, read_seisan_database)
from robustraqn.core.templates_creation import (
    create_template_objects, _shorten_tribe_streams)
from robustraqn.utils.quality_metrics import read_ispaq_stats
from robustraqn.utils.spectral_tools import (
    Noise_model, attach_noise_models, get_updated_inventory_with_noise_models)
from robustraqn.utils.bayesloc import update_cat_from_bayesloc
Logger.info('Module import done')


# %%

if __name__ == "__main__":
    parallel_backend('loky')

    templateFolder = 'Templates'
    parallel = False
    thread_parallel = False
    total_cores = (SLURM_CPUS or 40)
    cores = total_cores
    remove_response = True
    noise_balancing = True
    make_templates = True
    add_array_picks = True
    add_large_aperture_array_picks = False
    balance_power_coefficient = 2
    samp_rate = 20
    lowcut = 1.0
    highcut = 9.9
    prepick = 1.0
    template_length_long = 90.0
    min_n_traces = 13
    min_snr = 3
    apply_agc = True
    use_weights = True
    weight_current_noise_level = True
    # List of bayesloc output folders from which to read corrected hypocetners and picks
    bayesloc_event_solutions = None
    bayesloc_path = [
        '~/Documents2/Ridge/Relocation/Bayesloc/Ridge_INTEU_09a_oceanic_04',
        '~/Documents2/Ridge/Relocation/Bayesloc/Ridge_INTEU_09b_continental',
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

    n_threads = 1
    if thread_parallel:
        n_threads = int(total_cores / cores)
    # Set thread limit explicitly
    try:
        os.environ["NUMEXPR_MAX_THREADS"] = str(n_threads)
    except:
        pass

    working_on_cluster = True
    xcorr_func = 'fmf'
    seisan_rea_path = 'Seisan/INTEU'
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

    # Read in detection stations
    det_sta_f = open('stations_selection.dat', "r+")
    selected_stations = [line.strip() for line in det_sta_f.readlines()]
    det_sta_f.close()

    Logger.info('Reading in metadata and data')
    relevant_stations = get_all_relevant_stations(
        selected_stations, sta_translation_file=sta_translation_file)

    # FULL dataset:
    sfiles = glob.glob(os.path.join(seisan_rea_path, '198[7-9]/??/*.S??????'))
    sfiles += glob.glob(os.path.join(seisan_rea_path, '199?/??/*.S??????'))
    sfiles += glob.glob(os.path.join(seisan_rea_path, '20??/??/*.S??????'))
    sfiles.sort(key = lambda x: x[-6:] + x[-19:-9], reverse=True)

    startday = UTCDateTime(1988,1,1,0,0,0)
    endday = UTCDateTime(2023,12,31,0,0,0)

    custom_epoch = UTCDateTime(1960, 1, 1, 0, 0, 0)
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
    task_id = None
    try:
        SLURM_ARRAY_TASK_ID = int(os.environ['SLURM_ARRAY_TASK_ID'])
        task_id = SLURM_ARRAY_TASK_ID
        SLURM_ARRAY_TASK_COUNT = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
        # Split sfiles / events-lists into similarly-sized lists to run across
        # multiple nodes. need to merge output files afterwards.
        # Keep days together in one chunk:
        uniq_days = sorted(
            list(set([sfile[-6:] + sfile[-19:-17] for sfile in sfiles])))
        sfile_chunk = []
        for jd, uniq_day in enumerate(uniq_days):
            # only deal with the day that is relevant for this array job
            if jd % SLURM_ARRAY_TASK_COUNT != SLURM_ARRAY_TASK_ID:
                continue
            sfile_chunk += [sfile for sfile in sfiles
                            if sfile[-6:] + sfile[-19:-17] == uniq_day]
        sfiles = sfile_chunk
        Logger.info(
            'This is SLURM array task %s (task count: %s) for a chunk of %s '
            'events / files.', str(SLURM_ARRAY_TASK_ID),
            str(SLURM_ARRAY_TASK_COUNT), len(sfiles))
        if len(sfiles) == 0:
            Logger.info('No sfiles in chunk, quitting template creation job')
            quit()
    except Exception as e:
        Logger.info('This is not a SLURM array task.')
        pass
    Logger.info(
        'This task will run with %s parallel workers, with up to %s '
        'threads each, on %s cores in total', cores, n_threads,
        total_cores)


    # %% make templates
    ispaq_full = None
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
        remove_response=remove_response, output='DISP',
        noise_balancing=noise_balancing, ignore_bad_data=True,
        balance_power_coefficient=balance_power_coefficient,
        apply_agc=apply_agc, make_pretty_plot=False, normalize_NSLC=True,
        parallel=parallel, cores=cores, thread_parallel=thread_parallel,
        n_threads=n_threads,
        write_out=True, write_individual_templates=False,
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
        bayesloc_event_solutions=bayesloc_path, agency_id='BER',
        custom_epoch=custom_epoch, find_event_without_id=True, s_diff=4,
        max_bayes_error_km=100, add_arrivals=True, update_phase_hints=True,
        keep_best_fit_pick_only=True, remove_1_suffix=True,
        min_phase_probability=0,
        task_id=task_id, wavetool_path=wavetool_path)
    Logger.info('Created new set of templates.')
