#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2021

@author: felix
"""

# %%
import sys
sys.settrace
import os, glob, math, calendar, platform
import numpy as np
import pandas as pd
from importlib import reload
import statistics as stats

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


# import quality_metrics, spectral_tools, load_events_for_detection
# reload(quality_metrics)
# reload(load_events_for_detection)
# sys.path.insert(1, os.path.expanduser("~/Documents2/NorthSea/Elgin/Detection"))
from robustraqn.quality_metrics import (
    create_bulk_request, get_waveforms_bulk, read_ispaq_stats)
from robustraqn.load_events_for_detection import (
    prepare_detection_stream, init_processing, init_processing_wRotation,
    print_error_plots, get_all_relevant_stations, normalize_NSLC_codes,
    reevaluate_detections)
from robustraqn.spectral_tools import (
    Noise_model, get_updated_inventory_with_noise_models)
from robustraqn.templates_creation import create_template_objects
from robustraqn.lag_calc_postprocessing import (check_duplicate_template_channels,
                                     postprocess_picked_events)
from robustraqn.processify import processify
from robustraqn.fancy_processify import fancy_processify

import logging
Logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
EQCS_logger = logging.getLogger('EQcorrscan')
EQCS_logger.setLevel(logging.ERROR)



# @processify
def pick_events_for_day(
    date, det_folder, templatePath, ispaq, client, tribe, short_tribe=Tribe(),
    only_request_detection_stations=True, relevantStations=[],
    sta_translation_file='',
    noise_balancing=False, remove_response=False, inv=Inventory(),
    parallel=False, cores=None, check_array_misdetections=False,
    write_party=False, new_threshold=None, n_templates_per_run=1,
    archives=[], request_fdsn=False,
    min_det_chans=1, sfile_path='Sfiles', operator='EQC'):
    """
    Day-loop for picker
    """
    
    # Read in party of detections and check whether to proceeed
    dayparty = Party()
    current_day_str = date.strftime('%Y-%m-%d')
    party_file = os.path.join(
        det_folder, 'UniqueDet*' + current_day_str + '.tgz')
    try:
        dayparty = dayparty.read(party_file)
    except Exception as e:
        Logger.warning('Error reading parties for ' + party_file)
        Logger.warning(e)
        return
    if not dayparty:
        return
    Logger.info('Read in party of %s families for %s from %s.',
                str(len(dayparty)), current_day_str, party_file)

    # replace the old templates in the detection-families with those for
    # picking (these contain more channels)
    # dayparty = replace_templates_for_picking(dayparty, tribe)

    # Rethreshold if required
    dayparty = Party(dayparty).rethreshold(
        new_threshold=new_threshold, new_threshold_type='MAD',
        abs_values=True)
    # If there are no detections, then continue with next day
    if not dayparty:
        Logger.info('No detections left after re-thresholding for %s '
                    + 'families on %s', str(len(dayparty)),
                    current_day_str)
        return
    Logger.info('Starting to pick events with party of %s families for %s',
                str(len(dayparty)), current_day_str)

    # Choose only stations that are relevant for any detection on that day.
    if only_request_detection_stations:
        requiredStations = set([tr.stats.station for fam in dayparty
                                for tr in fam.template.st])
        requiredStations = list(
            set(relevantStations).intersection(requiredStations))
        #requiredStations = set.intersection(relevantStations, requiredStations)
        requiredStations = get_all_relevant_stations(
            requiredStations, sta_translation_file=sta_translation_file)
    # Start reading in data for day
    starttime = UTCDateTime(pd.to_datetime(date))
    starttime_req = starttime - 15*60
    endtime = starttime + 60*60*24
    endtime_req = endtime + 15*60

    # Create a smart request, i.e.: request only recordings that match
    # the quality metrics criteria and that best match the priorities.
    bulk, day_stats = create_bulk_request(
        starttime_req, endtime_req, stats=ispaq,
        parallel=parallel, cores=cores,
        stations=requiredStations, location_priority=['10','00',''],
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
        Logger.warning('No waveforms requested for %s', current_day_str)

    # Read in continuous data and prepare for processing
    # day_st = client.get_waveforms_bulk(bulk)
    day_st = get_waveforms_bulk(client, bulk, parallel=parallel,
                                cores=cores)
    Logger.info(
        'Successfully read in waveforms for bulk request of %s NSLC-'
        + 'objects for %s - %s.', len(bulk), str(starttime)[0:19],
        str(endtime)[0:19])
    day_st = prepare_detection_stream(
        day_st, tribe, parallel=parallel, cores=cores,
        try_despike=False)
    #daily_plot(day_st, year, month, day, data_unit="counts",
    #           suffix='resp_removed')

    day_st = init_processing(
        day_st, starttime=starttime, endtime=endtime,
        remove_response=remove_response, inv=inv, parallel=parallel,
        cores=cores, min_segment_length_s=10,
        max_sample_rate_diff=1, skip_interp_sample_rate_smaller=1e-7,
        interpolation_method='lanczos', 
        skip_check_sampling_rates=[20, 40, 50, 66, 75, 100, 500],
        taper_fraction=0.005, downsampled_max_rate=None,
        noise_balancing=noise_balancing, balance_power_coefficient=2)
    original_stats_stream = day_st.copy()
    
    # WHY NEEDED HERE????
    # day_st.merge(method=0, fill_value=0, interpolation_samples=0)
    # Normalize NSLC codes
    day_st = normalize_NSLC_codes(
        day_st, inv, sta_translation_file=sta_translation_file,
        std_network_code="NS", std_location_code="00",
        std_channel_prefix="BH")
    
    # If there is no data for the day, then continue on next day.
    if not day_st.traces:
        Logger.warning('No data for detection on %s, continuing' +
                    ' with next day.', current_day_str)
        return

    # Check for erroneous detections of real signals (mostly caused by smaller
    # seismic events near one of the arrays). Solution: check whether templates
    # with shorter length increase detection value - if not; it's not a
    # desriable detection.
    if check_array_misdetections:
        if len(short_tribe) < len(tribe):
            Logger.error(
                'Missing short templates for detection-reevaluation.')
        else:
            dayparty = reevaluate_detections(
                dayparty, short_tribe, stream=day_st, threshold
                =new_threshold-2, trig_int=3.0, threshold_type='MAD',
                overlap='calculate', plotDir='ReDetectionPlots',
                plot=False, fill_gaps=True, ignore_bad_data=True,
                daylong=True, ignore_length=True, min_chans=min_det_chans,
                concurrency='multiprocess', parallel_process=parallel,
                cores=cores, xcorr_func='time_domain',
                group_size=n_templates_per_run, process_cores=cores,
                time_difference_threshold=8, detect_value_allowed_error=30,
                return_party_with_short_templates=True)
        if not dayparty:
            Logger.warning('Party of families of detections is empty.')
            return
        if write_party:
            detection_file_name = os.path.join(
                'ReDetections_MAD9', 'UniqueDet_short_' + current_day_str)
            dayparty.write(
                detection_file_name, format='tar', overwrite=True)
            dayparty.write(
                detection_file_name + '.csv', format='csv', overwrite=True)

    # Compute a minimum-CC value that makes sense for the templates
    # avg_cc = stats.mean(
    #     [d.detect_val / d.no_chans for f in dayparty for d in f])

    #min_cc = avg_cc + (1 - avg_cc) * avg_cc * 3
    # min_cc = avg_cc * 0.8 #/ 2
    # TODO: would be better to make min_cc depend on the average CC and
    #       the standard deviation of CC across the channels, but then I
    #       would have to save the standard deviation in the tribe-file.

    # min_cc = min(abs(avg_cc * 0.6), 0.5)
    # Logger.info('I will run lag-calc with min_cc of %s', str(min_cc))
    
    # Use the same filtering and sampling parameters as with templates!
    #day_st = pre_processing.dayproc(
    #    day_st, lowcut=2.5, highcut=8.0, filt_order=4, samp_rate=100,
    #    debug=0, starttime=starttime, ignore_length=True,
    #    seisan_chan_names=False, parallel=True, num_cores=cores)
    picked_catalog = Catalog()
    picked_catalog = dayparty.lag_calc(
        day_st, pre_processed=False, shift_len=0.8, min_cc=0.4,
        min_cc_from_mean_cc_factor=0.6,
        horizontal_chans=['E', 'N', '1', '2'], vertical_chans=['Z'],
        interpolate=False, plot=False, overlap='calculate',
        parallel=parallel, cores=cores, daylong=True, ignore_bad_data=True,
        ignore_length=True)
    # try:
    #except LagCalcError:
    #    pass
    #    Logger.error("LagCalc Error on " + str(year) +
    #           str(month).zfill(2) + str(day).zfill(2))

    export_catalog = postprocess_picked_events(
        picked_catalog, dayparty, original_stats_stream,
        write_sfiles=True, sfile_path=sfile_path, operator=operator,
        all_channels_for_stations=relevantStations, extract_len=240,
        write_waveforms=True, archives=archives, request_fdsn=request_fdsn,
        template_path=templatePath, min_pick_stations=8,
        min_picks_on_detection_stations=3, origin_longitude=7.92,
        origin_latitude=59.85, origin_depth=200)
    
    return export_catalog




# %% Now run the day-loop

if __name__ == "__main__":
    #Set the path to the folders with continuous data:
    archive_path = '/data/seismo-wav/SLARCHIVE'
    # archive_path2 = '/data/seismo-wav/EIDA/archive'
    client = Client(archive_path)
    # client2 = Client(archive_path2)

    sta_translation_file = "station_code_translation.txt"
    selectedStations = ['ASK','BER','BLS5','DOMB','FOO','HOMB','HYA','KMY',
                        'ODD1','SKAR','SNART','STAV','SUE','KONO','DOMB',
                        #'NAO01','NB201','NBO00','NC204','NC303','NC602',
                        'NAO00','NAO01','NAO02','NAO03','NAO04','NAO05',
                        'NB200','NB201','NB202','NB203','NB204','NB205',
                        'NBO00','NBO01','NBO02','NBO03','NBO04','NBO05',
                        'NC200','NC201','NC202','NC203','NC204','NC205',
                        'NC300','NC301','NC302','NC303','NC304','NC305',
                        'NC400','NC401','NC402','NC403','NC404','NC405',
                        'NC600','NC601','NC602','NC603','NC604','NC605',
                        'STRU']
    # selectedStations = ['ASK','BER']
    # Add some extra stations from Denmark / Germany / Netherlands
    # addStations =  ['NAO00','NAO02','NAO03','NAO04','NAO05',
    #                 'NB200','NB202','NB203','NB204','NB205',
    #                 'NBO00','NBO01','NBO02','NBO03','NBO04','NBO05',
    #                 'NC200','NC201','NC202','NC203','NC205',
    #                 'NC300','NC301','NC302','NC304','NC305',
    #                 'NC400','NC401','NC402','NC403','NC404','NC405',
    #                 'NC600','NC601','NC603','NC604','NC605']

    relevantStations = get_all_relevant_stations(
        selectedStations, sta_translation_file=sta_translation_file)
    # addStations = get_all_relevant_stations(
    #     addStations, sta_translation_file=sta_translation_file)
    # allStations = relevantStations + addStations

    startday = UTCDateTime(2021,4,1,0,0,0)
    endday = UTCDateTime(2021,4,30,0,0,0)

    invFile = '~/Documents2/ArrayWork/Inventory/NorSea_inventory.xml'
    invFile = os.path.expanduser(invFile)
    inv = get_updated_inventory_with_noise_models(
        os.path.expanduser(invFile),
        pdf_dir='~/repos/ispaq/WrapperScripts/PDFs/',
        outfile='inv.pickle', check_existing=True)

    templatePath ='Templates'
    #templatePath='LagCalcTemplates'
    parallel = True
    cores = 40
    # det_folder = 'Detections_onDelta'
    det_folder = 'ReDetections_MAD9'

    remove_response = False
    noise_balancing = False
    check_array_misdetections = False
    write_party = True
    # threshold = 11
    new_threshold = 14
    n_templates_per_run = 30
    min_det_chans = 15
    only_request_detection_stations = True

    # Read templates from file
    Logger.info('Starting template reading')
    tribe = Tribe().read('TemplateObjects/Templates_min21tr_27.tgz')
    Logger.info('Tribe archive readily read in')
    if check_array_misdetections:
        short_tribe = Tribe().read(
            'TemplateObjects/short_Templates_min21tr_27.tgz')
        Logger.info('Short-tribe archive readily read in')
    n_templates = len(tribe)

    #Check templates for duplicate channels
    tribe = check_duplicate_template_channels(tribe)

    # Read in and process the daylong data
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
        pick_events_for_day(
            date=date, det_folder=det_folder, templatePath=templatePath,
            ispaq=ispaq, client=client, relevantStations=relevantStations,
            only_request_detection_stations=only_request_detection_stations,
            noise_balancing=noise_balancing, remove_response=remove_response,
            inv=inv, parallel=parallel, cores=cores,
            check_array_misdetections=check_array_misdetections,
            write_party=write_party, new_threshold=new_threshold,
            n_templates_per_run=n_templates_per_run,
            min_det_chans=min_det_chans)


