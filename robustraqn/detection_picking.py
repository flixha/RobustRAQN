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
import difflib

from obspy.core.event import Catalog, Event, Origin
from obspy.core.utcdatetime import UTCDateTime
from obspy import read_inventory, Inventory, Stream
# from obspy.clients.filesystem.sds import Client
from robustraqn.obspy.clients.filesystem.sds import Client

from eqcorrscan.core.match_filter import (Tribe, Party)
from eqcorrscan.core.lag_calc import LagCalcError
from eqcorrscan.utils.pre_processing import dayproc

# import quality_metrics, spectral_tools, load_events_for_detection
# reload(quality_metrics)
# reload(load_events_for_detection)
from robustraqn.quality_metrics import (
    create_bulk_request, get_waveforms_bulk, read_ispaq_stats,
    get_parallel_waveform_client)
from robustraqn.load_events_for_detection import (
    prepare_detection_stream, init_processing, init_processing_wRotation,
    get_all_relevant_stations, normalize_NSLC_codes, reevaluate_detections,
    try_apply_agc)
from robustraqn.event_detection import (
    prepare_day_overlap, get_multi_obj_hash, append_list_completed_days)
from robustraqn.spectral_tools import (
    Noise_model, get_updated_inventory_with_noise_models)
from robustraqn.lag_calc_postprocessing import (
    check_duplicate_template_channels, postprocess_picked_events,
    add_origins_to_detected_events)
from robustraqn.seismic_array_tools import array_lag_calc
from robustraqn.processify import processify
from robustraqn.fancy_processify import fancy_processify

import logging
Logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
EQCS_logger = logging.getLogger('EQcorrscan')
EQCS_logger.setLevel(logging.ERROR)


def prepare_and_update_party(dayparty, tribe, day_st):
    """
    If the template was updated since the detection run, then the party and its
    detections need to be updated with some information (pick-times, channels)
    """
    if len(tribe) == 0:
        return dayparty
    for family in dayparty:
        try:
            pick_template = tribe.select(family.template.name)
        except IndexError:
            Logger.error(
                'Could not find picking-template %s for detection family',
                family.template.name)
            template_names = [templ.name for templ in tribe]
            template_name_match = difflib.get_close_matches(
                family.template.name, template_names)
            if len(template_name_match) >= 1:
                template_name_match = template_name_match[0]
            else:
                Logger.warning(
                    'Did not find corresponding picking template for %s, '
                    + 'using original detection template instead.',
                    family.template.name)
                continue
            Logger.warning(
                'Found template with name %s, using instead of %s',
                template_name_match, family.template.name)
            pick_template = tribe.select(template_name_match)
        detect_chans = set([(tr.stats.station, tr.stats.channel)
                            for tr in family.template.st])
        pick_chans = set([(tr.stats.station, tr.stats.channel)
                            for tr in pick_template.st])
        for detection in family:
            new_pick_channels = list(pick_chans.difference(detect_chans))
            detection.chans = detection.chans + new_pick_channels
            # Check that I only compare the first pick on each channel
            # (detection allows multiple picks per channel, picking does
            # not yet).
            detections_earliest_tr_picks = detection.event.picks
            # if len(family.template.st) > len(pick_template.st):
            detections_earliest_tr_picks = [
                pick for pick in detection.event.picks
                if pick.time == min(
                    [p.time for p in detection.event.picks
                        if p.waveform_id.id == pick.waveform_id.id])]
            # Picks need to be adjusted when the user changes the templates
            # between detection and picking (e.g., add new stations or 
            # chagne picks). Here we need to find the time difference
            # between the picks of the new template and the picks of the
            # detection so that we can add corrected picks for the
            # previously missing stations/channels to the detection.
            time_diffs = []
            if len(detections_earliest_tr_picks) == 0:
                continue
            for det_pick in detections_earliest_tr_picks:
                templ_picks = [
                    pick for pick in pick_template.event.picks
                    if pick.waveform_id.id == det_pick.waveform_id.id]
                if len(templ_picks) == 1:
                    _time_diff = det_pick.time - templ_picks[0].time
                    time_diffs.append(_time_diff)
                elif len(templ_picks) > 1:
                    msg = ('Lag-calc does not support two picks on the same '
                           'trace, check your picking-template {name}, trace '
                           ' {trace}, phase {phase}!'.format(
                               name=detection.template_name,
                               trace=det_pick.waveform_id.id,
                               phase=det_pick.phase_hint))
                    raise LagCalcError(msg)
            # for det_tr in family.template.st:
            #     pick_tr = pick_template.st.select(id=det_tr.id)
            #     if len(pick_tr) == 1:
            #         time_diffs.append(det_tr.stats.starttime -
            #                           pick_tr[0].stats.starttime)
            # Use mean time diff in case any picks were corrected slightly
            time_diff = np.nanmean(time_diffs)
            if np.isnan(time_diff):
                time_diff = 0
            else:
                # Do a sanity check in case incorrect picks (e.g., P
                # instead of S) were fixed for a new template
                if abs(time_diff - np.nanmedian(time_diffs)) > 1:
                    Logger.error(
                        'Template used for detection of and picking of %s '
                        'differ. Adjusting the pick times resulted in '
                        'unexpectedly large differences between channels. '
                        'This may point to problematic picks in one of the'
                        ' templates.', detection.id)
            detection.event = pick_template.event.copy()
            for pick in detection.event.picks:
                pick.time = pick.time + time_diff  # add template prepick
            # when updating picks we have to add channels to detection
            day_st_chans = set([(tr.stats.station, tr.stats.channel)
                                for tr in day_st])
            templ_st_chans = set([(tr.stats.station, tr.stats.channel)
                                  for tr in pick_template.st])
            detection.chans = sorted(
                day_st_chans.intersection(templ_st_chans))
        family.template = pick_template
    return dayparty


# @processify
def pick_events_for_day(
        date, det_folder, template_path, ispaq, clients, tribe, dayparty=None,
        short_tribe=Tribe(), det_tribe=Tribe(), stations_df=None,
        only_request_detection_stations=True, apply_array_lag_calc=False,
        relevant_stations=[], sta_translation_file='', let_days_overlap=True,
        noise_balancing=False, balance_power_coefficient=2,
        remove_response=False, inv=Inventory(),
        apply_agc=False, agc_window_sec=5,
        parallel=False, cores=None, io_cores=1,
        check_array_misdetections=False, xcorr_func='fmf', arch='cpu',
        concurrency='concurrent', trig_int=12, minimum_sample_rate=20,
        time_difference_threshold=8, detect_value_allowed_error=60,
        threshold_type='MAD', new_threshold=None, n_templates_per_run=1,
        archives=[], request_fdsn=False, min_det_chans=1, shift_len=0.8,
        min_cc=0.4, min_cc_from_mean_cc_factor=0.6, extract_len=240,
        interpolate=True, use_new_resamp_method=True,
        write_party=False, ignore_cccsum_comparison=True,
        all_vert=True, all_horiz=True, vertical_chans=['Z', 'H'],
        horizontal_chans=['E', 'N', '1', '2', 'X', 'Y'],
        sfile_path='Sfiles', write_to_year_month_folders=False,
        operator='EQC', day_hash_file=None, **kwargs):
    """
    Day-loop for picker
    """

    current_day_str = date.strftime('%Y-%m-%d')

    if day_hash_file is not None:
        # Check if this date has already been processed with the same settings
        # i.e., current date and a settings-based hash exist already in file
        Logger.info('Checking if a run with the same parameters has been '
                    'performed before...')
        settings_hash = get_multi_obj_hash(
            [tribe.templates, relevant_stations, remove_response, inv, ispaq,
            noise_balancing, balance_power_coefficient, xcorr_func, arch,
            trig_int, new_threshold, threshold_type, min_det_chans,
            minimum_sample_rate, archives, request_fdsn, shift_len, min_cc,
            min_cc_from_mean_cc_factor, extract_len, all_vert, all_horiz,
            check_array_misdetections, short_tribe, write_party,
            vertical_chans, horizontal_chans, det_folder, template_path,
            time_difference_threshold, minimum_sample_rate, apply_agc,
            agc_window_sec, interpolate, use_new_resamp_method,
            ignore_cccsum_comparison])
        # Check hash against existing list
        try:
            day_hash_df = pd.read_csv(day_hash_file, names=["date", "hash"])
            if ((day_hash_df['date'] == current_day_str) &
                    (day_hash_df['hash'] == settings_hash)).any():
                Logger.info(
                    'Day %s already processed: Date and hash match entry in '
                    'date-hash list, skipping this day.', current_day_str)
                return
        except FileNotFoundError:
            pass

    # Read in party of detections and check whether to proceeed
    if dayparty is None:
        dayparty = Party()
        party_file = os.path.join(
            det_folder, 'UniqueDet*' + current_day_str + '.tgz')
        try:
            dayparty = dayparty.read(party_file)
        except Exception as e:
            Logger.warning('Could not read in any parties for ' + party_file)
            Logger.warning(e)
            return
        if not dayparty:
            return
        Logger.info('Read in party of %s families for %s from %s.',
                    str(len(dayparty)), current_day_str, party_file)
        # replace the old templates in the detection-families with those for
        # dayparty = Party([f for f in dayparty if f.template.name == '2021_10_07t19_59_36_80_templ'])
        # dayparty = Party([f for f in dayparty if f.template.name == '2018_02_23t21_15_51_38_templ'])
        # dayparty[0].detections = [dayparty[0].detections[3]]
        # dayparty = Party([f for f in dayparty if f.template.name == '2004_10_14t10_17_46_70_templ'])
        # 2004_10_14t10_17_46_70_templ_20190501_181532100000
        # dayparty = Party([f for f in dayparty if f.template.name == '2019_06_04t13_41_43_80_templ'])
        # dayparty = Party([f for f in dayparty if f.template.name == '2005_02_26t15_41_57_20_templ'])
        # dayparty[0].detections = [dayparty[0].detections[0]]

        # 2019_10_15t02_43_11_50_templ_20190802_072130169538

        # picking (these contain more channels)
        # dayparty = replace_templates_for_picking(dayparty, tribe)

    # dayparty = Party(dayparty[65])  # DEBUG
    # dayparty = Party([family for family in dayparty  # DEBUG
    #                   if family.template.name.startswith('2018_05_07t18_46')])
    # Logger.info('Retaining only detections for template %s',
    #             dayparty[0].template.name)

    # Rethreshold if required  # 2021_10_07t19_59_36_80_templ
    if new_threshold is not None:
        dayparty = Party(dayparty).rethreshold(
            new_threshold=new_threshold, new_threshold_type='MAD',
            abs_values=True)
    # If there are no detections, then continue with next day
    if not dayparty:
        Logger.info('No detections left after re-thresholding for %s '
                    + 'families on %s', str(len(dayparty)),
                    current_day_str)
        append_list_completed_days(
            file=day_hash_file, date=current_day_str, hash=settings_hash)
        return

    Logger.info('Starting to pick events with party of %s families for %s',
                str(len(dayparty)), current_day_str)
    # Choose only stations that are relevant for any detection on that day.
    required_stations = relevant_stations
    if only_request_detection_stations:
        required_stations = set([tr.stats.station for fam in dayparty
                                for tr in fam.template.st])
        required_stations = list(
            set(relevant_stations).intersection(required_stations))
        # required_stations =set.intersection(relevant_stations,required_stations)
        required_stations = get_all_relevant_stations(
            required_stations, sta_translation_file=sta_translation_file)
    # Start reading in data for day
    starttime = UTCDateTime(pd.to_datetime(date))
    starttime_req = starttime - 15 * 60
    endtime = starttime + 60 * 60 * 24
    endtime_req = endtime + 15 * 60
    # Create a smart request, i.e.: request only recordings that match
    # the quality metrics criteria and that best match the priorities.
    bulk_request, bulk_rejected, day_stats = create_bulk_request(
        inv.select(starttime=starttime_req, endtime=endtime_req),
        starttime_req, endtime_req, stats=ispaq,
        minimum_sample_rate=minimum_sample_rate,
        parallel=parallel, cores=cores,
        stations=required_stations, **kwargs)
    if not bulk_request:
        Logger.warning('No waveforms requested for %s', current_day_str)
        append_list_completed_days(
                file=day_hash_file, date=current_day_str, hash=settings_hash)
        return

    # Read in continuous data and prepare for processing
    day_st = Stream()
    for client in clients:
        Logger.info('Requesting waveforms from client %s', client)
        # client = get_parallel_waveform_client(client)
        # day_st += client.get_waveforms_bulk_parallel(
        #     bulk_request, parallel=parallel, cores=io_cores)
        day_st += client.get_waveforms_bulk(
            bulk_request, parallel=parallel, cores=io_cores)
    Logger.info(
        'Successfully read in waveforms for bulk request of %s NSLC-'
        + 'objects for %s - %s.', len(bulk_request), str(starttime)[0:19],
        str(endtime)[0:19])
    day_st = prepare_detection_stream(
        day_st, tribe, parallel=parallel, cores=cores,
        try_despike=False)
    original_stats_stream = day_st.copy()
    # daily_plot(day_st, year, month, day, data_unit="counts",
    #            suffix='resp_removed')
    # day_st = init_processing(
    #     day_st, starttime=starttime_req, endtime=endtime_req,
    #     remove_response=remove_response, inv=inv,
    #     noise_balancing=noise_balancing,
    #     pre_filt=[0.1, 0.2, 0.9 * nyquist_f, 0.95 * nyquist_f],
    #     parallel=parallel, cores=cores, **kwargs)
    
    nyquist_f = minimum_sample_rate / 2
    day_st = init_processing_wRotation(
            day_st, starttime=starttime_req, endtime=endtime_req,
            remove_response=remove_response, inv=inv,
            pre_filt=[0.1, 0.2, 0.9 * nyquist_f, 0.95 * nyquist_f],
            parallel=parallel, cores=cores,
            sta_translation_file=sta_translation_file,
            noise_balancing=noise_balancing,
            balance_power_coefficient=balance_power_coefficient, **kwargs)

    daylong = True
    if let_days_overlap:
        daylong = False
        tribe, short_tribe, day_st = prepare_day_overlap(
            tribe, short_tribe, day_st, starttime_req, endtime_req, **kwargs)
    else:
        tribe, short_tribe, day_st = prepare_day_overlap(
            tribe, short_tribe, day_st, starttime, endtime, overlap_length=0)

    # WHY NEEDED HERE????
    # day_st.merge(method=0, fill_value=0, interpolation_samples=0)
    # Normalize NSLC codes
    day_st = normalize_NSLC_codes(
        day_st, inv, sta_translation_file=sta_translation_file,
        std_network_code="NS", std_location_code="00",
        std_channel_prefix="BH")

    pre_processed = False
    if apply_agc and agc_window_sec:
        day_st, pre_processed = try_apply_agc(
            day_st, tribe, agc_window_sec=agc_window_sec, starttime=None,
            pre_processed=pre_processed, cores=cores, parallel=parallel,
            **kwargs)

    # Update parties for picking
    dayparty = prepare_and_update_party(dayparty, tribe, day_st)

    # If there is no data for the day, then continue on next day.
    if not day_st.traces:
        Logger.warning('No data for detection on %s, continuing'
                       ' with next day.', current_day_str)
        append_list_completed_days(
                file=day_hash_file, date=current_day_str, hash=settings_hash)
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
            dayparty, short_party = reevaluate_detections(
                dayparty, short_tribe, stream=day_st,
                threshold=new_threshold-2, trig_int=trig_int/4,
                threshold_type=threshold_type,
                overlap='calculate', plotDir='ReDetectionPlots',
                plot=False, fill_gaps=True, ignore_bad_data=True,
                daylong=daylong, ignore_length=True, min_chans=min_det_chans,
                pre_processed=pre_processed,
                parallel_process=parallel, cores=cores,
                xcorr_func=xcorr_func, concurrency=concurrency, arch=arch,
                # xcorr_func='time_domain', concurrency='multiprocess',
                group_size=n_templates_per_run, process_cores=cores,
                time_difference_threshold=time_difference_threshold,
                detect_value_allowed_error=detect_value_allowed_error,
                return_party_with_short_templates=True, **kwargs)
        if not dayparty:
            Logger.warning('Party of families of detections is empty.')
            append_list_completed_days(
                file=day_hash_file, date=current_day_str, hash=settings_hash)
            return
        if write_party:
            if not os.path.exists('Re' + det_folder):
                os.mkdir('Re' + det_folder)
            detection_file_name = os.path.join(
                'Re' + det_folder, 'UniqueDet_short_' + current_day_str)
            dayparty.write(
                detection_file_name, format='tar', overwrite=True)
            dayparty.write(
                detection_file_name + '.csv', format='csv', overwrite=True)

    # Check if I can do pre-processing just once:
    if apply_array_lag_calc and not pre_processed:
        lowcuts = list(set([tp.lowcut for tp in tribe]))
        highcuts = list(set([tp.highcut for tp in tribe]))
        filt_orders = list(set([tp.filt_order for tp in tribe]))
        samp_rates = list(set([tp.samp_rate for tp in tribe]))
        if (len(lowcuts) == 1 and len(highcuts) == 1 and
                len(filt_orders) == 1 and len(samp_rates) == 1):
            Logger.info(
                'All templates have the same trace-processing parameters. '
                'Preprocessing data once for lag-calc and array-lag-calc.')
            day_st = dayproc(
                day_st, lowcut=lowcuts[0], highcut=highcuts[0],
                filt_order=filt_orders[0], samp_rate=samp_rates[0],
                starttime=starttime, parallel=parallel, num_cores=cores,
                ignore_length=False, seisan_chan_names=False, fill_gaps=True,
                ignore_bad_data=False, fft_threads=1)
            pre_processed = True
    picked_catalog = Catalog()
    picked_catalog = dayparty.copy().lag_calc(
        day_st, pre_processed=pre_processed, shift_len=shift_len,
        min_cc=min_cc, min_cc_from_mean_cc_factor=min_cc_from_mean_cc_factor,
        all_vert=all_vert, all_horiz=all_horiz,
        horizontal_chans=horizontal_chans, vertical_chans=vertical_chans,
        interpolate=interpolate, use_new_resamp_method=use_new_resamp_method,
        parallel=parallel, cores=cores, daylong=daylong,
        ignore_cccsum_comparison=ignore_cccsum_comparison, **kwargs)
    # try:
    # except LagCalcError:
    #    pass
    #    Logger.error("LagCalc Error on " + str(year) +
    #           str(month).zfill(2) + str(day).zfill(2))
    picks_per_event = [len([pk for pk in ev.picks]) for ev in picked_catalog]
    min_picks_per_event = min(picks_per_event) or 0
    max_picks_per_event = max(picks_per_event) or 0
    Logger.info('Got %s events with at least %s and at most %s picks',
                len(picked_catalog), min_picks_per_event, max_picks_per_event)

    picked_catalog = add_origins_to_detected_events(
        picked_catalog, dayparty, tribe=tribe)

    if apply_array_lag_calc:
        picked_catalog = array_lag_calc(
            day_st, picked_catalog, dayparty, tribe, stations_df,
            min_cc=min_cc, pre_processed=pre_processed, shift_len=shift_len,
            min_cc_from_mean_cc_factor=min(min_cc_from_mean_cc_factor, 0.999),
            all_vert=all_vert, all_horiz=all_horiz,
            horizontal_chans=horizontal_chans, vertical_chans=vertical_chans,
            parallel=parallel, cores=cores, daylong=daylong,
            interpolate=interpolate,
            use_new_resamp_method=use_new_resamp_method,
            ignore_cccsum_comparison=ignore_cccsum_comparison, **kwargs)

    export_catalog = postprocess_picked_events(
        picked_catalog, dayparty, tribe, original_stats_stream,
        det_tribe=det_tribe, write_sfiles=True, sfile_path=sfile_path,
        operator=operator, all_channels_for_stations=relevant_stations,
        extract_len=extract_len, write_waveforms=True, archives=archives,
        request_fdsn=request_fdsn, template_path=template_path,
        min_pick_stations=8, min_picks_on_detection_stations=3,
        write_to_year_month_folders=write_to_year_month_folders,
        parallel=parallel, cores=io_cores, **kwargs)

    append_list_completed_days(
        file=day_hash_file, date=current_day_str, hash=settings_hash)

    return export_catalog


# %% Now run the day-loop
if __name__ == "__main__":
    # Set the path to the folders with continuous data:
    archive_path = '/data/seismo-wav/SLARCHIVE'
    # archive_path2 = '/data/seismo-wav/EIDA/archive'
    client = Client(archive_path)
    # client2 = Client(archive_path2)

    sta_translation_file = "station_code_translation.txt"
    selected_stations = ['ASK','BER','BLS5','DOMB','FOO','HOMB','HYA','KMY',
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
    # selected_stations = ['ASK','BER']
    # Add some extra stations from Denmark / Germany / Netherlands
    # add_stations =  ['NAO00','NAO02','NAO03','NAO04','NAO05',
    #                 'NB200','NB202','NB203','NB204','NB205',
    #                 'NBO00','NBO01','NBO02','NBO03','NBO04','NBO05',
    #                 'NC200','NC201','NC202','NC203','NC205',
    #                 'NC300','NC301','NC302','NC304','NC305',
    #                 'NC400','NC401','NC402','NC403','NC404','NC405',
    #                 'NC600','NC601','NC603','NC604','NC605']

    relevant_stations = get_all_relevant_stations(
        selected_stations, sta_translation_file=sta_translation_file)
    # add_stations = get_all_relevant_stations(
    #     add_stations, sta_translation_file=sta_translation_file)
    # all_stations = relevant_stations + add_stations

    startday = UTCDateTime(2021,4,1,0,0,0)
    endday = UTCDateTime(2021,4,30,0,0,0)

    inv_file = '~/Documents2/ArrayWork/Inventory/NorSea_inventory.xml'
    inv_file = os.path.expanduser(inv_file)
    inv = get_updated_inventory_with_noise_models(
        os.path.expanduser(inv_file),
        pdf_dir='~/repos/ispaq/WrapperScripts/PDFs/',
        outfile='inv.pickle', check_existing=True)

    template_path ='Templates'
    #template_path='LagCalcTemplates'
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
                stations=relevant_stations, startyear=current_year,
                endyear=current_year, ispaq_prefixes=['all'],
                ispaq_suffixes=['simpleMetrics','PSDMetrics'],
                file_type = 'parquet')
        pick_events_for_day(
            date=date, det_folder=det_folder, template_path=template_path,
            ispaq=ispaq, clients=[client], relevant_stations=relevant_stations,
            only_request_detection_stations=only_request_detection_stations,
            noise_balancing=noise_balancing, remove_response=remove_response,
            inv=inv, parallel=parallel, cores=cores,
            check_array_misdetections=check_array_misdetections,
            write_party=write_party, new_threshold=new_threshold,
            n_templates_per_run=n_templates_per_run,
            min_det_chans=min_det_chans)


