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
from obspy.geodetics.base import degrees2kilometers, locations2degrees
# from obspy.clients.filesystem.sds import Client
from robustraqn.obspy.clients.filesystem.sds import Client

from eqcorrscan.core.match_filter import (Tribe, Party, Template, Family)
from eqcorrscan.core.lag_calc import LagCalcError
from eqcorrscan.utils.pre_processing import dayproc, shortproc
# from eqcorrscan.core.template_gen import _template_gen

# import quality_metrics, spectral_tools, load_events_for_detection
# reload(quality_metrics)
# reload(load_events_for_detection)
from robustraqn.quality_metrics import (
    create_bulk_request, get_waveforms_bulk, read_ispaq_stats,
    get_parallel_waveform_client)
from robustraqn.load_events_for_detection import (
    prepare_detection_stream, init_processing, init_processing_w_rotation,
    get_all_relevant_stations, normalize_NSLC_codes, reevaluate_detections,
    try_apply_agc)
from robustraqn.templates_creation import (_shorten_tribe_streams)
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
from robustraqn.obspy_utils import _quick_copy_stream

import logging
Logger = logging.getLogger(__name__)
#logging.basicConfig(
#    level=logging.INFO,
#    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
EQCS_logger = logging.getLogger('EQcorrscan')
EQCS_logger.setLevel(logging.ERROR)


def prepare_and_update_party(
        dayparty, tribe, day_st, max_template_origin_shift_seconds=10,
        max_template_origin_shift_km=50, all_vert=True, all_horiz=True,
        vertical_chans=['Z', 'H'],
        horizontal_chans=['E', 'N', '1', '2', 'X', 'Y'],
        parallel=False, cores=1):
    """
    If the template was updated since the detection run, then the party and its
    detections need to be updated with some information (pick-times, channels,
    template name)
    """
    if len(tribe) == 0:
        return dayparty
    # Check if all processing parameters of the full tribe are the same - then
    # make sure the newly added templates have the same processing parameters
    lowcuts = list(set([tp.lowcut for tp in tribe]))
    highcuts = list(set([tp.highcut for tp in tribe]))
    filt_orders = list(set([tp.filt_order for tp in tribe]))
    samp_rates = list(set([tp.samp_rate for tp in tribe]))
    process_lengths = list(set([tp.process_length for tp in tribe]))
    prepicks = list(set([tp.prepick for tp in tribe]))
    # trace_offset is not needed for the original tribe
    re_process = False
    if (len(lowcuts) == 1 and len(highcuts) == 1 and
        len(filt_orders) == 1 and len(samp_rates) == 1):
        re_process = True
        lowcut = lowcuts[0]
        highcut = highcuts[0]
        filt_order = filt_orders[0]
        samp_rate = samp_rates[0]
        process_length = process_lengths[0]
        prepick = prepicks[0]
    # Loop through party and check if the templates are available for picking!
    for family in dayparty:
        try:
            pick_template = tribe.select(family.template.name)
        except IndexError:
            Logger.error(
                'Could not find picking-template %s for detection family',
                family.template.name)
            template_names = [templ.name for templ in tribe]
            template_name_matches = difflib.get_close_matches(
                family.template.name, template_names, n=100)
            found_ok_match = False
            if len(template_name_matches) >= 1:
                choose_index = 0
                # Loop through matches with similar template names, and check
                # by how far their origin differs from detection template. Only
                # accept alternative template within specific bounds.
                while (not found_ok_match and
                       choose_index <= len(template_name_matches) - 1):
                    template_name_match = template_name_matches[choose_index]
                    pick_template = tribe.select(template_name_match)
                    p_origin = pick_template.event.origins[0]
                    d_origin = family.template.event.origins[0]
                    time_difference = abs(p_origin.time - d_origin.time)
                    if time_difference > max_template_origin_shift_seconds:
                        choose_index += 1
                        continue
                    dist_deg = locations2degrees(
                        p_origin.latitude, p_origin.longitude,
                        d_origin.latitude, d_origin.longitude)
                    location_difference_km = degrees2kilometers(dist_deg)
                    if location_difference_km > max_template_origin_shift_km:
                        choose_index += 1
                    else:
                        found_ok_match = True
            if not found_ok_match:
                Logger.warning(
                    'Did not find corresponding picking template for %s, '
                    + 'using original detection template instead.',
                    family.template.name)
                add_tribe = Tribe(family.template)
                # Check if this template stream should be adjusted to fit rest
                new_tribe = Tribe()
                if re_process:
                    for template in add_tribe:
                        # Need to process stream and define template with the
                        # corrected parameters
                        template_st = shortproc(
                            template.st, lowcut=lowcut, highcut=highcut,
                            filt_order=filt_order, samp_rate=samp_rate,
                            parallel=False, num_cores=cores,
                            ignore_length=False, seisan_chan_names=False,
                            fill_gaps=True, ignore_bad_data=False,
                            fft_threads=1)
                        add_templ = Template(
                            name=template.name, event=template.event,
                            st=template_st, lowcut=lowcut, highcut=highcut,
                            samp_rate=samp_rate, filt_order=filt_order,
                            process_length=process_length, prepick=prepick)
                        new_tribe += add_templ
                    add_tribe = new_tribe
                # Check that there are no duplicate channels in template
                tribe += check_duplicate_template_channels(
                    add_tribe, all_vert=all_vert,
                    all_horiz=all_horiz, vertical_chans=vertical_chans,
                    horizontal_chans=horizontal_chans, parallel=False)
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
                    if pick.waveform_id.id == det_pick.waveform_id.id
                    and (pick.phase_hint[0] == det_pick.phase_hint[0]
                         if pick.phase_hint and det_pick.phase_hint else True)]
                if len(templ_picks) > 1:
                    # Could happen when there are two picks with different phas
                    # e hints at the same time, and both are earliest on trace.
                    templ_picks = [pick for pick in templ_picks
                                   if pick.phase_hint == det_pick.phase_hint]
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
                    two_sigma_td = 2 * stats.stdev(time_diffs)
                    time_outliers = [tdiff for tdiff in time_diffs
                                     if abs(tdiff - time_diff) > two_sigma_td]
                    Logger.error(
                        'Template used for detection and picking of %s differ.'
                        ' Adjusting pick times resulted in unexpectedly large '
                        'differences between %s channels with time-differences'
                        ' greater 2xStdDev. This may point to problematic '
                        'picks in one of the templates. (hope the new one is '
                        'right!)', detection.id, len(time_outliers))
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
        for detection in family:
            detection.template_name = pick_template.name
    return dayparty


#@fancy_processify
def pick_events_for_day(
        date, det_folder, template_path, ispaq, clients, tribe, dayparty=None,
        short_tribe=Tribe(), short_tribe2=Tribe(), det_tribe=Tribe(),
        stations_df=None, only_request_detection_stations=True,
        apply_array_lag_calc=False,
        relevant_stations=[], sta_translation_file='', let_days_overlap=True,
        all_chans_for_stations=[],
        noise_balancing=False, balance_power_coefficient=2,
        remove_response=False, output='DISP', inv=Inventory(),
        apply_agc=False, agc_window_sec=5, blacklisted_templates=[],
        parallel=False, cores=None, io_cores=1, plot=False, multiplot=False,
        check_array_misdetections=False, xcorr_func='fmf', arch='precise',
        re_eval_thresh_factor=0.6, use_weights=False, min_pick_stations=5,
        min_picks_on_detection_stations=4, min_n_station_sites=4,
        concurrency='concurrent', trig_int=12, minimum_sample_rate=20,
        time_difference_threshold=1, detect_value_allowed_reduction=2.5,
        threshold_type='MAD', new_threshold=None, n_templates_per_run=1,
        archives=[], archive_types=[], request_fdsn=False,
        pick_xcorr_func=None, min_det_chans=1, shift_len=0.8,
        min_cc=0.4, min_cc_from_mean_cc_factor=None,
        min_cc_from_median_cc_factor=0.98, extract_len=240,
        interpolate=True, use_new_resamp_method=True,
        write_party=False, ignore_cccsum_comparison=True,
        all_vert=True, all_horiz=True, vertical_chans=['Z', 'H'],
        horizontal_chans=['E', 'N', '1', '2', 'X', 'Y'],
        sfile_path='Sfiles', write_to_year_month_folders=False,
        operator='EQC', day_hash_file=None, copy_data=True,
        redecluster=False, clust_trig_int=30, decluster_metric='thresh_exc',
        hypocentral_separation=False, min_chans=13, absolute_values=True,
        compute_relative_magnitudes=False, mag_min_cc_from_mean_cc_factor=None,
        mag_min_cc_from_median_cc_factor=1.2, **kwargs):
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
            [tribe.templates, relevant_stations, remove_response, output,
             inv, ispaq, noise_balancing, balance_power_coefficient,
             xcorr_func, arch, trig_int, new_threshold, threshold_type,
             min_det_chans, minimum_sample_rate, archives, request_fdsn,
             shift_len, min_cc, min_cc_from_mean_cc_factor, 
             min_cc_from_median_cc_factor, extract_len,
             all_vert, all_horiz, check_array_misdetections,
             short_tribe.templates, short_tribe2.templates,
             re_eval_thresh_factor, detect_value_allowed_reduction,
             time_difference_threshold,
             vertical_chans, horizontal_chans, det_folder, template_path,
             time_difference_threshold, minimum_sample_rate, apply_agc,
             agc_window_sec, interpolate, use_new_resamp_method,
             ignore_cccsum_comparison, min_pick_stations,
             min_picks_on_detection_stations, min_n_station_sites,
             compute_relative_magnitudes, mag_min_cc_from_mean_cc_factor,
             mag_min_cc_from_median_cc_factor])
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
            dayparty = dayparty.read(party_file, cores=cores)
        except Exception as e:
            Logger.warning('Could not read in any parties for ' + party_file)
            Logger.warning(e)
            return
        if not dayparty:
            return
        Logger.info(
            'Read in party of %s families, %s detections, for %s from %s.',
            len(dayparty.families), len(dayparty), current_day_str, party_file)
        # Exclude detections for templates that are blacklisted
        if len(blacklisted_templates) > 0:
            remove_family_list = [
                family for family in dayparty
                if family.template.name in blacklisted_templates]
            dayparty = Party([family for family in dayparty
                              if family not in remove_family_list])
        # replace the old templates in the detection-families with those for
        # dayparty = Party([f for f in dayparty if f.template.name == '2021_10_07t19_59_36_80_templ'])
        # dayparty = Party([f for f in dayparty if f.template.name == '2018_02_23t21_15_51_38_templ'])
        # dayparty[0].detections = [dayparty[0].detections[0]]
        # dayparty = Party([f for f in dayparty if f.template.name == '2015_02_14t07_19_44_39_templ'])
        
        # fam = dayparty.select('2022_04_25t09_31_30_18_templ')
        # dets = [det for det in fam if det.id ==
        #         '2022_04_25t09_31_30_18_templ_20220425_003728600000']
        # dayparty = Party(Family(template=fam.template, detections=dets))

        # 2019_10_15t02_43_11_50_templ_20190802_072130169538

        # picking (these contain more channels)
        # dayparty = replace_templates_for_picking(dayparty, tribe)

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
        if day_hash_file is not None:
            append_list_completed_days(
                file=day_hash_file, date=current_day_str, hash=settings_hash)
        return

    if redecluster:
        n_det_init = len([d for f in dayparty for d in f])
        dayparty = dayparty.decluster(
            trig_int=clust_trig_int, timing='detect', metric=decluster_metric,
            hypocentral_separation=hypocentral_separation,
            min_chans=min_chans, absolute_values=absolute_values)
        n_det1 = len([d for f in dayparty for d in f])
        # 2nd pass to decluster based on origin times
        dayparty = dayparty.decluster(
            trig_int=clust_trig_int, timing='origin', metric=decluster_metric,
            hypocentral_separation=hypocentral_separation,
            min_chans=min_chans, absolute_values=absolute_values)
        n_det2 = len([d for f in dayparty for d in f])
        Logger.info(
            'Re-declustering party: Out of %s initial detections, after two-'
            'stage declustering, %s (%s on 1st round) detections are left',
            n_det_init, n_det2, n_det1)

    Logger.info('Starting to pick events with party of %s families for %s',
                len(dayparty.families), current_day_str)
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
        if day_hash_file is not None:
            append_list_completed_days(
                file=day_hash_file, date=current_day_str,
                hash=settings_hash)
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
    # original_stats_stream = day_st.copy()
    original_stats_stream = _quick_copy_stream(day_st)
    # daily_plot(day_st, year, month, day, data_unit="counts",
    #            suffix='resp_removed')
    # day_st = init_processing(
    #     day_st, starttime=starttime_req, endtime=endtime_req,
    #     remove_response=remove_response, inv=inv,
    #     noise_balancing=noise_balancing,
    #     pre_filt=[0.1, 0.2, 0.9 * nyquist_f, 0.95 * nyquist_f],
    #     parallel=parallel, cores=cores, **kwargs)
    
    nyquist_f = minimum_sample_rate / 2
    day_st = init_processing_w_rotation(
        day_st, starttime=starttime_req, endtime=endtime_req,
        remove_response=remove_response, output=output, inv=inv,
        pre_filt=[0.1, 0.2, 0.9 * nyquist_f, 0.95 * nyquist_f],
        parallel=parallel, cores=cores,
        sta_translation_file=sta_translation_file,
        noise_balancing=noise_balancing,
        balance_power_coefficient=balance_power_coefficient, **kwargs)

    daylong = True
    tribes = [tribe, short_tribe, short_tribe2]
    if let_days_overlap:
        daylong = False
        tribes, day_st = prepare_day_overlap(
            tribes, day_st, starttime_req, endtime_req, **kwargs)
    else:
        tribes, day_st = prepare_day_overlap(
            tribes, day_st, starttime, endtime, overlap_length=0)
    tribe, short_tribe, short_tribe2 = tribes

    # WHY NEEDED HERE????
    # day_st.merge(method=0, fill_value=0, interpolation_samples=0)
    # Normalize NSLC codes
    day_st, trace_id_change_dict = normalize_NSLC_codes(
        day_st, inv, sta_translation_file=sta_translation_file,
        std_network_code="NS", std_location_code="00",
        std_channel_prefix="BH")

    # If there is no data for the day, then continue on next day.
    if not day_st.traces:
        Logger.warning('No data for detection on %s, continuing'
                       ' with next day.', current_day_str)
        if day_hash_file is not None:
            append_list_completed_days(
                file=day_hash_file, date=current_day_str, hash=settings_hash)
        return

    # Check if I can do pre-processing just once:
    pre_processed = False
    if ((apply_array_lag_calc or apply_agc or check_array_misdetections or
            compute_relative_magnitudes) and not pre_processed):
        lowcuts = list(set([tp.lowcut for tp in tribe]))
        highcuts = list(set([tp.highcut for tp in tribe]))
        filt_orders = list(set([tp.filt_order for tp in tribe]))
        samp_rates = list(set([tp.samp_rate for tp in tribe]))
        if (len(lowcuts) == 1 and len(highcuts) == 1 and
                len(filt_orders) == 1 and len(samp_rates) == 1):
            Logger.info(
                'All templates have the same trace-processing parameters. '
                'Preprocessing data once for detection checking, agc, lag-calc'
                ', array-lag-calc, and relative magnitudes.')
            day_st = shortproc(
                day_st, lowcut=lowcuts[0], highcut=highcuts[0],
                filt_order=filt_orders[0], samp_rate=samp_rates[0],
                starttime=starttime, parallel=parallel, num_cores=cores,
                ignore_length=False, seisan_chan_names=False, fill_gaps=True,
                ignore_bad_data=False, fft_threads=1)
            pre_processed = True

    if apply_agc and agc_window_sec:
        day_st, pre_processed = try_apply_agc(
            day_st, tribe, agc_window_sec=agc_window_sec, starttime=None,
            pre_processed=pre_processed, cores=cores, parallel=parallel,
            **kwargs)

    # Update parties for picking
    dayparty = prepare_and_update_party(
        dayparty, tribe, day_st, all_horiz=all_horiz, all_vert=all_vert,
        parallel=parallel, cores=cores)

    # Check if the selection fmf / fmf2 backends for CPU makes sense (to not
    # request GPU backends unintentionally)
    if (arch == 'precise' and
            concurrency not in ['multiprocess', 'multithread']):
        concurrency = 'multiprocess'

    # Check for erroneous detections of real signals (mostly caused by smaller
    # seismic events near one of the arrays). Solution: check whether templates
    # with shorter length increase detection value - if not; it's not a
    # desriable detection.
    if check_array_misdetections:
        for shortt in [short_tribe, short_tribe2]:
            if len(shortt) < len(tribe) and len(shortt) > 0:
                Logger.error(
                    'Missing %s short templates for detection-reevaluation in '
                    'picking-tribe, adding them from the detection-templates.',
                    (len(tribe) - len(shortt)))
                # may need to shorten extra templates that were not part of the
                # short_tribe for picking, but that we can retrieve from
                # detections
                tribe_tr = tribe[0].st[0]
                if len(shortt) > 0:
                    s_tribe_tr = shortt[0].st[0]
                    # TODO: here I assume that template are all the same length
                    #       and same trace offset - do I need extra check?
                    tribe_len_pct = (
                        (s_tribe_tr.stats.endtime - s_tribe_tr.stats.starttime)
                        / (tribe_tr.stats.endtime - tribe_tr.stats.starttime))
                    # Find trace_offset that is used as start of 2nd shorttribe
                    try:
                        trace_offset = shortt[0].trace_offset
                    except AttributeError:
                        trace_offset = 0
                    existing_templ_names = [templ.name for templ in shortt]
                    # Find templates that need to be added to picking-tribe
                    extra_tribe = Tribe(
                        [templ for templ in tribe
                        if templ.name not in existing_templ_names])
                    # Check that there are no duplicate channels in template
                    extra_tribe = check_duplicate_template_channels(
                        extra_tribe, all_vert=all_vert, all_horiz=all_horiz,
                        parallel=parallel, cores=cores)
                    shortt += _shorten_tribe_streams(
                        extra_tribe, tribe_len_pct=tribe_len_pct,
                        min_n_traces=min_chans, trace_offset=trace_offset)
            elif len(shortt) == 0:
                Logger.error('Missing all short templates, cannot do detection'
                             '-reevaluation')
            else:
                Logger.info('Got short tribe with %s templates. ready for '
                             'reevaluation.', len(shortt))
        if len(short_tribe) > 0:
            dayparty, short_party = reevaluate_detections(
                dayparty, short_tribe, stream=day_st,
                threshold=new_threshold-1, trig_int=trig_int/4,
                threshold_type=threshold_type,
                re_eval_thresh_factor=re_eval_thresh_factor,
                overlap='calculate', plotDir='ReDetectionPlots',
                plot=plot, multiplot=multiplot, fill_gaps=True,
                ignore_bad_data=True, daylong=daylong, ignore_length=True,
                min_chans=min_det_chans, pre_processed=pre_processed,
                parallel_process=parallel, cores=cores,
                xcorr_func=xcorr_func, arch=arch, concurrency=concurrency,
                # xcorr_func='time_domain', concurrency='multiprocess',
                group_size=n_templates_per_run, process_cores=cores,
                time_difference_threshold=time_difference_threshold,
                detect_value_allowed_reduction=detect_value_allowed_reduction,
                return_party_with_short_templates=True,
                min_n_station_sites=min_n_station_sites,
                use_weights=use_weights, copy_data=copy_data, **kwargs)
        if len(short_tribe2) > 0:
            dayparty, short_party2 = reevaluate_detections(
                dayparty, short_tribe2, stream=day_st,
                threshold=new_threshold-1, trig_int=trig_int/4,
                threshold_type=threshold_type,
                re_eval_thresh_factor=re_eval_thresh_factor*0.9,
                overlap='calculate', plotDir='ReDetectionPlots',
                plot=plot, multiplot=multiplot, fill_gaps=True,
                ignore_bad_data=True, daylong=daylong, ignore_length=True,
                min_chans=min_det_chans, pre_processed=pre_processed,
                parallel_process=parallel, cores=cores,
                xcorr_func=xcorr_func, arch=arch, concurrency=concurrency,
                # xcorr_func='time_domain', concurrency='multiprocess',
                group_size=n_templates_per_run, process_cores=cores,
                time_difference_threshold=time_difference_threshold,
                detect_value_allowed_reduction=(
                    detect_value_allowed_reduction * 1.2),
                return_party_with_short_templates=True,
                min_n_station_sites=min_n_station_sites,
                use_weights=use_weights, copy_data=copy_data, **kwargs)
        if not dayparty:
            Logger.warning('Party of families of detections is empty.')
            if day_hash_file is not None:
                append_list_completed_days(
                    file=day_hash_file, date=current_day_str,
                    hash=settings_hash)
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

    picked_catalog = Catalog()
    picked_catalog = dayparty.copy().lag_calc(
        day_st, pre_processed=pre_processed, shift_len=shift_len,
        min_cc=min_cc, min_cc_from_mean_cc_factor=min_cc_from_mean_cc_factor,
        min_cc_from_median_cc_factor=min_cc_from_median_cc_factor,
        all_vert=all_vert, all_horiz=all_horiz,
        horizontal_chans=horizontal_chans, vertical_chans=vertical_chans,
        interpolate=interpolate, use_new_resamp_method=use_new_resamp_method,
        daylong=daylong, ignore_cccsum_comparison=ignore_cccsum_comparison,
        xcorr_func=pick_xcorr_func, concurrency=concurrency,
        parallel=parallel, cores=cores,
        **kwargs)
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
            min_cc_from_mean_cc_factor=min_cc_from_mean_cc_factor,
            min_cc_from_median_cc_factor=min_cc_from_median_cc_factor,
            all_vert=all_vert, all_horiz=all_horiz,
            horizontal_chans=horizontal_chans, vertical_chans=vertical_chans,
            xcorr_func=pick_xcorr_func, parallel=parallel, cores=cores,
            daylong=daylong, interpolate=interpolate,
            use_new_resamp_method=use_new_resamp_method,
            ignore_cccsum_comparison=ignore_cccsum_comparison, **kwargs)

    export_catalog = postprocess_picked_events(
        picked_catalog, dayparty, tribe, original_stats_stream,
        det_tribe=det_tribe, day_st=day_st, pre_processed=pre_processed,
        write_sfiles=True, sfile_path=sfile_path,
        operator=operator, all_chans_for_stations=all_chans_for_stations,
        extract_len=extract_len, write_waveforms=True, archives=archives,
        sta_translation_file=sta_translation_file, archive_types=archive_types,
        request_fdsn=request_fdsn, template_path=template_path,
        min_n_station_sites=min_n_station_sites,
        min_pick_stations=min_pick_stations, 
        min_picks_on_detection_stations=min_picks_on_detection_stations,
        write_to_year_month_folders=write_to_year_month_folders,
        compute_relative_magnitudes=compute_relative_magnitudes,
        min_mag_cc=min_cc, remove_response=remove_response, output=output,
        mag_min_cc_from_mean_cc_factor=mag_min_cc_from_mean_cc_factor,
        mag_min_cc_from_median_cc_factor=mag_min_cc_from_median_cc_factor,
        absolute_values=absolute_values,
        parallel=parallel, cores=io_cores, **kwargs)

    if day_hash_file is not None:
        append_list_completed_days(
            file=day_hash_file, date=current_day_str, hash=settings_hash)

    return export_catalog


# %% Now run the day-loop    ### TEST ###
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


