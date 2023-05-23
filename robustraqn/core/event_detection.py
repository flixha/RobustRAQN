#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main wrapper to run event detection on one given day of data.
Integrates - data quality metrics
           - seismic array processing
           - first round of detection QC

Created on Sun Jul 16 17:23:45 2017

@author: felix
"""

# %%

import os, gc, math, matplotlib
import warnings
import glob

import pandas as pd
from importlib import reload
import numpy as np
import pickle
import hashlib
from collections import Counter
from joblib import Parallel, delayed
import logging

from obspy.core.inventory.inventory import Inventory
from obspy import UTCDateTime
from obspy.core.util.attribdict import AttribDict
from obspy.core.util.deprecation_helpers import ObsPyDeprecationWarning

from eqcorrscan.core.match_filter import Template, Tribe
from eqcorrscan.core.match_filter.party import Party, Family

from robustraqn.obspy.core import Stream, Trace
from robustraqn.utils.quality_metrics import (
    create_bulk_request, get_waveforms_bulk, read_ispaq_stats)
from robustraqn.core.load_events import (
    prepare_detection_stream, print_error_plots, reevaluate_detections,
    multiplot_detection, try_apply_agc)
from robustraqn.utils.obspy import _quick_copy_stream
from robustraqn.utils.processify import processify
from robustraqn.utils.fancy_processify import fancy_processify
from robustraqn.obspy.clients.filesystem.sds import Client

Logger = logging.getLogger(__name__)


# TODO: option to use C|C| instead of C for correlation stats
# - in line 476 multi_corr.c         # fftw
# - in line 51, 85 in time_corr.c    # time domain
# - in line 134 in matched_filter.cu # fmf-gpu
# - in line 109 in matched_filter.c  # fmf-cpu


def read_bulk_test(client, bulk, parallel=False, cores=None):
    # Read in continuous data and prepare for processing
    st = get_waveforms_bulk(client, bulk, parallel=parallel, cores=cores)
    return st


def append_list_completed_days(file, date, hash):
    """
    Append the date string and a configuration hash to the hash list file.

    :type file: str
    :param file: Path to the file to append to.
    :type date: str
    :param date: Date string to append.
    :type hash: str
    :param hash: Hash string to append.
    """
    # setting_hash = hashlib.md5()
    # setting_hash.update(kwargs)
    if file is None:
        return
    with open(file, "a+") as list_completed_days:
        list_completed_days.write(str(date) + ',' + str(hash) + '\n')


def prepare_day_overlap(
        tribes, stream, starttime_req, endtime_req, overlap_length=600,
        **kwargs):
    """
    Set processing parameters to take care of 10 minutes overlap between days.

    :type tribes: list of `EQcorrscan.core.match_filter.tribe.Tribe`
    :param tribes: List of tribes to be processed.
    :type stream: `obspy.core.stream.Stream`
    :param stream: Stream to be processed.
    :type starttime_req: `obspy.core.utcdatetime.UTCDateTime`
    :param starttime_req: Starttime of the requested data.
    :type endtime_req: `obspy.core.utcdatetime.UTCDateTime`
    :param endtime_req: Endtime of the requested data.
    :type overlap_length: int
    :param overlap_length: Length of the overlap in seconds.

    :rtype: tuple
    :param: Tuple of tribes and stream with updated processing parameters.
    """
    # TODO: implement variable overlap
    starttime_overlap = starttime_req + 5 * 60
    endtime_overlap = endtime_req - 5 * 60
    process_length = endtime_overlap - starttime_overlap  # in seconds
    for tribe in tribes:
        for templ in tribe:
            templ.process_length = process_length
    stream.trim(starttime=starttime_overlap, endtime=endtime_overlap)
    Logger.info('Earliest trace start: %s',
                 min([tr.stats.starttime for tr in stream]))
    return tribes, stream


def get_multi_obj_hash(hash_object_list):
    """
    Function to compute a recreateable hash for a list of objects. This hash
    should not change if the objects are not changed, but if any relevant
    content in the objects or parameters changes, then the hash should change.

    :type hash_object_list: list
    :param hash_object_list: List of objects to be included in the hash.

    :rtype: str
    :return: Hash string.
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
        if isinstance(obj, str):
            hash = hashlib.md5(obj.encode('utf-8'))
        if hash is None:
            try:
                # E.g. for tribe, stream, or inventory (tries to loop 1/2/3 x)
                hash = hashlib.md5(str(obj).encode('utf-8'))
                hash = hashlib.md5(str(sorted([
                    str(item) for item in obj])).encode('utf-8'))
                hash = hashlib.md5(str(sorted([
                    str(item) for subobj in obj
                    for item in subobj])).encode('utf-8'))
                hash = hashlib.md5(str(sorted([
                    str(item) for subobj in obj for subsubobj in subobj
                    for item in subsubobj])).encode('utf-8'))
            except (TypeError, ValueError):
                pass
        if hash is not None:
            try:
                hash = hash.hexdigest()
            except AttributeError:
                pass
            hash_list.append(hash)
    settings_hash = hashlib.md5(str(hash_list).encode('utf-8')).hexdigest()
    return settings_hash


def calculate_events_for_party(party, parallel=False, cores=None):
    """
    Calculate events for each detection in party, allowing for parallel
    execution.

    :type party: `EQcorrscan.core.match_filter.party.Party`
    :param party: Party of families of detections that each need a new event.
    :type parallel: bool
    :param parallel: Whether to run in parallel.
    :type cores: int
    :param cores: Number of cores to use for parallel processing.

    :rtype: `EQcorrscan.core.match_filter.party.Party`
    :return: Party with events for each detection attached
    """
    if parallel:
        Logger.info(
            'Adding origins to remaining %s short detections in '
            'parallel.', len([d for fam in party for d in fam]))
        # with parallel_backend('multiprocessing', n_jobs=cores):
        detections = [det for family in party for det in family]
        # Make simplified templates that contain only trace headers, no data,
        # to speed up copying of templates to workers:
        simplified_templates = []
        for family in party:
            if len(family) == 0:
                continue
            full_templ = family.template
            # Use quick copy function for ~20 % speedup
            new_template_st = _quick_copy_stream(
                full_templ.st, deepcopy_data=False)
            # Set data arrays to first element alone to save time when sending
            # to workers
            for trace in new_template_st:
                trace.__dict__['data'] = trace.__dict__['data'][:1]
            # Create simplified template for quicker Pool initialization
            simple_template = Template(
                name=full_templ.name, st=new_template_st,
                lowcut=full_templ.lowcut, highcut=full_templ.highcut,
                samp_rate=full_templ.samp_rate,
                filt_order=full_templ.filt_order,
                process_length=full_templ.process_length,
                prepick=full_templ.prepick, event=full_templ.event)
            # template_streams.append(new_template_st)
            for detection in family:
                simplified_templates.append(simple_template)
        detections = Parallel(n_jobs=cores)(
            delayed(detection._calculate_event)(
                template=simplified_templates[i], template_st=None,
                estimate_origin=True, correct_prepick=True,
                use_simplified_origin=True)
            for i, detection in enumerate(detections))
        # Sort in detections with events and origins
        jdet = 0
        for jf, family in enumerate(party):
            new_detections = []
            for jd, detection in enumerate(family):
                # detection = detections[jd]
                new_detections.append(detections[jdet])
                jdet += 1
            family.detections = new_detections
    else:
        Logger.info('Adding origins to %s remaining short detections.',
                    len([d for fam in party for d in fam]))
        with warnings.catch_warnings(category=ObsPyDeprecationWarning):
            for family in party:
                for detection in family:
                    _ = detection._calculate_event(
                        template=family.template, template_st=None,
                        estimate_origin=True, correct_prepick=True)
    return party


# @processify
def run_day_detection(
        # MAIN INPUT OBJECTS
        clients, tribe, date, ispaq, selected_stations, day_st=Stream(),
        sta_translation_file=os.path.expanduser(
            "~/Documents2/ArrayWork/Inventory/station_code_translation.txt"),
        inv=Inventory(),
        # INIT PROCESSING CONFIG
        output='DISP', noise_balancing=False,
        balance_power_coefficient=2, let_days_overlap=True,
        minimum_sample_rate=20, min_chans=10,
        apply_agc=False, agc_window_sec=5,
        suppress_arraywide_steps=True,
        # MULTICORE / BACKEND CONFIG
        parallel=False, cores=1, io_cores=1, remove_response=False,
        n_templates_per_run=20, xcorr_func='fftw', use_weights=False,
        concurrency=None, arch='precise',
        threshold_type='MAD', threshold=10, trig_int=0, min_n_station_sites=4,
        # ARRAY DETECTION CONFIG
        check_array_misdetections=False,
        short_tribe=Tribe(), short_tribe2=Tribe(),
        re_eval_thresh_factor=0.6, time_difference_threshold=3,
        detect_value_allowed_reduction=2.5,
        # DECLUSTER CONFIG
        decluster_metric='thresh_exc', hypocentral_separation=200,
        absolute_values=True, 
        # OUTPUT SETUP
        write_party=False, detection_path='Detections', redetection_path=None,
        copy_data=True,
        return_stream=False, dump_stream_to_disk=False, day_hash_file=None,
        skip_days_with_existing_events=True,
        multiplot=False,
        **kwargs):
    """
    Main wrapper function to run reading, initial processing, detection etc. on
    one day of data.
    """
    current_day_str = date.strftime('%Y-%m-%d')
    # Check if party file for current day already exists
    if skip_days_with_existing_events:
        detection_file_name = os.path.join(detection_path,
                                           'UniqueDet' + current_day_str)
        day_party_files = glob.glob(detection_file_name + '.tgz')
        if len(day_party_files) > 0:
            Logger.info('Skipping day %s as detection file %s already exists.',
                        current_day_str, day_party_files[0])
            if not return_stream and dump_stream_to_disk:
                return
            else:
                return [Party(), Stream()]


    Logger.info('Starting detection run for day %s', str(date)[0:10])
    if (arch == 'precise' or arch == 'CPU' and
            concurrency not in ['multiprocess', 'multithread']):
        concurrency = 'multiprocess'

    # When copy_data=False selected for EQcorrscan, then we have to copy the 
    # data here.
    # TODO: use quicker tribe copy function.
    if not copy_data:
        tribe = tribe.copy()
        short_tribe = short_tribe.copy()
        short_tribe2 = short_tribe2.copy()

    # Split tribe into chunks to save memory in computing-intense parts.
    n_templates = len(tribe)
    if n_templates == 0:
        msg = 'Cannot detect events with an empty tribe!'
        raise ValueError(msg)
    n_runs = math.ceil(n_templates / n_templates_per_run)

    starttime = UTCDateTime(pd.to_datetime(date))
    starttime_req = starttime - 15 * 60
    endtime = starttime + 60 * 60 * 24
    endtime_req = endtime + 15 * 60

    if not os.path.exists(detection_path):
        os.mkdir(detection_path)
    if redetection_path is None:
        redetection_path = 'Re' + detection_path
    if not os.path.exists(redetection_path):
        os.mkdir(redetection_path)

    # Keep input safe:
    day_st = _quick_copy_stream(day_st)
    if len(day_st) == 0:
        # Differentiate between different servers, e.g., Echo and Delta (
        # deadlocks here otherwise)
        # req_parallel = False
        # if int(platform.release()[0]) >= 4 and parallel:
        #     req_parallel = parallel
        # Create a smart request, i.e.: request only recordings that match
        # the quality metrics criteria and that best match the priorities.
        bulk_request, bulk_rejected, day_stats = create_bulk_request(
            inv.select(starttime=starttime_req, endtime=endtime_req),
            starttime_req, endtime_req, stats=ispaq,
            parallel=False, cores=1, stations=selected_stations,
            minimum_sample_rate=minimum_sample_rate, **kwargs)
        Logger.debug('Bulk request: %s', bulk_request)
        if not bulk_request:
            Logger.warning('No waveforms requested for %s - %s',
                           str(starttime)[0:19], str(endtime)[0:19])
            if not return_stream and dump_stream_to_disk:
                return
            else:
                return [Party(), Stream()]

    # Check if this date has already been processed with the same settings
    # i.e., current date and a settings-based hash exist already in file
    if day_hash_file is not None:
        Logger.info('Checking if a run with the same parameters has been '
                    'performed before...')
        settings_hash = get_multi_obj_hash(
            [tribe.templates, selected_stations, remove_response, inv, output,
             day_stats, noise_balancing, balance_power_coefficient, xcorr_func,
             arch, trig_int, threshold, re_eval_thresh_factor, min_chans,
             multiplot, check_array_misdetections, suppress_arraywide_steps,
             short_tribe, short_tribe2,
             write_party, detection_path, redetection_path,
             time_difference_threshold, minimum_sample_rate,
             min_n_station_sites, apply_agc, agc_window_sec, use_weights])
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

        # Read in continuous data and prepare for processing
        day_st = Stream()
        for client in clients:
            Logger.info('Requesting waveforms from client %s', client)
            day_st += client.get_waveforms_bulk(
                bulk_request, parallel=parallel, cores=io_cores)

        Logger.info(
            'Successfully read in %s traces for bulk request of %s NSLC-'
            'objects for %s - %s.', len(day_st), len(bulk_request),
            str(starttime)[0:19], str(endtime)[0:19])
        day_st = prepare_detection_stream(
            day_st, tribe, parallel=parallel, cores=cores, try_despike=False,
            ispaq=day_stats)
        # Do initial processing (rotation, stats normalization, merging)
        # by parallelization across three-component seismogram sets.
        nyquist_f = minimum_sample_rate / 2
        day_st = day_st.init_processing_w_rotation(
            starttime=starttime_req, endtime=endtime_req,
            # day_st, starttime=starttime, endtime=endtime,
            remove_response=remove_response, output=output, inv=inv,
            pre_filt=[0.1, 0.2, 0.9 * nyquist_f, 0.95 * nyquist_f],
            parallel=parallel, cores=cores,
            suppress_arraywide_steps=suppress_arraywide_steps,
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
        # # Normalize NSLC codes
        # day_st, trace_id_change_dict = day_st.normalize_nslc_codes(
        #     inv, parallel=False, cores=cores,
        #     std_network_code="NS", std_location_code="00",
        #     std_channel_prefix="BH",
        #     sta_translation_file="station_code_translation.txt")

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
        # Do updates for all relevant tribes (also the short ones)
        for xtribe in [tribe, short_tribe, short_tribe2]:
            if len(xtribe) > 0:
                for templ in xtribe:
                    station_trace_counter = Counter(
                        [tr.stats.station for tr in templ.st])
                    for tr in templ.st:
                        # Get trace snr from ispaq-stats - need to calc noise-
                        # amp in relevant frequency band? - when resp removed,
                        # just compare total rms amp in matched_filter
                        try:
                            station_weight_factor = (
                                tr.stats.extra.station_weight_factor)
                        except AttributeError as e:
                            station_weight_factor = 1
                        try:
                            trace_rms_snr = tr.stats.extra.rms_snr
                        except AttributeError:
                            trace_rms_snr = 1
                        if not hasattr(tr.stats, 'extra'):
                            tr.stats.extra = AttribDict()
                        # Higher weight with higher SNR
                        tr.stats.extra.weight = (
                            np.sqrt(trace_rms_snr) *
                            # trace_rms_snr ** (1/3) *
                            # Lower weight with more traces per station
                            1 / (station_trace_counter[
                                tr.stats.station] ** (1/3)) *
                            # Extra weight factor, e.g. for array stations
                            station_weight_factor)

    # temporarily adjust process_length parameters
    daylong = True
    if let_days_overlap:
        daylong = False
        tribes, day_st = prepare_day_overlap(
            [tribe, short_tribe, short_tribe2], day_st, starttime_req,
            endtime_req)
        tribe, short_tribe, short_tribe2 = tribes

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
            stream=day_st, threshold=threshold, trig_int=trig_int,
            threshold_type=threshold_type, overlap='calculate', plot=False,
            plotDir='DetectionPlots', daylong=daylong,
            fill_gaps=True, ignore_bad_data=False, ignore_length=True, 
            pre_processed=pre_processed,
            parallel_process=parallel, cores=cores,
            xcorr_func=xcorr_func, concurrency=concurrency, arch=arch,
            group_size=n_templates_per_run, full_peaks=False,
            extract_detections=False, estimate_origin=True, output_event=False,
            save_progress=False, process_cores=cores, spike_test=False,
            use_weights=use_weights, copy_data=copy_data, **kwargs)
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
                 n_families, n_detections)

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
    Logger.info('Got a party of %s families with %s detections on %s.',
                 n_families, len(party), str(starttime)[0:10])

    # Check if there were no detections and we need to return an empty party.
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
                group_size=n_templates_per_run, process_cores=cores,
                time_difference_threshold=time_difference_threshold,
                detect_value_allowed_reduction=detect_value_allowed_reduction,
                return_party_with_short_templates=True,
                min_n_station_sites=min_n_station_sites,
                use_weights=use_weights, copy_data=copy_data, **kwargs)
            if len(short_tribe2) > 0:
                party, short_party = reevaluate_detections(
                    party, short_tribe2, stream=day_st,
                    threshold=threshold, trig_int=trig_int,
                    threshold_type=threshold_type,
                    re_eval_thresh_factor=re_eval_thresh_factor*0.9,
                    overlap='calculate', plotDir='ReDetectionPlots',
                    plot=False, fill_gaps=True, ignore_bad_data=True,
                    daylong=daylong, ignore_length=True,
                    # min_chans=min_det_chans,
                    pre_processed=pre_processed,
                    parallel_process=parallel, cores=cores,
                    xcorr_func=xcorr_func, arch=arch, concurrency=concurrency,
                    group_size=n_templates_per_run, process_cores=cores,
                    time_difference_threshold=time_difference_threshold,
                    detect_value_allowed_reduction=(
                        detect_value_allowed_reduction * 2),
                    return_party_with_short_templates=True,
                    min_n_station_sites=1,
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

            short_party = calculate_events_for_party(
                short_party, parallel=parallel, cores=cores)
            short_party = short_party.decluster(
                trig_int=trig_int, timing='detect', metric=decluster_metric,
                hypocentral_separation=hypocentral_separation,
                min_chans=min_chans, absolute_values=absolute_values)
            n_det1 = len([d for f in short_party for d in f])
            # 2nd pass to decluster based on origin times
            short_party = short_party.decluster(
                trig_int=trig_int, timing='origin', metric=decluster_metric,
                hypocentral_separation=hypocentral_separation,
                min_chans=min_chans, absolute_values=absolute_values)
            n_det2 = len([d for f in short_party for d in f])
            Logger.info('Short party: After two-stage declustering, %s '
                        '(%s on 1st round) detections are left',  n_det2, n_det1)
            # TODO: maybe the order should be:
            # check array-misdet -decluster party -compare short-party vs party
            if write_party:
                detection_file_name = os.path.join(
                    redetection_path, 'UniqueDet_short_' + current_day_str)
                short_party.write(
                    detection_file_name, format='tar', overwrite=True,
                    max_events_per_file=20)
                short_party.write(
                    detection_file_name + '.csv', format='csv', overwrite=True)

    # Add origins to detections (avoid it in .detect() for speed reasons))
    party = calculate_events_for_party(party, parallel=parallel, cores=cores)
    # Decluster detection and save them to files
    # metric='avg_cor' isn't optimal when one detection may only be good on
    # very few channels - i.e., allowing higher CC than any detection made on
    # many channels
    party = party.decluster(trig_int=trig_int, timing='detect',
                            metric=decluster_metric, min_chans=min_chans,
                            hypocentral_separation=hypocentral_separation,
                            absolute_values=absolute_values)
    n_det1 = len([d for f in party for d in f])
    # 2nd pass to decluster based on origin times - to avoid duplication of
    # events that ran in different batches with different nan-traces.
    party = party.decluster(trig_int=trig_int, timing='origin',
                            metric=decluster_metric, min_chans=min_chans,
                            hypocentral_separation=hypocentral_separation,
                            absolute_values=absolute_values)
    n_det2 = len([d for f in party for d in f])
    Logger.info('After two-stage declustering, %s (%s on 1st round) detections'
                ' are left', n_det2, n_det1)

    if write_party:
        detection_file_name = os.path.join(detection_path,
                                           'UniqueDet' + current_day_str)
        party.write(detection_file_name, format='tar', overwrite=True,
                    max_events_per_file=20)
        party.write(detection_file_name + '.csv', format='csv',
                    overwrite=True)

    if multiplot:
        multiplot_detection(party, tribe, day_st, out_folder='DetectionPlots')
    if dump_stream_to_disk:
        pickle.dump(day_st, open('tmp_st.pickle', "wb"), protocol=-1)
        pickle.dump(party, open('tmp_party.pickle', "wb"), protocol=-1)
    gc.collect()
    return_st = Stream()
    if return_stream:
        return_st = day_st
    Logger.debug('Size of party is: %s', len(pickle.dumps(party)))
    Logger.debug('Size of return_st is: %s', len(pickle.dumps(return_st)))
    if not return_stream and dump_stream_to_disk:
        return
    else:
        return [party, return_st]

