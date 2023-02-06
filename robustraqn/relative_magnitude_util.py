

# %%
import os, glob, matplotlib, sys, difflib, pickle
from os import times
import pandas as pd
from importlib import reload
import numpy as np
from joblib import Parallel, delayed, parallel_backend

from timeit import default_timer
import logging
Logger = logging.getLogger(__name__)

from obspy import UTCDateTime, Stream
from obspy.core.event import (Catalog, Magnitude, StationMagnitude, Amplitude,
                              StationMagnitudeContribution, WaveformStreamID,
                              QuantityError)
from obspy.core.inventory.inventory import Inventory
from obspy.core.event import ResourceIdentifier, CreationInfo, Event, Comment
from obspy.core.event.header import (
    AmplitudeCategory, AmplitudeUnit, EvaluationMode, EvaluationStatus,
    ATTRIBUTE_HAS_ERRORS)
from obspy.io.nordic.core import read_nordic, write_select, _write_nordic

from obsplus.events.validate import attach_all_resource_ids

from multiprocessing import Pool, cpu_count, current_process, get_context
from eqcorrscan.utils.correlate import pool_boy
from eqcorrscan.core.match_filter import Tribe, Template, read_detections
from eqcorrscan.core.match_filter.party import Party
from eqcorrscan.utils.mag_calc import relative_magnitude
from eqcorrscan.utils.pre_processing import shortproc

from robustraqn.obspy_utils import _quick_copy_stream


def _remove_uncorrected_traces(stream=Stream()):
    """
    Removes those traces from a stream where the response was not removed.
    """
    remove_traces = []
    for trace in stream:
        remove_tr = True
        # TODO: write and read processing info from tribe json / xml
        # Right now, tribes / parties do not save processing information.
        if hasattr(trace.stats, 'processing'):
            for proc_info in trace.stats.processing:
                if 'remove_response' in proc_info:
                    remove_tr = False
            if remove_tr:
                remove_traces.append(trace)
    for rm_trace in remove_traces:
        Logger.debug(
            'Removing trace %s for mag-calc because instrument response was ',
            'not removed.', rm_trace)
        stream.remove(rm_trace)
    return stream


def compute_relative_event_magnitude(
        detection, detected_event=None, j_ev=0, day_st=Stream(), party=Party(),
        tribe=Tribe(), templ2=Template(),
        detection_template_names=[], write_events=False, mag_out_dir=None,
        accepted_magnitude_types=['ML', 'Mw', 'MW'],
        accepted_magnitude_agencies=['BER', 'NOA'],
        magnitude_agency_type_pref=[
            ('BER', 'ML'), ('BER', 'Mw'), ('BER', 'MW'), ('NOA', 'ML')],
        min_snr=1.1, min_cc=0.25, min_cc_from_mean_cc_factor=None,
        min_n_relative_amplitudes=2, check_rel_amp_deviations=True,
        noise_window_start=-40, noise_window_end=-29.5,
        signal_window_start=-0.5, signal_window_end=10,
        use_s_picks=True, correlations=None, shift=0.35,
        return_correlations=True, correct_mag_bias=False,
        pre_processed=False, remove_response=False, output='DISP',
        parallel=False, cores=1, n_threads=1, **kwargs):
    """
    Compute relative magnitudes with specific criteria on SNR, minimum number
    of amplitude measurements etc.
    """
    if detection is None and detected_event is None:
        msg = ("Detection and detected event cannot both be unknown when " +
               " computig relative magnitude.")
        raise NotImplementedError(msg)

    if detected_event is None:
        detected_event = detection.event
    # keep input events safe
    # detected_event = detected_event.copy()
    Logger.debug('Event %s: start function', j_ev)
    attach_all_resource_ids(detected_event)
    Logger.debug('Event %s: attached resIDs', j_ev)

    Logger.info('Event %s: Trying to compute relative magnitude for %s',
                j_ev, detected_event.short_str())

    # If detection is not known: find detection based on detected event
    if detection is None:
        Logger.debug('Detection is None, retrieving detection from party.')
        detection_id = None
        # find the template-name with which the event has been detected
        for comment in detected_event.comments:
            if 'EQC_detection_id: ' in comment.text:
                detection_id = comment.text.lstrip('EQC_detection_id: ')
            elif 'eqcorrscan_template_' in comment.text:
                template_name2a = comment.text.lstrip('eqcorrscan_template_')
            elif 'Detected using template:' in comment.text:
                template_name2a = comment.text.lstrip(
                    'Detected using template: ')
        # now find the "template" in the large tribe (i.e. processed waveform)
        # that corresponds to the detected event
        template_name2b = templ2.name
        if template_name2a != template_name2b:
            Logger.error('Event  %s: Problem with template names! ', j_ev)
            return detected_event, pre_processed
        template_name2 = template_name2a

        # Find the name for the template that gives the cloest match 
        # between detection time and the earliest pick in the event
        day_detection_times = np.array(
            [d.detect_time for f in party for d in f])
        time_diffs = (abs(day_detection_times - min(
            [pick.time for pick in detected_event.picks])))
        if len(time_diffs) == 0:
            time_diffs = (
                abs(day_detection_times - (detected_event.preferred_origin() or
                                        detected_event.origins[0]).time))
        if len(time_diffs) == 0:
            Logger.error('Event  %s: no time information for event origin and '
                        'picks, cannot find matching detection.')
            return detected_event, pre_processed

        index = np.argmin(time_diffs)
        k = 0
        detection = None
        for family in party:
            for detection in family:
                if k == index:
                    break
                k += 1
            if k == index:
                break
    # Now detection and detection event are known
    template_name1 = detection.template_name

    Logger.debug('Detection is: %s, template is: %s', detection.id,
                 template_name1)
    # template_name1 should be read from Sfiles, but the writing of comments
    # with Nordic format is not yet supported (well now it is..)
    try:
        templ1 = tribe.select(template_name1)
    except IndexError:
        search_templ_name = template_name1
        template_name1 = difflib.get_close_matches(
            search_templ_name, detection_template_names, n=1)[0]
        Logger.warning(
            'Event  %s: Could not find exact match for template name %s, but '
            + 'found one that is similar: %s', j_ev, search_templ_name,
            template_name1)
        templ1 = tribe.select(template_name1)

    # templ2 = tribe_detected.select(template_name2)
    Logger.debug('Event %s: found matching template', j_ev)
    if len(day_st) > 0:
        detection_st = day_st
    elif hasattr(detection, "st"):
        detection_st = detection.st
    else:
        try:
            detection_st = _quick_copy_stream(templ2.st)
            pre_processed = True
        except (ValueError, AttributeError):
            msg = ("Need stream of detection to compute relative magnitude " +
                   " - attach stream to detection or supply new template")
            raise NotImplementedError(msg)

    # May nee to filter the data so that they match the template
    if not pre_processed:
        detection_st = shortproc(
            st=detection_st, lowcut=templ1.lowcut,
            highcut=templ1.highcut, filt_order=templ1.filt_order,
            samp_rate=templ1.samp_rate, parallel=parallel, num_cores=cores,
            fft_threads=n_threads)
    
    # When response should have been removed, check that it was indeed removed
    # from both template and detection - otherwise can't compute relative
    # amplitudes.
    if remove_response:
        detection_st = _remove_uncorrected_traces(detection_st)
        templ1_st = _quick_copy_stream(_remove_uncorrected_traces(templ1.st))
    else:
        templ1_st = _quick_copy_stream(templ1.st)

    Logger.debug(
        'Measure relative magnitude from streams with %s and %s traces',
        len(templ1.st), len(detection_st))
    if min_cc_from_mean_cc_factor is not None:
        min_cc = min(abs(detection.detect_val / detection.no_chans
                         * min_cc_from_mean_cc_factor),
                     min_cc)
        Logger.debug('Event %s: Setting minimum cc-threshold for relative '
                     'magnitude to %.5f', j_ev, min_cc)
    delta_mag, correlations = relative_magnitude(
        templ1_st, detection_st,
        templ1.event, detected_event,
        noise_window=(noise_window_start, noise_window_end),
        signal_window=(signal_window_start, signal_window_end),
        min_snr=min_snr, min_cc=min_cc, use_s_picks=use_s_picks,
        correlations=None, shift=shift,
        return_correlations=return_correlations,
        correct_mag_bias=correct_mag_bias,
        check_rel_amp_deviations=check_rel_amp_deviations, **kwargs)
        #  correct_mag_bias=True)
        # magnitude_method="UnnormalizedCC"

    mag_ccs = list()
    delta_mag_corr = dict()
    for seed_id, _delta_mag in delta_mag.items():
        cc = correlations[seed_id]
        mag_ccs.append(cc)
        # _delta_mag = _delta_mag / cc
        delta_mag_corr[seed_id] = _delta_mag

    Logger.debug('Event %s: computed %s delta-magnitudes', j_ev,
                 len(mag_ccs))

    # delta_mag_S = relative_magnitude(
    #     templ1_st, templ2.st, templ1.event, detected_event,
    #     noise_window=(-50, -30), signal_window=(-0.5, 20), min_snr=min_snr,
    #     min_cc=0.7, use_s_picks=True, correlations=None, shift=0.4,
    #     return_correlations=False, correct_mag_bias=True)
    # delta_mags.append([delta_mag])
    
    # if len(delta_mag) > 0 and len(delta_mag_S) > 0:
    #     break

    try:
        prev_mag = None
        prev_mags = []
        # Check if there are magnitudes for preferred agency + type combinations
        prev_mag_agency_type_tuple = [
            ((m.creation_info.agency_id, m.magnitude_type), m)
            for m in templ1.event.magnitudes if m.creation_info]
        previous_magnitudes = [
            prev_mag[1] for prev_mag in prev_mag_agency_type_tuple
            if prev_mag[0] in magnitude_agency_type_pref]
        # If there are no magnitudes for accepted agencies / types, allow others.
        if len(previous_magnitudes) == 0:
            if len(prev_mags) == 0:
                previous_magnitudes = [
                    m for m in templ1.event.magnitudes
                    if m.magnitude_type in accepted_magnitude_types
                    and (m.creation_info.agency_id in accepted_magnitude_agencies
                        if m.creation_info else True)]
        prev_mags = [m.mag for m in previous_magnitudes]
    
    # try:
    #     previous_magnitudes = []
    #     for mag_pref in magnitude_agency_type_pref:
    #         if len(previous_magnitudes) 
    #         try:
    #             previous_magnitudes = [
    #                 m for m in templ1.event.magnitudes
    #                 if m.magnitude_type == mag_pref[1]
    #                 and (m.creation_info.agency_id == mag_pref[0]
    #                      if m.creation_info else False)]
    #         except Exception as e:
    #             pass
        # # First try to be string about accepted magnitude agencies
        # try:
        #     previous_magnitudes = [
        #         m for m in templ1.event.magnitudes
        #         if m.magnitude_type in accepted_magnitude_types
        #         and (m.creation_info.agency_id in accepted_magnitude_agencies
        #                 if m.creation_info else False)]
        # except Exception as e:
        #     pass
        # If there are no magnitudes for accepted agencies, allow others.
        # if len(prev_mags) == 0:
        #     previous_magnitudes = [
        #         m for m in templ1.event.magnitudes
        #         if m.magnitude_type in accepted_magnitude_types
        #         and (m.creation_info.agency_id in accepted_magnitude_agencies
        #             if m.creation_info else True)]
        # prev_mags = [m.mag for m in previous_magnitudes]
    except Exception as e:
        Logger.warning(e)
        Logger.warning(
            "Event  %s: No template magnitude, relative magnitudes cannot be "
            "computed for %s", j_ev, detected_event.short_str())
        return detected_event, pre_processed
    if len(prev_mags) > 0:
        prev_mag = np.mean(prev_mags)
    else:
        Logger.warning(
            "Event %s: No template magnitudes, relative magnitudes cannot be "
            "computed for %s", j_ev, detected_event.short_str())
        return detected_event, pre_processed

    Logger.debug('Event %s: found %s previous template-magnitudes (%s)',
                 j_ev, len(prev_mags), str(prev_mags))
    # new_mags_rel = [delta_mag[key] for key in delta_mag]
    # new_mag = prev_mag + np.mean(new_mags_rel)
    
    # if not np.isnan(new_mag) and len(new_mags_rel) > 0:
    #     detected_event.magnitudes.append(Magnitude(
    #         # "resource_id"=ResourceIdentifier,
    #         #"mag", float=ATTRIBUTE_HAS_ERRORS,
    #         mag=new_mag,
    #         magnitude_type='ML',
    #         origin_id=detected_event.preferred_origin().resource_id,
    #         # "method_id"=ResourceIdentifier,
    #         station_count=len(delta_mag),
    #         # "azimuthal_gap"=float,
    #         evaluation_mode=EvaluationMode('automatic'),
    #         evaluation_status=EvaluationStatus('preliminary'),
    #         creation_info=CreationInfo(agency_id='BER', author='FH',
    #                                    creation_time=UTCDateTime(),
    #                                    version=None)))

    # [sm.mag for sm in detected_event.station_magnitudes]
    delta_mags = [_delta_mag[1] for _delta_mag in delta_mag_corr.items()]
    # av_mag = prev_mag + np.dot(
    #     np.array(delta_mags), np.array(mag_ccs)) / np.sum(mag_ccs)z
    av_mag = np.nan
    if len(delta_mags) > 0:
        # av_mag = prev_mag + np.mean(delta_mags)
        # better use median to exclude outliers?!
        av_delta_mag = np.median(delta_mags)
        av_mag = prev_mag + av_delta_mag
    else:
        return detected_event, pre_processed

    # Add station magnitudes
    sta_contrib = []
    amp_contrib = []
    for seed_id, _delta_mag in delta_mag_corr.items():
        if np.isnan(prev_mag) or np.isnan(_delta_mag):
            continue
        # Add relative amplitude measurement:
        rel_amp = 10 ** _delta_mag
        # Adding amplitude mainly useful if there is a related pick
        pick_ids = [p.resource_id for p in detected_event.picks
                    if p.waveform_id.get_seed_string() == seed_id]
        for pick_id in pick_ids:
            amp = Amplitude(pick_id=pick_id, type='Arel',
                            generic_amplitude=rel_amp, unit='dimensionless',
                            waveform_id=WaveformStreamID(seed_string=seed_id),
                            magnitude_hint='MR')
            amp_contrib.append(amp)
        if len(pick_ids) == 0:
            amp = Amplitude(generic_amplitude=rel_amp, unit='dimensionless',
                            waveform_id=WaveformStreamID(seed_string=seed_id),
                            type='Arel', magnitude_hint='MR')
            amp_contrib.append(amp)
        # Add station magnitude contribution
        station_mag_residual = _delta_mag - av_delta_mag
        sta_mag = StationMagnitude(
            mag=prev_mag + _delta_mag,
            amplitude_id=amp.resource_id,
            magnitude_type='MR', station_magnitude_type='MR',
            mag_errors=QuantityError(uncertainty=station_mag_residual),
            method_id=ResourceIdentifier("relative"),
            waveform_id=WaveformStreamID(seed_string=seed_id),
            creation_info=CreationInfo(
                # agency_id='BER',
                agency_id='eqcorrscan.core.lag_calc',
                author="EQcorrscan",
                creation_time=UTCDateTime()))
        detected_event.station_magnitudes.append(sta_mag)
        sta_contrib.append(StationMagnitudeContribution(
            station_magnitude_id=sta_mag.resource_id,
            weight=1.))
    detected_event.amplitudes += amp_contrib

    Logger.debug(
        'Event %s: created %s stationMagnitudes (%s)', j_ev,
        len(sta_contrib),
        str(['{0:.2f}'.format(stamag.mag)
             for stamag in detected_event.station_magnitudes]))

    Logger.debug('Event %s: The %s delta-magnitudes are: %s', j_ev,
                 len(delta_mags),
                 str(['{0:.2f}'.format(dm) for dm in delta_mags]))

    if not np.isnan(av_mag) and len(delta_mags) >= min_n_relative_amplitudes:
        # Compute average magnitude
        detected_event.magnitudes.append(
            Magnitude(
                mag=av_mag, magnitude_type='ML',
                origin_id=(detected_event.preferred_origin() or
                        detected_event.origins[0]).resource_id,
                method_id=ResourceIdentifier("relative"),
                station_count=len(delta_mag),
                evaluation_mode=EvaluationMode('automatic'),
                station_magnitude_contributions=sta_contrib,
                creation_info=CreationInfo(
                    agency_id='EQC', author="EQcorrscan",
                    creation_time=UTCDateTime())))
        Logger.info(
            'Event no. %s, %s: added median magnitude %.3f for %s station '
            'magnitudes.', j_ev, detected_event.short_str(),
            detected_event.magnitudes[-1].mag, len(sta_contrib))
        delta_mag = np.round(np.median(delta_mags), 2) + 0
        mag_str = '    '
        if len(previous_magnitudes) == 1:
            try:
                agency_id = previous_magnitudes[0].creation_info.agency_id
            except AttributeError:
                pass
            if agency_id is None:
                agency_id = '   '
            if len(agency_id) > 3:
                agency_id = agency_id[0:3]
            try:
                mag_type = previous_magnitudes[0].magnitude_type
                mag_type = mag_type[-1]  # Only retain last letter of magtype
            except (AttributeError, IndexError):
                mag_type = ' '
            mag_str = mag_type + agency_id
        mag_comment = Comment(
            text=('Template magnitude: {prev_mag: 5.2f}{mag_str:4s},' +
                  ' magnitude-delta: {delta_mag: 6.2f}').format(
                      prev_mag=prev_mag, mag_str=mag_str, delta_mag=delta_mag),
            creation_info=CreationInfo(agency_id='RR', author='RR'))
        detected_event.comments.append(mag_comment)

    # Write out Nordic files:
    if write_events:
        if mag_out_dir and not os.path.exists(mag_out_dir):
            os.makedirs(mag_out_dir)
        _write_nordic(
            detected_event, filename=None, userid='fh', evtype='L',
            outdir=mag_out_dir, overwrite=True, high_accuracy=True,
            nordic_format='NEW')
        Logger.debug('Event %s: wrote Nordic file', j_ev)

    # Avoid reprocessing full day-stream on next call to relative mag util
    if len(day_st) > 0:
        pre_processed = True
    else:
        pre_processed = False
    return detected_event, pre_processed