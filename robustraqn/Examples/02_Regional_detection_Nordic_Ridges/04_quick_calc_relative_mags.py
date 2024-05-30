"""
This script (re-)computes relative magnitudes quickly and robustly for events
with associated detections and templates.

See MAIN part for parameter choice
"""


# %%
import os
import glob
from pathlib import Path
import pandas as pd

from joblib import Parallel, delayed

from obspy.io.nordic.core import read_nordic, _write_nordic
from obspy import UTCDateTime, Inventory
from obspy.core.event import WaveformStreamID, Catalog, Comment

from eqcorrscan.core.match_filter import Tribe, Party, Family, Detection
from eqcorrscan.utils.pre_processing import shortproc

from robustraqn.core.load_events import (
    load_event_stream, _read_nordic, taper_trace_segments,
    read_seisan_database)
from robustraqn.core.event_postprocessing import (
    extract_stream_for_picked_events)
from robustraqn.utils.relative_magnitude import (
    compute_relative_event_magnitude)
from robustraqn.utils.spectral_tools import (
    get_updated_inventory_with_noise_models)
from robustraqn.obspy.core import Trace, Stream

import logging
Logger = logging.getLogger(__name__)
LOGFMT = "%(asctime)s\t%(name)40s:%(lineno)s\t%(funcName)20s()\t%(levelname)s\t%(message)s"
logging.basicConfig(level=logging.INFO, format=LOGFMT)


def _write_sfile(event, sfile_path, operator='EQC', wavfile=None,
                 write_to_year_month_folders=True):
    orig_time = event.origins[0].time
    sfile_rea_path = os.path.join(sfile_path,
                                    '{:04}'.format(orig_time.year),
                                    '{:02}'.format(orig_time.month))
    evtype = event.event_descriptions[0].text[0]
    try:
        sfile_name = event.extra['sfile_name']['value']
    except KeyError:
        sfile_name = orig_time.strftime(
            '%d-%H%M-%S' + evtype + '.S%Y%m')
    if write_to_year_month_folders:
        # Make Year/month structure in folder if it does not exist
        Path(sfile_rea_path).mkdir(parents=True, exist_ok=True)
        sfile_out = os.path.join(sfile_rea_path, sfile_name)
    else:
        sfile_out = os.path.join(sfile_path, sfile_name)
    _write_nordic(event, sfile_out, userid=operator, evtype=evtype,
                  wavefiles=wavfile, high_accuracy=False,
                  nordic_format='NEW')


def quick_process_relative_amplitudes(
    sfile, tribe=Tribe(), inv=Inventory(), archives=[], archive_types=[],
    j_event=0,
    remove_response=True, output='DISP',
    mag_min_cc_from_mean_cc_factor=None, mag_min_cc_from_median_cc_factor=1.2,
    min_mag_cc=0.2, absolute_values=True,
    operator='EQC', seisan_wav_path=None,
    write_to_year_month_folders=True, sfile_path=None,
    parallel=False, cores=1):
    """
    Function to quick process relative amplitudes in parallel.
    """
    Logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format=LOGFMT)

    Logger.info('Processing sfile %s', sfile)
    # 1. Read in event and waveform
    select = _read_nordic(sfile, nordic_format='NEW')
    event = select[0]

    for comment in event.comments:
        if 'Detected using template: ' in comment.text:
            template_name = comment.text.split('Detected using template: ')[1]
        if 'detect_val=' in comment.text:
            detect_val = float(comment.text.split('detect_val=')[1])
        if 'EQC_detection_id: ' in comment.text:
            id = comment.text.split('EQC_detection_id: ')[1]
            # 20220425_030736900000
            detect_time = UTCDateTime(id[29:-6]) + float(id[-6:]) / 1000000
        if 'EQC_threshold: ' in comment.text:
            thrs = comment.text.split('EQC_threshold: ')[1]
            threshold_type = thrs[6:18].strip()
            threshold_input = float(thrs[23:31].strip())
            threshold_set = float(thrs[37:45].strip())
        if 'Waveform-filename: ' in comment.text:
            wavfile = comment.text.split('Waveform-filename: ')[1]
        no_chans = len([pick for pick in event.picks])

    # Select template
    try:
        ttribe = Tribe(tribe.select(template_name))
    except IndexError:
        Logger.error(
            'No template %s in tribe for sfile %s, copying sfile without '
            'adjusting magnitude.', template_name, sfile)
        # Overwrite file if it exists
        _write_sfile(event, sfile_path, operator,
                     write_to_year_month_folders, wavfile)
        return
    lowcut = ttribe[0].lowcut
    highcut = ttribe[0].highcut
    samp_rate = ttribe[0].samp_rate

    # Remove previous relative amplitudes from event
    event.amplitudes = [amp for amp in event.amplitudes if amp.type != 'Arel']
    event.picks = [pick for pick in event.picks if pick.phase_hint != 'Arel']
    event.magnitudes = [mag for mag in event.magnitudes
                        if mag.creation_info.agency_id != 'EQC']

    if seisan_wav_path is None:
        seisan_wav_path = os.path.split(sfile)[0]
        # os.path.join(seisan_wav_path, os.path.split(sfile)[0])

    stream = load_event_stream(event, seisan_wav_path=seisan_wav_path,
                               selected_stations=selected_stations)

    # Remove relative-magnitude comment
    event.comments = [comment for comment in event.comments
                      if ' magnitude-delta:' not in comment.text]
    for pick in event.picks:
        # Make the event picks a tiny bit earlier so that the time window
        # matches better compared to the template trace
        pick.time -= 0.05

    # May need to reload stream from archive and write a new wavfile if missing
    if stream is None or len(stream) == 0:
        Logger.error('No waveform data for sfile %s', sfile)
        # 2. Reconstruct a detection from event+waveform and detection
        #    information
        detection = Detection(
            template_name=template_name, detect_time=detect_time,
            no_chans=no_chans, detect_val=detect_val,
            threshold=threshold_set, typeofdet='corr',
            threshold_type=threshold_type, threshold_input=threshold_input,
            chans=[pick.waveform_id.id for pick in event.picks],
            event=event, id=event.resource_id.id)
        party = Party(families=[Family(
            template=ttribe[0], detections=[detection], catalog=[event])])
        wavefiles, detection_list = extract_stream_for_picked_events(
            Catalog([event]), party,template_path=None, 
            archives=archives, archive_types=archive_types,
            original_stats_stream=Stream(), det_tribe=ttribe,
            request_fdsn=False, wav_out_dir=sfile_path,
            write_waveforms=True, write_to_year_month_folders=True,
            extract_len=800,
            all_chans_for_stations=[], sta_translation_file=None,
            only_relevant_stations=True, parallel=False, cores=1)
        wavfile = wavefiles[0]
        event.comments.append(
            Comment(text='Waveform-filename: %s' % Path(wavfile).parent))
        stream = load_event_stream(event, seisan_wav_path=seisan_wav_path,
                                   selected_stations=selected_stations)
        if stream is None or len(stream) == 0:
            _write_sfile(event, sfile_path)
            Logger.error('Still no waveform data for sfile %s after extra'
                         'request, copying Sfile ', sfile)
            return

    # Preprocess stream
    nyquist_f = samp_rate / 2
    starttime = stream[0].stats.starttime
    pre_filt = [0.1, 0.2, 0.9 * nyquist_f, 0.95 * nyquist_f]

    # Same processing as for making templates:
    stream = stream.mask_consecutive_zeros(min_run_length=None)
    # Taper all the segments
    stream = taper_trace_segments(stream)
    if remove_response:
        stream = stream.try_remove_responses(
            inv, output=output, taper_fraction=0.1,
            pre_filt=pre_filt, parallel=parallel, cores=cores,
            gain_traces=True, water_level=60)
    stream = stream.detrend(type='simple')

    # TODO: CHECK!!!!!!!!!!!!!!
    # init_processing leads to ~25 % different amplitudes compare to the 
    # processing in create_templates. That is a problem that I need to fix.
    # But this script here appears to produce the correct 1:1 relative 
    # magnitudes for selfdetection, so issue may be somewhere in initprocessing
    # of picking method.

    # stream = stream.init_processing_w_rotation(
    #     starttime=starttime, endtime=stream[0].stats.endtime,
    #     remove_response=remove_response, output=output,
    #     inv=inv, pre_filt=pre_filt,
    #     gain_traces=True, water_level=60, # 10
    #     sta_translation_file='',
    #     parallel=False, cores=cores, n_threads=1,
    #     # parallel=parallel, cores=cores, n_threads=1,
    #     suppress_arraywide_steps=False,
    #     min_segment_length_s=10, max_sample_rate_diff=1,
    #     taper_fraction=0.1, # taper_fraction=0.005,
    #     detrend_type='simple', downsampled_max_rate=samp_rate,
    #     std_network_code=None, std_location_code=None, std_channel_prefix=None,
    #     #std_network_code="NS", std_location_code="00", std_channel_prefix="BH")
    # )

    stream = shortproc(
        stream, lowcut=lowcut, highcut=highcut,
        filt_order=4, samp_rate=samp_rate,
        starttime=starttime, parallel=parallel, num_cores=cores,
        ignore_length=False, seisan_chan_names=False, fill_gaps=True,
        ignore_bad_data=True, fft_threads=1)
    pre_processed = True

    # Adjust pick trace ids in Template to match stream
    for template in ttribe:
        for pick in template.event.picks:
            # Logger.info('%s', pick.waveform_id.id)
            matching_chan = [
                tr for tr in stream
                if tr.stats.station == pick.waveform_id.station_code
                and tr.stats.channel[-1] == pick.waveform_id.channel_code[-1]]
            templ_traces = template.st.select(id=pick.waveform_id.id)
            # At this point there should be only one matching channel
            if len(list(set([tr.id for tr in matching_chan]))) > 1:
                raise ValueError('More than one matching channel found')
            elif len(matching_chan) == 0:
                continue
            pick.waveform_id = WaveformStreamID(
                *matching_chan[0].id.split('.'))
            # also change trace id in template
            for tr in templ_traces:
                tr.id = matching_chan[0].id

    # 2. Reconstruct a detection from event+waveform and detection information
    detection = Detection(
        template_name=template_name, detect_time=detect_time,
        no_chans=no_chans, detect_val=detect_val,
        threshold=threshold_set, typeofdet='corr',
        threshold_type=threshold_type, threshold_input=threshold_input,
        chans=[pick.waveform_id.id for pick in event.picks],
        event=event, id=event.resource_id.id)
    detection.st = stream

    # 3 . compute relative magnitude
    try:
        detection_event, pre_processed = compute_relative_event_magnitude(
            detection=detection, detected_event=event,
            j_ev=j_event, day_st=None, party=Party(), tribe=ttribe,
            templ2=None, detection_template_names=[], write_events=False,
            accepted_magnitude_types=['ML', 'Mw', 'MW'],
            accepted_magnitude_agencies=['BER', 'NAO'],
            mag_out_dir=None, min_cc=min_mag_cc,
            absolute_values=absolute_values,
            min_cc_from_mean_cc_factor=mag_min_cc_from_mean_cc_factor,
            min_cc_from_median_cc_factor=mag_min_cc_from_median_cc_factor,
            return_correlations=True, correct_mag_bias=False,
            remove_response=remove_response, output=output,
            pre_processed=pre_processed,
            parallel=parallel, cores=cores, n_threads=n_threads,
            min_amp_ratio_log_deviation=0.1) #, **kwargs)
    except Exception as e:
        Logger.error(
            'Failed computing relative magnitude for sfile %s: %s', sfile, e)
        if write_sfiles:
            # Overwrite file if it exists
            _write_sfile(event, sfile_path, operator,
                         write_to_year_month_folders, wavfile)
            return
    # Copy over magnitudes from detection-event to export-event
    event.magnitudes += detection_event.magnitudes
    event.station_magnitudes += detection_event.station_magnitudes
    # event.amplitudes += detection_event.amplitudes
    for amplitude in event.amplitudes:
        if amplitude.pick_id is None:  # Cannot resolve seed-id
            continue
        # May need to update pick resource id in amplitude
        ref_pick = amplitude.pick_id.get_referred_object()
        if ref_pick not in event.picks:
            pick_ids = [p.resource_id for p in event.picks
                        if p.waveform_id == ref_pick.waveform_id]
            for pick_id in pick_ids:
                amplitude.pick_id = pick_id
    for comment in detection_event.comments:
        if comment not in event.comments:
            event.comments.append(comment)
        if 'magnitude-delta' in comment.text:
            # Comment line:
            # Template magnitude:  3.90wBER, magnitude-delta:   0.29, std:  0.07, n:    2   3
            # Extract delta mag:
            delta_mag = float(
                comment.text.split('magnitude-delta:')[1].split(',')[0])

    # Do a sanity check:
    # If the delta magnitude is above 0.0 or even -0.2 (i.e., detection larger
    # than template)
    # template_magnitude = ttribe[0].event.magnitudes[-1].mag
    # if detection_event.magnitudes[-1] >= delta_mag
    if delta_mag >= 0.0:
        # then check the relative amplitudes measured at the three closest
        # stations
        rel_amps = [amp for amp in event.amplitudes if amp.type == "Arel"]
        amp_distances = []
        closest_amps = []
    # - if none of these are above close to 1.0,
    # - then the detection is probably a false detection

    if write_sfiles:
        # Overwrite file if it exists
        _write_sfile(event, sfile_path, operator, write_to_year_month_folders,
                     wavfile)


# %% 
############################# MAIN #########################################

if __name__ == "__main__":
    # Parameters for quickly calculating relative magnitudes in a robust way
    parallel = True  # whether to run parallelization at all
    event_parallel = True  # whether to use event-based parallelization
    cores = 40
    n_threads = 1
    # see compute_relative_event_magnitude
    mag_min_cc_from_mean_cc_factor = None
    mag_min_cc_from_median_cc_factor = 1.2
    min_mag_cc = 0.2  # see compute_relative_event_magnitude
    output = 'VEL'  # physical unit of traces to be used for relative amplit.
    remove_response = True
    absolute_values = True
    samp_rate = 20.0
    lowcut = 3.0
    highcut = 9.9
    operator = 'EQC'  # Output operator code

    check_supported_templates = True  # Whther to check supported templates
    write_sfiles = True  # Whether to write out templates

    # File locations
    seisan_wav_path = 'Sfiles_10'
    templ_rea_path = '../Seisan/INTEU/'
    rea_path = 'Sfiles_10_updated_otimes2'
    sfile_path = 'Sfiles_10_updated_otimes_mags'
    write_to_year_month_folders = True
    archives=['/nas/seismo-wav/SLARCHIVE'],
    archive_types=['SDS'],


    inv_file = '~/Documents2/ArrayWork/Inventory/NorSea_inventory.xml'
    # inv = read_inventory(os.path.expanduser(inv_file))
    inv = get_updated_inventory_with_noise_models(
        inv_file=os.path.expanduser(inv_file),
        pdf_dir=os.path.expanduser('~/repos/ispaq/WrapperScripts/PDFs/'),
        check_existing=True,
        outfile=os.path.expanduser(
            '~/Documents2/ArrayWork/Inventory/inv.pickle'))

    det_sta_f = open('stations_selection.dat', "r+")
    selected_stations = [line.strip() for line in det_sta_f.readlines()]
    det_sta_f.close()
    # selected_stations = ['DAG', 'KBS', 'ROEST', 'GILDE', 'LOSSI']

    tribe = Tribe()
    tribe._read_from_folder(
        'TemplateObjects/Unpack/Templates_min13tr_15652', cores=cores)

    # Update event magnitudes in tribe events:
    for templ in tribe:
        # templ.event.magnitudes = []
        # templ.event.station_magnitudes = []
        sfile_name = templ.event.extra['sfile']['value']
        templ_sfile = os.path.join(
            templ_rea_path, sfile_name[-6:-2], sfile_name[-2:], sfile_name)
        try:
            upd_event = read_nordic(templ_sfile)[0]
        except AttributeError:
            Logger.info('Failed reading sfile %s', templ_sfile)
            continue
        templ.event.magnitudes = upd_event.magnitudes
        templ.event.station_magnitudes = upd_event.station_magnitudes

    # %%
    sfiles = glob.glob(rea_path + '/????/??/*.S??????')
    sfiles2 = glob.glob(os.path.join(sfile_path, '????/??/*.S??????'))

    # Remove the Sfile-main folder from the paths:
    try:
        sfiles_df = pd.DataFrame(pd.DataFrame(sfiles)[0].str.split('/')
                                ).applymap(lambda x: '/'.join(x[1:]))
        sfiles2_df = pd.DataFrame(pd.DataFrame(sfiles2)[0].str.split('/')
                                ).applymap(lambda x: '/'.join(x[1:]))
        # Find the difference (the sfiles that haven't been corrected for mag)
        sfiles3 = list(set(sfiles_df[0]).difference(set(sfiles2_df[0])))
        sfiles = [os.path.join(rea_path, sfile) for sfile in sfiles3]
    except KeyError:  # Most likely no files in sfiles or sfiles2
        pass

    sfiles.sort(key = lambda x: x[-6:] + x[-19:-9], reverse=False)

    supported_template_names = [templ.name for templ in tribe]
    template_names = []
    origin_times = []
    station_match_strs = []
    ok_sfiles = []

    for j_event, sfile in enumerate(sfiles):
        select = _read_nordic(sfile, nordic_format='NEW')
        event = select[0]

        # Checks whether all templates in the Sfiles are found in the supplied
        # ttribe
        if check_supported_templates:
            break_event_proc = False
            for comment in event.comments:
                if 'Detected using template: ' in comment.text:
                    detection_template = comment.text.split(
                        'Detected using template: ')[1]
                    if detection_template not in supported_template_names:
                        break_event_proc = True
            if break_event_proc:
                # _write_sfile(event, sfile_path, operator,
                #          write_to_year_month_folders, wavfile)
                Logger.info('Skipping event processing for sfile %s.', sfile)
                continue
        # add sfile to processing list
        ok_sfiles.append(sfile)

        origin_times.append((event.preferred_origin() or event.origins[0]).time)
        ev_stations = list(set([pick.waveform_id.station_code
                                for pick in event.picks]))
        ev_station_fnmatch_str = '@(' + '|'.join(ev_stations) + ')'
        station_match_strs.append(ev_station_fnmatch_str)

        # 2. Find associated template name
        for comment in event.comments:
            if 'Detected using template: ' in comment.text:
                template_name = comment.text.split('Detected using template: ')[1]
                break
        template_names.append(template_name)


    # %%
    # Serial execution
    if not event_parallel:
        quick_process_relative_amplitudes(
            sfile, tribe=tribe, inv=inv, j_event=j_event,
            archives=archives, archive_types=archive_types,
            remove_response=remove_response, output=output,
            mag_min_cc_from_mean_cc_factor=mag_min_cc_from_mean_cc_factor,
            mag_min_cc_from_median_cc_factor=mag_min_cc_from_median_cc_factor,
            min_mag_cc=0.2, absolute_values=True,
            operator='EQC', seisan_wav_path=seisan_wav_path,
            write_to_year_month_folders=True, sfile_path=sfile_path,
            parallel=False, cores=1)

    # Parallel execution
    if event_parallel:
        event_ids = range(len(ok_sfiles))
        Logger.info('Starting parallel processing of %s events', len(sfiles))
        Parallel(n_jobs=cores)(delayed(
            quick_process_relative_amplitudes)(
                sfile, tribe=Tribe(tribe.select(template_name)),
                inv=inv.select(time=otime, station=station_match_str),
                j_event=j_event,
                remove_response=remove_response, output=output,
                mag_min_cc_from_mean_cc_factor=mag_min_cc_from_mean_cc_factor,
                mag_min_cc_from_median_cc_factor=mag_min_cc_from_median_cc_factor,
                min_mag_cc=0.2, absolute_values=True,
                operator='EQC', seisan_wav_path=seisan_wav_path,
                write_to_year_month_folders=True, sfile_path=sfile_path,
                parallel=False, cores=1)
            for sfile, template_name, otime, station_match_str, j_event in
            zip(ok_sfiles, template_names, origin_times, station_match_strs,
                event_ids))
        Logger.info('Job completed successfully.')

# In obspy.core.inventory.Network, replace fnmatch with wcmatch to make this
# work:
# inv.select(time=event.preferred_origin().time, station=station_match_str)

# %%
