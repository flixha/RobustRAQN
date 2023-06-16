#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main wrapper to create templates from events according to a range of criteria
in parallel. Takes into account:
  - data quality metrics
  - event observation setup and event location uncertainties
  - seismic arrays

@author: Felix Halpaap
"""

# %%
import os
import glob
# from importlib import reload
from multiprocessing import Pool, cpu_count, get_context
from multiprocessing.pool import ThreadPool
from joblib import Parallel, delayed
import pandas as pd
from timeit import default_timer
import numpy as np

from obspy.core.event import Catalog, read_events
from obspy.core.utcdatetime import UTCDateTime
from obspy.geodetics.base import gps2dist_azimuth
from obspy.geodetics import kilometers2degrees, degrees2kilometers
from obspy.core.event import Event
from obspy.io.nordic.core import read_nordic
from obspy.core.inventory.inventory import Inventory
from obspy.core.util.attribdict import AttribDict

from obsplus import events_to_df
from obsplus.stations.pd import stations_to_df

from eqcorrscan.utils import pre_processing
from eqcorrscan.core import template_gen
from eqcorrscan.utils.catalog_utils import filter_picks
from eqcorrscan.core.match_filter import Template, Tribe
from eqcorrscan.utils.plotting import pretty_template_plot
from eqcorrscan.utils.correlate import pool_boy

from robustraqn.core.load_events import (
    get_all_relevant_stations, load_event_stream, check_template,
    prepare_picks, fix_phasehint_capitalization, taper_trace_segments)
from robustraqn.core.seismic_array import (
    extract_array_picks, add_array_station_picks, get_station_sites,
    LARGE_APERTURE_SEISARRAY_PREFIXES, get_updated_stations_df,
    mask_array_trace_offsets, get_array_stations_from_df,
    SEISARRAY_STATIONS)
from robustraqn.utils.bayesloc import update_cat_from_bayesloc
from robustraqn.utils.spectral_tools import st_balance_noise
from robustraqn.utils.quality_metrics import (
    create_bulk_request, get_parallel_waveform_client)
from robustraqn.utils.obspy import _quick_copy_stream
from robustraqn.obspy.clients.filesystem.sds import Client
from robustraqn.obspy.core import Trace, Stream

import logging
Logger = logging.getLogger(__name__)


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


def _quick_tribe_copy(tribe):
    """Function to quickly copy tribe with effiecient stream copy.

    :param tribe: Tribe holding templates that needs to be copied.
    :type tribe: class:`eqcorrscan.core.match_filter.Tribe`
    :return: Tribe
    :rtype: class:`eqcorrscan.core.match_filter.Tribe`
    """
    new_template_list = []
    for template in tribe:
        new_template_st = _quick_copy_stream(template.st)
        new_template = Template(
            name=template.name, st=new_template_st, lowcut=template.lowcut,
            highcut=template.highcut, samp_rate=template.samp_rate,
            filt_order=template.filt_order,
            process_length=template.process_length, prepick=template.prepick,
            event=template.event)
        new_template_list.append(new_template)
    new_tribe = Tribe(new_template_list)
    return new_tribe


def _shorten_tribe_streams(
        tribe, trace_offset=0, tribe_len_pct=0.2, max_tribe_len=None,
        min_n_traces=0, write_out=False, make_pretty_plot=False,
        prefix='short', noise_balancing=False, apply_agc=False,
        write_individual_templates=False, check_len_strict=True,
        inplace=False, equalize_scaling=False, max_taper_percentage=0.05):
    """
    Create shorter templates from a tribe of longer templates

    :param tribe: Tribe with long templates
    :type tribe: class:`eqcorrscan.core.match_filter.Tribe`
    :param trace_offset:
        offset as ratio of the template's length at which to start the new
        templates, defaults to 0
    :type trace_offset: int, optional
    :param tribe_len_pct:
        length of the new shortened templates, as ratio of the long template's
        length, defaults to 0.2
    :type tribe_len_pct: float, optional
    :param max_tribe_len:
        maximum absolute length in s for new templates, defaults to None
    :type max_tribe_len: float, optional
    :param min_n_traces:
        minimum number of traces to keep template, defaults to 0
    :type min_n_traces: int, optional
    :param write_out: Whether to write new tribe to file, defaults to False
    :type write_out: bool, optional
    :param make_pretty_plot: whether to plot new template, defaults to False
    :type make_pretty_plot: bool, optional
    :param prefix:
        name-prefix for writing out the shortened tribe, defaults to 'short'
    :type prefix: str, optional
    :param noise_balancing:
        whether the tribe was noise-balanced (will be part of filename),
        defaults to False
    :type noise_balancing: bool, optional
    :param apply_agc:
        whether agc was applied to tribe, will be part of filename,
        defaults to False
    :type apply_agc: bool, optional
    :param write_individual_templates:
        whether to write out each template in its own file, defaults to False
    :type write_individual_templates: bool, optional
    :param check_len_strict:
        whether to strictly check the length of the shortened templates to all
        be the same number of samples, defaults to True
    :type check_len_strict: bool, optional
    :raises AssertionError: occurrs when traces do not have the same length
    :return: shortened tribe
    :rtype:  class:`eqcorrscan.core.match_filter.Tribe`
    """
    if len(tribe) == 0:
        return tribe
    if tribe_len_pct is not None:
        new_templ_len = (tribe[0].st[0].stats.endtime -
                         tribe[0].st[0].stats.starttime) * tribe_len_pct
    else:
        new_templ_len = max_tribe_len

    if inplace:  # Overwrite instead of copy
        short_tribe = tribe
    else:  # Make a copy
        short_tribe = _quick_tribe_copy(tribe)
    for templ in short_tribe:
        templ.trace_offset = trace_offset
        for tr in templ.st:
            if new_templ_len is not None:
                tr.trim(
                    starttime=tr.stats.starttime + trace_offset,
                    endtime=tr.stats.starttime + new_templ_len + trace_offset)

                # Try this to make sure catalog_to_dd.write_correlations works
                # exactly the same with presliced traces: -TODO: BUT IT DOESNT.
                # There is one sample difference in dt-values with presliced streams.

                # If there is one sample too many after this remove the first one
                # by convention
                n_samples_intended = new_templ_len * tr.stats.sampling_rate
                if len(tr.data) == n_samples_intended + 1:
                    tr.data = tr.data[1:len(tr.data)]
                # if tr.stats.endtime - tr.stats.starttime != extract_len:
                if tr.stats.npts < n_samples_intended:
                    Logger.warning(
                        "Insufficient data ({rlen} s) for {tr_id}, discarding. "
                        "Check that your traces are at least of length {length} s,"
                        #" with a pre_pick time of at least {prepick} s!".format(
                        "".format(
                            rlen=tr.stats.endtime - tr.stats.starttime,
                            tr_id=tr.id, length=new_templ_len))
                    continue

            if equalize_scaling:
                # tr.data = tr.data / np.nanmax(np.abs(tr.data))
                tr.data = tr.data / np.sqrt(np.nanmean(tr.data ** 2))
            # DETREND and TAPER!
            tr.taper(max_percentage=max_taper_percentage, type='cosine').detrend()
            # Cast to float32 to save memory
            tr.data = tr.data.astype(np.float32)
        if len(templ.st) >= min_n_traces:
            templ_name = templ.name
            # orig = templ.event.preferred_origin() or templ.event.origins[0]
            # templ_name = str(orig.time)[0:22] + '_' + 'templ'
            # templ_name = templ_name.lower().replace('-', '_')\
            #     .replace(':', '_').replace('.', '_').replace('/', '')
            # make a nice plot
            if make_pretty_plot:
                image_name = os.path.join('TemplatePlots',
                                          prefix + '_' + templ_name)
                pretty_template_plot(
                    templ.st, background=False, event=templ.event,
                    sort_by='distance', show=False, return_figure=False,
                    size=(25, 50), save=True, savefile=image_name)
            Logger.debug("Made shortened template %s", templ_name)
            Logger.debug(templ)
    # Check that all traces are same length
    if check_len_strict:
        stempl_lengths = list(set(
            [tr.stats.npts for templ in short_tribe for tr in templ.st]))
        # assert len(stempl_lengths) == 1, "Template traces differ in length"
        if not len(stempl_lengths) == 1:
            raise AssertionError("short tribe has traces of unequal length " +
                                 str(stempl_lengths))
    # Set up the file labeling depending on processing parameters
    label = ''
    if noise_balancing:
        label = label + 'balNoise_'
    if apply_agc:
        label = label + 'agc_'
    if write_out:
        short_tribe.write(
            'TemplateObjects/' + prefix + 'Templates_min'
            + str(min_n_traces) + 'tr_' + label + str(len(short_tribe)))
        # max_events_per_file=10)
    if write_individual_templates:
        for templ in short_tribe:
            templ.write('Templates/' + prefix + templ.name + '.mseed',
                        format="MSEED")
    return short_tribe


def check_template_event_errors_ok(
        origin, max_horizontal_error_km=None, max_depth_error_km=None,
        max_time_error_s=None, file='', min_latitude=None, max_latitude=None,
        min_longitude=None, max_longitude=None, **kwargs):
    """
    Function to check origin errors gracefully for the different ways erros
    can be recorded.

    :param origin: origin for which to check location errors
    :type origin: class:`obspy.core.event.origin`
    :param max_horizontal_error_km:
        maximum horizontal error in km to pass check, defaults to None
    :type max_horizontal_error_km: float, optional
    :param max_depth_error_km:
        maximum depth error in km to pass check, defaults to None
    :type max_depth_error_km: float, optional
    :param max_time_error_s:
        maximum time error in km to pass check, defaults to None
    :type max_time_error_s: float, optional
    :param file: event-file
        for which error was checked (only needed for logging), defaults to ''
    :type file: str, optional
    :param min_latitude: minimum latitude to pass check, defaults to None
    :type min_latitude: float, optional
    :param max_latitude: maximum latitude to pass check, defaults to None
    :type max_latitude: float, optional
    :param min_longitude: minimum longitude to pass check, defaults to None
    :type min_longitude: float, optional
    :param max_longitude: maximum longitude to pass check, defaults to None
    :type max_longitude: float, optional
    :return: Whether origin passed all checks or not.
    :rtype: bool
    """
    # Do not use event as template if any errors are above threshold
    if not origin:
        Logger.info(
            'Rejected template, event has no origin, cannot check errors.')
        return True
    # Check location
    if min_latitude and origin.latitude:
        if origin.latitude < min_latitude:
            return False
    if max_latitude and origin.latitude:
        if origin.latitude > max_latitude:
            return False
    if min_longitude and origin.longitude:
        if origin.longitude < min_longitude:
            return False
    if max_longitude and origin.longitude:
        if origin.longitude > max_longitude:
            return False

    # # Check horizontal error
    if max_horizontal_error_km:
        max_hor_error = list()
        if origin.origin_uncertainty:
            max_hor_error.append(
                origin.origin_uncertainty.max_horizontal_uncertainty / 1000)
        else:
            if (origin.longitude_errors
                    and origin.longitude_errors.uncertainty):
                max_hor_error.append(degrees2kilometers(
                    origin.longitude_errors.uncertainty))
            if (origin.latitude_errors
                    and origin.latitude_errors.uncertainty):
                max_hor_error.append(degrees2kilometers(
                    origin.latitude_errors.uncertainty))
        if max_hor_error:
            max_hor_error = max(max_hor_error)
            if (max_hor_error and max_hor_error > max_horizontal_error_km):
                Logger.info(
                    'Rejected template: event %s (file %s): horizontal error '
                    'too large (%s).', str(origin.time)[0:19], file,
                    str(max_hor_error))
                return False
    # Check depth error
    if max_depth_error_km:
        if (origin.depth_errors and origin.depth_errors.uncertainty):
            max_depth_error = origin.depth_errors.uncertainty / 1000
            if max_depth_error > max_depth_error_km:
                Logger.info(
                    'Rejected template: event %s (file %s): depth error too '
                    'large (%s).', str(origin.time)[0:19], file,
                    str(max_depth_error))
                return False
    # Check time error
    if max_time_error_s:
        if (origin.time_errors and origin.time_errors.uncertainty):
            max_time_error = origin.time_errors.uncertainty
            if max_time_error > max_time_error_s:
                Logger.info(
                    'Rejected template: event %s (file %s): time error too '
                    'large (%s).', str(origin.time)[0:19], file,
                    str(max_time_error))
                return False
    return True


def _create_template_objects(
        events_files=[], selected_stations=[], template_length=60,
        lowcut=2.5, highcut=9.9, min_snr=5.0, prepick=0.5, samp_rate=20,
        seisan_wav_path=None, inv=Inventory(), clients=[],
        remove_response=False, output='VEL', noise_balancing=False,
        balance_power_coefficient=2, ground_motion_input=[],
        apply_agc=False, agc_window_sec=5,
        min_n_traces=8, min_n_station_sites=4,
        write_individual_templates=False, templ_path='Templates/',
        make_pretty_plot=False, prefix='',
        check_template_strict=True, allow_channel_duplication=True,
        normalize_NSLC=True, add_array_picks=False, stations_df=pd.DataFrame(),
        add_large_aperture_array_picks=False, suppress_arraywide_steps=True,
        ispaq=None, sta_translation_file="station_code_translation.txt",
        std_network_code='NS', std_location_code='00', std_channel_prefix='BH',
        vertical_chans=['Z', 'H'],
        horizontal_chans=['E', 'N', '1', '2', 'X', 'Y'],
        stations_with_verticals_for_s=[],
        bayesloc_event_solutions=None, erase_mags=False,
        wavetool_path='/home/felix/Software/SEISANrick/PRO/linux64/wavetool',
        parallel=False, cores=1, thread_parallel=False, n_threads=1,
        unused_kwargs=True, *args, **kwargs):
    """
    Internal function to create template objects from event files or event
    objects, to be run in parallel.

    :param events_files: List of event files or event objects.
    :type events_files: list
    :param selected_stations: List of selected stations, defaults to None
    :type selected_stations: list, optional
    :param template_length: Length of template in seconds, defaults to 60
    :type template_length: float, optional
    :param lowcut: Lowcut frequency for bandpass filter, defaults to 2.5
    :type lowcut: float, optional
    :param highcut: Highcut frequency for bandpass filter, defaults to 9.9
    :type highcut: float, optional
    :param min_snr: Minimum signal-to-noise ratio for template creation,
    :type min_snr: float, optional
    :param prepick:
        Time before the pick in seconds to start the template, defaults to 0.5
    :type prepick: float, optional
    :param samp_rate: Sampling rate of the template, defaults to 20 Hz
    :type samp_rate: float, optional
    :param seisan_wav_path:
        Path to single-event waveform files linked from Seisan S-file,
        defaults to None (then data should be read from a client).
    :type seisan_wav_path: str, optional
    :param inv: Inventory object, defaults to None
    :type inv: :class:`obspy.core.inventory.inventory.Inventory`, optional
    :param clients: List of obspy FDSN client objects, defaults to None
    :type clients:
        list of :class:`obspy.clients.fdsn.client.Client`, or other clients
        with the same API.
    :param remove_response: Remove instrument response, defaults to False
    :type remove_response: bool, optional
    :param output: Output type of the template, defaults to 'DISP'
    :type output: str, optional
    :param noise_balancing: Whether to balance the noise, defaults to False
    :type noise_balancing: bool, optional
    :param balance_power_coefficient:
        Power coefficient for noise balancing, defaults to 2 (see
        :func:`robustraqn.utils.spectral_tools.balance_noise`)
    :type balance_power_coefficient: float, optional
    :param ground_motion_input:
        List of ground motion input types, can be one of 'DISP', 'VEL', 'ACC',
        defaults to []
    :type ground_motion_input: list, optional
    :param apply_agc:
        Whether to apply automatic gain control, defaults to False
    :type apply_agc: bool, optional
    :param agc_window_sec: AGC window length in seconds, defaults to 5
    :type agc_window_sec: float, optional
    :param min_n_traces:
        Minimum number of traces to define a set of waveforms as template.
    :type min_n_traces: int, optional
    :param min_n_station_sites: Minimum number of stations to define a template
    :type min_n_station_sites: int, optional
    :param write_individual_templates:
        Write out individual template files, defaults to False.
    :type write_individual_templates: bool, optional
    :param templ_path: Path to write individual template files to, defaults to
    :type templ_path: str, optional
    :param make_pretty_plot:
        Make a pretty plot of the template, defaults to False
    :type make_pretty_plot: bool, optional
    :param prefix: Prefix for the template name, defaults to ''
    :type prefix: str, optional
    :param check_template_strict:
        Check template strictly for NaNs, zeroes, and same trace length,
        defaults to True
    :type check_template_strict: bool, optional
    :param allow_channel_duplication:
        Allow channel duplication, defaults to True
    :type allow_channel_duplication: bool, optional
    :param normalize_NSLC: Normalize NSLC codes, defaults to True
    :type normalize_NSLC: bool, optional
    :param add_array_picks:
        Add picks for neighboring stations on a seismic array, defaults to
        False.
    :type add_array_picks: bool, optional
    :param stations_df:
        Stations dataframe containing station information like coordinates
    :type stations_df: :class:`pandas.DataFrame`, optional
    :param add_large_aperture_array_picks:
        Add picks for large aperture arrays (these can also be parts of a local
        seismic network if defined as such).
    :type add_large_aperture_array_picks: bool, optional
    :param large_aperture_array_df: Large aperture array dataframe
    :type large_aperture_array_df: :class:`pandas.DataFrame`, optional
    :type kwargs: dict, optional
    :param kwargs:
        Additional keyword arguments to be passed to
        eqcorrscan.utils.preprocessing.shortproc.

    :rtype: tuple
    :return:
        Tuple of template names and picks that were rejected due to data
        quality metrics indicating poor data quality.
    """
    if isinstance(events_files[0], str):
        input_type = 'sfiles'
        events_files.sort(key=lambda x: x[-6:])
    elif isinstance(events_files[0], Event):
        input_type = 'events'
        events_files = Catalog(sorted(events_files, key=lambda x: (
            x.preferred_origin() or x.origins[0]).time))
    else:
        NotImplementedError(
            'event_files has to be a filename-string or an obspy event')

    Logger.info('Start work on %s sfiles / events to create templates...',
                len(events_files))
    tribe = Tribe()
    template_names = []
    catalog_for_templates = Catalog()
    catalog = Catalog()
    bulk_rejected = []

    day_st = Stream()
    if clients and len(events_files) > 1:
        Logger.info('Requesting waveform data for one day of candidate '
                    'template events')
        if input_type == 'sfiles':
            # The earliest event time of the day
            day_starttime = UTCDateTime(
                events_files[0][-6:] + os.path.split(events_files[0])[-1][0:2])
            # The latest event time of the day
            checktime = UTCDateTime(events_files[-1][-6:] +
                                    os.path.split(events_files[-1])[-1][0:2])
        else:
            orig_t = (events_files[0].preferred_origin() or
                      events_files[0].origins[0]).time
            day_starttime = UTCDateTime(orig_t.year, orig_t.month, orig_t.day)
            # orig_t = events_files[-1].preferred_origin().time
            orig_t = (events_files[-1].preferred_origin() or
                      events_files[-1].origins[0]).time
            checktime = UTCDateTime(orig_t.year, orig_t.month, orig_t.day)
        # Request the whole day plus/minus a bit more
        starttime = day_starttime - 30 * 60
        endtime = starttime + 24.5 * 60 * 60
        # Check against time of the last sfile / event in batch - it should be
        # fully covered in the 25-hour stream
        if (checktime - day_starttime) < 60 * 60 * 24.5:
            if ispaq is not None:
                bulk_request, bulk_rejected, day_stats = (
                    create_bulk_request(
                        inv.select(starttime=starttime, endtime=endtime),
                        starttime=starttime, endtime=endtime, stats=ispaq,
                        stations=selected_stations,
                        minimum_sample_rate=samp_rate, **kwargs))
            else:
                bulk_request = [("??", s, "*", "?H?", starttime, endtime)
                                for s in selected_stations]
            for client in clients:
                Logger.info('Requesting waveforms from client %s', client)
                outtic = default_timer()
                client = get_parallel_waveform_client(client)
                add_st = client.get_waveforms_bulk(
                    bulk_request, parallel=parallel, cores=cores)
                outtoc = default_timer()
                Logger.info(
                    'Received %s traces from client for whole day, which took:'
                    ' {0:.4f}s'.format(outtoc - outtic), str(len(add_st)))
                day_st += add_st
            clients = []  # Set empty so not to request archive data again
            if len(day_st) == 0:
                Logger.warning('Did not find any waveforms for date %s.',
                               str(day_starttime))

    wavnames = []
    # Loop over all S-files that each contain one event
    for j, event_file in enumerate(events_files):
        if input_type == 'sfiles':
            Logger.info('Working on S-file: ' + event_file)
            select, wavname = read_nordic(
                event_file, return_wavnames=True, unused_kwargs=unused_kwargs,
                **kwargs)
            if bayesloc_event_solutions:
                Logger.info('Updating catalog from bayesloc solutions')
                select = update_cat_from_bayesloc(
                    select, bayesloc_event_solutions, **kwargs)
            # add wavefile-name to output
            wavnames.append(wavname[0])
            event = select[0]
            event_str = event_file
            sfile = event_file
            # Save original filename in event.extra.sfile
            if not hasattr(event, 'extra'):
                event['extra'] = AttribDict()
            event.extra.update({'sfile': {
                'value': os.path.basename(event_file), 'namespace':
                    'https://seis.geus.net/software/seisan/node239.html'}})
        else:
            event_str = event_file.short_str()
            Logger.info('Working on event: ' + event_str)
            sfile = ''
            event = event_file
        if erase_mags:
            event.magnitudes = []
        relevant_stations = get_all_relevant_stations(
            selected_stations, sta_translation_file=sta_translation_file)
        # Test if there are any comments on the event that tell me to skip it:
        for comment in event.comments:
            if "blacklisted template" in comment.text:
                continue
        # TODO: maybe I should select the "best" origin somewhere (e.g.,
        # smallest errors, largest number of stations etc) - Bayesloc origin is
        # added in position 0.
        origin = event.preferred_origin() or event.origins[0]

        if not check_template_event_errors_ok(
                origin, file=event_str, **kwargs):
            continue

        # TODO: if there are a lot of different picks at one station, e.g.:
        #       P, Pn, Pb, Pg, S, Sn, Sb, Sg; then throw out some so that there
        #       is less trace overlap and trace duplication (affects memory
        #       usage!)

        # Load picks and normalize
        tmp_catalog = filter_picks(
            Catalog([event]), stations=relevant_stations)
        if not tmp_catalog:
            Logger.info('Rejected template: no event for %s after filtering',
                        event_str)
            continue
        event = tmp_catalog[0]
        if not event.picks:
            Logger.info('Rejected template: event %s has no picks after '
                        'filtering', event.short_str())
            continue
        catalog += event

        # Add picks at array stations if requested
        if add_array_picks:
            # need to fix phase hints once alreay here
            event = fix_phasehint_capitalization(event)
            array_picks_dict = extract_array_picks(event=event)
            event = add_array_station_picks(
                event=event, array_picks_dict=array_picks_dict,
                stations_df=stations_df, **kwargs)
            if add_large_aperture_array_picks:
                array_picks_dict = extract_array_picks(
                    event=event,
                    seisarray_prefixes=LARGE_APERTURE_SEISARRAY_PREFIXES)
                Logger.info('Adding array picks for large aperture arrays')
                event = add_array_station_picks(
                    event=event, array_picks_dict=array_picks_dict,
                    stations_df=stations_df,
                    seisarray_prefixes=LARGE_APERTURE_SEISARRAY_PREFIXES,
                    **kwargs)

        #######################################################################
        # Load and quality-control stream and picks for event
        wavef = load_event_stream(
            event, sfile, seisan_wav_path, relevant_stations, clients=clients,
            st=day_st.copy(), min_samp_rate=samp_rate, pre_event_time=prepick,
            template_length=template_length, bulk_rejected=bulk_rejected,
            wavetool_path=wavetool_path)
        if wavef is None or len(wavef) == 0:
            Logger.info('Rejected template: event %s for sfile %s has no '
                        'waveforms available', event.short_str(), sfile)
            continue

        # Check for array-wide steps in the data
        if suppress_arraywide_steps:
            wavef = mask_array_trace_offsets(
                wavef, split_taper_stream=False, **kwargs)
        wavef = wavef.mask_consecutive_zeros(min_run_length=None)
        # Taper all the segments
        wavef = taper_trace_segments(wavef)

        # A useful pre-filter makes waterlevel less critical, but also avoids
        # other issues like overamplication of high-frequency noise on the
        # Greenland stations with steep FIR filter response.
        nyquist_f = samp_rate / 2
        if 'pre_filt' in kwargs.keys():
            pre_filt = kwargs['pre_filt']
            kwargs.pop('pre_filt')
        else:
            pre_filt = [0.1, 0.2, 0.9 * nyquist_f, 0.95 * nyquist_f]
        if remove_response:
            wavef = wavef.try_remove_responses(
                inv, output=output, taper_fraction=0.15,
                pre_filt=pre_filt, parallel=parallel, cores=cores,
                thread_parallel=thread_parallel, n_threads=n_threads, **kwargs)
            if origin.latitude is None or origin.longitude is None:
                Logger.warning('Could not compute distances for event %s.',
                               event.short_str())
            else:
                for tr in wavef:
                    try:
                        tr.stats.distance = gps2dist_azimuth(
                            origin.latitude, origin.longitude,
                            tr.stats.coordinates.latitude,
                            tr.stats.coordinates.longitude)[0]
                    except Exception as e:
                        Logger.warning(
                            'Could not compute distance for event %s -'
                            ' trace %s', event.short_str(), tr.id)
        wavef = wavef.detrend(type='simple')

        # TODO: Set tr.stats.extra.day_noise_level based on ispaq metrics
        for tr in wavef:
            if not hasattr(tr.stats, 'extra'):
                tr.stats.extra = AttribDict()
            tr.stats.extra.update({'day_noise_level': 1})

        # standardize all codes for network, station, location, channel
        if normalize_NSLC:
            wavef, trace_id_change_dict = wavef.normalize_nslc_codes(
                inv, sta_translation_file=sta_translation_file,
                std_network_code=std_network_code,
                std_location_code=std_location_code,
                std_channel_prefix=std_channel_prefix)

        # Do noise-balancing by the station's PSDPDF average
        if noise_balancing:
            # if not hasattr(wavef, "balance_noise"):
            #     bound_method = st_balance_noise.__get__(wavef)
            #     wavef.balance_noise = bound_method
            wavef = wavef.filter(
                'highpass', freq=0.1, zerophase=True).detrend()
            wavef = st_balance_noise(
                wavef, inv,
                balance_power_coefficient=balance_power_coefficient,
                ground_motion_input=ground_motion_input,
                sta_translation_file=sta_translation_file)
            for tr in wavef:
                try:
                    tr = tr.taper(
                        max_percentage=0.10, type='hann', max_length=15,
                        side='both').detrend('linear')
                except ValueError as e:
                    Logger.error('Could not detrend trace %s:', tr)
                    Logger.error(e)
                    continue

        # TODO: copy picks for horizontal channels to the other horizontal
        #       channel so that picks can once again be lag-calc-picked on both
        #       horizontals.
        event = prepare_picks(
            event=event, stream=wavef, normalize_NSLC=normalize_NSLC, inv=inv,
            sta_translation_file=sta_translation_file,
            vertical_chans=vertical_chans, horizontal_chans=horizontal_chans,
            stations_with_verticals_for_s=stations_with_verticals_for_s,
            **kwargs)
        # Extra checks for sampling rate and length of trace - if a trace is
        # very short, resample will fail.
        st = Stream()
        for tr in wavef.copy():
            factor = tr.stats.sampling_rate / float(samp_rate)
            if tr.stats.sampling_rate < 0.99 * samp_rate:
                Logger.info(
                    'Removed trace %s because its sample rate (%s) is too low',
                    tr.stats.sampling_rate)
            elif (tr.stats.npts / factor) < samp_rate:
                Logger.info('Removed trace %s because it has only %s samples.',
                            tr, str(tr.stats.npts))
            else:
                st += tr

        # Extract relevant kwargs for pre_processing.shortproc
        extra_kwargs = dict()
        for key, value in kwargs.items():
            if key in ['starttime', 'endtime', 'seisan_chan_names' 'fill_gaps',
                       'ignore_length', 'ignore_bad_data', 'fft_threads']:
                extra_kwargs.update({key: value})
        wavef = pre_processing.shortproc(
            st=st, lowcut=lowcut, highcut=highcut, filt_order=4,
            samp_rate=samp_rate, parallel=False, num_cores=1,
            fft_threads=n_threads, **extra_kwargs)
        # data_envelope = obspy.signal.filter.envelope(st_filt[0].data)

        # Make the templates from picks and waveforms
        catalog_for_templates += event
        # TODO : this is where a lot of calculated picks are thrown out
        template_st = template_gen._template_gen(
            picks=event.picks, st=wavef, length=template_length, swin='all',
            prepick=prepick, all_vert=True, all_horiz=True,
            delayed=True, min_snr=min_snr, horizontal_chans=horizontal_chans,
            vertical_chans=vertical_chans)  # , **kwargs)
        # Cast to float32 to save memory (Obspy likes to make it float64 but
        # that is not necessary for our purposes).
        for tr in template_st:
            tr.data = tr.data.astype(np.float32)
        # quality-control template
        if len(template_st) == 0:
            Logger.info('Rejected template: event %s (sfile %s): no traces '
                        'with matching picks that fulfill quality criteria.',
                        event.short_str(), sfile)
            continue
        # Apply AGC if required
        if apply_agc:
            Logger.info('Applying AGC to template stream')
            wavef = wavef.agc(agc_window_sec, **kwargs)
        if check_template_strict:
            template_st = check_template(
                template_st, template_length, remove_nan_strict=True,
                allow_channel_duplication=allow_channel_duplication,
                max_perc_zeros=5)

        # Reduce weights for the individual stations in an array:
        # Set station_weight_factor according to number of array stations
        if add_array_picks:
            unique_stations = list(dict.fromkeys(
                [tr.stats.station for tr in template_st]))
            station_sites = get_station_sites(unique_stations)
            # Get a list of the sites for each array
            n_station_sites_list = [station_sites.count(site)
                                    for site in station_sites]
            for tr in template_st:
                tr.stats.extra.station_weight_factor = 1
                for u_sta, n_station_sites in zip(unique_stations,
                                                  n_station_sites_list):
                    if tr.stats.station == u_sta:
                        # Try a reduction factor of 1 / sqrt(n_sites)
                        sfact = 1 / np.sqrt(n_station_sites)
                        tr.stats.extra.station_weight_factor = sfact
                        if not hasattr(tr.stats, 'extra'):
                            tr.stats.extra = AttribDict()
                        if hasattr(tr.stats.extra, 'weight'):
                            tr.stats.extra.update(
                                {'weight': tr.stats.extra.weight * sfact})

        # templ_name = str(event.origins[0].time) + '_' + 'templ'
        orig = event.preferred_origin() or event.origins[0]
        templ_name = ''
        # Make sure that template doesn't get a name that exists already
        # TODO: this only takes care of duplicate names if events/sfiles are in
        #       the same batch within parallel-loop (e.g., ok for day batches).
        nt = 0
        while templ_name in template_names or templ_name == '':
            if templ_name in template_names and templ_name != '':
                Logger.warning(
                    'Template name %s already taken, trying with a different '
                    'name.', templ_name)
            templ_time = orig.time + nt
            templ_name = str(templ_time)[0:22] + '_' + 'templ'
            templ_name = (templ_name.lower().replace('-', '_').replace(
                ':', '_').replace('.', '_').replace('/', ''))
            nt += 1
        template_names.append(templ_name)
        t = Template(name=templ_name, event=event, st=template_st,
                     lowcut=lowcut, highcut=highcut, samp_rate=samp_rate,
                     filt_order=4, process_length=86400.0, prepick=prepick)
        # Check that minimum number of station sites is fulfilled, considering
        # that one array counts only as one station site:
        unique_stations = list(set([
            p.waveform_id.station_code for p in t.event.picks]))
        n_station_sites = len(list(set(get_station_sites(unique_stations))))

        if (len(t.st) >= min_n_traces and
                n_station_sites >= min_n_station_sites):
            if write_individual_templates:
                templ_filename = os.path.join(templ_path,
                                              templ_name + '.mseed')
                t.write(templ_filename, format="MSEED")
            tribe += t
            # make a nice plot
            sfile_path, sfile_name = os.path.split(sfile)
            if make_pretty_plot:
                image_name = os.path.join('TemplatePlots',
                                          prefix + '_' + templ_name)
                pretty_template_plot(
                    template_st, background=wavef, event=event,
                    sort_by='distance', show=False, return_figure=False,
                    size=(12, 16), save=True, savefile=image_name)
            Logger.info("Made template %s for sfile %s", templ_name, sfile)
            Logger.info(t)
        else:
            if n_station_sites < min_n_station_sites:
                Logger.info("Rejected template %s (sfile %s): too few unique "
                            " station sites %s < %s", templ_name, sfile,
                            str(n_station_sites), str(min_n_station_sites))
            else:
                Logger.info("Rejected template %s (sfile %s): too few traces: "
                            "%s < %s", templ_name, sfile, str(len(t.st)),
                            str(min_n_traces))

    # clusters = tribe.cluster(method='space_cluster', d_thresh=1.0, show=True)
    template_list = []
    for j, templ in enumerate(tribe):
        template_list.append((templ.st, j))

    return (tribe, wavnames)

    # TODO: include clustering of templates here to reduce size of tribe?
    # clusters = cluster(template_list=template_list, show=True,
    #       corr_thresh=0.3, allow_shift=True, shift_len=2, save_corrmat=False,
    #       cores=16)
    # groups = cluster(template_list=template_list, show=False,
    #                  corr_thresh=0.3, cores=8)
    # group_streams = [st_tuple[0] for st_tuple in groups[0]]
    # stack = linstack(streams=group_streams)
    # PWSstack = PWS_stack(streams=group_streams)


def reset_preferred_magnitude(tribe, mag_preference_priority=[('ML', 'BER')]):
    """
    Function that resets the preferred_magnitude to link to a user-chosen
    magnitude type and agency.

    :type tribe: eqcorrscan.
    :param tribe: tribe for which preferred magnitudes should be fixed
    :type mag_preference_priority: list of tuples
    :param mag_preference_priority:
        magnitude type and agency id e.g., [('ML', 'BER')]. Any one value can
        also be None.
    """
    for templ in tribe:
        for mag_pref in reversed(mag_preference_priority):
            for mag in templ.event.magnitudes:
                if ((mag_pref[0] == mag.magnitude_type or None) and
                        (mag_pref[1] == mag.creation_info.agency_id or None)):
                    templ.event.preferred_magnitude_id = mag.resource_id
    return tribe


def _make_date_list(catalog, unique=True, sorted=True):
    """Make list of the unique days / dates for events in a catalog

    :param catalog: _description_
    :type catalog: type:`obspy.core.event.Catalog`
    :param unique:
        whether to provide a unique list of dayes, defaults to True
    :type unique: bool, optional
    :param sorted: whether to sort dates, defaults to True
    :type sorted: bool, optional
    :return: list of dates for which catalog contains events
    :rtype: list of type:`obspy.core.UTCDateTime`
    """
    date_list = []
    for event in catalog:
        orig = event.preferred_origin() or event.origins[0]
        date_list.append(str(UTCDateTime(
            orig.time.year, orig.time.month, orig.time.day))[0:10])
    if unique:
        date_list = list(set(date_list))
    if sorted:
        date_list = sorted(date_list)
    return date_list


def create_template_objects(
        sfiles=[], catalog=None, selected_stations=[],
        event_stations_filter=[], catalog_df=None, template_length=60,
        lowcut=2.5, highcut=9.9, min_snr=5.0, prepick=0.5, samp_rate=20,
        seisan_wav_path=None, sfiles_include_path=None,
        clients=[], inv=Inventory(),
        remove_response=False, output='VEL', noise_balancing=False,
        balance_power_coefficient=2, ground_motion_input=[],
        apply_agc=False, agc_window_sec=5,
        min_n_traces=8, write_out=False, write_individual_templates=False,
        out_folder='TemplateObjects', check_template_strict=True,
        templ_path='Templates', prefix='', make_pretty_plot=False,
        allow_channel_duplication=True, normalize_NSLC=True, ispaq=None,
        add_array_picks=False, add_large_aperture_array_picks=False,
        sta_translation_file="station_code_translation.txt",
        std_network_code='NS', std_location_code='00', std_channel_prefix='BH',
        vertical_chans=['Z', 'H'],
        wavetool_path='/home/felix/Software/SEISANrick/PRO/linux64/wavetool',
        horizontal_chans=['E', 'N', '1', '2', 'X', 'Y'],
        stations_with_verticals_for_s=[],
        erase_mags=False,
        parallel=False, cores=1, thread_parallel=False, n_threads=1,
        max_events_per_file=200, task_id=None,
        *args, **kwargs):
    """Wrapper for create-template-function

    :type sfiles: list of str
    :param sfiles: list of sfiles to use for template creation
    :type catalog: obspy.core.event.Catalog
    :param catalog:
        catalog of events to use for template creation, required if sfiles are
        not provided
    :type selected_stations: list of str
    :param selected_stations: list of stations to use for template creation
    :type event_stations_filter: list of str
    :param event_stations_filter: TBD TODO
    :type catalog_df: pandas.DataFrame
    :param catalog_df:
        Dataframe of catalog with events to use for template creation. If this
        dataframe is supplied, then starting up the parallel workers can be
        accellerated considerably.
    :type template_length: float
    :param template_length: length of templates in seconds
    :type lowcut: float
    :param lowcut: lowcut for bandpass filter
    :type highcut: float
    :param highcut: highcut for bandpass filter
    :type min_snr: float
    :param min_snr: minimum signal-to-noise ratio for template creation
    :type prepick: float
    :param prepick: time before pick to start template
    :type samp_rate: float
    :param samp_rate:
        target sampling rate for templates (lower sampling rate traces will be
        excluded)
    :type seisan_wav_path: str
    :param seisan_wav_path: path to event-based wav files (e.g., in seisan REA)
    :type clients: list of obspy.fdsn.client.Client
    :param clients: list of clients for data retrieval
    :type inv: obspy.core.inventory.inventory.Inventory
    :param inv: inventory to use for response removal
    :type remove_response: bool
    :param remove_response: whether to remove responses from traces
    :type output: str
    :param output: output units for response-corrected traces (VEL, ACC, DISP)
    :type noise_balancing: bool
    :param noise_balancing:
        whether to balance noise levels in templates according to the station
        noise level.
    :type balance_power_coefficient: float
    :param balance_power_coefficient:
        to-the-power coefficient for noise balancing, defaults to 2
    :type ground_motion_input: list of str
    :param ground_motion_input:
        list of ground motion input types (VEL, ACC, DISP), only needed in case
        the trace's stats do not contain the information
    :type apply_agc: bool
    :param apply_agc: whether to apply automatic gain control to traces
    :type agc_window_sec: float
    :param agc_window_sec: length of AGC window in seconds
    :type min_n_traces: int
    :param min_n_traces:
        minimum number of traces required for defining a valid template
    :type write_out: bool
    :param write_out: whether to write templates to disk
    :type write_individual_templates: bool
    :param write_individual_templates:
    :type out_folder: str
    :param out_folder: folder to write templates to
    :type check_template_strict: bool
    :param check_template_strict:
    :type templ_path: str
    :param templ_path: path to write templates to
    :type prefix: str
    :param prefix: prefix for template files
    :type make_pretty_plot: bool
    :param make_pretty_plot: whether to make a pretty plot of the template
    :type allow_channel_duplication: bool
    :param allow_channel_duplication:
        whether to allow channel duplication in templates (e.g., 2 traces for
        Z channel at same station, one for Pn and one for Pg phase)
    :type normalize_NSLC: bool
    :param normalize_NSLC:
        whether to normalize NSLC codes to standard codes (this is useful
        when network, station, or channel names have changed through station
        life cycle)
    :type ispaq: pd.DataFrame
    :param ispaq: dataframe containing ispaq data quality metrics
    :type add_array_picks: bool
    :param add_array_picks:
        whether to try to add picks and waveforms for each station within a
        seismic array to a template
    :type add_large_aperture_array_picks: bool
    :param add_large_aperture_array_picks:
        whether to handle parts of the network as large aperture arrays,
        e.g., a combination of subarrays (e.g., NOA array), or any geometry
        of closely spaced stations
    :type sta_translation_file: str
    :param sta_translation_file:
        path to station translation file that lists equivalent station codes
    :type std_network_code: str
    :param std_network_code: standard network code
    :type std_location_code: str
    :param std_location_code: standard location code
    :type std_channel_prefix: str
    :param std_channel_prefix: standard channel prefix
    :type vertical_chans: list of str
    :param vertical_chans: list of vertical channel names
    :type wavetool_path: str
    :param wavetool_path: path to wavetool binary
    :type horizontal_chans: list of str
    :param horizontal_chans: list of horizontal channel names
    :type erase_mags: bool
    :param erase_mags:
        whether to erase magnitudes from events prior to template creation
    :type parallel: bool
    :param parallel: whether to parallelize template creation
    :type cores: int
    :param cores: number of cores to use for parallelization
    :type max_events_per_file: int
    :param max_events_per_file:
        maximum number of events per tribe file, can be set to read tribe in
        parallel later

    :rtype: tuple of `EQcorrscan.core.match_filter.tribe.Tribe` and list of str
    :return:
        Tribe of templates and list of waveform paths read in for each template
    """
    # Get only relevant inventory information to make Pool-startup quicker
    new_inv = Inventory()
    if inv is not None:
        for sta in selected_stations:
            new_inv += inv.select(station=sta)
    stations_df = get_updated_stations_df(inv)
    not_cat_or_sfiles_msg = (
        'Provide either sfiles with file paths to events, or provide catalog '
        ' with events.')

    # Split the task into batches to save time on reading from archive just
    # once for each day.
    if parallel and ((sfiles and len(sfiles) > 1) or
                     (catalog and len(catalog) > 1)):
        if cores is None:
            cores = min(len(sfiles), cpu_count())
        # Run in batches to save time on reading from archive only once per day
        day_stats_list = []
        unique_date_list = []
        event_file_batches = []
        if sfiles:
            Logger.info('Preparing file batches from provided filenames')
            if len(sfiles) > cores and clients:
                sfiles_df = pd.DataFrame(sfiles, columns=['sfiles'])
                # Create day-column with efficient pandas functinos
                sfiles_df['day'] = (
                    sfiles_df.sfiles.str[-6:-2] + '-' +
                    sfiles_df.sfiles.str[-2:] + '-' +
                    pd.DataFrame(
                        sfiles_df['sfiles'].apply(os.path.split).tolist(),
                        index=sfiles_df.index).iloc[:, -1].str[0:2])
                unique_date_list = sorted(list(set(sfiles_df['day'])))
                sfile_groups = sfiles_df.groupby('day')
                event_file_batches = [
                    list(sfile_groups.get_group(unique_date_utc)['sfiles'])
                    for unique_date_utc in unique_date_list]
            else:
                event_file_batches = [[sfile] for sfile in sfiles]
                unique_date_list = [str(UTCDateTime(
                    sfile[-6:] + os.path.split(sfile)[-1][0:2]))[0:10]
                    for sfile in sfiles]
        elif catalog:
            Logger.info('Preparing event batches from provided catalog')
            if len(catalog) > cores and clients:
                unique_date_list = _make_date_list(catalog, unique=True,
                                                   sorted=True)
                # TODO: speed up event batch creation
                cat_df = events_to_df(catalog)
                cat_df['events'] = catalog.events
                cat_df['day'] = cat_df.time.astype(str).str[0:10]
                event_groups = cat_df.groupby('day')
                event_file_batches = [
                    event_groups.get_group(unique_date_utc).events
                    for unique_date_utc in unique_date_list]
            else:
                event_file_batches = [[event] for event in catalog]
                unique_date_list = _make_date_list(catalog, unique=False,
                                                   sorted=False)
        else:
            raise NotImplementedError(not_cat_or_sfiles_msg)

        # Split ispaq-stats into batches if they exist
        if ispaq is not None and len(ispaq) > 0:
            Logger.info('Preparing quality metrics batches.')
            if ispaq.index.name != 'startday':
                ispaq['startday'] = ispaq['start'].str[0:10]
                ispaq = ispaq.set_index(['startday'])
            if 'short_target' not in ispaq.columns:
                ispaq['short_target'] = ispaq['target'].str[3:-2]
            ispaq_groups = ispaq.groupby('startday')
            # Create batch list of ispaq-stats
            for unique_date_utc in unique_date_list:
                # Now that "startday" is set as index, split into batches:
                try:
                    day_stats_list.append(ispaq_groups.get_group(
                        unique_date_utc))
                except KeyError:
                    Logger.warning(
                        'No data quality metrics for %s', unique_date_utc)
                    day_stats_list.append(None)
        # Just create extra references to same ispaq-stats dataframe (should
        # not consume extra memory with references only).
        if not day_stats_list:
            for unique_date_utc in unique_date_list:
                day_stats_list.append(ispaq)
        # Check if we can get the selected stations to shorten the inventory
        # sent to the workers
        station_match_strs = []
        selected_station_lists = []
        # Check if inventory.select supports fnmatch-based extended globbing
        # (not in all obspy releases):
        tmp_inv = inv.select(station='@(*[A-Z]*|*[A-Z]*)')
        fnmatch_supported_inv_select = False
        if len(tmp_inv) > 0:
            fnmatch_supported_inv_select = True
        Logger.info(
            'Preparing station match strings for %s event file batches '
            '(%s events) to limit the size of the transmitted inventory.',
            len(event_file_batches), len(sfiles or catalog))
        # Make 'sfile' the index of the catalog_df for faster lookups
        if catalog_df is not None and len(catalog_df) > 0:
            catalog_df = catalog_df.set_index(['sfile'], inplace=False)
        if stations_df is not None and len(stations_df) > 0:
            if add_array_picks:
                array_stations_dict = get_array_stations_from_df(stations_df)
            # array_sites=array_sites)
            if add_large_aperture_array_picks:
                large_array_stations_dict = get_array_stations_from_df(
                    stations_df,  # # array_sites=array_sites,
                    seisarray_prefixes=LARGE_APERTURE_SEISARRAY_PREFIXES)
        for event_file_batch in event_file_batches:
            ev_station_fnmatch_str = '*'
            # If there's only 1 file in batch, retrieve stations; otherwise
            # it's okay to send the full inventory onwards to the workers.
            # TODO: create the station match string from the station-list for
            #       each event, or dataframe to avoid reading each event.
            # if event_stations_filter:
            #    event_stations_filter.iloc[1]
            ev_stations = []
            if len(event_file_batch) == 1:
                event_file = event_file_batch[0]
                if isinstance(event_file, str):
                    # Quickest way to get stations is from catalog dataframe
                    if catalog_df is not None:
                        # Get the filename, with / without REA path
                        if sfiles_include_path:
                            event_file_name = os.path.join(*os.path.normpath(
                                event_file).split(os.sep)[-4:])
                        else:  # Get only the filename without the path
                            event_file_name = os.path.split(event_file)[-1]
                        # 'sfile' is the index, gives quick lookup
                        event_df = catalog_df.loc[event_file_name]
                        # TODO: make sure that there cannot be two events with
                        #       same filename when merging two databases.
                        #       May need to use the full path to the file..
                        if isinstance(event_df, pd.DataFrame):
                            event_df = event_df.iloc[0]
                        # ev_stations = list(event_df.stations)
                        ev_stations = [
                            sta.strip()
                            for sta in event_df['stations'].split(',')]
                    # no stations could happen when dataframe doesnt contain
                    # station info or when there is no dataframe
                    if len(ev_stations) == 0:
                        # If no catalog dataframe is available, check whether
                        # to read the event from a file
                        event = read_nordic(event_file, **kwargs)[0]
                        ev_stations = list(set([pick.waveform_id.station_code
                                                for pick in event.picks]))
                elif isinstance(event_file, Event):
                    # 3rd alt: event is already an obspy Event object
                    event = event_file
                    ev_stations = list(set([pick.waveform_id.station_code
                                            for pick in event.picks]))
                # When we get a list of the stations for the event, we may
                # need to add array stations so that the relevant inventory
                # is transmitted.
                if add_array_picks:
                    array_sites = list(set(get_station_sites(ev_stations)))
                    # array_stations = get_array_stations(array_sites)
                    array_stations = []
                    for array_site in array_sites:
                        try:
                            array_stations += array_stations_dict[
                                SEISARRAY_STATIONS[array_site]]
                        except KeyError:
                            continue
                    ev_stations = list(set(ev_stations + array_stations))
                if add_large_aperture_array_picks:
                    large_array_sites = set(get_station_sites(
                        ev_stations,
                        seisarray_prefixes=LARGE_APERTURE_SEISARRAY_PREFIXES))
                    large_array_stations = []
                    for large_array_site in large_array_sites:
                        try:
                            large_array_stations += large_array_stations_dict[
                                SEISARRAY_STATIONS[large_array_site]]
                        except KeyError:
                            continue
                    ev_stations = list(set(ev_stations + large_array_stations))
                ev_stations = list(
                    set(ev_stations).intersection(set(selected_stations)))
                ev_station_fnmatch_str = '@(' + '|'.join(ev_stations) + ')'
                Logger.debug('Prepared event batch: Event %s stations: %s',
                             event_file, ev_stations)
            if fnmatch_supported_inv_select:
                station_match_strs.append(ev_station_fnmatch_str)
            else:
                station_match_strs.append('*')
            if len(ev_stations) > 0:
                selected_station_lists.append(ev_stations)
            else:
                selected_station_lists.append(selected_stations)

        Logger.info('Start parallel template creation.')
        # With multiprocessing as backend, this parallel loop seems to be more
        # resistant to running out of memory; hence can run with more cores
        # simultaneously. However, if we are scartching memory, then more cores
        # will not have any speedup (but it's still more stable with multiproc)
        res_out = Parallel(n_jobs=cores, backend='multiprocessing')(
            delayed(_create_template_objects)(
                events_files=event_file_batch.copy(),
                # selected_stations=selected_stations,
                selected_stations=selected_station_lists[nbatch],
                template_length=template_length,
                lowcut=lowcut, highcut=highcut, min_snr=min_snr,
                prepick=prepick, samp_rate=samp_rate,
                seisan_wav_path=seisan_wav_path, clients=clients,
                inv=new_inv.select(time=UTCDateTime(unique_date_list[nbatch]),
                                   station=station_match_strs[nbatch]),
                remove_response=remove_response, output=output,
                noise_balancing=noise_balancing,
                balance_power_coefficient=balance_power_coefficient,
                apply_agc=apply_agc, agc_window_sec=agc_window_sec,
                ground_motion_input=ground_motion_input,
                write_out=write_out, templ_path=templ_path, prefix=prefix,
                min_n_traces=min_n_traces, make_pretty_plot=make_pretty_plot,
                check_template_strict=check_template_strict,
                allow_channel_duplication=allow_channel_duplication,
                normalize_NSLC=normalize_NSLC, add_array_picks=add_array_picks,
                add_large_aperture_array_picks=add_large_aperture_array_picks,
                stations_df=stations_df, ispaq=day_stats_list[nbatch],
                sta_translation_file=sta_translation_file,
                std_network_code=std_network_code,
                std_location_code=std_location_code,
                std_channel_prefix=std_channel_prefix,
                vertical_chans=vertical_chans,
                horizontal_chans=horizontal_chans,
                stations_with_verticals_for_s=stations_with_verticals_for_s,
                wavetool_path=wavetool_path,
                erase_mags=erase_mags,
                parallel=False, cores=1,
                thread_parallel=thread_parallel, n_threads=n_threads,
                *args, **kwargs)
            for nbatch, event_file_batch in enumerate(event_file_batches))

        tribes = [r[0] for r in res_out if r is not None and len(r[0]) > 0]
        wavnames = [r[1][0] for r in res_out
                    if r is not None and len(r[1]) > 0]
        tribe = Tribe(templates=[templ for tri in tribes for templ in tri])
    else:
        if sfiles:
            events_files = sfiles
        elif catalog:
            events_files = catalog
        else:
            raise NotImplementedError(not_cat_or_sfiles_msg)
        Logger.info('Start serial template creation.')
        (tribe, wavnames) = _create_template_objects(
            events_files=events_files,
            selected_stations=selected_stations,
            template_length=template_length,
            lowcut=lowcut, highcut=highcut, min_snr=min_snr,
            prepick=prepick, samp_rate=samp_rate,
            seisan_wav_path=seisan_wav_path, clients=clients, inv=new_inv,
            remove_response=remove_response, output=output,
            noise_balancing=noise_balancing,
            balance_power_coefficient=balance_power_coefficient,
            ground_motion_input=ground_motion_input,
            min_n_traces=min_n_traces, make_pretty_plot=make_pretty_plot,
            write_out=write_out, templ_path=templ_path, prefix=prefix,
            parallel=parallel, cores=cores,
            check_template_strict=check_template_strict,
            allow_channel_duplication=allow_channel_duplication,
            add_array_picks=add_array_picks, stations_df=stations_df,
            add_large_aperture_array_picks=add_large_aperture_array_picks,
            ispaq=ispaq, normalize_NSLC=normalize_NSLC,
            sta_translation_file=sta_translation_file,
            std_network_code=std_network_code,
            std_location_code=std_location_code,
            std_channel_prefix=std_channel_prefix,
            vertical_chans=vertical_chans, horizontal_chans=horizontal_chans,
            stations_with_verticals_for_s=stations_with_verticals_for_s,
            wavetool_path=wavetool_path, erase_mags=erase_mags,
            thread_parallel=thread_parallel, n_threads=n_threads,
            *args, **kwargs)

    # Remove horizontal traces with P picks / vertical traces with S picks for
    # stations that shouldn't have them
    # (EQcorrscan's template_gen only adds S-picks to verticals if we include
    # horizontal channels in vertical_chans. BUt then it also adds P picks to
    # horizontal channels, so we want to remove these after. and we only want
    # to allows S-picks on Z channels at selected stations (maybe)
    # horizontal channels, bu
    for templ in tribe:
        # Here I want to keep S-picks on Z, but not P-picks on horizontal
        # channels:
        templ.st = Stream([tr for tr in templ.st if not (
            tr.stats.channel[-1] in horizontal_chans
            and tr.stats.extra.phase_hint[0] == 'P')])
        # stations_with_verticals_for_s # TODO: limit to specific stations
        if stations_with_verticals_for_s:
            templ.st = Stream([tr for tr in templ.st if not (
                tr.stats.channel[-1] in vertical_chans
                and tr.stats.station not in stations_with_verticals_for_s
                and tr.stats.extra.phase_hint[0] == 'S'
            )])

    # Make sure that there are no empty streams in tribe
    tribe = Tribe([templ for templ in tribe if len(templ.st) > 0])
    tribe = tribe.sort()
    # Add labels to the output files to indicate changes in processing
    label = ''
    if noise_balancing:
        label = label + 'balNoise_'
    if apply_agc:
        label = label + 'agc_'
    if task_id is not None:
        label = label + 'chunk_' + "{:02d}".format(task_id) + '_'
    if write_out:
        tribe_file_name = os.path.join(
            out_folder, prefix + 'Templates_min' + str(min_n_traces) +
            'tr_' + label + str(len(tribe)))
        Logger.info('Created %s templates, writing to tribe %s...',
                    len(tribe), tribe_file_name)
        tribe.write(tribe_file_name, max_events_per_file=max_events_per_file,
                    cores=cores)  # max_events_per_file=10)
        if write_individual_templates:
            for templ in tribe:
                templ.write(os.path.join(templ_path, prefix + templ.name +
                                         '.mseed'), format="MSEED")
    return tribe, wavnames


# %%
