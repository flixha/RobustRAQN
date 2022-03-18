#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Felix Halpaap
"""

# %%
import os
import glob
# from importlib import reload
from multiprocessing import Pool, cpu_count, get_context
from multiprocessing.pool import ThreadPool
from joblib import Parallel, delayed, parallel_backend
import pandas as pd

# from obspy import read_events, read_inventory
from obspy.core.event import Catalog
from obspy.core.utcdatetime import UTCDateTime
from obspy.geodetics.base import gps2dist_azimuth
from obspy.core.stream import Stream
# from obspy.core.util.base import TypeError
# from obspy.core.event import Event
from obspy.io.nordic.core import read_nordic
from obspy.core.inventory.inventory import Inventory

from obsplus.stations.pd import stations_to_df

# reload(eqcorrscan)
from eqcorrscan.utils import pre_processing
from eqcorrscan.core import template_gen
from eqcorrscan.utils.catalog_utils import filter_picks
from eqcorrscan.core.match_filter import Template, Tribe
# from eqcorrscan.utils.clustering import cluster
# from eqcorrscan.utils.stacking import linstack, PWS_stack
from eqcorrscan.utils.plotting import pretty_template_plot
from eqcorrscan.utils.correlate import pool_boy

# import load_events_for_detection
# import spectral_tools
# reload(load_events_for_detection)
from robustraqn.load_events_for_detection import (
    normalize_NSLC_codes, get_all_relevant_stations, load_event_stream,
    try_remove_responses, check_template, prepare_picks)
from robustraqn.spectral_tools import (
    st_balance_noise, Noise_model, get_updated_inventory_with_noise_models)
from robustraqn.seimic_array_tools import (
    extract_array_picks, add_array_station_picks,
    LARGE_APERTURE_SEISARRAY_PREFIXES, get_updated_stations_df)


import logging
Logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


def _shorten_tribe_streams(
        tribe, tribe_len_pct=0.2, max_tribe_len=None,
        min_n_traces=0, write_out=False, make_pretty_plot=False,
        prefix='short', noise_balancing=False,
        write_individual_templates=False):
    """
    Create shorter templates from a tribe of longer templates
    """
    if len(tribe) == 0:
        return tribe
    if tribe_len_pct is not None:
        new_templ_len = (tribe[0].st[0].stats.endtime -
                         tribe[0].st[0].stats.starttime) * tribe_len_pct
    else:
        new_templ_len = max_tribe_len
    short_tribe = tribe.copy()
    for templ in short_tribe:
        for tr in templ.st:
            tr.trim(starttime=tr.stats.starttime,
                    endtime=tr.stats.starttime + new_templ_len)
        if len(templ.st) >= min_n_traces:
            templ_name = str(
                templ.event.preferred_origin().time)[0:22] + '_' + 'templ'
            templ_name = templ_name.lower().replace('-', '_')\
                .replace(':', '_').replace('.', '_').replace('/', '')
            # make a nice plot
            if make_pretty_plot:
                image_name = os.path.join('TemplatePlots',
                                          prefix + '_' + templ_name)
                pretty_template_plot(
                    templ.st, background=False, event=templ.event,
                    sort_by='distance', show=False, return_figure=False,
                    size=(25, 50), save=True, savefile=image_name)
            Logger.info("Made template" + templ_name)
            Logger.info(templ)
    label = ''
    if noise_balancing:
        label = label + 'balNoise_'
    if write_out:
        short_tribe.write('TemplateObjects/' + prefix + 'Templates_min'
                    + str(min_n_traces) + 'tr_' + label + str(len(short_tribe)))
                    #max_events_per_file=10)
    if write_individual_templates:
        for templ in short_tribe:
            templ.write('Templates/' + prefix + templ.name + '.mseed',
                        format="MSEED")
    return short_tribe


def _create_template_objects(
        sfiles, selectedStations, template_length, lowcut, highcut, min_snr,
        prepick, samp_rate, seisan_wav_path, inv=Inventory(), clients=[],
        remove_response=False, noise_balancing=False,
        balance_power_coefficient=2, ground_motion_input=[],
        min_n_traces=8, write_out=False, templ_path='Templates/' ,
        make_pretty_plot=False, prefix='',
        check_template_strict=True, allow_channel_duplication=True,
        normalize_NSLC=True, add_array_picks=False, stations_df=pd.DataFrame(),
        add_large_aperture_array_picks=False,
        sta_translation_file="station_code_translation.txt",
        std_network_code='NS', std_location_code='00', std_channel_prefix='BH',
        parallel=False, cores=1, *args, **kwargs):
    """
    """
    sfiles.sort(key=lambda x: x[-6:])
    tribe = Tribe()
    template_names = []
    catalogForTemplates = Catalog()
    catalog = Catalog()

    wavnames = []
    # Loop over all S-files that each contain one event
    for j, sfile in enumerate(sfiles):
        Logger.info('Working on S-file: ' + sfile)
        select, wavname = read_nordic(sfile, return_wavnames=True)
        relevantStations = get_all_relevant_stations(
            selectedStations, sta_translation_file=sta_translation_file)
        origin = select[0].preferred_origin()

        # Load picks and normalize
        tempCatalog = filter_picks(select, stations=relevantStations)
        if not tempCatalog:
            continue
        event = tempCatalog[0]
        if not event.picks:
            continue
        tempCatalog = Catalog()
        tempCatalog += event
        catalog += event
        #######################################################################
        # Load and quality-control stream and picks for event
        wavef = load_event_stream(
            event, sfile, seisan_wav_path, relevantStations, clients=clients,
            min_samp_rate=samp_rate, pre_event_time=prepick,
            template_length=template_length)
        if wavef is None or len(wavef) == 0:
            Logger.error('Event %s for sfile %s has no waveforms available',
                         event.short_str(), sfile)
            return None

        if remove_response:
            wavef = try_remove_responses(
                wavef, inv, taper_fraction=0.15, pre_filt=[0.01, 0.05, 45, 50],
                parallel=parallel, cores=cores, **kwargs)
            for tr in wavef:
                try:
                    tr.stats.distance = gps2dist_azimuth(
                        origin.latitude, origin.longitude,
                        tr.stats.coordinates.latitude,
                        tr.stats.coordinates.longitude)[0]
                except Exception as e:
                    Logger.warning('Could not compute distance for event %s -'
                                   ' trace %s', event.short_str(), tr.id)
                    Logger.warning(e)
        wavef = wavef.detrend(type='simple')

        # standardize all codes for network, station, location, channel
        if normalize_NSLC:
            wavef = normalize_NSLC_codes(
                wavef, inv, sta_translation_file=sta_translation_file,
                std_network_code=std_network_code,
                std_location_code=std_location_code,
                std_channel_prefix=std_channel_prefix)

        # Do noise-balancing by the station's PSDPDF average
        if noise_balancing:
            # if not hasattr(wavef, "balance_noise"):
            #     bound_method = st_balance_noise.__get__(wavef)
            #     wavef.balance_noise = bound_method
            wavef = wavef.filter('highpass', freq=0.1, zerophase=True
                                 ).detrend()
            wavef = st_balance_noise(
                wavef, inv,
                balance_power_coefficient=balance_power_coefficient,
                ground_motion_input=ground_motion_input)
            wavef = wavef.detrend('linear').taper(
                0.15, type='hann', max_length=30, side='both')

        # Add picks at array stations if requested
        if add_array_picks:
            array_picks_dict = extract_array_picks(event=event)
            event = add_array_station_picks(
                event=event, array_picks_dict=array_picks_dict,
                stations_df=stations_df, **kwargs)
            if add_large_aperture_array_picks:
                array_picks_dict = extract_array_picks(event=event)
                Logger.info('Adding array picks for large aperture arrays')
                event = add_array_station_picks(
                    event=event, array_picks_dict=array_picks_dict,
                    stations_df=stations_df,
                    seisarray_prefixes=LARGE_APERTURE_SEISARRAY_PREFIXES,
                    **kwargs)

        event = prepare_picks(
            event=event, stream=wavef, normalize_NSLC=normalize_NSLC, inv=inv,
            sta_translation_file=sta_translation_file, *args, **kwargs)

        wavef = pre_processing.shortproc(
            st=wavef, lowcut=lowcut, highcut=highcut, filt_order=4,
            samp_rate=samp_rate, parallel=False, num_cores=1)
        # data_envelope = obspy.signal.filter.envelope(st_filt[0].data)

        # Make the templates from picks and waveforms
        catalogForTemplates += event
        ### TODO : this is where a lot of calculated picks are thrown out
        templateSt = template_gen._template_gen(
            picks=event.picks, st=wavef, length=template_length, swin='all',
            prepick=prepick, all_horiz=True, plot=False, delayed=True,
            min_snr=min_snr)
        # quality-control template
        if len(templateSt) == 0:
            continue
        if check_template_strict:
            templateSt = check_template(
                templateSt, template_length, remove_nan_strict=True,
                allow_channel_duplication=allow_channel_duplication,
                max_perc_zeros=5)

        # templ_name = str(event.origins[0].time) + '_' + 'templ'
        templ_name = str(event.preferred_origin().time)[0:22] + '_' + 'templ'
        templ_name = templ_name.lower().replace('-', '_')\
            .replace(':', '_').replace('.', '_').replace('/', '')
        # templ_name = templ_name.lower().replace(':','_')
        # templ_name = templ_name.lower().replace('.','_')
        # template.write('TemplateObjects/' + templ_name + '.mseed',
        # format="MSEED")
        template_names.append(templ_name)
        # except:
        #    print("WARNING: There was an issue creating a template for " +
        # sfile)
        # t = Template().construct(
        #     method=None,picks=event.picks, st=templateSt,length=7.0,
        #     swin='all', prepick=0.2, all_horiz=True, plot=False,
        #     delayed=True, min_snr=1.2, name=templ_name, lowcut=2.5,
        #     highcut=8.0,samp_rate=20, filt_order=4,event=event,
        #     process_length=300.0)
        t = Template(name=templ_name, event=event, st=templateSt,
                     lowcut=lowcut, highcut=highcut, samp_rate=samp_rate,
                     filt_order=4, process_length=86400.0, prepick=prepick)
        # highcut=8.0, samp_rate=20, filt_order=4, process_length=86400.0,

        if len(t.st) >= min_n_traces:
            if write_out:
                templ_filename = os.path.join(templ_path, templ_name + '.mseed')
                t.write(templ_filename, format="MSEED")
            tribe += t
            # add wavefile-name to output
            wavnames.append(wavname[0])

            # make a nice plot
            sfile_path, sfile_name = os.path.split(sfile)
            if make_pretty_plot:
                image_name = os.path.join('TemplatePlots',
                                        prefix + '_' + templ_name)
                pretty_template_plot(
                    templateSt, background=wavef, event=event, sort_by='distance',
                    show=False, return_figure=False, size=(25, 50), save=True,
                    savefile=image_name)
            Logger.info("Made template" + templ_name)
            Logger.info(t)

    # clusters = tribe.cluster(method='space_cluster', d_thresh=1.0, show=True)
    template_list = []
    for j, templ in enumerate(tribe):
        template_list.append((templ.st, j))

    return (tribe, wavnames)

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
        #templ.event.preferred_magnitude
        for mag_pref in reversed(mag_preference_priority):
            for mag in templ.event.magnitudes:
                if ((mag_pref[0] == mag.magnitude_type or None) and
                        (mag_pref[1] == mag.creation_info.agency_id or None)):
                    templ.event.preferred_magnitude_id = mag.resource_id
    return tribe


def create_template_objects(
        sfiles, selectedStations, template_length, lowcut, highcut, min_snr,
        prepick, samp_rate, seisan_wav_path, clients=[], inv=Inventory(),
        remove_response=False, noise_balancing=False,
        balance_power_coefficient=2, ground_motion_input=[],
        min_n_traces=8, write_out=False, templ_path='Templates',
        prefix='', make_pretty_plot=False,
        check_template_strict=True, allow_channel_duplication=True,
        normalize_NSLC=True, add_array_picks=False,
        sta_translation_file="station_code_translation.txt",
        std_network_code='NS', std_location_code='00', std_channel_prefix='BH',
        parallel=False, cores=1, *args, **kwargs):
    """
      Wrapper for create-template-function
    """
    # Get only relevant inventory information to make Pool-startup quicker
    new_inv = Inventory()
    for sta in selectedStations:
        new_inv += inv.select(station=sta)
    stations_df = get_updated_stations_df(inv)

    if parallel and len(sfiles) > 1:
        if cores is None:
            cores = min(len(sfiles), cpu_count())

        # Check if I can allow multithreading in each of the parallelized
        # subprocesses:
        thread_parallel = False
        n_threads = 1
        # if cores > 2 * len(sfiles):
        #     thread_parallel = True
        #     n_threads = int(cores / len(sfiles))

        # Is this I/O or CPU limited task?
        # Test on bigger problem (350 templates):
        # Threadpool: 10 minutes vs Pool: 7 minutes
        # TODO: the problem is deep-copying of the inventory to the threads/
        # processes. inv can be empty Inv, or chosen more carefully for the
        # problem to speed this up.
        # e.g. with : channel='?H?', latitude=59, longitude=2, maxradius=7

        # with pool_boy(Pool=get_context("spawn").Pool, traces=len(sfiles),
        #               n_cores=cores) as pool:
        #     results = (
        #         [pool.apply_async(
        #             _create_template_objects,
        #             ([sfile], selectedStations, template_length,
        #                 lowcut, highcut, min_snr, prepick, samp_rate,
        #                 seisan_wav_path),
        #             dict(
        #                 inv=new_inv.select(
        #                     time=UTCDateTime(sfile[-6:] + sfile[-19:-9])),
        #                 clients=clients,
        #                 remove_response=remove_response,
        #                 noise_balancing=noise_balancing,
        #                 balance_power_coefficient=balance_power_coefficient,
        #                 ground_motion_input=ground_motion_input,
        #                 write_out=False, min_n_traces=min_n_traces,
        #                 make_pretty_plot=make_pretty_plot, prefix=prefix,
        #                 check_template_strict=check_template_strict,
        #                 allow_channel_duplication=allow_channel_duplication,
        #                 normalize_NSLC=normalize_NSLC,
        #                 sta_translation_file=sta_translation_file,
        #                 std_network_code=std_network_code,
        #                 std_location_code=std_location_code,
        #                 std_channel_prefix=std_channel_prefix,
        #                 parallel=thread_parallel, cores=n_threads)
        #             ) for sfile in sfiles])
        # # try:
        # res_out = [res.get() for res in results]

        res_out = Parallel(n_jobs=cores)(
            delayed(_create_template_objects)(
                [sfile], selectedStations, template_length, lowcut, highcut,
                min_snr, prepick, samp_rate, seisan_wav_path, clients=clients,
                inv=new_inv.select(
                    time=UTCDateTime(sfile[-6:] + sfile[-19:-14])),
                remove_response=remove_response,
                noise_balancing=noise_balancing,
                balance_power_coefficient=balance_power_coefficient,
                ground_motion_input=ground_motion_input,
                write_out=write_out, templ_path=templ_path, prefix=prefix,
                min_n_traces=min_n_traces, make_pretty_plot=make_pretty_plot, 
                check_template_strict=check_template_strict,
                allow_channel_duplication=allow_channel_duplication,
                normalize_NSLC=normalize_NSLC, add_array_picks=add_array_picks,
                stations_df=stations_df,
                sta_translation_file=sta_translation_file,
                std_network_code=std_network_code,
                std_location_code=std_location_code,
                std_channel_prefix=std_channel_prefix,
                parallel=thread_parallel, cores=n_threads,
                *args, **kwargs)
            for sfile in sfiles)

        tribes = [r[0] for r in res_out if r is not None and len(r[0]) > 0]
        wavnames = [r[1][0] for r in res_out
                    if r is not None and len(r[0]) > 0]
        tribe = Tribe(templates=[tri[0] for tri in tribes if len(tri) > 0])
        # except IndexError:
        #    tribe = Tribe()
        #    wavnames = ()

        # pool.close()
        # pool.join()
        # pool.terminate()
    else:
        (tribe, wavnames) = _create_template_objects(
            sfiles, selectedStations, template_length, lowcut, highcut,
            min_snr, prepick, samp_rate, seisan_wav_path, inv=new_inv,
            clients=clients,
            remove_response=remove_response, noise_balancing=noise_balancing,
            balance_power_coefficient=balance_power_coefficient,
            ground_motion_input=ground_motion_input,
            min_n_traces=min_n_traces, make_pretty_plot=make_pretty_plot,
            write_out=write_out, templ_path=templ_path, prefix=prefix,
            parallel=parallel, cores=cores,
            check_template_strict=check_template_strict,
            allow_channel_duplication=allow_channel_duplication,
            normalize_NSLC=normalize_NSLC,
            sta_translation_file=sta_translation_file,
            std_network_code=std_network_code,
            std_location_code=std_location_code,
            std_channel_prefix=std_channel_prefix,
            *args, **kwargs)

    label = ''

    if noise_balancing:
        label = label + 'balNoise_'
    if write_out:
        tribe.write('TemplateObjects/' + prefix + 'Templates_min'
                    + str(min_n_traces) + 'tr_' + label + str(len(tribe)))
                    #max_events_per_file=10)
        for templ in tribe:
            templ.write(os.path.join(
                templ_path, prefix + templ.name + '.mseed'), format="MSEED")

    return tribe, wavnames


# %% ############## MAIN ###################

if __name__ == "__main__":
    seisan_rea_path = '../SeisanEvents/'
    seisan_wav_path = '../SeisanEvents/'
    selectedStations = ['ASK','BER','BLS5','DOMB','EKO1','FOO','HOMB','HYA',
                        'KMY','MOL','ODD1','SKAR','SNART','STAV','SUE','KONO',
                        'BIGH','DRUM','EDI','EDMD','ESK','GAL1','GDLE','HPK',
                        'INVG','KESW','KPL','LMK','LRW','PGB1','MUD',
                        'EKB','EKB1','EKB2','EKB3','EKB4','EKB5','EKB6','EKB7',
                        'EKB8','EKB9','EKB10','EKR1','EKR2','EKR3','EKR4',
                        'EKR5','EKR6','EKR7','EKR8','EKR9','EKR10',
                        'NAO00','NAO01','NAO02','NAO03','NAO04','NAO05',
                        'NB200','NB201','NB202','NB203','NB204','NB205',
                        'NBO00','NBO01','NBO02','NBO03','NBO04','NBO05',
                        'NC200','NC201','NC202','NC203','NC204','NC205',
                        'NC300','NC301','NC302','NC303','NC304','NC305',
                        'NC400','NC401','NC402','NC403','NC404','NC405',
                        'NC600','NC601','NC602','NC603','NC604','NC605'] 
    # selectedStations = ['NAO01', 'NAO03', 'NB200']
    # selectedStations = ['ASK', 'BER', 'NC602']
    # 'SOFL','OSL',
    invFile = '~/Documents2/ArrayWork/Inventory/NorSea_inventory.xml'
    inv = get_updated_inventory_with_noise_models(
        os.path.expanduser(invFile),
        pdf_dir='~/repos/ispaq/WrapperScripts/PDFs/', check_existing=True,
        outfile=os.path.expanduser('~/Documents2/ArrayWork/Inventory/inv.pickle'))

    template_length = 40.0
    parallel = True
    noise_balancing = False
    cores = 20

    sfiles = glob.glob(os.path.join(seisan_rea_path, '*L.S??????'))
    sfiles = glob.glob(os.path.join(seisan_rea_path, '24-1338-14L.S201909'))
    # sfiles = glob.glob(os.path.join(seisan_rea_path, '04-1734-46L.S200706'))
    # sfiles = glob.glob(os.path.join(seisan_rea_path, '24-0101-20L.S200707'))
    # sfiles = glob.glob(os.path.join(seisan_rea_path, '20-1814-05L.S201804'))
    # sfiles = glob.glob(os.path.join(seisan_rea_path, '01-0545-55L.S201009'))
    # sfiles = glob.glob(os.path.join(seisan_rea_path, '30-0033-00L.S200806'))
    sfiles = glob.glob(os.path.join(seisan_rea_path, '05-1741-44L.S202101'))
    sfiles.sort(key=lambda x: x[-6:])

    highcut = 9.9
    if noise_balancing:
        lowcut = 0.5
    else:
        lowcut = 2.5

    # create_template_objects(sfiles, selectedStations, inv,
    tribe, wavenames = create_template_objects(
        sfiles, selectedStations, template_length, lowcut, highcut,
        min_snr=4.0, prepick=0.2, samp_rate=20.0, inv=inv,
        remove_response=True, seisan_wav_path=seisan_wav_path,
        noise_balancing=noise_balancing, min_n_traces=3,
        parallel=parallel, cores=cores, write_out=False, make_pretty_plot=True)
