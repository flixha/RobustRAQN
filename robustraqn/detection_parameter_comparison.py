#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 17:23:45 2017

@author: felix
"""

# %%
# Do some magic to avoid memory congestion; and getting stuck on the queue that
# processify creates to run each day's detection:
#### ATTENTION: this is apparently dangerous in some cases and needs testing/
# debugging the underlying issue
# from signal import signal, SIGPIPE, SIG_DFL
# signal(SIGPIPE, SIG_DFL)
# from processify import processify


import os, gc, glob, math, pickle, calendar, datetime
from importlib import reload
import numpy as np
import statistics as stats
from multiprocessing import Process, Queue, Manager

import logging
Logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)
logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")

from obspy import read_inventory
from obspy import UTCDateTime
from obspy.core.event import Catalog
from obspy.clients.filesystem.sds import Client

from eqcorrscan.core.match_filter import Tribe
from eqcorrscan.core.match_filter.party import Party
#import quality_metrics, spectral_tools, load_events_for_detection,\
#    event_detection, templates_creation
# reload(quality_metrics)
# reload(load_events_for_detection)
from quality_metrics import *
from load_events_for_detection import *
from event_detection import run_day_detection
from createTemplates_object import create_template_objects
from spectral_tools import Noise_model, get_updated_inventory_with_noise_models

#Set the path to the folders with continuous data:
archive_path = '/data/seismo-wav/SLARCHIVE'
client = Client(archive_path)

selectedStations = ['ASK','BER','BLS5','DOMB','EKO1','FOO','HOMB','HYA','KMY',
                    'MOL','ODD1','SKAR','SNART','STAV','SUE','KONO',
                    'BIGH','DRUM','EDI','EDMD','ESK','GAL1','GDLE','HPK',
                    'INVG','KESW','KPL','LMK','LRW','PGB1',
                    'EKB','EKB1','EKB2','EKB3','EKB4','EKB5','EKB6','EKB7',
                    'EKB8','EKB9','EKB10','EKR1','EKR2','EKR3','EKR4','EKR5',
                    'EKR6','EKR7','EKR8','EKR9','EKR10',
                    'NAO00','NAO01','NAO02','NAO03','NAO04','NAO05',
                    'NB200','NB201','NB202','NB203','NB204','NB205',
                    'NBO00','NBO01','NBO02','NBO03','NBO04','NBO05',
                    'NC200','NC201','NC202','NC203','NC204','NC205',
                    'NC300','NC301','NC302','NC303','NC304','NC305',
                    'NC400','NC401','NC402','NC403','NC404','NC405',
                    'NC600','NC601','NC602','NC603','NC604','NC605']
#selectedStations = ['ASK','BER','RUND']
# selectedStations  = ['ASK', 'BLS5', 'KMY', 'ODD1', 'NAO01', 'ESK', 'EDI', 'KPL']
# selectedStations  = ['ODD1', 'ESK']
relevantStations = get_all_relevant_stations(
    selectedStations, sta_translation_file="station_code_translation.txt")

seisanREApath = '../SeisanEvents/'
seisanWAVpath = '../SeisanEvents/'
sfiles = glob.glob(os.path.join(seisanREApath, '*L.S??????'))
sfiles.sort(key = lambda x: x[-6:])
# sfiles = sorted(glob.glob(os.path.join(seisanREApath, '30-0033-00L.S200806')))

self_detection_test = False
if self_detection_test:
    # Or do a systematic test of all events: but always exclude the event itself
    # from the templates to be created.
    startdays =\
        [UTCDateTime(int(sfiles[n][-6:-2]), int(sfiles[n][-2:]),
                    int(sfiles[n][-19:-17]), 0,0,0) for n in range(0,len(sfiles))]
    enddays = startdays
    # Take all the other events except the one of the day that is being tested
    sfile_lol = [[sfiles[i] for i in range(0,len(sfiles)) if i!=n]
                for n in range(0,len(sfiles))]
else:
    # startdays = [UTCDateTime(2011,7,20,0,0,0),
    #              UTCDateTime(2018,10,25,0,0,0),
    #              UTCDateTime(2019,7,10,0,0,0)]
    startdays = [UTCDateTime(2014,6,6,0,0,0),
                 UTCDateTime(2015,6,16,0,0,0),
                 UTCDateTime(2018,4,14,0,0,0),
                 UTCDateTime(2018,10,25,0,0,0)]
    enddays = startdays
    sfile_lol = [sfiles for d in startdays]


invFile = '~/Documents2/ArrayWork/Inventory/NorSea_inventory.xml'
inv = get_updated_inventory_with_noise_models(
    os.path.expanduser(invFile), pdf_dir='~/repos/ispaq/WrapperScripts/PDFs/',
    outfile='inv.pickle', check_existing=True, plot_station_pdf=True)

ispaq_folder=\
    '/home/felix/repos/ispaq/WrapperScripts/Parquet_database/csv_parquet'


# @processify
def compare_event_picks(party, catalog, sfile, st, sta_translation_file,
                        max_time_diff_s=3, min_cc=0.7, parallel=False,
                        cores=1):
    """
    Compare the automatic picks from a detected catalog against the manual
    picks from an S-file
    """
    catalog = party.lag_calc(
        st, pre_processed=False, shift_len=0.8,
        min_cc=min_cc, horizontal_chans=['E', 'N', '1', '2'],
        vertical_chans=['Z'], interpolate=False, plot=False,
        overlap='calculate', parallel=parallel, cores=cores)
    
    select = read_nordic(sfile, return_wavnames=False)
    s_event = select[0]
    s_event = prepare_picks(s_event, st,
        std_network_code='NS', std_location_code='00', std_channel_prefix='BH',
        sta_translation_file=sta_translation_file)
    
    good_picks_lol = list()
    bad_picks_lol = list()
    missing_picks_lol = list()
    for j, event in enumerate(catalog):
        # Good picks: Picks that lag-calc was able to retrieve and that agree 
        # with  manually set picks for the event.
        good_picks = list()
        # Missing picks: Picks that were not set even though (a) corresponding
        # waveforms exist in stream and (b) corresponding trace for pick exists
        # in template
        missing_picks = list()
        bad_picks = list()
        
        for pick in event.picks:
            pick_is_good = False
            for s_pick in s_event.picks:
                same_id = (pick.waveform_id == s_pick.waveform_id)
                same_phase = (pick.phase_hint.upper()[0] == 
                              s_pick.phase_hint.upper()[0]) 
                time_diff = pick.time - s_pick.time
                if same_id and same_phase and time_diff < max_time_diff_s:
                    good_picks.append(pick)
                    pick_is_good = True
                    break
            if not pick_is_good:
                bad_picks.append(pick)

        for s_pick in s_event.picks:
            not_picked = (s_pick not in good_picks and s_pick not in bad_picks)
            could_have_picked = False
            for family in party:
                if family.template.name in event.comments[1].text:
                    for tr in family.template.st:
                        if s_pick.waveform_id.id == tr.id:
                            could_have_picked = True
                            break
                if could_have_picked:
                    break
            #for tr in party[j].template.st:

            if not_picked and could_have_picked:
                missing_picks.append(s_pick)

        good_picks_lol.append(good_picks)
        bad_picks_lol.append(bad_picks)
        missing_picks_lol.append(missing_picks)
        gc.collect()
        
    return [good_picks_lol, bad_picks_lol, missing_picks_lol]


# %% ######################### LOOP OVER Parameters ###########################

def main():
    parallel = True
    cores = 50
    n_templates_per_run = 20
    # day-function currently causes a deadlock from the 2nd iteration onward.
    # Workaround right now is to dump the stream to disk and read it back in
    # instead of piping it back from the processified function
    dump_stream_to_disk = False
    
    cc_choice = 'best_cc'
    samp_rate = 20.0
    # samp_rate = 40.0
    # Set the range of parameters to test, and choose a set of default parameters
    
    # template_length_l = np.concatenate(
    #     [np.arange(2, 27, 4), np.arange(30, 50, 6), np.arange(56, 101, 12),
    #     np.arange(110, 190, 20)])
    # template_length_l = np.round(np.logspace(np.log10(2), np.log10(240), 19), 1)
    template_length_l = np.round(np.linspace(1.4, 12, 19)**2.3)
    # [np.arange(2, 27, 4), np.arange(30, 56, 5), np.arange(60, 101, 10), [120]])
    
    # lowcut_l = np.concatenate(
    #     [np.array([0.1, 0.15, 0.2, 0.3, 0.4]), np.arange(0.55, 0.9, 0.15),
    #     np.arange(1.1, 3.3, 0.3), np.arange(3.9, 7.5, 0.8)])
    # lowcut_l = np.round(np.logspace(np.log10(0.1), np.log10(6), 19), 2)
    lowcut_l = np.round(np.linspace(0.46, 1.85, 19)**3.0, 2)
    # lowcut_l = np.concatenate(
    #     [np.array([0.1, 0.15, 0.2, 0.3, 0.4]), np.arange(0.55, 0.9, 0.15),
    #     np.arange(1.1, 3.3, 0.3), np.arange(3.9, 6.0, 0.8)])
    # [np.arange(0.1, 0.4, 0.1), np.arange(0.55, 0.9, 0.15), np.arange(1.05, 4.0, 0.25)])
    #highcut_l = np.arange(2.799, 10.0, 0.4)
    highcut_l = np.round(np.logspace(np.log10(2.6), np.log10(9.9), 19), 1)
    #np.arange(2.799, 20.0, 0.95)
    min_snr_l = np.round(np.logspace(np.log10(1.2), np.log10(20), 19), 1)
    # np.arange(1.2, 10.5, 0.5)
    prepick_l = np.arange(0.0, 1.9, 0.1)
    noise_balancing = [False, True]
    noise_balancing = [False]
    variable_parameters = [template_length_l]
    # variable_parameters = [template_length_l, lowcut_l, highcut_l, min_snr_l] #,
                        #prepick_l]

    # variable_parameters = [
    #     np.arange(30,31,1), np.arange(2.5,2.6,1), np.arange(8.0,8.1,1),
    #     np.arange(4.0,4.1,1), np.arange(0.2,0.3,1), [False]]
    # default_parameters = [30, 2.5, 8.0, 4.0, 0.2, True]
    # default_parameters = [30, 2.5, 9.99, 4.0, 0.5]
    default_parameters = [60, 2.5, 9.9, 1.5, 0.5]

    # default_parameters = [30, 0.1, 9.99, 4.0, 0.5]
    # template_length_l[0] = 150
    ispaq = pd.DataFrame()
    previous_date = UTCDateTime(1,1,1,0,0,0)

    for k, sfile_list in enumerate(sfile_lol):
        startday = startdays[k]
        endday = enddays[k]
        
        for noise_bal in noise_balancing:
            # if k < 6:
            #     continue
            day_st = Stream()
            party_lol = list()
            pick_lol = list()
            for j, variable_parameter in enumerate(variable_parameters):
                party_list = list()
                pick_list = list()
                run_parameters = default_parameters.copy()
                
                # if j != 0:
                #     continue
                
                for p, par_value in enumerate(variable_parameter):
                    # if p != 8:
                    #     continue
                    
                    # get the right parameters for this parameter-check run
                    run_parameters[j] = par_value
                    template_length, lowcut, highcut, min_snr, prepick =\
                        run_parameters

                    Logger.info('Making new templates')
                    
                    #tribe = Tribe().read('TemplateObjects/Templates_min8tr_8.tgz')
                    tribe = create_template_objects(
                        sfile_list, selectedStations, template_length,
                        lowcut, highcut, min_snr, prepick, samp_rate,
                        seisanWAVpath, inv=inv, remove_response=True,
                        noise_balancing=noise_bal, min_n_traces=1,
                        parallel=True, cores=cores, write_out=False,
                        make_pretty_plot=False)

                    n_templates = len(tribe)
                    n_runs = math.ceil(n_templates / n_templates_per_run)
                    # For each day, read in data and run detection
                    date_list = pd.date_range(
                        startday.datetime, endday.datetime).tolist()
                    
                    if len(date_list) > 1:
                        day_st = Stream()
                    for date in date_list:
                        return_stream = False
                        if date != previous_date:
                            return_stream = True
                        previous_date = date
                        
                        # Load in Mustang-like ISPAQ stats for the whole year
                        if date.strftime("%Y-%m-%d") not in ispaq.index:
                            ispaq = read_ispaq_stats(
                                folder=ispaq_folder, stations=selectedStations,
                                startyear=date.year, endyear=date.year,
                                ispaq_prefixes=['all'],
                                ispaq_suffixes=['simpleMetrics','PSDMetrics'],
                                file_type = 'parquet')
                        
                        # return_stream=True is currently causing a deadlock                                            
                        [party, day_st2] = run_day_detection(
                            tribe=tribe, date=date, ispaq=ispaq, inv=inv,
                            remove_response=True, day_st=day_st,
                            parallel=parallel, cores=cores,
                            selectedStations=selectedStations,
                            trig_int=3, noise_balancing=noise_bal, 
                            balance_power_coefficient=2, multiplot=False,
                            write_party=False, return_stream=True, 
                            dump_stream_to_disk=dump_stream_to_disk)
                        
                        party = party.decluster(
                            trig_int=40, timing='detect', min_chans=10,
                            metric='thresh_exc', absolute_values=True)
                        
                        if len(day_st2) > 0:
                            day_st = day_st2
                        if dump_stream_to_disk:
                            day_st = pickle.load(open("tmp_st.pickle", "rb"))
                            party = pickle.load(open("tmp_party.pickle", "rb"))

                        party_list.append(party.copy())

                        # TODO: let lag-calc picker pick the arrivals, and then
                        # compare the picks against the manual picks. Compute ratio
                        # of good vs. bad picks (and add plot in other script for 
                        # these stats).                    
                        # Pick events, then compare automatic against manual picks
                        if len(party) == 0:
                            pick_list.append((list(), list(), list()))
                            continue
                        
                        picking_party = Party([f for f in party if len(f) > 0])
                        picked_catalog = Catalog()
                        
                        # min_cc = threshold / n_channels * 10
                        # Or better use the minimum CC as reference here? Average
                        # CC seems to be higher for weaker earthquakes as templates
                        # min_cc = min([min([d.threshold / d.no_chans for d in f])
                        #     for f in party])
                        if cc_choice == 'best_cc':
                            best_cc = max(
                                [max([d.detect_val / d.no_chans for d in f])
                                for f in party])
                            # min_cc = best_cc * 1.5
                            # min_cc = best_cc + (1 - best_cc) * 0.2
                            min_cc = best_cc + (1 - best_cc) * best_cc / 3
                            Logger.info(
                                'Average CC of the best correlations is %s, setting'
                                + ' minimum CC for lag-calc to %s', best_cc,
                                min_cc)
                        elif cc_choice == 'av_cc':
                            avg_cc = stats.mean(
                                [stats.mean([d.threshold / d.no_chans for d in f])
                                for f in party])
                            # min_cc = avg_cc * 4
                            min_cc = avg_cc + (1 - avg_cc) * avg_cc * 3
                            Logger.info(
                                'Average CC of the correlations is %s, setting'
                                + ' minimum CC for lag-calc to %s', avg_cc,
                                min_cc)
                        
                        if self_detection_test:
                            res = compare_event_picks(
                                picking_party, picked_catalog, sfiles[k],
                                day_st, sta_translation_file=
                                "station_code_translation.txt",
                                min_cc=min_cc, max_time_diff_s=2,
                                parallel=parallel, cores=cores)
                            [good_picks, bad_picks, missing_picks] = res
                            pick_list.append((good_picks, bad_picks,
                                            missing_picks))
                        else:
                            pick_list.append((list(), list(), list()))
                            
                        n = gc.collect()
                        Logger.info("Number of unreachable objects collected "
                                    + "by GC: %s", str(n))
                        Logger.info("Uncollectable garbage: %s", gc.garbage)
                        
                party_lol.append(party_list)
                pick_lol.append(pick_list)
                
            if noise_bal:
                pickle_file = 'ParameterTests_07/party_comparison_'\
                    + str(startday)[0:10] + 'wiNoiseBal.pickle'
            else:
                pickle_file = 'ParameterTests_07/party_comparison_'\
                    + str(startday)[0:10] + 'noNoiseBal.pickle'
            pickle.dump([party_lol, pick_lol, variable_parameters],
                       open(pickle_file, "wb" ), -1)


if __name__ == "__main__":
    main()

# %%


# tribe = create_template_objects(sfile_list, selectedStations, 20, 2.5, 9.9, 3, 0.5, 20.0, seisanWAVpath, inv=inv, remove_response=True, noise_balancing=True, min_n_traces=1, parallel=True, cores=10, write_out=False, make_pretty_plot=False)
