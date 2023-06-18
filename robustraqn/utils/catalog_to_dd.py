
import pandas as pd
import logging
Logger = logging.getLogger(__name__)



def _read_correlation_file(existing_corr_file, event_id_mapper=None):
    """
    From Eleanor R. H. Mestel (user erhmestel) in eqcorrscan pull request # 516:

    Read in existing correlation file into a dictionary of existing pairs
    :type existing_corr_file: str
    :param existing_corr_file:
        File path for existing correlations file.
    :rtype: dict
    :return: Existing pairs in format {id1: [id2, id3 ...], id2: [id3 ...]}
    """
    existing_pairs = {}
    counter = 0
    with open(existing_corr_file, "r") as f:
        for line in f:
            cols = line.split()
            if cols[0] == "#":
                in_event_id_1 = int(cols[1])
                in_event_id_2 = int(cols[2])
                in_XX = cols[3] ## not clear to me what this 0.0 is

                for key, value in event_id_mapper.items():
                    if value == in_event_id_1:
                        in_eventid1 = key
                    elif value == in_event_id_2:
                        in_eventid2 = key

                if in_eventid1 in existing_pairs:
                    existing_pairs[in_eventid1].append(in_eventid2)
                else:
                    existing_pairs.update({in_eventid1: [in_eventid2]}) 

                counter +=1

            #    # tried to set it up with differential_times dictionary 
            #    # but can't make full _DTObs because no tt1 & tt2 just dt
            #    
            #    diff_time = _EventPair(
            #                event_id_1=in_event_id_1,
            #                event_id_2=in_event_id_2)

            #elif len(cols) == 4:
            #    in_sta = cols[0]
            #    in_dt = float(cols[1])
            #    in_weight = float(cols[2])
            #    in_phase = cols[3]

            #
            #    diff_time.obs.append(
            #            _DTObs(station=in_sta,
            #                   tt1=XX,
            #                   tt2=XX, weight=in_weight,
            #                   phase=in_phase))

    Logger.info(
        f"{counter} existing correlation measurements from {existing_corr_file}")
    return existing_pairs




# %%

import os
import pandas as pd
import numpy as np
import glob
import csv
from collections import defaultdict
from wcmatch import fnmatch
import logging
import swifter

from concurrent.futures import ThreadPoolExecutor

# from robustraqn.core.seismic_array import SEISARRAY_PREFIXES
from robustraqn.utils.catalog_to_dd import _read_correlation_file
from robustraqn.utils.growclust import read_evlist_file

SEISARRAY_PREFIXES = [
     '@(ARCES|AR[ABCDE][0-9])', '@(SPITS|SP[ABC][0-5])',
    '@(BEAR|BJO|BJO1|BEA[1-6])',
]

Logger = logging.getLogger(__name__)
log_format = "%(asctime)s\t%(name)40s:%(lineno)s\t%(funcName)20s()\t%(levelname)s\t%(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format)

# @profile
def _read_correlation_file_quick(existing_corr_file, SEISARRAY_PREFIXES,
                                 t_diff_max=None):
    """
    Quickly read in a dt.cc file and return a dictionary of dictionaries of
    dataframes with the dt.cc values for each event pair.
    """
    cc_df = pd.read_csv(existing_corr_file, delim_whitespace=True, header=None,
                        names=['station', 'dt', 'cc', 'phase'])
    if t_diff_max is not None:
        # Drop rows where station is not "#" and where abs(dt) > t_diff_max
        cc_df = cc_df[
            (cc_df['station'] == "#") | (abs(cc_df['dt']) <= t_diff_max)]

    Logger.info('Adding columns for each seismic array...')
    # Add columns for each seismic array
    for seisarray_prefix in SEISARRAY_PREFIXES:
        cc_df[seisarray_prefix] = cc_df.swifter.progress_bar(False).apply(
            lambda row: fnmatch.fnmatch(row['station'], seisarray_prefix,
                                        flags=fnmatch.EXTMATCH),
            axis=1)

    Logger.info('Splitting dt.cc file into event pairs...')
    event_pairs = cc_df[cc_df['station'] == "#"]
    cc_vals = cc_df[cc_df['station'] != "#"]
    # Group rows with non-consecutive index values into separate dataframes;
    # these are the different event pairs
    list_of_dfs = np.split(cc_vals, np.flatnonzero(
        np.diff(cc_vals.index) != 1) + 1)
    
    # list_of_dfs = [
    #     cc_df.iloc[event_pairs.iloc[n].name + 1 : event_pairs.iloc[n+1].name]
    #     for n in range(len(event_pairs)-1)]


    # Create a dictionary of dictionaries, with the first key being the
    # master event id, the second key being the worker event id, and the values
    # being the dataframe of dt.cc values for that event pair.
    event_pair_dict = defaultdict(dict)
    # Loop through all event pairs in the dt.cc file
    Logger.info('Sorting arrival dataframes into dicts...')
    for event_pair, dt_df in zip(event_pairs.iterrows(), list_of_dfs):
        master_id = int(event_pair[1]['dt'])
        worker_id = int(event_pair[1]['cc'])
        event_pair_dict[master_id][worker_id] = dt_df
    return event_pair_dict, cc_df


def _filter_master_arrivals(master_dict, master_id=None,
                            update_event_pair_dict=False):
    Logger.info('Filtering master event %s for array arrivals', master_id)
    remove_indices = []
    remove_dfs = []
    for worker_id, dt_df in master_dict.items():
        not_best_array_phase_picks_list = []
        # For each array, loop through all phase types
        for seisarray_prefix in SEISARRAY_PREFIXES:
            # array_picks = dt_df[dt_df.swifter.progress_bar(False).apply(
            #     lambda row: fnmatch.fnmatch(
            #         row['station'], seisarray_prefix, flags=fnmatch.EXTMATCH),
            #     axis=1)]
            array_picks = dt_df[dt_df[seisarray_prefix]]
            if len(array_picks) <= 1:
                continue
            # Get all phase types recorded for each array
            array_pick_phases = array_picks['phase'].unique()
            for phase, array_phase_picks in array_picks.groupby(
                    array_picks['phase']):
                if len(array_phase_picks) <= 1:
                    continue
                # Select the highest-CC phase type for each array and
                # remove others. Find the best observation for this phase
                # at this array:
                best_phase_loc = np.argmax(array_phase_picks['cc'])
                best_array_phase_pick = array_phase_picks.iloc[
                    best_phase_loc]
                best_array_phase_pick_name = best_array_phase_pick.name
                # Find the other observations for this phase at this array:
                not_best_array_phase_picks = array_phase_picks[
                    array_phase_picks.index != best_array_phase_pick_name]
                not_best_array_phase_picks_list.append(
                    not_best_array_phase_picks)
        # Remove the array picks that are not the best observation for this
        # phase:
        if len(not_best_array_phase_picks_list) != 0:
            for phases in not_best_array_phase_picks_list:
                for idx in list(phases.index.values):
                    remove_indices.append(idx)
            if update_event_pair_dict:
                remove_df = pd.concat(not_best_array_phase_picks_list)
                dt_df.drop(remove_df.index, inplace=True)
                remove_dfs.append(remove_df)
    return remove_indices


def filter_correlation_file_for_array_arrivals(
        event_pair_dict, cc_df, update_event_pair_dict=False,
        parallel=False, cores=None):
    """
    """
    # for event_pair, dt_df in zip(event_pairs.iterrows(), list_of_df):
    #     master_id = int(event_pair[1]['dt'])
    #     worker_id = int(event_pair[1]['cc'])
    #     event_pair_dict[master_id][worker_id] = dt_df
    # filetered_event_pair_dict = defaultdict(dict)
    remove_dfs = []
    remove_indices = []

    if not parallel:
        for master_id, master_dict in event_pair_dict.items():
            remove_indices = []
            remove_indices += _filter_master_arrivals(
                master_dict, master_id=master_id,
                update_event_pair_dict=update_event_pair_dict)
    else:
        # remove_index_lists = 
        with ThreadPoolExecutor(max_workers=40) as executor:
            # Because numpy releases GIL threading can use multiple cores
            results = executor.map(_filter_master_arrivals,
                                    event_pair_dict.values())
            for res in results:
                remove_indices += res
        
        # Logger.info('Filtering master event %s for array arrivals', master_id)
        # for worker_id, dt_df in master_dict.items():
        #     not_best_array_phase_picks_list = []
        #     # For each array, loop through all phase types
        #     for seisarray_prefix in SEISARRAY_PREFIXES:
        #         # array_picks = dt_df[dt_df.swifter.progress_bar(False).apply(
        #         #     lambda row: fnmatch.fnmatch(
        #         #         row['station'], seisarray_prefix, flags=fnmatch.EXTMATCH),
        #         #     axis=1)]
        #         array_picks = dt_df[dt_df[seisarray_prefix]]
        #         if len(array_picks) <= 1:
        #             continue
        #         # Get all phase types recorded for each array
        #         array_pick_phases = array_picks['phase'].unique()
        #         for phase, array_phase_picks in array_picks.groupby(
        #                 array_picks['phase']):
        #             if len(array_phase_picks) <= 1:
        #                 continue
        #             # Select the highest-CC phase type for each array and
        #             # remove others. Find the best observation for this phase
        #             # at this array:
        #             best_phase_loc = np.argmax(array_phase_picks['cc'])
        #             best_array_phase_pick = array_phase_picks.iloc[
        #                 best_phase_loc]
        #             best_array_phase_pick_name = best_array_phase_pick.name
        #             # Find the other observations for this phase at this array:
        #             not_best_array_phase_picks = array_phase_picks[
        #                 array_phase_picks.index != best_array_phase_pick_name]
        #             not_best_array_phase_picks_list.append(
        #                 not_best_array_phase_picks)
        #     # Remove the array picks that are not the best observation for this
        #     # phase:
        #     if len(not_best_array_phase_picks_list) != 0:
        #         for phases in not_best_array_phase_picks_list:
        #             for idx in list(phases.index.values):
        #                 remove_indices.append(idx)
        #         if update_event_pair_dict:
        #             remove_df = pd.concat(not_best_array_phase_picks_list)
        #             dt_df.drop(remove_df.index, inplace=True)
        #             remove_dfs.append(remove_df)
    # Remove all event pairs that have no picks left after filtering:
    Logger.info('Removing all array arrivals from full dataframe...')
    # big_remove_df = pd.concat(remove_dfs)
    # cc_df.drop(big_remove_df.index, inplace=True)
    # cc_df = cc_df[cc_df.index]

    # sbad_df = cc_df.index.isin(remove_indices)
    # cc_df = cc_df[~bad_df]
    mask = np.ones(len(cc_df), dtype=bool)
    mask[remove_indices] = False
    cc_df = cc_df[mask]
    # cc_df[~cc_df.index.isin(remove_indices)]
    return event_pair_dict, cc_df



# Loop over runs
def filter_dt_file_for_arrays(folder, SEISARRAY_PREFIXES, t_diff_max=None,
                              parallel=False, cores=None):
    """
    """
    # evlist_file = glob.glob(folder + '/evlist.txt')[0]
    # evlist_df = read_evlist_file(evlist_file)
    # event_id_mapper = {ev.event_id: ev.event_id
    #                    for ev in evlist_df.itertuples()}
    dt_file = glob.glob(os.path.join(folder, 'dt.cc'))[0]
    # existing_pairs = _read_correlation_file(existing_corr_file=dt_file,
    #                                         event_id_mapper=event_id_mapper)
    Logger.info('Reading correlation file: ' + dt_file)
    event_pair_dict, cc_df = _read_correlation_file_quick(
        dt_file, SEISARRAY_PREFIXES, t_diff_max=t_diff_max)

    # Need to reset index now:
    cc_df.reset_index(drop=True, inplace=True)

    Logger.info('Filtering correlation file for array arrivals')
    # Select the highest-CC phase type for each array and remove the others
    event_pair_dict,  cc_df = filter_correlation_file_for_array_arrivals(
        event_pair_dict, cc_df, parallel=parallel, cores=cores)

    out_file = os.path.join(folder, 'dt_filt.cc')
    Logger.info('Formatting output lines for file: ' + out_file)
    # Write the new dt.cc file
    header_df = cc_df[cc_df['station'] == "#"]
    header_str_df = (
        header_df.station +
        header_df.dt.apply(lambda x: '{:10.0f}'.format(x)) +
        header_df.cc.apply(lambda x: '{:10.0f} '.format(x)) +
        header_df.phase)
        # header_df.dt.apply(lambda x: str(int(x)))

    nonheader_df = cc_df[cc_df['station'] != "#"]
    nonheader_str_df = (
        nonheader_df.station.apply(lambda x: '{:6s}'.format(x)) +
        nonheader_df.dt.apply(lambda x: '{:9.3f}'.format(x)) +
        nonheader_df.cc.apply(lambda x: '{:7.4f} '.format(x)) +
        nonheader_df.phase)

    # Combine headers and nonheader lines, sort by index into right order
    cc_df['print_str'] = pd.concat([header_str_df, nonheader_str_df]).sort_index()

    Logger.info('Writing correlation file: ' + out_file)
    # Write out the new dt.cc file (only the string-printed column of df)
    np.savetxt(out_file, cc_df['print_str'].values, fmt='%s')


# %%

if __name__ == '__main__':

    folders = ['HypoDD_files/02_JanMayen',
            'HypoDD_files/03_MohnKnip', 'HypoDD_files/05_LenaGakkel',
            'HypoDD_files/06_SSvalbard', 'HypoDD_files/07_NSvalbard',
            'HypoDD_files/08_WSvalbard', 'HypoDD_files/09_Knipovich',
            'HypoDD_files/10_Molloy', 'HypoDD_files/11_LenaTrough']
    # folders = ['HypoDD_files/02_JanMayen']
    # folders = ['HypoDD_files/03_MohnKnip']
    # folders = ['HypoDD_files/05_LenaGakkel']
    folders = ['HypoDD_files/06_SSvalbard']
    # folders = ['HypoDD_files/07_NSvalbard']
    # folders = ['HypoDD_files/08_WSvalbard']
    # folders = ['HypoDD_files/09_Knipovich']
    # folders = ['HypoDD_files/10_Molloy']
    # folders = ['HypoDD_files/11_LenaTrough']

    folders = ['HypoDD_files/MohnRidgeTest_202204_wINTEU']

    t_diff_max = 15

    for folder in folders:
        filter_dt_file_for_arrays(folder, SEISARRAY_PREFIXES, t_diff_max=15)