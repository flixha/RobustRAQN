
import os
import pandas as pd
import numpy as np
import glob
from collections import defaultdict
from wcmatch import fnmatch
# import swifter
from joblib import Parallel, delayed

# from concurrent.futures import ThreadPoolExecutor

# from robustraqn.core.seismic_array import SEISARRAY_PREFIXES
# from robustraqn.utils.growclust import read_evlist_file

import logging

# Need to set up logging so that loky backend does proper logging of progress
log_format = "%(asctime)s\t%(name)40s:%(lineno)s\t%(funcName)20s()\t%(levelname)s\t%(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format)

Logger = logging.getLogger(__name__)

SEISARRAY_PREFIXES = [
     '@(ARCES|AR[ABCDE][0-9])', '@(SPITS|SP[ABC][0-5])',
    '@(BEAR|BJO|BJO1|BEA[1-6])',
]



def _read_correlation_file_quick(
        existing_corr_file, SEISARRAY_PREFIXES, t_diff_max=None,
        return_event_pair_dicts=False, return_list_of_dfs=True,
        float_dtype=np.float32):
    """
    Quickly read in a dt.cc file and return a dictionary of dictionaries of
    dataframes with the dt.cc values for each event pair.
    """
    # Specify dtypes to save some memory on dt- and cc-columns,
    # np.float32 should be enough precision for dt-measurements and
    # pyarrow-string saves 90 % compared to object dtype
    cc_df = pd.read_csv(
        existing_corr_file, delim_whitespace=True, header=None,
        names=['station', 'dt', 'cc', 'phase'], dtype={
            'station': "string[pyarrow]", 'dt': float_dtype,
            'cc': float_dtype, 'phase': "string[pyarrow]"})
    if t_diff_max is not None:
        # Drop rows where station is not "#" and where abs(dt) > t_diff_max
        cc_df = cc_df[
            (cc_df['station'] == "#") | (abs(cc_df['dt']) <= t_diff_max)]

    # Need to reset index after removing rows with abs(dt) > t_diff_max
    cc_df.reset_index(drop=True, inplace=True)
    Logger.info('Adding columns for each seismic array...')
    # Add columns for each seismic array
    for seisarray_prefix in SEISARRAY_PREFIXES:
        cc_df[seisarray_prefix] = np.zeros(len(cc_df), dtype=bool)
        # cc_df[seisarray_prefix] = cc_df.swifter.progress_bar(False).apply(
        #     lambda row: fnmatch.fnmatch(row['station'], seisarray_prefix,
        #                                 flags=fnmatch.EXTMATCH), axis=1)
        uniq_array_stations = set(fnmatch.filter(
            cc_df.station, seisarray_prefix, flags=fnmatch.EXTMATCH))
        for array_station in uniq_array_stations:
            cc_df[seisarray_prefix] = cc_df[seisarray_prefix] | (
                cc_df['station'] == array_station)

    Logger.info('Splitting dt.cc file into event pairs...')
    event_pairs = cc_df[cc_df['station'] == "#"]
    cc_vals = cc_df[cc_df['station'] != "#"]
    # Group rows with non-consecutive index values into separate dataframes;
    # these are the different event pairs

    # option 1: split at non-consecutive index values
    # list_of_dfs = np.split(cc_vals, np.flatnonzero(
    #     np.diff(cc_vals.index) != 1) + 1)

    # option 2: select by index of pairs
    # list_of_dfs = [
    #     cc_df.loc[event_pairs.iloc[n].name + 1 : event_pairs.iloc[n+1].name]
    #     for n in range(len(event_pairs)-1)]

    # option 3: use numpy arrays for indexing
    start_index = event_pairs.index.values + 1
    # Initialize stop_index array
    stop_index = np.zeros(len(event_pairs), dtype=int)
    stop_index[:-1] = event_pairs.index.values[1:] - 1
    # Set last index to last row of cc_df
    stop_index[-1] = cc_df.index.values[-1]
    list_of_dfs = [cc_df.loc[start_index[n] : stop_index[n]]
                   for n in range(len(event_pairs))]

    # Create a dictionary of dictionaries, with the first key being the
    # master event id, the second key being the worker event id, and the values
    # being the dataframe of dt.cc values for that event pair.
    if return_event_pair_dicts:
        event_pair_dict = defaultdict(dict)
        # Loop through all event pairs in the dt.cc file
        Logger.info('Sorting arrival dataframes into dicts...')
        for event_pair, dt_df in zip(event_pairs.iterrows(), list_of_dfs):
            master_id = int(event_pair[1]['dt'])
            worker_id = int(event_pair[1]['cc'])
            event_pair_dict[master_id][worker_id] = dt_df
        return event_pair_dict, cc_df
    elif return_list_of_dfs:
        return list_of_dfs, cc_df


def _filter_master_arrivals(
        master_dict, master_id=None, update_event_pair_dict=False,
        return_event_pair_dicts=False, return_list_of_dfs=True, dt_df=None,
        filter_all_stations=False, n_jobs=None):
    if return_event_pair_dicts:
        Logger.info('Filtering master event %s for array arrivals', master_id)
    elif return_list_of_dfs:
        if n_jobs is not None:
            # Show log information, but only in steps of 1 % (need to compare
            # how far we have gotten, but pay attention to rounding effects)
            precision = len(str(n_jobs)) - 1
            progress = round(master_id / n_jobs, precision)
            if progress in np.arange(0, 1, 0.01):
                Logger.info(
                    'Filtering master-worker event pair for array arrivals'
                    ' done: %s %% ( %s / %s )', round(progress * 100, 1),
                    master_id, n_jobs)
    remove_indices = []
    remove_dfs = []
    # Work through each event pair for one master event
    if return_event_pair_dicts:
        master_dict = master_dict
    # Work through one dataframe at a time
    elif return_list_of_dfs:
        master_dict = {1: dt_df}
    for worker_id, dt_df in master_dict.items():
        # not_best_array_phase_picks_list = []
        not_best_phase_pick_indices = []
        # For each array, loop through all phase types
        for seisarray_prefix in SEISARRAY_PREFIXES:
            # array_picks = dt_df[dt_df.swifter.progress_bar(False).apply(
            #     lambda row: fnmatch.fnmatch(
            #         row['station'], seisarray_prefix, flags=fnmatch.EXTMATCH),
            #     axis=1)]
            if dt_df[seisarray_prefix].sum() <= 1:
                continue
            array_picks = dt_df[dt_df[seisarray_prefix]]
            # Get all phase types recorded for each array
            # array_pick_phases = array_picks['phase'].unique()
            for phase, array_phase_picks in array_picks.groupby(
                    array_picks['phase']):
                if len(array_phase_picks) <= 1:
                    continue
                # Select the highest-CC phase type for each array and
                # remove others. Find the best observation for this phase
                # at this array:
                # best_phase_loc = np.argmax(array_phase_picks['cc'])
                # Get the index of the best phase observation:
                best_array_phase_pick_name = array_phase_picks['cc'].idxmax()
                # Find the other observations for this phase at this array:
                not_best_array_phase_pick_index = array_phase_picks.index[
                    array_phase_picks.index != best_array_phase_pick_name
                    ].values
                not_best_phase_pick_indices.append(
                    not_best_array_phase_pick_index)
        # Now filter individual stations (not part of arrays) for multiple
        # measurements of same arrival at the same station:
        if filter_all_stations:
            non_array_picks = dt_df[~dt_df[SEISARRAY_PREFIXES].any(axis=1)]
            for phase, phase_picks in non_array_picks.groupby(
                    [non_array_picks['station'], non_array_picks['phase']]):
                # Select highest-CC pick for each phase at each station, remove
                # rest.
                if len(phase_picks) <= 1:
                    continue
                # Get the index of the best phase observation:
                best_phase_pick_name = phase_picks['cc'].idxmax()
                # Find the other observations for this phase at this array:
                not_best_phase_pick_index = phase_picks.index[
                    phase_picks.index != best_phase_pick_name].values
                not_best_phase_pick_indices.append(not_best_phase_pick_index)

        # Remove the array and station picks that are not the best observation
        # for this phase:
        if len(not_best_phase_pick_indices) != 0:
            for phase_indices in not_best_phase_pick_indices:
                remove_indices.extend(phase_indices)
            if update_event_pair_dict:
                remove_df = pd.concat(not_best_phase_pick_indices)
                dt_df.drop(remove_df.index, inplace=True)
                remove_dfs.append(remove_df)
    return remove_indices


def filter_correlation_file_for_array_arrivals(
        event_pair_dict, cc_df, update_event_pair_dict=False,
        return_event_pair_dicts=False, return_list_of_dfs=True,
        list_of_dfs=[], filter_all_stations=False, parallel=False, cores=None):
    """
    """
    remove_indices = []
    if return_event_pair_dicts:
        if not parallel:
            for master_id, master_dict in event_pair_dict.items():
                remove_indices += _filter_master_arrivals(
                    master_dict, master_id=master_id,
                    return_event_pair_dicts=return_event_pair_dicts,
                    return_list_of_dfs=False,
                    filter_all_stations=filter_all_stations,
                    update_event_pair_dict=update_event_pair_dict)
        else:
            # joblib is much quicker than threadpoolexecutor here..
            results = Parallel(n_jobs=cores)(delayed(_filter_master_arrivals)(
                event_pair_dict[master_id], master_id=master_id,
                return_event_pair_dicts=return_event_pair_dicts,
                return_list_of_dfs=False,
                filter_all_stations=filter_all_stations)
                for master_id in event_pair_dict.keys())
            for res in results:
                remove_indices += res
    elif return_list_of_dfs:
        if not parallel:
            for dt_df in list_of_dfs:
                remove_indices += _filter_master_arrivals(
                    None, master_id=master_id,
                    return_event_pair_dicts=False,
                    return_list_of_dfs=return_list_of_dfs,
                    filter_all_stations=filter_all_stations,
                    dt_df=dt_df)
        else:
            # joblib is much quicker than threadpoolexecutor here..
            results = Parallel(n_jobs=cores)(delayed(_filter_master_arrivals)(
                None, master_id=df_j, n_jobs=len(list_of_dfs),
                return_event_pair_dicts=False,
                return_list_of_dfs=return_list_of_dfs, dt_df=dt_df,
                filter_all_stations=filter_all_stations,)
                for df_j, dt_df in enumerate(list_of_dfs))
            for res in results:
                remove_indices += res

    # Remove all event pairs that have no picks left after filtering:
    Logger.info('Removing all array arrivals from full dataframe...')
    mask = np.ones(len(cc_df), dtype=bool)
    mask[remove_indices] = False
    cc_df = cc_df[mask]
    if return_event_pair_dicts:
        return event_pair_dict, cc_df
    elif return_list_of_dfs:
        return None, cc_df


def filter_dt_file_for_arrays(folder, SEISARRAY_PREFIXES, t_diff_max=None,
                              parallel=False, cores=None,
                              return_event_pair_dicts=False,
                              filter_all_stations=False,
                              return_list_of_dfs=True):
    """
    """
    dt_file = glob.glob(os.path.join(folder, 'dt.cc'))[0]
    Logger.info('Reading correlation file: ' + dt_file)
    if return_event_pair_dicts:
        event_pair_dict, cc_df = _read_correlation_file_quick(
            dt_file, SEISARRAY_PREFIXES, t_diff_max=t_diff_max,
            return_event_pair_dicts=return_event_pair_dicts,
            return_list_of_dfs=False)
        Logger.info('Filtering correlation file for array arrivals')
        # Select the highest-CC phase type for each array and remove the others
        event_pair_dict, cc_df = filter_correlation_file_for_array_arrivals(
            event_pair_dict, cc_df,
            return_event_pair_dicts=return_event_pair_dicts,
            return_list_of_dfs=False, filter_all_stations=filter_all_stations,
            parallel=parallel, cores=cores)
    elif return_list_of_dfs:
        list_of_dfs, cc_df = _read_correlation_file_quick(
            dt_file, SEISARRAY_PREFIXES, t_diff_max=t_diff_max,
            return_event_pair_dicts=False,
            return_list_of_dfs=return_list_of_dfs)
        Logger.info('Filtering correlation file for array arrivals')
        # Select the highest-CC phase type for each array and remove the others
        _, cc_df = filter_correlation_file_for_array_arrivals(
            None, cc_df, return_event_pair_dicts=False,
            return_list_of_dfs=True, list_of_dfs=list_of_dfs,
            filter_all_stations=filter_all_stations,
            parallel=parallel, cores=cores)

    # Need to reset index now (no?!)
    # cc_df.reset_index(drop=True, inplace=True)

    out_file = os.path.join(folder, 'dt_filt.cc')
    Logger.info('Formatting output lines for file: ' + out_file)
    # Write the new dt.cc file
    header_df = cc_df[cc_df['station'] == "#"]
    header_str_df = (
        header_df.station +
        header_df.dt.apply(lambda x: '{:10.0f}'.format(x)) +
        header_df.cc.apply(lambda x: '{:10.0f} '.format(x)) +
        header_df.phase)

    nonheader_df = cc_df[cc_df['station'] != "#"]
    nonheader_str_df = (
        nonheader_df.station.apply(lambda x: '{:6s}'.format(x)) +
        nonheader_df.dt.apply(lambda x: '{:9.3f}'.format(x)) +
        nonheader_df.cc.apply(lambda x: '{:7.4f} '.format(x)) +
        nonheader_df.phase)

    # Combine headers and nonheader lines, sort by index into right order
    cc_df['print_str'] = pd.concat([header_str_df, nonheader_str_df]
                                   ).sort_index()

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
    # folders = ['HypoDD_files/06_SSvalbard']
    # folders = ['HypoDD_files/07_NSvalbard']
    # folders = ['HypoDD_files/08_WSvalbard']
    # folders = ['HypoDD_files/09_Knipovich']
    # folders = ['HypoDD_files/10_Molloy']
    # folders = ['HypoDD_files/11_LenaTrough']

    #folders = ['HypoDD_files/MohnRidgeTest_202204_wINTEU']

    t_diff_max = 15

    for folder in folders:
        filter_dt_file_for_arrays(
            folder, SEISARRAY_PREFIXES, t_diff_max=t_diff_max,
            return_event_pair_dicts=False, return_list_of_dfs=True,
            filter_all_stations=False,
            parallel=True, cores=50)