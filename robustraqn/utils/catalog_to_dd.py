
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


def _add_seisarray_columns(cc_df, seisarray_prefix):
    """
    Internal function to add bool-columns to a dataframe that state whether
    observation is  part of a specific seismic array. For adding one column
    for each seismic array.

    Parameters:
    -----------
    :type cc_df: pandas.DataFrame
    :param cc_df: Dataframe with columns 'station' and 'dt' from a dt.cc file
    :type seisarray_prefix: str
    :param seisarray_prefix: String with prefix of seismic array, e.g. 'AR'

    :rtype: pandas.Series
    :return: Series with bool values stating whether observation is part of
    """
    cc_df[seisarray_prefix] = np.zeros(len(cc_df), dtype=bool)
    # cc_df[seisarray_prefix] = cc_df.swifter.progress_bar(False).apply(
    #     lambda row: fnmatch.fnmatch(row['station'], seisarray_prefix,
    #                                 flags=fnmatch.EXTMATCH), axis=1)
    uniq_array_stations = set(fnmatch.filter(
        cc_df.station, seisarray_prefix, flags=fnmatch.EXTMATCH))
    for array_station in uniq_array_stations:
        cc_df[seisarray_prefix] = cc_df[seisarray_prefix] | (
            cc_df['station'] == array_station)
    return cc_df[seisarray_prefix]


def _read_correlation_file_quick(
        existing_corr_file, SEISARRAY_PREFIXES, t_diff_max=None,
        return_event_pair_dicts=False, return_df_gen=True,
        excluded_event_ids=[], max_n_dt=100,
        float_dtype=np.float32, parallel=False, cores=None):
    """
    Function to quickly read in a dt.cc file and return either:
    - a dictionary of dictionaries of dataframes with the dt.cc values for each
      event pair.
    - a generator of dataframes with the dt.cc values for each event pair.

    Parameters:
    -----------
    :type existing_corr_file: str
    :param existing_corr_file: Path to dt.cc file
    :type SEISARRAY_PREFIXES: list
    :param SEISARRAY_PREFIXES:
        List of extended glob-strings that describe all stations belonging to
        a seismic array, e.g.: [
            '@(ARCES|AR[ABCDE][0-9])', '@(SPITS|SP[ABC][0-5])']
    :type t_diff_max: float
    :param t_diff_max:
        Maximum dt-cc value to keep in dataframe (Values larger than this will
        be removed, default is None)
    :type return_event_pair_dicts: bool
    :param return_event_pair_dicts:
        Whether to return a dictionary of dictionaries of dataframes with the
        dt.cc values for each event pair (default is False, then a generator
        for dataframes is returned)
    :type return_df_gen: bool
    :param return_df_gen:
        Whether to return a generator of dataframes with the dt.cc values for
        each event pair (default is True)
    :type float_dtype: numpy.dtype
    :param float_dtype:
        dtype to use for dt and cc columns (default is np.float32 which should
        give enough precision for dt measurements)
    :type parallel: bool
    :param parallel: Whether to parallelize adding columns for each array
    :type cores: int
    :param cores: Number of cores to use for parallelization

    :rtype: dict
    :return:
        Dictionary of dictionaries of dataframes with the dt.cc values (when
        return_event_pair_dicts is True)
    :rtype: generator
    :return:
        Generator of dataframes with the dt.cc values (when return_df_gen is
        True)
    """
    # if size of dt.cc file is larger than 10 GB or so (but what exactly?) use
    # python strings instead of pyarrow strings to avoid pyarrow overflows:
    string_dtype = "string[pyarrow]"
    if os.path.getsize(existing_corr_file) > 1.3 * 10e9:
        string_dtype = "string"
        Logger.info('Reverting to python strings because dt.cc file is very '
                    'large')
    else:
        string_dtype = "string[pyarrow]"
    # Specify dtypes to save some memory on dt- and cc-columns,
    # np.float32 should be enough precision for dt-measurements and
    # pyarrow-string saves 90 % compared to object dtype
    cc_df = pd.read_csv(
        existing_corr_file, delim_whitespace=True, header=None,
        names=['station', 'dt', 'cc', 'phase'], dtype={
            'station': string_dtype, 'dt': float_dtype,
            'cc': float_dtype, 'phase': string_dtype})
    # if the dataframe has more than 2147483647 rows, then convert pyarrow-
    # strings to python-strings to avoid pyarrow overflows:
    # Doesnt work
    # if len(cc_df) > 2147483647:
    #     Logger.info('Casting pyarrow strings to python strings because '
    #                 'dataframe has more than 2147483647 rows')
    #     cc_df['station'] = cc_df['station'].apply(lambda x: str(x))
    #     cc_df['phase'] = cc_df['phase'].apply(lambda x: str(x))
    if t_diff_max is not None:
        # Drop rows where station is not "#" and where abs(dt) > t_diff_max
        # For large dt.cc files (e.g., 1169809308 rows) there could be a 
        # pyarrow.lib.ArrowInvalid: offset overflow while concatenating arrays
        # due to bool[pyarrow], so better convert to python bool
        bool_array = ((cc_df['station'] == "#") |
                      (abs(cc_df['dt']) <= t_diff_max)).values.astype(bool)
        cc_df = cc_df[bool_array]

    # Remove consecutive duplicate rows to speed up later comparisons - such
    # rows are definitely not helpful to Growclust
    consec_dup_df = cc_df[np.all(cc_df.shift() == cc_df, axis=1)]
    cc_df.drop(consec_dup_df.index, inplace=True)

    # Need to exclude event pairs where one event is part of excluded_event_ids
    if len(excluded_event_ids) > 0:
        Logger.info('Removing event pairs for %s excluded events...',
                    len(excluded_event_ids))
        remove_indices = []
        # Find all rows with '#' in station column and check if they contain
        # excluded event ids
        pair_df = cc_df[cc_df.station == '#']
        # excluded_pair_df = pair_df[
        #     (np.sum([
        #         pair_df.dt == excluded_event_id
        #         for excluded_event_id in excluded_event_ids], axis=0) > 0) |
        #     (np.sum([
        #         pair_df.cc == excluded_event_id
        #         for excluded_event_id in excluded_event_ids], axis=0) > 0)]
        # Memory-saving numpy formulation:
        rsum1 = np.zeros(len(pair_df), dtype=np.int32)
        rsum2 = np.zeros(len(pair_df), dtype=np.int32)
        for excluded_event_id in excluded_event_ids:
            # Add 1 to rsum1 and rsum2 for each row where dt or cc is equal to
            # the event id, for each event id in excluded_event_ids.
            rsum1 += (pair_df['dt'] == excluded_event_id)
            rsum2 += (pair_df['cc'] == excluded_event_id)
        excluded_pair_df = pair_df[(rsum1 > 0) | (rsum2 > 0)]
        remove_indices.extend(excluded_pair_df.index)
        prev_event_id = 0
        for jr, row in excluded_pair_df.iterrows():
            if row['dt'] != prev_event_id:
                Logger.info('Removing event pairs for event id %s', row['dt'])
                prev_event_id = row['dt']
            # Limit the number of rows before starting the iteration, otherwise
            # this can take a long time...
            for kr, dt_row in cc_df.loc[row.name+1 : row.name+max_n_dt
                                        ].iterrows():
                if dt_row.station == '#':
                    break
                remove_indices.append(dt_row.name)

        # SLower version with iterrows across all rows:
        # remove_pair = False
        # for jr, row in cc_df.iterrows():
        #     if '#' in row['station']:
        #         if (row['cc'] in excluded_event_ids or
        #                 row['dt'] in excluded_event_ids):
        #             remove_pair = True
        #             remove_indices.append(jr)
        #         else:
        #             remove_pair = False
        #     else:
        #         if remove_pair:
        #             remove_indices.append(jr)
        cc_df.drop(remove_indices, inplace=True)
        Logger.info('Removed %s lines from cc-dataframe for excluded events',
                    len(remove_indices))

    # Need to reset index after removing rows with abs(dt) > t_diff_max
    cc_df.reset_index(drop=True, inplace=True)
    Logger.info('Adding columns for each seismic array...')
    # Add columns for each seismic array
    if not parallel:
        seisarray_columns = []
        for seisarray_prefix in SEISARRAY_PREFIXES:
            seisarray_columns.append(_add_seisarray_columns(
                cc_df, seisarray_prefix=seisarray_prefix))
        # Concatenate seisaray_column to the right of cc_df:
        # cc_df = pd.concat([cc_df, *seisarray_columns], axis=1)
    else:
        # Parallelize adding columns for each seismic array
        station_df = pd.DataFrame(cc_df['station'])
        arr_jobs = min(cores, len(SEISARRAY_PREFIXES))
        results = Parallel(n_jobs=arr_jobs)(delayed(_add_seisarray_columns)(
            station_df, seisarray_prefix)
                               for seisarray_prefix in SEISARRAY_PREFIXES)
        cc_df = pd.concat([cc_df, *results], axis=1)

        # cc_df[seisarray_prefix] = np.zeros(len(cc_df), dtype=bool)
        # # cc_df[seisarray_prefix] = cc_df.swifter.progress_bar(False).apply(
        # #     lambda row: fnmatch.fnmatch(row['station'], seisarray_prefix,
        # #                                 flags=fnmatch.EXTMATCH), axis=1)
        # uniq_array_stations = set(fnmatch.filter(
        #     cc_df.station, seisarray_prefix, flags=fnmatch.EXTMATCH))
        # for array_station in uniq_array_stations:
        #     cc_df[seisarray_prefix] = cc_df[seisarray_prefix] | (
        #         cc_df['station'] == array_station)

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
    # Drop dt- column because it's not needed for the workers - but this
    # actually makes it slower
    # Speed timings:
    # with list: 43 s for 16 MB dt-cc file
    # with list, drop dt: 102 s for 16 MB dt-cc file
    # with generator: 52 s for 16 MB dt-cc file, but much less memory usage
    dt_df_generator = (
        cc_df.loc[start_index[n] : stop_index[n]]  # .drop(columns='dt')
        for n in range(len(event_pairs)))
    # Save number of event pairs for return
    n_jobs = len(start_index)

    # Create a dictionary of dictionaries, with the first key being the
    # master event id, the second key being the worker event id, and the values
    # being the dataframe of dt.cc values for that event pair.
    if return_event_pair_dicts:
        event_pair_dict = defaultdict(dict)
        # Loop through all event pairs in the dt.cc file
        Logger.info('Sorting arrival dataframes into dicts...')
        for event_pair, dt_df in zip(event_pairs.iterrows(), dt_df_generator):
            master_id = int(event_pair[1]['dt'])
            worker_id = int(event_pair[1]['cc'])
            event_pair_dict[master_id][worker_id] = dt_df
        return event_pair_dict, cc_df, n_jobs
    elif return_df_gen:
        return dt_df_generator, cc_df, n_jobs

# @profile
def _filter_master_arrivals(
        master_dict, master_id=None, update_event_pair_dict=False,
        return_event_pair_dicts=False, return_df_gen=True, dt_df=None,
        filter_all_stations=False, n_jobs=None):
    """
    Function to filter the master event arrivals for a given master event id.

    Parameters
    ----------
    :type master_dict: dict
    :param master_dict: Dictionary of dictionaries, with the first key being
    :type master_id: int
    :param master_id: Master event id to filter arrivals for
    :type update_event_pair_dict: bool
    :param update_event_pair_dict: Whether to update the event pair dictionary
    :type return_event_pair_dicts: bool
    :param return_event_pair_dicts: Whether to return the event pair dictionary
    :type return_df_gen: bool
    :param return_df_gen:
        Whether to return the dataframe generator (instead of the dictionary)
    :type dt_df: pandas.DataFrame
    :param dt_df: Dataframe of dt.cc values for the master event
    :type filter_all_stations: bool
    :param filter_all_stations:
    :type n_jobs: int
    :param n_jobs: Number of jobs to split the dataframe into

    :rtype: list
    :return:
        List of indices for suboptimal observations to be removed from the
        dataframe.
    """
    if return_event_pair_dicts:
        Logger.info('Filtering master event %s for array arrivals', master_id)
    elif return_df_gen:
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
    elif return_df_gen:
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
            # pandas sum is quicker here than np.sum:
            if dt_df[seisarray_prefix].sum() <= 1:
                continue
            # This selection is quite slow:
            array_picks = dt_df[dt_df[seisarray_prefix]]
            # Get all phase types recorded for each array
            # array_pick_phases = array_picks['phase'].unique()

            # uniq_array_phases = set(array_picks['phase'].values)
            # for phase in uniq_array_phases:
            #     array_phase_picks = array_picks[array_picks['phase'] == phase]
            if len(array_picks) <= 1:
                continue

            # With this, total time: 20 s
            # for phase, array_phase_picks in array_picks.groupby(
            #         array_picks['phase'], sort=False):
            #     if len(array_phase_picks) <= 1:
            #         continue
            #     # Select the highest-CC phase type for each array and
            #     # remove others. Find the best observation for this phase
            #     # at this array:
            #     # best_phase_loc = np.argmax(array_phase_picks['cc'])
            #     # Get the index of the best phase observation:
            #     best_array_phase_pick_name = array_phase_picks['cc'].idxmax()
            #     # Find the other observations for this phase at this array:
            #     not_best_array_phase_pick_index = array_phase_picks.index[
            #         array_phase_picks.index != best_array_phase_pick_name
            #         ].values
            #     not_best_phase_pick_indices.append(
            #         not_best_array_phase_pick_index)

            # With this, total time: 16 s
            # Do grouping manually with defaultdict:
            groups = defaultdict(lambda: [])
            for phase, index, cc in zip(
                    array_picks.phase, array_picks.index, array_picks.cc):
                groups[phase].append((index, cc))
            for phase, index_cc_tuple in groups.items():
                best_phase_pick_local_index = np.argmax(
                    [cc for index, cc in index_cc_tuple])
                best_phase_pick_index = index_cc_tuple[
                    best_phase_pick_local_index][0]
                not_best_phase_pick_index = [
                    index for index, cc in index_cc_tuple
                    if index != best_phase_pick_index]
                not_best_phase_pick_indices.append(not_best_phase_pick_index)
                
                
        # Now filter individual stations (not part of arrays) for multiple
        # measurements of same arrival at the same station:
        if filter_all_stations:
            # This formulation with .any is quite slow:
            # non_array_picks = dt_df[~dt_df[SEISARRAY_PREFIXES].any(axis=1)]
            # Quicker? - select rows where all seisarray-
            non_array_picks = dt_df[np.sum([
                dt_df[seisarray_prefix]
                for seisarray_prefix in SEISARRAY_PREFIXES], axis=0) == 0]
            # pandas sum is slower here:
            # non_array_picks = dt_df[
            #     dt_df[SEISARRAY_PREFIXES].sum(axis=1) == 0]
            if len(non_array_picks) > 1:
                # Total 46 seconds for this part:
                # for phase, phase_picks in non_array_picks.groupby(
                #         [non_array_picks['station'], non_array_picks['phase']],
                #         sort=False):
                #     # Select highest-CC pick for each phase at each station, remove
                #     # rest.
                #     if len(phase_picks) <= 1:
                #         continue
                #     # Get the index of the best phase observation:
                #     best_phase_pick_name = phase_picks['cc'].idxmax()
                #     # Find the other observations for this phase at this array:
                #     not_best_phase_pick_index = phase_picks.index[
                #         phase_picks.index != best_phase_pick_name].values
                #     not_best_phase_pick_indices.append(not_best_phase_pick_index)

                # Total 20 seconds for this part:
                # Do grouping manually with defaultdict:
                groups = defaultdict(lambda: [])
                # station_phases = (non_array_picks['station'] + '_' +
                #                   non_array_picks['phase']).values
                # Much quicker to concatenate strings with list comprehension:
                station_phases = [
                    "{}_{}".format(sta, ph) for sta, ph in zip(
                        non_array_picks.station.values,
                        non_array_picks.phase.values)]
                for station_phase, index, cc in zip(
                        station_phases, non_array_picks.index,
                        non_array_picks.cc):
                    groups[station_phase].append((index, cc))
                for station_phase, index_cc_tuple in groups.items():
                    best_phase_pick_local_index = np.argmax(
                        [cc for index, cc in index_cc_tuple])
                    best_phase_pick_index = index_cc_tuple[
                        best_phase_pick_local_index][0]
                    not_best_phase_pick_index = [
                        index for index, cc in index_cc_tuple
                        if index != best_phase_pick_index]
                    not_best_phase_pick_indices.append(
                        not_best_phase_pick_index)

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


def filter_correlation_file_arrivals(
        event_pair_dict, cc_df, update_event_pair_dict=False,
        return_event_pair_dicts=False, return_df_gen=True,
        dt_df_generator=[], n_jobs=0, filter_all_stations=False,
        parallel=False, cores=None):
    """
    Function to filter a correlation file for array arrivals. This
    is done by finding the observations that are not the best observation for
    each phase type at each array / station, and removing them from the
    dataframe in place.

    :type event_pair_dict: dict
    :param event_pair_dict:
        Dictionary of event pairs, with master event ID as key and a dictionary
        of worker event IDs as value. Each worker event ID is a key to a
        dataframe of correlation coefficients for that event pair.
    :type cc_df: pd.DataFrame
    :param cc_df:
        The correlation dataframe to filter (basically the full dt-cc file).
    :type update_event_pair_dict: bool
    :param update_event_pair_dict:
        Whether to update the event pair dict with the filtered dataframes.
    :type return_event_pair_dicts: bool
    :param return_event_pair_dicts:
        Whether to return the event pair dict. (Quicker to not return it and
        only use the df-generator).
    :type return_df_gen: bool
    :param return_df_gen: Whether to return a generator of dataframes.
    :type dt_df_generator: generator
    :param dt_df_generator: Generator for event-pair dataframes.
    :type n_jobs: int
    :param n_jobs: Total number of jobs to be run.
    :type filter_all_stations: bool
    :param filter_all_stations:
        Whether to filter all stations for multiple observations of the same
        phase type, and not just array stations.
    :type parallel: bool
    :param parallel: Whether to run in parallel.
    :type cores: int
    :param cores: Number of cores to use.

    :rtype: generator
    :return: Generator of dataframes.
    """
    remove_indices = []
    if return_event_pair_dicts:
        if not parallel:
            for master_id, master_dict in event_pair_dict.items():
                remove_indices += _filter_master_arrivals(
                    master_dict, master_id=master_id,
                    return_event_pair_dicts=return_event_pair_dicts,
                    return_df_gen=False,
                    filter_all_stations=filter_all_stations,
                    update_event_pair_dict=update_event_pair_dict)
        else:
            # joblib is much quicker than threadpoolexecutor here..
            filt_jobs = min(len(event_pair_dict), cores)
            results = Parallel(n_jobs=filt_jobs)(
                delayed(_filter_master_arrivals)(
                    event_pair_dict[master_id], master_id=master_id,
                    return_event_pair_dicts=return_event_pair_dicts,
                    return_df_gen=False,
                    filter_all_stations=filter_all_stations)
                for master_id in event_pair_dict.keys())
            for res in results:
                remove_indices += res
    elif return_df_gen:
        if not parallel:
            for df_j, dt_df in enumerate(dt_df_generator):
                remove_indices += _filter_master_arrivals(
                    None, master_id=df_j, n_jobs=n_jobs,
                    return_event_pair_dicts=False,
                    return_df_gen=return_df_gen, dt_df=dt_df,
                    filter_all_stations=filter_all_stations)
        else:
            # joblib is much quicker than threadpoolexecutor here..
            filt_jobs = min(n_jobs, cores)
            results = Parallel(n_jobs=filt_jobs)(delayed(
                _filter_master_arrivals)(
                    None, master_id=df_j, n_jobs=n_jobs,
                    return_event_pair_dicts=False,
                    return_df_gen=return_df_gen, dt_df=dt_df,
                    filter_all_stations=filter_all_stations,)
                for df_j, dt_df in enumerate(dt_df_generator))
            for res in results:
                remove_indices += res

    # Remove all event pairs that have no picks left after filtering:
    Logger.info('Removing all array arrivals from full dataframe...')
    mask = np.ones(len(cc_df), dtype=bool)
    mask[remove_indices] = False
    cc_df = cc_df[mask]
    if return_event_pair_dicts:
        return event_pair_dict, cc_df
    elif return_df_gen:
        return None, cc_df


def filter_dt_file_for_arrays(folder, SEISARRAY_PREFIXES, t_diff_max=None,
                              return_event_pair_dicts=False,
                              filter_all_stations=False,
                              return_df_gen=True, backup_parquet_file=False,
                              excluded_event_ids=[],
                              parallel=False, cores=None):
    """
    Top level function to filter a correlation file for array arrivals. This
    function reads the correlation file, filters it, and writes the filtered
    dt-cc file to disk.

    :type folder: str
    :param folder: Path to folder containing dt.cc file.
    :type SEISARRAY_PREFIXES: list
    :param SEISARRAY_PREFIXES:
        List of extended glob-strings that describe all stations belonging to
        a seismic array, e.g.: [
            '@(ARCES|AR[ABCDE][0-9])', '@(SPITS|SP[ABC][0-5])']
    :type t_diff_max: float
    :param t_diff_max:
        Maximum dt-cc value to keep in dataframe (Values larger than this will
        be removed, default is None)
    :type return_event_pair_dicts: bool
    :param return_event_pair_dicts:
        Whether to return the event pair dict. (Quicker to not return it and
        just use the df-generator).
    :type filter_all_stations: bool
    :param filter_all_stations:
        Whether to filter all stations and not just array stations.
    :type return_df_gen: bool
    :param return_df_gen:
        Whether to return a generator of dataframes (default is True, this is
        the quickest option).
    :type parallel: bool
    :param parallel: Whether to run in parallel.
    :type cores: int
    :param cores: Number of cores to use.

    :rtype: tuple
    :return: Tuple of event pair dict and filtered dataframe.
    """
    dt_file = glob.glob(os.path.join(folder, 'dt.cc'))[0]
    # One can also supply a dt_uniq.cc file where consecutive duplicates have
    # already been removed with "uniq dt.cc dt_uniq.cc" (Linux)
    dt_uniq_file = glob.glob(os.path.join(folder, 'dt_uniq.cc'))
    # Check if dt_uniq_file is newere than dt_file:
    if len(dt_uniq_file) > 0:
        dt_uniq_file = dt_uniq_file[0]
        if os.path.getmtime(dt_uniq_file) > os.path.getmtime(dt_file):
            dt_file = dt_uniq_file
            Logger.info('Using dt_uniq.cc file instead of dt.cc file')
    dt_prefilt_file = sorted(glob.glob(os.path.join(folder, 'dt_CCmin*.cc')))
    if len(dt_prefilt_file) > 0:
        dt_prefilt_file = dt_prefilt_file[0]
        if os.path.getmtime(dt_prefilt_file) > os.path.getmtime(dt_file):
            dt_file = dt_prefilt_file
            Logger.info('Using %s file instead of dt.cc file', dt_prefilt_file)
    Logger.info('Reading correlation file: ' + dt_file)
    if return_event_pair_dicts:
        event_pair_dict, cc_df, n_jobs = _read_correlation_file_quick(
            dt_file, SEISARRAY_PREFIXES, t_diff_max=t_diff_max,
            return_event_pair_dicts=return_event_pair_dicts,
            excluded_event_ids=excluded_event_ids,
            return_df_gen=False, parallel=parallel, cores=cores)
        Logger.info('Filtering correlation file for array arrivals')
        # Select the highest-CC phase type for each array and remove the others
        event_pair_dict, cc_df, n_jobs = filter_correlation_file_arrivals(
            event_pair_dict, cc_df,
            return_event_pair_dicts=return_event_pair_dicts,
            return_df_gen=False, filter_all_stations=filter_all_stations,
            parallel=parallel, cores=cores)
    elif return_df_gen:
        dt_df_generator, cc_df, n_jobs = _read_correlation_file_quick(
            dt_file, SEISARRAY_PREFIXES, t_diff_max=t_diff_max,
            return_event_pair_dicts=False,
            return_df_gen=return_df_gen,
            excluded_event_ids=excluded_event_ids,
            parallel=parallel, cores=cores)
        Logger.info('Filtering correlation file for array arrivals')
        # Select the highest-CC phase type for each array and remove the others
        _, cc_df = filter_correlation_file_arrivals(
            None, cc_df, return_event_pair_dicts=False,
            return_df_gen=True, dt_df_generator=dt_df_generator,
            filter_all_stations=filter_all_stations, n_jobs=n_jobs,
            parallel=parallel, cores=cores)

    # Need to reset index now (no?!)
    # cc_df.reset_index(drop=True, inplace=True)
    if backup_parquet_file:
        cc_df.to_parquet(os.path.join(folder, 'dt_filt.cc.parquet'))

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
    # Pyarrow 10.0.1 struggles with str-array > 2GB, so we better convert to
    # python object-type string first
    if nonheader_df.memory_usage().sum() > 2147483646:
        header_df['station'] = header_df['station'].astype('str')
        header_df['phase'] = header_df['phase'].astype('str')
        header_str_df = header_str_df.astype(str)
        nonheader_df['station'] = nonheader_df['station'].astype('str')
        nonheader_df['phase'] = nonheader_df['phase'].astype('str')

    nonheader_str_df = (
        nonheader_df.station.apply(lambda x: '{:6s}'.format(x)) +
        nonheader_df.dt.apply(lambda x: '{:9.3f}'.format(x)) +
        nonheader_df.cc.apply(lambda x: '{:7.4f} '.format(x)) +
        nonheader_df.phase)
    
    # with open(out_file,'w') as file:
    #   df.to_string(file, columns=use_cols)

    # Combine headers and nonheader lines, sort by index into right order
    cc_df['print_str'] = pd.concat([header_str_df, nonheader_str_df]
                                   ).sort_index()

    Logger.info('Writing correlation file: ' + out_file)
    # Write out the new dt.cc file (only the string-printed column of df)
    np.savetxt(out_file, cc_df['print_str'].values, fmt='%s')



# %% TEST FUNCTION

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
            return_event_pair_dicts=False, return_df_gen=True,
            filter_all_stations=False,
            parallel=True, cores=50)