
import numpy as np
import matplotlib

import pandas as pd
import time

from obspy.core.event import Catalog, Event
from obspy.geodetics.base import degrees2kilometers, kilometers2degrees
from obspy.core.utcdatetime import UTCDateTime
from obsplus.events.validate import attach_all_resource_ids
from obsplus import events_to_df
from obsplus.utils.time import to_datetime64
from obsplus.constants import EVENT_DTYPES, TIME_COLUMNS
from obsplus.structures.dfextractor import DataFrameExtractor
from eqcorrscan.core.match_filter import Tribe

import pandas as pd

import logging
Logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")


def update_tribe_from_bayesloc(tribe, bayesloc_stats_out_file):
    """
    """
    cat = Catalog([t.event for t in tribe])
    cat = update_cat_from_bayesloc(cat, bayesloc_stats_out_file)
    return tribe


def update_cat_from_bayesloc(cat, bayesloc_stats_out_file):
    """
    Update a catalog's locations from a bayesloc-relocation run
    """
    # '/home/felix/Documents2/BASE/Detection/Bitdalsvatnet/Relocation/BayesLoc/'
    #     + 'Bitdalsvatnet_04_prior_955_N_Sg/output/origins_ned_stats.out',

    cat_backup = cat.copy()
    # for event in cat:
    #     attach_all_resource_ids(event)
    # remove arrivals to avoid error in conversion to dataframe
    for ev in cat:
        ev.preferred_origin().arrivals = []
    cat_df = events_to_df(cat)
    cat_df['events'] = cat.events

    bayes_df = pd.read_csv(bayesloc_stats_out_file, delimiter=' ')
    bayes_df = bayes_df.sort_values(by='time_mean')

    bayes_times = [time.gmtime(value) for value in bayes_df.time_mean.values]
    bayes_utctimes = [
        UTCDateTime(bt.tm_year, bt.tm_mon, bt.tm_mday, bt.tm_hour, bt.tm_min,
                    bt.tm_sec + (et - int(et)))
        for bt, et in zip(bayes_times, bayes_df.time_mean.values)]
    bayes_df['utctime'] = bayes_utctimes
    bayes_df['datetime'] = [butc._get_datetime() for butc in bayes_utctimes]


    # cat_SgLoc = read_seisan_database('Sfiles_MAD10_Saga_02_Sg')
    # cat_Sgloc_df = events_to_df(cat_SgLoc)
    # cat_Sgloc_df['events'] = cat_SgLoc.events
    s_diff = 3
    max_bayes_error_km = 50

    # put back arrivals
    for event, event_backup in zip(cat, cat_backup):
        event.preferred_origin().arrivals = event_backup.preferred_origin(
            ).arrivals

    # Code to sort in the new locations from BAYESLOC / Seisan into catalog
    for event in cat:
        bayes_orig = event.preferred_origin().copy()
        lower_dtime = (bayes_orig.time - s_diff)._get_datetime()
        upper_dtime = (bayes_orig.time + s_diff)._get_datetime()

        tmp_cat_df = bayes_df.loc[(bayes_df.datetime > lower_dtime) & (
            bayes_df.datetime < upper_dtime)]
        if len(tmp_cat_df) > 0:
            if (tmp_cat_df.time_sd.iloc[0] < s_diff * 3 and
                    tmp_cat_df.depth_sd.iloc[0] < max_bayes_error_km and
                    tmp_cat_df.north_sd.iloc[0] < max_bayes_error_km and
                    tmp_cat_df.east_sd.iloc[0] < max_bayes_error_km):
                bayes_orig.latitude = tmp_cat_df.lat_mean.iloc[0]
                bayes_orig.longitude = tmp_cat_df.lon_mean.iloc[0]
                bayes_orig.depth = tmp_cat_df.depth_mean.iloc[0] * 1000
                bayes_orig.time = tmp_cat_df.datetime.iloc[0]
                bayes_orig.latitude_errors.uncertainty = kilometers2degrees(
                    tmp_cat_df.north_sd.iloc[0])
                bayes_orig.longitude_errors.uncertainty = kilometers2degrees(
                    tmp_cat_df.east_sd.iloc[0])
                bayes_orig.depth_errors.uncertainty = (
                    tmp_cat_df.depth_sd.iloc[0] * 1000)
                bayes_orig.time_errors.uncertainty = (
                    tmp_cat_df.time_sd.iloc[0])
                Logger.info(
                    'Added origin solution from Bayesloc for event %s',
                    event.short_str())
            event.origins.append(bayes_orig)
            event.preferred_origin_id = bayes_orig.resource_id
            # TODO indicate that this solution is from Bayesloc
    return cat