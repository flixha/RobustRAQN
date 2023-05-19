
import pandas as pd
import csv
import numpy as np
import datetime

from obspy.core import UTCDateTime
from obspy.core.event import (
    Catalog, Origin, OriginUncertainty, CreationInfo, QuantityError,
    OriginQuality)
from obspy.geodetics.base import (
    degrees2kilometers, kilometers2degrees, locations2degrees)
from eqcorrscan.utils.catalog_to_dd import _generate_event_id_mapper

import logging
Logger = logging.getLogger(__name__)

# Silcence pandas warnings for updating rows
pd.options.mode.chained_assignment = None  # default='warn'

GC_HEADER_LIST = [
        'year', 'month', 'day', 'hour', 'minute', 'second', 'evid', 'latR',
        'lonR', 'depR', 'mag', 'qID', 'cID', 'nbranch', 'qnpair', 'qndiffP',
        'qndiffS', 'rmsP', 'rmsS', 'eh', 'ez', 'et', 'latC', 'lonC', 'depC']


def _growclust_event_str(event, event_id, no_magnitude_value=-2.0):
    """
    Make an event.dat style string for an event.

    :type event_id: int
    :param event_id: Must be an integer.
    """
    assert isinstance(event_id, int)
    try:
        origin = event.preferred_origin() or event.origins[0]
    except (IndexError, AttributeError):
        Logger.error("No origin for event {0}".format(event.resource_id.id))
        return
    try:
        magnitude = (event.preferred_magnitude() or event.magnitudes[0]).mag
    except (IndexError, AttributeError):
        Logger.warning("No magnitude")
        magnitude = no_magnitude_value
    try:
        time_error = origin.quality['standard_error'] or 0.0
    except (TypeError, AttributeError):
        Logger.warning('No time residual in header')
        time_error = 0.0

    z_err = ((origin.depth_errors.uncertainty if origin.depth_errors else 0.0)
             or 0.0) / 1000.
    # Note that these should be in degrees, but GeoNet uses meters.
    x_err = ((origin.longitude_errors.uncertainty
              if origin.longitude_errors else 0.0) or 0.0) / 1000.
    y_err = ((origin.latitude_errors.uncertainty
              if origin.latitude_errors else 0.0) or 0.0) / 1000.

    # Errors are in degrees
    x_err = degrees2kilometers(x_err)
    y_err = degrees2kilometers(y_err)
    x_err = (x_err + y_err) / 2
    x_err = max(x_err, y_err)

    event_str = (
        "{year:4d} {month:02d} {day:02d} {hour:2d} {minute:02d} "
        "{seconds:02d}.{microseconds:03d}  {latitude:8.5f} {longitude:10.5f}"
        " {depth:7.3f}   {magnitude:5.2f} {x_err:6.3f} {z_err:6.3f} "
        "{time_err:6.3f} {event_id:9d}".format(
            year=origin.time.year, month=origin.time.month,
            day=origin.time.day, hour=origin.time.hour,
            minute=origin.time.minute, seconds=origin.time.second,
            microseconds=round(origin.time.microsecond / 1e3),
            latitude=origin.latitude, longitude=origin.longitude,
            depth=origin.depth / 1000., magnitude=magnitude,
            x_err=x_err, z_err=z_err, time_err=time_error, event_id=event_id))
    return event_str


def write_event(catalog, event_id_mapper=None):
    """
    Write obspy.core.event.Catalog to a growclust phase-format evlist.txt file.

    :type catalog: obspy.core.event.Catalog
    :param catalog: A catalog of obspy events.
    :type event_id_mapper: dict
    :param event_id_mapper:
        Dictionary mapping event resource id to an integer event id for hypoDD.
        If this is None, or missing events then the dictionary will be updated
        to include appropriate event-ids. This should be of the form
        {event.resource_id.id: integer_id}

    :returns: dictionary of event-id mappings.
    """
    event_id_mapper = _generate_event_id_mapper(
        catalog=catalog, event_id_mapper=event_id_mapper)
    event_strings = [
        _growclust_event_str(event, event_id_mapper[event.resource_id.id])
        for event in catalog]
    event_strings = "\n".join(event_strings)
    with open("evlist.txt", "w") as f:
        f.write(event_strings)
    return event_id_mapper


def write_station(inventory, use_elevation=False, filename="stlist.txt"):
    """
    Write a GrowClust formatted station file.

    :type inventory: obspy.core.Inventory
    :param inventory:
        Inventory of stations to write - should include channels if
        use_elevation=True to incorporate channel depths.
    :type use_elevation: bool
    :param use_elevation: Whether to write elevations (requires hypoDD >= 2)
    :type filename: str
    :param filename: File to write stations to.
    """
    station_strings = []
    formatter = "{sta:<5s}{lat:>9.4f}{lon:>10.4f}"
    if use_elevation:
        formatter = " ".join([formatter, "{elev:>5.0f}"])

    for network in inventory:
        for station in network:
            parts = dict(sta=station.code, lat=station.latitude,
                         lon=station.longitude)
            if use_elevation:
                channel_depths = {chan.depth for chan in station}
                if len(channel_depths) == 0:
                    Logger.warning("No channels provided, using 0 depth.")
                    depth = 0.0
                else:
                    depth = channel_depths.pop()
                if len(channel_depths) > 1:
                    Logger.warning(
                        f"Multiple depths for {station.code}, using {depth}")
                parts.update(dict(elev=station.elevation - depth))
            station_strings.append(formatter.format(**parts))
    with open(filename, "w") as f:
        f.write("\n".join(station_strings))


def read_gc_cat_to_df(gc_cat_file, gc_header_list=GC_HEADER_LIST):
    """
    Read Growclust-output catalog into a dataframe.
    """
    with open(gc_cat_file) as f:
        reader = csv.reader(f, delimiter=' ', skipinitialspace=True)
        first_row = next(reader)
        num_cols = len(first_row)
    if num_cols == 25:
        pass
    elif num_cols == 26 and 'timeC' not in gc_header_list:
        # Custom Growclust outputs origin time change
        gc_header_list.append('timeC')
    if num_cols != len(gc_header_list):
        raise ValueError('Number of column headers for growclust cat-file does'
                         + ' not match number of columns in file')

    gc_df = pd.read_csv(gc_cat_file, delim_whitespace=True,
                        names=gc_header_list)
    return gc_df


def update_tribe_from_gc_file(tribe, gc_cat_file, max_diff_seconds=3):
    """
    """
    cat = Catalog([t.event for t in tribe])
    cat = update_cat_from_gc_file(cat, gc_cat_file,
                                  max_diff_seconds=max_diff_seconds)
    for ne, event in enumerate(cat):
        tribe[ne].event = event
    return tribe

# @profile
def update_cat_df_from_gc_file(full_cat_df, gc_cat_file,
                               max_diff_seconds=8,
                               max_reloc_distance_km=50,
                               return_relocated_events_only=False):
    """
    """
    # full_cat_df_backup = full_cat_df.copy()
    gc_df = read_gc_cat_to_df(gc_cat_file)
    # Convert depth to meters as is usual in obspy
    cat_df['depR'] = cat_df.depR * 1000
    gc_df['timestamp'] = pd.to_datetime(
        gc_df[['year', 'month', 'day', 'hour', 'minute', 'second']])
    gc_df['datetime'] = pd.to_datetime(gc_df.timestamp)
    # only need to check events that were relocated in Growclust
    gc_df = gc_df[~np.isnan(gc_df['rmsP'])]
    
    # limit full cat to the geographical region of interest
    full_cat_df = full_cat_df[
        (full_cat_df['latitude'] > gc_df['latR'].min() - 0.5) &
        (full_cat_df['latitude'] < gc_df['latR'].max() + 0.5) &
        (full_cat_df['longitude'] > gc_df['lonR'].min() - 0.5) &
        (full_cat_df['longitude'] < gc_df['lonR'].max() + 0.5)]

    # gc_df['utcdatetime'] = [UTCDateTime(gc_df.datetime.iloc[nr])
    #                         for nr in range(len(gc_df))]

    n_events = len(full_cat_df)
    # initialize new columns
    if 'growclustR' not in full_cat_df.columns:
        full_cat_df['growclustR'] = np.zeros(n_events, dtype=np.int8)
    if 'timeR' not in full_cat_df.columns:
        # full_cat_df['timeR'] = np.array([np.datetime64('NaT')
        #                                  for j in range(n_events)])
        full_cat_df['timeR'] =  np.array([pd.NaT for j in range(n_events)])
    if 'latR' not in full_cat_df.columns:
        full_cat_df['latR'] = np.nan * np.ones(n_events)
    if 'lonR' not in full_cat_df.columns:
        full_cat_df['lonR'] = np.nan * np.ones(n_events)
    if 'depR' not in full_cat_df.columns:
        full_cat_df['depR'] = np.nan * np.ones(n_events)
    if 'qID' not in full_cat_df.columns:
        full_cat_df['qID'] = np.nan * np.ones(n_events)
    if 'cID' not in full_cat_df.columns:
        full_cat_df['cID'] = np.nan * np.ones(n_events)
    if 'nbranch' not in full_cat_df.columns:
        full_cat_df['nbranch'] = np.nan * np.ones(n_events)
    if 'qnpair' not in full_cat_df.columns:
        full_cat_df['qnpair'] = np.nan * np.ones(n_events)
    if 'qndiffP' not in full_cat_df.columns:
        full_cat_df['qndiffP'] = np.nan * np.ones(n_events)
    if 'qndiffS' not in full_cat_df.columns:
        full_cat_df['qndiffS'] = np.nan * np.ones(n_events)
    if 'rmsP' not in full_cat_df.columns:
        full_cat_df['rmsP'] = np.nan * np.ones(n_events)
    if 'rmsS' not in full_cat_df.columns:
        full_cat_df['rmsS'] = np.nan * np.ones(n_events)
    if 'eh' not in full_cat_df.columns:
        full_cat_df['eh'] = np.nan * np.ones(n_events)
    if 'ez' not in full_cat_df.columns:
        full_cat_df['ez'] = np.nan * np.ones(n_events)
    if 'et' not in full_cat_df.columns:
        full_cat_df['et'] = np.nan * np.ones(n_events)
    if 'latC' not in full_cat_df.columns:
        full_cat_df['latC'] = np.nan * np.ones(n_events)
    if 'lonC' not in full_cat_df.columns:
        full_cat_df['lonC'] = np.nan * np.ones(n_events)
    if 'depC' not in full_cat_df.columns:
        full_cat_df['depC'] = np.nan * np.ones(n_events)

    # keep copy of original origin solution
    if 'otime' not in full_cat_df.columns:
        full_cat_df['otime'] = full_cat_df['time']
    if 'olat' not in full_cat_df.columns:
        full_cat_df['olat'] = full_cat_df['latitude']
    if 'olon' not in full_cat_df.columns:
        full_cat_df['olon'] = full_cat_df['longitude']
    if 'odepth' not in full_cat_df.columns:
        full_cat_df['odepth'] = full_cat_df['depth']

    # Loop through relocated events, find original event, and update it.
    for n_row, gc_event in gc_df.iterrows():
        lower_dtime = (
            gc_event.datetime - datetime.timedelta(seconds=max_diff_seconds))
        upper_dtime = (
            gc_event.datetime + datetime.timedelta(seconds=max_diff_seconds))

        tmp_cat_df = full_cat_df.loc[(full_cat_df.otime > lower_dtime)
                                     & (full_cat_df.otime < upper_dtime)]
        if len(tmp_cat_df) == 0:
            closest_time_diff = np.abs(
                full_cat_df.otime - gc_event.datetime).min()
            Logger.warning(
                'Could not find matching event within %s s time difference '
                '(closest: %s s) in catalog for relocated event %s, %s, %s, %s'
                , max_diff_seconds, closest_time_diff,
                gc_event.datetime, gc_event.latR, gc_event.lonR, gc_event.depR)
            continue
        # Check the distance between the relocated event and the original,
        # if there are multiple candidates select the one closest in time and
        # space.
        # distances = [degrees2kilometers(
        #     locations2degrees(gc_event.latR, gc_event.lonR,
        #                       tmp_cat_df.iloc[nr].olat,
        #                       tmp_cat_df.iloc[nr].olon)
        #     for nr in range(len(tmp_cat_df)))]
        nj = len(tmp_cat_df)
        distances = degrees2kilometers(locations2degrees(
            np.array([gc_event.latR for j in range(nj)]),
            np.array([gc_event.lonR for j in range(nj)]),
            tmp_cat_df.olat, tmp_cat_df.olon))
        # include depth difference in distance estimate if possible
        distances = np.array([
            distance if np.isnan(tmp_cat_df.iloc[nr].odepth)
            else np.sqrt(distance ** 2 + (
                gc_event.depR / 1000 - tmp_cat_df.iloc[nr].odepth / 1000) ** 2)
            for nr, distance in enumerate(distances)])
        closest_distance_km = min(distances)
        if min(distances) > max_reloc_distance_km:
            Logger.warning(
                'Could not find event within %s s time difference and %s km '
                'distance (closest: %s km) for relocated event %s, %s, %s, %s'
                , max_diff_seconds, max_reloc_distance_km, closest_distance_km,
                gc_event.datetime, gc_event.latR, gc_event.lonR, gc_event.depR)
            continue
        within_bounds_df = tmp_cat_df.loc[distances <= max_reloc_distance_km]
        tmp_event_index = np.argmin(distances)
        fc_index = tmp_cat_df.index[tmp_event_index]
        tmp_event = tmp_cat_df.iloc[tmp_event_index]
        # Now update the catalog with the relocated event information.
        if tmp_event.growclustR:
            Logger.warning(
                'Event %s, %s, %s, %s already has been assigned a Growclust-'
                'hypocenter (%s events within bounds), overwriting it with '
                '%s, %s, %s, %s',
                tmp_event.time, tmp_event.latR, tmp_event.lonR, tmp_event.depR,
                len(within_bounds_df),
                gc_event.datetime, gc_event.latR, gc_event.lonR, gc_event.depR)
        # Below is a quicker way to assign the new values in a row
        # full_cat_df['growclustR'].loc[fc_index] = True
        # full_cat_df['timeR'].loc[fc_index] = gc_event.datetime
        # full_cat_df['latR'].loc[fc_index] = gc_event.latR
        # full_cat_df['lonR'].loc[fc_index] = gc_event.lonR
        # full_cat_df['depR'].loc[fc_index] = gc_event.depR
        # full_cat_df['qID'].loc[fc_index] = gc_event.qID
        # full_cat_df['cID'].loc[fc_index] = gc_event.cID
        # full_cat_df['nbranch'].loc[fc_index] = gc_event.nbranch
        # full_cat_df['qnpair'].loc[fc_index] = gc_event.qnpair
        # full_cat_df['qndiffP'].loc[fc_index] = gc_event.qndiffP
        # full_cat_df['qndiffS'].loc[fc_index] = gc_event.qndiffS
        # full_cat_df['rmsP'].loc[fc_index] = gc_event.rmsP
        # full_cat_df['rmsS'].loc[fc_index] = gc_event.rmsS
        # full_cat_df['eh'].loc[fc_index] = gc_event.eh
        # full_cat_df['ez'].loc[fc_index] = gc_event.ez
        # full_cat_df['et'].loc[fc_index] = gc_event.et
        # full_cat_df['latC'].loc[fc_index] = gc_event.latC
        # full_cat_df['lonC'].loc[fc_index] = gc_event.lonC
        # full_cat_df['depC'].loc[fc_index] = gc_event.depC
        # # update origin in catalog with GC solution
        # full_cat_df['time'].loc[fc_index] = gc_event.datetime
        # full_cat_df['latitude'].loc[fc_index] = gc_event.latR
        # full_cat_df['longitude'].loc[fc_index] = gc_event.lonR
        # full_cat_df['depth'].loc[fc_index] = gc_event.depR

        full_cat_df.at[fc_index, 'growclustR'] = True
        full_cat_df.at[fc_index, 'timeR'] = gc_event.datetime
        full_cat_df.at[fc_index, 'latR'] = gc_event.latR
        full_cat_df.at[fc_index, 'lonR'] = gc_event.lonR
        full_cat_df.at[fc_index, 'depR'] = gc_event.depR
        full_cat_df.at[fc_index, 'qID'] = gc_event.qID
        full_cat_df.at[fc_index, 'cID'] = gc_event.cID
        full_cat_df.at[fc_index, 'nbranch'] = gc_event.nbranch
        full_cat_df.at[fc_index, 'qnpair'] = gc_event.qnpair
        full_cat_df.at[fc_index, 'qndiffP'] = gc_event.qndiffP
        full_cat_df.at[fc_index, 'qndiffS'] = gc_event.qndiffS
        full_cat_df.at[fc_index, 'rmsP'] = gc_event.rmsP
        full_cat_df.at[fc_index, 'rmsS'] = gc_event.rmsS
        full_cat_df.at[fc_index, 'eh'] = gc_event.eh
        full_cat_df.at[fc_index, 'ez'] = gc_event.ez
        full_cat_df.at[fc_index, 'et'] = gc_event.et
        full_cat_df.at[fc_index, 'latC'] = gc_event.latC
        full_cat_df.at[fc_index, 'lonC'] = gc_event.lonC
        full_cat_df.at[fc_index, 'depC'] = gc_event.depC
        # update origin in catalog with GC solution 
        full_cat_df.at[fc_index, 'time'] = gc_event.datetime
        full_cat_df.at[fc_index, 'latitude'] = gc_event.latR
        full_cat_df.at[fc_index, 'longitude'] = gc_event.lonR
        full_cat_df.at[fc_index, 'depth'] = gc_event.depR

    if return_relocated_events_only:
        sel_df = full_cat_df[full_cat_df['growclustR'] == True]
        return sel_df
    return full_cat_df


def update_cat_from_gc_file(cat, gc_cat_file, max_diff_seconds=3):
    """
    Growclust eh, ez errors are median absolute deviations of the bootstrap
    distribution.
    """
    cat_backup = cat.copy()
    gc_df = read_gc_cat_to_df(gc_cat_file)
    gc_df['timestamp'] = pd.to_datetime(
        gc_df[['year', 'month', 'day', 'hour', 'minute', 'second']])
    gc_df['datetime'] = pd.to_datetime(gc_df.timestamp)
    gc_df['utcdatetime'] = [UTCDateTime(gc_df.datetime.iloc[nr])
                            for nr in range(len(gc_df))]

    # put back arrivals
    for event, event_backup in zip(cat, cat_backup):
        event_orig = event.preferred_origin() or event.origins[0]
        backup_orig = (
            event_backup.preferred_origin() or event_backup.origins[0])
        event_orig.arrivals = backup_orig.arrivals

    # Code to sort in the new locations from growclust into catalog
    for event in cat:
        cat_orig = (event.preferred_origin() or event.origins[0]).copy()
        lower_dtime = (cat_orig.time - max_diff_seconds)._get_datetime()
        upper_dtime = (cat_orig.time + max_diff_seconds)._get_datetime()

        tmp_cat_df = gc_df.loc[
            (gc_df.datetime > lower_dtime) & (gc_df.datetime < upper_dtime)]
        if len(tmp_cat_df) > 0:
            # Only change relocated events
            if (tmp_cat_df.eh.iloc[0] != -1
                 and not np.isnan(tmp_cat_df.eh.iloc[0])
                 and tmp_cat_df.ez.iloc[0] != -1
                 and not np.isnan(tmp_cat_df.ez.iloc[0])
                 and tmp_cat_df.et.iloc[0] != -1
                 and not np.isnan(tmp_cat_df.et.iloc[0])):
                # gc_orig.latitude = tmp_cat_df.latR.iloc[0]
                # gc_orig.longitude = tmp_cat_df.lonR.iloc[0]
                # gc_orig.depth = tmp_cat_df.depR.iloc[0] * 1000
                # gc_orig.time = tmp_cat_df.datetime.iloc[0]
                # gc_orig.latitude_errors.uncertainty = kilometers2degrees(
                #     tmp_cat_df.eh.iloc[0])
                # gc_orig.longitude_errors.uncertainty = kilometers2degrees(
                #     tmp_cat_df.eh.iloc[0])
                # gc_orig.depth_errors.uncertainty = (tmp_cat_df.ez.iloc[0])
                # gc_orig.time_errors.uncertainty = (tmp_cat_df.et.iloc[0])

                # OriginQuality(used_station_count=)
                origin_uncertainty = OriginUncertainty(
                    horizontal_uncertainty=tmp_cat_df.eh.iloc[0] * 1000,
                    preferred_description="horizontal uncertainty")

                gc_orig = Origin(
                    force_resource_id=True,
                    latitude=tmp_cat_df.latR.iloc[0],
                    latitude_errors=QuantityError(
                        uncertainty=kilometers2degrees(tmp_cat_df.eh.iloc[0])),
                    longitude=tmp_cat_df.lonR.iloc[0],
                    longitude_errors=QuantityError(
                        uncertainty=kilometers2degrees(tmp_cat_df.eh.iloc[0])),
                    depth=tmp_cat_df.depR.iloc[0] * 1000,
                    depth_errors=QuantityError(
                        uncertainty=tmp_cat_df.ez.iloc[0] * 1000),
                    time=tmp_cat_df.datetime.iloc[0],
                    time_errors=QuantityError(
                        uncertainty=tmp_cat_df.et.iloc[0]),
                    origin_uncertainty=origin_uncertainty,
                    creation_info=CreationInfo(agency_id='BER', author='GC'))
                # arrivals=cat_orig.arrivals)
                Logger.info(
                    'Added origin solution from Growclust for event %s',
                    event.short_str())
                # TODO: update arrivals list
                # TODO: add growclust clustering and RMS output to origin
                event.origins.append(gc_orig)
                event.preferred_origin_id = gc_orig.resource_id
    return cat
