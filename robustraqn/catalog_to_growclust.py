
from defusedxml import NotSupportedError
import pandas as pd
import csv

from obspy.core import UTCDateTime
from obspy.core.event import (
    Origin, OriginUncertainty, CreationInfo, QuantityError, OriginQuality)
from obspy.geodetics.base import degrees2kilometers, kilometers2degrees
from eqcorrscan.utils.catalog_to_dd import _generate_event_id_mapper

import logging
Logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")


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

    z_err = (origin.depth_errors.uncertainty or 0.0) / 1000.
    # Errors are in degrees
    x_err = degrees2kilometers(origin.longitude_errors.uncertainty or 0.0)
    y_err = degrees2kilometers(origin.latitude_errors.uncertainty or 0.0)
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
    elif num_cols == 26:  # Custom Growclust outputs origin time change
        gc_header_list.append('timeC')
    if num_cols != len(gc_header_list):
        raise ValueError('Number of column headers for growclust cat-file does'
                         + ' not match number of columns in file')

    gc_df = pd.read_csv(gc_cat_file, delim_whitespace=True,
                        names=gc_header_list)
    return gc_df


def update_cat_from_gc_file(cat, gc_cat_file, max_diff_seconds=3):
    """
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
        event.preferred_origin().arrivals = event_backup.preferred_origin(
            ).arrivals

    # Code to sort in the new locations from growclust into catalog
    for event in cat:
        cat_orig = event.preferred_origin().copy()
        lower_dtime = (cat_orig.time - max_diff_seconds)._get_datetime()
        upper_dtime = (cat_orig.time + max_diff_seconds)._get_datetime()

        tmp_cat_df = gc_df.loc[
            (gc_df.datetime > lower_dtime) & (gc_df.datetime < upper_dtime)]
        if len(tmp_cat_df) > 0:
            # Only change relocated events
            if (tmp_cat_df.eh.iloc[0] != -1 and
                    tmp_cat_df.ez.iloc[0] != -1 and
                    tmp_cat_df.et.iloc[0] != -1):
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
                # OriginUncertainty()
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


