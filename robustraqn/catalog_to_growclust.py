
from eqcorrscan.utils.catalog_to_dd import _generate_event_id_mapper
import logging
Logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")


def _growclust_event_str(event, event_id):
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
        magnitude = 0.0
    try:
        time_error = origin.quality['standard_error'] or 0.0
    except (TypeError, AttributeError):
        Logger.warning('No time residual in header')
        time_error = 0.0

    z_err = (origin.depth_errors.uncertainty or 0.0) / 1000.
    # Note that these should be in degrees, but GeoNet uses meters.
    x_err = (origin.longitude_errors.uncertainty or 0.0) / 1000.
    y_err = (origin.latitude_errors.uncertainty or 0.0) / 1000.
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