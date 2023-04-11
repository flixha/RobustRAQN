
# %%
import os
import numpy as np

import pandas as pd
import numpy as np
import time

import textwrap
from datetime import datetime
from joblib.parallel import Parallel, delayed
from wcmatch import fnmatch
from collections import Counter

from obspy.core.event import (
    Catalog, Event, QuantityError, OriginQuality, OriginUncertainty)
from obspy import Inventory
from obspy.geodetics.base import degrees2kilometers, kilometers2degrees
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.event import Origin, Pick, Arrival
from obspy.core.event.base import CreationInfo, WaveformStreamID
from obspy.taup import TauPyModel
from obspy.core.util.attribdict import AttribDict
from obspy.io.nordic.ellipse import Ellipse

from robustraqn.core.seismic_array import get_station_sites

import logging
Logger = logging.getLogger(__name__)


def _cc_round(num, dp):
    """
    Convenience function to take a float and round it to dp padding with zeros
    to return a string

    :type num: float
    :param num: Number to round
    :type dp: int
    :param dp: Number of decimal places to round to.

    :returns: str

    >>> print(_cc_round(0.25364, 2))
    0.25
    """
    num = round(num, dp)
    num = '{0:.{1}f}'.format(num, dp)
    return num


def readSTATION0(path, stations):
    """
    From old version of EQCorrscan, written by Calum Chamberlain.

    Read a Seisan STATION0.HYP file on the path given.

    Outputs the information, and writes to station.dat file.

    :type path: str
    :param path: Path to the STATION0.HYP file
    :type stations: list
    :param stations: Stations to look for

    :returns: List of tuples of station, lat, long, elevation
    :rtype: list

    >>> readSTATION0('eqcorrscan/tests/test_data', ['WHFS', 'WHAT2', 'BOB'])
    [('WHFS', -43.261, 170.359, 60.0), ('WHAT2', -43.2793, \
170.36038333333335, 95.0), ('BOB', 41.408166666666666, \
-174.87116666666665, 101.0)]
    """
    stalist = []
    f = open(path + '/STATION0.HYP', 'r')
    for line in f:
        line_sta = line[1:6].strip()
        wildcarded_stations = [sta for sta in stations
                               if '?' in sta or '*' in sta]
        if (line_sta in stations or fnmatch.fnmatch(
                line_sta, wildcarded_stations, flags=fnmatch.EXTMATCH)):
            station = line[1:6].strip()
            # Format is either ddmm.mmS/N or ddmm(.)mmmS/N
            lat = line[6:14].replace(' ', '0')
            if lat[-1] == 'S':
                NS = -1
            else:
                NS = 1
            if lat[4] == '.':
                lat = (int(lat[0:2]) + float(lat[2:-1]) / 60) * NS
            else:
                # degrees = lat[0:2]
                # if degrees.strip() == '':
                #     degrees = '0'
                # # minutes = lat[2:4]
                # # if ' ' in minutes:
                # #     minutes = minutes.replace(' ', '0')
                # # decimal_minutes = lat[4:-1]
                # # if ' ' in decimal_minutes:
                # #     decimal_minutes = decimal_minutes.replace(' ', '0')
                lat = (int(lat[0:2]) + float(lat[2:4] + '.' + lat[4:-1]) /
                       60) * NS
                # lat = (
                #     int(degrees) + float(minutes + '.' + decimal_minutes)
                #        60) * NS
            lon = line[14:23].replace(' ', '0')
            if lon[-1] == 'W':
                EW = -1
            else:
                EW = 1
            if lon[5] == '.':
                lon = (int(lon[0:3]) + float(lon[3:-1]) / 60) * EW
            else:
                # degrees = lon[0:3]
                # if degrees.strip() == '':
                #     degrees = '0'
                # minutes = lon[3:5]
                # if ' ' in minutes:
                #     minutes = minutes.replace(' ', '0')
                # decimal_minutes = lon[5:-1]
                # if ' ' in decimal_minutes:
                #     decimal_minutes = decimal_minutes.replace(' ', '0')
                lon = (int(
                    lon[0:3]) + float(lon[3:5] + '.' + lon[5:-1]) / 60) * EW
                # lon = (
                #  int(degrees) + float(minutes + '.' + decimal_minutes) /
                #        60) * EW
            try:
                elev = float(line[23:-1].strip())
            except ValueError:  # Elevation could be empty
                elev = 0.0
            # Note, negative altitude can be indicated in 1st column
            if line[0] == '-':
                elev *= -1
            stalist.append((station, lat, lon, elev))
    f.close()
    f = open('station0.dat', 'w')
    for sta in stalist:
        line = ''.join([sta[0].ljust(5), _cc_round(sta[1], 4).ljust(10),
                        _cc_round(sta[2], 4).ljust(10),
                        _cc_round(sta[3] / 1000, 4).rjust(7), '\n'])
        f.write(line)
    f.close()
    return stalist


def write_station(inventory, station0_file=None, stations=[],
                  filename="station.dat", write_only_once=True):
    """
    Write a GrowClust formatted station file.

    :type inventory: obspy.core.Inventory
    :param inventory:
        Inventory of stations to write - should include channels if
        use_elevation=True to incorporate channel depths.
    :type station0_file: str
    :param station0_file: Path to a station0 file to read in stations from.
    :type stations: list
    :param stations: List of stations to write to file.
    :type filename: str
    :param filename: File to write stations to.
    :type write_only_once: bool
    :param write_only_once: Whether to write stations only once to file.

    :rtype: list
    :returns: List of station names written to file.
    """
    station_strings = []
    # formatter = "{sta:<5s}{lat:>9.4f}{lon:>10.4f}{elev:>10.3f}"
    formatter = "{sta:<s} {lat:>.4f} {lon:>.4f} {elev:>.3f}"
    # %s %.4f %.4f %.3f
    known_station_names = []
    for network in inventory:
        for station in network:
            parts = dict(sta=station.code, lat=station.latitude,
                         lon=station.longitude, elev=station.elevation / 1000)
            # if use_elevation:
            #     channel_depths = {chan.depth for chan in station}
            #     if len(channel_depths) == 0:
            #         Logger.warning("No channels provided, using 0 depth.")
            #         depth = 0.0
            #     else:
            #         depth = channel_depths.pop()
            #     if len(channel_depths) > 1:
            #         Logger.warning(
            #             f"Multiple depths for {station.code}, using {depth}")
            #     parts.update(dict(elev=station.elevation - depth))
            if write_only_once:  # Write only one location per station
                if station.code in known_station_names:
                    continue
            # Don't write station if no picks in catalog
            if station.code not in stations:
                continue
            station_strings.append(formatter.format(**parts))
            known_station_names.append(station.code)
    if station0_file is not None:
        stalist = readSTATION0(station0_file, stations)
        for line in stalist:
            if write_only_once:
                if line[0] in known_station_names:
                    continue
            parts = dict(sta=line[0], lat=line[1], lon=line[2],
                         elev=line[3] / 1000)
            station_strings.append(formatter.format(**parts))
            known_station_names.append(line[0])
    with open(filename, "w") as f:
        f.write('sta_id lat lon elev\n')
        f.write("\n".join(station_strings))
    return known_station_names


def _select_best_origin(
        event, evid, priors=[], fix_depth=10, default_depth=10,
        def_dep_error_m=10000, default_lat=None, default_lon=None,
        def_hor_error_deg=None, def_time_error_s=5,
        default_start=datetime(1970, 1, 1, 0, 0, 0)):
    """
    Select the best origin (considering how much information it has available)
    and append it to priors

    :type event: :class:`~obspy.core.event.Event`
    :param event: Event to select origin from
    :type evid: int
    :param evid: Event ID
    :type priors: list
    :param priors: List of prior origins to append to:
    :type fix_depth: float
    :param fix_depth: Depth to fix all origins to (in km)
    :type default_depth: float
    :param default_depth: Default depth to use if no depth is available
    :type def_dep_error_m: float
    :param def_dep_error_m: Default depth error to use if no depth error is
    :type default_lat: float
    :param default_lat: Default latitude to use if no lat is available
    :type default_lon: float
    :param default_lon: Default longitude to use if no lon is available
    :type def_hor_error_deg: float
    :param def_hor_error_deg:
        Default horizontal error to use if no error is available
    :type def_time_error_s: float
    :param def_time_error_s: Default time error to use if no error is available
    :type default_start: :class:`~datetime.datetime`
    :param default_start: Default start time to use if no time is available
    :type default_start: :class:`~datetime.datetime`
    :param default_start:
        Default reference epoch start time to use, defaults to the Linux epoch
        default of 1970-01-01T00:00:00. Consider to use earlier epoch starts
        for earthquake catalogs that start earlier..

    :rtype: tuple
    :return:
        Tuple of (origin, origin_epoch_time, hor_uncertainty_km,
        depth_uncertainty_km, time_uncertainty_s)
    """
    orig = event.preferred_origin() or event.origins[0]
    # Select the best origin - it should have lat/lon, and the smaller the
    # the error / RMS the better (?)
    if orig.latitude is None or orig.longitude is None:
        for origin in event.origins:
            if origin.latitude is not None and orig.longitude is not None:
                orig = origin
    # Fix depth to a common depth for a cluster so that a-priori depth doesn't
    # influence final depths.
    if fix_depth is not None:
        orig.depth = fix_depth
    orig_epoch_time = (
        orig.time._get_datetime() - default_start).total_seconds()

    # Check uncertainties of the origins
    hor_uncertainty_km = None
    if (orig.origin_uncertainty is not None and
            orig.origin_uncertainty.max_horizontal_uncertainty):
        hor_uncertainty_km = (
            orig.origin_uncertainty.max_horizontal_uncertainty / 1000)
    elif (orig.origin_uncertainty is not None and
            orig.origin_uncertainty.horizontal_uncertainty):
        hor_uncertainty_km = (
            orig.origin_uncertainty.horizontal_uncertainty / 1000)
    elif (orig.longitude_errors is not None
          and orig.longitude_errors.uncertainty is not None
          and orig.latitude_errors is not None
          and orig.latitude_errors.uncertainty):
        hor_uncertainty_km = degrees2kilometers((
            (orig.latitude_errors.uncertainty or def_hor_error_deg) +
            (orig.longitude_errors.uncertainty or def_hor_error_deg)) / 2)
    elif (orig.longitude_errors is not None
          and orig.longitude_errors.uncertainty is not None):
        hor_uncertainty_km = degrees2kilometers(
            orig.longitude_errors.uncertainty)
    elif (orig.latitude_errors is not None
          and orig.latitude_errors.uncertainty is not None):
        hor_uncertainty_km = degrees2kilometers(
            orig.latitude_errors.uncertainty)
    if hor_uncertainty_km is None:
        hor_uncertainty_km = degrees2kilometers(def_hor_error_deg)

    if orig.depth is not None:
        o_depth = orig.depth / 1000
    else:
        o_depth = default_depth / 1000

    if orig.depth_errors.uncertainty is not None:
        o_depth_error = orig.depth_errors.uncertainty / 1000
    else:
        o_depth_error = def_dep_error_m / 1000

    priors.append((
        evid, orig.latitude or default_lat, orig.longitude or default_lon,
        # orig.origin_uncertainty.max_horizontal_uncertainty / 1000 or 30,
        hor_uncertainty_km, o_depth, o_depth_error,
        orig_epoch_time, orig.time_errors.uncertainty or def_time_error_s))
    return orig, priors


def _get_traveltime(models, degree, depth, phase, model_cutoff_distances=[]):
    """
    Retrieve the traveltime for a given phase, distance and depth from a
    velocity model.

    :type models: list
    :param models:
        List of velocity models to use (see model_cutoff_distances parameter
        when supplying multiple models). Supply models as
        ´obspy.taup.TauPyModel´
    :type degree: float
    :param degree: Distance in degrees
    :type depth: float
    :param depth: Depth in km
    :type phase: str
    :param phase: Phase name
    :type model_cutoff_distances: list
    :param model_cutoff_distances: List of cutoff distances in degrees for the
        velocity models. The first model in the list will be used for all
        distances smaller than the first cutoff distance, etc. If no cutoff
        is given, the first model will be used for all distances.

    :rtype: float
    :return: Traveltime in seconds
    """
    if len(model_cutoff_distances) < len(models):
        model = models[0]
    else:
        # Select the correct model according to the cutoff distance of each
        # model and the source-receiver distance in degrees
        model = None
        for jm, cutoff_dist in enumerate(model_cutoff_distances):
            if jm == 0:
                if degree <= cutoff_dist:
                    model = models[jm]
            elif (degree <= cutoff_dist
                    and degree > model_cutoff_distances[jm-1]):
                model = models[jm]
        if model is None:
            Logger.error('No velocity model applicable at distance %s',
                         degree)
    if phase in ['P', 'S']:  # For short phase name, get quickest phase
        phase_list = [phase, phase.lower(), phase + 'g', phase + 'n']
        try:
            arrivals = model.get_travel_times(depth, degree,
                                              phase_list=phase_list)
            return min([arrival.time for arrival in arrivals])
        except (IndexError, ValueError):
            return np.nan
    else:
        try:
            arrival = model.get_travel_times(depth, degree, phase_list=[phase])
            return arrival[0].time
        except IndexError:
            # if there's no path, try with an upgoing path Pg --> P / p
            try:
                arrival = model.get_travel_times(
                    depth, degree, phase_list=[phase[0]])
                return arrival[0].time
            except IndexError:
                try:
                    arrival = model.get_travel_times(
                        depth, degree, phase_list=[phase[0].lower()])
                    return arrival[0].time
                except IndexError:
                    return np.nan


def _get_nordic_event_id(event, return_generic_nordic_id=False):
    """
    Get the event id that is mentioned in the most recent comment in event read
    from Nordic file.

    :type event: :class:`~obspy.core.event.event.Event`
    :param event: Event to get the nordic event id from.
    :type return_generic_nordic_id: bool
    :param return_generic_nordic_id:
        If True, return a generic nordic event if no "real" Nordic event id is
        found.

    :rtype: int
    :return: Nordic event id.
    """
    if hasattr(event, 'extra') and 'nordic_event_id' in event.extra.keys():
        event_id = event.extra['nordic_event_id']['value']
    else:
        # sort comments by date they were written
        comments = event.comments.copy()
        comments = [comment for comment in comments
                    if len(comment.text.split(' ')) > 1]
        comments = sorted(comments, key=lambda x: x.text.split(' ')[1],
                          reverse=True)
        try:
            event_id = int(
                [com.text.split('ID:')[-1].strip('SdLRD ')
                 for com in comments if 'ID:' in com.text][0])
        except IndexError:
            event_id = None
    # Return generic nordic ID in case no specific ID is saved in file.
    if event_id is None and return_generic_nordic_id:
        event_id = int(event.short_str(
            )[0:19].replace('-', '').replace(':', '').replace('T', ''))
    if isinstance(event_id, str):
        event_id = int(event_id)
    return event_id


def write_ttimes_files(
        models, model_cutoff_distances=[], mod_names=[], out_name=None,
        phase_list=['P', 'Pg', 'Pn', 'S', 'Sg', 'Sn', 'P1', 'S1'],
        min_depth=0, max_depth=40, depth_step=2, min_degrees=0, max_degrees=10,
        min_tele_degrees=13,
        degree_step=0.2, outpath='.', print_first_arriving_PS=True,
        full_teleseismic_phases=False, parallel=False, cores=40):
    """
    Write a traveltime lookup table file with travel times for each supplied
    phase, for a list of velocity models used at different distances

    :type models: list
    :param models:
        List of velocity models to use (see model_cutoff_distances
        parameter when supplying multiple models). Supply models as
        ´obspy.taup.TauPyModel´
    :tpye model_cutoff_distances: list
    :param model_cutoff_distances:
        List of cutoff distances in degrees for the velocity models. The first
        model in the list will be used for all distances smaller than the first
        cutoff distance, etc. If no cutoff is given, the first model will be
        used for all distances.
    :type mod_names: list
    :param mod_names:
        List of names for the velocity models. If not supplied, the model names
        will be taken from the model file names.
    :type out_name: str
    :param out_name: Name of the output file
    :type phase_list: list
    :param phase_list: List of phases to calculate travel times for
    :type min_depth: float
    :param min_depth: Minimum depth to calculate travel times for
    :type max_depth: float
    :param max_depth: Maximum depth to calculate travel times for
    :type depth_step: float
    :param depth_step: Depth step to calculate travel times for
    :type min_degrees: float
    :param min_degrees: Minimum epicentral distance to calculate travel times
    :type max_degrees: float
    :param max_degrees: Maximum epicentral distance to calculate travel times
    :type degree_step: float
    :param degree_step: Epicentral distance step to calculate travel times
    :type outpath: str
    :param outpath: Path to write the output file to
    :type print_first_arriving_PS: bool
    :param print_first_arriving_PS:
        Print out additional files for the the first arriving P and S phases
        "P1" and "S1" (default: True)
    :type full_teleseismic_phases: bool
    :param full_teleseismic_phases:
        For full telsesmic phases, use as lower distance limit the distance in
        min_tele_degrees.
    :type parallel: bool
    :param parallel: Calculate travel times in parallel
    :type cores: int
    :param cores: Number of cores to use for parallel calculation
    :type min_tele_degrees: float
    :param min_tele_degrees:
        Minimum epicentral distance to calculate travel times for teleseismic
        phases

    :rtype: None
    :return: None
    """
    used_models = []
    for jm, model in enumerate(models):
        if not isinstance(model, TauPyModel):
            mod_name = str(model.split('/')[-1].split('.')[0])
            model = TauPyModel(model)
        else:
            if mod_names:
                mod_name = mod_names[jm]
            else:
                mod_name = 'def_'
        used_models.append(model)
        mod_names.append(mod_name)
    models = used_models
    # model = TauModel.from_file('NNSN1D_plusAK135.npz')
    # a list of epicentral distances without a travel time, and a flag:
    # notimes = []
    # plotted = False

    # calculate the arrival times and plot vs. epicentral distance:
    depths = np.arange(min_depth, max_depth, depth_step)
    # degrees = np.linspace(min_degrees, max_degrees, npoints)
    out_name = out_name or mod_names[0]
    degrees = np.arange(min_degrees, max_degrees, degree_step)
    for phase in phase_list:
        if full_teleseismic_phases:
            if phase in ['P', 'S', 'pP', 'sP', 'pS', 'sS']:
                degrees = np.arange(min_tele_degrees, 360, degree_step)
            else:
                degrees = np.arange(min_degrees, max_degrees, degree_step)
        ttfile = out_name + '.' + phase
        ttfile = os.path.expanduser(os.path.join(outpath, ttfile))
        Logger.info('Writing traveltime table to file %s', ttfile)
        f = open(ttfile, 'wt')
        f.write((
            '# travel-time table for velocityModel=%s and phase=%s. Generated '
            + 'with Obspy.taup and write_ttimes_files.\n') % (out_name, phase))
        f.write('%s  Number of depth samples at the following depths (km):\n' %
                len(depths))
        depths_str = ''.join(['%8.2f' % depth for depth in depths])
        depths_str = '\n'.join(textwrap.wrap(depths_str, 80))
        f.write(depths_str + '\n')
        f.write(('%s  131  Number of distance samples at the following ' +
                 'distances (deg):\n') % len(degrees))
        degree_str = ''.join(['%8.2f' % deg for deg in degrees])
        degree_str = '\n'.join(textwrap.wrap(degree_str, 80))
        f.write(degree_str + '\n')
        comp_phase = phase
        if phase in ['P1', 'S1']:
            comp_phase = phase[0]
            print_first_arriving_PS = True
        for depth in depths:
            if parallel:
                ttimes = Parallel(n_jobs=cores)(delayed(_get_traveltime)(
                    models, degree, depth, comp_phase,
                    model_cutoff_distances=model_cutoff_distances)
                                                for degree in degrees)
            else:
                ttimes = [_get_traveltime(
                    models, degree, depth, comp_phase,
                    model_cutoff_distances=model_cutoff_distances)
                          for degree in degrees]

            # make sure P and S are the first arrivals -check if Pn or Sn are
            # quicker at specific depth/distance
            if print_first_arriving_PS:
                if phase in ['P1', 'S1']:
                    n_ttimes = Parallel(n_jobs=cores)(delayed(_get_traveltime)(
                        models, degree, depth, comp_phase + 'n',
                        model_cutoff_distances=model_cutoff_distances)
                                                      for degree in degrees)
                    b_ttimes = Parallel(n_jobs=cores)(delayed(_get_traveltime)(
                        models, degree, depth, comp_phase + 'b',
                        model_cutoff_distances=model_cutoff_distances)
                                                      for degree in degrees)
                    ttimes = [np.nanmin([ttimes[j], n_ttimes[j], b_ttimes[j]])
                              for j, ttime in enumerate(ttimes)]
            f.write('Travel time at depth = %8.2f km.\n' % depth)
            for ttime in ttimes:
                f.write('%11.4f\n' % ttime)
        f.close()


def _fill_bayes_origin_quality(event, orig):
    """
    Fill in origin quality information from Bayesloc output origin.

    :type event: :class:`~obspy.core.event.event.Event`
    :param event: Event to fill in origin quality information for.
    :type orig: :class:`~obspy.core.event.origin.Origin`
    :param orig: Origin to fill in origin quality information for.

    :rtype: :class:`~obspy.core.event.origin.OriginQuality`
    :return: Origin quality information.
    """
    if orig is not None and orig.quality is not None:
        orig_quality = orig.quality
    else:
        if event is not None:
            used_station_count = len(set(
                [pick.waveform_id.station_code for pick in event.picks
                    if pick.waveform_id is not None]))
        else:
            used_station_count = 0
        orig_quality = OriginQuality(used_station_count=used_station_count)
    return orig_quality


def _fill_bayes_origin_uncertainty(row):
    """
    Fill in an uncertainty ellipse into an obspy origin from Bayesloc output.

    :type row: :class:`~pandas.core.series.Series`
    :param row: Row of Bayesloc output dataframe.

    :rtype: :class:`~obspy.core.event.origin.OriginUncertainty`
    :return: Origin uncertainty information.
    """
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    cov = [[row.east_sd, row.north_east_cor],
           [row.north_east_cor, row.north_sd]]
    # IF there are any nans or infs:
    if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
        ellipse = AttribDict()
        ellipse.a = 0
        ellipse.b = 0
        ellipse.theta = 0
    else:
        ellipse = Ellipse.from_cov(cov, center=(0, 0))
    origin_uncertainty = None
    if ellipse:
        origin_uncertainty = OriginUncertainty(
            max_horizontal_uncertainty=ellipse.a * 1000.,
            min_horizontal_uncertainty=ellipse.b * 1000.,
            azimuth_max_horizontal_uncertainty=ellipse.theta,
            preferred_description="uncertainty ellipse")
    if ellipse is None:
        Logger.error(
            'Event %s: Unertainty ellipse could not be created, maybe covarian'
            'ce values are not positive definite.', row.ev_id)
        min_unc = min(row.east_sd, row.north_sd)
        max_unc = max(row.east_sd, row.north_sd)
        # TODO: fix how this would set the ellipse azimuth incorrectly
        origin_uncertainty = OriginUncertainty(
            # horizontal_uncertainty=np.mean([min_unc, max_unc]),
            max_horizontal_uncertainty=max_unc,
            min_horizontal_uncertainty=min_unc,
            azimuth_max_horizontal_uncertainty=0,
            preferred_description="uncertainty ellipse")
    return origin_uncertainty


def read_bayesloc_origins(
        bayesloc_origins_ned_stats_file, cat=Catalog(),
        custom_epoch=None, agency_id='', find_event_without_id=False,
        s_diff=3, max_bayes_error_km=100, read_all_iterations=False):
    """
    Read bayesloc origins into Catalog and pandas datafrane.

    :type bayesloc_origins_ned_stats_file: str
    :param bayesloc_origins_ned_stats_file:
        Path to Bayesloc origins file or the bayesloc execution directory.
    :type cat: :class:`~obspy.core.event.catalog.Catalog`
    :param cat: Catalog to add bayesloc origins to.
    :type custom_epoch: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param custom_epoch: Custom epoch to use for bayesloc origins.
    :type agency_id: str
    :param agency_id: Agency ID to use for bayesloc origins.
    :type find_event_without_id: bool
    :param find_event_without_id: Find event in catalog without ID.
    :type s_diff: float
    :param s_diff:
        Maximum difference in seconds between bayesloc origin time and event
        to be considered the same event.
    :type max_bayes_error_km: float
    :param max_bayes_error_km:
        Maximum error between bayesloc origin and event to be matched as the
        same event.
    :type read_all_iterations: bool
    :param read_all_iterations:
        Whether to read all iterations of bayesloc origin realizations (this
        can be slow for large datasets). Default is False.
    :type return_df: bool
    :param return_df: Whether to return the dataframe of bayesloc origins.
    :type return_cat: bool
    :param return_cat: Whether to return the catalog of bayesloc origins.
    :type return_bayesloc_event_ids: bool
    :param return_bayesloc_event_ids: Whether to return the bayesloc event IDs.

    :rtype: :class:`~obspy.core.event.catalog.Catalog`
    :return: Catalog of events with bayesloc origins.
    """
    bayesloc_solutions_added = False
    catalog_empty = False
    if len(cat) == 0:
        catalog_empty = True
    # cat_backup = cat.copy()

    # Skip if all events have bayesloc solutions already
    bayesloc_event_ids = []
    for event in cat:
        bayesloc_event_id = None
        for origin in event.origins:
            try:
                bayesloc_event_id = origin.extra.get(
                    'bayesloc_event_id')['value']
                break
            except (AttributeError, TypeError):
                continue
        bayesloc_event_ids.append(bayesloc_event_id)
    if len(cat) > 0 and all([id is not None for id in bayesloc_event_ids]):
        return cat, pd.DataFrame(), False

    bayes_df = pd.read_csv(bayesloc_origins_ned_stats_file, delimiter=' ')
    bayes_df = bayes_df.sort_values(by='time_mean')

    # This assumes a default starttime of 1970-01-01T00:00:00
    bayes_times = [time.gmtime(value) for value in bayes_df.time_mean.values]
    # Allow a custom epoch time, e.g., one that starts before 1970
    if custom_epoch is not None:
        if isinstance(custom_epoch, UTCDateTime):
            custom_epoch = custom_epoch.datetime
        if not isinstance(custom_epoch, datetime):
            raise TypeError(
                'custom_epoch needs to be of type datetime.datetime or ' +
                'UTCDateTime')
        # 1970-01-01T00:00:00 is the default epoch start time.
        epoch_correction_s = (
            custom_epoch - datetime(1970, 1, 1, 0, 0, 0)).total_seconds()
        bayes_times = [time.gmtime(value + epoch_correction_s)
                       for value in bayes_df.time_mean.values]
    bayes_utctimes = [
        UTCDateTime(bt.tm_year, bt.tm_mon, bt.tm_mday, bt.tm_hour, bt.tm_min,
                    bt.tm_sec + (et - int(et)))
        for bt, et in zip(bayes_times, bayes_df.time_mean.values)]
    bayes_df['utctime'] = bayes_utctimes
    bayes_df['datetime'] = [butc.datetime for butc in bayes_utctimes]

    # remove arrivals to avoid error in conversion to dataframe
    orig = None
    arrivals = []
    for ev in cat:
        try:
            orig = ev.preferred_origin() or ev.origins[0]
            # orig.arrivals = []
        except (AttributeError, KeyError):
            pass
    # Code below not needed I think....
    # cat_df = None
    # cat_df = events_to_df(cat)
    # cat_df['events'] = cat.events
    # # put back arrivals
    # for event, event_backup in zip(cat, cat_backup):
    #     backup_orig = (
    #         event_backup.preferred_origin() or event_backup.origins[0])
    #     try:
    #         orig.arrivals = backup_orig.arrivals
    #     except AttributeError:
    #         pass

    # Code to sort in the new locations from BAYESLOC / Seisan into catalog
    if catalog_empty:
        bayes_df = bayes_df.reset_index()
        for row in bayes_df.itertuples():
            origin_quality = _fill_bayes_origin_quality(None, orig)
            origin_uncertainty = _fill_bayes_origin_uncertainty(row)
            # Keep arrivals from previous origin - those will be updated with
            # Bayesloc-stats later.
            bayes_orig = Origin(
                latitude=row.lat_mean,
                longitude=row.lon_mean,
                depth=row.depth_mean * 1000,
                time=row.datetime,
                latitude_errors=QuantityError(uncertainty=kilometers2degrees(
                    row.north_sd)),
                longitude_errors=QuantityError(uncertainty=kilometers2degrees(
                    row.east_sd)),
                depth_errors=QuantityError(uncertainty=row.depth_sd * 1000),
                time_errors=QuantityError(uncertainty=row.time_sd),
                creation_info=CreationInfo(
                    agency_id=agency_id, author='Bayesloc'),
                arrivals=arrivals,
                quality=origin_quality,
                origin_uncertainty=origin_uncertainty)
            bayes_orig.extra = {
                'bayesloc_event_id': {
                    'value': row.ev_id,
                    'namespace': 'Bayesloc'}}
            # TODO: add full covariance matrix for north-east-depth-time
            #       covariances and other uncertainties
            new_event = Event(
                origins=[bayes_orig],
                creation_info=CreationInfo(
                    agency_id=agency_id, author='Bayesloc'))
            cat.append(new_event)
    else:
        for event in cat:
            cat_orig = (event.preferred_origin() or event.origins[0]).copy()
            arrivals = cat_orig.arrivals.copy()
            nordic_event_id = _get_nordic_event_id(
                event, return_generic_nordic_id=True)
            tmp_cat_df = bayes_df.loc[bayes_df.ev_id == int(nordic_event_id)]
            if find_event_without_id and len(tmp_cat_df) == 0:
                lower_dtime = (cat_orig.time - s_diff)._get_datetime()
                upper_dtime = (cat_orig.time + s_diff)._get_datetime()
                tmp_cat_df = bayes_df.loc[(bayes_df.datetime > lower_dtime) & (
                    bayes_df.datetime < upper_dtime)]

            if len(tmp_cat_df) > 0:
                # Make new origin with all the data
                bayes_orig = Origin()
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
                bayes_orig.creation_info = CreationInfo(
                    agency_id=agency_id, author='Bayesloc')
                bayes_orig.arrivals = arrivals
                bayes_orig.quality = _fill_bayes_origin_quality(
                    event, bayes_orig)
                bayes_orig.origin_uncertainty = _fill_bayes_origin_uncertainty(
                    tmp_cat_df)
                if (tmp_cat_df.time_sd.iloc[0] < s_diff * 3 and
                        tmp_cat_df.depth_sd.iloc[0] < max_bayes_error_km and
                        tmp_cat_df.north_sd.iloc[0] < max_bayes_error_km and
                        tmp_cat_df.east_sd.iloc[0] < max_bayes_error_km):
                    Logger.info(
                        'Added origin solution from Bayesloc for event %s',
                        event.short_str())
                    bayesloc_solutions_added = True
                    # Put bayesloc origin as first in list
                    # new_orig_list = list()
                    # new_orig_list.append(bayes_orig)
                    # new_orig_list.append(event.origins)
                    # event.origins = new_orig_list
                    bayes_orig.extra = {
                        'bayesloc_event_id': {
                            'value': tmp_cat_df.iloc[0].ev_id,
                            'namespace': 'Bayesloc'}}
                    # event.origins.append(bayes_orig)
                    event.origins = [bayes_orig] + event.origins
                    event.preferred_origin_id = bayes_orig.resource_id
                # else:
    if read_all_iterations:
        bayesloc_origins_file = os.path.join(os.path.split(
            bayesloc_origins_ned_stats_file)[0], 'origins.out')
        cat, bayes_df = _cat_add_origin_iterations(
            cat, bayes_df, custom_epoch=custom_epoch,
            bayesloc_origins_file=bayesloc_origins_file)
    return cat, bayes_df, bayesloc_solutions_added


def _cat_add_origin_iterations(cat, bayes_df, bayesloc_origins_file,
                               custom_epoch=None):
    """
    Read information from origins.out file about all iterations of the Bayesloc
    MCMC.

    :type cat: :class:`~obspy.core.event.catalog.Catalog`
    :param cat: Catalog with events to add origin iterations to.
    :type bayes_df: :class:`pandas.DataFrame`
    :param bayes_df: DataFrame with Bayesloc origin information.
    :type bayesloc_origins_file: str
    :param bayesloc_origins_file: Path to Bayesloc origins.out file.
    :type custom_epoch:
        :class:`datetime.datetime` or
        :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param custom_epoch: Custom epoch time to use for Bayesloc origin times.

    :rtype: :class:`~obspy.core.event.catalog.Catalog`
    :return: Catalog with events to add origin iterations to.
    """
    # file header:
    # chain_nr iter_nr ev_id lat lon depth time
    origin_df = pd.read_csv(bayesloc_origins_file, delimiter=' ')
    # origin_df.datetime = origin.time

    # This assumes a default starttime of 1970-01-01T00:00:00
    origin_times = [time.gmtime(value) for value in origin_df.time.values]
    # Allow a custom epoch time, e.g., one that starts before 1970
    if custom_epoch is not None:
        if isinstance(custom_epoch, UTCDateTime):
            custom_epoch = custom_epoch.datetime
        if not isinstance(custom_epoch, datetime):
            raise TypeError(
                'custom_epoch needs to be of type datetime.datetime or ' +
                'UTCDateTime')
        epoch_correction_s = (
            custom_epoch - datetime(1970, 1, 1, 0, 0, 0)).total_seconds()
        origin_times = [time.gmtime(value + epoch_correction_s)
                        for value in origin_df.time.values]
    origin_utctimes = [
        UTCDateTime(bt.tm_year, bt.tm_mon, bt.tm_mday, bt.tm_hour, bt.tm_min,
                    bt.tm_sec + (et - int(et)))
        for bt, et in zip(origin_times, origin_df.time.values)]
    origin_df['utctime'] = origin_utctimes
    origin_df['datetime'] = [butc.datetime for butc in origin_utctimes]
    # Attach dataframe to each Bayesloc origin
    for event in cat:
        added_origin_iterations = False
        for orig in event.origins:
            # Check if there are origin iterations in the dataframe, then
            # select all iterations for this origin and attach as dataframe to
            # origin.
            if (hasattr(orig, 'extra')
                    and 'bayesloc_event_id' in orig.extra.keys()
                    and orig.extra['bayesloc_event_id']['value']
                    in origin_df.ev_id.values):
                orig.extra['origin_iterations'] = (
                    origin_df.loc[origin_df['ev_id'] ==
                                  orig.extra['bayesloc_event_id']['value']])
                added_origin_iterations = True
        if not added_origin_iterations:
            Logger.debug(
                'Could not find origin corresponding to origin iteration '
                'statistics for event %s', event.short_str())
    return cat, bayes_df


def read_bayesloc_arrivals(arrival_file, custom_epoch=None):
    """
    Read Bayesloc arrival file.

    :type arrival_file: str
    :param arrival_file: Path to Bayesloc arrivals.out file.
    :type custom_epoch:
    :param custom_epoch:
        Custom epoch time start to use for Bayesloc origin times.

    :rtype: :class:`pandas.DataFrame`
    :return: DataFrame with Bayesloc arrival information.
    """
    arrival_df = pd.read_csv(arrival_file, delimiter=' ')

    # This assumes a default starttime of 1970-01-01T00:00:00
    arrival_times = [time.gmtime(value) for value in arrival_df.time.values]
    # Allow a custom epoch time, e.g., one that starts before 1970
    if custom_epoch is not None:
        if isinstance(custom_epoch, UTCDateTime):
            custom_epoch = custom_epoch.datetime
        if not isinstance(custom_epoch, datetime):
            raise TypeError(
                'custom_epoch needs to be of type datetime.datetime')
        # custom_epoch = datetime.datetime(1960,1,1,0,0,0)
        epoch_correction_s = (
            custom_epoch - datetime(1970, 1, 1, 0, 0, 0)).total_seconds()
        arrival_times = [time.gmtime(value + epoch_correction_s)
                         for value in arrival_df.time.values]
    bayes_utctimes = [
        UTCDateTime(bt.tm_year, bt.tm_mon, bt.tm_mday, bt.tm_hour, bt.tm_min,
                    bt.tm_sec + (et - int(et)))
        for bt, et in zip(arrival_times, arrival_df.time.values)]
    arrival_df['utctime'] = bayes_utctimes
    arrival_df['datetime'] = [butc._get_datetime() for butc in bayes_utctimes]
    return arrival_df


def read_bayesloc_phases(phases_file):
    """
    Function to read output/phases_freq_stats.out file.

    :type phases_file: str
    :param phases_file: Path to Bayesloc phases_freq_stats.out file.

    :rtype: :class:`pandas.DataFrame`
    :return: DataFrame with Bayesloc phase information.
    """
    phases_df = pd.read_csv(phases_file, delimiter=' ')
    return phases_df


def get_bayesloc_filepath(
        path_or_file, default_output_file='origins_ned_stats.out'):
    """
    Check if a string is either the final path or the directory to a Bayesloc
    run - in the latter case, return the path to the relevant file.

    :type path_or_file: str
    :param path_or_file: Path to Bayesloc output file or directory.
    :type default_output_file: str
    :param default_output_file: Default Bayesloc output file name.

    :rtype: str
    :return: Path to Bayesloc output file.
    """
    if os.path.isdir(path_or_file):
        test_f = os.path.join(path_or_file, default_output_file)
        if os.path.isfile(test_f):
            path_or_file = test_f
        else:
            test_f = os.path.join(path_or_file, 'output', default_output_file)
            if os.path.isfile(test_f):
                path_or_file = test_f
            else:
                msg = ('bayesloc_stats_out_file should be the origin-output ' +
                       ' file or the directory of the Bayesloc run.')
                TypeError(msg)
    return path_or_file


def add_bayesloc_arrivals(arrival_file, catalog=Catalog(), custom_epoch=None):
    """
    Add bayesloc arrivals to a catalog

    :type arrival_file: str
    :param arrival_file: Path to Bayesloc arrivals.out file.
    :type catalog: :class:`obspy.core.event.Catalog`
    :param catalog: Catalog to add Bayesloc arrivals to.
    :type custom_epoch:
        :class:`datetime.datetime` or :class:`obspy.UTCDateTime`
    :param custom_epoch:
        Custom epoch time start to use for Bayesloc origin times.
    :type phases_file: str
    :param phases_file: Path to Bayesloc phases_freq_stats.out file.

    :rtype: :class:`obspy.core.event.Catalog`
    :return: Catalog with Bayesloc arrivals added.
    """
    # Check if any of the events in the catalog can be updated with arrivals,
    # if not return to save time
    bayes_origins_in_cat = False
    for event in catalog:
        bayesloc_event_id = None
        for origin in event.origins:
            try:
                bayesloc_event_id = origin.extra.get(
                    'bayesloc_event_id')['value']
                bayes_origins_in_cat = True
            except (AttributeError, TypeError):
                continue
            continue
    if not bayes_origins_in_cat:
        Logger.info('There are no bayesloc origins in catalog, not trying to '
                    'add Bayesloc arrivals either.')
        return catalog

    # TODO: what to do if there is no phases file yet
    phases_file = get_bayesloc_filepath(
        arrival_file, default_output_file='phases_freq_stats.out')
    arrival_file = get_bayesloc_filepath(
        arrival_file, default_output_file='arrival.dat')

    if len(catalog) == 0:
        raise TypeError(
            'Catalog is empty, use utils.bayesloc.read_bayesloc_origins')
    arrival_df = read_bayesloc_arrivals(
        arrival_file, custom_epoch=custom_epoch)
    if phases_file is not None:
        phases_df = read_bayesloc_phases(phases_file)
        # need to remove event_id column so they don't become duplicated

        if len(phases_df) == len(arrival_df):
            # Actually, input and output are not sorted exactly in the same way
            # output phases_freq_stats is sorted by:
            # event_id, earliest arrival time at station (but all picks at
            # station together), pick times at station

            # Logger.info('Merging input and output arrival dataframes...')

            # # Get a dataframe with unique combinations of ev_id + sta_id
            # unique_evid_station_df = arrival_df[
            #     ['ev_id', 'sta_id']].drop_duplicates()
            # # Group into dataframes that share same ev_id and sta_id
            # event_station_groups = arrival_df.groupby(['ev_id', 'sta_id'])
            # event_station_dfs = [
            #     event_station_groups.get_group((row.ev_id, row.sta_id))
            #     for row in unique_evid_station_df.itertuples()]
            # # Sort each dataframe internally ?? (needed?)
            # for event_station_df in event_station_dfs:
            #     # Set new column to earliest arrival time within dataframe
            #     event_station_df['earliest_event_station_time'] = min(
            #         event_station_df.time)
            # # Concatenate all dataframes
            # sorted_arrival_df = pd.concat(event_station_dfs)

            arrival_df['earliest_event_station_time'] = arrival_df.groupby(
                ['ev_id', 'sta_id'])['time'].transform('min')
            # Sort full dataframe by: ev_id, earliest arrival time per station
            sorted_arrival_df = arrival_df.sort_values(
                ['ev_id', 'earliest_event_station_time', 'sta_id', 'time'],
                ascending=[True, True, True, True], ignore_index=True)
            # Set a new index within each group, which increases with pick-time
            sorted_arrival_df['internal_index'] = sorted_arrival_df.groupby(
                ['ev_id', 'sta_id', 'phase']).cumcount()

            # Write the same type of internal index into phases_df (here we can
            # assume that within each group, phases_df is sorted by pick time)
            phases_df['internal_index'] = phases_df.groupby(
                ['ev_id', 'sta_id', 'phase']).cumcount()
            arrival_df = sorted_arrival_df.merge(
                phases_df, on=['ev_id', 'sta_id', 'phase', 'internal_index'])
            # Assert that the order of ev_id, sta_id_phase matches in both dfs:
            # assert(sorted_arrival_df[['ev_id', 'sta_id', 'phase']].equals(
            #     phases_df[['ev_id', 'sta_id', 'phase']]))
            # phases_df = phases_df.rename(columns={
            #    'ev_id': 'ev_id2', 'sta_id': 'sta_id2', 'phase': 'phase2'})
            # phases_df = phases_df.drop(columns=['ev_id', 'sta_id', 'phase'])

            # Concatenate arrival- and phases-output files:
            # arrival_df = pd.concat([sorted_arrival_df, phases_df], axis=1)
            # Logger.info('Done merging input and output arrival dataframes.')

            # Check where the dataframes still differ
            # diff_df = arrival_df[arrival_df['sta_id'] != arrival_df[
            #     'sta_id2']]
            # df = (
            #     df.assign(key=df.groupby('c2')['c1'].transform('max'))
            #     .sort_values(['key', 'c2', 'c1'], ascending=False,
            #      ignore_index=True).drop(columns=['key'])

            # Maybe this is a way to write it quicker, but doesn't work yet.
            # arrival_df.assign(
            #  earliest_event_station_time=arrival_df.groupby(
            #      ['ev_id', 'sta_id']))['time'].transform('min').sort_values(
            #         ['ev_id', 'earliest_event_station_time'], ascending=True,
            #          ignore_index=True).drop(columns=[
            #     'earliest_event_station_time'])
            # groupby(by='ev_id')
        else:
            Logger.error(
                'There are %s arrivals and %s output stats for phases, can ',
                'not merge stats together...', len(arrival_df), len(phases_df))

    add_picks = False
    # Add picks and arrivals if there are no picks in catalog yet
    if len([p for ev in catalog for p in ev.picks]) == 0:
        add_picks = True

    for event in catalog:
        bayesloc_event_id = None
        for origin in event.origins:
            try:
                bayesloc_event_id = origin.extra.get(
                    'bayesloc_event_id')['value']
                bayesloc_origin = origin
            except (AttributeError, TypeError):
                continue
        if bayesloc_event_id is None:
            Logger.info(
                'Cannot add bayesloc arrivals, there is no Bayesloc solution '
                ' for event %s', event.short_str())
            continue

        # TODO: Add residuals to each arrival
        #       - bayesloc does not output residuals directly
        #       . compute residuals from arrival times and corections for:
        #           station, phase, event, distance with files
        #       - apply residual corrections factors from those files to
        #         residuals: tte_station_shifts_stats.out
        #                    tte_station_phase_shifts_stats.out,
        #                    tte_phase_dist_scales_stats.out
        #                    atep_event_factors_stats.out
        #
        # Select arrivals / picks
        event_arrival_df = arrival_df[arrival_df.ev_id == bayesloc_event_id]
        Logger.info('Adding bayesloc-arrivals to origin for event %s',
                    event.short_str())
        for row in event_arrival_df.itertuples():
            pick_found = False
            # If there are already picks in catalog, try to find the one that
            # matches the bayesloc-arrival.
            if not add_picks:
                pick = [p for p in event.picks
                        if p.waveform_id.station_code == row.sta_id and
                        p.time == row.utctime and p.phase_hint == row.phase]
                if len(pick) == 1:
                    pick_found = True
                    new_pick = pick[0]
                # If picks are called "P1" or "S1", find by time / first letter
                elif (len(pick) == 0 and len(row.phase) > 1 and
                        row.phase[-1] == '1'):
                    pick = [p for p in event.picks
                            if p.waveform_id.station_code == row.sta_id and
                            p.time == row.utctime and p.phase_hint and
                            p.phase_hint[0] == row.phase.removesuffix('1')]
                    if len(pick) == 1:
                        pick_found = True
                        new_pick = pick[0]
                if not pick_found:
                    continue
            # If there are no picks in catalog, add all arrivals as new picks.
            if add_picks or not pick_found:
                new_pick = Pick(
                    waveform_id=WaveformStreamID(station_code=row.sta_id),
                    time=row.utctime,
                    phase_hint=row.phase)
                event.picks.append(new_pick)
            # Add arrivals to origin
            # - time_correction: sum of bayesloc corrections
            # - also need to add: most likely phase hint and probability for
            #   input phase hint
            # First, let's try to find the corresponding arrival in bayesloc-
            # origin that was copied over from previous location (contains data
            # like azimuth, apparent velocity, and residuals that Bayesloc does
            # not know):
            existing_arrival = None
            if pick_found:
                for arrival in bayesloc_origin.arrivals:
                    if arrival.pick_id == new_pick.resource_id:
                        existing_arrival = arrival
            if existing_arrival is not None:
                # Copy arrival and adjust based on Bayesloc output
                new_arrival = existing_arrival
            else:
                # Create new arrival based on all information from Bayesloc
                new_arrival = Arrival(pick_id=new_pick.resource_id,
                                      phase=row.phase,
                                      # time_correction=,  # TODO
                                      # time_residual=,
                                      )
            nsp = 'Bayesloc'
            if not hasattr(new_arrival, 'extra'):
                new_arrival.extra = AttribDict()
            try:
                new_arrival.extra.update(
                    {'original_phase': {'value': row.phase, 'namespace': nsp}})
            except AttributeError:
                pass
            try:
                new_arrival.extra.update(
                    {'prob_as_called':
                        {'value': row.prob_as_called, 'namespace': nsp}})
            except AttributeError:
                pass
            try:
                new_arrival.extra.update(
                    {'most_prob_phase':
                        {'value': row.most_prob_phase, 'namespace': nsp}})
            except AttributeError:
                pass
            # Need to find the field with the probability for the new phasehint
            try:
                new_arrival.extra.update({
                    'prob_as_suggested':
                        {'value': getattr(row, row.most_prob_phase),
                         'namespace': nsp}})
            except AttributeError:
                pass
            try:
                new_arrival.extra.update({'n': {'value': row.n,
                                                'namespace': nsp}})
            except AttributeError:
                pass

            # TODO could add probabilities for all other the other phase hints,
            #      that the pick/arrival could refer to, but that may not be
            #      very useful.
            if new_arrival not in bayesloc_origin.arrivals:
                bayesloc_origin.arrivals.append(new_arrival)
    return catalog


def update_tribe_from_bayesloc(tribe, bayesloc_stats_out_file,
                               custom_epoch=None):
    """
    Update ´eqcorrscan.core.match_filter.Tribe´ with Bayesloc results.

    :type tribe: eqcorrscan.core.match_filter.Tribe
    :param tribe: Tribe to update
    :type bayesloc_stats_out_file: str
    :param bayesloc_stats_out_file: Path to Bayesloc output file
    :type custom_epoch: :class:`obspy.core.utcdatetime.UTCDateTime`
    :param custom_epoch: Custom epoch to use for Bayesloc output

    :rtype: eqcorrscan.core.match_filter.Tribe
    :return: Updated tribe
    """
    cat = Catalog([t.event for t in tribe])
    cat = update_cat_from_bayesloc(cat, bayesloc_stats_out_file,
                                   custom_epoch=custom_epoch)
    for ne, event in enumerate(cat):
        tribe[ne].event = event
    return tribe


def read_bayesloc_events(bayesloc_output_folder, custom_epoch=None):
    """
    Read Bayesloc output files and return a catalog.

    :type bayesloc_output_folder: str
    :param bayesloc_output_folder: Path to Bayesloc output folder
    :type custom_epoch: :class:`obspy.core.utcdatetime.UTCDateTime`
    :param custom_epoch: Custom epoch to use for Bayesloc output

    :rtype: :class:`obspy.core.event.Catalog`
    :return: Catalog with Bayesloc events
    """
    catalog, bayes_df, _ = read_bayesloc_origins(
        bayesloc_origins_ned_stats_file=os.path.join(
            bayesloc_output_folder, 'origins_ned_stats.out'),
        custom_epoch=custom_epoch)
    add_bayesloc_arrivals(
        arrival_file=os.path.join(
            bayesloc_output_folder, 'arrivals.out'),
        catalog=catalog, custom_epoch=custom_epoch)
    return catalog


def update_cat_from_bayesloc(
        cat, bayesloc_stats_out_files, custom_epoch=None, agency_id='',
        find_event_without_id=False, s_diff=3, max_bayes_error_km=100,
        read_all_iterations=False,
        add_arrivals=False, update_phase_hints=False,
        keep_best_fit_pick_only=False, remove_1_suffix=False,
        min_phase_probability=0, **kwargs):
    """
    Update a catalog's locations from a bayesloc-relocation run.

    :type cat: :class:`obspy.core.event.Catalog`
    :param cat: Catalog to update
    :type bayesloc_stats_out_files: str or list
    :param bayesloc_stats_out_files: A list of files containing the
        output from bayesloc_stats_out.py
    :type agency_id: str
    :param agency_id: The agency id to use for the events
    :type custom_epoch: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param custom_epoch: An optional custom epoch to use for the event
        times
    :type s_diff: float
    :param s_diff: The maximum difference in seconds between the event and
        the bayesloc origin time to consider the event to be the same.
    :type max_bayes_error_km: float
    :param max_bayes_error_km: The maximum error in km to consider the
        bayesloc origin to be the same as the event origin.
    :type read_all_iterations: bool
    :param read_all_iterations: If True, read in all bayesloc iterations, if
        False, only read in the final iteration
    :type add_arrivals: bool
    :param add_arrivals: If True, add an arrival for each phase
    :type update_phase_hints: bool
    :param update_phase_hints: If True, update the phase hints
    :type keep_best_fit_pick_only: bool
    :param keep_best_fit_pick_only:
        If True, only keep the arrival with the highest likelihood
    :type remove_1_suffix: bool
    :param remove_1_suffix: If True, remove the '_1' suffix from the
        phase hints
    :type min_phase_probability: float
    :param min_phase_probability: Remove any arrivals with a phase
        probability less than this

    :rtype: :class:`obspy.core.event.Catalog`
    :returns: Updated catalog.
    """
    if isinstance(bayesloc_stats_out_files, str):
        bayesloc_stats_out_files = [bayesloc_stats_out_files]

    # Loop through multiple bayesloc output folders
    for bayesloc_stats_out_file in bayesloc_stats_out_files:
        bayesloc_folder = bayesloc_stats_out_file
        bayesloc_stats_out_file = get_bayesloc_filepath(
            bayesloc_stats_out_file,
            default_output_file='origins_ned_stats.out')

        cat, _, bayesloc_solutions_added = read_bayesloc_origins(
            bayesloc_stats_out_file, cat=cat, custom_epoch=custom_epoch,
            agency_id=agency_id, find_event_without_id=find_event_without_id,
            s_diff=s_diff, max_bayes_error_km=max_bayes_error_km,
            read_all_iterations=read_all_iterations)
        if not bayesloc_solutions_added:
            continue
        if add_arrivals:
            cat = add_bayesloc_arrivals(
                bayesloc_folder, catalog=cat, custom_epoch=custom_epoch)
        if update_phase_hints:
            cat = _update_bayesloc_phase_hints(
                cat, remove_1_suffix=remove_1_suffix)
        # Keep only the best fitting pick only when there are multiple
        # picks with the same phase-hint for same event at station
        # Can be selected based on highest probability OR smallest residual
        if keep_best_fit_pick_only:
            cat = _select_bestfit_bayesloc_picks(
                cat, min_phase_probability=min_phase_probability)
    return cat


def _select_bestfit_bayesloc_picks(cat, min_phase_probability=0):
    """
    Select the best fitting pick for each phase-hint at each station.

    :type cat: :class:`obspy.core.event.Catalog`
    :param cat: Catalog to update
    :type min_phase_probability: float
    :param min_phase_probability: Remove any arrivals with a phase probability
        less than this (default: 0)
    :type remove_1_suffix: bool
    :param remove_1_suffix: If True, remove the '_1' suffix from the phase

    :rtype: :class:`obspy.core.event.Catalog`
    :returns: Updated catalog.
    """
    for event in cat:
        bayesloc_event_id = None
        bayesloc_origin = None
        for origin in event.origins:
            try:
                bayesloc_event_id = origin.extra.get(
                    'bayesloc_event_id')['value']
                break
            except (AttributeError, TypeError):
                continue
        if bayesloc_event_id is not None:
            Logger.info(
                'Event %s: Sorting out duplicate picks: keeping only those '
                'picks / arrivals that best fit.', event.short_str())
            bayesloc_origin = origin
        else:
            continue
        uniq_bayes_phases = list(set([
            (arrival.pick_id.get_referred_object().waveform_id.station_code,
             arrival.phase)
            for arrival in bayesloc_origin.arrivals
            if arrival.pick_id.get_referred_object() is not None]))
        for station, phase in uniq_bayes_phases:
            if station is None:
                continue
            similar_arrivals = [
                arrival for arrival in bayesloc_origin.arrivals
                if arrival.pick_id.get_referred_object() is not None and
                phase == arrival.phase and station ==
                arrival.pick_id.get_referred_object().waveform_id.station_code]
            # Can save some time here if there is only 1 pick:
            if len(similar_arrivals) == 1 and min_phase_probability == 0:
                continue
            similar_picks = [
                arrival.pick_id.get_referred_object()
                for arrival in similar_arrivals
                if arrival.pick_id.get_referred_object() is not None]
            phase_probabilities = [
                arrival.extra.prob_as_called.value
                for arrival in similar_arrivals
                if hasattr(arrival, 'extra') and
                hasattr(arrival.extra, 'prob_as_called')]
            if len(phase_probabilities) == 0:
                continue
            max_phase_probability = max(phase_probabilities)
            max_phase_prob_idx = np.argmax(phase_probabilities)
            # Keep only the best one
            if max_phase_probability > min_phase_probability:
                Logger.debug(
                    'Event %s: There are %s picks for %s for station %s, '
                    'keeping only the best fitting pick (best: %s, worst: %s) '
                    'above probability %s.', event.short_str(),
                    len(similar_picks), phase, station, max_phase_probability,
                    min(phase_probabilities), min_phase_probability)
            else:
                Logger.debug(
                    'Event %s: There are %s picks for %s for station %s, '
                    'but probably (best: %s, worst: %s) none are properly '
                    'assigned with probability > %s.',
                    event.short_str(), len(similar_picks), phase, station,
                    max_phase_probability, min(phase_probabilities),
                    min_phase_probability)
            for j, (arrival, pick) in enumerate(
                    zip(similar_arrivals, similar_picks)):
                if (j == max_phase_prob_idx and
                        max_phase_probability > min_phase_probability):
                    continue
                bayesloc_origin.arrivals.remove(arrival)
                event.picks.remove(pick)
    return cat


def _update_bayesloc_phase_hints(cat, remove_1_suffix=False):
    """
    Update arrivals and picks with phase hints indicated by Bayesloc.

    :type cat: :class:`obspy.core.event.Catalog`
    :param cat: Catalog to update
    :type remove_1_suffix: bool
    :param remove_1_suffix: If True, remove the '_1' suffix from the phase

    :rtype: :class:`obspy.core.event.Catalog`
    :returns: Updated catalog.
    """
    for event in cat:
        bayesloc_event_id = None
        bayesloc_origin = None
        for origin in event.origins:
            try:
                bayesloc_event_id = origin.extra.get(
                    'bayesloc_event_id')['value']
                break
            except (AttributeError, TypeError):
                continue
        if bayesloc_event_id is not None:
            Logger.info('Updating phase hints for event %s', event.short_str())
            bayesloc_origin = origin
            for arrival in bayesloc_origin.arrivals:
                pick = arrival.pick_id.get_referred_object()
                if hasattr(arrival, 'extra'):
                    if (arrival.extra.original_phase.value !=
                            arrival.extra.most_prob_phase.value):
                        # Rename pick phase according to Bayesloc
                        if arrival.pick_id.get_referred_object() is None:
                            continue
                        try:
                            is_different_phase_type = (
                                arrival.phase[0] ==
                                arrival.extra.most_prob_phase.value[0])
                            if not is_different_phase_type:
                                Logger.debug(
                                    '%s, %s: Output phase from bayesloc '
                                    'changes phase hint from %s to %s',
                                    event.short_str(), str(pick.waveform_id),
                                    arrival.phase,
                                    arrival.extra.most_prob_phase.value)
                        except IndexError:
                            pass
                        arrival.phase = arrival.extra.most_prob_phase.value
                        arrival.extra.prob_as_called.value = (
                            arrival.extra.prob_as_suggested.value)
                        pick.phase_hint = arrival.extra.most_prob_phase.value
                    # original_phase prob_as_called most_prob_phase
                # When first arrival phase is called like "P1" or "S1", may
                # need to rename:
                if (remove_1_suffix and len(arrival.phase) > 1 and
                        arrival.phase[-1] == '1'):
                    arrival.phase.removesuffix('1')
                if pick is not None and pick.phase_hint != arrival.phase:
                    pick.phase_hint = arrival.phase
    return cat


def write_arrival_file(
        cat, path='.', split_into_months=False,
        custom_epoch=datetime(1970, 1, 1, 0, 0, 0),
        abs_min_n_stations=0, abs_min_n_phases=0, allowed_phases=[
            'P', 'S', 'Pn', 'Pg', 'Sn', 'Sg', 'P1', 'S1', 'Pb', 'Sb'],
        known_station_names=[],
        min_n_station_sites=0, minimum_phases_per_station=0,
        inv=Inventory(), compare_station_to_inventory=False,
        min_n_teleseismic_stations=0, include_good_regional_events=True,
        min_n_regional_stations=0, min_n_obs_stations=0,
        min_latitude=None, max_latitude=None,
        min_longitude=None, max_longitude=None,
        fix_depth=None, default_depth=10, def_dep_error_m=15000,
        default_lat=None, default_lon=None, def_hor_error_deg=None,
        def_time_error_s=5, obs_prefixes=()):
    """
    Function to write bayesloc arrival file.

    :type cat: :class:`obspy.core.event.Catalog`
    :param cat: Catalog to write
    :type path: str
    :param path: Path to write arrival file to
    :type split_into_months: bool
    :.param split_into_months:
        If True, split catalog into 12 separate arrival files, with the
        big events overlapping between files, but otherwise only events from
        each month printed into each file (this is useful to chunk up a big
        bayesloc relocation setup)
    :type custom_epoch: :class:`datetime.datetime`
    :param custom_epoch:
        Custom epoch start time to use for bayesloc files (bayesloc only works
        with seconds since epoch start, not with time objects)
    :type abs_min_n_stations: int
    :param abs_min_n_stations: Absolute minimum number of stations
    :type abs_min_n_phases: int
    :param abs_min_n_phases: Absolute minimum number of phases
    :type allowed_phases: list
    :param allowed_phases: List of allowed phases
    :type known_station_names: list
    :param known_station_names: List of known station names
    :type min_n_station_sites: int
    :param min_n_station_sites: Minimum number of station sites
    :type minimum_phases_per_station: int
    :param minimum_phases_per_station: Minimum number of phases per station
    :type inv: :class:`obspy.core.inventory.inventory.Inventory`
    :param inv: Inventory to use for station information
    :type compare_station_to_inventory: bool
    :param compare_station_to_inventory: If True, compare station names to
        inventory
    :type min_n_teleseismic_stations: int
    :param min_n_teleseismic_stations:
        Minimum number of teleseismic stations for event to be included
        as a teleseismic event.
    :type include_good_regional_events: bool
    :param include_good_regional_events:
        If True, include good regional events as "anchor" events that
        overlap between months / files.
    :type min_n_regional_stations: int
    :param min_n_regional_stations: Minimum number of regional stations
    :type min_latitude: float
    :param min_latitude: Minimum latitude for event to be included in file
    :type max_latitude: float
    :param max_latitude: Maximum latitude for event to be included in file
    :type min_longitude: float
    :param min_longitude: Minimum longitude for event to be included in file
    :type max_longitude: float
    :param max_longitude: Maximum longitude for event to be included in file
    :type fix_depth: float
    :param fix_depth: Fix depth to this value, if None, use depth from event
    :type default_depth: float
    :param default_depth: Default depth to use if no depth is given
    :type def_dep_error_m: float
    :param def_dep_error_m:
        Default depth error to use if no depth error is given
    :type default_lat: float
    :param default_lat: Default latitude to use if no latitude is given
    :type default_lon: float
    :param default_lon: Default longitude to use if no longitude is given
    :type def_hor_error_deg: float
    :param def_hor_error_deg: Default horizontal error to use if no error is
        given
    :type def_time_error_s: float
    :param def_time_error_s: Default time error to use if no error is given
    :type obs_prefixes: list
    :param obs_prefixes: List of prefixes to use for ocean bottom seismometer
        stations (these can be treated as more important than others)

    :rtype: list
    :return: List of arrival files written
    """
    Logger.info('Step 2.6 Collecting arrivals that fulfill criteria')
    phases_per_station = Counter([pick.waveform_id.station_code
                                  for event in cat
                                  for pick in event.picks])

    # Write one output file for all events in each of the 12 months of all yrs
    if split_into_months:
        months_list = [[mo] for mo in np.arange(1, 13)]
        Logger.info(
            'Splitting input files into %s unqiue bayesloc runs / folders',
            len(months_list))
    else:
        months_list = [list(np.arange(1, 13))]

    for im, months in enumerate(months_list):
        im = im + 1
        ev_ids = list()
        arrivals = []
        priors = []
        folder_suffix = ""
        if split_into_months:
            folder_suffix = '_' + '{run:02d}'.format(run=im)
        run_folder = path + folder_suffix
        for j, event in enumerate(cat):
            orig = event.preferred_origin() or event.origins[0]
            if min_latitude and orig.latitude and orig.latitude < min_latitude:
                continue
            if max_latitude and orig.latitude and orig.latitude > max_latitude:
                continue
            if (min_longitude and orig.longitude
                    and orig.longitude < min_longitude):
                continue
            if (max_longitude and orig.longitude
                    and orig.longitude > max_longitude):
                continue
            # Set minimum number of stations and phases
            unique_stations_list = list(set([p.waveform_id.station_code
                                            for p in event.picks]))
            n_stations = len(unique_stations_list)
            n_phases = len(list(set([(p.waveform_id.station_code, p.phase_hint)
                                    for p in event.picks])))
            n_station_sites = len(
                list(set(get_station_sites(unique_stations_list))))

            sta_dist_tuples = []
            for arrival in orig.arrivals:
                pick = arrival.pick_id.get_referred_object()
                if pick is None:
                    continue
                sta_dist_tuples.append((pick.waveform_id.station_code,
                                        arrival.distance))
            n_teleseismic_stations = len(list(set(
                sdt[0] for sdt in sta_dist_tuples if sdt[1] and sdt[1] > 15)))
            # distances = event_distances[j]
            # n_teleseismic_stations = len(list(set(
            #     [p.waveform_id.station_code for jp, p in
            #       enumerate(event.picks)
            #     if distances[jp] is not None and distances[jp] > 15])))

            if obs_prefixes:
                n_obs_stations = len(list(set(
                    [p.waveform_id.station_code for p in event.picks
                     if p.waveform_id.station_code.startswith(obs_prefixes)])))

            # Absolute minimum number of phases and stations - else DO NOT use
            if (n_stations < abs_min_n_stations or n_phases < abs_min_n_phases
                    or n_station_sites < min_n_station_sites):
                Logger.info(
                    'Not enough stations / sites / phases: %s stations, %s '
                    'sites  (%s teleseismic), %s phases', str(n_stations),
                    str(n_station_sites), str(n_teleseismic_stations),
                    str(n_phases))
                continue

            has_local_solution = False
            if orig.creation_info.agency_id in ['AWI']:
                has_local_solution = True
            # First round of relocations: select only events with at least 5?
            # teleseismic arrivals to pin down residuals better
            if (n_teleseismic_stations >= min_n_teleseismic_stations or
                    n_obs_stations >= min_n_obs_stations):
                pass
            elif (include_good_regional_events and
                    n_stations >= min_n_regional_stations):
                pass
            elif has_local_solution:
                pass
            else:
                # When splitting into one Bayesloc run across all equal months,
                # then keep even worse observed events
                # if not strict or (split_into_months and
                #       orig.time.month in months):
                if split_into_months and orig.time.month in months:
                    pass
                else:
                    continue

            orig = event.preferred_origin()
            evid = _get_nordic_event_id(
                event, return_generic_nordic_id=True)
            # Make sure no event id is duplicated
            while evid in ev_ids:
                evid = evid + 1
            ev_ids.append(evid)
            for pick in event.picks:
                if pick.phase_hint in allowed_phases:
                    # fix P- and S- phase names to Pg / Pn / Sg / Sn?
                    # Sort out picks from stations where there's too few picks:
                    if (phases_per_station[pick.waveform_id.station_code]
                            < minimum_phases_per_station):
                        Logger.error(
                            'Station %s has not enough phases (%s < %s), '
                            'skipping pick', pick.waveform_id.station_code,
                            phases_per_station[pick.waveform_id.station_code],
                            minimum_phases_per_station)
                        continue
                    if compare_station_to_inventory:
                        sel_inv = inv.select(
                            station=pick.waveform_id.station_code)
                        if len(sel_inv) == 0:
                            Logger.error(
                                'Station %s not in inventory, skipping pick',
                                pick.waveform_id.station_code)
                            continue
                        # Check for case mismatch
                        if (sel_inv.networks[0].stations[0].code
                                != pick.waveform_id.station_code):
                            continue
                    epoch_time = (pick.time._get_datetime() - custom_epoch
                                  ).total_seconds()
                    arrivals.append(
                        (evid, pick.waveform_id.station_code.upper(),
                         pick.phase_hint, epoch_time))
            # Get info for PRIOR file
            orig, priors = _select_best_origin(
                event, evid, priors, fix_depth=fix_depth,
                default_depth=default_depth, def_dep_error_m=def_dep_error_m,
                default_lat=default_lat, default_lon=default_lon,
                def_hor_error_deg=def_hor_error_deg,
                def_time_error_s=def_time_error_s, default_start=custom_epoch)

        Logger.info('Step 3.2 Writing %s arrivals to arrival file',
                    str(len(arrivals)))
        arrfile = 'arrival.dat'

        if not os.path.exists(run_folder):
            os.makedirs(run_folder)
        arrivalfile = os.path.join(run_folder, arrfile)
        f = open(arrivalfile, 'wt')
        f.write('ev_id sta_id phase time\n')
        for arrival in arrivals:
            eid, scode, phase, time = arrival
            # Only write out arrivals for known stations - bayesloc may
            # otherwise crash
            if known_station_names and scode not in known_station_names:
                continue
            f.write('%i %s %s %.3f\n' % (eid, scode, phase, time))
        f.close()

        Logger.info('Step 3.3 Writing %s origins to prior file',
                    str(len(priors)))
        prifile = 'prior.dat'
        priorfile = os.path.join(run_folder, prifile)
        f = open(priorfile, 'wt')
        f.write('ev_id lat_mean lon_mean dist_sd depth_mean depth_sd ' +
                'time_mean time_sd\n')
        for prior in priors:
            evid, plat, plon, dist_sd, pdepth, dep_sd, ptime, time_sd = prior
            # f.write('%i %9.4f %9.4f %4.1f %6.1f %4.1f %16.3f %4.1f\n' %
            #      (evid, plat, plon, dist_sd, pdepth, dep_sd, ptime, time_sd))
            f.write('%i %9.4f %9.4f %7.1f %6.1f %7.1f %16.3f %6.1f\n' %
                    (evid, plat, plon, dist_sd, pdepth, dep_sd, ptime, time_sd)
                    )
        f.close()


def write_station(inventory, station0_file=None, stations_dat_file=None,
                  station_df=None, default_elev=0,
                  stations=[], filename="station.dat", write_only_once=True):
    """
    Write a GrowClust formatted station file.

    :type inventory: obspy.core.Inventory
    :param inventory:
        Inventory of stations to write - should include channels if
        use_elevation=True to incorporate channel depths.
    :type station0_file: str
    :param station0_file: Path to station0 file to read station names from.
    :type stations_dat_file: str
    :param stations_dat_file: Path to stations.dat file to read station names
    :type station_df: pandas.DataFrame
    :param station_df: DataFrame of station names to read station names from.
    :type default_elev: float
    :param default_elev: Default elevation to use if not in inventory.
    :type stations: list
    :param stations: List of stations to write to file.
    :type filename: str
    :param filename: File to write stations to.
    :type write_only_once: bool
    :param write_only_once: Only write stations to file once.

    :rtype: list
    :return: List of station strings.
    """
    station_strings = []
    # formatter = "{sta:<5s}{lat:>9.4f}{lon:>10.4f}{elev:>10.3f}"
    # formatter = "{sta:<s} {lat:>.4f} {lon:>.4f} {elev:>.3f}"
    formatter = "{sta:5s} {lat:9.4f} {lon:10.4f} {elev:7.3f}"
    # %s %.4f %.4f %.3f
    known_station_names = []
    for network in inventory:
        for station in network:
            parts = dict(sta=station.code, lat=station.latitude,
                         lon=station.longitude, elev=station.elevation / 1000)
            # if use_elevation:
            #     channel_depths = {chan.depth for chan in station}
            #     if len(channel_depths) == 0:
            #         Logger.warning("No channels provided, using 0 depth.")
            #         depth = 0.0
            #     else:
            #         depth = channel_depths.pop()
            #     if len(channel_depths) > 1:
            #         Logger.warning(
            #             f"Multiple depths for {station.code}, using {depth}")
            #     parts.update(dict(elev=station.elevation - depth))
            if write_only_once:  # Write only one location per station
                if station.code in known_station_names:
                    continue
            # Don't write station if no picks in catalog
            if station.code not in stations:
                continue
            station_strings.append(formatter.format(**parts))
            known_station_names.append(station.code)
    if station0_file is not None:
        if station0_file:
            stalist_1 = readSTATION0(station0_file, stations)
            for line in stalist_1:
                if write_only_once:
                    if line[0] in known_station_names:
                        continue
                parts = dict(sta=line[0], lat=line[1], lon=line[2],
                             elev=line[3] / 1000)
                station_strings.append(formatter.format(**parts))
                known_station_names.append(line[0])
        if stations_dat_file:
            stalist_2 = read_stations_dat(stations_dat_file, stations)
            for line in stalist_2:
                if write_only_once:
                    if line[0] in known_station_names:
                        continue
                parts = dict(sta=line[0], lat=line[1], lon=line[2],
                             elev=line[3] / 1000)
                station_strings.append(formatter.format(**parts))
                known_station_names.append(line[0])
    if station_df is not None:
        known_station_names += list(station_df.station)
        for row in station_df.iterrows():
            if 'elev' in row[1].keys():
                elev = row[1].elev
            elif 'elevation' in row[1].keys():
                elev = row[1].elevation
            else:
                elev = default_elev
            parts = dict(sta=row[1].station[-5:], lat=row[1].lat,
                         lon=row[1].lon, elev=elev/1000)
            station_strings.append(formatter.format(**parts))
    with open(filename, "w") as f:
        f.write('sta_id lat lon elev\n')
        f.write("\n".join(station_strings))
    return known_station_names


def read_stations_dat(path, stations):
    """
    function to read ISF (International seismic format) station.dat-file

    :type path: str
    :param path: path to station.dat-file
    :type stations: list
    :param stations: list of stations to read

    :rtype: list
    :return: list of station strings
    """
    if path is None:
        return None
    stalist = []
    try:
        f = open(path + '/stations.dat', 'r')
    except (FileNotFoundError, NotADirectoryError):
        f = open(path, 'r')
    for line in f:
        if line[0:5].strip() in stations:
            station = line[0:5].strip()
            # Format is either ddmm.mmS/N or ddmm(.)mmmS/N
            lat = line[6:14].replace(' ', '0')
            if lat[-1] == 'S':
                NS = -1
            else:
                NS = 1
            lat = (int(lat[0:2]) + float(lat[2:4]) / 60
                   + float(lat[4:-1]) / (60 * 60)) * NS
            lon = line[15:25].replace(' ', '0')
            if lon[-1] == 'W':
                EW = -1
            else:
                EW = 1
            if lon[5] == '.':
                lon = (int(lon[0:3]) + float(lon[3:-1]) / 60) * EW
            else:
                lon = (int(lon[0:3]) + float(lon[3:5]) / 60
                       + float(lon[5:-1]) / (60 * 60)) * EW
            try:
                elev = float(line[25:32].strip())
            except ValueError:  # Elevation could be empty
                elev = 0.0
            site = line[32:].strip()
            stalist.append((station, lat, lon, elev, site))
    f.close()
    f = open('station0.dat', 'w')
    for sta in stalist:
        line = ''.join([sta[0].ljust(5), _cc_round(sta[1], 4).ljust(10),
                        _cc_round(sta[2], 4).ljust(10),
                        _cc_round(sta[3] / 1000, 4).rjust(7), '\n'])
        f.write(line)
    f.close()
    return stalist


# %% TEST TEST TEST
if __name__ == "__main__":
    import logging
    Logger = logging.getLogger(__name__)
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")

    from obspy.io.nordic.core import read_nordic
    catalog = read_nordic(
        '/home/seismo/WOR/felix/R/SEI/REA/INTEU/2020/12/14-1935-58R.S202012')
    # '/home/seismo/WOR/felix/R/SEI/REA/INTEU/2021/07/05-0423-41R.S202107')
    # 'a8598f5d-d25d-4b69-8547-298589d29bc3'

    Logger.info('Updating catalog from bayesloc solutions')
    bayesloc_path = [
        '/home/felix/Documents2/Ridge/Relocation/Bayesloc/Ridge_INTEU_09a_oceanic_09',
        '/home/felix/Documents2/Ridge/Relocation/Bayesloc/Ridge_INTEU_09a_oceanic_10b']

    update_cat_from_bayesloc(
        catalog, bayesloc_path, custom_epoch=UTCDateTime(1960, 1, 1, 0, 0, 0),
        agency_id='BER', find_event_without_id=True, s_diff=3,
        max_bayes_error_km=100, add_arrivals=True, update_phase_hints=True,
        keep_best_fit_pick_only=True, remove_1_suffix=True,
        min_phase_probability=0)


# %%
