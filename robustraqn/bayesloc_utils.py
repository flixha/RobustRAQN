
# %%
import os
from xml.dom.minidom import Attr
from attr import Attribute
import numpy as np
import matplotlib

import pandas as pd
import numpy as np
import time

import textwrap
from datetime import datetime
from joblib.parallel import Parallel, delayed

from obspy.core.event import (
    Catalog, Event, QuantityError, OriginQuality, OriginUncertainty)
from obspy.geodetics.base import degrees2kilometers, kilometers2degrees
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.event import Origin, Pick, Arrival
from obspy.core.event.base import CreationInfo, WaveformStreamID
from obspy.taup import TauPyModel
from obspy.core.util.attribdict import AttribDict
from obspy.io.nordic.ellipse import Ellipse

from obsplus.events.validate import attach_all_resource_ids
from obsplus import events_to_df
from obsplus.utils.time import to_datetime64
from obsplus.constants import EVENT_DTYPES, TIME_COLUMNS
from obsplus.structures.dfextractor import DataFrameExtractor

import pandas as pd

import logging
Logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")


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
        if line[1:6].strip() in stations:
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
                degrees = lat[0:2]
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
                # lat = (int(degrees) + float(minutes + '.' + decimal_minutes) /
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
                lon = (int(lon[0:3]) + float(lon[3:5] + '.' + lon[5:-1]) /
                      60) * EW
                # lon = (int(degrees) + float(minutes + '.' + decimal_minutes) /
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
    :type use_elevation: bool
    :param use_elevation: Whether to write elevations (requires hypoDD >= 2)
    :type filename: str
    :param filename: File to write stations to.
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
    priors.append((
        evid, orig.latitude or default_lat, orig.longitude or default_lon,
        # orig.origin_uncertainty.max_horizontal_uncertainty / 1000 or 30,
        degrees2kilometers(
            ((orig.latitude_errors.uncertainty or def_hor_error_deg) +
             (orig.longitude_errors.uncertainty or def_hor_error_deg)) / 2),
        #    (orig.longitude_errors.uncertainty +
        #     orig.longitude_errors.uncertainty) / 2
        (orig.depth or default_depth) / 1000,
        (orig.depth_errors.uncertainty or def_dep_error_m) / 1000,
        orig_epoch_time, orig.time_errors.uncertainty or def_time_error_s))
    return orig, priors


def _get_traveltime(model, degree, depth, phase):
    """
    """
    if phase in ['P', 'S']:  # For short phase name, get quickest phase
        phase_list = [phase, phase.lower(), phase + 'g', phase + 'n']
        try:
            arrivals = model.get_travel_times(depth, degree,
                                              phase_list=phase_list)
            return(min([arrival.time for arrival in arrivals]))
        except (IndexError, ValueError):
            return(np.nan)
    else:
        try:
            arrival = model.get_travel_times(depth, degree, phase_list=[phase])
            return(arrival[0].time)
        except IndexError:
            # if there's no path, try with an upgoing path Pg --> P / p
            try:
                arrival = model.get_travel_times(
                    depth, degree, phase_list=[phase[0]])
                return(arrival[0].time)
            except IndexError:
                try:
                    arrival = model.get_travel_times(
                        depth, degree, phase_list=[phase[0].lower()])
                    return(arrival[0].time)
                except IndexError:
                    return(np.nan)


def _get_nordic_event_id(event):
    """
    Get the event id that is mentioned in the most recent comment in event read
    from Nordic file.
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
        event_id = int(
            [com.text.split('ID:')[-1].strip('SdLRD ')
            for com in comments if 'ID:' in com.text][0])
    return event_id


def write_ttimes_files(
        model, mod_name=None,
        phase_list=['P', 'Pg', 'Pn', 'S', 'Sg', 'Sn', 'P1', 'S1'],
        min_depth=0, max_depth=40, depth_step=2, min_degrees=0, max_degrees=10,
        degree_step=0.2, outpath='.', print_first_arriving_PS=True,
        parallel=False, cores=40):
    """
    """
    if not isinstance(model, TauPyModel):
        mod_name = str(model.split('/')[-1].split('.')[0])
        model = TauPyModel(model)
    else:
        if mod_name is not None:
            mod_name = mod_name
        else:
            mod_name = 'def_'
    # model = TauModel.from_file('NNSN1D_plusAK135.npz')
    # a list of epicentral distances without a travel time, and a flag:
    # notimes = []
    #plotted = False
    
    # calculate the arrival times and plot vs. epicentral distance:
    depths = np.arange(min_depth, max_depth, depth_step)
    # degrees = np.linspace(min_degrees, max_degrees, npoints)
    degrees = np.arange(min_degrees, max_degrees, degree_step)
    for phase in phase_list:
        ttfile = mod_name + '.' + phase
        ttfile = os.path.expanduser(os.path.join(outpath, ttfile))
        Logger.info('Writing traveltime table to file %s', ttfile)
        f = open(ttfile, 'wt')
        f.write((
            '# travel-time table for velocityModel=%s and phase=%s. Generated '
            + 'with Obspy.taup and write_ttimes_files.\n') % (mod_name, phase))
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
                    model, degree, depth, comp_phase) for degree in degrees)
            else:
                ttimes = [_get_traveltime(model, degree, depth, comp_phase)
                          for degree in degrees]

            # make sure P and S are the first arrivals -check if Pn or Sn are
            # quicker at specific depth/distance
            if print_first_arriving_PS:
                if phase in ['P1', 'S1']:
                    n_ttimes = Parallel(n_jobs=cores)(delayed(_get_traveltime)(
                        model, degree, depth, comp_phase + 'n')
                                                      for degree in degrees)
                    b_ttimes = Parallel(n_jobs=cores)(delayed(_get_traveltime)(
                        model, degree, depth, comp_phase + 'b')
                                                      for degree in degrees)
                    ttimes = [np.nanmin([ttimes[j], n_ttimes[j], b_ttimes[j]])
                              for j, ttime in enumerate(ttimes)]

            # ttimes = []
            # for degree in degrees:
            #     try:
            #         arrival = model.get_travel_times(
            #             depth, degree,phase_list=[phase])
            #         # convert time from minutes to seconds # NO?
            #         ttimes.append(arrival[0].time)
            #     except IndexError:
            #         # if there's no path, try with an upgoing path (e.g., make
            #         # 'Pg' to 'p' to complete travetimetable)
            #         try:
            #             arrival = model.get_travel_times(
            #                 depth, degree, phase_list=[phase[0].lower()])
            #         except IndexError:
            #             ttimes.append(np.nan)
                # ax = arrivals.plot_times(phase_list=phase_list, show=True,
                #                          ax=ax, plot_all=True)
            f.write('Travel time at depth = %8.2f km.\n' % depth)
            for ttime in ttimes:
                f.write('%11.4f\n' % ttime)
        f.close()


def _fill_bayes_origin_quality(event, orig):
    """
    """
    if orig is not None and orig.quality is not None:
        orig_quality = orig.quality
    else:
        orig_quality = OriginQuality(
            used_station_count=len(set(
                [pick.waveform_id.station_code for pick in event.picks
                    if pick.waveform_id is not None])))
    return orig_quality


def _fill_bayes_origin_uncertainty(row):
    """
    Uncertainty ellipse from Bayesloc output:
    row: pandas dataframe Series or dataframe row
    """
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    cov = [[row.east_sd, row.north_east_cor],
            [row.north_east_cor, row.north_sd]]
    ellipse = Ellipse.from_cov(cov, center=(0, 0))
    origin_uncertainty = None
    if ellipse:
        origin_uncertainty = OriginUncertainty(
            max_horizontal_uncertainty=ellipse.a * 1000.,
            min_horizontal_uncertainty=ellipse.b * 1000.,
            azimuth_max_horizontal_uncertainty=ellipse.theta,
            preferred_description="uncertainty ellipse")
    return origin_uncertainty


def read_bayesloc_origins(
        bayesloc_origins_ned_stats_file, cat=Catalog(),
        custom_epoch=None, agency_id='', find_event_without_id=False,
        s_diff=3, max_bayes_error_km=100):
    """
    Read bayesloc origins into Catalog and pandas datafrane
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
            except AttributeError:
                continue
        bayesloc_event_ids.append(bayesloc_event_id)
    if all([id is not None for id in bayesloc_event_ids]):
        return cat, pd.DataFrame(), False

    bayes_df = pd.read_csv(bayesloc_origins_ned_stats_file, delimiter=' ')
    bayes_df = bayes_df.sort_values(by='time_mean')

    # This assumes a default starttime of 1970-01-01T00:00:00
    bayes_times = [time.gmtime(value) for value in bayes_df.time_mean.values]
    # Allow a custom epoch time, e.g., one that starts before 1970
    if custom_epoch is not None:
        # if not isinstance(custom_epoch, np.datetime64):
        # raise TypeError(
        #         'custom_epoch needs to be of type numpy.datetime64')
        # np.datetime64('1960-01-01T00:00:00') - 
        #     np.datetime64('1970-01-01T00:00:00')
        if isinstance(custom_epoch, UTCDateTime):
            custom_epoch = custom_epoch.datetime
        if not isinstance(custom_epoch, datetime):
            raise TypeError(
                'custom_epoch needs to be of type datetime.datetime or ' +
                'UTCDateTime')
        # custom_epoch = datetime.datetime(1960,1,1,0,0,0)
        epoch_correction_s = (
            custom_epoch - datetime(1970,1,1,0,0,0)).total_seconds()
        bayes_times = [time.gmtime(value + epoch_correction_s)
                       for value in bayes_df.time_mean.values]
    bayes_utctimes = [
        UTCDateTime(bt.tm_year, bt.tm_mon, bt.tm_mday, bt.tm_hour, bt.tm_min,
                    bt.tm_sec + (et - int(et)))
        for bt, et in zip(bayes_times, bayes_df.time_mean.values)]
    bayes_df['utctime'] = bayes_utctimes
    bayes_df['datetime'] = [butc.datetime for butc in bayes_utctimes]

    # cat_SgLoc = read_seisan_database('Sfiles_MAD10_Saga_02_Sg')
    # cat_Sgloc_df = events_to_df(cat_SgLoc)
    # cat_Sgloc_df['events'] = cat_SgLoc.events

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
                quality=orig_quality,
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
            nordic_event_id = _get_nordic_event_id(event)
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
                bayes_orig.creation_info=CreationInfo(
                    agency_id=agency_id, author='Bayesloc')
                bayes_orig.arrivals = arrivals
                origin_quality = _fill_bayes_origin_quality(event, bayes_orig)
                origin_uncertainty = _fill_bayes_origin_uncertainty(tmp_cat_df)
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
    return cat, bayes_df, bayesloc_solutions_added


def read_bayesloc_arrivals(arrival_file, custom_epoch=None):
    """
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
            custom_epoch - datetime(1970,1,1,0,0,0)).total_seconds()
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
    """
    phases_df = pd.read_csv(phases_file, delimiter=' ')
    return phases_df


def get_bayesloc_filepath(
        path_or_file, default_output_file='origins_ned_stats.out'):
    """
    Check if a string is either the final path or the directory to a Bayesloc
    run - in the latter case, return the path to the relevant file.
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
            except AttributeError:
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
            'Catalog is empty, use bayesloc_utils.read_bayesloc_origins')
    arrival_df = read_bayesloc_arrivals(
        arrival_file, custom_epoch=custom_epoch)
    if phases_file is not None:
        phases_df = read_bayesloc_phases(phases_file)
        # need to remove event_id column so they don't become duplicated
        phases_df = phases_df.drop(columns=['ev_id', 'sta_id', 'phase'])
        if len(phases_df) == len(arrival_df):
            arrival_df = pd.concat([arrival_df, phases_df], axis=1)
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
            except AttributeError:
                continue
        if bayesloc_event_id is None:
            Logger.info(
                'Cannot add bayesloc arrivals, there is no Bayesloc solution '
                ' for event %s', event.short_str())
            continue

        # Need to read in residuals files - which ones?
        # Either read directly  from file XXX
        # or compute residuals from arrival times and corections for:
        # station, phase, event, distance with files
        # tte_station_shifts_stats.out
        # tte_station_phase_shifts_stats.out,
        # tte_phase_dist_scales_stats.out
        # 
        # atep_event_factors_stats.out
        # 
        # Select arrivals / picks
        event_arrival_df = arrival_df[arrival_df.ev_id==bayesloc_event_id]
        Logger.info('Adding bayesloc-arrivals to origin for event %s',
                    event.short_str())
        for row in event_arrival_df.itertuples():
            pick_found = False
            # If there are already picks in catalog, try to find the one that
            # matches the bayesloc-arrival.
            if not add_picks:
                pick = [p for p in event.picks
                        if p.waveform_id.station_code==row.sta_id and
                        p.time==row.utctime and p.phase_hint==row.phase]
                if len(pick) == 1:
                    pick_found = True
                    new_pick = pick[0]
                # If picks are called "P1" or "S1", find by time / first letter
                elif (len(pick) == 0 and len(row.phase) > 1 and
                        row.phase[-1] == '1'):
                    pick = [p for p in event.picks
                        if p.waveform_id.station_code==row.sta_id and
                        p.time==row.utctime and p.phase_hint and
                        p.phase_hint[0]==row.phase.removesuffix('1')]
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
            # - time_residual: hmm... does Bayesloc output residual?
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
                                      #time_residual=,
                                      )
            nsp = 'Bayesloc'
            if not hasattr(new_arrival, 'extra'):
                new_arrival.extra = AttribDict()
            # bayes_orig.extra = {
            #         'bayesloc_event_id': {
            #             'value': row.ev_id,
            #             'namespace': 'Bayesloc'}}
            new_arrival.extra.update({'original_phase':
                {'value': row.phase, 'namespace': nsp}})
            new_arrival.extra.update({'prob_as_called':
                {'value': row.prob_as_called, 'namespace': nsp}})
            new_arrival.extra.update({'most_prob_phase': 
                {'value': row.most_prob_phase, 'namespace': nsp}})
            # Need to find the field with the probability for the new phasehint
            new_arrival.extra.update({
                'prob_as_suggested': 
                    {'value': getattr(row, row.most_prob_phase),
                     'namespace': nsp}})
            new_arrival.extra.update({'n': {'value': row.n, 'namespace': nsp}})
            # TODO could add probabilities for all other the other phase hints,
            #      that the pick/arrival could refer to, but that may not be 
            #      very useful.
            if new_arrival not in bayesloc_origin.arrivals:
                bayesloc_origin.arrivals.append(new_arrival)
        # try:
        #     origin = [orig for orig in event.origins
        #               if orig.creation_info.author=='Bayesloc'][0]
        # except (AttributeError, KeyError, IndexError):
        #     Logger.warning('Could not find Bayesloc origin for event %s',
        #                    event.short_str())
        #     continue
    return catalog


def update_tribe_from_bayesloc(tribe, bayesloc_stats_out_file,
                               custom_epoch=None):
    """
    """
    cat = Catalog([t.event for t in tribe])
    cat = update_cat_from_bayesloc(cat, bayesloc_stats_out_file,
                                   custom_epoch=custom_epoch)
    for ne, event in enumerate(cat):
        tribe[ne].event = event
    return tribe


def read_bayesloc_events(bayesloc_output_folder, custom_epoch=None):
    """
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
        add_arrivals=False, update_phase_hints=False,
        keep_best_fit_pick_only=False, remove_1_suffix=False,
        min_phase_probability=0, **kwargs):
    """
    Update a catalog's locations from a bayesloc-relocation run
    """
    # '/home/felix/Documents2/BASE/Detection/Bitdalsvatnet/Relocation/BayesLoc/'
    #     + 'Bitdalsvatnet_04_prior_955_N_Sg/output/origins_ned_stats.out',
    
    if isinstance(bayesloc_stats_out_files, str):
        bayesloc_stats_out_files = [bayesloc_stats_out_files]
        
    # Loop through multiple bayesloc output folders
    for bayesloc_stats_out_file in bayesloc_stats_out_files:
        bayesloc_folder = bayesloc_stats_out_file
        bayesloc_stats_out_file = get_bayesloc_filepath(
            bayesloc_stats_out_file, default_output_file='origins_ned_stats.out')
        
        cat, _, bayesloc_solutions_added = read_bayesloc_origins(
            bayesloc_stats_out_file, cat=cat, custom_epoch=custom_epoch,
            agency_id=agency_id, find_event_without_id=find_event_without_id,
            s_diff=s_diff, max_bayes_error_km=max_bayes_error_km)
        if not bayesloc_solutions_added:
            continue
        # TODO indicate that this solution is from Bayesloc
        # TODO: load phase probabilities, take the one that is most likely
        #       the "correct" pick for each phase, and fix picks
        #       accordingly.
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
    """
    for event in cat:
        bayesloc_event_id = None
        bayesloc_origin = None
        for origin in event.origins:
            try:
                bayesloc_event_id = origin.extra.get(
                    'bayesloc_event_id')['value']
                break
            except AttributeError:
                continue
        if bayesloc_event_id is not None:
            Logger.info(
                'Event %s: Sorting out duplicate picks: keeping only those picks /'
                ' arrivals that best fit.', event.short_str())
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
            rel_arrivals = [
                arrival for arrival in bayesloc_origin.arrivals
                if arrival.pick_id.get_referred_object() is not None and
                phase == arrival.phase and station ==
                arrival.pick_id.get_referred_object().waveform_id.station_code]
            # Can save some time here if there is only 1 pick:
            if len(rel_arrivals) == 1 and min_phase_probability == 0:
                continue
            rel_picks = [
                arrival.pick_id.get_referred_object()
                for arrival in rel_arrivals
                if arrival.pick_id.get_referred_object() is not None]
            phase_probabilities = [arrival.extra.prob_as_called.value
                                   for arrival in rel_arrivals]
            if len(phase_probabilities) == 0:
                continue
            max_phase_probability = max(phase_probabilities)
            max_phase_prob_idx = np.argmax(phase_probabilities)
            # Keep only the best one
            if max_phase_probability > min_phase_probability:
                Logger.debug(
                    'Event %s: There are %s picks for %s for station %s, '
                    'keeping only the best fitting pick above probability %s.',
                    event.short_str(), len(rel_picks), phase, station,
                    min_phase_probability)
            else:
                Logger.debug(
                    'Event %s: There are %s picks for %s for station %s, '
                    'but probably (max: %s) none are properly assigned.',
                    event.short_str(), len(rel_picks), phase, station,
                    max_phase_probability)
            for j, (arrival, pick) in enumerate(zip(rel_arrivals, rel_picks)):
                if (j == max_phase_prob_idx and
                        max_phase_probability > min_phase_probability):
                    continue
                bayesloc_origin.arrivals.remove(arrival)
                event.picks.remove(pick)
    return cat


def _update_bayesloc_phase_hints(cat, remove_1_suffix=False):
    """
    Update arrivals and picks with phase hints indicated by Bayesloc.
    """
    for event in cat:
        bayesloc_event_id = None
        bayesloc_origin = None
        for origin in event.origins:
            try:
                bayesloc_event_id = origin.extra.get(
                    'bayesloc_event_id')['value']
                break
            except AttributeError:
                continue
        if bayesloc_event_id is not None:
            Logger.info('Updating phase hints for event %s', event.short_str())
            bayesloc_origin = origin
            for arrival in bayesloc_origin.arrivals:
                if hasattr(arrival, 'extra'):
                    if (arrival.extra.original_phase.value !=
                            arrival.extra.most_prob_phase.value):
                        # Rename pick phase according to Bayesloc
                        if arrival.pick_id.get_referred_object() is None:
                            continue
                        pick = arrival.pick_id.get_referred_object()
                        arrival.phase = arrival.extra.most_prob_phase.value
                        arrival.extra.prob_as_called.value = (
                            arrival.extra.prob_as_suggested.value)
                        pick.phase_hint = arrival.extra.most_prob_phase.value
                    # original_phase prob_as_called most_prob_phase
                # When first arrival phase is called like "P1" or "S1", may need to
                # rename:
                if (remove_1_suffix and len(arrival.phase[-1]) > 1 and
                        arrival.phase[-1] == '1'):
                    arrival.phase.removesuffix('1')
    return cat



# %%
    # cat = read_bayesloc_events('/home/felix/Software/Bayesloc/Example_Ridge_Gibbons_2017/output')




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
        # '/home/seismo/WOR/felix/R/SEI/REA/INTEU/2020/12/14-1935-58R.S202012')
        '/home/seismo/WOR/felix/R/SEI/REA/INTEU/2021/07/05-0423-41R.S202107')
    

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
