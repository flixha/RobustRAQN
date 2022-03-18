
# %%
import os
from cmath import sin
from collections import defaultdict
from obspy import read_inventory
# import wcmatch
from wcmatch import fnmatch 
import pandas as pd
import numpy as np
import math

import logging
Logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(name)40s:%(lineno)s\t%(funcName)20s()\t%(levelname)s\t%(message)s")

from collections import Counter

from obspy import Stream
from obspy.signal.util import next_pow_2, util_geo_km
from obspy.io.nordic.core import read_nordic, write_select
from obspy.signal.array_analysis import get_timeshift
from obspy.geodetics import (degrees2kilometers, kilometers2degrees,
                             gps2dist_azimuth, locations2degrees)
from obspy.taup.velocity_model import VelocityModel
from obspy.core.event import (Catalog, Pick, Arrival, WaveformStreamID,
                              CreationInfo)
from obspy import UTCDateTime
from obsplus.stations.pd import stations_to_df

from robustraqn.spectral_tools import get_updated_inventory_with_noise_models

SEISARRAY_PREFIXES = [
    'NAO*', 'NBO*', '@(NB2*|NOA)', 'NC2*', 'NC3*', 'NC4*', 'NC6*',
    'NR[ABCD][0-9]',
    '@(ARCES|AR[ABCDE][0-9])', '@(SPITS|SP[ABC][0-5])', '@(BEAR|BJO*|BEA[1-6])',
    'OSE[0-9][0-9]', 'EKO[0-9]*', 'GRA[0-9][0-9]',
    '@(EKA|ESK|EKB*|EKR*)', '@(ILAR|IL[0-3][0-9])', '@(YKA|YKA*[0-9])',
    '@(HN[AB][0-6]|BAS02)',
    '@(OBS[0-6]|OBS1[1-2]'  # OBS
]

LARGE_APERTURE_SEISARRAY_PREFIXES = [
    '@(N[ABC][O2346]*|NOA|NR[ABCD][0-9]|NAO*|NBO*)',
    '@(ISF|BRBA|BRBB|BRB)',  # Svalbard west
    # Vestland
    '@(KMY|KMY2|NWG22)',
    '@(ODD|ODD1|NWG21|BAS1[0-4]|BAS19|BAS2[0-3]|REIN)',
    '@(BER|ASK|RUND|HN[AB][0-6]|BAS0[3-6]|BAS0D|BAS1[5-7]|ASK[0-8]|SOTS|TB2S|OSGS|ESG|EGD|SOTS|ARNS|BT2[13]',
    # Midtnorge
    '@(AKN|JH1[02]|JH0[89])',
    '@(DOMB|JH0[34]|JH11)',
    '@(LADE|LENS|ODLO|TBLU|TRON|NWG01|N6004|N6005|N6006|N6007|N6008)',
    '@(NSS|SA35)',
    # Nordland
    '@(ROEST|N2RO|N2VA)',
    '@(VAGH|KONS|STOK|LEIR|RAUS|MOR8|FLOS|STOK1|STOK2|NBB13|NBB14|NBB15)',
    '@(GILDE|NBB05|NBB30|MELSS|NBB17|N1304)',  # South 2 of NOrdland
    '@(FAUS|N2TV|NBB08|N2ST)',                 # North 2 of Nordland
    '@(VBYGD|STEI|N2LO|N2DI|N2HS|N2IH)']       # Northern part of Nordland


REF_STATIONS = ['ARCES', 'SPITS', 'NC2', 'NAO', 'NBO', 'NB2', 'NOA', 'NC2',
                'NC3', 'NC4', 'NC6', 'NRA0', 'BEAR', 'YKA', 'ILAR', 'EKA']

SEISARRAY_REF_STATIONS = {
    seisarray_prefix: ref_station
    for ref_station in REF_STATIONS for seisarray_prefix in SEISARRAY_PREFIXES
    if fnmatch.fnmatch(ref_station, seisarray_prefix, flags=fnmatch.EXTMATCH)}

SEISARRAY_REF_EQUIVALENT_STATIONS = {
  'ARCES': 'ARA0',
  'SPITS': 'SPA0',
  'NC2': 'NC200',
  'NAO': 'NAO00',
  'NBO': 'NBO00',
  'NB2': 'NB200',
  'NOA': 'NB200',
  'NC2': 'NC200',
  'NC3': 'NC300',
  'NC4': 'NC400',
  'NC6': 'NC600',
  'NRA0': 'NC602',
  'BEAR': 'BEA4',
  'YKA': 'YKR8',
  'ILAR': 'IL01'
}

# Notes:
# May be able to treat southern part of Nordland stations as a large array (70 x 70 km)


"""
1. Template Creation:
 1.1 prepare array picks:
     - load list of stations
     - check which picks are part of an array
     - group picks by array
     - for each array pick:
       - compute array arrivals 
         - first check for BAZ and app-vel measurments
         - or use origin and computed BAZ+app-vel
     - keep average arrival for each array station based on one or more 
       array-arrival computations
     - check which array traces do not have an associated pick and append a
       pick for these
 1.2 MAYBE stack array waveforms in templates:
    - align waveforms based on picks
    - stack waveforms
    - adjust arrival time and replace each unstacked trace with the stacked one
2. Detection
3. Picking
"""


def get_array_stations_from_df(stations_df=pd.DataFrame(),
                               seisarray_prefixes=SEISARRAY_PREFIXES):
    # array_list is a list of tuples, with the first tuple element containing
    # the array-prefix and the 2nd tuple element containing a list of stations
    # at the array.
    # seisarray_list = list()
    """
    Returns a dict containing the array prefixes and all stations that belong
    to the corresponding array prefix.
    """
    seisarray_dict = dict()
    single_station_list = stations_df.copy()
    # alternative pattern 'E*K!(O)*' (works only  in wcmatch, not glob)
    # ''E*K[!O]*'
    for seisarray_prefix in seisarray_prefixes:
        seisarray_station_list = list()
        for resp in stations_df.iterrows():
            if fnmatch.fnmatch(resp[1].station, seisarray_prefix,
                               flags=fnmatch.EXTMATCH):
                seisarray_station_list.append(resp[1].station)
                # single_station_list.remove(station)
        # seisarray_list.append((seisarray_prefix, seisarray_station_list))
        if len(seisarray_station_list) > 0:
            seisarray_dict[seisarray_prefix] = list(
              set(seisarray_station_list))

    return seisarray_dict


def filter_array_stations_df(stations_df=pd.DataFrame(), seisarray_prefix=''):
    """
    Extract the stations_df for the stations of an array, described by a Unix-
    style pattern in seisarray_prefix.
    """
    array_stations_df = pd.DataFrame(
        [resp for index, resp in stations_df.iterrows()
         if fnmatch.fnmatch(resp.station, seisarray_prefix,
                            flags=fnmatch.EXTMATCH)])
    # TODO: only return unique stations?
    # array_stations_df 
    return array_stations_df


def extract_array_picks(event, seisarray_prefixes=SEISARRAY_PREFIXES):
    """
    Return dict of dicts with array-prefix (keys) and relevant phase_hints
    (values / keys) and the relevant picks (values).
    """
    array_picks_dict = dict()
    # array_picks_dict = defaultdict(dict)
    # defaultdict(list)

    for seisarray_prefix in seisarray_prefixes:
        array_picks_list = list()
        for pick in event.picks:
            station = pick.waveform_id.station_code
            if fnmatch.fnmatch(station, seisarray_prefix,
                               flags=fnmatch.EXTMATCH):
                array_picks_list.append(pick)
                # single_station_list.remove(station)
        phase_hints = list(set([
            pick.phase_hint for pick in array_picks_list
            if pick.phase_hint and pick.phase_hint[0] in 'PS']))
        pha_picks_dict = dict()
        for phase_hint in phase_hints:
            pha_picks_dict[phase_hint] = [pick for pick in array_picks_list
                                          if pick.phase_hint == phase_hint]
            # seisarray_list.append((seisarray_prefix, seisarray_station_list))
        if len(pha_picks_dict.keys()) > 0:
            array_picks_dict[seisarray_prefix] = pha_picks_dict

    return array_picks_dict


def extract_array_stream(st, seisarray_prefixes=SEISARRAY_PREFIXES):
    """
    Return dict with array-prefix (keys) and relevant streams (values) .
    """
    array_streams_dict = dict()
    # array_picks_dict = defaultdict(dict)
    # defaultdict(list)
    
    for seisarray_prefix in seisarray_prefixes:
        array_st = Stream()
        for tr in st:
            if fnmatch.fnmatch(tr.stats.station, seisarray_prefix,
                               flags=fnmatch.EXTMATCH):
                array_st += tr
        # seisarray_list.append((seisarray_prefix, seisarray_station_list))
        if len(array_st) > 0:
            array_streams_dict[seisarray_prefix] = array_st
    return array_streams_dict


def get_geometry(stations_df, array_prefix='', coordsys='lonlat',
                 return_center=False):
    """
    Modified from obspy.signal.array_analysis.get_geometry to allow calculation
    for a stations dataframe.
    (:copyright: The ObsPy Development Team (devs@obspy.org))
    
    Method to calculate the array geometry and the center coordinates in km

    :param stream: Stream object, the trace.stats dict like class must
        contain an :class:`~obspy.core.util.attribdict.AttribDict` with
        'latitude', 'longitude' (in degrees) and 'elevation' (in km), or 'x',
        'y', 'elevation' (in km) items/attributes. See param ``coordsys``
    :param coordsys: valid values: 'lonlat' and 'xy', choose which stream
        attributes to use for coordinates
    :param return_center: Returns the center coordinates as extra tuple
    :return: Returns the geometry of the stations as 2d :class:`numpy.ndarray`
            The first dimension are the station indexes with the same order
            as the traces in the stream object. The second index are the
            values of [lat, lon, elev] in km
            last index contains center [lat, lon, elev] in degrees and km if
            return_center is true
    """
    array_stations_df = filter_array_stations_df(
        stations_df=stations_df, seisarray_prefix=array_prefix)
    
    nstat = len(array_stations_df)
    center_lat = 0.
    center_lon = 0.
    center_h = 0.
    geometry = np.empty((nstat, 3))

    for i, (index, resp) in enumerate(array_stations_df.iterrows()):
        if coordsys == 'lonlat':
            geometry[i, 0] = resp.longitude
            geometry[i, 1] = resp.latitude
            geometry[i, 2] = resp.elevation
        elif coordsys == 'xy':
            geometry[i, 0] = resp.x
            geometry[i, 1] = resp.y
            geometry[i, 2] = resp.elevation
    Logger.debug("Array geometry coordsys = %s", coordsys)

    if coordsys == 'lonlat':
        center_lon = geometry[:, 0].mean()
        center_lat = geometry[:, 1].mean()
        center_h = geometry[:, 2].mean()
        for i in np.arange(nstat):
            x, y = util_geo_km(center_lon, center_lat, geometry[i, 0],
                               geometry[i, 1])
            geometry[i, 0] = x
            geometry[i, 1] = y
            geometry[i, 2] -= center_h
    elif coordsys == 'xy':
        geometry[:, 0] -= geometry[:, 0].mean()
        geometry[:, 1] -= geometry[:, 1].mean()
        geometry[:, 2] -= geometry[:, 2].mean()
    else:
        raise ValueError("Coordsys must be one of 'lonlat', 'xy'")

    if return_center:
        return np.c_[geometry.T,
                     np.array((center_lon, center_lat, center_h))].T
    else:
        return geometry, array_stations_df


def find_array_picks_baz_appvel(
        event, phase_hints=None, seisarray_prefixes=SEISARRAY_PREFIXES,
        array_picks_dict=None, array_baz_dict=None, array_app_vel_dict=None,
        vel_mod=None, vel_mod_file=os.path.join(
            os.path.dirname(__file__), 'models','NNSN1D_plusAK135.tvel')):
    """
    For one event, for every phase, find all picks at the same array, find the
    backazimuth and the apparent velocity, either from measurements or from the
    computed arrivals. Return dictionaries of picks, baz, and app-vel for each
    phase.
    
    returns 2-level dictionaries (1st level keys are array-prefixes, 2nd level
    keys are phase-hints) for:
    - all picks of same phase-hint at each array
    - backazimuth of the phase at each array (may be averaged)
    - apparent velocity of the phase at each array (may be averaged)
    
    phase_hints: can be None (check all phases automatically) or list of phases
                for which to find/compute baz and app-vel
    """
    if vel_mod is None:
        vel_mod = VelocityModel.read_tvel_file(vel_mod_file)
    update_array_picks = False
    update_bazs = False
    update_app_vels = False
    # Check if user supplied any values - then do not update /overwrite those
    if not array_picks_dict:
        array_picks_dict = extract_array_picks(
            event, seisarray_prefixes=seisarray_prefixes)
        update_array_picks = True
    if not array_baz_dict:
        array_baz_dict = dict()
        update_bazs = True
    if not array_app_vel_dict:
        array_app_vel_dict = dict()
        update_app_vels = True
    # Loop over array-prefixes
    update_phase_hints = False
    if phase_hints is None:
        update_phase_hints = True
    for seisarray_prefix in array_picks_dict.keys():
        if update_phase_hints:
            phase_hints = array_picks_dict[seisarray_prefix].keys()
        if update_bazs:
            array_baz_dict[seisarray_prefix] = dict()
        if update_app_vels:
            array_app_vel_dict[seisarray_prefix] = dict()
        for phase_hint in phase_hints:
            Logger.debug('Looking for backazimuth and apparent velocity for ' +
                         'array %s, phase %s', seisarray_prefix, phase_hint)
            # if update_array_picks:
            #     array_picks_dict[seisarray_prefix][phase_hint] = [
            #         pick for pick in array_picks
            #         # phase_picks --> pha_picks_dict
            #         if pick.phase_hint == phase_hint]
            if update_bazs:
                array_baz_dict[seisarray_prefix][phase_hint] = np.mean(
                    [pick.backazimuth
                     for pick in array_picks_dict[seisarray_prefix][phase_hint]
                     if pick.backazimuth is not None])
            if update_app_vels:
                array_app_vel_dict[seisarray_prefix][phase_hint] = np.mean(
                    [degrees2kilometers(1.0 / pick.horizontal_slowness)
                     for pick in array_picks_dict[seisarray_prefix][phase_hint]
                     if pick.horizontal_slowness is not None])

            # If there is no origin (e.g., for newly picked events),
            # then don't try to compute BAZ or app-vel
            can_calculate_baz_appvel = True
            origin = event.preferred_origin()
            if origin is None:
                try: 
                    origin = event.origins[0]
                except IndexError as e:
                    continue
            if origin is None:
                can_calculate_baz_appvel = False
                
            # if there is no measurements for BAZ, compute it from arrivals:
            if update_bazs and can_calculate_baz_appvel:
                calculated_bazs = []
                if np.isnan(array_baz_dict[seisarray_prefix][phase_hint]):
                    for pick in array_picks_dict[seisarray_prefix][phase_hint]:
                        for arrival in origin.arrivals:
                            if (arrival.pick_id == pick.resource_id and
                                    arrival.azimuth is not None):
                                calculated_bazs.append(
                                    (arrival.azimuth + 180) % 360)
                    array_baz_dict[seisarray_prefix][phase_hint] = np.mean(
                        calculated_bazs)
            # compute apparent velocity if there is not measurement for phase
            if update_app_vels and can_calculate_baz_appvel:
                calculated_app_vels = []
                if np.isnan(array_app_vel_dict[seisarray_prefix][phase_hint]):
                    for pick in array_picks_dict[seisarray_prefix][phase_hint]:
                        for arrival in origin.arrivals:
                            # takeoff_angle is incidence angle read from Seisan
                            # TODO compute from: incidence angle and velocity
                            #  model should be simple calculation with topmost
                            # velocity and AIN
                            if (arrival.pick_id == pick.resource_id and
                                    arrival.takeoff_angle is not None):
                                if pick.phase_hint[0] == 'P':
                                    vel = vel_mod.layers[0][2]
                                elif  pick.phase_hint[0] == 'S':
                                    vel = vel_mod.layers[0][4]
                                else:
                                    continue
                                app_vel = vel / math.sin(math.radians(
                                    arrival.takeoff_angle))
                                calculated_app_vels.append(app_vel)
                    array_app_vel_dict[seisarray_prefix][phase_hint] = np.mean(
                        calculated_app_vels)

    return array_picks_dict, array_baz_dict, array_app_vel_dict


def add_array_station_picks(
        event, stations_df, array_picks_dict=None, array_baz_dict=None,
        array_app_vel_dict=None, baz=None, app_vel=None,
        seisarray_prefixes=SEISARRAY_PREFIXES,
        vel_mod_file=os.path.join(os.path.dirname(__file__), 'models',
                                  'NNSN1D_plusAK135.tvel'), **kwargs):
    """
    Returns all picks at array stations from 
        # ARRAY stuff
        if backazimuth or apparent_velocity are defined as anytyhing else than
        None, then they overwrite all baz / app-vel values

    1.1 prepare array picks:
     - load list of stations
     - check which picks are part of an array
     - group picks by array
     - for each array pick:
       - compute array arrivals 
         - first check for BAZ and app-vel measurments
         - or use origin and computed BAZ+app-vel
     - keep average arrival for each array station based on one or more 
       array-arrival computations
     - check which array stations do not have an associated pick and append a
       pick for these
    """
    n_picks_before = len(event.picks)
    vel_mod = VelocityModel.read_tvel_file(vel_mod_file)
    # 2. For each array:
    #   2.1 collect all equivalent picks from the same array
    #       lets either user supply baz- and app-vel dicts, or lets code find
    #       the values
    array_picks_dict, array_baz_dict, array_app_vel_dict = (
        find_array_picks_baz_appvel(
            event, array_picks_dict=array_picks_dict,
            seisarray_prefixes=seisarray_prefixes, vel_mod=vel_mod))

    #   2.2. get array delays for relevant arrival.
    for seisarray_prefix, pha_picks_dict in array_picks_dict.items():
        # array_picks --> pha_picks_dict
        array_geometry, array_stations_df = get_geometry(
            stations_df, array_prefix=seisarray_prefix, coordsys='lonlat',
            return_center=False)
        array_center = get_geometry(
            array_stations_df, array_prefix=seisarray_prefix,
            coordsys='lonlat', return_center=True)

        # # Find array reference: it is either:
        # #   1. array beam ref that is defined inventory
        # # #    (station.site.name contains beam)
        # array_beam_name = [
        #     resp.station for index, resp in array_stations_df.iterrows()
        #     if 'BEAM' in resp.site_name.upper()]
        # array_ref_station = None
        # if array_beam_name:
        #     array_ref_station = array_beam_name[0]
        # #   2. station that is closest to the center of the array
        # if array_ref_station is None:
        #     distances = np.sqrt(
        #         degrees2kilometers(locations2degrees(
        #             stations_df.latitude, stations_df.longitude,
        #             array_center[1][0], array_center[1][0])) ** 2 +
        #         ((stations_df.elevation - array_center[1][2]) / 1000) ** 2)
        #     min_distance_index = np.argmin(distances)
        #     array_ref_station = (
        #         array_stations_df.iloc[min_distance_index].station)
        
        # # Correct array geometry to make the reference station the array center
        # # array_stations_df[array_stations_df.station==array_ref_station].iloc[0]
        # idx = None
        # for j, (index, resp) in enumerate(array_stations_df.iterrows()):
        #     if resp.station == array_ref_station:
        #         idx = j
        # if idx is not None:
        #     array_geometry_for_ref = array_geometry - array_geometry[idx]
            
            
        # # get array timeshifts and align traces by timeshift
        # for arrival in origin.arrivals:
        #     pick = arrival.pick_id.get_referred_object()
        #     if arrival.phase == 'P' and pick.waveform_id.station_code == refStation:
        #         # Extract/Compute horizontal slowness (in quakeML is stored as s/deg)
        #         if pick.horizontal_slowness is not None:
        #             velocity = 1.0 / pick.horizontal_slowness
        #             pick.horizontal_slowness = 1/8.2
        #         elif pick.time is not None and origin.time is not None\
        #                 and arrival.distance is not None:
        #             dist_deg = arrival.distance
        #             dist_km = degrees2kilometers(dist_deg)
        #             # for the relevant pick, get station coordinates
        #             inv2 = inv.select(station=pick.waveform_id.station_code)
        #             sta_lat = inv2.networks[0].stations[0].channels[0].latitude
        #             sta_lon = inv2.networks[0].stations[0].channels[0].longitude
        #             # compute back azimuth based on origin and station coordinates
        #             gcd, az, baz = gps2dist_azimuth(origin.latitude, origin.longitude,
        #                                             sta_lat, sta_lon)
        #             pick.backazimuth = baz
        #             #pick.horizontal_slowness = (pick.time - origin.time) / dist_deg
        #             pick.horizontal_slowness = (pick.time - origin.time) / dist_km
        #             pick.horizontal_slowness = 1/8.2
        #             velocity = arrival.distance / (pick.time - origin.time)
        #         sll_x = math.cos(math.radians(baz)) * pick.horizontal_slowness
        #         sll_y = math.sin(math.radians(baz)) * pick.horizontal_slowness
        #         timeshifts = get_timeshift(array_geometry, sll_x, sll_y, 0, 1, 1)
        #         ref_arrival = arrival
        #         ref_pick = pick

        #         timeshifts2 = -array_geometry[:,2] / 1000 * 1/3
        #         break

        phase_hints = pha_picks_dict.keys()

        # phase_hints = list(set([
        #     pick.phase_hint for pick in array_picks
        #     if pick.phase_hint and pick.phase_hint[0] in 'PS']))
    
        # array_picks_dict = dict()
        # array_baz_dict = dict()
        # array_app_vel_dict = dict()
        # for phase_hint in phase_hints:
        #     Logger.info('Working on array %s, phase %s', seisarray_prefix,
        #                 phase_hint)
        #     array_picks_dict[phase_hint] = [pick for pick in array_picks
        #                                     if pick.phase_hint == phase_hint]
        #     array_baz_dict[phase_hint] = np.mean(
        #         [pick.backazimuth
        #          for pick in array_picks_dict[phase_hint]
        #          if pick.backazimuth is not None])
        #     array_app_vel_dict[phase_hint] = np.mean(
        #         [degrees2kilometers(1.0 / pick.horizontal_slowness)
        #          for pick in array_picks_dict[phase_hint]
        #          if pick.horizontal_slowness is not None])

        #     # if there is no measurements for BAZ, compute it from arrivals:
        #     calculated_bazs = []
        #     if np.isnan(array_baz_dict[phase_hint]):
        #         for pick in array_picks_dict[phase_hint]:
        #             for arrival in event.preferred_origin().arrivals:
        #                 if (arrival.pick_id == pick.resource_id and
        #                         arrival.azimuth is not None):
        #                     calculated_bazs.append(
        #                         (arrival.azimuth + 180) % 360)
        #         array_baz_dict[phase_hint] = np.mean(calculated_bazs)
        #     # compute apparent velocity if there is not measurement for phase
        #     calculated_app_vels = []
        #     if np.isnan(array_app_vel_dict[phase_hint]):
        #         for pick in array_picks_dict[phase_hint]:
        #             for arrival in event.preferred_origin().arrivals:
        #                 if (arrival.pick_id == pick.resource_id and
        #                         # arrival.distance is not None and
        #                         arrival.takeoff_angle is not None):
        #                     # takeoff_angle is incidence angle when read from Seisan
        #                     # TODO compute from: incidence angle and velocity model
        #                     #      should be simple calculation with topmost
        #                     #      velocity and AIN
        #                     if pick.phase_hint[0] == 'P':
        #                         vel = vel_mod.layers[0][2]
        #                     elif  pick.phase_hint[0] == 'S':
        #                         vel = vel_mod.layers[0][4]
        #                     else:
        #                         continue
        #                     app_vel = vel / math.sin(math.radians(
        #                         arrival.takeoff_angle))
        #                     calculated_app_vels.append(app_vel)
        #         array_app_vel_dict[phase_hint] = np.mean(calculated_app_vels)


        # 2.3 compute average array pick at reference site.
        #     i now have BAZ, app-vel, distance, and velocity model
        # app_vel = degrees2kilometers(1.0 / pick.horizontal_slowness)
        for phase_hint in phase_hints:
            Logger.info(
                'Computing average picks for %s at stations of array %s',
                phase_hint, seisarray_prefix)
            if app_vel is None:
                app_vel = array_app_vel_dict[seisarray_prefix][phase_hint]
            horizontal_slowness_km = 1.0 / app_vel
            if baz is None:
                baz = array_baz_dict[seisarray_prefix][phase_hint]
            # Check for missing information for array arrival
            if (baz is None or np.isnan(baz) or horizontal_slowness_km is None
                    or np.isnan(horizontal_slowness_km)):
                Logger.error(
                    'Cannot compute timeshifts for array %s arrival %s - '
                    'missing backazimuth (%s) and/or slowness (%s). You may '
                    'need to locate or update event %s.', seisarray_prefix,
                    phase_hint, str(baz), str(horizontal_slowness_km),
                    event.short_str())
                continue
            sll_x = math.cos(math.radians(baz)) * horizontal_slowness_km
            sll_y = math.sin(math.radians(baz)) * horizontal_slowness_km
            # timeshifts2 = -array_geometry[:,2] / 1000 * 1/3
            pick_time_list = []
            pick_time_list_ns = []
            #   2.4 compute theoretical arrivals at all array statinos
            for pick in array_picks_dict[seisarray_prefix][phase_hint]:
                # Correct array geometry to make the reference station the array center
                # first find the index of the pick-station in the geometry
                idx = None
                for j, (index, resp) in enumerate(array_stations_df.iterrows()):
                    # if resp.station == array_ref_station:
                    if resp.station == pick.waveform_id.station_code:
                        idx = j
                if idx is not None:
                    array_geometry_for_pick_sta = (
                        array_geometry - array_geometry[idx])
                else:
                    continue
                timeshifts = get_timeshift(array_geometry_for_pick_sta,
                                           sll_x, sll_y, 0, 1, 1)
                # for timeshift in timeshifts:
                #     if pick.time is None or np.isnan(timeshift[0][0]):
                #         Logger.warning(
                #             'NaN in pick time (%s) for %s (%s), shift: %s',
                #             pick.waveform_id.get_seed_string(), pick.phase_hint,
                #             str(pick.time), str(timeshift[0][0]))
                pick_time_list.append([pick.time - timeshift[0][0]
                                       for timeshift in timeshifts])
                pick_time_list_ns.append([(pick.time - timeshift[0][0])._ns
                                          for timeshift in timeshifts])
            # pick-time average needs to be calculated from epoch-seconds
            pick_times_av = [
                UTCDateTime(time_av)
                for time_av in (np.mean(pick_time_list_ns, axis=0) / 1e9)]
            # 3. add picks for array stations that did not have pick to pick-list
            # Only add pick for array station if there is no equivalent pick
            # for that station yet (check phase_hint and station.code)
            for row_n, pick_time in enumerate(pick_times_av):
                pick_tuples = [(pick.phase_hint, pick.waveform_id.station_code)
                               for pick in event.picks]
                # Get the most common value for onset and polarity for picks of
                # the same phase at the array
                onset = None
                onset_counter = Counter(
                    [pick.onset for pick in array_picks_dict[
                        seisarray_prefix][phase_hint]]).most_common()
                if onset_counter:
                    onset = onset_counter[0][0]
                # Get most common polarity
                polarity = None
                polarity_counter = Counter([
                    pick.polarity for pick in array_picks_dict[
                        seisarray_prefix][phase_hint]]).most_common()
                if polarity_counter:
                    polarity = polarity_counter[0][0]
                # Get most common Net/Loc/Chn descriptor
                new_waveform_id = array_picks_dict[
                    seisarray_prefix][phase_hint][0].waveform_id.copy()
                nlc_counter = Counter([
                    (pick.waveform_id.network_code,
                     pick.waveform_id.location_code,
                     pick.waveform_id.channel_code)
                    for pick in array_picks_dict[
                        seisarray_prefix][phase_hint]]).most_common()
                if nlc_counter:
                    net, loc, chn = nlc_counter[0][0]
                    new_waveform_id = WaveformStreamID(
                        network_code=net, location_code=loc, channel_code=chn)
                station = array_stations_df.iloc[row_n].station
                if ((phase_hint, station) not in pick_tuples):
                    # new_pick = array_picks_dict[seisarray_prefix][phase_hint][0].copy()
                    Logger.debug('Adding pick for %s at array-station %s',
                                 phase_hint, station)
                    new_waveform_id.station_code = station
                    new_pick = Pick(
                        time=pick_time, phase_hint=phase_hint,
                        waveform_id=new_waveform_id,
                        horizontal_slowness=kilometers2degrees(
                            1 / horizontal_slowness_km),
                        # backazimuth=array_baz_dict[seisarray_prefix][phase_hint],
                        backazimuth=baz,
                        onset=onset,
                        polarity=polarity,
                        evaluation_mode='automatic',
                        creation_info=CreationInfo(agency_id='RR'))
                    # add a pick-.weight if there is one
                    if 'extra' in pick.keys():
                        new_pick['extra'] = dict()
                        new_pick['extra']['nordic_pick_weight'] = dict()
                        new_pick['extra']['nordic_pick_weight']['value'] = (
                            pick['extra']['nordic_pick_weight']['value'])
                        new_pick['extra']['nordic_pick_weight'][
                            'namespace'] = (pick['extra']['nordic_pick_weight'
                                                          ]['namespace'])
                    event.picks.append(new_pick)

    # 3. add picks for array stations that did not have pick to pick-list
    # out_picks = event.picks.copy()
    # for seisarray in seisarrays:
    #     for pick in array_picks:
    #         if pick.waveform_id, pick.phase_hint not in [
    #                 (p.waveform_id, p.phase_hint) for p in picks]:
    #             out_picks += pick
    # event.picks = out_picks
    n_picks_after = len(event.picks)
    Logger.info(
        'Added %s computed picks for array stations (before: %s, after: %s)',
        str(n_picks_after - n_picks_before), n_picks_before, n_picks_after)

    return event


def array_lac_calc(
        st, picked_cat, party, tribe, stations_df,
        seisarray_prefixes=SEISARRAY_PREFIXES,
        min_cc=0.4, pre_processed=False, shift_len=0.8,
        min_cc_from_mean_cc_factor=0.6,
        horizontal_chans=['E', 'N', '1', '2'], vertical_chans=['Z'],
        interpolate=False, plot=False, overlap='calculate',
        parallel=False, cores=None, daylong=True, ignore_bad_data=True,
        ignore_length=True, **kwargs):
    """
    Obtain a cross-correlation pick for the whole array.

    Here is how to do it:
    1. for every array:
        - select all waveforms for the array
        - for every phase:
            - run lag_calc. Do not allows single traces to shift, but only
              allow the whole stream to move together.
            - min_cc should be selected based on the number of array stations
              and other contributing factors.
        - add one single pick for the array's reference station for each phase
    """
    tribe_array_picks_dict = dict()
    for family in party:
        tribe_array_picks_dict[family.template.name] = extract_array_picks(
            event=family.template.event)
    phase_hints = list(set([
        phase_hint
        for templ_name, array_picks_dict in tribe_array_picks_dict.items()
        for seisarray_prefix, phase_picks_dict in array_picks_dict.items()
        for phase_hint in phase_picks_dict.keys()]))

    array_st_dict = extract_array_stream(
        st, seisarray_prefixes=seisarray_prefixes)

    # tribe = Tribe([family.template for family in party])

    for phase_hint in phase_hints:
        # TODO do I better need to check what's in the party vs stream?
        # Need to select only the traces for relevant picks
        # array_party = Party()
        # for family in party:
        #     for detection in family:
        #         for station, chan in detection.chans:
                    
        for seisarray_prefix in array_st_dict.keys():
            # Factor to relax cc-requirement by - noise of stacked traces
            # should in theory reduce by sqrt(n_traces)
            # TODO - remove the factor 10 below after testing!!!!!
            cc_relax_factor = np.sqrt(len(array_st_dict[seisarray_prefix]))
            Logger.info('Preparing traces for array %s, for picking phase %s' +
                        ' with lag-calc. CC relax factor is %s',
                        seisarray_prefix, phase_hint, str(cc_relax_factor))
            # From (preprocessed?) stream select only traces for current array
            array_catalog = Catalog()
            array_catalog = party.copy().lag_calc(
                array_st_dict[seisarray_prefix], shift_len=0.0,
                pre_processed=pre_processed, min_cc=min_cc/cc_relax_factor,
                min_cc_from_mean_cc_factor=min_cc_from_mean_cc_factor/cc_relax_factor,
                horizontal_chans=horizontal_chans,
                vertical_chans=vertical_chans, interpolate=interpolate,
                plot=plot, overlap=overlap, parallel=parallel, cores=cores,
                daylong=daylong, ignore_bad_data=ignore_bad_data,
                ignore_length=ignore_length, **kwargs)
            
            Logger.info('Got new array picks for %s events.',
                        str(len(array_catalog)))
            # sort picks into previously lag-calc-picked catalog
            for event in array_catalog:
                # find the event that was picked from the same detection for
                # the whole network
                ref_station_code = SEISARRAY_REF_STATIONS[seisarray_prefix]
                ref_equi_stacode = SEISARRAY_REF_EQUIVALENT_STATIONS[
                    ref_station_code]
                picked_event = [ev for ev in picked_cat
                                if ev.resource_id == event.resource_id][0]
                existing_ref_picks = [
                    pick for pick in picked_event.picks
                    if (pick.waveform_id.station_code == ref_station_code and
                        pick.phase_hint == phase_hint)]
                existing_ref_equivalent_picks = [
                    pick for pick in picked_event.picks
                    if (pick.waveform_id.station_code == ref_equi_stacode and
                        pick.phase_hint == phase_hint)]
                # if there's already a pick for the array's reference beam
                # station, then I don't need to compute it and can save time.
                if len(existing_ref_picks) > 0:
                    Logger.info('There is already a pick for array beam %s, '
                                'phase %s, not adding any more picks.',
                                seisarray_prefix, phase_hint)
                    continue
                if len(existing_ref_equivalent_picks) > 0:
                    Logger.info(
                        'Adding array pick for array %s beam for phase %s.',
                        seisarray_prefix, phase_hint)
                    for equi_pick in existing_ref_equivalent_picks:
                        new_waveform_id = WaveformStreamID(
                            network_code=equi_pick.waveform_id.network_code,
                            station_code=ref_station_code,
                            location_code=equi_pick.waveform_id.location_code,
                            channel_code=equi_pick.waveform_id.channel_code)
                        new_pick = Pick(
                            time=equi_pick.time,
                            phase_hint=phase_hint,
                            waveform_id=new_waveform_id,
                            onset=equi_pick.onset,
                            polarity=equi_pick.polarity,
                            evaluation_mode='automatic',
                            creation_info=CreationInfo(agency_id='RR'))
                        # TODO can I add baz and app-vel here?
                        # horizontal_slowness=kilometers2degrees(
                        #     1 / horizontal_slowness_km),
                        # backazimuth=array_baz_dict[seisarray_prefix][phase_hint],
                        # backazimuth=baz,
                                            
                        picked_event.picks.append(new_pick)
                    continue  # avoid new computations

                # compute the pick at the array's reference station
                # Take the BAZ and app-vel from the template, but take the
                # pick-dict from the newly detected event
                # BAZ are in picks or arrivals of the template-event.
                Logger.info(
                    'Calculating arrival time at array %s beam site for phase '
                    ' %s', seisarray_prefix, phase_hint)
                template_names = [
                    comment.text.removeprefix('Detected using template: ')
                    for comment in event.comments
                    if 'Detected using template:' in comment.text]
                if template_names:
                    detection_template_name = template_names[0]
                else:
                    continue
                template_event = [templ.event for templ in tribe
                                  if templ.name == detection_template_name]
                if len(template_event) == 1:
                    template_event = template_event[0].copy()
                else:
                    Logger.error('Could not find template %s that was used ' +
                                 'for detection, not calculating lags',
                                 detection_template_name)
                    continue
                try:
                    array_picks_dict, array_baz_dict, array_app_vel_dict = (
                        find_array_picks_baz_appvel(
                            template_event, phase_hints=[phase_hint],
                            seisarray_prefixes=[seisarray_prefix]))
                except KeyError as e:
                    Logger.error(
                        'Cannot find backazimuth or apparent velocity for '
                        'array %s, phase %s, for template %s. Cannot compute '
                        'pick for array beam.', seisarray_prefix, phase_hint,
                        detection_template_name)
                    continue
                # Compute arrival at the reference station
                if stations_df is None:
                    msg = ('Need dataframe of station locations to compute '
                           + 'array beam pick')
                    raise ValueError(msg)
                event_with_array_picks = add_array_station_picks(
                    event, stations_df=stations_df, array_picks_dict=None,
                    array_baz_dict=array_baz_dict,
                    array_app_vel_dict=array_app_vel_dict,
                    seisarray_prefixes=[seisarray_prefix],
                    # TODO - need  to send baz and app-vel!!!!!
                    baz=array_baz_dict[seisarray_prefix][phase_hint],
                    app_vel=array_app_vel_dict[seisarray_prefix][phase_hint])
                ref_station = SEISARRAY_REF_STATIONS[seisarray_prefix]
                ref_picks = [pick for pick in event_with_array_picks.picks
                            if pick.waveform_id.station_code == ref_station]

                Logger.info('Got %s picks for reference station, adding these'
                            + ' to the event (has %s picks now)',
                            str(len(ref_picks)), str(len(picked_event.picks)))
                # Add new lag-calc pick for array's reference station
                for ref_pick in ref_picks:
                    # TODO: Add baz and app-vel to picks
                    # horizontal_slowness=kilometers2degrees(
                    #     1 / array_app_vel_dict[phase_hint]),
                    # backazimuth=array_baz_dict[phase_hint],
                    ref_pick.backazimuth = array_baz_dict[
                        seisarray_prefix][phase_hint]
                    ref_pick.horizontal_slowness = kilometers2degrees(
                         1 / array_app_vel_dict[seisarray_prefix][phase_hint])
                    picked_event.picks.append(ref_pick)

    return picked_cat


def add_array_waveforms_to_template(template, stream,
                                    seisarray_prefixes=SEISARRAY_PREFIXES):
    """
    function that
    - considers whether there are waveforms + picks in template that are
      recorded at an array
    - checks whether there are other waveforms for the same array in the stream
      (which are not yet part of template)
    - calls function to compute pick; adds waveform template for all array
      stations
    """
    return template


def stack_template_waveforms(template, seisarray_prefixes=SEISARRAY_PREFIXES):
    """
    function that
    - considers all waveforms recorded at one array
    - checks SNR; if SNR very high then no stacking required; if SNR low then
      stack
    - stacks waveforms recorded at the array
    - performs quality check on how well the stack correlates with the single
      trace
    - replaces each template seismogram at the array stations with the stacked
      seismogram
    """
    return template



# %% TEST
if __name__ == "__main__":
    from obspy.clients.filesystem.sds import Client
    from eqcorrscan.core.match_filter import Party, Tribe
    from robustraqn.templates_creation import create_template_objects
    from robustraqn.event_detection import run_day_detection
    from robustraqn.detection_picking import pick_events_for_day
    from robustraqn.quality_metrics import read_ispaq_stats
    from robustraqn.lag_calc_postprocessing import (
        check_duplicate_template_channels, postprocess_picked_events)

    parallel = True
    cores = 32
    make_templates = False
    make_detections = False

    # sfile = '/home/felix/Documents2/BASE/Detection/Bitdalsvatnet//SeisanEvents_06/15-
    # 2112-49L.S202108'
    # sfile = '/home/seismo/WOR/felix/R/SEI/REA/INTEU/2018/12/24-0859-35R.S201812'
    # sfile = '/home/seismo/WOR/felix/R/SEI/REA/INTEU/2000/07/18-0156-40R.S200007'
    sfile = '/home/felix/Documents2/BASE/Detection/Bitdalsvatnet/SeisanEvents_07/05-1742-43L.S202101'
    cat = read_nordic(sfile)
    event = cat[0]
    picks_before = str(len(event.picks))
    # array_picks_dict = extract_array_picks(event)
    # event2 = add_array_station_picks(event, array_picks_dict, stations_df)
    
    inv_file = '~/Documents2/ArrayWork/Inventory/NorSea_inventory.xml'
    # inv = read_inventory(os.path.expanduser(invile))
    inv = get_updated_inventory_with_noise_models(
        inv_file=os.path.expanduser(inv_file),
        pdf_dir=os.path.expanduser('~/repos/ispaq/WrapperScripts/PDFs/'),
        check_existing=True,
        outfile=os.path.expanduser(
            '~/Documents2/ArrayWork/Inventory/inv.pickle'))
    #inv = read_inventory(os.path.expanduser(inv_file))
    stations_df = stations_to_df(inv)
    # Add site names to stations_df (for info on array beams)
    site_names = []
    if 'site_name' not in stations_df.columns:
        for network in inv.networks:
            for station in network.stations:
                for channel in station.channels:
                    site_names.append(station.site.name)
    stations_df['site_name'] = site_names

    # seisarray_dict = get_array_stations_from_df(stations_df=stations_df)
    array_picks_dict = extract_array_picks(event=event)


# %%
    event2 = add_array_station_picks(
        event=event, array_picks_dict=array_picks_dict,
        stations_df=stations_df)

    write_select(catalog=Catalog([event2]), filename='array.out', userid='RR',
                 evtype='R', wavefiles=None, high_accuracy=True,
                 nordic_format='NEW')
    print(
        'Picks before: ' + picks_before +
        ' Picks after: ' + str(len(event2.picks)))

    selected_stations = ['ASK','BER','BLS5','DOMB','FOO','HOMB','HYA','KMY',
                        'ODD1','SKAR','SNART','STAV','SUE','KONO',
                        #'NAO01','NB201','NBO00','NC204','NC303','NC602',
                        'NAO00','NAO01','NAO02','NAO03','NAO04','NAO05',
                        'NB200','NB201','NB202','NB203','NB204','NB205',
                        'NBO00','NBO01','NBO02','NBO03','NBO04','NBO05',
                        'NC200','NC201','NC202','NC203','NC204','NC205',
                        'NC300','NC301','NC302','NC303','NC304','NC305',
                        'NC400','NC401','NC402','NC403','NC404','NC405',
                        'NC600','NC601','NC602','NC603','NC604','NC605']

    # selected_stations = ['ASK', 'BLS5', 'KMY', 'ODD1','SKAR']
    seisan_wav_path = (
        '/home/felix/Documents2/BASE/Detection/Bitdalsvatnet/SeisanEvents_07')
    if make_templates:
        tribe, _ = create_template_objects(
            [sfile], selected_stations, template_length=40, lowcut=4.0,
            highcut=19.9, min_snr=3, prepick=0.3, samp_rate=40.0,
            min_n_traces=13, seisan_wav_path=seisan_wav_path,
            inv=inv, remove_response=False, output='VEL', add_array_picks=True,
            parallel=parallel, cores=cores, write_out=True,
            templ_path='tests/data/Templates', make_pretty_plot=False,
            normalize_NSLC=True)
    else:
        tribe = Tribe().read('TemplateObjects/Templates_min13tr_1.tgz')

    pick_tribe = check_duplicate_template_channels(tribe.copy())

    date = pd.to_datetime('2021-01-05')
    ispaq = read_ispaq_stats(folder=
        '~/repos/ispaq/WrapperScripts/Parquet_database/csv_parquet',
        stations=selected_stations, startyear=date.year,
        endyear=date.year, ispaq_prefixes=['all'],
        ispaq_suffixes=['simpleMetrics','PSDMetrics'],
        file_type = 'parquet')

    client = Client('/data/seismo-wav/SLARCHIVE')
    if make_detections:
        [party, day_st] = run_day_detection(
            client=client, tribe=tribe, date=date, ispaq=ispaq, 
            selectedStations=selected_stations, inv=inv, xcorr_func='fftw',
            concurrency='concurrent',  parallel=parallel, cores=cores,
            n_templates_per_run=1, threshold=10, trig_int=20, multiplot=False,
            write_party=True, detection_path='tests/data/Detections',
            min_chans=3, return_stream=True)
    else:
        party = Party().read('tests/data/Detections/UniqueDet2021-01-05.tgz')
    # party[0].detections = [party[0][10]]
    # party[0].detections = [party[0][0]]
        


# %%
    export_catalog = pick_events_for_day(
        tribe=pick_tribe, det_tribe=tribe, templatePath=None,
        date=date, det_folder='tests/data/Detections', dayparty=party,
        ispaq=ispaq, clients=[client], relevantStations=selected_stations,
        array_lag_calc=True, inv=inv, parallel=True, cores=cores,
        write_party=False, n_templates_per_run=1, min_det_chans=5, min_cc=0.4,
        interpolate=True, archives=['/data/seismo-wav/SLARCHIVE'], 
        sfile_path='tests/data/Sfiles', operator='feha', stations_df=stations_df)


# %%
# picked_catalog = array_lac_calc(
#     day_st, export_catalog, party, tribe, stations_df, min_cc=0.4,
#     pre_processed=False, shift_len=0.8, min_cc_from_mean_cc_factor=0.6,
#     horizontal_chans=['E', 'N', '1', '2'], vertical_chans=['Z'],
#     parallel=False, cores=1, daylong=True)
    
# %%
    # arr_st_dict = extract_array_stream(day_st)



# %%
# pick_tribe = check_duplicate_template_channels(tribe.copy())
# %%
