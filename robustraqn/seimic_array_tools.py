
# %%
import os
from collections import defaultdict
from obspy import read_inventory
# import wcmatch
from wcmatch import fnmatch 
import pandas as pd
import numpy as np
import math
import traceback

import logging
Logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format=("%(asctime)s\t%(name)40s:%(lineno)s\t%(funcName)" +
            "20s()\t%(levelname)s\t%(message)s"))

from collections import Counter, defaultdict

from obspy import Stream
from obspy.signal.util import next_pow_2, util_geo_km
from obspy.io.nordic.core import read_nordic, write_select
from obspy.signal.array_analysis import get_timeshift
from obspy.geodetics import (degrees2kilometers, kilometers2degrees,
                             gps2dist_azimuth, locations2degrees)
from obspy.taup import TauPyModel
from obspy.taup.velocity_model import VelocityModel
from obspy.core.event import (Catalog, Pick, Arrival, WaveformStreamID,
                              CreationInfo)
from obspy import UTCDateTime
from obsplus.stations.pd import stations_to_df
from eqcorrscan.core.match_filter import Party


# List of extended glob-expressions that match all stations within a seismic
# array. E.g., NC3* will match all stations starting with "NC3" into one
# seismic array.
SEISARRAY_PREFIXES = [
    'NAO*', 'NBO*', '@(NB2*|NOA)', 'NC2*', 'NC3*', 'NC4*', 'NC6*',
    'NR[ABCD][0-9]', '@(ASK|ASK[1-5])', '@(MOR|MOR[1-8])', '@(KTK|KTK[1-6])',
    '@(ARCES|AR[ABCDE][0-9])', '@(SPITS|SP[ABC][0-5])', '@(BEAR|BJO*|BEA[1-6])',
    'OSE[0-9][0-9]', 'EKO[0-9]*', 'GRA[0-9][0-9]', 'SNO[0-9][0-9]',
    '@(EKA|ESK|EKB*|EKR*)', '@(ILAR|IL[0-3][0-9])', '@(YKA|YKA*[0-9])',
    '@(HNAR|HN[AB][0-6]|BAS02)',
    '@(OBS[0-6]|OBS1[1-2]'  # OBS
]

LARGE_APERTURE_SEISARRAY_PREFIXES = [
    '@(N[ABC][O2346]*|NOA|NR[ABCD][0-9]|NAO*|NBO*)',
    '@(ISF|BRBA|BRBB|BRB)',  # Svalbard west
    # Vestland
    '@(BLS|BLS[1-5])',
    '@(KMY|KMY2|NWG22)',
    '@(ODD|ODD1|NWG21|BAS1[0-4]|BAS19|BAS2[0-3]|REIN)',
    '@(BER|ASK|RUND|HN[AB][0-6]|BAS0[3-6]|BAS0D|BAS1[5-7]|' +
    'ASK[0-8]|SOTS|TB2S|OSGS|ESG|EGD|SOTS|ARNS|BT2[13])',
    '@(STAV,NWG29)',
    # Oestlandet
    '@(KONO|KON01|KON02|KON03)',
    # Midtnorge
    '@(AKN|JH1[02]|JH0[89])',
    '@(MOL,NWG03,JH06)',
    '@(FOO,NWG14)',
    '@(DOMB|JH0[34]|JH11)',
    '@(LADE|LENS|ODLO|TBLU|TRON|NWG01|N6004|N6005|N6006|N6007|N6008|N6132|SA55A)',
    '@(NSS|SA35)',
    # Nordland
    '@(ROEST|N2RO|N2VA)',
    '@(LOF|N2SV|N2VI)',
    '@(VAGH|KONS|STOK|LEIR|RAUS|MOR8|FLOS|STOK1|STOK2|NBB13|NBB14|NBB15)',
    '@(GILDE|NBB05|NBB30|MELSS|NBB17|N1304)',  # South 2 of NOrdland
    '@(FAUS|N2TV|NBB08|N2ST)',                 # North 2 of Nordland
    '@(VBYGD|STEI|N2LO|N2DI|N2HS|N2IH)',       # Northern part of Nordland
    '@(KUA|RATU|NIKU|KOVU|KIR|KURU)',
    '@(SALU|SA15|SA15A|N7010)',
    '@(DUNU|SA16|N7017)',
    '@(NIKU|KOVU|RATU|KUA|KIR|SA13)',
    '@(KTK|HEF|LP71|LP81)',
    '@(VADS|SA05|SA05A|SA03)',
    '@(HAMF|SA04|SA07)',
    '@(KIF|SA12|N2VI)',
    '@(TRO|JETT)',
    '@(KEV|SA10|ARCES|AR[ABCDE][0-9])',
    '@(SOFL|FAR|IF0[1-9]|IF10|IF07)']             # ARCES / Kevo

# List of beam reference points ("stations") for seismic arrays where the array
# beam reference point has a name that is not the same as any of the stations.
REF_STATIONS = ['ARCES', 'SPITS', 'NC2', 'NAO', 'NBO', 'NB2', 'NOA', 'NC2',
                'NC3', 'NC4', 'NC6', 'NRA0', 'BEAR', 'YKA', 'ILAR', 'EKA',
                'HNAR',
                'OBS01', 'ASK', 'KTK1', 'MOR8', 'BLS5',
                'OSE05', 'EKO1', 'GRA01', 'SNO01', 'EKO15', 'SNO05',
                'BRBB', 'KMY', 'ODD1', 'STAV', 'KONO', 'AKN', 'MOL', 'FOO',
                'DOMB', 'LADE', 'NSS', 'ROEST', 'KONS', 'GILDE', 'FAUS',
                'KTK', 'VADS', 'HAMF', 'TRO', 'KEV']

SEISARRAY_REF_STATIONS = {
    seisarray_prefix: ref_station
    for ref_station in REF_STATIONS for seisarray_prefix in SEISARRAY_PREFIXES
    if fnmatch.fnmatch(ref_station, seisarray_prefix, flags=fnmatch.EXTMATCH)}

# List of array beam points that have the same location as an actual seismic
# station that is part of the seismic array.
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
  'ILAR': 'IL01',
  'HNAR': 'HNA0'
}
SEISARRAY_REF_EQUIVALENT_STATIONS = defaultdict(
    str, SEISARRAY_REF_EQUIVALENT_STATIONS.items())

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


def get_station_sites(stations, seisarray_prefixes=SEISARRAY_PREFIXES):
    """
    Return a list of stations sites, i.e., for arrays, return the array name,
    while for single stations return the station code.
    """
    station_sites = []
    for station in stations:
        try:
            check_prefix = None
            for seisarray_prefix in seisarray_prefixes:
                if fnmatch.fnmatch(station, seisarray_prefix,
                                   flags=fnmatch.EXTMATCH):
                    check_prefix = seisarray_prefix
            ref_station = SEISARRAY_REF_STATIONS[check_prefix]
        except KeyError:
            ref_station = station
        station_sites.append(ref_station)
    return station_sites


def get_array_stations_from_df(stations_df=pd.DataFrame(),
                               seisarray_prefixes=SEISARRAY_PREFIXES):
    # array_list is a list of tuples, with the first tuple element containing
    # the array-prefix and the 2nd tuple element containing a list of stations
    # at the array.
    # seisarray_list = list()
    """
    Returns a dict containing the array prefixes and all stations that belong
    to the corresponding array prefix.
    
    :type stations_df: pandas.DataFrame
    :param stations_df: obsplus-stations dataframe with station information.
    :type seisarray_prefixes: list
    :param seisarray_prefixes:
        List of extended-glob patterns that match all station codes within
        seismic arrays.
        
    :returns: dictionary of keys:
        Seismic-array prefixes and values: pandas.dataframe of all stations
        that belong to the seismic array.
    :rtype: dict
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

    :type stations_df: pandas.DataFrame
    :param stations_df: obsplus-stations dataframe with station information.
    :type seisarray_prefixes: list
    :param seisarray_prefixes:
        List of extended-glob patterns that match all station codes within
        seismic arrays.
        
    :returns: pandas.dataframe of all stations that belong to the seismic array
    :rtype: pandas.Dataframe    
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
    
    :type event: class:`obspy.core.event.Event`
    :param event:
        event for which array-picks are to be extracted into a dictionary.
    :type seisarray_prefixes: list
    :param seisarray_prefixes:
        List of extended-glob patterns that match all station codes within
        seismic arrays.
        
    :returns:
        dictionary of seismic-array prefixes and all picks belonging to
        stations of the array
    :rtype: dict
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
    Return dict with array-prefix (keys) and relevant streams (values).

    :type st: class:`obspy.core.stream.Stream`
    :param st:
        stream for which array-traces are to be extracted into a dictionary.
    :type seisarray_prefixes: list
    :param seisarray_prefixes:
        List of extended-glob patterns that match all station codes within
        seismic arrays.
        
    :returns:
        dictionary of seismic-array prefixes and a
        class:`obspy.core.stream.Stream` containing all traces recorded at the
        stations of the array
    :rtype: dict
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
        # if there's only data for one station it doesn't need array handling
        array_stas = set([tr.stats.station for tr in array_st])
        if len(array_st) > 1 and len(array_stas) > 1:
            array_streams_dict[seisarray_prefix] = array_st
    return array_streams_dict


def get_geometry(stations_df, array_prefix='', coordsys='lonlat',
                 return_center=False, center_coord_output='lonlat'):
    """
    Modified from obspy.signal.array_analysis.get_geometry to allow calculation
    for a stations dataframe.
    (:copyright: The ObsPy Development Team (devs@obspy.org))
    
    Method to calculate the array geometry and the center coordinates in km

    :type stations_df: pandas.DataFrame
    :param stations_df:
        obsplus-stations dataframe with station information. Can contain either
        'latitude', 'longitude' (in degrees) and 'elevation' (in km), or 'x',
        'y', 'elevation' (in km) columns. See param ``coordsys``
    :type coordsys: str
    :param coordsys: valid values: 'lonlat' and 'xy', choose which dataframe
        columns to use for coordinates
    :type return_center: bool
    :param return_center: Returns the center coordinates as extra tuple
    :type center_coord_output: str
    :param center_coord_output:
        valid values: 'lonlat' and 'xy', choose which coordinate system
        returned center coordinates should be in

    :returns:
        Returns the geometry of the stations as 2d :class:`numpy.ndarray`. The
        first dimension are the station indexes with the same order as the
        stations in the dataframe object. The second index are the values of
        [lat, lon, elev] in km, last index contains center coordinates
        ([lat, lon, elev] or [x, y, z], see ``center_coord_output``)
    :rtype: :class:`numpy.ndarray`
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
            geometry[i, 2] = resp.elevation / 1000
        elif coordsys == 'xy':
            geometry[i, 0] = resp.x
            geometry[i, 1] = resp.y
            geometry[i, 2] = resp.elevation / 1000
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
        if center_coord_output == 'lonlat':
            center_geometry = np.array((center_lon, center_lat, center_h))
        elif center_coord_output == 'xy':
            center_geometry =  np.array((geometry[:, 0].mean(),
                                         geometry[:, 1].mean(),
                                         geometry[:, 2].mean()))
        else:
            raise ValueError(
                "center_coord_output must be one of 'lonlat', 'xy'")
        return np.c_[geometry.T, center_geometry].T
    else:
        return geometry, array_stations_df


def find_array_picks_baz_appvel(
        event, phase_hints=None, seisarray_prefixes=SEISARRAY_PREFIXES,
        array_picks_dict=None, array_baz_dict=None, array_app_vel_dict=None,
        vel_mod=None, taup_mod=None, mod_file=os.path.join(
            os.path.dirname(__file__), 'models','NNSN1D_plusAK135')):
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

    :type event: class:`obspy.core.event.Event`
    :param event:
        event for which array-picks are to be extracted into a dictionary.
    :type phase_hints: list or None
    :param phase_hints:
        list of phase-hints for which array picks should be extraced. ´´None´´
        means that the function returns information for all phase-hints for
        which picks exist at each array.
    :type seisarray_prefixes: list
    :param seisarray_prefixes:
        List of extended-glob patterns that match all station codes within
        seismic arrays.
    :type array_picks_dict: dict
    :param array_picks_dict:
        dictionary of seismic-array prefixes and a
        class:`obspy.core.stream.Stream` containing all traces recorded at the
        stations of the array (output from
        robustraqn.seismic_array_tools.extract_array_picks).
    :type array_baz_dict: dict or None
    :param array_baz_dict:
        2-level dictionary of seismic-array-prefixes (keys), phase_hints (keys)
        and backazimuths of the phase at the array (values). Leave as ´´None´´
        to let function find available information in the event.
    :type array_app_vel_dict: dict or None
    :param array_app_vel_dict:
        2-level dictionary of seismic-array-prefixes (keys), phase_hints (keys)
        and apparent velocities (in km/s) of the phase at the array (values).
        Leave as ´´None´´ to let function find available information in the
        event.
    :type vel_mod: class:`obspy.taup.velocity_model.VelocityModel`
    :param vel_mod:
        Velocity model that is to be used for computing time delays between
        stations of an array considering back-azimuth and apparent velocity.
    :type mod_file: str
    :param mod_file:
        Path to a model file (*.tvel and *.npz) containing a taup / velocity
        model conforming to class:`obspy.taup.velocity_model.VelocityModel`
        
    :returns:
        Tuple of three dictionaries of seismic-array prefixes and phase-hints
        as keys (2 levels), and the picks, backazimuth value, and apparent
        velocity value for each phase at each array (as dict-values).
    :rtype: tuple of (dict, dict, dict)
    """
    if vel_mod is None:
        vel_mod = VelocityModel.read_tvel_file(mod_file + '.tvel')
    if taup_mod is None:
        taup_mod = TauPyModel(mod_file + '.npz')
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
                baz_mean = np.nanmean(
                    [pick.backazimuth
                     for pick in array_picks_dict[seisarray_prefix][phase_hint]
                     if pick.backazimuth is not None])
                #if baz_mean is not None and not np.isnan(baz_mean):
                array_baz_dict[seisarray_prefix][phase_hint] = baz_mean
            if update_app_vels:
                app_vel_mean = np.nanmean(
                    [degrees2kilometers(1.0 / pick.horizontal_slowness)
                     for pick in array_picks_dict[seisarray_prefix][phase_hint]
                     if pick.horizontal_slowness is not None])
                #if app_vel_mean is not None and not np.isnan(app_vel_mean):
                array_app_vel_dict[seisarray_prefix][phase_hint] = app_vel_mean

            # If there is no origin (e.g., for newly picked events),
            # then don't try to compute BAZ or app-vel
            can_calculate_baz_appvel = True
            # TODO: select the origin with the most complete information
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
            # compute apparent velocity if there is no measurement for phase
            if update_app_vels and can_calculate_baz_appvel:
                calculated_app_vels = []
                if np.isnan(array_app_vel_dict[seisarray_prefix][phase_hint]):
                    for pick in array_picks_dict[seisarray_prefix][phase_hint]:
                        arrivals = [arr for arr in origin.arrivals if
                                    arr.pick_id == pick.resource_id and
                                    arr.distance is not None]
                        for arrival in arrivals:
                            # takeoff_angle known, but not incidence angle
                            if pick.phase_hint[0] == 'P':
                                vel = vel_mod.layers[0][2]
                            elif pick.phase_hint[0] == 'S':
                                vel = vel_mod.layers[0][4]
                            else:
                                continue
                            # Compute indicence angle at station with taup
                            try:
                                taup_arrivals = taup_mod.get_travel_times(
                                    source_depth_in_km=origin.depth / 1000,
                                    distance_in_degree=arrival.distance,
                                    phase_list=[arrival.phase])
                            except (TypeError, ValueError) as e:
                                Logger.exception(
                                    'Taupy failed when computing phase %s for '
                                    'array %s: ', phase_hint, seisarray_prefix)
                                # Logger.error(e)
                                Logger.error(traceback.print_exception(
                                    e, value=e, tb=e.__traceback__, limit=1))
                                continue
                            if not taup_arrivals:
                                continue
                            inc_angle = taup_arrivals[0].incident_angle
                            # TODO: correct? or get it from arrival.ray_param?
                            app_vel = vel / math.sin(math.radians(inc_angle))
                            calculated_app_vels.append(app_vel)
                            # Save incident angle with arrival
                            # try:
                            #     if arrival.extra.incident_angle:
                            #         continue
                            # except KeyError:
                            #     arrival.extra = {'incident_angle': {
                            #         'value': inc_angle, 'namespace': ''}}
                    app_vel = np.nanmean(calculated_app_vels)
                    if not np.isnan(app_vel):
                        array_app_vel_dict[seisarray_prefix][phase_hint] = (
                            app_vel)
    # TODO: if baz or app-vel ar none, compute them from 1-D velmodel rays:
    # taup_arrivals = taup_mod.get_travel_times(
    #     source_depth_in_km=origin.depth / 1000,
    #     distance_in_degree=arrival.distance,
    #     phase_list=[arrival.phase])
    # if not arrivals:
    #     continue
    # inc_angle = taup_arrivals[0].incident_angle
    # app_vel = vel / math.sin(math.radians(inc_angle))
    # calculated_app_vels.append(app_vel)
    # horizontal_slowness = arrival.ray_param * 360  # check!

    return array_picks_dict, array_baz_dict, array_app_vel_dict


def _check_extra_info(new_pick, pick):
    """
    Check that all attribute-keys required for a newly created pick are
    properly assigned for handling Nordic pick weight. If any attributes are
    missing, the pick may not conform to the xml-scheme for Quakeml and there
    will be an error on writing a quakeml-file. The function defines a Nordic
    pick-weight for a the new pick if there is one in the "old"" pick that it
    is compared to.

    :type new_pick: class:`obspy.core.event.Pick`
    :param new_pick:
        pick which shall be updated with pick-weight info from pick.
    :type pick: class:`obspy.core.event.Pick`
    :param pick:
        pick which shall be used to update pick-weight info in new_pick.

    :returns:
        pick with updated Nordic weight and namespace in
        pick.extra.nordic_pick_weight.
    :rtype: class:`obspy.core.event.Pick`
    """
    weight = None
    try:
        weight_val = new_pick['extra']['nordic_pick_weight']['value']
        weight = weight_val
    except KeyError:
        weight_val = None
    if 'extra' in pick.keys() or 'extra' in new_pick.keys():
        new_pick['extra'] = dict()
        new_pick['extra']['nordic_pick_weight'] = dict()
        try:
            weight_val_old = pick['extra']['nordic_pick_weight']['value']
            weight = weight_val_old
        except KeyError:
            weight_val_old = None
        if weight_val is not None and weight_val_old is not None:
            if int(weight_val) > int(weight_val_old):
                weight = weight_val
    # Set all info for pick weight
    if weight is not None:
        new_pick.extra = {
            'nordic_pick_weight': {
            'value': str(weight),
            'namespace': 'https://seis.geus.net/software/seisan/node239.html'}}
    return new_pick


def _check_existing_and_add_pick(event, new_pick):
    """
    pick to event, but only if event does not have similar pick yet
    """
    existing_array_picks_tuples = [
        (p.waveform_id.station_code, p.waveform_id.channel_code, p.phase_hint)
        for p in event.picks]
    new_pick_tuple = (new_pick.waveform_id.station_code,
                      new_pick.waveform_id.channel_code, new_pick.phase_hint)
    if new_pick_tuple not in existing_array_picks_tuples:
        event.picks.append(new_pick)
    return event


def add_array_station_picks(
        event, stations_df, array_picks_dict=None, array_baz_dict=None,
        array_app_vel_dict=None, baz=None, app_vel=None,
        seisarray_prefixes=SEISARRAY_PREFIXES, min_array_distance_factor=10,
        mod_file=os.path.join(os.path.dirname(__file__), 'models',
                                  'NNSN1D_plusAK135'), **kwargs):
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

    :type event: class:`obspy.core.event.Event`
    :param event:
        event for which array-picks are to be extracted into a dictionary.
    :type stations_df: pandas.DataFrame
    :param stations_df: obsplus-stations dataframe with station information.
    :type array_picks_dict: dict
    :param array_picks_dict:
        dictionary of seismic-array prefixes and a
        class:`obspy.core.stream.Stream` containing all traces recorded at the
        stations of the array (output from
        robustraqn.seismic_array_tools.extract_array_picks).
    :type array_baz_dict: dict or None
    :param array_baz_dict:
        2-level dictionary of seismic-array-prefixes (keys), phase_hints (keys)
        and backazimuths of the phase at the array (values). Leave as ´´None´´
        to let function find available information in the event.
    :type array_app_vel_dict: dict or None
    :param array_app_vel_dict:
        2-level dictionary of seismic-array-prefixes (keys), phase_hints (keys)
        and apparent velocities (in km/s) of the phase at the array (values).
        Leave as ´´None´´ to let function find available information in the
        event.
    :type baz: float or None
    :param baz:
        backazimuth value that should be used for all computations. Leave as
        ´´None´´ so as not to overwrite array/phase-specific values.
    :type app_vel: float or None
    :param app_vel:
        apparent-velocity value (km/s) that should be used for all
        computations. Leave as ´´None´´ so as not to overwrite array/phase-
        specific values.
    :type seisarray_prefixes: list
    :param seisarray_prefixes:
        List of extended-glob patterns that match all station codes within
        seismic arrays.
    :type min_array_distance_factor: int or float
    :param min_array_distance_factor:
        Factor that defines the minimum distance between seismic event and
        seismic array that needs to fulfilled so that a seismic array is
        treated as an array rather than single stations (e.g., compute arrival
        times at individual stations based on assumed plane wave). The function
        checks that the event-array distance is larger than the factor times
        the array's aperture.
    :type mod_file: str
    :param mod_file:
        Path to a *.tvel-file containing a velocity model conforming to
        class:`obspy.taup.velocity_model.VelocityModel`
        
    :returns:
        Event with picks added for all individual stations at seismic arrays
        for which picks could be computed from array arrivals.

    :rtype: class:`obspy.core.event.Event`
    """
    n_picks_before = len(event.picks)
    vel_mod = VelocityModel.read_tvel_file(mod_file + '.tvel')
    taup_mod = TauPyModel(mod_file + '.npz')
    update_app_vel = True
    if app_vel is not None:
        update_app_vel = False
    update_baz = True
    if baz is not None:
        update_baz = False
    # 2. For each array:
    #   2.1 collect all equivalent picks from the same array
    #       lets either user supply baz- and app-vel dicts, or lets code find
    #       the values
    array_picks_dict, array_baz_dict, array_app_vel_dict = (
        find_array_picks_baz_appvel(
            event, array_picks_dict=array_picks_dict, vel_mod=vel_mod,
            seisarray_prefixes=seisarray_prefixes, taup_mod=taup_mod))
    # Try to find the best origin - the preferred one if it has lon/lat, else
    # check further
    origin = event.preferred_origin()
    if origin is None or origin.latitude is None or origin.longitude is None:
        for orig in event.origins:
            if (orig is not None and orig.latitude is not None and
                    orig.longitude is not None):
                origin = orig
                break

    #   2.2. get array delays for relevant arrival.
    for seisarray_prefix, pha_picks_dict in array_picks_dict.items():
        # array_picks --> pha_picks_dict
        array_geometry, array_stations_df = get_geometry(
            stations_df, array_prefix=seisarray_prefix, coordsys='lonlat',
            return_center=False)
        array_geo_center = get_geometry(
            array_stations_df, array_prefix=seisarray_prefix,
            coordsys='lonlat', return_center=True, center_coord_output='xy')
        array_center = array_geo_center[-1]

        # Compute array apperture - find maximum distance between array station
        # and array center, times two.
        array_aperture = [
            np.sqrt((ar_geo[0] - array_center[0]) ** 2 +
                    (ar_geo[1] - array_center[1]) ** 2 +
                    (ar_geo[2] - array_center[2]) ** 2)
            for ar_geo in array_geometry]
        array_aperture = 2 * max(array_aperture)
        # Can't check array-aperture vs distance if there's no origin solution.
        if (origin is not None and origin.latitude is not None and
                origin.longitude is not None):
            event_array_dist = degrees2kilometers(
                locations2degrees(origin.latitude, origin.longitude,
                                  array_center[1], array_center[0]))
            event_array_dist = np.sqrt(
                event_array_dist ** 2 + origin.depth ** 2)
            # Check if array is far enough from event to assume plane wave
            if (event_array_dist / min_array_distance_factor
                    <= array_aperture):
                Logger.info(
                    'Distance between event %s and array %s is too small, not '
                    'computing array arrivals.', event.short_str(),
                    seisarray_prefix)
                continue

        phase_hints = pha_picks_dict.keys()
        # 2.3 compute average array pick at reference site.
        #     i now have BAZ, app-vel, distance, and velocity model
        # app_vel = degrees2kilometers(1.0 / pick.horizontal_slowness)
        for phase_hint in phase_hints:
            Logger.info(
                'Computing average picks for %s at stations of array %s',
                phase_hint, seisarray_prefix)
            if update_app_vel:
                app_vel = array_app_vel_dict[seisarray_prefix][phase_hint]
            horizontal_slowness_spkm = 1.0 / app_vel
            if update_baz:
                baz = array_baz_dict[seisarray_prefix][phase_hint]
            # Check for missing information for array arrival
            if (baz is None or np.isnan(baz)
                    or horizontal_slowness_spkm is None
                    or np.isnan(horizontal_slowness_spkm)):
                Logger.error(
                    'Cannot compute timeshifts for array %s arrival %s - '
                    'missing backazimuth (%s) and/or slowness (%s). You may '
                    'need to locate or update event %s.', seisarray_prefix,
                    phase_hint, str(baz), str(horizontal_slowness_spkm),
                    event.short_str())
                continue
            sll_x = math.cos(math.radians(baz)) * horizontal_slowness_spkm
            sll_y = math.sin(math.radians(baz)) * horizontal_slowness_spkm
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
                    Logger.error(
                        'Missing information for array station %s, cannot com'
                        'pute array arrivals.', pick.waveform_id.station_code)
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
            # In case list is empty in case of missing station locations:
            if not pick_time_list_ns:
                continue
            # pick-time average needs to be calculated from epoch-seconds
            pick_time_averages = np.mean(pick_time_list_ns, axis=0) / 1e9
            pick_times_av_utc = [UTCDateTime(time_av)
                             for time_av in (pick_time_averages)]
            # 3. add picks for array stations that did not have pick to pick-list
            # Only add pick for array station if the>re is no equivalent pick
            # for that station yet (check phase_hint and station.code)
            for row_n, pick_time in enumerate(pick_times_av_utc):
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
                        horizontal_slowness=1 / kilometers2degrees(
                            1 / horizontal_slowness_spkm),
                        # backazimuth=array_baz_dict[seisarray_prefix][phase_hint],
                        backazimuth=baz,
                        onset=onset,
                        polarity=polarity,
                        evaluation_mode='automatic',
                        creation_info=CreationInfo(agency_id='RR'))
                    new_pick.extra = {'nordic_pick_weight': {'value': 2}}
                    new_pick = _check_extra_info(new_pick, pick)
                    # Check that similar array-pick isn't in pick-list yet
                    event =_check_existing_and_add_pick(event, new_pick)

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


def _find_associated_detection_id(party, event_index):
    """
    For the n-th event in a catalog received from lag-calc, find the associated
    detection for the event and return the detection id.
    """
    nd = 0
    for family in party:
        for detection in family:
            if nd == event_index:
                detection_id = detection.id
                return detection_id
            nd += 1
    return ''


def _check_picks_within_shiftlen(party, event, detection_id, shift_len):
    """
    Check whether pick is within shift_len
    """
    keep_picks = list()
    detection_events = [det.event for fam in party for det in fam
                        if det.id == detection_id]
    if len(detection_events) != 1:
        Logger.error(
            'Found more than one matching detection events when comparing '
            'new array picks against detection %s', detection_id)
        return event
    detection_event = detection_events[0]
    for pick in event.picks:
        detection_picks = [
            p for p in detection_event.picks
            if p.phase_hint == pick.phase_hint and
            (p.waveform_id.station_code == pick.waveform_id.station_code)]
        if len(detection_picks) != 1:
            continue
        detection_pick = detection_picks[0]
        if abs(detection_pick.time - pick.time) <= shift_len:
            keep_picks.append(pick)
        else:
            Logger.info('Pick for phase %s for array station %s outside '
                        'shift_len for detection %s', pick.phase_hint,
                        pick.waveform_id.station_code, detection_id)
    event.picks = keep_picks
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
    Obtain a cross-correlation pick for seismic arrays. This function should be
    invoked after `EQcorrscan.core.lag_calc` has computed picks for individual
    traces.

    Here is what this function does
    1. for every array:
        - select all waveforms for the array
        - for every phase:
            - run lag_calc. Do not allows single traces to shift, but only
              allow the whole stream to move together.
            - min_cc should be selected based on the number of array stations
              and other contributing factors.
        - add one single pick for the array's reference station for each phase

    :type st: obspy.core.stream.Stream
    :param st: All the data needed to cut from - can be a gappy Stream.
    :type picked_cat: class:`obspy.core.event.catalog`
    :type picked_cat:
        catalog containing events for which picks have been obtained on
        individual channels with lag_calc.
    :type party: `EQcorrscan.core.match_filter.tribe`
    :param party:
        Party of families of detections for the events in `picked_cat`.
    :type tribe: `EQcorrscan.core.match_filter.tribe`
    :param tribe: tribe of templates used for detection of events.
    :type stations_df: pandas.DataFrame
    :param stations_df: obsplus-stations dataframe with station information.
    :type seisarray_prefixes: list
    :param seisarray_prefixes:
        List of extended-glob patterns that match all station codes within
        seismic arrays.
    :type pre_processed: bool
    :param pre_processed:
        Whether the stream has been pre-processed or not to match the
        templates. See note below.
    :type shift_len: float
    :param shift_len:
        Shift length allowed for the pick in seconds, will be plus/minus
        this amount - default=0.8
    :type min_cc: float
    :param min_cc:
        Minimum cross-correlation value to be considered a pick,
        default=0.4.
    :type min_cc_from_mean_cc_factor: float
    :param min_cc_from_mean_cc_factor:
        If set to a value other than None, then the minimum cross-
        correlation value for a trace is set individually for each
        detection based on:
        min(detect_val / n_chans * min_cc_from_mean_cc_factor, min_cc).
        default is 0.6.
    :type horizontal_chans: list
    :param horizontal_chans:
        List of channel endings for horizontal-channels, on which S-picks
        will be made.
    :type vertical_chans: list
    :param vertical_chans:
        List of channel endings for vertical-channels, on which P-picks
        will be made.
    :type cores: int
    :param cores:
        Number of cores to use in parallel processing, defaults to one.
    :type interpolate: bool
    :param interpolate:
        Interpolate the correlation function to achieve sub-sample
        precision.
    :type plot: bool
    :param plot:
        To generate a plot for every detection or not, defaults to False
    :type parallel: bool
    :param parallel: Turn parallel processing on or off.
    :type process_cores: int
    :param process_cores:
        Number of processes to use for pre-processing (if different to
        `cores`).
    :type ignore_length: bool
    :param ignore_length:
        If using daylong=True, then dayproc will try check that the data
        are there for at least 80% of the day, if you don't want this check
        (which will raise an error if too much data are missing) then set
        ignore_length=True.  This is not recommended!
    :type ignore_bad_data: bool
    :param ignore_bad_data:
        If False (default), errors will be raised if data are excessively
        gappy or are mostly zeros. If True then no error will be raised,
        but an empty trace will be returned (and not used in detection).
        
    :returns:
        catalog containing events where lag-calc based cross-correlation picks
        have been added for phases arriving at seismic arrays.
    :rtype picked_cat: class:`obspy.core.event.catalog`
    """
    if len(picked_cat) == 0:
        return picked_cat
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
            array_party = Party(
                [family.copy() for family in party
                 if len(
                     [tr for tr in family.template.st if fnmatch.fnmatch(
                         tr.stats.station, seisarray_prefix,
                         flags=fnmatch.EXTMATCH)]) > 0])
            # Factor to relax cc-requirement by - noise of stacked traces
            # should in theory reduce by sqrt(n_traces)
            cc_relax_factor = np.sqrt(len(array_st_dict[seisarray_prefix]))
            Logger.info('Preparing traces for array %s, for picking phase %s'
                        ' with lag-calc. CC relax factor is %s',
                        seisarray_prefix, phase_hint, str(cc_relax_factor))
            # From (preprocessed?) stream select only traces for current array
            array_catalog = Catalog()
            array_catalog = array_party.lag_calc(
                array_st_dict[seisarray_prefix], shift_len=0.0,
                pre_processed=pre_processed, min_cc=min_cc/cc_relax_factor,
                min_cc_from_mean_cc_factor=(
                    min_cc_from_mean_cc_factor/cc_relax_factor),
                horizontal_chans=horizontal_chans,
                vertical_chans=vertical_chans, interpolate=interpolate,
                plot=plot, overlap=overlap, daylong=daylong,
                parallel=parallel, cores=cores, process_cores=1,
                ignore_bad_data=ignore_bad_data,
                ignore_length=ignore_length, **kwargs)

            Logger.info('Got new array picks for %s events.',
                        str(len(array_catalog)))
            # sort picks into previously lag-calc-picked catalog
            for i_event, event in enumerate(array_catalog):
                # find the event that was picked from the same detection for
                # the whole network
                detection_id = _find_associated_detection_id(
                    party=array_party, event_index=i_event)
                # Check whether pick is within shift_len
                event = _check_picks_within_shiftlen(
                    party=array_party, event=event, detection_id=detection_id,
                    shift_len=shift_len)
                if len(event.picks) == 0:
                    continue
                # Figure out beam / reference station pick from all traces
                try:
                    ref_station_code = SEISARRAY_REF_STATIONS[seisarray_prefix]
                except KeyError:
                    Logger.warning(
                        'No reference station for array %s defined, cannot'
                        ' add pick.', seisarray_prefix)
                    continue
                ref_equi_stacode = SEISARRAY_REF_EQUIVALENT_STATIONS[
                    ref_station_code]
                picked_event = [ev for ev in picked_cat
                                if ev.resource_id == event.resource_id]
                # There may be no matching picked event if CCCSUM check (lag-
                # calc vs match_filter CCC) failed in xcorr_pick_family.
                if len(picked_event) == 0:
                    continue
                picked_event = picked_event[0]
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
                    Logger.info(
                        'There is already a pick for array beam %s, phase %s, '
                        'detection %s, not adding any more picks.',
                        seisarray_prefix, phase_hint, detection_id)
                    continue
                if len(existing_ref_equivalent_picks) > 0:
                    Logger.info(
                        'Adding array pick for array %s beam for phase %s, '
                        'detection %s.', seisarray_prefix, phase_hint,
                        detection_id)
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
                        # Set a reduced Nordic pick weight for automatically
                        # added pick
                        new_pick.extra = {'nordic_pick_weight': {'value': 2}}
                        new_pick = _check_extra_info(new_pick, equi_pick)
                        # TODO can I add baz and app-vel here?
                        # horizontal_slowness=1 / kilometers2degrees(
                        #     1 / horizontal_slowness_spkm),
                        # backazimuth=array_baz_dict[seisarray_prefix][
                        #   phase_hint],
                        # backazimuth=baz,
                        picked_event = _check_existing_and_add_pick(
                            picked_event, new_pick)
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
                    Logger.error('Could not find template %s that was used '
                                 'for detection, not calculating lags',
                                 detection_template_name)
                    continue
                try:
                    array_picks_dict, array_baz_dict, array_app_vel_dict = (
                        find_array_picks_baz_appvel(
                            template_event, phase_hints=[phase_hint],
                            seisarray_prefixes=[seisarray_prefix]))
                except KeyError as e:
                    Logger.warning(
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
                    baz=array_baz_dict[seisarray_prefix][phase_hint],
                    app_vel=array_app_vel_dict[seisarray_prefix][phase_hint])
                ref_station = SEISARRAY_REF_STATIONS[seisarray_prefix]
                ref_picks = [pick for pick in event_with_array_picks.picks
                            if pick.waveform_id.station_code == ref_station
                            and pick.phase_hint == phase_hint]

                Logger.info('Got %s picks for reference station, adding these'
                            + ' to the event (has %s picks now)',
                            str(len(ref_picks)), str(len(picked_event.picks)))
                # Add new lag-calc pick for array's reference station
                for ref_pick in ref_picks:
                    ref_pick.backazimuth = array_baz_dict[
                        seisarray_prefix][phase_hint]
                    app_vel = array_app_vel_dict[seisarray_prefix][phase_hint]
                    # TODO: This should never be nan really
                    if np.isnan(app_vel):
                        continue
                    ref_pick.horizontal_slowness = 1 / kilometers2degrees(
                         app_vel)
                    picked_event.picks.append(ref_pick)

    return picked_cat


def get_updated_stations_df(inv):
    """
    Updated stations-dataframe with a column that contains the site-names for
    each station.

    :type stations_df: pandas.DataFrame
    :param stations_df: obsplus-stations dataframe with station information.

    :returns: obsplus-stations dataframe with station information.
    :rtype: pandas.DataFrame
    """
    if inv is None:
        return pd.DataFrame()
    stations_df = stations_to_df(inv)
    # Add site names to stations_df (for info on array beams)
    site_names = []
    if 'site_name' not in stations_df.columns:
        for network in inv.networks:
            for station in network.stations:
                for channel in station.channels:
                    site_names.append(station.site.name)
    stations_df['site_name'] = site_names
    return stations_df


# def add_array_beam_waveforms_to_template(
#         template, stream, seisarray_prefixes=SEISARRAY_PREFIXES):
#     """
#     TODO
#     function that
#     - considers whether there are waveforms + picks in template that are
#       recorded at an array
#     - checks whether there are other waveforms for the same array in the stream
#       (which are not yet part of template)
#     - calls function to compute pick; adds waveform template for all array
#       stations
#     """
#     return template


# def stack_template_waveforms(template, seisarray_prefixes=SEISARRAY_PREFIXES):
#     """
#     TODO
#     function that
#     - considers all waveforms recorded at one array
#     - checks SNR; if SNR very high then no stacking required; if SNR low then
#       stack
#     - stacks waveforms recorded at the array
#     - performs quality check on how well the stack correlates with the single
#       trace
#     - replaces each template seismogram at the array stations with the stacked
#       seismogram
#     """
#     return template


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
    from robustraqn.spectral_tools import get_updated_inventory_with_noise_models

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
            selected_stations=selected_stations, inv=inv, xcorr_func='fftw',
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
        tribe=pick_tribe, det_tribe=tribe, template_path=None,
        date=date, det_folder='tests/data/Detections', dayparty=party,
        ispaq=ispaq, clients=[client], relevant_stations=selected_stations,
        array_lag_calc=True, inv=inv, parallel=True, cores=cores,
        write_party=False, n_templates_per_run=1, min_det_chans=5, min_cc=0.4,
        interpolate=True, archives=['/data/seismo-wav/SLARCHIVE'], 
        sfile_path='tests/data/Sfiles', operator='feha', stations_df=stations_df)


# %%
    picked_catalog = array_lac_calc(
        day_st, export_catalog, party, tribe, stations_df, min_cc=0.4,
        pre_processed=False, shift_len=0.8, min_cc_from_mean_cc_factor=0.6,
        horizontal_chans=['E', 'N', '1', '2'], vertical_chans=['Z'],
        parallel=False, cores=1, daylong=True)
    
# %%
    # arr_st_dict = extract_array_stream(day_st)



# %%
# pick_tribe = check_duplicate_template_channels(tribe.copy())
# %%
