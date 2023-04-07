
import numpy as np
import pandas as pd

from timeit import default_timer

from obspy.core.trace import Trace
from obspy.core.trace import _add_processing_info
from obspy.core.inventory import Inventory
from obspy.core.util.attribdict import AttribDict

import logging
Logger = logging.getLogger(__name__)
#logging.basicConfig(
#    level=logging.INFO,
#    format=("%(asctime)s\t%(name)40s:%(lineno)s\t%(funcName)20s()\t%" +
#            "(levelname)s\t%(message)s"))


@_add_processing_info
def automatic_gain_control(self, agc_window_sec, agc_method='gismo',
                           method_exec='new', **kwargs):
# def _agc(st, agc_window_sec, agc_method='gismo', method_exec='old'):
    """
    Apply automatic gain correction (AGC) to traces in an ObsPy Stream object.
    This function is copied and modified (sped up and monkey-pathced) from
    package  https://github.com/uafgeotools/rtm which is originally published
    under MIT license:

MIT License

Copyright (c) 2019-2022 The University of Alaska Fairbanks

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

    Args:
        st (:class:`~obspy.core.stream.Stream`): Stream containing waveforms to
            be processed
        agc_window_sec (int or float): AGC window [s]. A shorter time window results
            in a more aggressive AGC effect (i.e., increased gain for quieter
            signals)
        agc_method (str): One of `'gismo'` or `'walker'` (default: `'gismo'`)

            * `'gismo'` A Python implementation of ``agc.m`` from the GISMO
              suite:

              https://github.com/geoscience-community-codes/GISMO/blob/master/core/%40correlation/agc.m

              It preserves the relative amplitudes of traces (i.e. doesn't
              normalize) but is limited in how much in can boost quiet sections
              of waveform.

            * `'walker'` An implementation of the AGC algorithm described in
              Walker *et al.* (2010), paragraph 22:

              https://doi.org/10.1029/2010JB007863

              (The code is adopted from Richard Sanderson's version.) This
              method scales the amplitudes of the resulting traces between
              :math:`[-1, 1]` (or :math:`[0, 1]` for envelopes) so inter-trace
              amplitudes are not preserved. However, the method produces a
              stronger AGC effect which may be desirable depending upon the
              context.

    Returns:
        :class:`~obspy.core.stream.Stream`: Copy of input Stream with AGC
        applied
    """
    Logger.debug('Applying AGC to %s', self)
    if len(self.data) == 0:
        Logger.warning('Empty trace, cannot apply AGC to %s', self)
        return self
    if agc_method == 'gismo':
        win_samp = int(self.stats.sampling_rate * agc_window_sec)
        if win_samp >= self.stats.npts:
            Logger.warning(
                'AGC window (%s samples) is longer than trace %s, hence '
                'amplitudes remain unchanged.', str(win_samp), self)
            return self
        elif method_exec == 'old':
            scale = np.zeros(self.count() - 2 * win_samp)
            for i in range(-1 * win_samp, win_samp + 1):
                scale = scale + np.abs(
                    self.data[win_samp + i : win_samp + i + scale.size])
        elif method_exec == 'new':
            # Quicker version of above code (?)
            # slower:
            # scales = np.zeros((self.count() - 2 * win_samp, 2 * win_samp + 3))
            # for ns, i in enumerate(range(-1 * win_samp, win_samp + 1)):
            #     scales[:, ns] = np.abs(self.data[
            #         win_samp + i:win_samp + i + scale.size])
            # scale = np.sum(scales, 1)
            n_width = 2 * win_samp + 1
            # faster!
            # scale = scipy.convolve(
            #     self.data, np.ones(n_width), mode='valid') / n_width
            # even faster!
            # scale = np.convolve(np.abs(self.data), np.ones(n_width),
            #                     'valid') / n_width
            # fastest!
            scale = np.array(pd.DataFrame(np.abs(self.data)).rolling(
                n_width, min_periods=n_width, center=True).mean())
            scale = scale[win_samp : -win_samp].ravel()

        # Fill any Zeros in scale
        if np.any(scale==0): # or np.any(np.isnan(scale))
            Logger.debug('Filling some zeros in AGC scale.')
            scale_s = pd.Series(scale)
            scale_s.replace(to_replace=0, method='ffill', inplace=True)
            scale = scale_s.values

        scale = scale / scale.mean()  # Using max() here may better
                                      # preserve inter-trace amplitudes

        # Fill out the ends of scale with its first/last values
        try:
            scale = np.hstack((np.ones(win_samp) * scale[0],
                               scale,
                               np.ones(win_samp) * scale[-1]))

            self.data = self.data / scale  # "Scale" the data, sample-by-sample
            # self.stats.processing.append('AGC applied via '
            #                             f'_agc(agc_window_sec={agc_window_sec}, '
            #                             f'method=\'{agc_method}\')')
        except IndexError:
            Logger.error('Could not compute AGC scaling for %s', self)

    elif agc_method == 'walker':
        half_win_samp = int(self.stats.sampling_rate * agc_window_sec / 2)
        if half_win_samp >= self.stats.npts:
            Logger.warning(
                'AGC window (%s samples) is longer than trace %s, hence '
                'amplitudes remain unchanged.', str(half_win_samp), self)
            return self
        elif method_exec == 'old':
            scale = []
            for i in range(half_win_samp, self.count() - half_win_samp):
                # The window is centered on index i
                scale_max = np.abs(self.data[
                    i - half_win_samp : i + half_win_samp]).max()
                scale.append(scale_max)
        elif method_exec == 'new':
            n_width = 2 * half_win_samp
            scale = np.array(pd.DataFrame(np.abs(self.data)).rolling(
                n_width, min_periods=n_width, center=True).max())
            scale = scale[half_win_samp : -half_win_samp].ravel()
            # need to remove last sample so that it is the same as old
            # implementation -- or NOT?!
            # scale = scale[:-1]
        # Fill any Zeros in scale
        if np.any(scale==0): # or np.any(np.isnan(scale))
            Logger.debug('Filling some zeros in AGC scale.')
            scale_s = pd.Series(scale)
            scale_s.replace(to_replace=0, method='ffill', inplace=True)
            scale = scale_s.values

        try:
            # Fill out the ends of scale with its first/last values
            scale = np.hstack((np.ones(half_win_samp) * scale[0],
                            scale,
                            np.ones(half_win_samp) * scale[-1]))
            self.data = self.data / scale  # "Scale" the data, sample-by-sample
            # self.stats.processing.append('AGC applied via '
            #                             f'_agc(agc_window_sec={agc_window_sec}, '
            #                             f'agc_method=\'{agc_method}\')')
        except IndexError:
            Logger.error('Could not compute AGC scaling for %s', self)
    else:
        raise ValueError(f'AGC method \'{agc_method}\' not recognized. Method '
                         'must be either \'gismo\' or \'walker\'.')
    return self


def try_remove_response(self, inv, taper_fraction=0.05, pre_filt=None,
                        output='VEL', gain_traces=True, water_level=10,
                        **kwargs):
    """Internal function that tries to remove the response from a trace

    :param tr: trace for response-removal
    :type tr: :class:`obspy.core.trace.Trace`
    :param inv: inventory containing all available responses
    :type inv: :class:`obspy.core.inventory.Inventory`
    :param taper_fraction: fraction of trace to be tapered, defaults to 0.05
    :type taper_fraction: float, optional
    :param pre_filt:
        list of optional pre-filtering parameters, defaults to None
    :type pre_filt: list of float, optional
    :param output:
        physical output magnitude after response removal (displacement,
        velocity or acceleration), defaults to 'DISP'
    :type output: 'DISP', 'VEL', or 'ACC', optional
    :param gain_traces:
        whether to multiply traces with a gain-factor so that corrected trace-
        values fit into float16 / float32 variables (else: float16-error),
        defaults to True
    :type gain_traces: bool, optional

    :return:
        trace with response removed (if no response found, dummy filter can be
        applied to trace)
    :rtype: :class:`obspy.core.trace.Trace`
    """
    # remove response
    outtic = default_timer()
    found_matching_resp = False
    try:
        self.remove_response(
            inventory=inv, output=output, water_level=water_level,
            pre_filt=pre_filt, zero_mean=True, taper=True,
            taper_fraction=taper_fraction)
        sel_inv = inv.select(
            network=self.stats.network, station=self.stats.station,
            location=self.stats.location, channel=self.stats.channel,
            time=self.stats.starttime)
        found_matching_resp = True
    except Exception as e:
        # Try to find the matching response
        found_matching_resp, self, sel_inv = self.try_find_matching_response(
            inv)
        if not found_matching_resp:
            Logger.warning('Finally cannot remove reponse for %s - no match '
                           'found', str(self))
            Logger.warning(e)
        else:
            # TODO: what if trace's location code is empty, and there are
            # multiple instruments at one station that both match the trace in
            # a channel code?
            try:
                self.remove_response(
                    inventory=sel_inv, output=output, water_level=water_level,
                    pre_filt=pre_filt, zero_mean=True, taper=True,
                    taper_fraction=taper_fraction)
            except Exception as e:
                found_matching_resp = False
                Logger.warning('Finally cannot remove reponse for %s - no '
                               'match found', str(self))
                Logger.warning(e)
        # IF reponse isn't found, then adjust amplitude to something
        # similar to the properly corrected traces
        if not found_matching_resp:
            self.data = self.data / 1e6

    # TODO: remove this once proper writing / reading of trace
    #       processing is implemented somewhere else...
    if not hasattr(self.stats, 'extra'):
        self.stats.extra = AttribDict()
    # Use int (0) instead of Fale for better I/O compatibility - bool is
    # subtype of int anyways
    self.stats.extra.update({'response_removed': 0})
    if found_matching_resp:
        # keep this information in trace.stats.extra for now to allow
        # proper relative amplitde / magnitude calculation
        self.stats.extra.update({'response_removed': 1})
    # Set station coordinates
    # initialize
    self.stats["coordinates"] = {}
    self.stats["coordinates"]["latitude"] = np.NaN
    self.stats["coordinates"]["longitude"] = np.NaN
    self.stats["coordinates"]["elevation"] = 0
    self.stats["coordinates"]["depth"] = 0
    self.stats['distance'] = np.NaN
    # try to set coordinates from channel-info; but using station-info is
    # also ok
    stachan_info = None
    try:
        stachan_info = sel_inv.networks[0].stations[0]
        stachan_info = sel_inv.networks[0].stations[0].channels[0]
    except Exception as e:
        Logger.warning('Cannot find metadata for trace %s', self.id)
        pass
    try:
        self.stats["coordinates"]["latitude"] = stachan_info.latitude
        self.stats["coordinates"]["longitude"] = stachan_info.longitude
        self.stats["coordinates"]["elevation"] = stachan_info.elevation
        self.stats["coordinates"]["depth"] = stachan_info.depth
    except Exception as e:
        Logger.warning('Could not set all station coordinates for %s.', self.id)
    # Gain all traces to avoid a float16-zero error
    # basically converts from m to um (for displacement) - nm
    if gain_traces:
        self.data = self.data * 1e6
    # Now convert back to 32bit-double to save memory ! (?)
    # if np.dtype(self.data[0]) == 'float64':
    self.data = np.float32(self.data)
    try:
        self.stats.mseed.encoding = 'FLOAT32'
    except (KeyError, AttributeError):
        pass
    # before response removed: self.data is in int32
    # after response removed, self.data is in float64)

    outtoc = default_timer()
    if (outtoc - outtic) > 3:
        Logger.debug(
            'Response-removal of trace %s took: {0:.4f}s'.format(
                outtoc - outtic), self.id)

    # Check that data are not NaN:
    if np.isnan(self.data).any():
        Logger.warning('Data for trace %s contain NaN after response-removal,'
                       + ' will discard this trace.', str(self))
        return None

    return self


def try_find_matching_response(self, inv, **kwargs):
    """Try to remove response from one trace

    :param tr: trace for response-removal
    :type tr: :class:`obspy.core.trace.Trace`
    :param inv: inventory containing all available responses
    :type inv: :class:`obspy.core.inventory.Inventory`

    :return: trace with response removed
    :rtype: :class:`obspy.core.trace.Trace`

    Note: If code doesn't find the response, then assume that the trace's
          metadata lack network or location code. Look for reponse in inv-
          entory that has the same station code, and check start/endtimes
          of channel - correct trace stats if there's a match.
          :returns: bool, trace, inventory

    Logic:
    1. only location code is empty:
    2. neither location nor network codes are empty, but there is a response
       for an empty location code.
    2. network code is empty
    3. if not found, try again and allow any location code
    4 if not found, check if the channel code may contain a space or zero in
       the middle
       5 if not found, allow space in channel and empty network
       6 if not found, allow space in channel and empty location
       7 if not found, allow space in channel, empty network, and
         empty location
    """
    found = False
    # 1. only location code is empty:
    if ((self.stats.location == '' or self.stats.location == '--')
            and not self.stats.network == ''):
        temp_inv = inv.select(network=self.stats.network,
                             station=self.stats.station,
                             channel=self.stats.channel)
        found = False
        for network in temp_inv.networks:
            for station in network.stations:
                for channel in station.channels:
                    if self.response_stats_match(channel):
                        self.stats.location = channel.location_code
                        inv = self.return_matching_response(
                            inv, network, station, channel)
                        return True, self, inv
    # 2. neither location nor network codes are empty, but there is a response
    #    for an empty location code.
    if (not (self.stats.location == '' or self.stats.location == '--') and
            not self.stats.network == ''):
        temp_inv = inv.select(network=self.stats.network,
                             station=self.stats.station,
                             channel=self.stats.channel)
        found = False
        for network in temp_inv.networks:
            for station in network.stations:
                for channel in station.channels:
                    if self.response_stats_match(channel):
                        self.stats.location = channel.location_code
                        inv = self.return_matching_response(
                            inv, network, station, channel)
                        return True, self, inv
    # 2. network code is empty
    if self.stats.network == '':
        temp_inv = inv.select(station=self.stats.station,
                             location=self.stats.location,
                             channel=self.stats.channel)
        found = False
        for network in temp_inv.networks:
            for station in network.stations:
                # chan_codes = [c.code for c in station.channels]
                for channel in station.channels:
                    if self.response_stats_match(channel):
                        self.stats.network = network.code
                        inv = self.return_matching_response(
                            inv, network, station, channel)
                        return True, self, inv
    # 3. if not found, try again and allow any location code
    if self.stats.network == '':
        temp_inv = inv.select(station=self.stats.station,
                             channel=self.stats.channel)
        found = False
        for network in temp_inv.networks:
            for station in network.stations:
                for channel in station.channels:
                    if self.response_stats_match(channel):
                        self.stats.network = network.code
                        self.stats.location = channel.location_code
                        inv = self.return_matching_response(
                            inv, network, station, channel)
                        return True, self, inv
    # 4 if not found, check if the channel code may contain a space in
    #   the middle
    if self.stats.channel[1] == ' ' or self.stats.channel[1] == '0':
        temp_inv = inv.select(network=self.stats.network,
                             station=self.stats.station,
                             location=self.stats.location,
                             channel=self.stats.channel[0] + '?' +
                             self.stats.channel[2])
        found = False
        for network in temp_inv.networks:
            for station in network.stations:
                for channel in station.channels:
                    if self.response_stats_match(channel):
                        self.stats.channel = channel.code
                        inv = self.return_matching_response(
                            inv, network, station, channel)
                        return True, self, inv
        # 5 if not found, allow space in channel and empty network
        temp_inv = inv.select(station=self.stats.station,
                             location=self.stats.location,
                             channel=self.stats.channel[0] + '?' +
                             self.stats.channel[2])
        found = False
        for network in temp_inv.networks:
            for station in network.stations:
                for channel in station.channels:
                    if self.response_stats_match(channel):
                        self.stats.network = network.code
                        self.stats.channel = channel.code
                        inv = self.return_matching_response(
                            inv, network, station, channel)
                        return True, self, inv
        # 6 if not found, allow space in channel and empty location
        temp_inv = inv.select(network=self.stats.network,
                             station=self.stats.station,
                             channel=self.stats.channel[0] + '?' +
                             self.stats.channel[2])
        found = False
        for network in temp_inv.networks:
            for station in network.stations:
                for channel in station.channels:
                    if self.response_stats_match(channel):
                        self.stats.location = channel.location_code
                        self.stats.channel = channel.code
                        inv = self.return_matching_response(
                            inv, network, station, channel)
                        return True, self, inv
        # 7 if not found, allow space in channel, empty network, and
        #   empty location
        temp_inv = inv.select(station=self.stats.station,
                             channel=self.stats.channel[0] + '?' +
                             self.stats.channel[2])
        found = False
        for network in temp_inv.networks:
            for station in network.stations:
                for channel in station.channels:
                    if self.response_stats_match(channel):
                        self.stats.network = network.code
                        self.stats.location = channel.location_code
                        self.stats.channel = channel.code
                        inv = self.return_matching_response(
                            inv, network, station, channel)
                        return True, self, inv
    return found, self, Inventory()


def response_stats_match(self, channel, **kwargs):
    """
    Check whether some criteria (validity period, sampling rate) of the
    inventory-response and the trace match

    :param tr: trace for which to check the information in the channel-response
    :type tr: :class:`obspy.core.trace.Trace`
    :param channel: _description_
    :type channel: :class:`obspy.core.inventory.Channel`

    :return: Whether the channel response matches with the trace.
    :rtype: bool
    """
    sample_rate_diff = abs(channel.sample_rate - self.stats.sampling_rate)
    if (channel.start_date <= self.stats.starttime
            and (channel.end_date >= self.stats.endtime
                 or channel.end_date is None)
            and sample_rate_diff < 1):
        return True
    else:
        return False


def return_matching_response(self, inv, network, station, channel, **kwargs):
    """Return the first matching response for a trace.

    :param tr: trace for which to return matching response
    :type tr: :class:`obspy.core.trace.Trace`
    :param inv: inventory with all available responses
    :type inv: :class:`obspy.core.inventory.Inventory`
    :param network: selected network
    :type network: :class:`obspy.core.inventory.Network`
    :param station: selected station
    :type station: :class:`obspy.core.inventory.Station`
    :param channel: selected channel
    :type channel: :class:`obspy.core.inventory.Channel`

    :return: Inventory containing matching responses only
    :rtype: :class:`obspy.core.inventory.Inventory`
    """
    inv = inv.select(
        network=network.code, station=station.code, channel=channel.code,
        location=channel.location_code, starttime=self.stats.starttime,
        endtime=self.stats.endtime)
    if len(inv.networks) > 1 or len(inv.networks[0].stations) > 1\
            or len(inv.networks[0].stations[0].channels) > 1:
        Logger.debug('Found more than one matching response for trace, '
                     + 'returning all.')
    return inv.copy()



# TODO: maybe monkey patch these functions onto Trace ?
# def balance_noise(self, inv, balance_power_coefficient=2,
# def _try_remove_responses(tr, inv, taper_fraction=0.05, pre_filt=None,
# def load_events_for_detection.py:def _init_processing_per_channel(
# def load_events_for_detection.py:def _init_processing_per_channel_wRotation(

# TODO: why does this file fail?    IndexError for scale..
# from obspy import read
# from robustraqn.obspy.core import Stream, Trace
# st = read('/data/seismo-wav/NNSN_/1988/07/88070213.1349J88')
# st.agc(agc_window_sec=10)


Trace.automatic_gain_control = automatic_gain_control
Trace.agc = automatic_gain_control
Trace.try_find_matching_response = try_find_matching_response
Trace.response_stats_match = response_stats_match
Trace.try_remove_response = try_remove_response
Trace.return_matching_response = return_matching_response