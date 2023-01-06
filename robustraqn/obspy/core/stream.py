
import numpy as np
import pandas as pd

from obspy.core import Stream
from robustraqn.obspy.core.trace import Trace

import logging
Logger = logging.getLogger(__name__)
#logging.basicConfig(
#    level=logging.INFO,
#    format=("%(asctime)s\t%(name)40s:%(lineno)s\t%(funcName)20s()\t%" +
#            "(levelname)s\t%(message)s"))


# class Stream(object):
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
    for tr in self:
        tr.agc(agc_window_sec=agc_window_sec, agc_method=agc_method,
                method_exec=method_exec)
    return self


# TODO: maybe monkey patch these functions onto Stream ?
# def extract_array_stream(st, seisarray_prefixes=SEISARRAY_PREFIXES):
# def normalize_NSLC_codes(st, inv, std_network_code="NS",
# def try_remove_responses(stream, inventory, taper_fraction=0.05, pre_filt=None,
# def check_normalize_sampling_rate(
# load_events_for_detection.py:def init_processing(day_st, starttime, endtime, remove_response=False,
# load_events_for_detection.py:def init_processing_wRotation(
# load_events_for_detection.py:def prepare_detection_stream(
# load_events_for_detection.py:def parallel_detrend(st, parallel=True, cores=None, type='simple'):
# load_events_for_detection.py:def parallel_merge(st, method=0, fill_value=None, interpolation_samples=0,
# load_events_for_detection.py:def robust_rotate(stream, inventory, method="->ZNE"):
# load_events_for_detection.py:def parallel_rotate(st, inv, cores=None, method="->ZNE"):


Stream.automatic_gain_control = automatic_gain_control
Stream.agc = automatic_gain_control
