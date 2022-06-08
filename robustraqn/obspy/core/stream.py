
import numpy as np
import pandas as pd

from obspy.core import Stream
from robustraqn.obspy.core.trace import Trace

import logging
Logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format=("%(asctime)s\t%(name)40s:%(lineno)s\t%(funcName)20s()\t%" +
            "(levelname)s\t%(message)s"))
"[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"


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

    # if agc_method == 'gismo':
    #     for tr in self:
    #         win_samp = int(tr.stats.sampling_rate * agc_window_sec)
    #         if win_samp >= tr.stats.npts:
    #             Logger.warning(
    #                 'AGC window (%s samples) is longer than trace %s, hence '
    #                 'amplitudes remain unchanged.', str(win_samp), tr)
    #             continue
    #         if method_exec == 'old':
    #             scale = np.zeros(tr.count() - 2 * win_samp)
    #             for i in range(-1 * win_samp, win_samp + 1):
    #                 scale = scale + np.abs(
    #                     tr.data[win_samp + i : win_samp + i + scale.size])
    #         elif method_exec == 'new':
    #             # Quicker version of above code (?)
    #             # slower:
    #             # scales = np.zeros((tr.count() - 2 * win_samp, 2 * win_samp + 3))
    #             # for ns, i in enumerate(range(-1 * win_samp, win_samp + 1)):
    #             #     scales[:, ns] = np.abs(tr.data[
    #             #         win_samp + i:win_samp + i + scale.size])
    #             # scale = np.sum(scales, 1)
    #             n_width = 2 * win_samp + 1
    #             # faster!
    #             # scale = scipy.convolve(
    #             #     tr.data, np.ones(n_width), mode='valid') / n_width
    #             # even faster!
    #             # scale = np.convolve(np.abs(tr.data), np.ones(n_width),
    #             #                     'valid') / n_width
    #             # fastest!
    #             scale = np.array(pd.DataFrame(np.abs(tr.data)).rolling(
    #                 n_width, min_periods=n_width, center=True).mean())
    #             scale = np.transpose(scale[~np.isnan(scale)])
    #         scale = scale / scale.mean()  # Using max() here may better
    #                                       # preserve inter-trace amplitudes
    #         # Fill out the ends of scale with its first/last values
    #         scale = np.hstack((np.ones(win_samp) * scale[0],
    #                            scale,
    #                            np.ones(win_samp) * scale[-1]))
    #         tr.data = tr.data / scale  # "Scale" the data, sample-by-sample
    #         tr.stats.processing.append('AGC applied via '
    #                                    f'_agc(agc_window_sec={agc_window_sec}, '
    #                                    f'method=\'{agc_method}\')')

    # elif agc_method == 'walker':
    #     for tr in self:
    #         half_win_samp = int(tr.stats.sampling_rate * agc_window_sec / 2)
    #         if half_win_samp >= tr.stats.npts:
    #             Logger.warning(
    #                 'AGC window (%s samples) is longer than trace %s, hence '
    #                 'amplitudes remain unchanged.', str(half_win_samp), tr)
    #             continue
    #         scale = []
    #         if method_exec == 'old':
    #             for i in range(half_win_samp, tr.count() - half_win_samp):
    #                 # The window is centered on index i
    #                 scale_max = np.abs(tr.data[i - half_win_samp:
    #                                         i + half_win_samp]).max()
    #                 scale.append(scale_max)
    #         elif method_exec == 'new':
    #             n_width = 2 * half_win_samp + 1
    #             scale = np.array(pd.DataFrame(np.abs(tr.data)).rolling(
    #                 n_width, min_periods=n_width, center=True).max())
    #             scale = np.transpose(scale[~np.isnan(scale)])

    #         # Fill out the ends of scale with its first/last values
    #         scale = np.hstack((np.ones(half_win_samp) * scale[0],
    #                            scale,
    #                            np.ones(half_win_samp) * scale[-1]))
    #         tr.data = tr.data / scale  # "Scale" the data, sample-by-sample
    #         tr.stats.processing.append('AGC applied via '
    #                                    f'_agc(agc_window_sec={agc_window_sec}, '
    #                                    f'agc_method=\'{agc_method}\')')
    # else:
    #     raise ValueError(f'AGC method \'{agc_method}\' not recognized. Method '
    #                      'must be either \'gismo\' or \'walker\'.')
    return self


Stream.automatic_gain_control = automatic_gain_control
Stream.agc = automatic_gain_control