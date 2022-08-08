
# %% 
import os

import pandas as pd
import wcmatch
from wcmatch import fnmatch, glob  # need wcmatch for extglob support
from string import punctuation
from copy import deepcopy
import itertools
import numpy as np
from pyfftw.interfaces import scipy_fftpack
from pyfftw.interfaces import scipy_fft  # these fft-functions return complex

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from cycler import cycler

# from scipy import stats
import statistics as stats
from scipy.interpolate import interp1d

import pickle

import obspy
from obspy import read_inventory
from obspy import read as obspyread
# from obspy.core.stream import Stream
# from obspy.core.event import Event
from obspy import UTCDateTime
# from obspy.io.mseed import InternalMSEEDError
# from obspy.imaging.cm import pqlx
from obspy.signal.spectral_estimation import get_nlnm

# from robustraqn.quality_metrics import ()
import logging
Logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(name)40s:%(lineno)s\t%(funcName)20s()\t%(levelname)s\t%(message)s")
from robustraqn.seismic_array_tools import SEISARRAY_PREFIXES


def balance_noise(self, inv, balance_power_coefficient=2,
                  water_level_above_5s_in_db=-150, ground_motion_input=[],
                  sta_translation_file=None):
    """
    Normalize the frequency content of the seismogram recorded at a station
    by the station's noise profile.
    """
    if len(self.data) == 0:
        Logger.warning('Cannot balance trace by noise PDF, there is no data '
                       'for %s', str(self))
        return self
    else:
        Logger.debug('Balancing noise with power coefficient: %s',
                     str(balance_power_coefficient))

    tr_range = self.data.max() - self.data.min()

    # TODO: once Obspy has FrequencyTrace classes implemented, use these
    #       instead of manually converting to frequency domain
    # from eqcorrscan.utils.libnames import _load_cdll
    # utilslib = _load_cdll('libutils')
    # utilslib.normxcorr_fftw
    #
    # fdat = np.fft.rfft(self.data, n=self.stats.npts) # 2.91 s
    # f = np.fft.rfftfreq(self.stats.npts) * self.stats.sampling_rate
    # fdat = spectrum(self.data, win=1, nfft=tr.stats.npts) # 1.95 s
    # fdat = pyfftw.interfaces.scipy_fftpack.rfft(self.data, n=self.stats.npts)
    # 1.24 s
    # fdat = scipy.fftpack.rfft(tr.data, n=tr.stats.npts) # 1.77 s

    orig_npts = self.stats.npts
    fast_len = scipy_fft.next_fast_len(orig_npts)

    fdat = scipy_fft.rfft(self.data, n=fast_len)
    f = scipy_fft.rfftfreq(fast_len) * self.stats.sampling_rate

    try:
        sta_inv = inv.select(station=self.stats.station)
        if len(sta_inv) == 0 and sta_translation_file is not None:
            sta_fortransl_dict, sta_backtrans_dict = (
                load_station_translation_dict(file=sta_translation_file))
            if self.stats.station in sta_backtrans_dict:
                sta_inv = inv.select(station=sta_backtrans_dict.get(
                    self.stats.station))
        if len(sta_inv) > 0:
            sta_inv = sta_inv.networks[0].station[0]

        noise_model = sta_inv.noise_model.copy()
        for j, freq in enumerate(noise_model.frequencies):
            if 1/freq > 5:
                if noise_model.decibels[j] < water_level_above_5s_in_db:
                    noise_model.decibels[j] = water_level_above_5s_in_db
        # noise_model = Noise_model(noise_model.frequencies,
        #                           noise_model.decibels)

        f_filter = noise_model.frequencies
        amp_filter = noise_model.amplitude
    except Exception as e:
        # Logger.warning('Cannot balance trace by noise PDF, there is no noise'
        #                ' model available for %s', str(self))
        Logger.exception(e)
        Logger.warning(
            'Cannot balance trace by noise PDF, there is no noise model '
            'available for %s %s. Using dummy bandpass filter 3-50 Hz',
            self.id, str(self.stats.starttime))
        f_filter = np.array([1, 3, 50, 100])
        amp_filter = np.array([10e5, 1, 1, 10e5])
        # return self

    # Transform noise spectra to the physical units of the trace (this assumes
    # that spectra are for ground velocity)
    try:
        process_info = [pi for pi in self.stats.processing
                        if "remove_response" in pi][-1]
        ground_m_type = process_info.partition("::output='"
                                               )[2].partition("'::")[0]
    except Exception as e:
        ground_m_type = []

    if   "VEL"  in ground_m_type or "VEL"  in ground_motion_input:
        pass  # don't need to correct then
    elif "ACC"  in ground_m_type or "ACC"  in ground_motion_input:
        # * np.sqrt(-1)
        amp_filter = amp_filter * (2 * np.pi * f_filter)
    elif "DISP" in ground_m_type or "DISP" in ground_motion_input:
        # * np.sqrt(-1)
        amp_filter = amp_filter / (2 * np.pi * f_filter)
    else:
        Logger.warning("Cannot resolve what is the input ground motion type "
                       + "for %s, assuming velocity.", str(self))

    # find maximum possibly useful frequency
    f_max_append = np.maximum(2 * f_filter[-1], self.stats.sampling_rate)
    # append values to define filter up to lower & upper frequency boundary
    f_filter = np.append(0, f_filter)
    amp_filter = np.append(amp_filter[0], amp_filter)
    f_filter = np.append(f_filter, f_max_append)
    amp_filter = np.append(amp_filter, amp_filter[-1])

    # normalize filter by its average so that resulting amplitude is approx
    # same level
    amp_filter = amp_filter / stats.median(amp_filter)

    # interpolate PSDPDF-based filter onto all frequencies
    func = interp1d(f_filter, amp_filter)
    amp_filter_interp = func(f)
    # over-emphasize low-noise frequencies
    amp_filter_interp = amp_filter_interp ** balance_power_coefficient
    # normalize filter by its average so that resulting amplitude is approx
    # same level (better normalize before interpolation so it's always the
    # same!)
    # amp_filter_interp = amp_filter_interp / amp_filter_interp.mean()

    # multiply data with PSDPDF-based filter in frequency domain
    amp_filtered = fdat / amp_filter_interp
    # convert back to time domain
    # dat = np.fft.irfft(amp_filtered, n=self.stats.npts)
    dat = scipy_fft.irfft(amp_filtered, n=fast_len)

    # Set trace data to filtered data and amplify to previous range
    self.data = dat
    # Remove the extra points that were added for fast fft
    # Convert (back) to float32, there's no point with memory-heavy float64
    self.data = np.float32(self.data[0:orig_npts])

    return self


# Would be nice to monkey-patch this method onto the Stream-class
# class Stream(obspy.core.stream.Stream):

def st_balance_noise(
        self, inv, balance_power_coefficient=2, ground_motion_input=[],
        water_level_above_5s_in_db=-150, sta_translation_file=None):
    """
    Normalize the frequency content of a stream by the mean / mode / median
    noise PDF at that station site.
    """
    for tr in self:
        # Add balancing-function as bound method to Trace
        # adding the function causes errors once running it in parallel
        # bound_method = tr_balance_noise.__get__(tr)
        # bound_method = types.MethodType(tr_balance_noise, tr)

        # if not hasattr(tr, "balance_noise"):
        # tr.balance_noise = bound_method
        Logger.debug('Trying to balance noise for ' + str(tr))
        # tr.balance_noise(inv)
        tr = balance_noise(
            tr, inv, balance_power_coefficient=balance_power_coefficient,
            ground_motion_input=ground_motion_input,
            water_level_above_5s_in_db=water_level_above_5s_in_db,
            sta_translation_file=sta_translation_file)

    return self


def sum_station_pdf(inv, pdf_dir, network, station, location="*",
                    channel="???"):
    """
    create a mega pdf for one network / channel / location / station
    """
    # Get list of existing PDFs for individual NSLC
    pdffiles = wcmatch.glob.glob(
        os.path.join(pdf_dir, network, station, network + "." + station + "."
                     + location + "." + channel + ".*_PDF.csv"),
        flags=wcmatch.glob.EXTGLOB)

    # Sum all individual station PDFs to produce composite pdf
    # or read results from a previous calculation
    freq_u, db_u = findPDFBounds(pdffiles)
    freq_u_str = findUniqFreq(pdffiles)

    station_pdf_folder = 'StationPDFs'
    if not os.path.exists(station_pdf_folder):
        os.makedirs(station_pdf_folder)

    outpdffile = os.path.join(
        station_pdf_folder, network + '.' + station + '.' + location + '.'
        + channel)

    pdf = calcMegaPDF(
        freq_u, freq_u_str, db_u, pdffiles, outpdffile=outpdffile)
    newpdf_norm = normalize_pdf(pdf, freq_u)

    # compute mean / mode / median (or some percentile)
    perc = 50
    frac = perc/100.
    # Calculate specified percentiles
    # sampling_rates = df.SampleRate.unique()
    # TODO: once all response are ingested in EIDA node, replace this list with
    #       a unique list of sampling rates taken from the fedcat-file
    # sampling_rates = [20, 40, 50, 66, 75, 80, 100, 200, 500]

    sample_rates_df = find_unique_sampling_rates(inv, network, station,
                                                 location, channel)
    fnyqs = 0.5 * sample_rates_df
    freq_perc, db_perc = find_percentile(freq_u, db_u, newpdf_norm, frac,
                                         ax=None, plotline=False, fnyqs=fnyqs)

    return pdf, freq_u, db_u, freq_perc, db_perc


def normalize_pdf(pdf, freq_u):
    """ Normalize PDF since MUSTANG returns hit counts not %
    """
    newpdf_norm = np.zeros(shape=pdf.shape, dtype=np.float_)
    for i in range(len(freq_u)):
        if np.sum(pdf[i, :]) > 0:
            newpdf_norm[i, :] = pdf[i, :]/np.sum(pdf[i, :])
        else:
            newpdf_norm[i, :] = pdf[i, :]*0

    return newpdf_norm


def find_unique_sampling_rates(inv, network, station, location, channel):
    """
    Return unique list of sampling rates for the stations/location/channel
    """
    sample_rates_list = list()
    # do station=* to allow extended globbing patterns for stations
    inv = inv.select(network=network, station='*', location=location,
                     channel=channel)
    for inv_network in inv.networks:
        for inv_station in inv_network.stations:
            if wcmatch.fnmatch.fnmatch(inv_station.code, station,
                                       flags=wcmatch.fnmatch.EXTMATCH):
                for inv_channel in inv_station.channels:
                    sample_rates_list.append(inv_channel.sample_rate)
    sample_rates = sorted(list(set(sample_rates_list)))
    # sample_rates_df = pd.DataFrame(sample_rates, columns=['sample_rate'])
    sample_rates_df = np.array(sample_rates)

    return sample_rates_df


def calcMegaPDF(freq_u, freq_u_str, db_u, pdffiles, outpdffile='megapdf.npy'):
    '''
    based on a previous version by Emily Wolin. Integrated from
    https://github.com/ewolin/HighFreqNoiseMustang_paper

    Add together all PSDPDFs in pdffiles!
    And save as .npy file for easier reading later
    '''
    # Set up dictionaries to convert freq and db to integers.
    # Use integers for freq to avoid floating point errors
    # and make binning faster.
    i_f = np.arange(len(freq_u_str))
    fd = dict(zip(freq_u_str, i_f))

    i_db = np.arange(len(db_u))
    dbd = dict(zip(db_u, i_db))
    pdf = np.zeros((len(i_f), len(i_db)), dtype=np.int_)

    # Sum all files to make mega-pdf
    Logger.info('Adding individual PDFs to composite, please wait...')
    logfile = open('pdffiles.txt', 'w')
    for infile in pdffiles:
        try:
            freq = np.loadtxt(infile, unpack=True, delimiter=',', usecols=0)
            db, hits = np.loadtxt(infile, unpack=True, delimiter=',',
                                  usecols=[1, 2], dtype=np.int_)
        except OSError:
            Logger.warning('NSLC time period {} not available, so I cannot '
                           'check NLNM.'.format(infile))
            continue

        # check microseism looks ok
        microseism_ok = isMicroseismOk(freq, db, hits, db_tol=20,
                                       max_hits_perc=20, f_min=0.2, f_max=0.4)
        # microseism_ok = True
        if microseism_ok:
            logfile.write('{0}\n'.format(infile.split('/')[-1]))
            Logger.info(infile.split('/')[-1])
            for i in range(len(hits)):
                f1 = freq[i]
                db1 = db[i]
                hit1 = hits[i]

                i_f1 = fd[str(f1)]
                i_db1 = dbd[db1]
                pdf[i_f1, i_db1] += hit1
        else:
            Logger.warning('rejected based on microseismic band: %s', infile)
    logfile.close()

    # Save PDF to a numpy file so we can read+plot it easily later
    np.save(outpdffile, [pdf, freq_u, db_u])

    # outpdftext = open('megapdf.txt', 'w')
    # outpdftext.write('#freq db hits\n')
    # for i_f in range(len(freq_u)):
    #    for i_db in range(len(db_u)):
    #         outpdftext.write('{0} {1} {2}\n'.format(freq_u[i_f], db_u[i_db],
    #                          pdf[i_f,i_db]))
    # outpdftext.close()
    Logger.info('Finished calculating composite PDF.')
    Logger.info('See pdffiles.txt for list of individual PDFs summed.')

    return pdf


def isMicroseismOk(freq, db, hits, db_tol=5, max_hits_perc=20, f_min=0.2,
                   f_max=0.4):
    '''
    copyright Emily Wolin. Integrated from
    https://github.com/ewolin/HighFreqNoiseMustang_paper

    Check that a PDF does not fall too far below the Peterson NLNM
    '''
    # find total number of PSDs in PDF
    mode_freq = stats.mode(freq)
    ih = np.where(freq == mode_freq)
    n_psds = hits[ih].sum()

    T_nlnm, db_nlnm = get_nlnm()
    f_nlnm = 1./T_nlnm
    nlnm_interp = interp1d(np.log10(f_nlnm), db_nlnm)

    icheck = np.where((freq <= f_max) & (freq >= f_min))
    f_check = freq[icheck]
    # print(np.unique(f_check))
    # print(f_check)
    db_nlnm_check = nlnm_interp(np.log10(f_check))
    db_check = db[icheck]
    hits_check = hits[icheck]
    dbdiff = db_nlnm_check - db_check
    # define max_hits from allowed percentage and total number of PSDs
    max_hits = max_hits_perc / 100 * n_psds
    ibelow, = np.where(dbdiff > db_tol)
    hits_below = hits_check[ibelow]
    f_check_below = f_check[ibelow]

    ibelow = np.intersect1d(np.where(dbdiff > db_tol),
                            np.where(hits_check > max_hits))
    # hits_below = np.where(ibelow > max_hits)
    # check if more than a selected percentage of PSDs for any frequency in
    # microseism band exceeds allowed number of exceeding PSDs:
    isok = True
    for f in set(f_check_below):
        fi = np.where(f_check_below == f)
        if hits_below[fi].sum() > max_hits:
            isok = False

    # if len(ibelow) == 0:
    #     isok = True
    # else:
    #     isok = False
    return isok


def findPDFBounds(pdffiles):
    '''
    copyright Emily Wolin. Integrated from
    https://github.com/ewolin/HighFreqNoiseMustang_paper

    Get lists of unique frequencies and dBs
    from all individual PDF files
    '''
    Logger.info('finding list of unique freq, dB')

    j = -1
    first_pdffile_found = False
    while not first_pdffile_found:
        j += 1
        Logger.info(pdffiles[j])
        try:
            df_single = pd.read_csv(pdffiles[j], skiprows=5,
                                    names=['freq', 'db', 'hits'])
            first_pdffile_found = True
        except FileNotFoundError as e:
            Logger.error(e)
            Logger.info(
                'Looking for next available NSLC time period with PDF file.')
    # df_single = pd.read_csv(pdffiles[0], skiprows=5,
    #                        names=['freq', 'db', 'hits'])
    freq_u = df_single.freq.unique()
    db_u = df_single.db.unique()
    for i in range(j, len(pdffiles)):
        try:
            df_single = pd.read_csv(pdffiles[i], skiprows=5,
                                    names=['freq', 'db', 'hits'])
            freq_u = np.unique(np.append(freq_u, df_single.freq.unique()))
            db_u = np.unique(np.append(db_u, df_single.db.unique()))
        except FileNotFoundError as e:
            Logger.warning(e)
            Logger.info('Continuing with next NSLC time period with PDF file.')
            continue
    db_u.sort()
    freq_u.sort()
    # np.save('freq_u.npy', freq_u)
    # np.save('db_u.npy', db_u)
    return freq_u, db_u


def findUniqFreq(pdffiles):
    '''
    copyright Emily Wolin. Integrated from
    https://github.com/ewolin/HighFreqNoiseMustang_paper

    Find unique frequency values as *strings*
    for quick lookup in calcMegaPDF.
    '''
    j = -1
    first_pdffile_found = False
    while not first_pdffile_found:
        j += 1
        Logger.info(pdffiles[j])
        try:
            df_single = pd.read_csv(pdffiles[j], skiprows=5,
                                    names=['freq', 'db', 'hits'],
                                    dtype={'freq': 'str'})
            freq_u = df_single.freq.unique()
            first_pdffile_found = True
        except FileNotFoundError as e:
            Logger.error(e)
            Logger.info(
                'Looking for next available NSLC time period with PDF file.')

    # df_single = pd.read_csv(pdffiles[0], skiprows=5,
    #                         names=['freq', 'db', 'hits'],
    #                         dtype={'freq':'str'})
    for i in range(j, len(pdffiles)):
        try:
            df_single = pd.read_csv(pdffiles[i], skiprows=5,
                                    names=['freq', 'db', 'hits'],
                                    dtype={'freq': 'str'})
            freq_u = np.unique(np.append(freq_u, df_single.freq.unique()))
        except FileNotFoundError as e:
            Logger.warning(e)
            Logger.info(
                'Looking for next available NSLC time period with PDF file.')
    outfile = open('freq_u_str.txt', 'w')
    for i in range(len(freq_u)):
        outfile.write('{0}\n'.format(freq_u[i]))
    outfile.close()
    isort = np.argsort(freq_u.astype('float'))
    freq_u = freq_u[isort]
    # np.save('freq_u_str.npy', freq_u)
    return freq_u


def find_percentile(freq_u, db_u, newpdf_norm, perc, ax, fnyqs=[],
                    plotline=True):
    '''
    copyright Emily Wolin. Integrated from
    https://github.com/ewolin/HighFreqNoiseMustang_paper

    Given a (normalized) PDF, find the dB levels of a given percentile.
    Ignore frequencies between 0.75*fnyq and fnyq to cut out spikes.
    '''
    # surely there must be something in numpy or scipy that does this
    # but I haven't hunted it down yet.
    nfreq = len(freq_u)
    db_perc = -999 * np.ones(nfreq)
    for i in range(nfreq):
        fslice = newpdf_norm[i, :]
        dum = 0
        for j in range(len(fslice)):
            dum += fslice[j]
            if(dum >= perc):
                db_perc[i] = db_u[j]
                break
    # plot and/or write percentile line
    # ignoring spikes near Nyquist frequency(ies)
    i_use, = np.where(freq_u < 200)
    freq_u = freq_u[i_use]
    db_perc = db_perc[i_use]
    i_use, = np.where(freq_u < 100)
    for fnyq in fnyqs:
        i_nonyq, = np.where((freq_u > fnyq) | (freq_u < 0.75 * fnyq))
        i_use = np.intersect1d(i_use, i_nonyq)
    freq_perc = freq_u[i_use]
    db_perc = db_perc[i_use]
    if plotline:
        ax.plot(1./freq_perc, db_perc,
                label='{0:.1f}%'.format(100*perc), linewidth=1)
    # outname = 'Percentiles/percentile_{0:.1f}.txt'.format(100*perc)
    # outfile = open(outname,'w')
    # outfile.write('#freq dB\n')
    # for i in range(len(db_perc)):
    #    outfile.write('{0} {1}\n'.format(freq_perc[i], db_perc[i]))
    # outfile.close()
    # print('Wrote {0} percentile to {1}'.format(100*perc, outname))
    return freq_perc, db_perc


def plot_pdf(pdf, freq_u, db_u, station, out_folder, outfile_name, inv,
             plot_legend=True):
    """
    copyright Emily Wolin. Extracted from
    https://github.com/ewolin/HighFreqNoiseMustang_paper

    plot the MegaPdf
    """
    # cmap = 'gray_r'

    cmap = get_custom_ispaq_cmap()

    # vmax = 0.05 # original
    vmax = 0.15
    fig, ax = setupPSDPlot()

    # Plot normalized PDF!
    im = ax.pcolormesh(1./freq_u, db_u, pdf.T*100, cmap=cmap, vmax=vmax*100)
    fig.colorbar(im, cax=fig.axes[1], label='Probability (%)')

    # Calculate and plot specified percentiles
    sample_rates = find_unique_sampling_rates(inv, '*', station, "*", "*")
    #     inv, "*", station.replace('_','*'), "*", "*") # OLD fh
    fnyqs = 0.5 * sample_rates

    percentiles = [10, 25, 50, 90]
    for perc in percentiles:
        frac = perc/100.
        freq_perc, db_perc = find_percentile(freq_u, db_u, pdf, frac,
                                             ax, plotline=True, fnyqs=fnyqs)

    smallest_period = 0.01
    largest_period = 200
    # Add frequency axis to plot
    ax_freq = ax.twiny()
    ax_freq.set_xlabel('Frequency (Hz)')
    ax_freq.semilogx()
    ax_freq.set_xlim(1 / smallest_period, 1 / largest_period)
    ax_freq.xaxis.set_label_position('top')
    ax_freq.tick_params(axis='x', top=True, labeltop=True)

    # Final setup: Draw legend and set period axis limits
    if plot_legend:
        ax.legend(ncol=2, loc='lower center', fontsize='6', handlelength=3)
    ax.set_xlim(smallest_period, largest_period)
    ax.set_title(station)
    plt.tight_layout()

    # Save images of plots
    # for imgtype in ['png', 'eps']:
    for imgtype in ['png']:
        fig.savefig(os.path.join(out_folder, outfile_name + '_pdf.'+imgtype),
                    dpi=600)
        Logger.info('saved plot as ' + outfile_name + '.{0}'.format(imgtype))
    plt.close()


def get_custom_ispaq_cmap():
    """
    Extracted from ispaq.ispaq.PDF_aggregator.plot_PDF
    """
    # Set up plotting -- color map
    cmap = plt.get_cmap('gist_rainbow_r', 3000)
    # don't want whole spectrum
    cmaplist = [cmap(i) for i in range(cmap.N)][100::]

    # convert the first nchange to fade from white
    nchange = 100
    for i in range(nchange):
        first = cmaplist[nchange][0]
        second = cmaplist[nchange][1]
        third = cmaplist[nchange][2]
        scaleFactor = (nchange-1-i)/float(nchange)
        df = ((1-first)  * scaleFactor) + first
        ds = ((1-second) * scaleFactor) + second
        dt = ((1-third)  * scaleFactor) + third
        cmaplist[i] = (df, ds, dt, 1)

    cmaplist[0] = (1, 1, 1, 1)
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    return cmap


def setupPSDPlot():
    '''
    copyright Emily Wolin. Extracted from
    https://github.com/ewolin/HighFreqNoiseMustang_paper
    Set up a plot with Peterson noise model for plotting PSD curves.
       x axis = period (s)
       y axis = decibels '''
    # Set paths to various noise models

    if '__file__' in locals():
        file = __file__
    else:
        file = '.'
    codedir = os.path.dirname(file)
    piecewise = os.path.join(codedir, 'PiecewiseModels')
    nhnm = np.loadtxt(piecewise+'/peterson_HNM.mod', unpack=True)
    nlnm = np.loadtxt(piecewise+'/peterson_LNM.mod', unpack=True)
    nhnb = np.loadtxt(piecewise+'/High_T-vs-dB.txt', unpack=True)
    nlportb = np.loadtxt(piecewise+'/Low_Port_T-vs-dB.txt', unpack=True)
    nlpermb = np.loadtxt(piecewise+'/Low_Perm_T-vs-dB.txt', unpack=True)
    # nlportb = np.loadtxt(piecewise+'/stitch.txt', unpack=True)

    # Set up axes
    width = 6
    fig = plt.figure(figsize=(width, width/1.4))
    gs_plots = gridspec.GridSpec(1, 2, width_ratios=[1, 0.05])
    ax = fig.add_subplot(gs_plots[0, 0])
    ax_cb = fig.add_subplot(gs_plots[0, 1])

    # colorlist = ['gold', '#ff7f0e', '#d62728']
    # colorlist = ['green', 'gold', '#ff7f0e', '#d62728']
    colorlist = ['gold', '#ff7f0e', '#d62728', 'green']
    ax.set_prop_cycle(cycler('color', colorlist))

    ax.semilogx()
    ax.set_xlim(0.05, 200)
    ax.set_xlabel('Period (s)')
    ax.set_ylabel(r'Power (dB[m$^2$/s$^4$/Hz])')
    ax.set_ylim(-200, -60)

    # Plot Peterson noise models
    ax.plot(nhnm[0], nhnm[1], linewidth=2, color='black', label='NHNM/NLNM')
    ax.plot(nlnm[0], nlnm[1], linewidth=2, color='black')

    # icheck, = np.where((nlnm[0] >= 1./0.4)&(nlnm[0]<=1./0.2))
    f_min = 0.2
    f_max = 0.4
    icheck, = np.where((nlnm[0] >= 1./f_max)&(nlnm[0]<=1./f_min))
    ax.plot(nlnm[0][icheck], nlnm[1][icheck]-5, linewidth=1, color='grey',
            linestyle='--')
    # print(nlnm[0][icheck])

    # Plot high-frequency extensions
    # ax.plot(nlpermb[0], nlpermb[1], color='grey', ls='-.', lw=3, 
    #         label='Low Permanent Baseline')
    ax.plot(nlportb[0], nlportb[1], linewidth=3, ls='--', color='grey',
            label='Low Portable Baseline')
    ax.plot(nhnb[0], nhnb[1], color='grey', ls=(0, (1, 1)), lw=3,
            label='High Baseline')
    # Plot Brune corner frequency grid
    # plotBrune(ax)
    # dummy point w/no labels in case we don't plot HF noise models
    # ax.plot(np.zeros(1), np.zeros([1]), color='w', alpha=0, label=' ')

    return fig, ax


class Noise_model:
    """
    Noise_model class, to be attached to an
        obspy.core.inventory.station.Station

    :type value: float
    :param value: Latitude value
    :type measurement_method: str
    :param measurement_method: Method used in the measurement.
    """
    _minimum = -90
    _maximum = 90
    _unit = "DEGREES"

    def __init__(self, frequencies, decibels):
        """
        :type frequencies: ndarray
        :param frequencies: frequency sampling values
        :type decibels: ndarray
        :param decibels:
            decibel values for the spectral power of the displacement values,
            e.g. in units of 10log(m^2/s^4/Hz)[dB]; or in relative units.
        """
        self.frequencies = frequencies
        self.decibels = decibels
        # self.decibles = 10 * log10(power)
        self.power = 10 ** (decibels / 10)
        self.amplitude = 10 ** (decibels / (2 * 10))
        # super(Noise_model, self).__init__(frequencies, decibels)

    def copy(self):
        """
        Provide deep copy of object
        """
        return deepcopy(self)

    @property
    def power(self):
        return 10 ** (self.decibels / 10)

    @property
    def amplitude(self):
        return 10 ** (self.decibels / (2 * 10))

    @amplitude.setter
    def amplitude(self, amp):
        self.decibels = 20 * np.log10(amp)

    @power.setter
    def power(self, pow):
        self.decibels = 10 * np.log10(pow)


class Station(obspy.core.inventory.station.Station):
    """
    Adds noise_model as attribute to Station
    """

    @property
    def noise_model(self):
        return self._noise_model

    @noise_model.setter
    def noise_model(self, value):
        if isinstance(value, Noise_model):
            self._noise_model = value
        else:
            self._noise_model = Noise_model(value)

    # @property
    # def ppsd(self):
    #     return self._ppsd

    # @ppsd.setter
    # def ppsd(self, value):
    #     if isinstance(value, PPSD):
    #         self._ppsd = value
    #     else:
    #         self._ppsd = PPSD(value)


def attach_single_noise_model(inv, pdf_dir, network="*", station="*",
                              location="*", channel="[ESBHCDFNML]??",
                              plot_station_pdf=False):
    """
    """
    pdf, freq_u, db_u, freq_perc, db_perc = sum_station_pdf(
        inv, pdf_dir, network, station, location=location, channel=channel)

    if plot_station_pdf:
        pdf_norm = normalize_pdf(pdf, freq_u)
        file_name = station.strip(punctuation)
        plot_pdf(pdf_norm, freq_u, db_u, station, 'StationPDFs',
                 file_name, inv, plot_legend=False)

    noise_model = Noise_model(frequencies=freq_perc, decibels=db_perc)

    for net in inv.networks:
        for sta in net.stations:
            if (wcmatch.fnmatch.fnmatch(net.code, network) and
                wcmatch.fnmatch.fnmatch(
                    sta.code, station, flags=wcmatch.fnmatch.EXTMATCH)):
                if not hasattr(sta, "noise_model"):
                    setattr(sta, "noise_model", noise_model)

    return inv


# %% CUSTOM FUNCTIONS FOR MY OWN USE # FH 2020-11-03

def attach_noise_models(inv, pdf_dir, outfile='inv.pickle',
                        plot_station_pdf=False):
    """
    Attach a noise model to each station in an inventory for which the pdf_dir
    contains ISPAQ Mustang-style PSDPDF data files. Save the resulting
    inventory to outfile for reloading at a later point
    """

    all_station_pdf_dirs = glob.glob(os.path.join(pdf_dir, "??", "*"))
    station_list = list()
    for sta_pdf_dir in all_station_pdf_dirs:
        station_list.append(sta_pdf_dir.split("/")[-1])
    station_list = sorted(list(set(station_list)))

    # array_list is a list of tuples, with the first tuple element containing
    # the array-prefix and the 2nd tuple element containing a list of stations
    # at the array.
    seisarray_list = list()
    single_station_list = station_list.copy()
    seisarray_prefixes = SEISARRAY_PREFIXES
    # alternative pattern 'E*K!(O)*' (works only  in wcmatch, not glob)
    # ''E*K[!O]*'
    for seisarray_prefix in seisarray_prefixes:
        seisarray_station_list = list()
        for station in station_list.copy():
            if wcmatch.fnmatch.fnmatch(station, seisarray_prefix,
                                       flags=wcmatch.fnmatch.EXTMATCH):
                seisarray_station_list.append(station)
                single_station_list.remove(station)
        seisarray_list.append((seisarray_prefix, seisarray_station_list))

    single_station_list = sorted(list(set(single_station_list)))

    # For seismic arrays, build one megaPDF across all stations and attach the
    # same noise model to all stations.
    for j, seisarray in enumerate(seisarray_list):
        seisarray_prefix = seisarray[0]
        seisarray_stations = seisarray[1]
        try:
            inv = attach_single_noise_model(
                inv, pdf_dir, network="*", station=seisarray_prefix+'*',
                location="*", channel="[BHSE]H?",
                plot_station_pdf=plot_station_pdf)
        except Exception as e:
            Logger.warning('Cannot add noise model for array %s: %s',
                           seisarray_prefix, e)

    # For singular seismic stations, make a megaPDF for all channels together
    # and attach it to the station.
    for station in single_station_list:
        try:
            inv = attach_single_noise_model(
                inv, pdf_dir, network="*", station=station, location="*",
                channel="[ESBHCDFNML]??", plot_station_pdf=plot_station_pdf)
        except Exception as e:
            Logger.warning('Cannot add noise model for station %s: %s',
                           station, e)

    # inv.write('test_inv.xml', format="STATIONXML")
    pickle.dump(inv, open(outfile, "wb"), protocol=4)

    return inv


def plot_noise_profiles(inv, stations):
    """
    Plot the noise profiles from all station into one plot
    """
    fig, axes = plt.subplots(1, 1, sharey='row', figsize=(10, 6))
    for network in inv.networks:
        for station in network.stations:
            if station.code in stations:
                if hasattr(station, "noise_model"):
                    axes.plot(station.noise_model.frequencies,
                              station.noise_model.decibels)
    plt.xscale('log')
    fig.show()


def test_noise_balancing():
    """
    Function to demonstrate/test the balancing of a waveform by the station's
    noise profile
    """
    # pyfftw.interfaces.scipy_fftpack

    inv = pickle.load(open("inv.pickle", "rb"))

    # starttime = UTCDateTime(2020, 7, 22, 15, 6, 00)
    # endtime = UTCDateTime(2020, 7, 22, 15, 6, 30)
    # st = obspyread(
    #  '/data/seismo-wav/SLARCHIVE/2020/NS/ASK/HHZ.D/NS.ASK.00.HHZ.D.2020.204')

    starttime = UTCDateTime(2019, 9, 24, 0, 0, 0)
    endtime = UTCDateTime(2019, 9, 25, 0, 0, 0)
    st = obspyread(
        '/data/seismo-wav/SLARCHIVE/2019/NS/ASK/HHZ.D/NS.ASK.00.HHZ.D.2019.267')

    tr = st[0]
    tr = tr.detrend().taper(0.01, type='hann', max_length=None, side='both')

    tr.copy().filter(
        type='bandpass', freqmin=2, freqmax=10.0, zerophase=True
        ).trim(starttime=starttime, endtime=endtime).plot()

    # Add balancing-function as bound method to Trace
    # if not hasattr(tr, "balance_noise"):
    #     bound_method = tr_balance_noise.__get__(tr)
    #     tr.balance_noise = bound_method
    tr = balance_noise(tr, inv)

    # tr.data = dat * (tr_range / (dat.max() - dat.min()))
    tr.filter(type='bandpass', freqmin=0.5, freqmax=15.0)
    # tr.plot()
    tr.copy().trim(starttime=starttime, endtime=endtime).plot()


def plot_all_station_pdfs():
    """
    Plot the mega-PSDPDFs for all stations that are asaved as numpy-files in a
    folder, including percentiles.
    """
    inv = pickle.load(open("inv.pickle", "rb"))

    for file in glob.glob('StationPDFs/*.npy'):
        try:
            pdf, freq_u, db_u = np.load(open(file, "rb"), allow_pickle=True)
            pdf_norm = normalize_pdf(pdf, freq_u)
            station_name = file.split(".")[1]
            plot_pdf(pdf_norm, freq_u, db_u, station_name, 'StationPDFs',
                     station_name, inv, plot_legend=False)
        except OSError as err:
            Logger.warning(err)
            Logger.warning('Could not load file %s, continuing with next '
                           'station', file)
            continue


def get_updated_inventory_with_noise_models(
        inv_file, pdf_dir='~/repos/ispaq/WrapperScripts/PDFs/',
        outfile='inv.pickle', check_existing=True, plot_station_pdf=False):
    """
    """
    # Check whether an updated inventory exists already, and overwrite only if
    # option check_existing is set to False
    if os.path.exists(outfile):
        if (os.path.getmtime(outfile) >= os.path.getmtime(inv_file)
                and check_existing):
            Logger.info('Not updating noise models because an updated '
                        + 'inventory with noise models exists already.')
            inv_p_file = open(outfile, "rb")
            inv = pickle.load(inv_p_file)
            inv_p_file.close()
            return inv

    inv = read_inventory(os.path.expanduser(inv_file))
    pdf_dir = os.path.expanduser(pdf_dir)
    inv = attach_noise_models(inv, pdf_dir=pdf_dir, outfile=outfile,
                              plot_station_pdf=plot_station_pdf)
    return inv


# %%

# TODO - one could attach the whole PSDPDF as obspy.PPSD-object to the
# inventory, and add methods to read7write from/to ISPAQ Mustang style files.

# class PPSD(obspy.signal.PPSD):

#     def load_from_mustang_parquet(file):
#         """
#         load PPSD object from mustang PDF file saved as parquet file
#         """
#         read_ispaq_stats(
#             folder, networks=['??'], stations=['*'],
#             ispaq_prefixes='all', ispaq_suffixes='simpleMetrics',
#             file_type='csv', startyear=1970, endyear=2030)

#     def load_from_mustang_csv(file):
#         """
#         load PPSD object from mustang PDF file saved as csv file
#         """

#     def get_median(file):
#         """
#         median of PPSD"
#         """


# inv = pickle.load( open( "inv.pickle", "rb" ) )

# Can I add the station's noise model to the Station object itself?

    # def add_noise_model(file):
    #    self.

# ppsd.db_bins ===
# ppsd.db_bin_centers === db_u
# ppsd.db_bin_edges

# ppsd.period_bin_centers
# ppsd.period_bin_left_edges
# ppsd.period_bin_right_edges
# ppsd.period_xedges

# counts:
# ppsd.current_histogram === pdf

# PPSD.__init__(stats, metadata, skip_on_gaps=False, db_bins=(-200, -50, 1.0),
#               ppsd_length=3600.0, overlap=0.5, special_handling=None,
#               period_smoothing_width_octaves=1.0, period_step_octaves=0.125,
#               period_limits=None, **kwargs)

# multiply spectrum of waveform by PSDPDF

# %%
