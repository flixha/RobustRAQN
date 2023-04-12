

from eqcorrscan.core.match_filter.party import Party

def reevaluate_detections(
        self, short_tribe, stream, threshold_type='MAD', threshold=9,
        re_eval_thresh_factor=0.6, trig_int=40.0, overlap='calculate',
        plot=False, multiplot=False, plotdir='DetectionPlots',
        daylong=False, fill_gaps=False, ignore_bad_data=False,
        ignore_length=True, pre_processed=False,
        parallel_process=False, cores=None, xcorr_func='fftw',
        concurrency=None, arch='precise', group_size=1, full_peaks=False,
        save_progress=False, process_cores=None, spike_test=False, min_chans=4,
        time_difference_threshold=3, detect_value_allowed_reduction=2.5,
        return_party_with_short_templates=False, min_n_station_sites=4,
        use_weights=False, copy_data=True, **kwargs):
    """
    This function takes a set of detections and reruns the match-filter
    detection with a set of templates that are shortened to XX length. Only if
    the detections are also significant (e.g., better detect_value) with the
    shorter templates, then they are retained. Other detections that do not
    pass this test are considered misdetections, which can often happen when
    seismic arrays are involved in detection and there is a seismic event near
    one of the arrays.

    :type self: :class:`eqcorrscan.core.match_filter.party.Party`
    :param self: Party containing the detections to reevaluate.
    :param short_tribe:
        Tribe (shortened compared to detection tribe), containing the templates
        to use for reevaluation of detections.
    :type stream: :class:`obspy.core.stream.Stream`
    :param stream: Stream containing the background data.
    :type threshold_type: str
    :param threshold_type: Threshold type to use for detection
    :type threshold: float
    :param threshold: Threshold to use for detection
    :type re_eval_thresh_factor: float
    :param re_eval_thresh_factor:
        Factor to multiply the original threshold by for match_filter detection
        with short template.
    :type trig_int: float
    :param trig_int: Trigger interval in seconds to use for detection
    :type overlap: float or str
    :param overlap:
    :type plot: bool
    :param plot: Whether to plot the detections
    :type multiplot: bool
    :param multiplot:
        Whether to plot the detections in a nice multi-channel plot.
    :type plotdir: str
    :param plotdir: Directory to save the plots to.
    :type daylong: bool
    :param daylong: Whether the data are daylong or not.
    :type fill_gaps: bool
    :param fill_gaps: Whether to fill gaps in the data or not.
    :type ignore_bad_data: bool
    :param ignore_bad_data: Whether to ignore bad data in EQcorrscan or not.
    :type ignore_length: bool
    :param ignore_length: Whether to ignore trace length or not.
    :type pre_processed: bool
    :param pre_processed: Whether the data are pre-processed or not.
    :type parallel_process: bool
    :param parallel_process:
    :type cores: int
    :param cores:
    :type xcorr_func: str
    :param xcorr_func: Cross-correlation function to use.
    :type concurrency: str
    :param concurrency:
        Concurrency to use for multiprocessing, can be one of 'concurrent',
        'multiprocess', 'multithread'. For more details see
        :func:`eqcorrscan.utils.correlate.get_stream_xcorr`.
    :type arch: str
    :param arch: Architecture of fmf / fmf2 to use, can be 'GPU' or 'CPU'.
    :type group_size: int
    :param group_size: Size of template group to process at once.
    :type full_peaks: bool
    :param full_peaks: Whether to use full peaks or not.
    :type save_progress: bool
    :param save_progress: Whether to save progress or not.
    :type process_cores: int
    :param process_cores: Number of cores to use for processing.
    :type spike_test: bool
    :param spike_test: Whether to use spike test or not.
    :type min_chans: int
    :param min_chans:
        Minimum number of channels to accept a detection as significant.
    :type time_difference_threshold: float
    :param time_difference_threshold:
        Time difference threshold in seconds between detection from long and
        short templates.
    :type detect_value_allowed_reduction: float
    :param detect_value_allowed_reduction:
        Allowed reduction in detect_value between detections from long and
        short templates.
    :type return_party_with_short_templates: bool
    :param return_party_with_short_templates:
        Whether to return the party with short templates or with long templates
        attached to the detections.
    :type min_n_station_sites: int
    :param min_n_station_sites:
        Minimum number of station sites to accept a detection. This is to avoid
        spurious detections that are only due to one array (i.e., one site.)
    :type use_weights: bool
    :param use_weights: Whether to use weights or not.
    :type copy_data: bool
    :param copy_data: Whether to copy the data at the start of EQcorrscan.
    :type kwargs: dict
    :param kwargs: Additional keyword arguments to pass to match_filter.

    :return: Party with detections that have been reevaluated.
    :rtype: :class:`eqcorrscan.core.match_filter.party.Party`
    """
    # Maybe do some checks to see if tribe and short_tribe have somewhat of the
    # same templates?

    # Check there's enough individual station sites for detection - otherwise
    # don't bother with the detection. This should avoid spurious picks that
    # are only due to one array.
    # TODO: this check should be executed before declustering in EQcorrscan
    # TODO: function should always return both the party for the long templates
    #       that have a short-template detection, and the party for the short
    #       templates.
    Logger.info('Start reevaluation of detections.')
    n_families_in = len(self.families)
    n_detections_in = len(self)
    # Get list of unique station names in party for station-site dict lookup
    unique_stations = list(set(
        [chan[0] for fam in self for det in fam for chan in det.chans]))
    station_sites_dict = seismic_array_tools.get_station_sites_dict(
        unique_stations)

    if min_n_station_sites > 1:
        checked_party = Party()
        for family in self:
            # family.copy()
            # checked_family.detections = []
            checked_family = Family(template=family.template, detections=[],
                                    catalog=None)
            for detection in family:
                # unique_stations = list(set([
                #     p.waveform_id.station_code
                #     for p in detection.event.picks]))
                # TODO: is there a way to speed up the checks on number of 
                #       station sites?
                unique_det_stations = list(set([chan[0]
                                            for chan in detection.chans]))
                # n_station_sites = len(list(set(
                #     get_station_sites(unique_stations))))
                # Get the number of station sites
                n_station_sites = len(list(set(
                    station_sites_dict[uniq_station]
                    for uniq_station in unique_det_stations)))
                if n_station_sites >= min_n_station_sites:
                    checked_family.detections.append(detection)
            if len(family.detections) > 0:
                # checked_party += checked_family
                checked_party.families.append(checked_family)
        Logger.info(
            'Checked party, %s detections fulfill minimum sites criterion.',
            len([det for fam in checked_party for det in fam]))
    else:
        checked_party = self
    long_party = checked_party
    n_detections_ok = len(long_party)

    # Need to scale factor slightly for fftw vs time-domain
    # (based on empirical observation)
    if xcorr_func == 'fftw':
        re_eval_thresh_factor = re_eval_thresh_factor * 1.1
    threshold = threshold * re_eval_thresh_factor

    # Select only the relevant templates
    det_templ_names = list(dict.fromkeys(
        [d.template_name for f in long_party for d in f]))
    short_tribe = Tribe(
        [short_tribe.select(templ_name) for templ_name in det_templ_names])
    # Find the relevant parts of the stream so as not to rerun the whole day:
    det_times = [d.detect_time for f in long_party for d in f]
    Logger.info(
        'Re-evaluating party to sort out misdetections, checking %s'
        + ' detections.', len(det_times))
    if len(det_times) == 0:
        return long_party, long_party
    earliest_det_time = min(det_times)
    latest_det_time = max(det_times)
    # if detections on significan part of the day:
    det_st = stream
    # if (latest_det_time - earliest_det_time) > 86400 * 0.5:
    #     Logger.info('Using full day for detection re-evaluation.')
    #     # then use whole day
    #     det_st = stream
    # else:
    #    #cut around half an hour before earliest and half an hour after latest
    #     # detection
    #     tr_start_times = [tr.stats.starttime for tr in stream.traces]
    #     tr_end_times = [tr.stats.endtime for tr in stream.traces]
    #     earliest_st_time = min(tr_start_times)
    #     latest_st_time = max(tr_end_times)
    #     starttime = earliest_det_time - (10 * 60)
    #     if starttime < earliest_st_time:
    #         starttime = earliest_st_time
    #     endtime = latest_det_time + (10 * 60)
    #     if endtime > latest_st_time:
    #         endtime = latest_st_time
    #     det_st = stream.trim(starttime=starttime, endtime=endtime)
    #     daylong = False
    # for temp in short_tribe:
    #     temp.process_len = endtime - starttime

    # rerun detection
    # TODO: if threshold is MAD, then I would have to set the threshold lower
    # than before. Or use other threshold here.
    short_party = short_tribe.detect(
        stream=det_st, threshold=threshold, trig_int=trig_int/10,
        threshold_type=threshold_type, overlap=overlap, plot=plot,
        plotdir=plotdir, daylong=daylong, pre_processed=pre_processed,
        fill_gaps=fill_gaps, ignore_bad_data=ignore_bad_data,
        ignore_length=ignore_length,
        parallel_process=parallel_process, cores=cores,
        concurrency=concurrency, xcorr_func=xcorr_func, arch=arch,
        group_size=group_size, output_event=False,
        full_peaks=full_peaks, save_progress=save_progress,
        process_cores=process_cores, spike_test=spike_test,
        use_weights=use_weights, copy_data=copy_data, **kwargs)
    # TODO: Sanity check: if there are ca. 1000 times or more detections
    #       for each long template, then rerun with higher threshold.

    # Check detections from short templates again the original set of
    # detections. If there is no short-detection for an original detection,
    # or if the detect_value is a lot worse, then remove original detection
    # from party.
    Logger.info(
        'Compare %s detections for short templates against %s existing detect'
        'ions', len([d for fam in short_party for d in fam]), n_detections_ok)
    return_party = Party()
    long_return_party = Party()
    short_return_party = Party()
    short_party_templ_names = [
        f.template.name for f in short_party if f is not None]
    for long_fam in long_party:
        if long_fam.template.name not in short_party_templ_names:
            continue  # do not retain the whole family
        # select matching family
        short_fam = short_party.select(long_fam.template.name)
        if len(short_fam) == 0:
            Logger.debug('Re-evaluation obtained no detections for %s.',
                          long_fam)
            continue
        short_det_times_np = np.array(
            [np.datetime64(d.detect_time.ns, 'ns') for d in short_fam])
        # Adjust by trace-offset when checking with an offset short tribe
        # (e.g., to check whether there is still significant correlation 
        # outside the time window for the first short tribe/templates)
        if hasattr(short_fam.template, 'trace_offset'):
            Logger.debug(
                'Template %s: Adjusting detection times with trace offset',
                short_fam.template.name)
            short_det_times_np += - np.timedelta64(
                int(short_fam.template.trace_offset * 1e9), 'ns')

        # Allow to return either partys with the original templates or the
        # short templates
        # if return_party_with_short_templates:
        #     return_family = Family(short_fam.template)
        # else:
        #     return_family = Family(long_fam.template)
        long_family = Family(long_fam.template)
        short_family = Family(short_fam.template)

        # Check detections for whether they fulfill the reevaluation-criteria.
        for det in long_fam:
            time_diffs = abs(
                short_det_times_np - np.datetime64(det.detect_time.ns, 'ns'))
            time_diff_thresh = np.timedelta64(
                int(time_difference_threshold * 1E9), 'ns')
            # If there is a short-detection close enough in time to the
            # original detection, then check detection values:
            # TODO: here I should pick the detection with the best detection-
            #       value within the time_difference_threshold, so I don't 
            #       pick a spurious value right next to it.
            if not any(time_diffs <= time_diff_thresh):
                Logger.debug('No detections within time-threshold found during'
                             ' re-evaluation of %s at %s', det.template_name,
                             det.detect_time)
            else:
                # Filter short-detections within time error threshold:
                t_diff_ind = np.arange(0, len(time_diffs))  # make index array
                # candidate_time_diffs = time_diffs[
                #     time_diffs <= time_diff_thresh]
                # Filter index array for detections within time threshold
                candidate_indices = t_diff_ind[time_diffs <= time_diff_thresh]
                # Find best detection within time threshold
                detection_values = [
                    detec.detect_val for id, detec in enumerate(short_fam)
                    if id in candidate_indices]
                cand_index = np.argmax(abs(np.array(detection_values)))
                sdi = candidate_indices[cand_index]
                # get the matching short-detection
                # sdi = np.argmin(time_diffs)
                short_det = short_fam[sdi]
                # If detection-value is now better or at least not a lot worse
                # within allowed error, only then keep the original detection.
                det_value_deviation_limit = abs(
                    det.detect_val / detect_value_allowed_reduction)
                # Compare MAD exceedance: If channels in templates have changed
                # between detection and picking, then detect_val may have
                # changed more than allowed, but as long as MAD is just a bit
                # the short-template detection should also be accepted.
                if threshold_type == 'MAD':
                    long_det_mad_exc = abs(det.detect_val / det.threshold)
                    mad_det_value_deviation_limit = (
                        long_det_mad_exc / detect_value_allowed_reduction)
                    short_det_mad_exc = abs(
                        short_det.detect_val / short_det.threshold)
                # Compare detection value for short vs long template
                if (abs(short_det.detect_val) >= det_value_deviation_limit):
                    # if return_party_with_short_templates:
                    #     return_family += short_det
                    # else:
                    #     return_family += det
                    long_family += det
                    short_family += short_det
                elif (threshold_type == 'MAD' and
                      short_det_mad_exc >= mad_det_value_deviation_limit):
                    long_family += det
                    short_family += short_det
                else:
                    Logger.info(
                        'Re-evaluation detections did not meet detection-value'
                        ' criterion for %s at %s (orig det. value: %s, new '
                        'det. value: %s, limit: %s', det.template_name,
                        det.detect_time, det.detect_val,
                        abs(short_det.detect_val), det_value_deviation_limit)
                    if threshold_type == 'MAD':
                        Logger.info(
                            'MAD values change for %s at %s (orig MAD exceedan'
                            'ce %s, new MAD exc. %s, limit: %s',
                            det.template_name, det.detect_time,
                            long_det_mad_exc, short_det_mad_exc,
                            mad_det_value_deviation_limit)
        # if len(return_family) >= 0:
        #     return_party += return_family
        if len(long_family) >= 0:
            # Quicker with append, to avoid checks between all templates:
            # long_return_party += long_family
            long_return_party.families.append(long_family)
            # short_return_party += short_family
            short_return_party.families.append(short_family)

    if len(long_return_party) == 0:
        n_detections = 0
        n_families = 0
    else:
    #     return_party = return_party.decluster(
    #         trig_int=trig_int, timing='detect', metric='thresh_exc',
    #         min_chans=min_chans, absolute_values=True)
        n_detections = len(long_return_party)
        n_families = len(long_return_party.families)
    Logger.info(
        'Re-evaluation of %s detections (%s families) finished, remaining are'
        ' %s detections (%s families).', n_detections_in, n_families_in,
        n_detections, n_families)
    if multiplot:
        multiplot_detection(long_return_party, short_tribe, det_st, **kwargs)

    return long_return_party, short_return_party



party.reevaluate_detections = reevaluate_detections