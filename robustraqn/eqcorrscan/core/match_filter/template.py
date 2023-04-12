

from eqcorrscan.core.match_filter.template import Template


def check_template(self, template_length, remove_nan_strict=True,
                   max_perc_zeros=5, allow_channel_duplication=True, **kwargs):
    """
    Function to check that templates do not contain NaNs or zeros, do not
    contain duplicate channels, and that all traces are the same length.
    
    :type st: :class:`obspy.core.stream.Stream`
    :param st: Stream of templates to check.
    :type template_length: float
    :param template_length: Length of templates in seconds.
    :type remove_nan_strict: bool
    :param remove_nan_strict: If True, will remove traces that contain NaNs
    :type max_perc_zeros: float
    :param max_perc_zeros: Maximum percentage of zeros allowed in a trace.
    :type allow_channel_duplication: bool
    :param allow_channel_duplication:
        If True, will allow duplicate channels, otherwise it will remove the
        later duplicated channel.

    :return: Stream of templates with NaNs removed.
    :rtype: :class:`obspy.core.stream.Stream`
    """
    # Now check the templates
    st = self.stream
    # Check that all traces are the same length:
    t_lengths = [len(tr.data) for tr in st]
    t_length_max = max(t_lengths)
    if any([t_item == t_length_max for t_item in t_lengths]):
        Logger.info('Template stream: %s has traces with unequal lengths.',
                    st[0].stats.starttime)
    # Check each trace
    k = 0
    channel_ids = list()
    st_copy = st.copy()
    for tr in st_copy:
        # Check templates for duplicate channels (happens when there are
        # P- and S-picks on the same channel). Then throw away the
        # S-trace (the later one) for now.
        if tr.id in channel_ids:
            for j in range(0, k):
                test_same_id_trace = st[j]
                if tr.id == test_same_id_trace.id:
                    # remove if the duplicate traces have the same start-time
                    if (tr.stats.starttime == test_same_id_trace.stats.starttime
                            and tr in st):
                        st.remove(tr)
                        continue
                    # if channel-duplication is forbidden, then throw away the
                    # later trace (i.e., S-trace)
                    elif not allow_channel_duplication:
                        st.remove(test_same_id_trace)
                        continue
        else:
            channel_ids.append(tr.id)
            k += 1

    st_copy = st.copy()
    for tr in st_copy:
        # Check that the trace is long enough
        if tr.stats.npts < template_length*tr.stats.sampling_rate and tr in st:
            st.remove(tr)
            Logger.info(
                'Trace %s %s is too short (%s s), removing from template.',
                tr.id, tr.stats.starttime, 
                str(tr.stats.npts / tr.stats.sampling_rate))
        # Check that the trace has no NaNs
        if remove_nan_strict and any(np.isnan(tr.data)) and tr in st:
            st.remove(tr)
            Logger.info('Trace %s contains NaNs, removing from template.',
                        tr.id)
        # Check that not more than 5 % of the trace is zero:
        n_nonzero = np.count_nonzero(tr.copy().detrend().data)
        # if sum(tr.copy().detrend().data==0) > tr.data.size*max_perc_zeros\
        if (n_nonzero < tr.data.size * (1-max_perc_zeros) and tr in st):
            st.remove(tr)
            Logger.info('Trace %s contains more than %s %% zeros, removing '
                        'from template.', tr.id, str(max_perc_zeros*100))
    self.stream = st
    return self


Template.check_template = check_template