
import copy
from robustraqn.obspy import Stream, Trace
from obspy import UTCDateTime


def _stream_quick_select(stream, seed_id):
    """
    4x quicker selection of traces in stream by full Seed-ID. Does not support
    wildcards or selection by network/station/location/channel alone.
    """
    net, sta, loc, chan = seed_id.split('.')
    stream = Stream(
        [tr for tr in stream
         if (tr.stats.network == net and
             tr.stats.station == sta and
             tr.stats.location == loc and
             tr.stats.channel == chan)])
    return stream


def _quick_copy_trace(trace, deepcopy_data=True):
    """
    Function to quickly copy a trace. Sets values in the traces' and trace
    header's dict directly, circumventing obspy's init functions.
    Speedup: from 37 us to 12 us per trace - 3x faster

    Warning: do not use to copy traces with processing history or response
    information.

    :type trace: :class:`obspy.core.trace.Trace`
    :param trace: Stream to quickly copy
    :type deepcopy_data: bool
    :param deepcopy_data:
        Whether to deepcopy trace data (with `deepcopy_data=False` expect up to
        20 % speedup, but use only when you know that data trace contents will
        not change or affect results).

    :rtype: :class:`obspy.core.trace.Trace`
    return: trace
    """
    new_trace = Trace()
    for key, value in trace.__dict__.items():
        if key == 'stats':
            new_stats = new_trace.stats
            for key_2, value_2 in value.__dict__.items():
                if isinstance(value_2, UTCDateTime):
                    new_stats.__dict__[key_2] = UTCDateTime(ns=value_2.ns)
                else:  # for scalars and strings
                    # This can not yet handle copy of complex stats like
                    # response object, processing history list, etc.
                    # copy.deepcopy(value_2)
                    new_stats.__dict__[key_2] = value_2
        elif deepcopy_data:
            # data needs to be deepcopied (and anything else, to be safe)
            new_trace.__dict__[key] = copy.deepcopy(value)
        else:  # No deepcopy, e.g. for NaN-traces with no effect on results
            new_trace.__dict__[key] = value
    return new_trace


def _quick_copy_stream(stream, deepcopy_data=True):
    """
    Function to quickly copy a stream.
    Speedup for simple trace:
        from 112 us to 44 (35) us per 3-trace stream - 2.8x (3.2x) faster

    Warning: use `deepcopy_data=False` (saves extra ~20 % time) only when the
             changing the data in the stream later does not change results
             (e.g., for NaN-trace or when data array will not be changed).

    This is what takes longest (1 empty trace, total time to copy 27 us):
    copy header: 18 us (vs create new empty header: 683 ns)

    Two points that can speed up copying / creation:
        1. circumvent trace.__init__ and trace.__set_attr__ by setting value
           directly in trace's __dict__
        2. when setting trace header, circumvent that Stats(header) is called
           when header is already a Stats instance

    :type stream: :class:`obspy.core.stream.Stream`
    :param stream: Stream to quickly copy
    :type deepcopy_data: bool
    :param deepcopy_data:
        Whether to deepcopy data (with `deepcopy_data=False` expect up to 20 %
        speedup, but use only when you know that data trace contents will not
        change or affect results).

    :rtype: :class:`obspy.core.stream.Stream`
    return: stream
    """
    new_traces = list()
    for trace in stream:
        new_traces.append(
            _quick_copy_trace(trace, deepcopy_data=deepcopy_data))
    return Stream(new_traces)