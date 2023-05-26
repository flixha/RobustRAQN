
from obspy import read
from robustraqn.obspy.core import Stream

# %% TESTS
if __name__ == "__main__":
    from obspy import UTCDateTime
    st = read(
        '/data/seismo-wav/SLARCHIVE/1999/NO/SPA0/SHZ.D/NO.SPA0.00.SHZ.D.1999.213')
    st2 = st.slice(starttime=UTCDateTime(1999,8,1,20,36,51),
                   endtime=UTCDateTime(1999,8,1,20,36,54))
    st = st.mask_consecutive_zeros( min_run_length=5)
    st = st.split()
    # Taper all the segments
    st = st.taper_trace_segments()
