
# %%

import os
from obspy.core import UTCDateTime
import robustraqn
from robustraqn.quality_metrics import (
    create_bulk_request, get_waveforms_bulk, read_ispaq_stats,
    get_parallel_waveform_client)


from obspy.clients.fdsn import Client

metrics_path = os.path.join(os.path.dirname(robustraqn.__file__),
                            'tutorial/data/ispaq/Parquet_database/csv/')

stations = ['BER']

# Read in data quality metrics
ispaq = read_ispaq_stats(
    folder=metrics_path, stations=stations, startyear=2020, endyear=2020,
    ispaq_prefixes=['all'], ispaq_suffixes=['simpleMetrics', 'PSDMetrics'],
    file_type = 'parquet')

# Create a bulk-request from data quality metrics
bulk_request, day_stats = create_bulk_request(
    starttime=UTCDateTime(2020, 1, 10), endtime=UTCDateTime(2020, 1, 11),
    stats=ispaq, parallel=True, cores=2,
    stations=stations, location_priority=['10','00',''],
    band_priority=['B','H','S','E','N'], instrument_priority=['H'],
    components=['Z','N','E','1','2'],
    min_availability=0.8, max_cross_talk=1,
    max_spikes=1000, max_glitches=1000, max_num_gaps=500,
    max_num_overlaps=1000, max_max_overlap=86400,
    max_dead_channel_lin=3, require_alive_channel_gsn=True,
    max_pct_below_nlnm=50, max_pct_above_nhnm=50,
    min_sample_unique=150, max_abs_sample_mean=1e7,
    min_sample_rms=1e-6, max_sample_rms=1e8,
    max_sample_median=1e6, min_abs_sample_average=(1, 1e-9),
    require_clock_lock=False, max_suspect_time_tag=86400)


client = Client('UIB-NORSAR')
# Monkey-patch client to allow parallel waveform request
client = get_parallel_waveform_client(client)

# Request the data with a bulk-request
st = client.get_waveforms_bulk_parallel(bulk_request, parallel=True,
                                            cores=2)
# alternative API:
# st = get_waveforms_bulk(client, bulk, parallel=True, cores=2)

# %%
