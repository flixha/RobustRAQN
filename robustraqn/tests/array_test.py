
# %% TEST

import os
import pandas as pd
from obspy.io.nordic.core import read_nordic, write_select
from obsplus.stations.pd import stations_to_df

from obspy.clients.filesystem.sds import Client
from obspy.core.event import Catalog
from eqcorrscan.core.match_filter import Party, Tribe
from robustraqn.templates_creation import create_template_objects
from robustraqn.event_detection import run_day_detection
from robustraqn.detection_picking import pick_events_for_day
from robustraqn.quality_metrics import read_ispaq_stats
from robustraqn.lag_calc_postprocessing import (
    check_duplicate_template_channels, postprocess_picked_events)
from robustraqn.spectral_tools import get_updated_inventory_with_noise_models
from robustraqn.seimic_array_tools import (
    add_array_station_picks, extract_array_picks, array_lac_calc)

parallel = True
cores = 32
make_templates = True
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

# %%

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
        templ_path='data/Templates', make_pretty_plot=False,
        normalize_NSLC=True)
else:
    tribe = Tribe().read(
        '/home/felix/Documents2/BASE/Detection/Bitdalsvatnet/TemplateObjects/Templates_min13tr_1.tgz')

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
        write_party=True, detection_path='data/Detections',
        min_chans=3, return_stream=True)
else:
    party = Party().read('data/Detections/UniqueDet2021-01-05.tgz')
# party[0].detections = [party[0][10]]
# party[0].detections = [party[0][0]]



# %%
export_catalog = pick_events_for_day(
    tribe=pick_tribe, det_tribe=tribe, template_path=None,
    date=date, det_folder='data/Detections', dayparty=party,
    ispaq=ispaq, clients=[client], relevant_stations=selected_stations,
    array_lag_calc=True, inv=inv, parallel=True, cores=cores,
    write_party=False, n_templates_per_run=1, min_det_chans=5, min_cc=0.4,
    interpolate=True, archives=['/data/seismo-wav/SLARCHIVE'], 
    sfile_path='data/Sfiles', operator='feha', stations_df=stations_df)



# %%
picked_catalog = array_lac_calc(
    day_st, export_catalog, party, tribe, stations_df, min_cc=0.4,
    pre_processed=False, shift_len=0.8, min_cc_from_mean_cc_factor=0.6,
    horizontal_chans=['E', 'N', '1', '2'], vertical_chans=['Z'],
    parallel=False, cores=1, daylong=True)



# %%
# Large aperature array test
from obspy.clients.filesystem.sds import Client
from eqcorrscan.core.match_filter import Party, Tribe
from robustraqn.templates_creation import create_template_objects
from robustraqn.event_detection import run_day_detection
from robustraqn.detection_picking import pick_events_for_day
from robustraqn.quality_metrics import read_ispaq_stats
from robustraqn.lag_calc_postprocessing import (
    check_duplicate_template_channels, postprocess_picked_events)
from robustraqn.spectral_tools import get_updated_inventory_with_noise_models
from robustraqn.obspy.clients.filesystem.sds import Client

parallel = False
cores = 1

inv_file = '~/Documents2/ArrayWork/Inventory/NorSea_inventory.xml'
# inv = read_inventory(os.path.expanduser(invile))
inv = get_updated_inventory_with_noise_models(
    inv_file=os.path.expanduser(inv_file),
    pdf_dir=os.path.expanduser('~/repos/ispaq/WrapperScripts/PDFs/'),
    check_existing=True,
    outfile=os.path.expanduser(
        '~/Documents2/ArrayWork/Inventory/inv.pickle'))
    
archive_path = '/data/seismo-wav/SLARCHIVE'
client = Client(archive_path)

sfiles = ['/home/felix/Documents2/Ridge//Seisan/INTEU/2021/07/05-0423-41R.S202107']
selected_stations = [
    'ARA0', 'KEV',
    'BJO1', 'BEAR', 'BEA1', 'BEA2', 'BEA3', 'BEA4', 'BEA5', 'BEA6',
    'STEI', 'LOF', 'TRO', 'JETT', 'STOK', 'KONS', 'RAUS', 'LEIR', 'GILDE',
    'MOR8', 'VAGH', 'ROEST', 'VBYGD', 'LOSSI']

tribe, _ = create_template_objects(
    sfiles, selected_stations, template_length=40, lowcut=4.0,
    highcut=19.9, min_snr=3, prepick=0.3, samp_rate=40.0,
    min_n_traces=13, seisan_wav_path=None,
    inv=inv, remove_response=False, output='VEL', add_array_picks=True,
    add_large_aperture_array_picks=True,
    parallel=parallel, cores=cores, write_out=True,
    templ_path='tests/data/Templates', make_pretty_plot=False,
    normalize_NSLC=True)