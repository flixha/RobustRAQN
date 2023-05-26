
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Felix Halpaap
"""

# %%
import os
import glob
# from importlib import reload
from multiprocessing import Pool, cpu_count, get_context
from multiprocessing.pool import ThreadPool
from re import A
from joblib import Parallel, delayed, parallel_backend
import pandas as pd
from itertools import groupby
from timeit import default_timer

from obsplus import events_to_df
from obsplus.stations.pd import stations_to_df

from robustraqn.core.templates_creation import create_template_objects
from robustraqn.utils.spectral_tools import (
    get_updated_inventory_with_noise_models)

import logging
Logger = logging.getLogger(__name__)

# %% ############## MAIN ###################

if __name__ == "__main__":
    seisan_rea_path = '../SeisanEvents/'
    seisan_wav_path = '../SeisanEvents/'
    selected_stations = ['ASK','BER','BLS5','DOMB','EKO1','FOO','HOMB','HYA',
                        'KMY','MOL','ODD1','SKAR','SNART','STAV','SUE','KONO',
                        'BIGH','DRUM','EDI','EDMD','ESK','GAL1','GDLE','HPK',
                        'INVG','KESW','KPL','LMK','LRW','PGB1','MUD',
                        'EKB','EKB1','EKB2','EKB3','EKB4','EKB5','EKB6','EKB7',
                        'EKB8','EKB9','EKB10','EKR1','EKR2','EKR3','EKR4',
                        'EKR5','EKR6','EKR7','EKR8','EKR9','EKR10',
                        'NAO00','NAO01','NAO02','NAO03','NAO04','NAO05',
                        'NB200','NB201','NB202','NB203','NB204','NB205',
                        'NBO00','NBO01','NBO02','NBO03','NBO04','NBO05',
                        'NC200','NC201','NC202','NC203','NC204','NC205',
                        'NC300','NC301','NC302','NC303','NC304','NC305',
                        'NC400','NC401','NC402','NC403','NC404','NC405',
                        'NC600','NC601','NC602','NC603','NC604','NC605'] 
    # selected_stations = ['NAO01', 'NAO03', 'NB200']
    # selected_stations = ['ASK', 'BER', 'NC602']
    # 'SOFL','OSL',
    inv_file = '~/Documents2/ArrayWork/Inventory/NorSea_inventory.xml'
    inv = get_updated_inventory_with_noise_models(
        os.path.expanduser(inv_file),
        pdf_dir='~/repos/ispaq/WrapperScripts/PDFs/', check_existing=True,
        outfile=os.path.expanduser('~/Documents2/ArrayWork/Inventory/inv.pickle'))

    template_length = 40.0
    parallel = True
    noise_balancing = False
    cores = 20

    sfiles = glob.glob(os.path.join(seisan_rea_path, '*L.S??????'))
    sfiles = glob.glob(os.path.join(seisan_rea_path, '24-1338-14L.S201909'))
    # sfiles = glob.glob(os.path.join(seisan_rea_path, '04-1734-46L.S200706'))
    # sfiles = glob.glob(os.path.join(seisan_rea_path, '24-0101-20L.S200707'))
    # sfiles = glob.glob(os.path.join(seisan_rea_path, '20-1814-05L.S201804'))
    # sfiles = glob.glob(os.path.join(seisan_rea_path, '01-0545-55L.S201009'))
    # sfiles = glob.glob(os.path.join(seisan_rea_path, '30-0033-00L.S200806'))
    sfiles = glob.glob(os.path.join(seisan_rea_path, '05-1741-44L.S202101'))
    sfiles.sort(key=lambda x: x[-6:])

    highcut = 9.9
    if noise_balancing:
        lowcut = 0.5
    else:
        lowcut = 2.5

    # create_template_objects(sfiles, selected_stations, inv,
    tribe, wavenames = create_template_objects(
        sfiles, selected_stations, template_length, lowcut, highcut,
        min_snr=4.0, prepick=0.2, samp_rate=20.0, inv=inv,
        remove_response=True, seisan_wav_path=seisan_wav_path,
        noise_balancing=noise_balancing, min_n_traces=3,
        parallel=parallel, cores=cores, write_out=False, make_pretty_plot=True)
