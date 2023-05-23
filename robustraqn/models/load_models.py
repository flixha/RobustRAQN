

import os
import glob
import numpy as np
from obspy.taup import TauPyModel
from obspy.taup.velocity_model import VelocityModel
from obspy.taup.taup_create import build_taup_model

# obspy taup velocity models

vel_model_files = glob.glob('*.tvel')
vel_model_files = glob.glob('barents16.tvel')

for vel_model_file in vel_model_files:
    build_taup_model(filename=vel_model_file, output_folder='.', verbose=True)


exit()

velmod = VelocityModel.read_tvel_file(filename='fescan.tvel')

model = TauPyModel(model="iasp91")

# model.model.ray_params



velmod = VelocityModel.read_tvel_file('/home/felix/Documents2/NorthSea/LowFrequencyPhase/Python/NNSN1D_plusAK135.tvel')
velmod = VelocityModel.read_tvel_file('/NNSN1D_plusAK135.tvel')
