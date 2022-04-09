

import os
import glob
import numpy as np
from obspy.taup import TauPyModel
from obspy.taup.velocity_model import VelocityModel

# obspy taup velocity models

vel_model_files = glob.glob('*.tvel')
model = TauPyModel(model="iasp91")

model.model.ray_params

VelocityModel.read_tvel_file(model_name='ispaq91')

velmod = VelocityModel.read_tvel_file('/home/felix/Documents2/NorthSea/LowFrequencyPhase/Python/NNSN1D_plusAK135.tvel')
velmod = VelocityModel.read_tvel_file('/NNSN1D_plusAK135.tvel')