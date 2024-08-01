import os
from pathlib import Path
import pickle as pkl

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import log
from numpy import abs, angle, pi, real, imag, sqrt, max, min
import scipy.signal as sig
from scipy.ndimage import gaussian_filter1d, gaussian_filter

from batch_analyzer import BatchAnalyzerOsc
import common as cmn


fpath_in = (r'D:\WORK\Salvador\repo\A1_model_old\data'
            r'\grid_batch_v34_batch56_tcebkg_4\grid_00004_data.pkl')

with open(fpath_in, 'rb') as fid:
    sim_res = pkl.load(fid)
    
cell_names = list(sim_res['simData']['V_soma'].keys())
cell_idx = [int(name[5:]) for name in cell_names]
cell_idx.sort()

pop_cells = {}
for cell_id in cell_idx:
    pop_name = sim_res['net']['cells'][cell_id]['tags']['pop']
    if pop_name not in pop_cells:
        pop_cells[pop_name] = []
    pop_cells[pop_name].append(f'cell_{cell_id}')
    
vdata = sim_res['simData']['V_soma']

pop_name = 'HTC'
plt.figure()
for cell in pop_cells[pop_name]:
    vv = vdata[cell]
    plt.plot(vv)
plt.title(pop_name)

plt.figure()
n = 1
plt.plot(vdata[pop_cells['TC'][n]], label='TC')
plt.plot(vdata[pop_cells['HTC'][n]], label='HTC')
plt.plot(vdata[pop_cells['IRE'][n]], label='IRE')
plt.legend()
