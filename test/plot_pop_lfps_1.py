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
            r'\grid_batch_v34_batch56_tcebkg_4\grid_00003_data.pkl')

with open(fpath_in, 'rb') as fid:
    sim_result = pkl.load(fid)    
lfp, tt, lfp_coords = BatchAnalyzerOsc.get_pop_lfps(sim_result)
tt /= 1000
yy = lfp_coords[:, 1]
pop_names = list(lfp.keys())

mask = (tt >= 1)

nx = 3
ny = 2
plt.figure()
for n, pop_name in enumerate(pop_names):
    plt.subplot(ny, nx, n + 1)
    ext = [tt[mask][0], tt[mask][-1], yy.max(), yy.min()]
    X = lfp[pop_name].T[:, mask]
    v = np.max(np.abs(X.ravel()))
    plt.imshow(X, aspect='auto', extent=ext, origin='upper', vmin=-v, vmax=v)
    plt.colorbar()
    plt.xlabel('Time')
    plt.ylabel('Depth')
    plt.title(pop_name)