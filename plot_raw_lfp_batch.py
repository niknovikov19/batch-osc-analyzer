import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
#from netpyne.plotting.plotter import colorList
import numpy as np
from numpy import abs, angle, pi, real, imag, sqrt, max, min
from scipy.ndimage import gaussian_filter1d

from batch_analyzer import BatchAnalyzerOsc
import common as cmn

matplotlib.use('qtagg')


dirpath_data = r'D:\WORK\Salvador\repo\A1_model_old\data\grid_batch_5sec_IT3_w_10'
par_names = ['osc_pop_name', 'osc_f']
ba = BatchAnalyzerOsc(dirpath_data, par_names)

pop_name_stim = 'IT3'

t_lim = (2000, 3000)

dirpath_out = Path(dirpath_data) / 'spect' / f'stim_{pop_name_stim}' / 'LFP_td'
os.makedirs(str(dirpath_out), exist_ok=True)

job_idx, ff_stim = ba.get_slice_idx({'osc_pop_name': pop_name_stim})

for job_id, fstim in zip(job_idx, ff_stim):

    print(f'==== fstim: {fstim} ====')

    #job_id = 0
    #sim_result = ba.load_job_sim_data(job_id)
    
    lfp, tt, lfp_coords = ba.get_job_lfp(job_id)
    yy = lfp_coords[:, 1]
    mask = (tt >= t_lim[0]) & (tt <= t_lim[1])
    tt = tt[mask]
    lfp = lfp[mask, :]
    
    par = ba.par_vals_lst[job_id]
    
    plt.figure(111)
    plt.clf()
    plt.show()
    plt.get_current_fig_manager().window.showMaximized()
    ext = [tt[0], tt[-1], yy.max(), yy.min()]
    plt.imshow(lfp.T, aspect='auto', extent=ext, origin='upper')
    plt.colorbar()
    plt.xlabel('Time')
    plt.ylabel('Depth')
    #s = ', '.join([f'{name}={val}' for name, val in par.items()])
    #plt.title(s)
    plt.title(f'LFP, fstim={fstim} Hz')
    
    #s = '_'.join([f'{name}={val}' for name, val in par.items()])
    fpath_out = dirpath_out / f'fstim={fstim}.png'
    plt.savefig(fpath_out)


# =============================================================================
# job_id = 10
# fstim = ff_stim[job_id]
# lfp, tt, lfp_coords = ba.get_job_lfp(job_id)
# yy = lfp_coords[:, 1]
# mask = (tt >= 2000) & (tt <= 3000)
# tt_ = tt[mask]
# lfp_ = lfp[mask, :]
# plt.figure()
# for n in [0, 4, 8, 12]:
#     plt.plot(tt_, lfp_[:, n].T, label=f'y={yy[n]}')
# plt.xlabel('Time')
# plt.legend()
# plt.title(f'fstim = {fstim}')
# =============================================================================
