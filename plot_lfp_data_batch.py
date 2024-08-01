import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
#import scipy.signal as sig
from scipy.ndimage import gaussian_filter1d, gaussian_filter

from batch_analyzer import BatchAnalyzerOsc
import sim_res_parser as srp

matplotlib.use('qtagg')


def smooth(X, sm0=0, sm1=0):
    if (sm0 == 0) and (sm1 == 0):
        return X
    if (sm0 != 0) and (sm1 != 0):
        return gaussian_filter(X, (sm0, sm1))
    if (sm0 != 0) and (sm1 == 0):
        return gaussian_filter1d(X, sm0, axis=0)
    if (sm0 == 0) and (sm1 != 0):
        return gaussian_filter1d(X, sm1, axis=-1)
    
def plot_layer_borders(sim_result, x0, x1, all_layers=True):
    if all_layers:
        pops_vis = ['IT2', 'IT3', 'IT5A', 'IT5B', 'IT6', 'ITP4']
    else:
        pops_vis = ['ITP4']
    for pop_name in pops_vis:
        yy_ = srp.get_pop_ylim(sim_result, pop_name)
        if pop_name == 'ITP4':
            col = 'k'
        else:
            col = 'k'
        for n in [0, 1]:
            plt.plot([x0, x1], [yy_[n], yy_[n]], f'{col}--')


#dirpath_data = r'D:\WORK\Salvador\repo\A1_model_old\data\grid_batch_5sec_IT3_w_10'
#dirpath_data = r'D:\WORK\Salvador\repo\A1_model_old\data\grid_batch_5sec_IT3_IT5A_IT5B'
dirpath_data =  r'D:\WORK\Salvador\repo\A1_model_old\data\grid_batch_v34_batch56_IT3osc_repbkg'
par_names = ['osc_pop_name', 'osc_f']

pop_name_stim = 'IT3'

# Time limits to analyze
t_lim = (1, 10)

# Reference channel
#ref_chan = None
ref_chan = 0

fmax = 150

# LFP or CSD
#pow_type = 'lfp'
#pow_type = 'bipolar'
pow_type = 'csd'

# Normalize each freq. power by its max. over channels
pow_norm = 'none'
#pow_norm = 'freq'
#pow_norm = 'band'

need_log = 1

# Smoothing window lengths
sm_f = 10
sm_d = 1

# Frequency bands
fband_low = (10, 30)
fband_high = (50, 150)

layer_groups = [(0, 950), (950, 1250), (1250, 2000)]
#layer_groups = [(0, 1000), (1000, 2000)]

need_smooth = 1

# Action flags
need_recalc = 0
need_reload = 1
need_plot_f = 0
need_plot_d = 1
need_plot_fd = 0


# Initialize batch result analyzer
ba = BatchAnalyzerOsc(dirpath_data, par_names)

# Calculate/load LFP data
lfp_par_names = {'t_lim', 'ref_chan', 'pow_type', 'fmax', 'pow_norm',
             'layer_groups', 'fbands'}
lfp_par_dict = {key: val for key, val in locals().items() if key in lfp_par_names}
if need_reload or 'lfp_data' not in locals():
    lfp_data = ba.calc_lfp_data(lfp_par_dict, need_recalc=need_recalc)
    
# Layer boundaries
if need_reload or 'sim_result' not in locals():
    sim_result = ba.load_job_sim_data(0)


# Output folder
dirpath_out_base = Path(dirpath_data) / 'spect' / f'stim_{pop_name_stim}'
dirpath_out = (
    dirpath_out_base / 
    f'lfp_ref={ref_chan}_pow={pow_type}_norm={pow_norm}_log={need_log}'
    )
os.makedirs(dirpath_out, exist_ok=True)

ff = lfp_data[0]['data']['freqs']
dd = lfp_data[0]['data']['depths']

# Select LFP data for a given input population
lfp_pow, fstim = ba.get_data_slice(
        lfp_data, ['data', 'W'], 
        {'osc_pop_name': pop_name_stim})

fvis_lim = (2, 150)
f_mask = ((ff >= fvis_lim[0]) & (ff <= fvis_lim[1]))

for W, f in zip(lfp_pow, fstim):
    print(f'f = {f}')
    Wsm = smooth(W[:, f_mask], sm_d, sm_f)
    if need_log:
        Wsm = np.log(Wsm)    
    plt.figure(111)
    plt.clf()
    ext = (ff[f_mask][0], ff[f_mask][-1], dd[-1], dd[0])
    plt.imshow(Wsm, aspect='auto', extent=ext, origin='upper', vmin=0, vmax=4.5)
               #vmin=-2.5, vmax=-0.25)
    plot_layer_borders(sim_result, ext[0], ext[1])
    plt.xlabel('Frequency')
    plt.ylabel('Depth')
    title_str = f'fstim={f:.02f}, {pow_type}, ref={ref_chan}, norm={pow_norm}, log={need_log}'
    plt.title(title_str)
    plt.draw()
    plt.colorbar()
    fpath_out = dirpath_out / f'fstim={f}.png'
    plt.savefig(fpath_out, dpi=300)
