import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from batch_analyzer import BatchAnalyzerOsc


dirpath_data = r'D:\WORK\Salvador\repo\A1_model_old\data\grid_batch_5sec_IT3_w_10'
par_names = ['osc_pop_name', 'osc_f']
ba = BatchAnalyzerOsc(dirpath_data, par_names)

time_range = (2, None)
rbin_sz = 0.001
corr_taper_width = 0.5
pops_incl = None

pop_name_stim = 'IT3'
ref = 'oscIT3'

#norm_type = 'none'
norm_type = 'ref'

need_smooth = 1

rate_data = ba.calc_rate_data(
    rbin_sz=rbin_sz, time_range=time_range, pops_incl=pops_incl,
    need_recalc=False)
spect_data_self = ba.calc_spect_data(
    rbin_sz=rbin_sz, time_range=time_range, pops_incl=pops_incl,
    corr_taper_width=corr_taper_width, ref='self', need_recalc=False)
spect_data_cross = ba.calc_spect_data(
    rbin_sz=rbin_sz, time_range=time_range, pops_incl=pops_incl,
    corr_taper_width=corr_taper_width, ref=ref, need_recalc=False)

#job_idx, par_vals_free = ba._get_slice_idx({'osc_pop_name': 'IT3'})
#rr, par_vals_free = ba.get_data_slice(
#    rate_data, ['data', 'IT5A', 'r'], {'osc_pop_name': 'IT3'})

dirpath_out_base = Path(dirpath_data) / 'spect' / f'stim_{pop_name_stim}'
dirpath_out = (
    dirpath_out_base / 
    f'psd_ref={ref}_ctaper={corr_taper_width}_sm={need_smooth}_norm={norm_type}'
    )
os.makedirs(dirpath_out, exist_ok=True)

ff = spect_data_cross[0]['data'][(ref, ref)]['ff']

pop_pairs = list(spect_data_cross[0]['data'].keys())
for n, pop_pair in enumerate(pop_pairs):
    print(f'{pop_pair[0]}-{pop_pair[1]}')
    psd_key = 'psd_c' if need_smooth else 'psd'
    psd, fstim = ba.get_data_slice(
            spect_data_cross, ['data', pop_pair, psd_key], 
            {'osc_pop_name': pop_name_stim})
    psd0 = {}
    for m in [0, 1]:
        psd0[m], _ = ba.get_data_slice(
                spect_data_self, ['data', (pop_pair[m], pop_pair[m]), psd_key], 
                {'osc_pop_name': pop_name_stim})
    if norm_type == 'ref':
        psd /= np.sqrt(psd0[1])
    psd = np.abs(np.array(psd))
    plt.figure(111)
    plt.clf()
    ext = [ff[0], ff[-1], fstim[0], fstim[-1]]
    plt.imshow(np.log(psd), aspect='auto', origin='lower', extent=ext)
    plt.xlabel('Activity freq.')
    plt.ylabel('Stimulation freq.')
    plt.title(f'{pop_pair[0]}-{pop_pair[1]}, stim: {pop_name_stim}')
    fname_out = f'{n}_{pop_pair[0]}-{pop_pair[1]}.png'
    plt.savefig(dirpath_out / fname_out)
