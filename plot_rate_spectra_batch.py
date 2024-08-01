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

# Firing rate calculation params
time_range = (2, None)
rbin_sz = 0.001
corr_taper_width = 0.5
pops_incl = None

# Stimulated and reference population
pop_name_stim = 'IT3'
ref = 'oscIT3'

# Firing rate smoothing
s = 5

# Action flags
need_plot_bars = 1
need_plot_r_fstim_group = 1
need_plot_r_pop_group = 1

# Calculate/load firing rate dynamics
rate_data = ba.calc_rate_data(
    rbin_sz=rbin_sz, time_range=time_range, pops_incl=pops_incl,
    need_recalc=False)

# Calculate/load firing rate (cross-)spectra
spect_data = ba.calc_spect_data(
    rbin_sz=rbin_sz, time_range=time_range, pops_incl=pops_incl,
    corr_taper_width=corr_taper_width, ref=ref, need_recalc=False)


pop_names = list(rate_data[0]['data'].keys())[:-1]
job_idx, ff_stim = ba.get_slice_idx({'osc_pop_name': pop_name_stim})
ff = spect_data[0]['data'][(pop_names[0], ref)]['ff']

# Output folders
dirpath_out = Path(dirpath_data) / 'spect' / f'stim_{pop_name_stim}'
dirpath_out_bar = dirpath_out / f'cross_ref={ref}_bar'
dirpath_out_r = dirpath_out / f'cross_ref={ref}_r_group=fstim_s={s}'
dirpath_out_r_pop = dirpath_out / f'cross_ref={ref}_r_group=pop_s={s}'
for dirpath in [dirpath_out_bar, dirpath_out_r]:
    os.makedirs(str(dirpath), exist_ok=True), dirpath_out_r_pop
    
tt = rate_data[0]['data'][ref]['tvec']

for job_id, fstim in enumerate(ff_stim):

    print(f'==== fstim = {fstim} ====')
    fvis_id = np.argmin(np.abs(ff - fstim))

    # Cross-spectra with reference population
    psd_cross = np.zeros(len(pop_names), dtype=np.complex128)
    for n, pop_name in enumerate(pop_names):
        psd = spect_data[job_id]['data'][(pop_name, ref)]['psd']
        rr = rate_data[job_id]['data'][pop_name]['rvec']
        psd /= max(rr)  # against spurious correlation due to large rate peaks
        psd_cross[n] = psd[fvis_id]

    # Bar plots for cross-spectral power and phase diff.
    if need_plot_bars:
        plt.figure(111)
        plt.clf()
        plt.show()
        plt.get_current_fig_manager().window.showMaximized()
        plt.subplot(2, 1, 1)
        h = abs(psd_cross)
        ax = plt.gca()
        ax.bar(pop_names, h)
        plt.xticks(rotation=90, fontsize=7)
        plt.title(f'Cross-spectral power (ref = {ref}, f = {fstim})', fontsize=8)
        plt.subplot(2, 1, 2)
        cols = np.tile(1 - sqrt(h / np.nanmax(h)), (3, 1)).T
        plt.bar(pop_names, np.angle(psd_cross), color=cols)
        plt.xticks(rotation=90, fontsize=7)
        plt.title(f'Phase difference (ref = {ref}, f = {fstim})', fontsize=8)
        fpath_out = dirpath_out_bar / f'fstim={fstim}.png'
        plt.tight_layout()
        plt.savefig(fpath_out, dpi=300)
    
    # Firing rate dynamics
    if need_plot_r_fstim_group:
        dirpath_out_r_ = dirpath_out_r / f'fstim={fstim}'
        os.makedirs(str(dirpath_out_r_), exist_ok=True)
        rvec0 = gaussian_filter1d(rate_data[job_id]['data'][ref]['rvec'], s)
        rvec0 = (rvec0 - rvec0.mean()) / rvec0.std()
        for m, pop_name in enumerate(pop_names):
            rvec = gaussian_filter1d(rate_data[job_id]['data'][pop_name]['rvec'], s)    
            rvec = (rvec - rvec.mean()) / rvec.std()
            plt.figure(111)
            plt.clf()
            plt.show()
            plt.get_current_fig_manager().window.showMaximized()
            plt.plot(tt, rvec0, label=ref)
            plt.plot(tt, rvec, 'k', label=pop_name)
            plt.legend()
            plt.xlabel('Time')
            plt.title('Firing rate, normalized')
            fpath_out = dirpath_out_r_ / f'{m}_{pop_name}_{ref}.png'
            plt.savefig(fpath_out, dpi=300)

if need_plot_r_pop_group:
    for m, pop_name in enumerate(pop_names):
        print(pop_name)
        dirpath_out_r_ = dirpath_out_r_pop / f'{m}_{pop_name}_{ref}'
        os.makedirs(str(dirpath_out_r_), exist_ok=True)
        for job_id, fstim in enumerate(ff_stim):
            rvec0 = gaussian_filter1d(rate_data[job_id]['data'][ref]['rvec'], s)
            rvec0 = (rvec0 - rvec0.mean()) / rvec0.std()
            rvec = gaussian_filter1d(rate_data[job_id]['data'][pop_name]['rvec'], s)    
            rvec = (rvec - rvec.mean()) / rvec.std()
            plt.figure(111)
            plt.clf()
            plt.show()
            plt.get_current_fig_manager().window.showMaximized()
            idx_vis = slice(100, -100)
            plt.plot(tt[idx_vis], rvec0[idx_vis], label=ref, linewidth=1)
            plt.plot(tt[idx_vis], rvec[idx_vis], 'k', label=pop_name, linewidth=1)
            plt.legend()
            plt.xlabel('Time')
            plt.title(f'Firing rate, normalized (f={fstim})')
            fpath_out = dirpath_out_r_ / f'fstim={fstim}.png'
            plt.savefig(fpath_out, dpi=300)
            

# =============================================================================
# # Amplitude and phase of cross-spectrum vs frequency (one pop. vs reference)
# plt.figure()
# pop_name = 'IT2'
# psd = spect_data[job_id]['data'][(pop_name, ref)]['psd']
# #psd_avg = np.mean(psd[fvis_mask])
# plt.subplot(2, 1, 1)
# plt.plot(ff, np.abs(psd))
# plt.plot(ff[fvis_mask], np.abs(psd[fvis_mask]), 'k.', markersize=15)
# #plt.plot(fvis, np.abs(psd_avg), 'k.', markersize=15)
# plt.xlim(0, 25)
# plt.title(f'Cross-spectral power ({pop_name}-{ref})')
# plt.subplot(2, 1, 2)
# for n in range(-4, 4):
#     col = 'r' if (n % 2) == 0 else 'r--'
#     plt.plot([min(ff), max(ff)], [n * pi, n * pi], col)
# phi = np.angle(psd)
# phi = cmn.make_closest_angle_seq(phi, mod=(2 * pi))
# plt.plot(ff, phi)
# plt.plot(ff[fvis_mask], phi[fvis_mask], 'k.', markersize=15)
# #plt.plot(fvis, np.angle(psd_avg) + 2 * pi, 'k.', markersize=15)
# plt.xlim(0, 25)
# plt.ylim(-15, 15)
# plt.title(f'Phase difference ({pop_name}-{ref})')
# plt.xlabel('Frequency')
# =============================================================================

# =============================================================================
# plt.figure()
# pop_name = 'IT2'
# psd = spect_data[job_id]['data'][(pop_name, ref)]['psd']
# plt.plot(np.real(psd), np.imag(psd))
# plt.plot(np.real(psd[fvis_mask]), np.imag(psd[fvis_mask]), 'k.')
# =============================================================================


