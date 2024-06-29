import os
from pathlib import Path

import matplotlib.pyplot as plt
from netpyne.plotting.plotter import colorList
import numpy as np
from numpy import abs, angle, pi, real, imag, sqrt, max, min
from scipy.ndimage import gaussian_filter1d

from batch_analyzer import BatchAnalyzerOsc
import common as cmn


dirpath_data = r'D:\WORK\Salvador\repo\A1_model_old\data\grid_batch_5sec_IT3_IT5A_IT5B'
par_names = ['osc_pop_name', 'osc_f']
ba = BatchAnalyzerOsc(dirpath_data, par_names)

time_range = (2, None)
rbin_sz = 0.001
corr_taper_width = 0.05
pops_incl = None

ref = 'IT3'

rate_data = ba.calc_rate_data(
    rbin_sz=rbin_sz, time_range=time_range, pops_incl=pops_incl,
    need_recalc=False)
spect_data = ba.calc_spect_data(
    rbin_sz=rbin_sz, time_range=time_range, pops_incl=pops_incl,
    corr_taper_width=corr_taper_width, ref=ref, need_recalc=False)

fstim = 11
ff_stim = np.unique([sd['params']['osc_f'] for sd in spect_data])
job_id = np.argmin(abs(ff_stim - fstim))

pop_names = list(rate_data[0]['data'].keys())

fvis = 11
df = 0.1
ff = spect_data[job_id]['data'][(pop_names[0], ref)]['ff']
fvis_mask = (ff >= (fvis - df)) & (ff <= (fvis + df))

psd_cross = {}
for n, pop_name in enumerate(pop_names):
    psd = spect_data[job_id]['data'][(pop_name, ref)]['psd']
    rr = rate_data[job_id]['data'][pop_name]['rvec']
    psd /= max(rr)
    psd_cross[pop_name] = psd[fvis_mask][0]

# Complex-valued cross-spectra at stim. freq. for every pop. vs reference
plt.figure()
for n, pop_name in enumerate(pop_names):
    psd = spect_data[job_id]['data'][(pop_name, ref)]['psd']
    psd = sqrt(np.abs(psd)) * np.exp(1j * np.angle(psd))
    col = colorList[n % len(colorList)]
    plt.plot(real(psd[fvis_mask]), imag(psd[fvis_mask]), '.',
             color=col, label=pop_name, markersize=10)
plt.plot(0, 0, 'k.')
plt.legend()
plt.xlabel('Re')
plt.ylabel('Im')

plt.figure()
h = np.array([abs(psd) for psd in psd_cross.values()])
#h = sqrt(h)
plt.bar(pop_names, h)
plt.xticks(fontsize=7)

# Amplitude and phase of cross-spectrum vs frequency (one pop. vs reference)
plt.figure()
pop_name = 'IT2'
psd = spect_data[job_id]['data'][(pop_name, ref)]['psd']
#psd_avg = np.mean(psd[fvis_mask])
plt.subplot(2, 1, 1)
plt.plot(ff, np.abs(psd))
plt.plot(ff[fvis_mask], np.abs(psd[fvis_mask]), 'k.', markersize=15)
#plt.plot(fvis, np.abs(psd_avg), 'k.', markersize=15)
plt.xlim(0, 25)
plt.title(f'Cross-spectral power ({pop_name}-{ref})')
plt.subplot(2, 1, 2)
for n in range(-4, 4):
    col = 'r' if (n % 2) == 0 else 'r--'
    plt.plot([min(ff), max(ff)], [n * pi, n * pi], col)
phi = np.angle(psd)
phi = cmn.make_closest_angle_seq(phi, mod=(2 * pi))
plt.plot(ff, phi)
plt.plot(ff[fvis_mask], phi[fvis_mask], 'k.', markersize=15)
#plt.plot(fvis, np.angle(psd_avg) + 2 * pi, 'k.', markersize=15)
plt.xlim(0, 25)
plt.ylim(-15, 15)
plt.title(f'Phase difference ({pop_name}-{ref})')
plt.xlabel('Frequency')

# =============================================================================
# plt.figure()
# pop_name = 'IT2'
# psd = spect_data[job_id]['data'][(pop_name, ref)]['psd']
# plt.plot(np.real(psd), np.imag(psd))
# plt.plot(np.real(psd[fvis_mask]), np.imag(psd[fvis_mask]), 'k.')
# =============================================================================

rvec1 = gaussian_filter1d(rate_data[job_id]['data']['IT3']['rvec'], 10)
rvec2 = gaussian_filter1d(rate_data[job_id]['data']['SOM5B']['rvec'], 10)
rvec1 = (rvec1 - rvec1.mean()) / rvec1.std()
rvec2 = (rvec2 - rvec2.mean()) / rvec2.std()
tt = rate_data[job_id]['data']['IT3']['tvec']
plt.figure()
plt.plot(tt, rvec1)
plt.plot(tt, rvec2)
