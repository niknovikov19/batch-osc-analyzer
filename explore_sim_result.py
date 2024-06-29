# -*- coding: utf-8 -*-
"""Explore the result of a model simulation.

@author: Nikita Novikov
"""

#import importlib
import json
#import logging
import os
from pathlib import Path
import pickle as pkl
import scipy as sc
import scipy.signal as sig
import sys

import matplotlib.pyplot as plt
from netpyne import specs, sim
import numpy as np

import common as cmn


dirpath_root = Path(r'D:\WORK\Salvador\repo\A1_model_old\data')

batch_name = 'grid_batch_IT3'

run_id = '00002'

rbin_sz = 0.001
corr_taper_width = 0.2

pops_included = ['oscIT3', 'IT2', 'IT3', 'ITP4', 'IT5A', 'IT5B', 'IT6']
pops_excluded = None

pop_name_0 = 'oscIT3'


# Load simulation result
dirpath_batch = dirpath_root / batch_name
fpath_sim = dirpath_batch / f'grid_{run_id}_data.pkl'
with open(fpath_sim, 'rb') as fid:
    sim_result = pkl.load(fid)
    
# Rate dynamics and spike times for each population
Tsim = cmn.get_sim_duration(sim_result) / 1000
pop_data = {}
for pop_name, pop in cmn.get_pop_params(sim_result).items():
    if (pops_included is not None) and (pop_name not in pops_included):
        continue
    if (pops_excluded is not None) and (pop_name in pops_excluded):
        continue
    print(f'Firing rate: {pop_name}')
    spike_times = cmn.get_netpyne_pop_spikes(sim_result, pop_name)
    cell_count = cmn.get_pop_size(sim_result, pop_name)
    tvec, rvec = cmn.calc_firing_rate_dynamics(
            spike_times, time_range=(0.2, Tsim), dt=rbin_sz,
            Ncells=cell_count)
    pop_data[pop_name] = {
        'cell_count': cell_count,
        'spike_times': spike_times,
        'tvec': tvec,
        'rvec': rvec,
        'r': np.mean(rvec)
    }
pop_names = list(pop_data.keys())
Npop = len(pop_data)

# Correlations of the firing rate signals
corr_data = {}
for pop1_name, pop1 in pop_data.items():
    for pop2_name, pop2 in pop_data.items():
        print(f'Correlation: {pop1_name} - {pop2_name}')
        rvec1 = pop1['rvec'] - pop1['rvec'] .mean()
        rvec2 = pop2['rvec'] - pop2['rvec'] .mean()
        cc, lags = cmn.calc_rvec_crosscorr(rvec1, rvec2, rbin_sz)
        H = sc.stats.norm.pdf(lags, scale=corr_taper_width)
        corr_data[(pop1_name, pop2_name)] = {
                'corr': cc, 'corr_c': cc * H, 'lags': lags}
        
plot_corr = 1
if plot_corr:
    plt.figure()
    plt_num = 1
    for pop1_name in pop_data:
        for pop2_name in pop_data:
            C = corr_data[(pop1_name, pop2_name)] 
            plt.subplot(Npop, Npop, plt_num)
            plt_num += 1
            cm = C['corr'].max()
            plt.plot([0, 0], [-cm, cm], 'r--')
            plt.plot(C['lags'], C['corr'])
            #plt.plot(C['lags'], C['corr_c'])
            plt.title(f'{pop1_name} - {pop2_name}')
            plt.xlim((-0.5, 0.5))
            
nx = 3
ny = 3
plt.figure()
pop_num = 0
for nx_ in range(nx):
    for ny_ in range(ny):
        if pop_num >= len(pop_names):
            break
        plt.subplot(ny, nx, pop_num + 1)
        pop1_name = pop_names[pop_num]
        pop2_name = pop_name_0
        C = corr_data[(pop1_name, pop2_name)]
        cm = C['corr'].max()
        plt.plot([0, 0], [-cm, cm], 'r--')
        plt.plot(C['lags'], C['corr'])
        #plt.plot(C['lags'], C['corr_c'])
        plt.title(f'{pop1_name} - {pop2_name}')
        plt.xlim((-0.5, 0.5))
        pop_num += 1
        
    
# Print mean firing rates
print('Firing rate: ')
for pop_name, pop in pop_data.items():
    print(f'{pop_name}: {pop["r"]}')
 
def norm(x):
    return (x - x.mean()) / x.std()
    
# Plot firing rate dynamics
plt.figure()
smooth_win = 10
for pop_name, pop in pop_data.items():
    if pop_name not in ['oscIT3', 'IT5B']:
        continue
    rvec_sm, tvec_sm = cmn.smooth_data(pop['rvec'], pop['tvec'],
                                       win_len=smooth_win)
    mask = tvec_sm > 0
    plt.plot(tvec_sm[mask], norm(rvec_sm[mask]), label=pop_name)
plt.xlabel('Time, s')
plt.title('Firing rate, Hz')
plt.legend()

# =============================================================================
# def fix_angle(phi):
#     pi = np.pi
#     phi = phi % (2 * pi)
#     phi[phi >= pi] -= pi
#     return phi
# =============================================================================

# Power spectral density
plt.figure(111)
plt.subplot(2, 1, 1)
for n, pop_name in enumerate(pop_data):
    if pop_name == pop_name_0:
        continue
    C = corr_data[(pop_name, pop_name)] 
    ff, psd = cmn.fft_smart(C['corr_c'], C['lags'], fmax=150)
    plt.plot(ff, np.abs(psd), label=pop_name)
    plt.xlim(2, 40)
    #plt.ylim(0, 500)
plt.title('PSD')
plt.legend()

from numpy import pi

# Phase difference
plt.figure(111)
plt.subplot(2, 1, 2)
for n, pop_name in enumerate(pop_data):
    C = corr_data[(pop_name, pop_name_0)] 
    ff, psd = cmn.fft_smart(C['corr_c'][:-1], C['lags'], fmax=150)
    phi = np.angle(psd)
    z = np.exp(1j * phi)
    z, ff = cmn.smooth_data(z, ff, win_len=10)
    phi = np.angle(z)
    phi = cmn.make_closest_angle_seq(phi, mod=(2 * pi), step=1)
    #phi = ((phi + pi) % (2 * pi)) - pi
    plt.plot(ff, phi, '-', label=pop_name)
plt.xlabel('Frequency')
plt.title(f'Phase diff. with {pop_name_0}')
plt.legend()

f0 = 15
df = 1
fband = (f0 - df, f0 + df)
W = {}
for pop1_name, pop1 in pop_data.items():
    for pop2_name, pop2 in pop_data.items():
        C = corr_data[(pop1_name, pop2_name)] 
        ff, psd = cmn.fft_smart(C['corr_c'][:-1], C['lags'], fmax=150)
        mask = (ff >= fband[0]) & (ff <= fband[1])
        W[(pop1_name, pop2_name)] = np.mean(psd[mask])

plt.figure()
for pop_name in pop_data:
    w = W[(pop_name, pop_name_0)]
    w00 = W[(pop_name_0, pop_name_0)]
    w0 = W[(pop_name, pop_name)] * w00
    w /= np.sqrt(np.abs(w0))
    w *= np.sign(np.real(w00))
    #w = np.sqrt(np.abs(w)) * np.exp(1j * np.angle(w))
    plt.plot([0, np.real(w)], [0, np.imag(w)], '.-',
             label=pop_name)
plt.xlabel('Re')
plt.ylabel('Im')
plt.title(f'Corr. with {pop_name_0}')
plt.legend()

# =============================================================================
# plt.figure()
# C = corr_data[('IT3', pop_name_0)] 
# ff, psd = cmn.fft_smart(C['corr'], C['lags'], fmax=100)
# #psd /= np.abs(psd)
# Nf = len(ff)
# col1 = np.array([1, 0, 0], ndmin=2)
# col2 = np.array([0, 0, 1], ndmin=2)
# q = np.linspace(0, 1, Nf)
# cols = q * col1.T + (1 - q) * col2.T
# plt.scatter(psd.real, psd.imag, c=cols.T)
# =============================================================================

