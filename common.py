# -*- coding: utf-8 -*-
"""Various functions used in the toolbox.

@author: Nikita Novikov
"""

import copy
#import inspect

import json
#import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import scipy as sc
import scipy.signal as sig
import time

#from netpyne import sim


def calc_rvec_crosscorr(rvec1, rvec2, bin_sz):
    """ Calculate cross-correlation of two firing rate vectors. """
    cc = np.correlate(rvec1, rvec2, mode='full') / len(rvec1)
    nbins = len(rvec1)
    lags = np.arange(-nbins+1, nbins) * bin_sz
    return cc, lags

def calc_rvec_autocorr(rvec, bin_sz):
    """ Calculate autocorrelation of a firing rate vector. """
    return calc_rvec_crosscorr(rvec, rvec, bin_sz)


def correlate2(x, y, lag_range=None):
    """ Similar to np.correlate(), but with a given range of lags. """
    if lag_range is None:
        return np.correlate(x, y, mode='full')
    #x_mean = np.mean(x)
    #y_mean = np.mean(y)
    x_mean = 0
    y_mean = 0
    #denominator = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
    denominator = 1
    result = []
    for lag in range(min(lag_range), max(lag_range) + 1):
        if lag == 0:
            numerator = np.sum((x - x_mean) * (y - y_mean))
        elif lag > 0:
            numerator = np.sum((y[:-lag] - y_mean) * (x[lag:] - x_mean))
        else:
            numerator = np.sum((y[-lag:] - y_mean) * (x[:lag] - x_mean))
        correlation = numerator / denominator
        result.append(correlation)
    return np.array(result)

def calc_rvec_crosscorr2(rvec1, rvec2, bin_sz, lag_range=None):
    if lag_range is None:
        return calc_rvec_crosscorr(rvec1, rvec2, bin_sz)
    else:
        lag_range_ = [int(lag / bin_sz) for lag in lag_range]
        cc = correlate2(rvec1, rvec2, lag_range_) / len(rvec1)
        lags = np.linspace(lag_range[0], lag_range[1], len(cc))
        return cc, lags


def _jitter_frate_vec(rvec, nbins_jit):
    rmat = rvec.reshape((nbins_jit,-1), order='F')
    np.random.shuffle(rmat)
    
def calc_rvec_crosscorr_nojit(rvec1, rvec2, bin_sz, nbins_jit, niter_jit=100,
                              lag_range=None):
    """Cross-correlogram of two firing rate signals, excluding jittered cov. """

    N = int(len(rvec1) / nbins_jit) * nbins_jit
    rvec1 = rvec1[:N]
    rvec2 = rvec2[:N]

    # Non-jittered covariance
    #c = np.correlate(rvec1, rvec2, mode='full')
    c, lags = calc_rvec_crosscorr2(rvec1, rvec2, bin_sz, lag_range)

    rvec1_jit = rvec1.copy()
    rvec2_jit = rvec2.copy()

    C_jit = np.nan * np.ones((niter_jit, len(c)))

    # Jittered covariances
    for n in range(niter_jit):
        _jitter_frate_vec(rvec1_jit, nbins_jit)
        _jitter_frate_vec(rvec2_jit, nbins_jit)
        #C_jit[n,:] = np.correlate(rvec1_jit, rvec2_jit, mode='full')
        C_jit[n,:], _ = calc_rvec_crosscorr2(
                rvec1_jit, rvec2_jit, bin_sz, lag_range)

    # Mean jittered cov, cov without jittered part
    c_jit = C_jit.mean(axis=0)
    c_nojit = c - c_jit
    
    return c_nojit, lags


def smooth_data(x, t=None, win_len=0, need_trim=False):
    """Smooth a signal. """
    
    if win_len==0:
        return t.copy(), x.copy()
    
    # Kernel
    w = np.ones(win_len,'d')
    w = w / w.sum()

    if need_trim:
        x1 = np.convolve(w, x, mode='valid')
    else:
        x1 = np.convolve(w, x, mode='same')
    
    if t is None:
        t1 = range(len(x1))    
    elif need_trim:
        t1 = t[win_len-1:]
    else:
        t1 = t.copy()
        
    return x1, t1

def resample(y, x, xnew):
    xnew = xnew[(xnew > np.min(x)) & (xnew < np.max(x))]
    f = sc.interpolate.interp1d(x, y, kind='cubic')
    return xnew, f(xnew)

def smooth_data_2(y, s, x, xnew=None):
    if xnew is not None:
        x, y = resample(y, x, xnew)
    dx = x[1] - x[0]
    T = s * 5
    Kx = np.arange(-T, T, dx)
    K = sc.stats.norm.pdf(Kx, scale=s)
    return x, np.convolve(y, K, mode='same') * dx

# =============================================================================
# # Calculate firing rate vector
# def calc_rvec(pop_name, bin_sz, t_range):
#     plt.ioff()
#     fig, sph = sim.analysis.plotSpikeHist(include=[pop_name], binSize=bin_sz,
#                                           timeRange=t_range, showFig=False)
#     plt.close(fig);
#     plt.ion()
#     rvec = sph['histoData'][0]
#     tvec = sph['histoT']
#     return rvec, tvec
# =============================================================================


def get_net_params(sim_result):
    return sim_result['net']['params']

def get_pop_params(sim_result, pop_name=None):
    if pop_name is None:
        return get_net_params(sim_result)['popParams']
    else:
        return get_net_params(sim_result)['popParams'][pop_name]

def get_pop_cell_gids(sim_result, pop_name):
    return sim_result['net']['pops'][pop_name]['cellGids']

def get_sim_data(sim_result):
    return sim_result['simData']

def get_pop_size(sim_result, pop_name):
    return len(get_pop_cell_gids(sim_result, pop_name))

def get_sim_duration(sim_result):
    return sim_result['simConfig']['duration']


def get_netpyne_pop_spikes(sim_result, pop_name, combine_cells=True):
    """Get times of spikes generated by a given population. """
    sim_data = get_sim_data(sim_result)
    pop_cell_idx = get_pop_cell_gids(sim_result, pop_name)
    spike_cell_idx = np.array(sim_data['spkid'])
    spkt = np.array(sim_data['spkt'])
    if combine_cells:
        #pop_spike_mask = np.array([cell_id in pop_cell_idx
        #                           for cell_id in sim_data['spkid']])
        pop_spike_mask = np.isin(sim_data['spkid'], pop_cell_idx)
        spike_times = spkt[pop_spike_mask] / 1000
    else:
        spike_times = []
        for cell_id in pop_cell_idx:
            mask = (spike_cell_idx == cell_id)
            spike_times.append(spkt[mask] / 1000)
    return spike_times


def calc_firing_rate_dynamics(spike_times, time_range, dt, Ncells=1,
                              epoch_len=None):
    """Calculate firing rate dynamics from combined spiketrains. """
    t1 = time_range[0]
    t2 = time_range[1]
    # Decrease the time range so it is a multiple of the epoch
    if epoch_len is not None:
        num_epochs = np.floor((time_range[1] - time_range[0]) / epoch_len)
        t2 = t1 + epoch_len * num_epochs
    else:
        num_epochs = 1
    # Get spike times within the given time range
    mask = (spike_times >= t1) & (spike_times <= t2)
    spike_times = spike_times[mask]
    # Put all the spikes into a single epoch
    if epoch_len is not None:
        spike_times = ((spike_times - t1) % epoch_len) + t1
        t2 = t1 + epoch_len
    # Transform: spike time -> sample number
    Nbins = int((t2 - t1) / dt)
    #spike_times = np.sort(spike_times)
    bin_idx = np.floor((spike_times - t1) / dt)
    bin_idx = bin_idx[(bin_idx >= 0) & (bin_idx < Nbins)]
    bin_idx = bin_idx.astype(np.int64)
    # Calculate firing rate dynamics
    rvec = np.bincount(bin_idx, minlength=Nbins)
    rvec = rvec / (dt * Ncells * num_epochs)
    # Time samples
    tvec = np.arange(Nbins, dtype=np.float64) * dt
    # Return the result
    return tvec, rvec


def calc_osc_amp(x, t, f):
    """Calculate complex amplitude at the frequency f. """
    T = t[-1] - t[0]
    dt = t[1] - t[0]
    u = np.cos(2 * np.pi * f * t)
    v = np.sin(2 * np.pi * f * t)
    #c_re = 2 * np.sum(u * x) * dt / T
    #c_im = -2 * np.sum(v * x) * dt / T
    c_re = 2 * np.sum(v * x) * dt / T
    c_im = 2 * np.sum(u * x) * dt / T
    c = c_re + 1j * c_im
    return c


def calc_osc_amp_from_spiketrain(spike_times, time_range, f, Ncells=1,
                                 taper=False):
    """Calculate complex amplitude at the frequency f from spiketrains. """
    # Decrease the time range so it is a multiple of the period
    T0 = 1 / f
    num_periods = np.floor((time_range[1] - time_range[0]) / T0)
    T = T0 * num_periods
    t1, t2 = (time_range[0], time_range[0] + T)
    # Get spike times within the given time range
    mask = (spike_times >= t1) & (spike_times <= t2)
    spike_times = spike_times[mask]
    spike_times -= t1
    # Calculate the coefficient
    #c_re = np.sum(np.cos(2 * np.pi * f * spike_times)) * 2 / (T * Ncells)
    #c_im = -np.sum(np.sin(2 * np.pi * f * spike_times)) * 2 / (T * Ncells)
    c_re_vec = np.sin(2 * np.pi * f * spike_times)
    c_im_vec = np.cos(2 * np.pi * f * spike_times)
    if taper:
        w_vec = 0.5 - 0.5 * np.cos(2 * np.pi * spike_times / T)
        c_re_vec *= w_vec
        c_im_vec *= w_vec
    c_re = np.sum(c_re_vec) * 2 / (T * Ncells)
    c_im = np.sum(c_im_vec) * 2 / (T * Ncells)
    c = c_re + 1j * c_im
    return c


def generate_sinusoid(A, f, tvec):
    return np.abs(A) * np.sin(2 * np.pi * f * tvec + np.angle(A))
    #return np.abs(A) * np.cos(2 * np.pi * f * tvec + np.angle(A))


def filter_signal(x, t, fband, order=3):
    fs = 1. / (t[1] - t[0])
    sos = sig.butter(order, fband, 'bandpass', output='sos', fs=fs)
    y = sig.sosfiltfilt(sos, x)
    return y

def remove_keys_except(d: dict, key_names: list):
    dsel = {key: d[key] for key in key_names}
    d.clear()
    d.update(dsel)
    
   
def deepcopy2(x):
    if isinstance(x, dict):
        return {key: deepcopy2(val) for key, val in x.items()}
    elif isinstance(x, list):
        return [deepcopy2(val) for val in x]
    elif hasattr(x, '__dict__'):
        y = copy.copy(x)
        for attr_name in x.__dict__:
            attr_val = getattr(x, attr_name)
            attr_val = deepcopy2(attr_val)
            setattr(y, attr_name, attr_val)
        return y
    else:
        return x


class Timer:
    def __init__(self):
        self.t0 = 0
        self.name = 'Timer'
    def start(self, name='Timer'):
        self.name = name
        self.t0 = time.time()
    def stop(self):
        dt = time.time() - self.t0
        print(f'{self.name}: dt = {dt}')
        

from collections.abc import Mapping, Container
from sys import getsizeof
 
def deep_getsizeof(o, ids=None):
    """Find the memory footprint of a Python object
 
    This is a recursive function that drills down a Python object graph
    like a dictionary holding nested dictionaries with lists of lists
    and tuples and sets.
 
    The sys.getsizeof function does a shallow size of only. It counts each
    object inside a container as pointer only regardless of how big it
    really is.
 
    :param o: the object
    :param ids:
    :return:
    """
    if ids is None:
        ids = set()
    
    d = deep_getsizeof
    if id(o) in ids:
        return 0
 
    r = getsizeof(o)
    ids.add(id(o))
 
    if isinstance(o, str):    # or isinstance(0, unicode):
        return r
 
    if isinstance(o, Mapping):
        return r + sum(d(k, ids) + d(v, ids) for k, v in o.iteritems())
 
    if isinstance(o, Container):
        return r + sum(d(x, ids) for x in o)
 
    return r 


def load_json_or_pkl(fpath):
    _, ext = os.path.splitext(fpath)
    if ext == '.json':
        with open(fpath, 'r', encoding='utf-8') as fid:
            res = json.load(fid)
    else:
        with open(fpath, 'rb') as fid:
            res = pickle.load(fid)
    return res


def fstr(template):
    return eval(f"f'{template}'")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


def fft_smart(x, t, fmax=None):
    f = np.fft.fftfreq(len(x), t[1] - t[0])
    y = np.fft.fft(x)
    mask = (f >= 0)
    if fmax:
        mask &= (f < fmax)
    return f[mask], y[mask]    
    
def make_closest_angle_seq(phi, mod=np.pi, step=1):
    #y = phi % mod
    #y = phi % (2 * np.pi)
    y = phi.copy()
    for n in range(step, len(y)):
        d = y[n] - y[n - step]
        #d = np.min([np.abs(d), mod - np.abs(d)]) * np.sign(d)
        d = d % mod
        if np.abs(d - mod) < np.abs(d):
            d -= mod
        y[n] = y[n - step] + d
    return y
     
  
def minmax(x):
    return (np.min(x), np.max(x))