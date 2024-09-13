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

import common as cmn
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


# Frequency bands
FBAND_LOW = (10, 30)
FBAND_HIGH = (50, 150)

LAYER_GROUPS = [(0, 950), (950, 1250), (1250, 2000)]


class LFPAnalyzer:
    
    def __init__(self, sim_result):
        self.sim_result = sim_result
        lfp, tt, lfp_coords = srp.get_lfp(sim_result)
        self.tvec = tt / 1000
        self.depth_vec = lfp_coords[:, 1]
        self.lfp = lfp.T
        
    def _preprocess_lfp(self, t_lim, ref_chan, pow_type):
        
        X = self.lfp
        tt = self.tvec

        # Select time range
        if t_lim is not None:
            t_idx = np.argwhere(((tt >= t_lim[0]) & (tt <= t_lim[1]))).ravel()
            tt = tt[t_idx]
            X = X[:, t_idx]

        # Re-reference
        if ref_chan is not None:
            X -= X[ref_chan, :]

        # Bipolar / CSD
        if pow_type == 'bipolar':
            X = np.concatenate([X[0, None, :], X], axis=0)
            X = np.diff(X, n=1, axis=0)
        if pow_type == 'csd':
            X = np.concatenate([X[0, None, :], X, X[-1, None, :]], axis=0)
            X = np.diff(X, n=2, axis=0)
          
        self.lfp_proc = X
        self.tvec_proc = tt
        
    def _calc_power(self, fmax, pow_norm, layer_groups, fbands):
        # Calculate spectral power
        nt = len(self.tvec_proc)
        h = sig.windows.hann(nt).reshape((1, nt))
        ff, W = cmn.fft_smart(self.lfp_proc * h, self.tvec_proc, fmax=fmax)
        W = np.abs(W)

        # Normalize each freq. power by its max. over channels
        if pow_norm == 'freq':
            W /= W.max(axis=0)
        
        self.freq_vec = ff
        self.W = W
        
        # Average power over depths within layer groups
        self.Wavg_d = {}
        for ylim in layer_groups:
            yy = self.depth_vec
            mask = (yy >= ylim[0]) & (yy <= ylim[1])
            self.Wavg_d[ylim] = W[mask, :].mean(axis=0)
            
        # Average power over frequencies within bands
        self.Wavg_f = {}
        for fband in fbands:
            w = W[:, (ff >= fband[0]) & (ff <= fband[1])].mean(axis=1)
            if pow_norm == 'band':
                w /= w.max()
            self.Wavg_f[fband] = w
            
    def analyze_lfp(self, t_lim=None, ref_chan=None, pow_type='lfp',
                    fmax=150, pow_norm='none', layer_groups=LAYER_GROUPS,
                    fbands=[FBAND_LOW, FBAND_HIGH]):
        self._preprocess_lfp(t_lim, ref_chan, pow_type)
        self._calc_power(fmax, pow_norm, layer_groups, fbands)
        res = {'tvec': self.tvec, 'lfp': self.lfp,
               'tvec_proc': self.tvec_proc, 'lfp_proc': self.lfp_proc,
               'freqs': self.freq_vec, 'depths': self.depth_vec,
               'W': self.W, 'Wavg_d': self.Wavg_d, 'Wavg_f': self.Wavg_f}
        return res

