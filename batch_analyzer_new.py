#import importlib
import json
import glob
#import logging
import os
from pathlib import Path
import pickle as pkl
import re

import dask.array as da
#import matplotlib.pyplot as plt
#from netpyne import specs, sim
import numpy as np
import pandas as pd
import scipy as sc
#import scipy.signal as sig
import xarray as xr

import common as cmn
import sim_res_parser as srp


def _get_key_seq(x, seq):
    """Get x[seq[0]][seq[1]][...] """
    for key in seq:
        x = x[key]
    return x


class BatchAnalyzer:
    
    def __init__(self, dirpath_data, par_names):
        self.dirpath_data = Path(dirpath_data)
        self.par_names = par_names
        self.job_idx = None
        self._read_param_values()
        
    def _read_param_values(self):
        """Read parameter value combinations from batch job configs. """
        
        # List of param files
        fname_templ = self.dirpath_data / 'grid_?????_params.json'
        job_templ = re.compile(r'grid_(\d{5})_params\.json')
        fnames = glob.glob(str(fname_templ))
        
        # Read the files and extract parameter values
        data = []
        for fname in fnames:
            fname_ = os.path.basename(fname)
            job_id = int(job_templ.match(fname_).group(1))
            with open(fname, 'r') as f:
                cfg = json.load(f)['simConfig']
                entry = {par: cfg[par] for par in self.par_names}
                entry['job_id'] = job_id
                data.append(entry)
        
        # Create xarray of job indices, with param values as coordinates
        df = pd.DataFrame(data)
        pivot_df = df.pivot_table(
            index=self.par_names, values='job_id')
        self.job_idx = pivot_df.to_xarray()
        
# =============================================================================
#     def _get_job_id(self, par_vals: dict):
#         """Get job index by parameter values. """
#         return 0
# =============================================================================
        
    def load_job_sim_data(self, job_id):
        """Load simulation result for a single param combination. """
        fpath_sim = self.dirpath_data / f'grid_{job_id:05d}_data.pkl'
        with open(fpath_sim, 'rb') as fid:
            sim_result = pkl.load(fid)
        return sim_result
    
# =============================================================================
#     def get_slice_idx(self, par_vals_slice):
#         """
#         Find and sort indexes of par_vals_lst for which par_vals_slice
#         is a subset of the dictionary from the corresponding par_vals_lst entry,
#         sorted by the values of the remaining parameter par_name_free.
#         """
#         # Determine the missing key (par_name_free)
#         par_name_free = next(par_name for par_name in self.par_names
#                              if par_name not in par_vals_slice)
# 
#         # Iterate over par_vals_lst and find matches to par_vals_slice
#         matches = []
#         for idx, entry in enumerate(self.par_vals_lst):
#             if all(entry.get(k) == v for k, v in par_vals_slice.items()):
#                 matches.append((idx, entry[par_name_free]))
#     
#         # Sort matches based on the par_name_free values and return the result
#         matches.sort(key=lambda x: x[1])
#         job_idx, par_vals_free = zip(*matches) if matches else ([], [])    
#         return list(job_idx), list(par_vals_free)
# =============================================================================
    
# =============================================================================
#     def get_data_slice(self, x: list, field_seq: list, par_vals_slice: dict):
#         job_idx, par_vals_free = self.get_slice_idx(par_vals_slice)
#         res = [_get_key_seq(x[job_id], field_seq) for job_id in job_idx]
#         return res, par_vals_free
# =============================================================================

    def _calc_job_rate_data(self, sim_result, rbin_sz, time_range=(0, None), 
                            pops_incl=None, pops_excl=None):
        """Rate dynamics and spike times for each population. """
        if time_range[1] is None:
            time_range = (
                time_range[0], cmn.get_sim_duration(sim_result) / 1000)
        job_rate_data = {}
        for pop_name in srp.get_pop_names(sim_result):
            if (pops_incl is not None) and (pop_name not in pops_incl):
                continue
            if (pops_excl is not None) and (pop_name in pops_excl):
                continue
            print(f'Firing rate: {pop_name}')
            spike_times = cmn.get_netpyne_pop_spikes(sim_result, pop_name)
            cell_count = cmn.get_pop_size(sim_result, pop_name)
            tvec, rvec = cmn.calc_firing_rate_dynamics(
                    spike_times, time_range=time_range, dt=rbin_sz,
                    Ncells=cell_count)
            job_rate_data[pop_name] = {
                'cell_count': cell_count,
                'spike_times': spike_times,
                'tvec': tvec,
                'rvec': rvec,
                'r': np.mean(rvec)
            }
        return job_rate_data
    
    def _gen_rate_data_path(self, rbin_sz, time_range=(0, None)):
        tstr = str(time_range).replace(', ', '_')
        fname = f'rate_data_(rbin={rbin_sz}_t={tstr}).nc'
        return self.dirpath_data / fname
    
    def calc_rate_data(self, rbin_sz, time_range=(0, None), 
                       pops_incl=None, pops_excl=None, need_recalc=False):
        """Calculate/load firing rate dynamics for every job. """
        
        # Load rate data
        fpath_rate_data = self._gen_rate_data_path(rbin_sz, time_range)
        if os.path.exists(fpath_rate_data) and not need_recalc:
            with open(fpath_rate_data, 'rb') as fid:
                rate_data = xr.open_dataset(fpath_rate_data, chunks={})
            return rate_data
            
        # Get pop. names and rate time bins from the 0-th parameter set results
        sim_result = self.load_job_sim_data(job_id=0)
        pop_names = srp.get_pop_names(sim_result)
        job_rate_data = self._calc_job_rate_data(
            sim_result, rbin_sz, time_range, pops_incl=pop_names[0])
        tvec = job_rate_data['tvec']
        
        # Allocate rate arrays
        npop, nt, npar = len(pop_names), len(tvec), len(self.par_names)
        sz0 = self.job_idx.shape
        ch0 = [-1] * npar
        d0, c0 = self.par_names, self.job_idx.coords
        r_ = da.full((*sz0, npop), np.nan, chunks=(ch0 + [1]))
        rvec_ = da.full((*sz0, nt, npop), np.nan, chunks=(ch0 + [-1, 1]))
        r = xr.DataArray(r_, coords={**c0, 'pop': pop_names},
                         dims=(*d0, 'pop'))
        rvec = xr.DataArray(rvec_, coords={**c0, 'time': tvec, 'pop': pop_names},
                            dims=(*d0, 'time', 'pop'))
        
        if os.path.exists(fpath_rate_data) and not need_recalc:
            with open(fpath_rate_data, 'rb') as fid:
                rate_data = pkl.load(fid)
        else:
            rate_data = []
            for job_id, job_par in enumerate(self.par_vals_lst):
                print(f'==== Job: {job_id} ====')
                sim_result = self.load_job_sim_data(job_id)
                job_rate_data = self._calc_job_rate_data(
                    sim_result, rbin_sz, time_range, pops_incl, pops_excl)
                rate_data.append({'data': job_rate_data, 'params': job_par})
            with open(fpath_rate_data, 'wb') as fid:
                pkl.dump(rate_data, fid)
                
        return rate_data
    
    
    def _is_pop_pair_used(self, pop1_name, pop2_name, ref=None):
        if ref is None:
            return True
        elif ref == 'self':
            return (pop1_name == pop2_name)
        elif ref == 'osc':
            return (pop2_name[:3] == 'osc')
        else:
            return pop2_name == ref
    
    def _calc_job_spect_data(self, job_rate_data, corr_taper_width,
                             fmax, ref=None):
        """Correlations and cross-specra of the firing rate signals. """
        job_spect_data = {}
        for pop1_name, pop1 in job_rate_data.items():
            for pop2_name, pop2 in job_rate_data.items():
                if not self._is_pop_pair_used(pop1_name, pop2_name, ref):
                    continue
                print(f'Correlation: {pop1_name} - {pop2_name}')
                rvec1 = pop1['rvec'] - pop1['rvec'] .mean()
                rvec2 = pop2['rvec'] - pop2['rvec'] .mean()
                dt = pop1['tvec'][1] - pop1['tvec'][0]
                cc, lags = cmn.calc_rvec_crosscorr(rvec1, rvec2, dt)
                H = sc.stats.norm.pdf(lags, scale=corr_taper_width)
                ff, psd = cmn.fft_smart(cc, lags, fmax=fmax)
                _, psd_c = cmn.fft_smart(cc * H, lags, fmax=fmax)
                job_spect_data[(pop1_name, pop2_name)] = {
                        'corr': cc, 'corr_c': cc * H, 'lags': lags,
                        'psd': psd, 'psd_c': psd_c, 'lags': lags, 'ff': ff}
        return job_spect_data
    
    def _gen_spect_data_path(self, rbin_sz, time_range=(0, None),
                             corr_taper_width=0.05, ref=None):
        tstr = str(time_range).replace(', ', '_')
        fname = f'spect_data_(rbin={rbin_sz}_t={tstr}_ctaper={corr_taper_width}_ref={ref}).pkl'
        return self.dirpath_data / fname
    
    def calc_spect_data(self, rbin_sz, time_range=(0, None), 
                       pops_incl=None, pops_excl=None, 
                       corr_taper_width=0.05, fmax=150, ref=None,
                       need_recalc=False):
        """Calculate/load correlations and cross-specra for every job. """
        
        fpath_spect_data = self._gen_spect_data_path(
            rbin_sz, time_range, corr_taper_width, ref)
        if os.path.exists(fpath_spect_data) and not need_recalc:
            with open(fpath_spect_data, 'rb') as fid:
                spect_data = pkl.load(fid)
        else:
            spect_data = []
            rate_data = self.calc_rate_data(
                rbin_sz, time_range, pops_incl, pops_excl)
            for job_id, job_par in enumerate(self.par_vals_lst):
                print(f'==== Job: {job_id} ====')
                job_rate_data = rate_data[job_id]['data']
                job_spect_data = self._calc_job_spect_data(
                    job_rate_data, corr_taper_width, fmax, ref)
                spect_data.append({'data': job_spect_data, 'params': job_par})
            with open(fpath_spect_data, 'wb') as fid:
                pkl.dump(spect_data, fid)
                
        return spect_data
    

    

