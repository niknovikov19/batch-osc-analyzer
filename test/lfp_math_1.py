import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter


# Function to load and preprocess the CSV file
def load_csv(file_path):
    df = pd.read_csv(file_path, sep=';', decimal=',', header=None)
    df.columns = ['x', 'y']
    return df

# Function to re-interpolate data
def interpolate_data(dfs, new_y):
    interpolated_data = {}
    for name, df in dfs.items():
        df_ = df.sort_values(by='y')
        f = np.interp(new_y, df_['y'], df_['x'])
        interpolated_data[name] = f
    return interpolated_data

# Load CSV files
dirpath_in = r'D:\WORK\Salvador\A1_project\lfp_csd_plots'
data_names = ['lfp0_exp_high', 'lfp0_exp_low',
              'bipolar_exp_high', 'bipolar_exp_low',
              'csd_exp_high', 'csd_exp_low']
dfs0 = {}
for name in data_names:
    fpath = Path(dirpath_in) / f'{name}.csv'
    dfs0[name] = load_csv(fpath)
    
# Determine the common x-coordinates for interpolation
min_y = max([df['y'].min() for df in dfs0.values()])
max_y = min([df['y'].max() for df in dfs0.values()])
yy = np.linspace(min_y, max_y, num=1000)  # Adjust num as needed
#yy = np.linspace(0, 1, 200)

# Re-interpolate all dataframes
dfs = interpolate_data(dfs0, yy)

#plt.figure()
#plt.plot(dfs['lfp0_exp_low'], yy)
#plt.plot(dfs['lfp0_exp_high'], yy)

P = {}
# =============================================================================
# P['low'] = {'lfp': dfs['lfp0_exp_low'],
#             'bip': dfs['bipolar_exp_low'],
#             'csd': dfs['csd_exp_low']}
# P['high'] = {'lfp': dfs['lfp0_exp_high'],
#             'bip': dfs['bipolar_exp_high'],
#             'csd': dfs['csd_exp_high']}
# =============================================================================
P['low'] = {'lfp': dfs['bipolar_exp_low'][::-1],
            'bip': dfs['csd_exp_low'][::-1],
            'csd': dfs['lfp0_exp_low']}
P['high'] = {'lfp': dfs['bipolar_exp_high'][::-1],
            'bip': dfs['csd_exp_high'][::-1],
            'csd': dfs['lfp0_exp_high']}

q, k = {}, {}
q['low'] = 1
q['high'] = 1
k['low'] = 1
k['high'] = 1
#dy = 1. / (1e2 * 2.3)
dy = 1. / (1e2 * 2)

yy = yy[:-2]

plt.figure()

for fband in ['low', 'high']:
    
    r2_lfp = P[fband]['lfp'] * q[fband]    # |z|^2
    r_lfp = np.sqrt(r2_lfp)                  # |z|
    dr_lfp = np.pad(np.diff(r_lfp), (0, 1)) / dy  # d|z|/dy
    #dr_lfp = np.diff(r_lfp) / dy
    
    r2_bip = P[fband]['bip'] * k[fband]  # |dz/dy|^2
    #r2_bip = r2_bip[::-1]
    
    dphi2_lfp = (r2_bip - dr_lfp**2) / r2_lfp         # (d_arg(z)/dy)^2
    dphi_lfp = np.sqrt(dphi2_lfp)
    phi_lfp = np.cumsum(dphi_lfp) * dy
    
    #d2r_lfp = np.pad(np.diff(dr_lfp), (0, 1)) / dy
    #d2phi_lfp = np.pad(np.diff(dphi_lfp), (0, 1)) / dy
    d2r_lfp = np.pad(np.diff(r_lfp, 2), (1, 1)) / dy**2
    d2phi_lfp = np.pad(np.diff(phi_lfp, 2), (1, 1)) / dy**2
    
    s = 50
    d2r_lfp = median_filter(d2r_lfp, s)
    d2phi_lfp = median_filter(d2phi_lfp, s)
    
    r2_csd = (d2phi_lfp**2 * r2_lfp + 4 * d2phi_lfp * dphi_lfp * dr_lfp * r_lfp +
              d2r_lfp**2 - 2 * d2r_lfp * dphi_lfp**2 * r_lfp + 
              dphi_lfp**4 * r2_lfp + 4 * dphi_lfp**2 * dr_lfp**2)
    
    r2_lfp = r2_lfp[:-2]
    phi_lfp = phi_lfp[:-2]
    r2_bip = r2_bip[:-2]
    r2_csd = r2_csd[:-2]
    
    r2_csd /= max(r2_csd)

    nx = 4
    plt.subplot(1, nx, 1)
    plt.plot(r2_lfp, yy)
    plt.title('LFP power')
    plt.subplot(1, nx, 2)
    plt.plot(r2_bip, yy)
    plt.title('Bip. power')
    plt.subplot(1, nx, 3)
    plt.plot(dphi_lfp[:-2], yy)
    plt.title('dphi/dy')
    #plt.plot(r2_csd, yy)
    #plt.plot(d2r_lfp[:-2], yy)
    #plt.title('CSD power')
    plt.subplot(1, nx, 4)
    #plt.plot(dr, yy)
    #plt.title('dr/dy')
    plt.plot(phi_lfp, yy)
    plt.title('phi')

