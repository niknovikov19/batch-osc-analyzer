import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from batch_analyzer import BatchAnalyzerOsc
import common as cmn


#dirname_in = 'grid_batch_v34_batch56_tcebkg_4'
#par_names = ['TC_ebkg_mult']

dirname_in = 'grid_batch_v34_batch56_trnibkg_2'
par_names = ['IRE_ibkg_mult']

dirpath_data = rf'D:\WORK\Salvador\repo\A1_model_old\data\{dirname_in}'
ba = BatchAnalyzerOsc(dirpath_data, par_names)

time_range = (1, None)
rbin_sz = 0.001
pops_incl = None

rate_data = ba.calc_rate_data(
    rbin_sz=rbin_sz, time_range=time_range, pops_incl=pops_incl,
    need_recalc=False)

pop_names = list(rate_data[0]['data'].keys())
pop_rates = {}
for pop_name in pop_names:
    pop_rates[pop_name], _ = ba.get_data_slice(
        rate_data, ['data', pop_name, 'r'], {})
_, par_vals = ba.get_data_slice(rate_data, ['data', pop_names[0], 'r'], {})

pop_groups = [
    ['NGF1', 'IT2',  'SOM2',  'PV2',  'VIP2',  'NGF2'],
    ['IT3',  '',     'SOM3',  'PV3',  'VIP3',  'NGF3'],
    ['ITP4', 'ITS4', 'SOM4',  'PV4',  'VIP4',  'NGF4'],
    ['IT5A', 'CT5A', 'SOM5A', 'PV5A', 'VIP5A', 'NGF5A'],
    ['IT5B', 'CT5B', 'SOM5B', 'PV5B', 'VIP5B', 'NGF5B', 'PT5B'],
    ['IT6',  'CT6',  'SOM6',  'PV6',  'VIP6',  'NGF6'],
    ['TC',   'TCM', ' HTC',   'IRE',  'IREM',  'TI',    'TIM']
    ]

colors = cmn.get_default_colors()

nx = 4
ny = 2
plt.figure()
group_id = 0
for ny_ in range(ny):
    for nx_ in range(nx):
        plt.subplot(ny, nx, ny_ * nx + nx_ + 1)
        if group_id >= len(pop_groups):
            break
        pop_group = pop_groups[group_id]
        group_id += 1
        for m, pop_name in enumerate(pop_group):
            if pop_name not in pop_rates:
                continue
            rr = pop_rates[pop_name]
            plt.plot(par_vals, rr, '.-', color=colors[m], label=pop_name)
        plt.legend()
        #plt.ylim(0.01, 300)
        plt.yscale('log', nonpositive='mask')
plt.subplot(ny, nx, (ny - 1) * nx + 1)
plt.xlabel(par_names[0])
plt.ylabel('Firing rate')

#df = pd.DataFrame(pop_rates, index=ebkg_vals).T
