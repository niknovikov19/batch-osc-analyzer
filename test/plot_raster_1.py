import collections
import pickle as pkl

from netpyne import sim, specs
from netpyne.analysis.utils import colorList
import numpy as np
from matplotlib import pyplot as plt
#import pandas as pd


#fpath_in = r"D:\WORK\Salvador\A1_project\zenodo\v35_batch9\v35_batch9\v35_batch9_0_0_data.pkl"
fpath_in = r'D:\WORK\Salvador\A1_project\zenodo\v34_batch56\v34_batch56_0_0_data.pkl'
#fpath_in = r'D:\WORK\Salvador\repo\A1_model_old\data\A1_paper\v34_batch56_10s_data.pkl'
#fpath_in = r"D:\WORK\Salvador\repo\A1_model_old\data\grid_batch_v34_batch56_tcebkg_4\grid_00002_data.pkl"

with open(fpath_in, 'rb') as fid:
    sim_res = pkl.load(fid)
    
sim.initialize()
sim.loadAll(fpath_in, instantiate=False)

pops = sim_res['net']['pops']
#incl = ['2', '3']
#incl = ['5', '6']
incl = ['6']
include = [pop for pop in pops if any(s in pop for s in incl)]

sim.analysis.plotRaster(orderInverse=True, include=include)
#plt.xlim(1500, 3000)

#sim.analysis.plotSpikeStats(include=include, stats=['rate'], timeRange=(1500, 3000))
#sim.analysis.plotSpikeStats(include=include, stats=['rate'])
sim.analysis.plotSpikeStats(stats=['rate'])
plt.xlim((0, 200))


# =============================================================================
# sim.plotting.plotRaster(
#     **{ 'include': ['allCells'], 
#         'saveFig': None, 
#         'showFig': False, 
#         'popRates': 'minimal', 
#         'orderInverse': False, 
#         'timeRange': (1.5, 3), 
#         'figSize': (3, 8), 
#         'lw': 0.3, 
#         'markerSize': 3, 
#         'marker': '.', 
#         'dpi': 300})
# =============================================================================
