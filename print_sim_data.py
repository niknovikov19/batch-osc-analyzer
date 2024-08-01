import numpy as np
import pickle as pkl


#fpath = r"D:\WORK\Salvador\repo\A1_model_old\data\grid_batch_v34_batch56_2\grid_00004_data.pkl"
fpath = r"D:\WORK\Salvador\repo\A1_model_old\data\A1_paper\v34_batch56_10s_data.pkl"

with open(fpath, 'rb') as fid:
    sim_res = pkl.load(fid)

pops = sim_res['net']['pops']

print('Num. cells')
for pop_name, pop in pops.items():
    n = pop['numCells']
    print(f'{pop_name}: {n}')
    
print('Y range')
for pop_name, pop in pops.items():
    if 'IT' in pop_name:
        yy = pop['ynormRange']
        print(f'{pop_name}: {yy[0]} - {yy[1]}')
        
pop = pops['IT3']
gids = pop['cellGids']
cells = [cell for cell in sim_res['net']['cells'] if cell['gid'] in gids]
yy = np.array([cell['tags']['y'] for cell in cells])
yy_norm = np.array([cell['tags']['ynorm'] for cell in cells])
