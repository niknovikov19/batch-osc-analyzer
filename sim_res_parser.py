import numpy as np

import common as cmn


def get_pop_names(sim_result):
    #return list(cmn.get_pop_params(sim_result).values())
    return list(sim_result['net']['pops'].values())

def get_lfp_coords(sim_result):
    cfg = sim_result['simConfig']
    lfp_coords = np.array(cfg['recordLFP'])
    return lfp_coords

def get_record_times(sim_result):
    cfg = sim_result['simConfig']
    dt = cfg['recordStep']
    T = cfg['duration']
    tt = np.arange(0, T, dt)
    return tt

def get_lfp(sim_result):
    lfp = np.array(sim_result['simData']['LFP']).T
    lfp_coords = get_lfp_coords(sim_result)
    tt = get_record_times(sim_result)
    return lfp, tt, lfp_coords

def get_pop_lfps(sim_result):
    lfp = {}
    for pop_name, pop_lfp in sim_result['simData']['LFPPops'].items():
        lfp[pop_name] = np.array(pop_lfp).T
    lfp_coords = get_lfp_coords(sim_result)
    tt = get_record_times(sim_result)
    return lfp, tt, lfp_coords

def get_pop_ylim(sim_res, pop_name):
    h = sim_res['simConfig']['sizeY']
    yy = sim_res['net']['pops'][pop_name]['tags']['ynormRange']
    return (yy[0] * h, yy[1] * h)

def get_layer_borders(sim_result):
    layer_yrange = {}
    for pop_name in sim_result['net']['pops']:
        layer_yrange[pop_name] = get_pop_ylim(sim_result, pop_name)
    return layer_yrange

    