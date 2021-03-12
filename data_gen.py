# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 18:41:44 2021

@author: nic
"""
from SimulSetup import Simulation
import pickle
import numpy as np

# Script for the data generation 

def data_gen():
    
    el_side = 40
    x = np.linspace(0,1,el_side).reshape(el_side,1)
    locs = ['left','right','top','bottom']
    bc_left = x**2
    bc_right = x**2
    bc_right = bc_right[::-1]
    bc_top = x**2
    bc_bottom = x**2
    bc_bottom = bc_bottom[::-1]
    
    bc_vals = [bc_left,bc_right,bc_top,bc_bottom]
    bc_dict = {'locations':locs,'values':bc_vals}
    
    sim = Simulation((el_side,el_side),bc_dict,cuda=True)
    
    simresults = sim.simulation_run_for(0.01,100)
    
    # access the nonzero elements of k_mattrix als further input
    k_mat = sim.k_mat.get()
    mat_indcs = np.array(np.where(k_mat > 0.)).T
    # trying to copy everywhere to get rid of the references
    nonzero = k_mat[mat_indcs[:,0],mat_indcs[:,1]].copy()
    
    return simresults, nonzero
    
if __name__ == '__main__':
    data_dict = {}
    data_dict['features'] = []
    data_dict['targets'] = []
    
    for i in range(10):
        
        print(f'starting simulation {i}...')
        res, nonzero_k_mat = data_gen()
        feature = res['temperature'][0].copy()
        target = res['temperature'][-1].copy()
        data_dict['features'].append((feature,nonzero_k_mat))
        data_dict['targets'].append(target)
        print(f'finished simulation {i}')
    
        
    # write all the data
    with open('training_data_big.sml','wb') as pd:
        pickle.dump(data_dict,pd)
