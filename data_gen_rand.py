# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 11:49:30 2021

@author: nwilhelm
"""

from SimulSetup import Simulation
import pickle
import numpy as np
import argparse

# Script for the data generation 

def data_gen(el_side):
    
    
    locs = ['left','right','top','bottom']
    bc_left = np.random.rand(el_side,1)
    bc_right = np.random.rand(el_side,1)
    
    bc_top = np.random.rand(el_side,1)
    bc_bottom = np.random.rand(el_side,1)
    
    
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
    parser = argparse.ArgumentParser(description = 'Args for Data creation')
    parser.add_argument('-s',
        '--sample',default=10,type=int,help='num of simulated samples')
    parser.add_argument('-n',
        '--num',default=40,type=int,help='number parts')
    parser.add_argument('-d',
        '--datname',default='train_data_val',help='name of the data')
    args = parser.parse_args()

    data_dict = {}
    data_dict['features'] = []
    data_dict['targets'] = []
    samples = args.sample
    ptnum = args.num
    dname = args.datname + '.sml'
    
    for i in range(samples):
        
        print(f'starting simulation {i}...')
        res, nonzero_k_mat = data_gen()
        feature = res['temperature'][0].copy()
        target = res['temperature'][-1].copy()
        data_dict['features'].append((feature,nonzero_k_mat))
        data_dict['targets'].append(target)
        print(f'finished simulation {i}')
    
        
    # write all the data
    with open(dname,'wb') as pd:
        pickle.dump(data_dict,pd)
