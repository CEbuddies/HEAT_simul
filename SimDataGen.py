# -*- coding: utf-8 -*-
"""
Created on Wen Mar  17 18:41:44 2021

@author: nic
"""
"""
Data creation class for the simulations 
Specify boundary conditions and so on
"""

from SimulSetup import Simulation
import pickle
import numpy as np
import argparse

class DataGen():

    def __init__(self,el_side,num_samples,bc_dict,out_name,cuda):
        self.el_side = el_side
        self.samples = num_samples
        self.bc_dict_path = bc_dict
        self.cuda = cuda
        self.out_name = out_name

        # data dict
        self.data_dict = {}
        self.data_dict['features'] = []
        self.data_dict['targets'] = []

    def read_boundaries(self):
        """ process boundary conditions """
        with open(self.bc_dict_path,'r') as pl:
            bc_dict = pickle.load(pl)
        return bc_dict

    def create_boundaries(self):
        pass

    def run(self):
        """ runs the simulations with given specifications """
        if self.bc_dict_path is not None:
            bc = self.read_boundaries()
        else:
            bc = self.create_boundaries()

        for i in range(self.samples):
            print(f'starting simulations run with {self.samples+1} samples')
            print(f'starting simulation {i+1} ...')
            sim = Simulation((self.el_side,self.el_side),bc,cuda=self.cuda)
            simresults = sim.simulation_run_for(0.01,100)
            k_mat = sim.k_mat.get()
            nonzero_k_mat = self.nonzeros(k_mat)

            # has to be copied for freeing the memory
            feature = simresults['temperature'][0].copy()
            target = simresults['temperature'][-1].copy()
            self.data_dict['features'].append((feature,nonzero_k_mat))
            self.data_dict['targets'].append(target)

            print(f'finished simulation {i+1}')

        with open(self.out_name,'wb') as pd:
            pickle.dump(self.data_dict,pd)

    def nonzeros(self,k_mat):

        # access the nonzero elements of k_mattrix als further input
        mat_indcs = np.array(np.where(k_mat > 0.)).T
        # trying to copy everywhere to get rid of the references
        nonzero = k_mat[mat_indcs[:,0],mat_indcs[:,1]].copy()

        return nonzero

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 'Args for Data creation')
    parser.add_argument('-s',
        '--sample',default=10,type=int,help='num of simulated samples')
    parser.add_argument('-n',
        '--num',default=40,type=int,help='number parts')
    parser.add_argument('-d',
        '--datname',default='train_data_val',help='name of the data')
    parser.add_argument('-b',
        '--bcpath',default='linear',help='path of bc dict')
    parser.add_argument('-c',
        '--cuda',default=True,help='cuda support')
    args = parser.parse_args()

    ## build the inputs 
    el_side = args.num
    num_samples = args.sample
    datname = args.datname + '.sml'
    bc_path = args.bcpath + '.bcd'
    cuda = args.cuda
    ## 

    dg = DataGen(el_side,num_samples,bc_path,data_name,cuda)
    dg.run()

