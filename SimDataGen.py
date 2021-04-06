# -*- coding: utf-8 -*-
"""
Created on Wen Mar  17 18:41:44 2021

@author: nic
"""
"""
Data creation class for the simulations 
Specify boundary conditions and so on

type: train - training data
type: val - validation data

so far data is automaticall rerouted to train_data (split in Dataset)
"""
"""
TODO
metadata creation (maybe)

current version supports only squared domain 
please adjust in init and in dirs_ as well as in the run methods 
then remove this remark 
"""

from SimulSetup import Simulation
import pickle
import numpy as np
import argparse
import random 

def dirs_(el_num,k_line,els_side):
"""
return the connectivity tuple per element 
"""

	col = el_num % lines
	line = int(el_num/lines)
	k_ = k_line.reshape(els_side,els_side)
	try:
	    north = k_[line-1,col]
	except:
		north = 0
	try:
		west = k_[line,col-1]
	except:
		west = 0
	try:
		east = k_[line,col+1]
	except:
		east = 0
	try:
		south = k_[line+1,col]
	except:
		south = 0
	return (north,west,east,south

class DataGen():

    def __init__(self,el_side,num_samples,bc_dict,out_name,cuda):
        self.el_side = el_side
        self.samples = num_samples
        # arparser automatically defaults to random - random creates of different BCs
        if bc_dict != 'random':
            self.bc_dict_path = bc_dict + '.bcd'
        else:
            self.bc_dict_path = None
        self.cuda = cuda
        self.out_name = out_name
        self.bc_name = ['left','right','top','bottom']
        # orientation direction is switched from one to another side 
        self.dirs = [-1,1]

        # data dict - final data from which dataset is to be created
        self.data_dict = {}
        self.data_dict['features'] = []
        self.data_dict['targets'] = []

    def read_boundaries(self):
        """ process boundary conditions from file """
        with open(self.bc_dict_path,'rb') as pl:
            bc_dict = pickle.load(pl)
        return bc_dict

    def get_random_bc_dict(self):
        """ creating dirichlet BC dict for each simulation

        Returns: dict - BC dict
        """

        def lin(x):
            return x
        def quad(x):
            return (x-0.5)**2
        def doub_quad(x):
            return 2*(x-0.5)**2
        def sin(x):
            return np.abs(np.sin(x))
        def four_sin(x):
            return np.abs(np.sin(4*x))
        def const(x):
            # return a random but complete filled vector
            return np.full((len(x),),np.random.rand())

        self.bc_switch = {'lin':lin,'quad':quad,'doub_quad':doub_quad,
                    'sin':sin,'four_sin':four_sin,'const':const}

        bc_val = []
        for side in self.bc_name:
            vec = np.linspace(0,1,self.el_side)
            # draw a random function to create the side
            bc_type_name = random.choice(list(self.bc_switch.keys())) 
            bc_vals_side = self.bc_switch[bc_type_name](vec)

            # switch direction
            bc_vals_side = bc_vals_side[::random.choice(self.dirs)]
            bc_vals_side = bc_vals_side.reshape(self.el_side,1)
            bc_val.append(bc_vals_side)
        # compose the dict
        bc_dict = {}
        bc_dict['locations'] = self.bc_name
        bc_dict['values'] = bc_val
        return bc_dict

    def run(self):
        """ runs the simulations with given specifications """
        
        print(f'starting simulations run with {self.samples} samples')
        for i in range(self.samples):
            if self.bc_dict_path is not None:
                bc = self.read_boundaries()
            else:
                bc = self.get_random_bc_dict()
            print(f'starting simulation {i+1} ...')
            sim = Simulation((self.el_side,self.el_side)
                    ,bc,cuda=self.cuda)
            simresults = sim.simulation_run_for(0.01,100)
            k_mat = sim.k_mat.get()
            nonzero_k_mat = self.nonzeros(k_mat)

            # has to be copied for freeing the memory
            feature = simresults['temperature'][0].copy()
            target = simresults['temperature'][-1].copy()
            self.data_dict['features'].append((feature,nonzero_k_mat))
            self.data_dict['targets'].append(target)

            print(f'finished simulation {i+1}')

        print(f'Writing to target file ' + self.out_name)
        with open(self.out_name,'wb') as pd:
            pickle.dump(self.data_dict,pd)

    def run_spat(self):
        """ runs the simulations with given specifications """
        
        print(f'starting simulations run with {self.samples} samples')
        for i in range(self.samples):
            if self.bc_dict_path is not None:
                bc = self.read_boundaries()
            else:
                bc = self.get_random_bc_dict()
            print(f'starting simulation {i+1} ...')
            sim = Simulation((self.el_side,self.el_side)
                    ,bc,cuda=self.cuda)
            simresults = sim.simulation_run_for(0.01,100)
            k_mat = sim.k_mat.get()
            # process k_mat to a three dim 
            k3 = np.zeros((self.el_side*self.el_side,4))
            for el_idx,k_line in enumerate(k_mat):
                k3[el_idx] = dirs_(el_idx,k_line,self.el_side)

            k3 = k3.reshape(self.el_side,self.el_side,4)

            # has to be copied for freeing the memory
            feature = simresults['temperature'][0].copy()
            target = simresults['temperature'][-1].copy()
            self.data_dict['features'].append((feature,k3))
            self.data_dict['targets'].append(target)

            print(f'finished simulation {i+1}')

        print(f'Writing to target file ' + self.out_name)
        with open(self.out_name,'wb') as pd:
            pickle.dump(self.data_dict,pd)

    def nonzeros(self,k_mat):
        """ Extracting the nonzero elements of the k-matrix

        In: np.array - k-matrix
        Out: np.array - nonzero elements (vector shaped)
        """
        # access the nonzero elements of k_mattrix as further input
        mat_indcs = np.array(np.where(k_mat > 0.)).T
        # trying to copy everywhere to get rid of the references
        nonzero = k_mat[mat_indcs[:,0],mat_indcs[:,1]].copy()

        # scaling
        n_min = np.min(nonzero)
        n_max = np.max(nonzero)
        nonzero = (nonzero - n_min)/(n_max - n_min)

        return nonzero
        


