# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 14:57:37 2021

@author: nic
"""
import numpy as np 
from collections import defaultdict
import pickle


class Simulation():
    
    def __init__(self,grid_size):
        
        # domain is shaped 1x1
        self.grid_el_p_side = grid_size[0]
        self.number_parts = grid_size[0]*grid_size[1]
        x = np.linspace(0,1,self.grid_el_p_side)
        y = x
        xy = np.meshgrid(x,y)
        self.xy_coords = np.array(xy).T.reshape(self.grid_el_p_side**2,2)
        
        # simul stettings 
        self.dT_tol = 0.1 # absolute value of Temperature change
        # squared simulation domain for start - systems matrix
        # randomly distributed 
        self.sys_mat = 0.025*np.random.rand(grid_size[0]**2,grid_size[0]**2)
        self.sys_mat = self.sys_mat*1 # theoretical area of 1 
        self.cont_area_p2p = 0.01
        self.sys_mat = self.sys_mat*self.cont_area_p2p
        # random starting temperature
        self.temp = np.random.rand(grid_size[0]*grid_size[1],1)
        self.c = 0.9
        self.m = 1.01
        self.ener = self.temp*self.c*self.m
        self.r = (1/grid_size[0])/2
        
        self.mesh = Mesh(grid_size[0],grid_size[1],1,1)
        self.connect, self.k_mat = self.mesh.mat_indices(3,3)
        
    def calc_sysmat(self):
        # suitable for nonlinear behavior as well
        return
    
    def random_sysmat(self):
        
        rand_sysmat = 0.02*np.random.rand(self.number_parts,self.number_parts)
        return 
    
    def find_pairs(self,number_parts):
        # not needed so far, maybe deprecate
        """
        finds the adjacent particles for the connectivity
        creates a dictionary that contains the connected parts
        in: a variable number of particles - necessary ?

        """
        # create empty dictionary 
        d_connect = defaultdict().fromkeys(range(number_parts))
        # outter loop that goes for every single vector
        for num1, coord in enumerate(self.xy_coords):
            temp_list = list()
            # inner loop for the dinstance calculation
            for num2, coord2 in enumerate(self.xy_coords):
                # calculate distance vector
                dist_v = coord2 - coord
                # euklidean norm 
                dist = np.round(np.linalg.norm(dist_v),4)
                if dist <= 2*self.r and dist > 0:
                    temp_list.append(num2)
                    
            d_connect[num1] = temp_list
            
        return d_connect
    
    def k_mat(self,number_parts):
        
        indexes, k_mat = self.mesh.mat_indices(3,3)
        
        
        return k_mat, indexes
    
    def mat_indices(self,lines,cols):
        
        list_idx = []
        
        
        for el_no in range(lines*cols):
            col_id = el_no % cols
            line_id = int((el_no-col_id)/3)
            list_idx.append(np.array([line_id,col_id]))
            
        return list_idx
        
                
    def part2part_grad(self):
        
        # create gradients 
        temp_mat = np.tile(self.temp,(1,self.number_parts))
        grad_T = temp_mat.T - temp_mat
        qmat = grad_T*self.k_mat
        # sum over j
        q_vec = qmat.sum(1)
        q_vec = q_vec.reshape(self.number_parts,1)
        
        return q_vec
    
    def fwd_euler(self,f,grad,dt):
        
        f_plus1 = f+grad*dt
        
        return f_plus1
    
    def temp_from_E(self):
        
        temp = self.ener/(self.c*self.m)
        return temp
    
    def store_data(self,data_to_store):
        
        with open('simuldata.smd','wb') as pd:
            pickle.dump(data_to_store,pd)
        
        return
    
    def simulation_run_while(self):
        # TODO - simulation until steady state 
        # find a measure for the most little gradient 
        grad = 10000
        while grad > 0.1:
            pass
        return
    
    def simulation_run_for(self,timestep,time,frames=10000):
        
        num_steps = int(round(time/timestep))
        temp_data = np.zeros((num_steps,self.number_parts))
        temp_grad_data = np.zeros((num_steps,self.number_parts))
        # wab spatial gradients!!!
        for t in range(num_steps):
            q_vec = self.part2part_grad()
            old_temp = self.temp
        
            # change this and also use adaptive solver 
            new_ener = self.fwd_euler(self.ener,q_vec,timestep)
            self.ener = new_ener
            # update temperature
            self.temp = self.temp_from_E()
            temp_data[t,:] = self.temp.T
            temp_grad = (self.temp - old_temp)/timestep
            temp_grad_data[t,:] = temp_grad.T
            time += 0.01
        # obtain time vector at the end 
        time_ = np.linspace(0,num_steps*timestep,num_steps)
        
        results = dict()
        results['time'] = time_
        results['temperature'] = temp_data
        results['temperature_grad'] = temp_grad_data
        
        
        return results
    
class Material():
    
    # build this up to create a whole object from it 
    def __init__(self):
        self.c = 0.55
        self.m = 1.1
        
class Mesh():
    #### meshclass
    def __init__(self,el1,el2,domain1,domain2):
        
        meshdist1 = domain1/el1
        meshdist2 = domain2/el2
        start_d1 = meshdist1/2
        start_d2 = meshdist2/2
        self.x = np.linspace(start_d1,domain1-start_d1,el1)
        self.y = np.linspace(start_d2,domain2-start_d1,el1)
        self.xy = np.meshgrid(self.x,self.y)
        self.coords = np.array(self.xy).T.reshape(el1*el2,2)
        
        
        return
    
    def mat_indices(self,lines,cols):
        
        list_idx = []
        k_mat = np.zeros((lines*cols,lines*cols))
        
        
        for el_no in range(lines*cols):
            col_id = el_no % cols
            line_id = int((el_no-col_id)/3)
            list_idx.append(np.array([line_id,col_id]))
            
        for k_line in range(9):
            for k_col in range(9):
                #if np.abs(np.sum(list_idx[k_line] - list_idx[k_col])) == 1:
                if (np.abs(list_idx[k_line][0] - list_idx[k_col][0]) 
                    + np.abs(list_idx[k_line][1] - list_idx[k_col][1])) == 1:
                    if k_col <= k_line:
                        k_mat[k_line,k_col] = k_mat[k_col,k_line]
                    else:
                        k_mat[k_line,k_col] = np.random.rand()*0.1
            
        return list_idx, k_mat 
        
        
        