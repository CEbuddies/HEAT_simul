# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 15:40:34 2021

@author: User
"""

import torch 
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle

class HEAT_Data(Dataset):
    
    def __init__(self):
        
        with open('training_data.sml','rb') as pl:
            xy = pickle.load(pl) # a dict of data
        
        # concatenate all the data together to list of np.arrays
        for idx,tup in enumerate(xy['features']):
            xy['features'][idx] = np.concatenate(tup)

        self.x = xy['features'] # a list of np.arrays (T_start,k_mat)
        
        self.y = xy['targets']
        self.nums = len(self.x)
        
    def __getitem__(self,index):
        # string create 
        return self.x[index], self.y[index]
        
    def __len__(self):
        
        return self.nums
    

        
