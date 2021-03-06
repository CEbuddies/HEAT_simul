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
        
        with open('smaples_tars.trn','rb') as pl:
            xy = pickle.load(pl) # a dict of data
        self.x = xy['samples'] # a list of tuples (T_start,k_mat)
        # alle x mal 1000 nehmen 
        self.y = xy['features']
        self.nums = len(self.x)
        
    def __getitem__(self,index):
        # string create 
        return self.x[index], self.y[index]
        
    def __len__(self):
        
        return self.nums
    
# use this in the pytorch lightning module
dataset = HEAT_Data()
dataloader = DataLoader(dataset=dataset,
                        batch_size=16,shuffle=True,num_workers=2)
        