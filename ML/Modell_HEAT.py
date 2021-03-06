# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 09:51:04 2021

@author: User
"""
import torch
from torch import nn
import torch.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from Dataset_HEAT import HEAT_Data


# Modell for HEAT-Learning 

class HEAT_model(pl.LightningModule):
    
    def __init__(self,in_,out_):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_,512),
            nn.ReLU(),
            nn.Linear(512,1024),
            nn.ReLU(),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Linear(1024,out_))
        
    def forward(self, x):
        
        res = self.net(x)
        return res
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=1e-3)
        return optimizer 
    
    def trainig_step(self, batch, batch_idx):
        
        start_config, end_config = batch # features and tars 
        # maybe do something to x - e.g. reshape
        predict = self(start_config)
        loss = nn.MSELoss()
        out = loss(predict,end_config)
        
        return {'loss':out}
    
    def train_dataloader(self):
        train_dataset = HEAT_Data()
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=16,shuffle=True)
    