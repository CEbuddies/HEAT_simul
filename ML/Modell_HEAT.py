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
from pytorch_lightning import Trainer


# Modell for HEAT-Learning 

class HEAT_model(pl.LightningModule):
    
    def __init__(self,in_,out_,parts=1600,nonzeros=1522):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_,4096),
            nn.ReLU(),
            nn.Linear(8192,8192),
            nn.ReLU(),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Linear(4096,out_))
        
    def forward(self, x):
        
        res = self.net(x)
        return res
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=1e-3)
        return optimizer 
    
    def training_step(self, batch, batch_idx):
        
        start_config, end_config = batch # features and tars 
        # slice up start_config
        temps = start_config[0]
        k_mat = start_config[1]
        inp_ = torch.cat((temps,k_mat),1)
        predict = self(inp_)
        loss = nn.MSELoss()
        out = loss(predict,end_config)
        
        return {'loss':out}
    
    def train_dataloader(self):
        train_dataset = HEAT_Data()
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=32,shuffle=True)
        
        return train_loader
    
if __name__ == '__main__':
    
    trainer = Trainer(fast_dev_run = True)
    model = HEAT_model(1600+1522,1600)
    trainer.fit(model)
    
    
