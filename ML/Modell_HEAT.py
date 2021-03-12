# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 09:51:04 2021

@author: nwilhelm
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
    
    def __init__(self,in_=14076,out_=1600,parts=1600,nonzeros=1522):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_,2**14),
            nn.ReLU(),
            nn.Linear(2**14,2**14),
            nn.ReLU(),
            nn.Linear(2**14,2**11),
            nn.ReLU(),
            nn.Linear(2**11,out_))
        
    def forward(self, x):
        
        res = self.net(x)
        return res
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=1e-3)
        return optimizer 
    
    def training_step(self, batch, batch_idx):
        
        start_config, end_config = batch # features and tars 
        # slice up start_config
        start_config = start_config.float() # float seems to work
        end_config = end_config.float()
        predict = self(start_config)
        loss = nn.MSELoss()
        out = loss(predict,end_config)
        
        return {'loss':out}
    
    def train_dataloader(self):
        train_dataset = HEAT_Data()
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=32,shuffle=True,num_workers=4)
        
        return train_loader
    
if __name__ == '__main__':
    
    # this one only for testing purposes
    # trainer = Trainer(fast_dev_run = True,gpus=1)
    trainer = Trainer(max_epochs=50,gpus=1)
    model = HEAT_model() # go with defaults for 40x40 domain
    trainer.fit(model)
    
    
