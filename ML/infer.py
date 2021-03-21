# author : Nic
# Inference for the model

from Modell_HEAT import HEAT_model
import pickle
import numpy as np
import torch
import argparse
# TODO add argparse for all files 
with open('val_data_lin.mlinp','rb') as pl:
    val_data = pickle.load(pl)

# create the model and load the statedict 
model = HEAT_model.load_from_checkpoint('final_chk_lin.ckpt')

for idx,el in enumerate(val_data['features']):
    val_data['features'][idx] = np.concatenate(el)

# list -> np.array 10x7480
inp_ = np.array(val_data['features'])
inp = torch.tensor(inp_).float()

val_out = model(inp)
# get it back to numpy 
val_out_np = val_out.detach().cpu().numpy()

# add to the dict 
val_data['prediction'] = val_out_np

with open('infered_data_lin.val','wb') as pd:
    pickle.dump(val_data,pd)

