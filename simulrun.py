# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 03:15:31 2021

@author: Nic
"""

# TODO: obtain gradient information
# obtain field 
# TODO: pickle stuff to binary instead of file
from SimulSetup import Simulation
from VisualizePartsim import VisualizePartsim
import numpy as np
import pickle
linepts = 8
colpts = 8
sim = Simulation((linepts,colpts))

print(sim.temp)
t = 0
temp_list = []
ener_grad = []
temp_grad = []
# k_mat
for timestep in range(10000):
    q_vec = sim.part2part_grad()
    old_temp = sim.temp

    new_ener = sim.fwd_euler(sim.ener,q_vec,0.01)
    sim.ener = new_ener
    # update temperature
    sim.temp = sim.temp_from_E()
    temp_list.append(sim.temp)
    temp_grad.append((old_temp - sim.temp)/0.01)
    t += 0.01
    


print(sim.temp)
temp_array = np.array(temp_list)
grad_array = np.array(temp_grad)
grad_array = grad_array.reshape(len(grad_array),linepts*colpts)
# still hardcoded - caution
temp_array = temp_array.reshape(len(temp_array),linepts*colpts)
vis = VisualizePartsim(temp_array)
vis.matshow_field(100,1)

