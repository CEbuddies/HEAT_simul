# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 20:43:49 2021

@author: Nic
"""
from VisualizePartsim import VisualizePartsim
from SimulSetup import Simulation
import time
import numpy as np

# simulation script - example
el_side = 40
x = np.linspace(0,1,el_side).reshape(el_side,1)
locs = ['left','right','top','bottom']
bc_left = x**2
bc_right = x**2
bc_right = bc_right[::-1]
bc_top = x**2
bc_bottom = x**2
bc_bottom = bc_bottom[::-1]

bc_vals = [bc_left,bc_right,bc_top,bc_bottom]
bc_dict = {'locations':locs,'values':bc_vals}

sim = Simulation((el_side,el_side),bc_dict)
t_start = time.time()
simresults = sim.simulation_run_for(0.01,100)
t_end = time.time()

vis = VisualizePartsim(simresults['temperature'])
vis.matshow_field(100,1)

print(f'time (simulation) elapsed: {t_end-t_start}s')