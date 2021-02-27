# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 20:43:49 2021

@author: Nic
"""
from VisualizePartsim import VisualizePartsim
from SimulSetup import Simulation

# simulation script - example
sim = Simulation((20,20))
simresults = sim.simulation_run_for(0.01,100)

vis = VisualizePartsim(simresults['temperature'])
vis.matshow_field(50,1)