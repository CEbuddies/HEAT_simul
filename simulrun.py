# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 03:15:31 2021

@author: User
"""
from SimulSetup import Simulation

sim = Simulation((3,3))

print(sim.temp)
t = 0
for timestep in range(10000):
    q_vec = sim.part2part_grad()

    new_ener = sim.fwd_euler(sim.ener,q_vec,0.01)
    sim.ener = new_ener
    # update temperature
    sim.temp = sim.temp_from_E()
    t += 0.01


print(sim.temp)
