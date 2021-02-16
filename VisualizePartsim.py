# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 16:39:07 2020

@author: User
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

class VisualizePartsim():
    """
    all the visualization is done in here
    """
    def __init__(self,trajec):
        self.valuepts = len(trajec) 
        self.traj = trajec
        shape = np.shape(trajec)
        self.sidesz = int(np.sqrt(shape[1]))
    def matshow_field(self,freq=10):
        freq = freq
        
        tmats = self.traj.reshape(self.valuepts,self.sidesz,self.sidesz)
        fig, ax1 = plt.subplots()
        plt.show()
        minval = np.min(tmats)
        maxval = np.max(tmats)
        perc_show = 0.1
        
        for i in range(int(perc_show*self.valuepts)):
            try:
                plotob = ax1.matshow(tmats[i],cmap=matplotlib.cm.plasma,interpolation='nearest',vmin=minval,vmax=maxval)
                plt.draw()
                if i == 0:
                    fig.colorbar(plotob,ax=ax1)
                if i % 10 == 0:
                    print('{0:d} of {1:d} plots done...'.format(i,int(self.valuepts*perc_show)))
                plt.pause(0.01)
                
            except KeyboardInterrupt:
                print('The visualisation has been terminated by the user...')
                #raise Exception
                break
            
    def plot_trajs(self):
        fig0, ax0 = plt.subplots()
        ax0.plot(range(self.valuepts),self.traj)
        #plt.close(fig0)
        
    def load_traj(self):
        """
        try to automatically load a corresponding file from the current directory
        """
        pass