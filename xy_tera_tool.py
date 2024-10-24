# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 17:33:09 2024

@author: matth
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import scipy as sc
import pandas as pd

test_path = r"C:\Users\matth\OneDrive - University of Birmingham\Y4 Project\BeamLensBiggerAperPar (1).txt"



class XY_File:
    
    def __init__(self, xy_file_path):
        self.xy_file_path = xy_file_path
        
        with open(self.xy_file_path) as file:
                self.pixel_row = file.readline().strip().split()
                
        self.pixel_x_length = float(self.pixel_row[0]) # mm
        self.pixel_y_length = float(self.pixel_row[1]) # mm
        
        self.df = pd.read_csv(self.xy_file_path, delim_whitespace=True, header=None, skiprows=1)
        self.time_data = self.df.iloc[:, 0]
        
        self.number_pixels = len(self.df.iloc[0,1:])
        
    def collect_single_pixel(self, pixel_index, plot=False):
        if plot:
            plt.plot(self.time_data, self.df.iloc[:, pixel_index])
            return self.df.iloc[:, pixel_index]
        else:
            return self.df.iloc[:, pixel_index]
    
    def create_grid(self, number_per_row):
        pass
    
    def create_test_row(self):
        fig, ax = plt.subplots(figsize=(12, 2))
        ax.set_xlim([0, self.number_pixels * self.pixel_x_length])
        ax.set_ylim([0, self.pixel_y_length])
        ax.set_aspect('auto')
        intensity_data = self.df.values
        
        self.heatmap = ax.imshow(intensity_data[0].reshape(1, -1), cmap='Greens_r', aspect='auto', extent=[0, self.number_pixels * self.pixel_x_length, 0, self.pixel_y_length])
        
        cbar = plt.colorbar(self.heatmap, ax=ax)
        cbar.set_label('Intensity')
        
        def update_heatmap(frame):
            # Update the heatmap with data from the next time step
            self.heatmap.set_array(intensity_data[frame].reshape(1, -1))
            return self.heatmap
            
        self.ani = FuncAnimation(fig, update_heatmap, frames=len(self.time_data), interval=0.0521E-9, repeat=False)
        
        plt.show()