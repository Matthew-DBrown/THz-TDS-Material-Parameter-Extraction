# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 15:18:42 2024

@author: matth
"""

import numpy as np
import matplotlib.pyplot as plt
import spectra as sp
import os
import noise_floor_time_constants as flr

markers = ['o', 's', '^', '*', 'd', 'v', 'p']
colors = ['firebrick', 'orangered', 'olive', 'forestgreen', 'steelblue', 'indigo', 'mediumvioletred']
tableau_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948', '#B07AA1']
viridis_colors = ['#440154', '#482777', '#3F4A8A', '#31688E', '#26828E', '#1F9E89', '#35B779']

def plot_averages(signals):
    marker_idx = 0
    color_idx = 0
    for group, signal in signals.items():
        plt.plot(signal.x_data, signal.y_data, linestyle='None', marker=markers[marker_idx], color=colors[color_idx], label=f"Time Constant: {group}ms", alpha=0.4)
        marker_idx += 1
        color_idx += 1
    
    plt.grid(True, color='k', linestyle='--', alpha=0.4)
    plt.legend(loc='upper right')
    
def plot_averages_fft(signals, linestyle_cond, logscale=True):
    marker_idx = 0
    color_idx = 0
    for group, signal in signals.items():
        freqs, fft_signal = signal.calc_fft(logscale)[0], signal.calc_fft(logscale)[1]
        plt.plot(freqs, fft_signal, linestyle=linestyle_cond, marker=markers[marker_idx], color=colors[color_idx], label=f"Time Constant: {group}ms", alpha=0.4)
        marker_idx += 1
        color_idx += 1
        
    plt.grid(True, color='k', linestyle='--', alpha=0.4)
    plt.legend(loc='upper right', fontsize=14)
        
def plot_average_fft_magnitude(signals, linestyle_cond):
    marker_idx = 0
    color_idx = 0
    plt.figure(figsize=(8,8))
    for group, signal in signals.items():
        signal.x_rounded = np.round(signal.x_data, 3)
        plt.magnitude_spectrum(signal.y_data, 1/0.169, scale='dB', color=viridis_colors[color_idx], marker=markers[marker_idx], markersize=6, linestyle='-', alpha=0.6, label=f"Time Constant: {group}ms")
        #plt.plot(x, y, linestyle=linestyle_cond, marker=markers[marker_idx], color=colors[color_idx], label=f"Time Constant: {group}ms", alpha=0.4)
        marker_idx += 1
        color_idx += 1
    
    #ax = plt.gca()
    #ax.patch.set_facecolor("k")    
    plt.xlabel("Frequency (THz)", fontsize=14)
    plt.ylabel("Magnitdue (dB)", fontsize=14)
    plt.xlim(0, 3)
    plt.grid(True, color='k', linestyle='--', alpha=0.4)
    #plt.legend(loc='upper right', fontsize=14)


def plot_average_fft_dB_test(signals, linestyle_cond):
    marker_idx = 0
    color_idx = 0
    plt.figure(figsize=(8,8))
    for group, signal in signals.items():
        signal.x_rounded = np.round(signal.x_data, 3)
        plt.plot(signal.calc_fft(logscale=False)[0], 20*np.log10(signal.calc_fft(logscale=False))[1], color=viridis_colors[color_idx], marker=markers[marker_idx], markersize=6, linestyle='-', alpha=0.6, label=f"Time Constant: {group}ms")
        #plt.magnitude_spectrum(signal.y_data, 1/0.169, scale='dB', color=viridis_colors[color_idx], marker=markers[marker_idx], markersize=6, linestyle='-', alpha=0.6, label=f"Time Constant: {group}ms")
        #plt.plot(x, y, linestyle=linestyle_cond, marker=markers[marker_idx], color=colors[color_idx], label=f"Time Constant: {group}ms", alpha=0.4)
        marker_idx += 1
        color_idx += 1
    
    #ax = plt.gca()
    #ax.patch.set_facecolor("k")    
    plt.xlabel("Frequency (THz)", fontsize=14)
    plt.ylabel("Magnitdue (dB)", fontsize=14)
    plt.xlim(0, 3)
    plt.grid(True, color='k', linestyle='--', alpha=0.4)
    #plt.legend(loc='upper right', fontsize=14)

if __name__ == "__main__":
    
    file_groups = {}
    
    parent_path = r"C:\Users\matth\OneDrive - University of Birmingham\Y4 Project\Lab_data\time_constant_investigation\focused"
    for file in os.listdir(parent_path):
        if file.endswith(".txt"):
            group = file[:4]
            if group not in file_groups:
                file_groups[group] = []
            file_groups[group].append(file)
    
    stacks = {}
    
    for group, files in file_groups.items():
        y_data = []
        for file in files:
            file_path = os.path.join(parent_path, file)
            
            df = np.loadtxt(file_path)
            y_column = df[:,1]
            y_data.append(y_column)
        
        stacks[group] = np.vstack(y_data)
        
    average_stacks = {}
    
    sample_signal = sp.Signal(os.path.join(parent_path, "0001ms_foc.txt"))
    
    for group, array in stacks.items():
        average_stacks[group] = np.average(array, axis=0)
        # plt.plot(sample_signal.x_data, average_stacks[group], label=f"Avg. of {group}ms time constant")
    
    #plt.legend(loc='upper right')
    #plt.show()
    
    signals = {}
    
    for group, avg_data in average_stacks.items():
        signal_name = f"Signal_{group}ms"
        
        temp_file_path = os.path.join(parent_path, "temp_file", f"{group}ms_avg.txt")
        
        combined_data = np.column_stack((sample_signal.x_data, avg_data))
        
        np.savetxt(temp_file_path, combined_data, fmt='%.6f')
        
        signals[group] = sp.Signal(temp_file_path)
    
    #plot_averages(signals)
    #plot_averages_fft(signals, '-', logscale=False)
    #plot_average_fft_magnitude(signals, '-')
    plot_average_fft_dB_test(signals, '-')
    color_idx = 0
    for group, signal in signals.items():
        line, f_min, f_max = flr.find_noise_floor(signal, 2.41)
        plt.hlines(line, 0, f_max, linewidth=1, color=viridis_colors[color_idx])
        color_idx += 1
    plt.legend(loc='lower left', fontsize=14)    
    plt.show()