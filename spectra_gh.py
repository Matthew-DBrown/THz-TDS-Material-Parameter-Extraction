# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 17:20:42 2024

@author: matth
"""

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import pandas as pd
import scipy as sc
import os
from scipy import constants

n_air = 1

def find_index(array, target):
    '''
    Parameters
    ----------
    array : Numpy array
        
    target : Float

    Returns
    -------
    Index in array of a value nearest target value.

    '''
    
    diff = np.abs(array - target)
    return np.argmin(diff)

def average(stack_of_arrays):
    pass # Perform the average on the arrays, might need to consider axes

def zero(ft_data):
    pass # Function to zero data? May be best to have this inside the class
    
def interpolate(data):
    pass # some interpolation? Material extraction. Again, maybe best in the class
    
def calculate_dynamic_range(spectrum):
    pass # Makes a plot like in Miguel paper. Class-Method / Function?

def Calculate_T_theory(n1_tilde, n2_tilde, n3_tilde, omega, L):
    # working off the assumption that FP(omega) = 1 i.e. window the signal to remove fabry-perot.
    return np.array([((2*n2_tilde*(n1_tilde+n3_tilde))/((n2_tilde+n1_tilde)*(n2_tilde+n3_tilde))) * np.exp(-(n2_tilde-n_air)*(omega*L/constants.c)*1j)])



class Signal:
    
    def __init__(self, signal_file, name=None):
        '''
        Parameters
        ----------
        signal_file : RAW String
            Takes the directory of the file, with '\' as '/'.

        '''
        self.signal_file = signal_file
        self.name = name
        self.extension = self.signal_file.rsplit('.', 1)[-1] # Extracts the filetype.
        if self.extension == 'csv':
            self.df = pd.read_csv(self.signal_file)
            self.x_data = self.df.iloc[:, 0]
            self.y_data = self.df.iloc[:, 1]
            self.data_size = len(self.y_data)
            # self.odu_data = self.df.iloc[:,2] # ODU's equivalent time measurement
        elif self.extension == 'txt':
            self.df = pd.read_csv(self.signal_file, delim_whitespace=True, header=None)
            self.x_data = self.df.iloc[:, 0]
            self.y_data = self.df.iloc[:, 1]
            self.data_size = len(self.y_data)
            # self.odu_data = self.df.iloc[:,2]
        elif self.extension == 'xlsx':
            pass
        else:
            print("Invalid Filetype!")
        
        #plt.plot(self.x_data, self.y_data, linestyle='-', marker='o', color='m', markersize=2, linewidth=1)
        #plt.xlabel("Time")
        #plt.ylabel("_____")

    def perform_fft(self, logscale=False):
        # Assumes equal interval rate
        # Something about FFT not that good for number of data =! 2^n
        self.x_rounded = np.round(self.x_data, 3)
        self.time_step = self.x_rounded[1]-self.x_rounded[0]
        self.fft_signal = np.fft.fft(self.y_data)
        n = len(self.x_data)
        self.fast_freqs = np.fft.fftfreq(n, self.time_step)
        # self.fft_signal_pos = self.fft_signal[:self.data_size//2]
        # self.fft_freq_array_pos = self.fft_freq_array[:self.data_size//2]
        
        plt.plot(self.fast_freqs[:self.data_size//2], np.abs(self.fft_signal[:self.data_size//2]), linestyle='-', marker='o', color='b')
        if logscale:
            plt.yscale('log')
        plt.title(self.signal_file)
        plt.ylabel("Arbitrary Units", fontsize=14)
        plt.xlabel("Frequency (THz)", fontsize=14)
        plt.grid(True, linestyle='--', color='k', alpha=0.4)
    
    def period_lombscargle(self, f_min, f_max, n, normalize):
        # Apparently Lombscargle uses angular frequencies!
        
        frequency_array = np.linspace(f_min, f_max, int(n), endpoint=True)/(2*np.pi)
        pgram = sc.signal.lombscargle(self.x_data, self.y_data, frequency_array, normalize=normalize)
        plt.title(f"Periodogram for: {self.signal_file}")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Amplitude (?)")
        plt.plot(frequency_array, pgram)
        plt.show()
    
    def calc_lombscargle(self, f_min, f_max, n, normalize):
        # Does not plot --> faster to collect data
        frequency_array = np.linspace(f_min, f_max, int(n), endpoint=True)/(2*np.pi)
        pgram = sc.signal.lombscargle(self.x_data, self.y_data, frequency_array, normalize=normalize)
        return frequency_array, pgram
    
    def calc_fft(self, logscale=False):
        self.x_rounded = np.round(self.x_data, 3)
        self.time_step = self.x_rounded[1]-self.x_rounded[0]
        self.fft_signal = np.fft.fft(self.y_data)
        n = len(self.x_data)
        self.fast_freqs = np.fft.fftfreq(n, self.time_step)
        if logscale:
            return self.fast_freqs[:len(self.x_data)//2], np.log10(np.abs(self.fft_signal[:len(self.x_data)//2]))
        else:
            return self.fast_freqs[:len(self.x_data)//2], np.abs(self.fft_signal[:len(self.x_data)//2])
     
    def calc_complex_fft(self, positive=True, logscale=False):
        self.x_rounded = np.round(self.x_data, 3)
        self.time_step = self.x_rounded[1] - self.x_rounded[0]
        self.fft_signal = np.fft.fft(self.y_data)
        n = len(self.x_data)
        self.fast_freqs = np.fft.fftfreq(n, self.time_step)
        if positive and not logscale:
            return self.fast_freqs[:len(self.x_data)//2], self.fft_signal[:len(self.x_data)//2]
        elif positive and logscale:
            return self.fast_freqs[:len(self.x_data)//2], np.log10(self.fft_signal[:len(self.x_data)//2])
        elif not positive and not logscale:
            return self.fast_freqs, self.fft_signal
        elif not positive and logscale:
            return self.fast_freqs, np.log10(self.fft_signal) # Pretty sure this will not work anyway as some values are negative
        
    def calculate_windowed_fft(self, t1, t2, plot=False, logscale_state=False):
        x_1_index = find_index(self.x_data, t1)
        x_2_index = find_index(self.x_data, t2)
        self.x_rounded_window = np.round(self.x_data, 3)[x_1_index:x_2_index]
        # self.window_timestep = self.x_rounded_window[1]-self.x_rounded_window[0]
        self.window_timestep = 0.167
        self.window_fft_signal = np.fft.fft(self.y_data[x_1_index:x_2_index])
        n = len(self.x_rounded_window)
        self.windowed_fast_freq = np.fft.fftfreq(n, self.window_timestep)
        if plot:
            if logscale_state:
                plt.plot(self.windowed_fast_freq[:n//2], np.abs(self.window_fft_signal[:n//2]))
                plt.yscale('log')
            else:
                plt.plot(self.windowed_fast_freq[:n//2], np.abs(self.window_fft_signal[:n//2]))
        else:
            if logscale_state==False:
                return self.windowed_fast_freq[:n//2], np.abs(self.window_fft_signal[:n//2])
            else:
                return self.windowed_fast_freq[:n//2], np.log10(np.abs(self.window_fft_signal[:n//2]))
    
    
    def re_plot(self):
        plt.plot(self.x_data, self.y_data, linestyle='-', marker='o', color='m', markersize=2, linewidth=1)
        plt.xlabel("Time")
        plt.ylabel("_____")
        if self.name:
            plt.title(f"{self.name}")
    
    def show_data(self):
        '''
        Can be handy to see.
        '''
        return self.x_data, self.y_data #, self.odu_data
    
    def spectrogram(self):
        self.x_rounded = np.round(self.x_data, 3)
        time_step = self.x_rounded[1]-self.x_rounded[0] * 10**(-12)
        
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
        ax1.plot(self.x_data, self.y_data)
        ax1.set_label("Signal")
        
        Pxx, freqs, bins, im = ax2.specgram(self.y_data, NFFT=256, Fs=1/time_step)
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Frequency (Hz)')
        
        plt.show()     
        
    def measured_transfer_function(self, fft_signal, reference):
        # In the frequency domain
        self.msrd_transfer_function = fft_signal / reference.calc_complex_fft()[1]
        return self.msrd_transfer_function
        
    def phase_transfer_function(self, plot=False, name=False):
        self.phase_data = np.angle(self.msrd_transfer_function)
        frequencies = self.calc_complex_fft()[0]
        if plot:
            fig = plt.figure(figsize=(5,5))
            plt.plot(frequencies, self.phase_data/np.pi, color='b')
            plt.xlabel("Frequency (THz)", fontsize=14)
            plt.xlim(np.min(frequencies), np.max(frequencies))
            plt.ylabel(r"Angle ($\pi$ radians)", fontsize=14)
            plt.hlines([-1, 1], np.min(frequencies), np.max(frequencies), linestyle='--', color='r')
            if name:
                plt.title(f"Phase of Measured Transfer Function\n{name}", fontsize=14)
            elif not name:
                plt.title(f"Phase of Measured Transfer Function\n{self.signal_file}")
            plt.show()
            
    def unwrap_phase_function(self):
        pass
            
class MultiSignals:
    
    def __init__(self, signals_path):
        self.signals_dic = {}
        
        for file in os.listdir(signals_path):
            if file.rsplit('.', 1)[-1] =="txt":
                if file not in self.signals_dic:
                    self.signals_dic[file] = Signal(os.path.join(signals_path, file), f"{file}")
            else:
                print(f'Ignored: {file}')
        
        self.signals_dic_nref = self.signals_dic.copy() #dictionary without reference
        self.signals_dic_nref.pop('reference.txt')
        y_data_list = [np.array(signal.y_data) for signal in self.signals_dic_nref.values()]
        self.array_stack_nref = np.vstack(y_data_list)
        
        self.reference_freqs = self.signals_dic['reference.txt'].calc_fft(logscale=False)[0]
        self.reference_fft = self.signals_dic['reference.txt'].calc_fft(logscale=False)[1]
                    
                
    def plot_all(self):
        for file, signal in self.signals_dic.items():
            plt.plot(signal.x_data, signal.y_data, label=f'{file}')
            plt.xlabel("Time (ps)")
            plt.ylabel("Amplitude (a.u.)")
        plt.legend(loc='upper right', fontsize=14)
        plt.grid(True, color=-'k', linestyle='--', alpha=0.5)
        plt.show()
        
    def calc_std_dev(self):
        self.std_dev_array = np.std(self.array_stack_nref, 0)
        return self.std_dev_array
    
    def calc_avg_array(self):
        self.avg_array = np.average(self.array_stack_nref, 0)
        return self.avg_array
    
    def plot_snr_average(self):
        fig, ax1 = plt.subplots()
        ax1.plot(self.signals_dic['reference.txt'].x_data, self.calc_avg_array(), color='b', linestyle='-', label="Average")
        ax1.plot(self.signals_dic['reference.txt'].x_data, 20*self.calc_std_dev(), color='g', linestyle='-', label ='Std Dev (x20)')
        ax1.set_ylabel("Amplitude (a.u.)", fontsize=14)
        ax1.set_xlabel("Time (ps)", fontsize=14)
        ax1.set_xlim(0, np.max(self.signals_dic['reference.txt'].x_data))
        ax1.grid(True, color='k', linestyle='--', alpha=0.5)
        ax2 = ax1.twinx()
        ax2.plot(self.signals_dic['reference.txt'].x_data, np.abs(self.calc_avg_array()/self.calc_std_dev()), color='r', linestyle='dotted', label='SNR')
        ax2.set_ylabel("SNR (a.u.)", fontsize=14)
        ax1.legend(loc='upper left', fontsize=14)
        ax2.legend(loc='upper right', fontsize=14)    
        fig.suptitle("Radix(6) Data")
  
    
    
    