# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 15:21:29 2024

@author: matth
"""

import tkinter as tk
import spectra as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy as sc
import pandas as pd
import os
from scipy import signal
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy import constants
from tqdm import tqdm
import sounddevice as sd
   
na_tilde = 1.00027-(1j*0)
     

class SampleRefPair:
    
    def __init__(self, sample_signal, reference_signal, measured_thickness, unwrap_range_selec=False, plot=True):
        # Both sample_signal and reference_signal are objects. These are best
        # derived from spectra MultiSignals class.
        self.sample_signal = sample_signal
        self.reference_signal = reference_signal
        self.measured_thickness = measured_thickness
        self.fft_freqs, self.fft_sample = sample_signal.calc_complex_fft()[0], sample_signal.calc_complex_fft()[1]
        self.fft_reference = reference_signal.calc_complex_fft()[1]
        self.measured_H = self.fft_sample / self.fft_reference
        self.time_domain_h = np.fft.ifft(self.measured_H, len(self.sample_signal.y_data))
        self.time_data = self.sample_signal.x_data * 1E-12 # ps
        
        # Finding delta t(thickness) for thickness range
        t_max_idx = sp.find_index(self.sample_signal.y_data, np.max(self.sample_signal.y_data)) # This finds the index of the peak
        t_min_idx = sp.find_index(self.reference_signal.y_data, np.max(self.reference_signal.y_data))
        self.delta_t_peaks = self.time_data[t_max_idx] - self.time_data[t_min_idx] # Finding time difference between peaks
        self.l_upper = (self.delta_t_peaks * constants.c) / 0.2 # As advised in metrology ref [5]
        self.l_lower = (self.delta_t_peaks * constants.c) / 7 # As advised in metrology ref [5]
        
        self.starting_ns = np.ones_like(self.fft_freqs) # These aren't used but they are arrays with the same shape as frequency array since there is a n,k for each frequency point
        self.starting_ks = np.zeros_like(self.fft_freqs)
        
        self.measured_phase_function = np.angle(self.measured_H)
        
        self.phase_peaks, _ = find_peaks(self.measured_phase_function, height=2, distance=1)
        M = 0
        self.unwrapped_phase = np.zeros_like(self.measured_phase_function)
        for i in range(len(self.measured_phase_function)):
            if i in self.phase_peaks:
                M += 1
            self.unwrapped_phase[i]  = self.measured_phase_function[i] - (M*2*np.pi)
        
        # Also confirming with Numpy
        self.backup_unwrap = np.unwrap(self.measured_phase_function)
        
        self.fit_x_min = float(input("Input Minimum Frequency for Analysis (THz): ")) * 1E12 # MISNOMER --> THIS IS LOWER BOUND OF PARAMETER FREQUENCY RANGE
        self.fit_x_max = float(input("Input Maximum Frequency for Analysis (THz): ")) * 1E12
        self.number_fp_pulses = int(input("Number of FP pulses expected: "))
        self.x_min_index = sp.find_index(self.fft_freqs, self.fit_x_min)
        self.x_max_index = sp.find_index(self.fft_freqs, self.fit_x_max)
        if unwrap_range_selec:
            self.unwrap_fit_min = float(input("Input Minimum Frequency for Phase Extrapolation (THz): ")) * 1E12
            self.unwrap_fit_max = float(input("Input Maximum Frequency for Phase Extrapolation (THz): ")) * 1E12
            self.unwrap_min_idx = sp.find_index(self.fft_freqs, self.unwrap_fit_min)
            self.unwrap_max_idx = sp.find_index(self.fft_freqs, self.unwrap_fit_max)

        self.fit_freqs = self.fft_freqs[self.unwrap_min_idx:self.unwrap_max_idx] # For the extrapolation
        self.freqs_interest = self.fft_freqs[self.x_min_index:self.x_max_index] # These are for material parameter extraction
        
        self.popt, self.pcov = sp.interpolate(self.unwrapped_phase[self.unwrap_min_idx:self.unwrap_max_idx], self.fit_freqs, self.unwrap_fit_min, self.unwrap_fit_max)
        self.perr = np.sqrt(np.diag(self.pcov))
        
        self.interpolated_unwrapped_H_phase = self.fft_freqs*self.popt[0] # Through the origin.
        
        self.popt_backup, self.pcov_backup = sp.interpolate(self.backup_unwrap[self.unwrap_min_idx:self.unwrap_max_idx], self.fit_freqs, self.unwrap_fit_min, self.unwrap_fit_max)
        self.perr_backup = np.sqrt(np.diag(self.pcov_backup))
        
        self.interpolated_unwrapped_H_phase_backup = self.fft_freqs*self.popt_backup[0]
        
        print(f"-----------------\nThe difference between both unwrap methods in the fit is: {np.abs(self.popt) - np.abs(self.popt_backup)}\n-----------------\n")
        
        
        self.fs = 1 / ((np.round(self.sample_signal.x_data[1], 3) - np.round(
            self.sample_signal.x_data[0], 3))*1e-12)
        x=1000*self.sample_signal.y_data
        
        Pxx, freqs, bins, _ = plt.specgram(x, NFFT=64, Fs=self.fs, mode='psd', noverlap=32, cmap='hot', vmin=-80, vmax=-60)
        #plt.close()
        
        if plot:
            #fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True, constrained_layout=False)
            #ax4 = fig.add_subplot(414)
            self.fig = plt.figure(figsize=(12, 12))
            gs = gridspec.GridSpec(3, 2, height_ratios=[2, 2, 2], hspace=0.4)

            # Create the first three subplots with a shared x-axis
            self.ax1 = self.fig.add_subplot(gs[0, 0])
            self.ax2 = self.fig.add_subplot(gs[1, 0], sharex=self.ax1)
            self.ax3 = self.fig.add_subplot(gs[2, 0], sharex=self.ax1)
            
            self.ax4 = self.fig.add_subplot(gs[0, 1])
            self.ax5 = self.fig.add_subplot(gs[1, 1])
            self.ax6 = self.fig.add_subplot(gs[2, 1])
            
            self.ax1.set_title(f"Sample: {self.sample_signal.name.rsplit('.txt')[0]}", fontsize=14)
            self.ax1.plot(self.fft_freqs, self.measured_phase_function, color='b', label=r'Measured $\angle$H')
            self.ax1.set_ylabel("Phase (rad)", fontsize=14)
            self.ax1.yaxis.set_label_coords(-0.1, 0.5)
            self.ax1.tick_params(axis='both', labelsize=11)
            
            self.ax2.plot(self.fft_freqs, self.unwrapped_phase, color='b', label='Manual unwrap')
            self.ax2.plot(self.fft_freqs, self.backup_unwrap, color='green', label='NumPy unwrap')
            self.ax2.plot(self.fft_freqs, self.popt[0]*self.fft_freqs, color='darkmagenta', label=r'Interpolated fit $\angle$H = '+f'({self.popt[0]:.2e}'+r'$\pm$'+f'{self.perr[0]:.2e})f')
            self.ax2.set_ylabel("Phase (rad)", fontsize=14)
            self.ax2.set_xlim(np.min(self.fft_freqs), np.max(self.fft_freqs))
            self.ax2.yaxis.set_label_coords(-0.1, 0.5)
            self.ax2.tick_params(axis='both', labelsize=11)
        
            self.ax3.plot(self.fft_freqs, 20*np.log10(np.abs(self.fft_reference)), 'r', label='Reference FFT')
            self.ax3.plot(self.fft_freqs, 20*np.log10(np.abs(self.fft_sample)), 'b', label='Sample FFT')
            self.ax3.hlines(0, np.min(self.fft_freqs), np.max(self.fft_freqs), 'k')
            self.ax3.set_ylabel("Magnitude (dB)", fontsize=14)
            self.ax3.set_xlabel("Frequency (Hz)", fontsize=14)
            self.ax3.yaxis.set_label_coords(-0.1, 0.5)
            self.ax3.tick_params(axis='both', labelsize=11)
            
            self.ax1.axvspan(np.min(self.freqs_interest), np.max(self.freqs_interest), color='lightblue', label='Parameter Extraction Region', alpha=0.8)
            self.ax1.axvspan(self.fft_freqs[self.unwrap_min_idx], self.fft_freqs[self.unwrap_max_idx], color='green', label='Phase Extrapolation Region', alpha=0.8)
            self.ax2.axvspan(np.min(self.freqs_interest), np.max(self.freqs_interest), color='lightblue', alpha=0.8)
            self.ax3.axvspan(np.min(self.freqs_interest), np.max(self.freqs_interest), color='lightblue', alpha=0.8)
            
            self.ax1.legend(loc='lower left', fontsize=14)
            self.ax2.legend(loc='lower left', fontsize=14)
            self.ax3.legend(loc='lower left', fontsize=14)
        
            self.ax1.grid(True, linestyle='--', alpha=0.6)
            self.ax2.grid(True, linestyle='--', alpha=0.6)
            self.ax3.grid(True, linestyle='--', alpha=0.6)
            self.ax4.grid(True, linestyle='--', alpha=0.6)
            self.ax6.grid(True, linestyle='--', alpha=0.6)
            
            self.ax4.plot(self.time_data, self.sample_signal.y_data, color='b', label='Sample')
            self.ax4.plot(self.time_data, self.reference_signal.y_data, color='r', label='Reference')
            self.ax4.set_ylabel("Amplitude (a.u.)", fontsize=14)
            self.ax4.set_xlabel("Time (s)", fontsize=14)
            self.ax4.set_xlim(np.min(self.time_data), np.max(self.time_data))
            #ax4.yaxis.set_label_coords(-0.1, 0.5)
            self.ax4.tick_params(axis='both', labelsize=11)
            self.ax4.legend(loc='upper right', fontsize=14)
            
            self.mesh = self.ax5.pcolormesh(bins, freqs, 20*np.log10(Pxx), shading='auto', cmap='viridis')
            self.ax5.set_title("Spectrogram of Sample")
            self.ax5.set_ylabel("Frequency (Hz)", fontsize=14)
            self.ax5.set_xlabel("Time (s)", fontsize=14)
            self.fig.colorbar(self.mesh, ax=self.ax5, orientation='horizontal', pad=0.2, fraction=0.05, label="dB")
            self.ax5.tick_params(axis='both', labelsize=11)
            
            plt.show()
        
    def FP_theory(self, ns, ks, l, f):
        # ns, ks are functions of frequency. Will span the region of characterisation
        ns_tilde = ns - (1j*ks)
        term = ((ns_tilde - na_tilde)/(ns_tilde + na_tilde))**2 * np.exp(-2*1j*ns_tilde*2*np.pi*f*l/constants.c)
        return (1-term)**-1
        
    def model_H(self, ns, ks, l, freqs, i, fp_count=None):
        if fp_count is not None:
            ns_tilde = ns - (1j*ks)
            fp = self.sigma_sum_FP(fp_count, ns, ks, l, freqs)
            ratio = (4*na_tilde*ns_tilde) / ((na_tilde + ns_tilde)**2)
            exponent = -1j*l*(ns_tilde - na_tilde)*2*np.pi*freqs / constants.c
            model = ratio * np.exp(exponent) * fp
            phase_model = np.unwrap(np.angle(model))
            popt, pcov = sp.interpolate(phase_model, freqs, self.unwrap_fit_min, self.unwrap_fit_max)
            interpolated_phase = (popt[0]*freqs)[i]
        
        elif fp_count is None:
            ns_tilde = ns - (1j*ks)
            fp = self.FP_theory(ns, ks, l, freqs)
            ratio = (4*na_tilde*ns_tilde) / (na_tilde + ns_tilde)**2
            exponent = -1j*l*(ns_tilde - na_tilde)*2*np.pi*freqs / constants.c
            model = ratio * np.exp(exponent) * fp
            phase_model = np.unwrap(np.angle(model))
            popt, pcov = sp.interpolate(phase_model, freqs, self.unwrap_fit_min, self.unwrap_fit_max)
            interpolated_phase = (popt[0]*freqs)[i]
        
        elif fp_count == 0:
            ns_tilde = ns - (1j*ks)
            fp = self.sigma_sum_FP(fp_count, ns, ks, l, freqs)
            ratio = (4*1*ns_tilde) / ((1 + ns_tilde)**2)
            exponent = -1j*l*(ns_tilde - 1)*2*np.pi*freqs / constants.c
            model = ratio * np.exp(exponent)
            phase_model = np.unwrap(np.angle(model))
            popt, pcov = sp.interpolate(phase_model, freqs, self.unwrap_fit_min, self.unwrap_fit_max)
            interpolated_phase = (popt[0]*freqs)[i]
        
        return model[i], interpolated_phase, phase_model[i]
    
    def calc_spectrogram(self, scale, play=False):
        fs = 1 / ((np.round(self.sample_signal.x_data[1], 3) - np.round(
            self.sample_signal.x_data[0], 3))*1e-12)
        
        x=scale*self.sample_signal.y_data
        if play:
            if fs !=44100:
                sd.play(1000*x, 44100)
        
        plt.figure(figsize=(8,6))
        plt.specgram(x, NFFT=8, Fs=fs, mode='psd', noverlap=7, cmap='hot', vmin=-80, vmax=-60)
        plt.ylabel("Frequency (Hz)", fontsize=14)
        plt.xlabel("Time (s)", fontsize=14)
        plt.colorbar(label='PSD')
        plt.title(f"Sample: {self.sample_signal.name}")
        plt.show()
    
    def sigma_sum_FP(self, number_fp, ns, ks, l, f):
        ns_tilde = ns - (1j*ks)
        total = 0
        equation = (((ns_tilde - na_tilde)/(ns_tilde + na_tilde))**2) * (
            np.exp(-2*1j*ns_tilde*2*np.pi*l*f/constants.c)) # CORRECTION MADE: USED TO BE 'na_tilde' INSTEAD OF 'ns_tilde' and f was missing
        for i in range(0, number_fp+1):
            total += equation**i
        
        return total


def init_params_guess(Pair, f, i, l, unwrappedH, absH):
    # i is the index of the frequency point
    i_map = sp.find_index(Pair.fft_freqs, f)
    ns_guess = 1 - ((constants.c/(2*np.pi*f*l))*unwrappedH[i_map])
    co = constants.c / (2*np.pi*f*l)
    ln_arg1 = 4*ns_guess / ((ns_guess+1)**2)
    ks_guess = co*(np.log(ln_arg1) - np.log(absH[i_map]))
    print(f"Iteration {i}:\nUnwrappedH[{i_map}] = {unwrappedH[i_map]}, frequency point = {f}")
    return ns_guess, ks_guess

def delta_tot(Pair, ns, ks, l, f_array, i, number_fp):
    i_map = sp.find_index(Pair.fft_freqs, f_array[i])
    old_i_arg_frequency = Pair.fft_freqs[np.argmin(Pair.measured_H[i])]
    delta_rho = np.log(np.abs(Pair.measured_H[i_map])) - np.log(np.abs(Pair.model_H(ns, ks, l, f_array, i, fp_count=Pair.number_fp_pulses)[0]))
    #delta_rho = np.log(np.abs(Pair.measured_H[i_map])) - np.log(np.abs(Pair.model_H(ns, ks, l, Pair.fft_freqs, i, fp_count=Pair.number_fp_pulses)[0]))
    delta_phi = Pair.interpolated_unwrapped_H_phase_backup[i_map] - Pair.model_H(ns, ks, l, f_array, i, fp_count=Pair.number_fp_pulses)[1]
    #delta_phi = Pair.interpolated_unwrapped_H_phase_backup[i_map] - Pair.model_H(ns, ks, l, Pair.fft_freqs, i, fp_count=Pair.number_fp_pulses)[1]
    delta_tot = delta_rho**2 + delta_phi**2 # for a single frequency point
    print(SamplePair.fft_freqs[i_map], " ", f_array[i], " ", old_i_arg_frequency)
    return delta_tot

    
def iterate_for_thickness(Pair, l):
    # Pair is an object from the class SampleRefPair
    f_array = Pair.freqs_interest    
    n_sample_real = []
    k_sample_imag = []
    
    unwrappedH = SamplePair.interpolated_unwrapped_H_phase
    absH = np.abs(SamplePair.measured_H)
    
    n_guess_list = []
    k_guess_list = []
    
    for i in range(len(f_array)):

        ns_guess, ks_guess = init_params_guess(Pair, f_array[i], i, l, unwrappedH, absH)[0], init_params_guess(Pair, f_array[i], i, l, unwrappedH, absH)[1]
        n_guess_list.append(ns_guess)
        k_guess_list.append(ks_guess)
        init_guess = [ns_guess, ks_guess]
        temp_fn = lambda t: delta_tot(
            Pair, t[0], t[1], l, f_array, i, Pair.number_fp_pulses)
        
        minimized_params = minimize(temp_fn, init_guess, method='Nelder-Mead')
        print("\n#############################################################")
        print(f"n_guess: {ns_guess} | k_guess = {ks_guess} | (i={i})")
        print(f"Frequency point {f_array[i]:.6e}: n = {minimized_params.x[0]} | k = {minimized_params.x[1]}")
        print("#############################################################")
        
        n_value = minimized_params.x[0]
        k_value = minimized_params.x[1]
        
        n_sample_real.append(n_value)
        k_sample_imag.append(k_value)   
    

        
    return n_sample_real, k_sample_imag, n_guess_list, k_guess_list


if __name__ == "__main__": 
    #test_path = r"C:\Users\matth\OneDrive - University of Birmingham\Y4 Project\Lab_data\rogers_tmm3\focused\241017_1"
    #test_path = r"C:\Users\matth\OneDrive - University of Birmingham\Y4 Project\Lab_data\rogers_rt_duriod_6002\Focused\201017_1"
    #test_path = r"C:\Users\matth\OneDrive - University of Birmingham\Y4 Project\Lab_data\rogers_RO3003\focused\241018"
    #test_path = r"C:\Users\matth\OneDrive - University of Birmingham\Y4 Project\Lab_data\radix_4.6\focused\241023\film_side"
    test_path = r"C:\Users\matth\OneDrive - University of Birmingham\Y4 Project\Lab_data\rogers_rt_duroid_5880LZ\focused\241018_1"
    #test_path = r"C:\Users\matth\OneDrive - University of Birmingham\Y4 Project\Lab_data\radix_2.8HT\focused\241024\film_side"
    #test_path = r"C:\Users\matth\OneDrive - University of Birmingham\Y4 Project\Lab_data\radix_2.8(1)\focused\241018"
    test_batch = sp.MultiSignals(test_path)
    
    #SamplePair = SampleRefPair(test_batch.signals_dic['rogerstmm3_foc_1.txt'], test_batch.signals_dic['reference.txt'], 3.155E-3, unwrap_range_selec=True, plot=True)
    #SamplePair = SampleRefPair(test_batch.signals_dic['radix_4.6_film_1.txt'], test_batch.signals_dic['reference.txt'], 3.010E-3, unwrap_range_selec=True, plot=True)
    SamplePair = SampleRefPair(test_batch.signals_dic['5880LZ_2.txt'], test_batch.signals_dic['reference.txt'], 3.075E-3, unwrap_range_selec=True, plot=True)
    #SamplePair = SampleRefPair(test_batch.signals_dic['2.8(6)_1.txt'], test_batch.signals_dic['reference.txt'], 3.010E-3, unwrap_range_selec=True, plot=True)
    #SamplePair = SampleRefPair(test_batch.signals_dic['6002_1.txt'], test_batch.signals_dic['reference.txt'], 3.07E-3)
    #SamplePair.calc_spectrogram(scale=1000)
    
   
    n_sample_real, k_sample_imag, n_guess_list, k_guess_list = iterate_for_thickness(SamplePair, SamplePair.measured_thickness)
    
    SamplePair.ax6.plot(SamplePair.freqs_interest, n_sample_real, marker='o', markersize=1, color='black', label=r'$n_s$')
    SamplePair.ax6.set_ylabel("Refractive Index", color="blue")
    SamplePair.ax6.tick_params(axis='y')
    SamplePair.ax6.set_ylim(1, np.max(n_sample_real)*1.05)
    
    ax6_right = SamplePair.ax6.twinx()
    ax6_right.plot(SamplePair.freqs_interest, k_sample_imag, color="red", label=r"$\kappa$")
    ax6_right.set_ylabel("Extinction Coefficient")
    ax6_right.set_ylim(0, np.max(k_sample_imag)*1.05)
    ax6_right.legend(loc="upper right", fontsize=14)
    SamplePair.ax6.legend(loc='lower right', fontsize=14)