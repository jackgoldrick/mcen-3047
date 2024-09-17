from matplotlib.patches import Circle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from osc import sim
import os

def read_csv(file_path='osc.csv', home=True):
    if home:
        file_path = os.path.join('./data/good/', file_path)
        
        
    df = pd.read_csv(file_path)
    
    df.columns = ['X', 'Y']
    return df



def plot_freq_pairs():
    
    freq_osc = [100, 200, 300, 400, 500, 600, 700]
    freq_daq = [100, 200, 300, 400, 500, 400, 300]
    circle = Circle((500, 500), radius=10.5, edgecolor='blue', facecolor='none')


    
    figure, ax = plt.subplots()
    plt.title('Measured Frequency Pairs of DAQ and Oscilliscope')
    plt.xlabel('Frequency Oscilliscope (Hz)')
    plt.ylabel('Frequency DAQ (Hz)')
    plt.plot(freq_osc, freq_daq, 'ro')
    ax.add_patch(circle)
    
    ax.annotate('Nyquist Frequency',
                xy=(500, 500), xycoords='data',
                xytext=(0.36, 0.68), textcoords='axes fraction',
                arrowprops=dict(facecolor='black', shrink=0.05),
                horizontalalignment='right', verticalalignment='top')
    
    plt.show()
    

    
def plot_alias_with_fft():
    df_t = read_csv(file_path='aliasing_700_10k10k.csv')
    df_fft = read_csv(file_path='aliasing_700_10k10k_fft.csv')
    
    figure, ax = plt.subplots(2)
    figure.suptitle('Clean sample of DAQ')
    ax[0].set(ylabel='Voltage (V)', xlabel='Time (s)')
    ax[0].set_xlim(0.05, 0.12)
    ax[1].set(ylabel='Amplitude (dB)', xlabel='Frequency (Hz)')
    ax[0].plot(df_t['X'], df_t['Y'])
    ax[1].plot(df_fft['X'], df_fft['Y'])
    plt.show()
    
    
def plot_full_alias_with_fft():
    
    
    # df_t = read_csv(file_path='aliasing_700_500.csv')
    df_fft = read_csv(file_path='aliasing_700_500_fft.csv')
    figure, ax = plt.subplots(2)
    figure.suptitle('Aliased sample of DAQ')
    ax[0].set(ylabel='Voltage (V)', xlabel='Time (s)')
    ax[0].set_xlim(0.05, 0.12)
    ax[1].set(ylabel='Amplitude (dB)', xlabel='Frequency (Hz)')
    ax[1].set_xlim(200, 800)
    ax[0].plot(df_t['X'], df_t['Y'])
    ax[1].plot(df_fft['X'], df_fft['Y'])
    plt.show()
    
        
plot_freq_pairs()
plot_alias_with_fft()
plot_full_alias_with_fft()