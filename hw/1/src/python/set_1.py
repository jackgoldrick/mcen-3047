import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import torch as tc
import pandas as pd
import os
import csv
class Set1:
    class Problem1:
        def record_audio(duration=7, fs=44100):
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
            sd.wait()
            return recording


        def get_num_samples(fs=44100, duration=7):
            return fs * duration

        def get_time_vector(self, fs=44100, duration=7):
            return np.linspace(0, duration, self.get_num_samples(fs, duration))

        def plot_audio_voltage(self, recording):
            fig = plt.figure()
            plt.title('Voltage of Audio Sample vs Time')
            plt.xlabel('Time (s)')
            plt.ylabel('Voltage (V)')
            plt.plot(self.get_time_vector(), recording)
            plt.show()
            
            
        def plot_audio_fft(recording, fs=44100):
            fig = plt.figure()
            plt.title('FFT of Audio Sample')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude')
            plt.plot(np.fft.fftfreq(len(recording), 1/fs), np.abs(np.fft.fft(recording)))
            plt.show()
            
        def run(self):
            recording = self.record_audio()
            self.plot_audio_voltage(recording)
            # plot_audio_fft(recording)
            
    class Problem2:
        def compute_function(n=5,duration=6, fs=44100):
            t = tc.Tensor(np.linspace(start=0, stop=duration, num=fs*duration))
            
            return tc.Tensor([(2 * np.cos(np.pi * i) * np.sin(np.pi * i * t )) / (np.pi * i) for i in range(n)]), t
            
        def plot_n_fourier_terms(self, n=5, duration=6, fs=44100):
            f, t = self.compute_function(n=n, duration=duration, fs=fs)
            fig = plt.figure()
            plt.title(f'First {n} Fourier Series Terms of Function f(t)')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.plot(t,f)
            plt.show()

        
        
        def run(self):
            self.plot_n_fourier_terms(n=5)
            self.plot_n_fourier_terms(n=20)
            self.plot_n_fourier_terms(n=400)
    
    class Problem3:
        """ 1. Takes a csv file, tone, that contains audio data. 
                i. The first column is the time vector
                ii. The second column is the audio data
            2. Executes a FFT on the audio
            3. plots the timeseries data
            4. plots the FFT data
            5. Saves the plots as images
            6. calculates the frequency of the tone from the least significant bit of the DAQ 
               and voltage scale"""
        def run(self):
            pass
        def read_csv(self, file_path='tone.csv', home=True):
            if home:
                file_path = os.path.join('../../data/', file_path)
                
                
            df = pd.read_csv(file_path)
            
            df.columns = ['time', 'audio']
            return df
        
        def plot_audio(self, df):
            fig = plt.figure()
            plt.title('Voltage of Audio Sample vs Time')
            plt.xlabel('Time (s)')
            plt.ylabel('Voltage (V)')
            plt.plot(df['time'], df['audio'])
            plt.show()
            
        def plot_audio_fft(self, df):
            fig = plt.figure()
            plt.title('FFT of Audio Sample')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude')
            plt.plot(np.fft.fftfreq(len(df['audio']), 1/44100), np.abs(np.fft.fft(df['audio'])))
            plt.show()
            
        def save_plots(self, df, file_path='results', home=True):
            if home:
                file_path = os.path.join('../../', file_path)
                
            self.plot_audio(df)
            self.plot_audio_fft(df)
            plt.savefig(file_path + '/problem3.png')

        def calculate_frequency(self, df):
            pass            
            
        