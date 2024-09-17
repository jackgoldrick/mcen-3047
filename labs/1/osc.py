import numpy as np

def sim(f=500, dt=10e-6, t_f=1, k=1, k_p=.000001):
    t = np.arange(0, t_f, dt)
    noise = np.random.normal()
    signal = (k + k_p * noise) *  np.sin(2 * np.pi * f * t)
    return signal, t

