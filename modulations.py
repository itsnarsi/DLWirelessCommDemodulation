# -*- coding: utf-8 -*-
import numpy as np
import scipy.signal.signaltools as sigtool
import scipy.signal as signal
from numpy.random import sample

import matplotlib.pyplot as plt

def genGrayCode(n):
    if n == 0:
        return ['']
    
    code1 = genGrayCode(n-1)
    code2 = []
    for codeWord in code1:
        code2 = [codeWord] + code2
        
    for i in range(len(code1)):
        code1[i] += '0'
    for i in range(len(code2)):
        code2[i] += '1'
    return code1 + code2    

class fskmod(object):
    def __init__(self, M, freq_sep, Fs, num_samples):
        """Frequency Shift Keying Modulation
        M              ->    Modulation Order
        freq_sep       ->    Frequency Seperation (Hz)
        Fs             ->    Frequency Sampling
                             Fs >= (M-1)*freq_sep
        num_samples    ->    Number of samples per symbol
        """
        
        self.M = M
        self.freq_sep = freq_sep
        self.Fs = Fs
        self.num_samples = num_samples
        
        self.symbol_order = [int(i,2) for i in genGrayCode(np.log2(self.M))]
                
        self.symbol_freq = np.arange(-1 * int(self.M/2), int(self.M/2), 1) * self.freq_sep + self.Fs * 0.5
        
        self.symbol_freq = self.symbol_freq[self.symbol_order]
        
    def modulate(self, data):
        
        data_len = len(data)
        
        t = np.arange(0,float(data_len*self.num_samples)/float(self.Fs),1/float(self.Fs), dtype=np.float32)
        
        m = np.zeros(0).astype(np.float32)
        for sym in data:
            m=np.hstack((m,np.multiply(np.ones(self.num_samples),self.symbol_freq[sym])))
            
        return np.cos(2*np.pi*np.multiply(m,t))
        
c = fskmod(16, 1000, 32000, 100)

X = np.random.randint(0,16, 500)
Y = c.modulate(X)