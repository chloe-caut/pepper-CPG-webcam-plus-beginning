#frequency analysys for hand detect version 1.


from pandas import read_csv
import numpy as np 
import pandas as pd 
from scipy import signal
import pywt
import matplotlib.pyplot as plt


path_file='/home/nootnoot/Documents/pepper-internship/position.csv'

def load_csv(path):
    df = read_csv(path_file, sep=';', header=None)
    t= df.iloc[:,0]
    X= df.iloc[:,1]
    Y= df.iloc[:,2]
    
    return df,t,X,Y


def fft_plot(Y,t):
    y_detrend = signal.detrend(Y)
    FFT =np.fft.fft(y_detrend)
    new_N=int(len(FFT)/2) 
    f_nat=1
    new_X = np.linspace(10**-12, f_nat/2, new_N, endpoint=True)
    new_Xph=1.0/(new_X)
    FFT_abs=np.abs(FFT)
    plt.plot(new_Xph,2*FFT_abs[0:int(len(FFT)/2.)]/len(new_Xph),color='blue')
    plt.xlabel('Period ($h$)',fontsize=20)
    plt.ylabel('Amplitude',fontsize=20)
    plt.title('(Fast) Fourier Transform Method Algorithm',fontsize=20)
    plt.grid(True)
    plt.xlim(0,200)
    plt.show()
    
    dt=0.1
    L=len(t)/10
    
    t = np.arange(0, L, dt)
    n = y_detrend.size
    freq = np.fft.fftfreq(n, d=dt)
    
    plt.subplot(211)
    plt.plot(t,y_detrend, 'r', label='movements landmark 0')
    plt.ylabel('y')
    plt.xlabel('n')
    plt.grid(True)

    
    plt.subplot(212)
    plt.plot(freq, FFT.real, label="real")
    plt.plot(freq, FFT.imag, label="imag")
    plt.legend()
    plt.show()

  

def main():
    df,t,X,Y = load_csv(path_file)
    fft_plot(Y,t)
    
    
if __name__ == '__main__':
    main()