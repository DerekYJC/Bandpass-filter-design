# -*- coding: utf-8 -*-

#%% Import the module
from scipy.signal import freqz
from scipy.signal import butter, filtfilt
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15, 'axes.linewidth': 2.5, 'axes.titlepad': 20,
                     'axes.labelsize': 'medium',  'axes.labelpad': 20,
                     'figure.dpi': 100, 'savefig.dpi': 100, 
                     'xtick.major.size': 10, 'xtick.major.width': 2,
                     'xtick.major.pad': 15, 'xtick.direction': 'in',
                     'ytick.major.size': 10, 'ytick.major.width': 2,
                     'ytick.major.pad': 15, 'ytick.direction': 'in'})

#%% Now let's see high frequency component
# Filter paramteres
fs     = 3051.76    # sample rate, Hz
freq_range = dict()
freq_range['delta'] = ( 0.1,   4) ## Low-pass should be better
freq_range['theta'] = (   4,   8)
freq_range['alpha'] = (   8,  12)
freq_range['beta']  = (12.5,  30)
freq_range['gamma'] = (  30, 100)

#%% Design Butterworth bandpass filter to the data
def butter_bandpass_v1(lowcut, highcut, fs, order_lowpass, order_highpass):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b_lowpass, a_lowpass = butter(order_lowpass, high, btype='low', analog=False)
    b_highpass, a_highpass = butter(order_highpass, low, btype='high', analog=False)
    return b_lowpass, a_lowpass, b_highpass, a_highpass

def butter_bandpass_v2(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, version, fs, order):
    if version == 'v1':
        b_lowpass, a_lowpass, b_highpass, a_highpass = butter_bandpass_v1(lowcut, 
                                                                          highcut, 
                                                                          fs, 
                                                                          order[0], 
                                                                          order[1])
        y = filtfilt(b_lowpass, a_lowpass, data)
        y = filtfilt(b_highpass, a_highpass, y)
    elif version == 'v2':
        b, a = butter_bandpass_v2(lowcut, highcut, fs, order)
        y = filtfilt(b, a, data)
    return y  

#%% Visualize the frequency response
def frequency_response(wave, order_v1_lowpass, order_v1_highpass, order_v2,
                       freq_range, fs):
    
    (lowcut, highcut) = freq_range[wave]
    thres = (lowcut + highcut) 
    
    # Get the filter coefficients so we can check its frequency response.
    # Bandpass filter design version 1 - highpass + lowpass
    
    b_lowpass, a_lowpass, b_highpass, a_highpass = butter_bandpass_v1(lowcut, 
                                                                      highcut, 
                                                                      fs, 
                                                                      order_v1_lowpass,
                                                                      order_v1_highpass)
    w_lowpass, h_lowpass = freqz(b_lowpass, a_lowpass, worN=50000)
    w_highpass, h_highpass = freqz(b_highpass, a_highpass, worN=50000)
    
    # Bandpass filter design version 2 - bandpass
    
    b, a = butter_bandpass_v2(lowcut, highcut, fs, order_v2)
    w, h = freqz(b, a, worN=50000)
    w_highpass, h_highpass = freqz(b_highpass, a_highpass, worN=50000)
    
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(0.5*fs*w/np.pi, np.abs(h), 'grey', linewidth=3,
            label='version 2: bandpass')
    ax.plot(0.5*fs*w_lowpass[w_lowpass>=thres*np.pi/fs]/np.pi, 
            np.abs(h_lowpass[w_lowpass>=thres*np.pi/fs]), 'b', linewidth=3, 
            label='version 1: lowpass + highpass')
    plt.legend(('version 2: bandpass', 'version 1: lowpass + highpass'), loc='upper right', fontsize=12)
    ax.plot(0.5*fs*w_highpass[w_highpass<=thres*np.pi/fs]/np.pi, 
            np.abs(h_highpass[w_highpass<=thres*np.pi/fs]), 'b', linewidth=3)
    ax.plot(lowcut, 0.5*np.sqrt(2), 'ko')
    ax.axvline(lowcut, color='k', linewidth=2)
    ax.plot(highcut, 0.5*np.sqrt(2), 'ko')
    ax.axvline(highcut, color='k', linewidth=2)
    plt.xlim(0, 1.5*highcut)
    plt.title("Bandpass Filter {} ({} - {} Hz) Frequency Response".format(wave, lowcut, highcut))
    plt.xlabel('Frequency [Hz]')
    plt.grid()
    #plt.xlim([0, 150])    
    plt.savefig("Bandpass Filter {} ({} - {} Hz) Frequency Response".format(wave, int(lowcut), int(highcut)))
    plt.show()

#%% Test the filter design with specific orders 
    
frequency_response('theta', order_v1_lowpass=6, order_v1_highpass=5, order_v2=3, 
                   freq_range=freq_range, fs=fs)   

frequency_response('alpha', order_v1_lowpass=7, order_v1_highpass=7, order_v2=3, 
                   freq_range=freq_range, fs=fs)  

frequency_response('beta', order_v1_lowpass=9, order_v1_highpass=7, order_v2=3, 
                   freq_range=freq_range, fs=fs)

frequency_response('gamma', order_v1_lowpass=13, order_v1_highpass=9, order_v2=5, 
                   freq_range=freq_range, fs=fs)
