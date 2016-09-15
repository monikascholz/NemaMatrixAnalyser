# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 17:05:04 2016
Nemamatrix reader.
@author: monika
"""
import numpy as np
import matplotlib.pylab as plt
import style
import custom_plots as cpl
from scipy.signal import butter, lfilter
from detect_peaks import detect_peaks

def readerFile(fname):
    """Read nemamatrix raw file."""
    data = np.loadtxt(fname, skiprows = 2, comments = '#')
    # read parameters and convert to dictionary
    params =  eval(open(fname).readline().rstrip().replace('true', 'True').replace('\x00', ''))
    return params, data
    
def convertData(data):
    """Convert digital signal to units (muV)."""
    conversionFactor = 0.75/2**24 #750 mVpp at 24 bit Digital signal
    return conversionFactor*data*10**6
    
def OneHzHighpass(data, fs):
    """According to Nemamatrix manual, this is a  1 Hz second-order Butterworth Highpass.
    fs is the sampling rate in Hz.
    Wn is cuttoff frequency in Hz relative to Nyquist frequency. Here 1 Hz/(0.5*sampling rate)."""
    b,a = butter(N = 2, Wn = 2.00/fs, btype='high', analog=False, output='ba')
    return lfilter(b, a, data)[fs*2:]
    
def Notch60Hz(data, fs):
    """60Hz Notch filter with second-order bandstop filters with corners at
    59Hz and 61Hz according to nemamatrix documentation."""
    b,a = butter(N = 2, Wn = [2.*59/fs, 2.0*61/fs], btype='bandstop', output='ba')
    return lfilter(b, a, data)
    
def LowPass(data, fs, lowfreq):
    """lowpass filter to smooth signal.lowfreq is cutoff in Hz"""
    b,a = butter(N = 2, Wn = lowfreq/fs/2., btype='low', analog=False, output='ba')
    return lfilter(b, a, data)#[fs*2:]

def findExtrema(data, heightCutoff, minimumPeakDistance, typ='Max'):
    """Find regional maxima/minima"""
    if typ=='Max':
        return detect_peaks(data, edge='rising',  mph=heightCutoff, mpd= minimumPeakDistance)
    if typ=='Min':
        return detect_peaks(data, edge='falling',  valley = True, mph=heightCutoff, mpd= minimumPeakDistance)

def findCorrespondingExtrema(extrema, data, lookahead, typ):
    """find corresponding max/min pairs"""
    corrExtrema = np.zeros(len(extrema))
    if typ == 'Max':
        for pindex, p in enumerate(extrema):
            corrExtrema[pindex] = np.argmax(data[p:p+lookahead])+p
    if typ == 'Min':
        for pindex, p in enumerate(extrema):
            corrExtrema[pindex] = np.argmin(data[p:lookahead+p])+p
    return corrExtrema
    
def spikeTriggeredAverage(spikeLocs, data, memorySize):  
    '''average signal around detected spikes.'''
    
    dataSpikes = np.zeros((len(spikeLocs), 2*memorySize))
    spikeLocs = spikeLocs[spikeLocs > memorySize]
    spikeLocs = spikeLocs[spikeLocs < len(data)-2*memorySize]
    
    for pindex, p in enumerate(spikeLocs):
        dataSpikes[pindex] = np.array(data[p-memorySize:p + memorySize])
    return dataSpikes
    

def nemamatrixReader(fname, outputImages, outputData):
    """reads nemamatrix files and performs some analysis."""
    # read file
    pars, data = readerFile(filename)
    samplingRate = float(pars['amp_sample_rate_hz'])
    
    # parameters
    memorySize = 200
    minimumPeakDistance = 80 # distance between minima or maxima (should correspond to maximal frequency of 5Hz = 100 frames)
    heightCutoffMax, heightCutoffMin = 30, 50
    correspondingPeakDistance = 0.3*samplingRate
    # convert data to microVolts
    data = convertData(data)
    # invert Data if not headfirst
    if pars['orientation'][2]=='tf':
        data = -data
        
    # use highpass and notch filters as described in nemamatrix documentation
    data = OneHzHighpass(data, samplingRate)
    data = Notch60Hz(data, samplingRate)
    
    #  spike triggered averaging for minima or maxima
    xPeaksMax = findExtrema(data, heightCutoff = heightCutoffMax, minimumPeakDistance = minimumPeakDistance, typ='Max')
    
    xPeaksMin = findExtrema(data, heightCutoff = heightCutoffMin, minimumPeakDistance = minimumPeakDistance, typ='Min')
    # plot data
#    plt.plot(data)
#    plt.plot(xPeaksMax, data[xPeaksMax], 'ro')
#    plt.plot(xPeaksMin, data[xPeaksMin], 'ko')
    #
    types = {0: 'Max', 1: 'Min'}
    labels = ['Peak-Trough time (ms)', 'Trough-Peak time (ms)']
    summaryData = {}
    fullData = {}
    fig = plt.figure(13,figsize=(7,10.5),dpi=100,edgecolor='k',facecolor='w') 
    for index, xPeaks in enumerate([xPeaksMax, xPeaksMin]):
        # plot minima or maxima triggered signal
        ax = plt.subplot(3,2,index+1)
        STA = spikeTriggeredAverage(spikeLocs = xPeaks, data = data, memorySize = memorySize)
        STAmean = np.average(STA, axis = 0)
        STAtime = np.arange(2*memorySize)/1.0/samplingRate*1000 # in ms
        STAstdev = np.std(STA, axis=0)/np.sqrt(len(STA))
        
        for line in STA:
            plt.plot(STAtime, line, color = style.UCgray[0], alpha = 0.01)
        plt.plot(STAtime , STAmean, lw= 1.5, color = 'orange')
        plt.fill_between(STAtime ,STAmean-STAstdev, STAmean+STAstdev, alpha = 0.5, color='orange')
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (uV)')
        plt.ylim([-150, 100])
        plt.xticks([0,300,600,900])
        plt.tight_layout()
        cpl.clean_axes(ax)
        
        summaryData['Peakheight{}'.format(types[index])] = [np.max(STAmean), STAstdev[np.argmax(STAmean)], len(STA)]
        fullData['STA{}'.format(types[index])] = [STAtime ,STAmean, STAstdev]
        # plot distribution of pumping rates
        ax = plt.subplot(3,2,index+3)
        frequencies = samplingRate/np.diff(xPeaks)
        bins = np.arange(0,6,0.1)
        x,y = cpl.histogram(frequencies, bins, normed=True)
        plt.step(x,y, where='pre', color = style.UCblue[0], lw=1, )
        cpl.fill_between_steps(ax, x,y, 0, step_where='pre', color = style.UCblue[0], alpha=0.5)
        plt.ylabel('Distribution of \n pumping rates')
        plt.xlabel('Pumping rate (s$^{-1}$)')
        plt.ylim([0,0.75])
        plt.yticks([0,0.25,0.5,0.75, 1.0])
        plt.tight_layout()
        cpl.clean_axes(ax)
        summaryData['PumpRate{}'.format(types[index])] = [np.mean(frequencies), np.std(frequencies), len(frequencies)]
        fullData['Rates{}'.format(types[index])] = frequencies
        
        ax = plt.subplot(3,2,5+index)
        corrPeaks = findCorrespondingExtrema(xPeaks, data, lookahead = correspondingPeakDistance, typ = types[1-index])
        frequencies = (corrPeaks-xPeaks)/samplingRate*1000 # in ms
        bins = np.arange(0,correspondingPeakDistance/samplingRate*1000,5)
        x,y = cpl.histogram(frequencies, bins, normed=True)
        plt.step(x,y, where='pre', color = style.UCblue[0], lw=1, )
        cpl.fill_between_steps(ax, x,y, 0, step_where='pre', color = style.UCblue[0], alpha=0.5)
        plt.ylabel('Distribution of  \n intra-pump spacing')
        plt.xlabel(labels[index])
        plt.yticks([])
        plt.tight_layout()
        cpl.clean_axes(ax)
        summaryData['Distance{}{}'.format(types[index], types[1-index])] = [np.mean(frequencies), np.std(frequencies), len(frequencies)]
        fullData['Distance{}{}'.format(types[index], types[1-index])] = frequencies
    plt.savefig(outputImages)
    return summaryData, fullData

filename = "/home/monika/Desktop/workspace spyder/ElectroPhysiology/data/daf16_07092016/nema_04123_2016-09-07_16-48-00.txt"
nemamatrixReader(filename)