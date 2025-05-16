import datetime
import os

import dv_processing as dv
import cv2 as cv
import argparse
from datetime import timedelta
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.signal import find_peaks
from scipy.signal import find_peaks
from scipy.signal import butter, lfilter
import time
import tkinter as tk
from tkinter import ttk, messagebox

import statistics
filename = "test"
root = tk.Tk()
root.title("Patient Data Input")
root.geometry("300x250")

# Labels and Entry Fields
tk.Label(root, text="Patient Number:").pack()
entry_patient = tk.Entry(root)
entry_patient.pack()


tk.Label(root, text="Distance (meters):").pack()
entry_distance = tk.Entry(root)
entry_distance.pack()

# Dropdown for Lighting Conditions
tk.Label(root, text="Lighting Conditions:").pack()
lighting_var = tk.StringVar()
lighting_dropdown = ttk.Combobox(root, textvariable=lighting_var, state="readonly")
lighting_dropdown['values'] = ("Good natural light", "Dark natural light", "fluorescent tube light", "led light")
lighting_dropdown.pack()

tk.Label(root, text="Reading Number:").pack()
entry_reading = tk.Entry(root)
entry_reading.pack()




def get_info():
    global filename
    # Create main window

    patient_number = entry_patient.get().strip()
    reading_number = entry_reading.get().strip()
    distance = entry_distance.get().strip()
    lighting = lighting_var.get()
    
    if not (patient_number and reading_number and distance and lighting):
        messagebox.showerror("Input Error", "All fields must be filled!")
        return
    
    filename = f"patient_{patient_number}_{distance}m_{lighting.replace(' ', '_')}_reading_{reading_number}"
    root.destroy()

# Submit Button
tk.Button(root, text="Generate Filename", command=get_info).pack(pady=10)
root.mainloop()

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

peakTimes = []
peaks = []
numEvents = []
polarities = []
positiveEvents = []
negativeEvents = []
x = []



parser = argparse.ArgumentParser(description='Show a preview of an iniVation event camera input.')
# Open the camera
camera = dv.io.CameraCapture()


# Event only configuration
config = dv.io.MonoCameraWriter.EventOnlyConfig("DVXplorer_sample", camera.getEventResolution())

# # Create the writer instance, it will only have a single event output stream.
writer = dv.io.MonoCameraWriter(os.path.join("recordings","042025",filename + ".aedat4"), config)

filter_chain = dv.EventFilterChain()
# Filter refractory period
filter_chain.addFilter(dv.RefractoryPeriodFilter(camera.getEventResolution()))
# Remove noise
filter_chain.addFilter(dv.noise.BackgroundActivityNoiseFilter(camera.getEventResolution(), backgroundActivityDuration=timedelta(milliseconds=1)))

          
# Apply filter chain and show preview
def slice_events(events):
    x.append(events.timestamps()[0])
    polarities.append(np.mean(events.polarities()))
    unique, counts = np.unique(events.polarities(), return_counts=True)
    writer.writeEvents(events)
    # Pass data to filter
    filter_chain.accept(events)
    # apply filtering
    filtered_events = filter_chain.generateEvents()
    numEvents.append(filtered_events.size())


def heartrateFilter(dict):
    if 1 < dict['frequency'] < 3:
        return True
    else:
        return False
    

def breathingFilter(dict):
    if dict['frequency'] < 1:
        return True
    else:
        return False

def extractMaximum(fft, freqs, dist, prom):
    peakdict = []

    peaks, heights = find_peaks(np.abs(fft), distance=dist, prominence=prom)
    for peak in peaks:
        peakdict.append({'frequency': freqs[peak], 'value' : np.abs(fft)[peak]})

    #breathingfreqs = filter(breathingFilter, peakdict)
    heartRatefreqs = filter(heartrateFilter, peakdict)
    try:
        possibleHeartRate = next(heartRatefreqs)
    except StopIteration:
        return {'frequency': 0, 'value': 0}
    # possibleBreathingRate = next(breathingfreqs)
    #print(max(heartRatefreqs, 'value'))
    for heartRate in heartRatefreqs:
        if heartRate['value'] > possibleHeartRate['value']:
            possibleHeartRate = heartRate

    return  possibleHeartRate

def extractMaximumRR(fft, freqs, dist, prom):
    peakdict = []

    peaks, heights = find_peaks(np.abs(fft), distance=dist, prominence=prom)
    for peak in peaks:
        peakdict.append({'frequency': freqs[peak], 'value' : np.abs(fft)[peak]})

    #breathingfreqs = filter(breathingFilter, peakdict)
    respRatefreqs = filter(breathingFilter, peakdict)
    try:
        possiblerespRate = next(respRatefreqs)
    except StopIteration:
        return {'frequency': 0, 'value': 0}
    # possibleBreathingRate = next(breathingfreqs)
    #print(max(heartRatefreqs, 'value'))
    for respRate in respRatefreqs:
        if respRate['value'] > possiblerespRate['value']:
            possiblerespRate = respRate

    return  possiblerespRate


def getPeakDistance(values, timeArray):
    peakdict = []
    peaks, heights = find_peaks(values, distance=10, prominence=1.5)
    print(peaks)
    for peak in peaks:
        print(timeArray[peak])
        print(values[peak])
        peakdict.append({'time': timeArray[peak], 'value' : values[peak]})
    return peakdict

def getAveragePeriod(peakTimes):
    accTimes = 0
    avg = 0
    for i in range(len(peakTimes)-1):
        if avg > 0:
            if (peakTimes[i + 1] - peakTimes[i]) > avg * 0.5:
                accTimes += peakTimes[i + 1] - peakTimes[i]
                avg = accTimes/(i+1)
        else:
            accTimes += peakTimes[i + 1] - peakTimes[i]
            avg = accTimes/(i+1)
    avg = accTimes/(len(peakTimes)-2)
    return avg



def getMedianPeriod(peakTimes):
    p = []
    for i in range(len(peakTimes)-1):
        p.append(peakTimes[i + 1] - peakTimes[i])
    median = statistics.median(p)

    return median


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def filters(numEvent, bpLimit1, bpLimit2, sgwindow, sgorder):
    numEventsRR = scipy.signal.savgol_filter(numEvent, sgwindow, sgorder, mode='interp')
    #numEventsRR = butter_bandpass_filter(numEvent,0.13, 0.7, 100, 2)

    hr = numEvent

    hr = butter_bandpass_filter(hr,bpLimit1, bpLimit2, 100, 5)
    hr = scipy.signal.savgol_filter(hr, 11, 5, mode='interp')


    return numEventsRR, hr


def calculateRespRate(values, times, numPeaks=10, min_distance=100):
    values = np.array(values)
    times = np.array(times)
    #analytic_signal = hilbert(values)
    #values = np.abs(analytic_signal)
    height_threshold = np.mean(values) + 0.3 * np.std(values)
    print("height threshold: ", height_threshold)
    # Estimate initial prominence
    prominence = np.std(values) * 0.4

    # Try detecting peaks 
    peaks, properties = find_peaks(values, distance=min_distance, prominence=prominence,height=height_threshold)

    # If too few peaks, reduce prominence gradually
    if len(peaks) < numPeaks:
        print("Too few peaks detected, adjusting prominence...")
        for scale in np.linspace(0.5, 0.1, 5):
            peaks, properties = find_peaks(values, distance=min_distance, prominence=np.std(values) * scale, height=height_threshold)
            if len(peaks) >= numPeaks:
                break

    # If still too few peaks, fallback to basic detection
    if len(peaks) < numPeaks:
        peaks, properties = find_peaks(values, distance=min_distance)

    peak_times = times[peaks]
    peak_values = values[peaks]

    # Require at least 2 peaks to calculate period
    if len(peak_times) < 2:
        return 0, [], []

    # Use median period for robustness
    periods = np.diff(peak_times)
    median_period = np.median(periods)
    resp_rate = (1 / median_period) * 60  # Convert to breaths per minute

    return resp_rate, list(peak_times), list(peak_values)


def calculateHeartRate(values, times, numPeaks=10, min_distance=50):
    values = np.array(values)
    times = np.array(times)
    height_threshold = np.mean(values) + 0.3 * np.std(values)
    prominence = np.std(values) * 0.4

    peaks, properties = find_peaks(values, distance=min_distance, prominence=prominence, height=height_threshold)

    if len(peaks) < numPeaks:
        for scale in np.linspace(0.5, 0.1, 5):
            peaks, properties = find_peaks(values, distance=min_distance, prominence=np.std(values) * scale, height=height_threshold)
            if len(peaks) >= numPeaks:
                break

    if len(peaks) < numPeaks:
        peaks, properties = find_peaks(values, distance=min_distance)

    peak_times = times[peaks]
    peak_values = values[peaks]

    if len(peak_times) < 2:
        return 0, [], []

    periods = np.diff(peak_times)
    trimmed_period = trim_mean(periods, proportiontocut=0.1)
    heart_rate = (1 / trimmed_period) * 60  # Convert to bpm

    return heart_rate, list(peak_times), list(peak_values)



def vital_parameters(y, t, visualize, title):
    start = t[0]
    for i in range(0, len(t)):    
        t[i] = (t[i] - start) / 1000000

    numEventsRR1, hr = filters(y, 0.8, 3, 25, 2)
    

    rr, peakTimes0, peaks0 = calculateRespRate(numEventsRR1, t, 7, 300)
    #hr_bpm, peakTimes1, peaks1 = calculateHeartRate(hr, t, 5, 50)
    hr_bpm, peakTimes1, peaks1 = calculateHeartRate(hr, t)

    sr = 100
    F_hr = np.fft.fft(hr)
    N = len(F_hr)
    n = np.arange(N)
    T = N/sr
    freq = n/T
    f = extractMaximum(F_hr, freq, 10, 1.5)
    hr_alt = f['frequency'] * 60

    F_rr = np.fft.fft(numEventsRR1)
    N_rr = len(F_rr)
    n_rr = np.arange(N_rr)
    T_rr = N_rr/sr
    freq_rr = n_rr/T_rr
    f_rr = extractMaximumRR(F_rr, freq_rr, 1, 1)
    rr_alt = f_rr['frequency'] * 60

    if visualize:
        #plt.title(title)
        plt.subplot(221, title="Filtered Respiration Signal" + str(rr) + " bpm")
        plt.plot(t, numEventsRR1, color="red")
        plt.xlabel('seconds')
        plt.ylabel('Events')
        plt.plot(peakTimes0, peaks0, marker="o", ls="", label="peaks")

        plt.subplot(222, title="FFT of Respiration Signal" + str(f_rr['frequency'] * 30) + " bpm")
        plt.stem(freq_rr, np.abs(F_rr), 'b', markerfmt=" ", basefmt="-b")
        plt.plot(f_rr['frequency'], f_rr['value'], marker="o", ls="", label="detected")
        plt.xlabel('Freq (Hz)')
        plt.ylabel('|FFT|')
        plt.xlim(0, 1)

        plt.subplot(223, title="Filtered Heart Signal" + str(hr_bpm) + " bpm")
        plt.plot(t, hr, color="red")
        plt.xlabel('seconds')
        plt.ylabel('Events')
        plt.plot(peakTimes1, peaks1, marker="o", ls="", label="peaks")

        plt.subplot(224, title="FFT of Heart Signal" + str(f['frequency'] * 60) + " bpm")
        plt.stem(freq, np.abs(F_hr), 'b', markerfmt=" ", basefmt="-b")
        plt.plot(f['frequency'], f['value'], marker="o", ls="", label="detected")
        plt.xlabel('Freq (Hz)')
        plt.ylabel('|FFT|')
        plt.xlim(0, 3)

        plt.show()

    return rr, hr_bpm, rr_alt, hr_alt

# Create an event slicer, this will only be used events only camera
slicer = dv.EventStreamSlicer()
slicer.doEveryTimeInterval(datetime.timedelta(milliseconds=10), slice_events)


def main():
    global filename
    global numEvents
    global polarities
    global x


    t_end = time.time() + 30 * 1
    while time.time() < t_end:
        # Get events
        events = camera.getNextEventBatch()
        print(t_end - time.time())
        # If no events arrived yet, continue reading
        if events is not None:
            slicer.accept(events)



    numEvents = np.array(numEvents)
    numEventsSG = scipy.signal.savgol_filter(numEvents, 25, 2, mode='interp')
    numEventsRR = scipy.signal.savgol_filter(numEvents, 55, 2, mode='interp')
    numEventsHr = scipy.signal.savgol_filter(numEvents, 25, 2, mode='interp')
    hr = butter_bandpass_filter(numEventsHr,1, 3, 100, 5)

    x = np.array(x)
    polarities = np.array(polarities)
    start = x[0]
    for i in range(0, x.size):    
        x[i] = (x[i] - start) / 1000

    sr = 1/((x[1] - x[0])/1000)
    #print(sr)
    F = np.fft.fft(numEventsSG)
    F_hr = np.fft.fft(hr)
    N = len(F)
    n = np.arange(N)
    T = N/sr
    freq = n/T

    peakTimes, peaks, rr = calculateRespRate(numEventsRR, x, 7)

    print('resp rate is ' + str(rr))
    print('heart rate is ' + str(extractMaximum(F_hr, freq)))
    f = open(os.path.join("recordings","042025", filename + ".txt"), "a")

    for i in range(0,len(numEvents)):
        line = str(x[i]) + ',' + str(numEvents[i]) + '\n'
        f.write(line)
    f.close()


    plt.subplot(221, title= "Number of Events")
    plt.plot(x, numEventsRR, color="red")
    plt.plot(peakTimes, peaks, marker="o", ls="", label="peaks")

    plt.subplot(222, title="FFT von F_hr")
    plt.stem(freq, np.abs(F_hr), 'b', \
            markerfmt=" ", basefmt="-b")
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')
    plt.xlim(0, 3)

    plt.subplot(223, title= "Number of Events HR")
    plt.plot(x, hr, color="red")
    #plt.plot(peakTimes, peaks, marker="o", ls="", label="peaks")

    plt.subplot(224, title= "Raw")
    plt.plot(x, numEvents, color="red")


    plt.show()


if __name__ == '__main__':
    main()

