import datetime

import dv_processing as dv
import cv2 as cv
import argparse
from datetime import timedelta
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.signal import find_peaks
from scipy.signal import butter, lfilter
from scipy.signal import hilbert
from scipy.stats import trim_mean
import statistics
import os
import tkinter as tk
from tkinter import simpledialog

import heartpy as hp


from collections import defaultdict

import gc
import csv

current_distance = ""
peakTimes = []
peaks = []
numEvents = []
polarities = []
positiveEvents = []
negativeEvents = []
x = []
visualize = True
visualizer = None

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

def subdirectory(path):
    rowlist = []
    for entry in os.scandir(path) :
        if entry.is_dir():
            subdirectory(os.path.join(path, entry.name))
        if entry.is_file() and ".aedat4" in entry.name:
            filename = entry.name
            print(filename)
            rowlist.append(path + filename)
    return rowlist


def predict(events):
    global peakTimes, peaks, numEvents, polarities, positiveEvents, negativeEvents, x
    if events.size() == 0:
        return
    x.append(events.timestamps()[0])
    numEvents.append(events.size())
    if visualize and visualizer is not None:
        #visualizer.accept(events)
        frame = visualizer.generateImage(events)
        cv.imshow("Preview", frame)
        cv.waitKey(1)



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


def getMedianPeriod(peakTimes):
    p = []
    for i in range(len(peakTimes)-1):
        p.append(peakTimes[i + 1] - peakTimes[i])
    median = statistics.median(p)

    return median



def calculateRespRate(values, times, numPeaks=10, min_distance=100):
    values = np.array(values)
    times = np.array(times)
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

def calculateHeartRateHeartPy(values, times):
    values = np.array(values)
    times = np.array(times)
    
    try:
        wd, m = hp.process(values, sample_rate=100)
        return m['bpm'], times[wd['peaklist']], values[wd['peaklist']]
    except Exception as e:
        print("HeartPy failed:", e)
        return 0, [], []


def extractMaximum(fft, freqs, dist, prom):
    peakdict = []

    peaks, heights = find_peaks(np.abs(fft), distance=dist, prominence=prom)
    for peak in peaks:
        peakdict.append({'frequency': freqs[peak], 'value' : np.abs(fft)[peak]})

    heartRatefreqs = filter(heartrateFilter, peakdict)
    try:
        possibleHeartRate = next(heartRatefreqs)
    except StopIteration:
        return {'frequency': 0, 'value': 0}
    for heartRate in heartRatefreqs:
        if heartRate['value'] > possibleHeartRate['value']:
            possibleHeartRate = heartRate

    return  possibleHeartRate

def extractMaximumRR(fft, freqs, dist, prom):
    peakdict = []

    peaks, heights = find_peaks(np.abs(fft), distance=dist, prominence=prom)
    for peak in peaks:
        peakdict.append({'frequency': freqs[peak], 'value' : np.abs(fft)[peak]})

    heartRatefreqs = filter(breathingFilter, peakdict)
    try:
        possibleHeartRate = next(heartRatefreqs)
    except StopIteration:
        return {'frequency': 0, 'value': 0}
    for heartRate in heartRatefreqs:
        if heartRate['value'] > possibleHeartRate['value']:
            possibleHeartRate = heartRate


    return  possibleHeartRate



def vital_parameters(y, t, visualize, title):
    start = t[0]
    for i in range(0, len(t)):    
        t[i] = (t[i] - start) / 1000000

    numEventsRR1, hr = filters(y, 0.8, 3, 25, 2)
    

    rr, peakTimes0, peaks0 = calculateRespRate(numEventsRR1, t, 7, 300)
    #hr_bpm, peakTimes1, peaks1 = calculateHeartRate(hr, t, 5, 50)
    hr_bpm, peakTimes1, peaks1 = calculateHeartRateHeartPy(hr, t)

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



def ask_signal_quality():
    quality = {"value": None}

    def set_quality(q):
        quality["value"] = q
        root.quit()

    root = tk.Tk()
    root.title("Signal Quality")
    root.geometry("300x100")

    tk.Label(root, text="Rate the signal quality:").pack(pady=5)
    tk.Button(root, text="Good", width=10, command=lambda: set_quality("Good")).pack(side=tk.LEFT, padx=10, pady=10)
    tk.Button(root, text="Medium", width=10, command=lambda: set_quality("Medium")).pack(side=tk.LEFT, padx=10, pady=10)
    tk.Button(root, text="Bad", width=10, command=lambda: set_quality("Bad")).pack(side=tk.LEFT, padx=10, pady=10)

    root.mainloop()
    root.destroy()
    return quality["value"]




def main():
    files = subdirectory("recordings\\042025\\042025\\")
   
    global peakTimes, peaks, numEvents, polarities, positiveEvents, negativeEvents, x

    for file in files:
        lastTs = 0
        peakTimes = []
        peaks = []
        numEvents = []
        polarities = []
        positiveEvents = []
        negativeEvents = []
        x = []
        global visualizer
        #file = "recordings\\042025\\042025\\patient_1_0,5m_Good_natural_light_reading_1.aedat4"

        #file = "recordings\\042025\\042025\\patient_1_1m_Good_natural_light_reading_2.aedat4"
        patient_number = file.split("\\")[-1].split("_")[1]
        reading_number = file.split("\\")[-1].split("_")[-1].split(".")[0]
        distance = file.split("\\")[-1].split("_")[2]

        reader = dv.io.MonoCameraRecording(file)
        slicer = dv.EventStreamSlicer()
        slicer.doEveryTimeInterval(datetime.timedelta(milliseconds=10), predict)

        filter_chain = dv.EventFilterChain()

        print("Recording started")
        print("Recording filename: ", file)
        # Run the loop while camera is still connected
        while reader.isRunning():
            # Read batch of events
            events = reader.getNextEventBatch()
            if events is not None:

                if events.timestamps()[-1] > lastTs:
                    filter_chain.accept(events)
                    filtered_events = filter_chain.generateEvents()
                    if filtered_events.size() > 0:
                        slicer.accept(filtered_events)
                    lastTs = events.timestamps()[-1]
                #predict(events)
        if len(numEvents) > 0:

            current_distance = distance  # Set global variable
            resp_rate, heart_rate, resp_rate_alt, heart_rate_alt = vital_parameters(numEvents, x, True, "Patient: " + patient_number + "Reading :" + reading_number)
            signal_quality = ""
            #signal_quality = ask_signal_quality()
            print(f"Patient {patient_number}, Reading {reading_number}: Respiration Rate = {resp_rate:.2f} bpm, Heart Rate = {heart_rate:.2f} bpm, alternative Resp Rate = {resp_rate_alt:.2f} bpm, alternative Heart Rate = {heart_rate_alt:.2f} bpm")
            output_csv = "output29042025.csv"
            file_exists = os.path.isfile(output_csv)

            with open(output_csv, "a", newline="") as csvfile:
                fieldnames = ['Patient', 'Reading', 'Distance', 'Respiration Rate (bpm)', 'Heart Rate (bpm)', 'Alt Respiration Rate (bpm)', 'Alt Heart Rate (bpm)']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                if not file_exists:
                    writer.writeheader()  # write header only once

                writer.writerow({
                    'Patient': patient_number,
                    'Reading': reading_number,
                    'Distance': distance,
                    'Respiration Rate (bpm)': f"{resp_rate:.2f}",
                    'Heart Rate (bpm)': f"{heart_rate:.2f}",
                    'Alt Respiration Rate (bpm)': f"{resp_rate_alt:.2f}",
                    'Alt Heart Rate (bpm)': f"{heart_rate_alt:.2f}",
                    #'Signal quality': signal_quality
                })
            del reader, slicer, events

        gc.collect()
        visualizer = None



if __name__ == "__main__":
    main()