## supress warnings for clean output
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import wave
import math
import glob
import time
import random
import pyaudio
import feather
import librosa
import argparse
import subprocess
import pretty_midi
import numpy as np
import pandas as pd
from matplotlib.mlab import find
from sklearn.decomposition import PCA
from scipy import signal
from bisect import bisect_left
from operator import itemgetter
from itertools import *
from pydub import AudioSegment


def takeClosest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
       return after
    else:
       return before

def parabolic(f, x):
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)

def freq_from_autocorr(audio_signal, sr):
    """Estimate frequency using autocorrelation."""
    # Calculate autocorrelation (same thing as convolution, but with one input
    # reversed in time), and throw away the negative lags
    audio_signal -= np.mean(audio_signal)  # Remove DC offset
    corr = signal.fftconvolve(audio_signal, audio_signal[::-1], mode='full')
    corr = corr[len(corr)/2:]
    # Find the first low point
    d = np.diff(corr)
    try:
        start = find(d > 0)[0]
        # Find the next peak after the low point (other than 0 lag).  This bit is
        # not reliable for long signals, due to the desired peak occurring between
        # samples, and other peaks appearing higher.
        i_peak = np.argmax(corr[start:]) + start
        i_interp = parabolic(corr, i_peak)[0]
        freq = sr / i_interp
    except IndexError as e:
        # index could not be found, set the pitch to frequency 0
        freq = float('nan')
    
    # The voiced speech of a typical adult male will have a fundamental frequency 
    # from 85 to 180 Hz, and that of a typical adult female from 165 to 255 Hz.
    # Lowest Bass E2 (82.41Hz) to Soprano to C6 (1046.50Hz)
    
    # This number can be set with insight from the full dataset
    # if freq < 80 or freq > 1000:
    if freq < 100 or freq > 400:
        freq = float('nan')
        
    return freq

def load(filename, sr=8000):
    """
    Load a wave file and return the signal, sample rate and number of channels.
    Can be any format that libsndfile supports, like .wav, .flac, etc.
    """
    signal, sample_rate = librosa.load(filename, sr=sr)
    channels = 1
    return signal, sample_rate, channels

def analyse_pair(head, midi_filename, freq_func=freq_from_autocorr, num_windows=10, fs=8000, frame_size=256):
    """Analyse a single input example from the MIR-QBSH dataset."""
    # set parameters 
    # sampling rate of the data samples is 8kHz or 8000Hz 
    # ground truth frame size is at 256
    # can oversample for increased resolution

    ## Load data and label
    fileroot = head + '/' + midi_filename
    
    # Load the midi data as well
    midi_file = midiroot + midi_filename + '.mid'
    # print(midiroot)
#     midi_data = pretty_midi.PrettyMIDI(midi_file)
    midi_data = None 
    # print(midi_data)
    # think about alignment to midi_data.instruments[0].notes
    # currently alignment to true midi is NOT handled

    # load data
    audio_signal, _, _ = load(fileroot+'.wav', fs)
    # print('audio_signal', audio_signal)
    # print('length of audio_signal', len(audio_signal))

    # Load matching true labelled values
    # The .pv file contains manually labelled pitch file, 
    # with frame size = 256 and overlap = 0. 
    with open(fileroot+'.pv', 'r') as f:
        y = []
        for line in f:
            if float(line) > 0:
                y.append(float(line))
            else:
                y.append(float('nan'))
    # length of the true pitch values should match 
    # the number of audio frames to analyse
    # print('length of true pitch values', len(y))

    size_match = len(y) == len(audio_signal)/frame_size
    # print('size match', size_match)

    num_frames = len(audio_signal) / frame_size
    # print('num_frames', num_frames)

    # extract pitches (candidates), y_hat

    # allow for window overlap to handle clean transitions
    # solve for num_frames * 4 and then average out for clean sample
    window_size = frame_size / num_windows
    y_hat_freq = []
    for n in range(num_frames):
        window_freq = []
        for i in range(num_windows):
            window_start = 0+n*frame_size+i*window_size
            window_end = frame_size+n*frame_size+i*window_size
            window_s = audio_signal[window_start:window_end]

            # this is where the magic happens
            # define the function to extract the frequency from the windowed signal
            window_freq.append(freq_func(window_s, fs))
        
        # append the median of the window frequencies, 
        # somewhat robust to anomalies
        y_hat_freq.append(np.median(window_freq))

    # One approach is to remove the outlier points by eliminating any 
    # points that were above (Mean + 2*SD) and any points below (Mean - 2*SD) 
    # before plotting the frequencies. This can not happen on the frequency itself
    # and rather happens on the frequency change, thus smoothing out huge leaps
    freq_change = np.abs(np.diff(y_hat_freq))
    
    # remove any points where the frequency change is drastic
    freq_mean = np.nanmean(freq_change)
    freq_std = np.nanstd(freq_change)
    
    # this is arbitraily set and may need to be tuned 
    freq_change_max = freq_mean + 2*freq_std

    bad_idx = np.argwhere(freq_change > freq_change_max).flatten()
    for i in bad_idx:
        y_hat_freq[i] = float('nan')
    
    # Convert the frequencies to midi notes
    y_hat = librosa.hz_to_midi(y_hat_freq)

    # print('y_hat', y_hat)
    # print('length of estimated pitch values', len(y_hat))

    # compare pitches with actual labels, y
    squared_error = (y-y_hat)**2
    absolute_error = abs(y-y_hat)
    mse = np.nanmean(squared_error)
    mae = np.nanmean(absolute_error)
    # print('MSE', mse)
    
    # create a version of the frequency distribution with no nans
    y_hat_freq_no_nan = [value for value in y_hat_freq if not math.isnan(value)]
    
    # clean up the pitches
    clean_y_hat = cleaned_midi_pitches(y_hat)    
    return audio_signal, midi_data, y, y_hat, clean_y_hat, y_hat_freq, y_hat_freq_no_nan, squared_error, mse, absolute_error, mae

def scale_linear_bycolumn(rawpoints, high=100.0, low=0.0):
    mins = np.min(rawpoints, axis=0)
    maxs = np.max(rawpoints, axis=0)
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng)

def save_extracted_pitches(clean_y_hat, fname, reconroot=None, fs=8000, frame_size=256):    
    # Save the extraced pitches 
    np.savetxt(fname + '.out', clean_y_hat, delimiter=',')
    
    # Save the midi rendition
    # Create a PrettyMIDI object
    song_recon = pretty_midi.PrettyMIDI()
    
    # Create an Instrument instance
    instrument_name = 'Bird Tweet'  # 'Bird Tweet'
    instrument_program = pretty_midi.instrument_name_to_program(instrument_name)
    instrument = pretty_midi.Instrument(program=instrument_program)
        
    # get the real indecies
    real_idx = np.argwhere(~np.isnan(clean_y_hat))

    # find groups of real valued signals
    consecutive_real = []
    for k, g in groupby(enumerate(real_idx.flatten()), lambda (i,x):i-x):
        consecutive_real.append(map(itemgetter(1), g))

    # initialize the note holding arrays
    start_times = []
    end_times = []
    pitches = []
    velocities = []
    
    time_per_step = float(frame_size / (fs * 1.0)) # 0.032 

    for group in consecutive_real:
        # only build note if it exists for longer than a single step
        if len(group) > 1:
            start = group[0] * time_per_step
            start_times.append(start)
            end_times.append(start + len(group) * time_per_step)
            pitches.append(clean_y_hat[group[0]])
            # can pull a dynamic velocity from the amplitude of the wave
            velocities.append(110)

    # need to scale over the range of pitches for the bird tweets
    # to sound anywhere near interesting... push to almost
    # the full range of the general midi bird pitches
    # print 'max and min', max(pitches), min(pitches)
    # pitches = scale_linear_bycolumn(pitches, high=110, low=10)
    # print 'new max and min', max(pitches), min(pitches)
    
    for i in range(len(pitches)):
        note = pretty_midi.Note(velocity=velocities[i], pitch=int(pitches[i]), 
                                start=start_times[i], 
                                end=end_times[i])
        print(note)
        # append the note to the instrument
        instrument.notes.append(note)
            
    # Add the instrument to the PrettyMIDI object
    song_recon.instruments.append(instrument)
    
    # Write out the MIDI data
    if reconroot is not None:
        revised_file_name = '-'.join(fname.split('/')[3:])[:-4] + '-' + instrument_name.replace(" ", "_") + '.recon.mid'
    else:
        revised_file_name = fname + '-' + instrument_name.replace(" ", "_") + '.recon.mid'
        reconroot = ''
    
    print reconroot + revised_file_name
    song_recon.write(reconroot + revised_file_name)
    
    midi_f = reconroot + revised_file_name
    wav_f = reconroot + revised_file_name[:-4] + '.wav'
    
    # play the sound immediately 
    # need to have the correct soundfont in the Dataset folder
    subprocess.call(['fluidsynth', 'Dataset/soundfonts/fluid_r3_gm2.sf2', midi_f, '--no-shell'])
    # subprocess.call(['fluidsynth', 'Dataset/soundfonts/Birdsongs_Arizona.sf2', midi_f, '--no-shell'])

    ## save as a wav file for easy comparison
    subprocess.call(['timidity', midi_f, '-Ow', '-o', wav_f])

    print('saved', wav_f)

    return song_recon

def cleaned_midi_pitches(old_y_hat):
    # find NaN gaps of 1 and infill
    # find larger NaN gaps and create segments
    # use median in segment to define note
    # use length of segment to define note duration

    # initialize our recovery midi pitches
    y_hat_clean = np.copy(old_y_hat)

    # get indecies of nan values
    nan_idx = np.argwhere(np.isnan(y_hat_clean))

    # interpolate over single nans 
    consecutive_nans = []
    for k, g in groupby(enumerate(nan_idx.flatten()), lambda (i,x):i-x):
        consecutive_nans.append(map(itemgetter(1), g))

    for group in consecutive_nans:
        if len(group) == 1:
            # print('single nan at ', group[0])
            if (group[0] == 0):
                y_hat_clean[group[0]] = y_hat_clean[group[0]+1]
            elif (group[0] == len(y_hat_clean) - 1):
                y_hat_clean[group[0]] = y_hat_clean[group[0]-1]
            else:
                # print(group[0], len(y_hat_clean))
                y_hat_clean[group[0]] = ((y_hat_clean[group[0]+1])+(y_hat_clean[group[0]-1]))/2
            # print(group[0], y_hat_clean[group[0]])

    # get idx of real valued signals
    real_idx = np.argwhere(~np.isnan(y_hat_clean))

    # find groups of real valued signals
    consecutive_real = []
    for k, g in groupby(enumerate(real_idx.flatten()), lambda (i,x):i-x):
        consecutive_real.append(map(itemgetter(1), g))
    # for each group, the median is a good estimate of the midi pitch class
    for group in consecutive_real:
        # find the median of the real note group
        y_hat_clean[group] = np.median(y_hat_clean[group])
    return y_hat_clean

def main(args=None):

    # parse the arg list
    fname = args.fname
    num_windows = args.num_windows
    frame_size = args.frame_size
    fs = args.fs

    if not fname:
        print('no input given, recording audio')
        format = pyaudio.paInt16
        channels = 1
        record_seconds = args.record_seconds
        millis = int(round(time.time() * 1000))
        input_fname = "Dataset/recordings/{}.wav".format(millis)
         
        audio = pyaudio.PyAudio()
         
        # start Recording
        stream = audio.open(format=format, channels=channels,
                        rate=fs, input=True,
                        frames_per_buffer=frame_size)
        print "recording..."
        frames = []
         
        for i in range(0, int(fs / frame_size * record_seconds)):
            data = stream.read(frame_size)
            frames.append(data)
        print "finished recording"
         
        # stop Recording
        stream.stop_stream()
        stream.close()
        audio.terminate()
         
        print('saving wave file', input_fname)
        waveFile = wave.open(input_fname, 'wb')
        waveFile.setnchannels(channels)
        waveFile.setsampwidth(audio.get_sample_size(format))
        waveFile.setframerate(fs)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()

        print('processing wave file')
        fname = input_fname

    # start the processing
    print('starting processing')
    audio_signal, _, _ = load(fname, fs)
    num_frames = len(audio_signal) / frame_size
    print('num_frames', num_frames)

    # define freq_func
    freq_func=freq_from_autocorr

    # allow for window overlap to handle clean transitions
    # solve for num_frames * 4 and then average out for clean sample
    window_size = frame_size / num_windows
    y_hat_freq = []
    for n in range(num_frames):
        window_freq = []
        for i in range(num_windows):
            window_start = 0+n*frame_size+i*window_size
            window_end = frame_size+n*frame_size+i*window_size
            window_s = audio_signal[window_start:window_end]

            # this is where the magic happens
            # define the function to extract the frequency from the windowed signal
            window_freq.append(freq_func(window_s, fs))

        # append the median of the window frequencies, 
        # somewhat robust to anomalies
        y_hat_freq.append(np.median(window_freq))

    # One approach is to remove the outlier points by eliminating any 
    # points that were above (Mean + 2*SD) and any points below (Mean - 2*SD) 
    # before plotting the frequencies. This can not happen on the frequency itself
    # and rather happens on the frequency change, thus smoothing out huge leaps
    freq_change = np.abs(np.diff(y_hat_freq))

    # remove any points where the frequency change is drastic
    freq_mean = np.nanmean(freq_change)
    freq_std = np.nanstd(freq_change)
    freq_change_max = freq_mean + freq_std
    bad_idx = np.argwhere(freq_change > freq_change_max).flatten()
    for i in bad_idx:
        y_hat_freq[i] = float('nan')

    # Convert the frequencies to midi notes
    y_hat = librosa.hz_to_midi(y_hat_freq)

    print('length of estimated pitch values', len(y_hat))

    # create a version of the frequency distribution with no nans
    y_hat_freq_no_nan = [value for value in y_hat_freq if not math.isnan(value)]

    # clean up the pitches
    clean_y_hat = cleaned_midi_pitches(y_hat)   
    song_recon = save_extracted_pitches(clean_y_hat, fname=fname, fs=fs, frame_size=frame_size)    

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
 
   parser.add_argument('--fname', default=None)
   parser.add_argument('--num_windows', default=10, type=int)
   parser.add_argument('--frame_size', default=1024, type=int)
   parser.add_argument('--fs', default=44100, type=int)
   parser.add_argument('--record_seconds', default=5, type=int)
   args = parser.parse_args()
   main(args=args)