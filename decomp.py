import os
import sys
import wave
import random
import librosa
import pretty_midi
import numpy as np
from pydub import AudioSegment
from pydub.playback import play
from pydub import AudioSegment
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def freq_from_autocorr(signal, fs):
    """Estimate frequency using autocorrelation
    Pros: Best method for finding the true fundamental of any repeating wave,
    even with strong harmonics or completely missing fundamental
    Cons: Not as accurate, doesn't work for inharmonic things like musical
    instruments, this implementation has trouble with finding the true peak
    """
    # Calculate autocorrelation (same thing as convolution, but with one input
    # reversed in time), and throw away the negative lags
    signal -= np.mean(signal)  # Remove DC offset
    corr = fftconvolve(signal, signal[::-1], mode='full')
    corr = corr[len(corr)/2:]

    # Find the first low point
    d = diff(corr)

    try:
      start = find(d > 0)[0]
      # Find the next peak after the low point (other than 0 lag).  This bit is
      # not reliable for long signals, due to the desired peak occurring between
      # samples, and other peaks appearing higher.
      i_peak = argmax(corr[start:]) + start
      i_interp = parabolic(corr, i_peak)[0]
      freq = fs / i_interp
    except IndexError as e:
      freq = float('nan')

    return freq

def load(filename):
    """
    Load a wave file and return the signal, sample rate and number of channels.
    Can be any format that libsndfile supports, like .wav, .flac, etc.
    """
    sample_rate = 16000
    signal, sample_rate = librosa.load(filename, sr=sample_rate)
    channels = 1
    return signal, sample_rate, channels

def random_bird():
    # Put the root of the bird data set here
	dir_root = '/Users/korymath/Desktop/brds/Dataset/wav/'
	audio_file = dir_root + random.choice(os.listdir(dir_root))
	return audio_file

# load the audio file into memory
signal, sample_rate, channels = load(random_bird())

# Signal
print('signal', signal)

# Frequency from autocorrelation
print('frequencies', freq_from_autocorr(signal, sample_rate))

# Using PyDub
audio = AudioSegment.from_wav(random_bird())
play(audio[:5000])

# Mix two bird noises
print('mixing two bird sounds')
sound1 = AudioSegment.from_file(random_bird())
sound2 = AudioSegment.from_file(random_bird())
played_togther = sound1.overlay(sound2)
sound2_starts_after_delay = sound1.overlay(sound2, position=5000)
sound2_repeats_until_sound1_ends = sound1.overlay(sound2, loop=True)
sound2_plays_twice = sound1.overlay(sound2, times=2)

# Start bird 2 after a delay
print('sound2_starts_after_delay')
play(sound2_starts_after_delay)

# Repeat sound2 until sound1 ends
print('sound2_repeats_until_sound1_ends')
play(sound2_repeats_until_sound1_ends)

# Play sound2 twice
print('sound2_plays_twice')
play(sound2_plays_twice)
