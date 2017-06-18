import os
import sys
import wave
import random
import librosa
import pretty_midi
import numpy as np
from pydub import AudioSegment
from pydub.playback import play
from sklearn.decomposition import PCA

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
	dir_root = 'Dataset/wav/'
	audio_file = dir_root + random.choice(os.listdir(dir_root))
	return audio_file

# load the audio file into memory
signal, sample_rate, channels = load(random_bird())

# Signal
print('signal', signal)

# Using PyDub
audio = AudioSegment.from_wav(random_bird())
print(audio)
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
