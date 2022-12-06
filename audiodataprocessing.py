import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from glob import glob
#glob allows us to list out all the files in a directory which will be helpfull when we want to read in a bunch of the wav files from this data set
import librosa
import librosa.display
import IPython.display as ipd
from itertools import cycle

sns.set.theme(style="white", palette=None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

#Reading in Audio Files
audio_files = glob('filename')----------------------------------------------------------
#Play audio file
ipd.Audio(audio_files[0])

y, sr = librosa.read(audio_files[0])
#this allows us to read in a file
#y is the raw data of the audio file and sr is an integer value of the sample rate
