import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from glob import glob

# import librosa
import librosa.display
import IPython.display as ipd

from itertools import cycle

sns.set_theme(style="white", palette=None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

# Reading in Audio Files
# audio_files = glob('Dream on.m4a')
audio_files = glob('../input/ravdess-emotional-speech-audio/*/*.wav')
# Play audio file
ipd.Audio(audio_files[0])

# y, sr = librosa.read(audio_files[0])
# this allows us to read in a file

# y is the raw data of the audio file and sr is an integer value of the sample rate

y, sr = librosa.load(audio_files[0])
print(f'y: {y[:10]}')
print(f'shape y: {y.shape}')
print(f'sr: {sr}')

pd.Series(y).plot(figsize=(10, 5),
                  lw=1,
                  title='Raw Audio Example',
                  color=color_pal[0])
plt.show()

# Trimming silence
y_trimmed, _ = librosa.effects.trim(y, top_db=20)
pd.Series(y_trimmed).plot(figsize=(10, 5),
                          lw=1,
                          title='Raw Audio Trimmed Example',
                          color=color_pal[1])
plt.show()

pd.Series(y[30000:30500]).plot(figsize=(10, 5),
                               lw=1,
                               title='Raw Audio Zoomed In Example',
                               color=color_pal[2])
plt.show()

# Spectrogram

D = librosa.stft(y)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
# S_db.shape

# Plot the transformed audio data
fig, ax = plt.subplots(figsize=(10, 5))
img = librosa.display.specshow(S_db,
                               x_axis='time',
                               y_axis='log',
                               ax=ax)
ax.set_title('Spectrogram Example', fontsize=20)
fig.colorbar(img, ax=ax, format=f'%0.2f')
plt.show()

# Mel Spectrogram

S = librosa.feature.melspectrogram(y=y,
                                   sr=sr,
                                   n_mels=128 * 2, )
S_db_mel = librosa.amplitude_to_db(S, ref=np.max)

fig, ax = plt.subplots(figsize=(10, 5))
# Plot the mel spectrogram
img = librosa.display.specshow(S_db_mel,
                               x_axis='time',
                               y_axis='log',
                               ax=ax)
ax.set_title('Mel Spectrogram Example', fontsize=20)
fig.colorbar(img, ax=ax, format=f'%0.2f')
plt.show()
