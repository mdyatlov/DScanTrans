import os
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy import fft

input_folder_name = '../../Data/DScan/20210201_SINIS_Transmission/TE/'

output_folder_name = 'images_sinis_trans'
if not os.path.exists(output_folder_name):
    os.makedirs(output_folder_name)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # getting colors for plots

# Defining data samples names
labels_array = ['No Clad REF 3', 'No Clad BENT 3', 'Er:YSZ REF 3', 'Er:YSZ BENT 3']

# -------------------------------------------------------------------------------------------------------------------- #
# TRANSMISSION SPECTRA ANALYSIS

fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)

for i in range(4):
    # Loading datafiles with optical spectrum
    data = np.loadtxt(input_folder_name + str(i + 1) + '.txt', skiprows=1)

    # Converting optical spectrum from dB to W
    data[:, 1] = 10 ** (data[:, 1] / 10) * 1e-3

    # Plotting plot
    axs[math.floor(i / 2)].plot(data[:, 0], data[:, 1], label=labels_array[i])

# Setting axes properties and creating legend
for i in range(2):
    axs[i].set(title='SINIS Transmission spectra',
               xlabel=r'Wavelength, nm', ylabel=r'Power, W',
               xlim=(1560, 1600))
    axs[i].legend()

# Showing and saving figure
plt.tight_layout()
plt.savefig(output_folder_name + '/spectra_example.png')
plt.savefig(output_folder_name + '/spectra_example.eps')
plt.show()
plt.clf()

# -------------------------------------------------------------------------------------------------------------------- #
# TRANSMISSION SPECTRA FOURIER ANALYSIS

fig, ax = plt.subplots()

for i in range(4):
    # Loading datafiles with optical spectrum
    data = np.loadtxt(input_folder_name + str(i + 1) + '.txt', skiprows=1)

    # Generating frequency grid
    N_freq = np.size(data, 0)
    dlambda = (max(data[:, 0]) - min(data[:, 0])) / N_freq * 1e-9
    freq_max = 1 / 2 / dlambda
    dfreq = freq_max * 2 / N_freq
    freq = np.arange(-freq_max, freq_max, dfreq)

    # Converting optical spectrum from dB to W
    data[:, 1] = 10 ** (data[:, 1] / 10) * 1e-3

    # Taking Fourier transform of the optical spectrum, normalizing it on max
    spec = abs(fft.fftshift(fft.fft(data[:, 1]))) / max(abs(fft.fftshift(fft.fft(data[:, 1]))))

    # Searching peaks we need
    for j in range(N_freq):
        if spec[j] > 0.15:
            print(labels_array[i]+' PEAK ({:.2e}, {:.2f})'.format(freq[j], spec[j]))

    # Plotting plot
    ax.plot(freq, spec, label=labels_array[i])

# Setting axes properties and creating legend
ax.set(title='Modulation frequencies of SINIS optical spectra',
       xlabel=r'Spatial frequency, m$^{-1}$', ylabel=r'Spectral density',
       xlim=(-5e8, 5e8))
ax.legend()

# Showing and saving figure
plt.tight_layout()
plt.savefig(output_folder_name + '/spectra_sinis.png')
plt.savefig(output_folder_name + '/spectra_sinis.eps')
plt.show()
plt.clf()