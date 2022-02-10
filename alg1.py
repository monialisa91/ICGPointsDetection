import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import re
from ecg_preprocess import ecg_preprocess
from ecg_preprocess import icg_preprocess
from scipy.signal import savgol_filter as sg
from pantomkins import pt
from ecg_R import points


def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]

def plotICGECG(data_icg, data_ecg):
    fig, axs = plt.subplots(2)

    axs[0].plot(np.arange(len(data_icg)), data_icg)
    axs[1].plot(np.arange(len(data_ecg)), data_ecg)
    plt.show()

# 1. DATA LOAD

files = glob("01_RawData/*BL.mat")
files.sort(key=natural_keys)

# 2. DATA PARAMETERS

lim = 1000
fs = 500

# a) ECG parameters

cutoff_lowECG = 30
order_lowECG = 4
cutoff_highECG = 5
order_highECG = 4

# b) ICG parameters

cutoff_lowICG = 20
order_lowICG = 2
cutoff_highICG = 5
order_highICG = 4


# 3. LOAD OF THE PREPROCESSED FILES

ecg = ecg_preprocess(files[0], lim, sampling_rate=fs, cutoff_low=cutoff_lowECG, cutoff_high=cutoff_highECG, order_low=order_lowECG, order_high=order_highECG)
data_ecg = ecg.sg_filter()
data_ecg = data_ecg.reshape(data_ecg.shape[0])

icg = icg_preprocess(files[0], lim, sampling_rate=fs, cutoff_low=cutoff_lowICG, cutoff_high=cutoff_highICG, order_low=order_lowICG, order_high=order_highICG)
data_icg = icg.baseline()

# 4. FIDUCIAL POINTS DETECTION

det = points(data_ecg, data_icg, fs)
Rpoints = det.R_peak_detection()
Cpoints = det.C_point_detection()
Bpoints = det.B_point_detection()
print(Bpoints)

plt.plot(np.arange(len(data_icg)), data_icg)
plt.scatter(Cpoints, data_icg[Cpoints])
plt.scatter(Bpoints, data_icg[Bpoints])


plt.show()


exit()

'''plt.plot(np.arange(len(ecg_pan)), ecg_pan)
plt.show()'''

