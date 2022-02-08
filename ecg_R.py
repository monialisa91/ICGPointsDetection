from pantomkins import pt
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

class points():

    def __init__(self, data_ecg, data_icg, fs):
        self.data_ecg = data_ecg
        self.data_icg = data_icg
        self.fs = fs

    def R_peak_detection(self):
        pan = pt(self.data_ecg, self.fs)
        data_pt = pan.fit()
        peaks = find_peaks(data_pt)[0]
        values = data_pt[np.array(peaks)]
        maksimum = np.sort(values)[-2:]
        thr = 0.8 * np.mean(maksimum)
        peaks_thr = np.where(values>thr)
        peaks_thr2 = peaks[peaks_thr]

        '''plt.plot(np.arange(len(data_pt)), data_pt)
        plt.scatter(peaks_thr2, data_pt[peaks_thr2])
        plt.axhline(thr)
        plt.show()'''

        return peaks_thr2

    def C_point_detection(self):
        R_points = self.R_peak_detection()
        C_points = []
        cc_interval = []
        for i in range(len(R_points)-1):
            pos0 = R_points[i]
            posk = R_points[i+1]
            print(pos0, posk)
            cc = self.data_icg[pos0:posk]
            cc_interval.append(posk-pos0)
            C_point = np.argmax(cc) + pos0
            C_points.append(C_point)
        c_mean = np.mean(np.array(cc_interval))
        pos0 = R_points[-1]
        posk = int(pos0 + c_mean)
        cc = self.data_icg[pos0:posk]
        C_point = np.argmax(cc) + pos0
        C_points.append(C_point)
        return np.array(C_points)


