from OMA import OMA_Module as oma
import numpy as np
import scipy
from matplotlib import pyplot as plt

if __name__ == "__main__":
    Fs = 2048

    # Cutoff frequency (band of interest)
    cutoff = 100

    # measurement duration
    t_end = 500

    # Threshold for MAC
    mac_threshold = 0.5

    # Decide if harmonic filtering is active
    filt = False

    # Decide if the modes need to be scaled (and where to find the data for scaling)
    scaling = True
    path = "Data/TiflisBruecke/"

    # Welch's Method Parameters
    window = 'hann'
    n_seg = 100
    overlap = 0.5
    zero_padding = False

    # import data (and plot)
    acc, Fs = oma.import_data(filename="Data/Data_040624_pos0410_wo_hammer.mat",
                              plot=False,
                              fs=Fs,
                              time=t_end,
                              detrend=True,
                              downsample=False,
                              cutoff=cutoff)

    data = acc[:, 0]
    n_per_seg = np.floor(len(data) / n_seg)  # divide into 8 segments
    n_overlap = np.floor(overlap * n_per_seg)  # Matlab uses zero overlap

    # Build CPSD-Matrix from acceleration data
    vf, mCPSD = scipy.signal.csd(data,
                                 data,
                                 fs=Fs,
                                 nperseg=n_per_seg,
                                 noverlap=n_overlap,
                                 window=window)

    plt.plot(vf, 20 * np.log10(mCPSD))
    plt.show()
