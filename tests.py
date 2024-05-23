import numpy as np
import scipy
import FDD_Module as fdd

if __name__ == "__main__":
    # Specify Sampling frequency
    Fs = 2048

    # import data (and plot)
    acc, Fs = fdd.import_data(filename='Data/ShakerOMA/acc_data_01_09_12_33_harmonic_35Hz.csv',
                              plot=False,
                              fs=Fs,
                              time=180,
                              detrend=True,
                              downsample=False,
                              gausscheck=False,
                              cutoff=100)

    f_harmonic = fdd.harmonic_est(data=acc, delta_f=0.5, f_max=100, fs=Fs, plot=True)
    print(f_harmonic)
