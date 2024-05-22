import numpy as np
import scipy
import FDD_Module as fdd

if __name__ == "__main__":
    # Specify Sampling frequency
    Fs = 1000

    # import data (and plot)
    acc, Fs = fdd.import_data(filename='Data/MatlabData/MDOF_Data_2.csv',
                              plot=False,
                              fs=Fs,
                              time=60,
                              detrend=False,
                              downsample=False,
                              gausscheck=False,
                              cutoff=100)

    f_harmonic = fdd.harmonic_est(data=acc, delta_f=2, f_max=100, fs=Fs, threshold=30)
    print(f_harmonic)