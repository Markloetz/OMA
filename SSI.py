import numpy as np
import matplotlib.pyplot as plt
from OMA import OMA_Module as oma
import scipy

if __name__ == '__main__':
    # Specify Sampling frequency
    Fs = 2048

    # Threshold for MAC
    mac_threshold = 0.85

    # import data (and plot)
    acc, Fs = oma.import_data(filename='Data/DataPlateHarmonicInfluence/acc_data_01_09_12_33_harmonic_22_5Hz.csv',
                              plot=False,
                              fs=Fs,
                              time=180,
                              detrend=True,
                              downsample=False,
                              cutoff=1000)

    # SSI Pseudocode
    # 1 Form Block-Hankel-Matrix
    # 2 Split Block-Hankel-Matrix
    # 3 Calculate Projection by QR-Decomposing the BHM
    # 4 Projection = Observability Matrix @ Kalman States
    # 5 SVD of Projection

    oma.ssi.ssi_proc(acc,
                     fs=Fs,
                     br=15)


