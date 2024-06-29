from matplotlib import pyplot as plt
import numpy as np
from scipy.io import loadmat
from OMA import OMA_Module as oma
import pickle

if __name__ == '__main__':
    '''Specify Parameters for OMA'''
    # Specify Sampling frequency
    Fs = 2048

    # Path of Measurement Files and other specifications
    file1 = "Data/TiflisBruecke/Data_040624_pos0708.mat"
    file2 = "Data/Data_040624_pos0410_wo_hammer.mat"
    file3 = "Data/TiflisBruecke2/Data_190624_pos_r1_07_08_r2.mat"

    # Cutoff frequency (band of interest)
    cutoff = 25

    # measurement duration
    t_end = 500

    # SSI-Parameters
    # Specify limits
    f_lim = 0.01  # Pole stability (frequency)
    z_lim = 0.01  # Pole stability (damping)
    mac_lim = 0.1  # Mode stability (MAC-Value)
    limits = [f_lim, z_lim, mac_lim]

    # block-rows
    ord_max = 50
    ord_min = 10

    '''Peak Picking Procedure on SV-diagram of the whole dataset'''
    # import data
    acc1, _ = oma.import_data(filename=file1,
                              fs=Fs,
                              time=t_end,
                              detrend=True,
                              cutoff=cutoff * 4,
                              downsample=False,
                              plot=False)
    acc2, _ = oma.import_data(filename=file2,
                              fs=Fs,
                              time=t_end,
                              detrend=True,
                              cutoff=cutoff * 4,
                              downsample=False,
                              plot=False)
    acc3, _ = oma.import_data(filename=file3,
                              fs=Fs,
                              time=t_end,
                              detrend=True,
                              cutoff=cutoff * 4,
                              downsample=False,
                              plot=False)

    # SSI
    # Perform SSI
    freqs, zeta, modes, _, _, status = oma.ssi.SSICOV(acc1,
                                                      dt=1 / Fs,
                                                      Ts=1,
                                                      ord_min=ord_min,
                                                      ord_max=ord_max,
                                                      limits=limits)

    # stabilization diag
    fig, ax = oma.ssi.stabilization_diag(freqs, status, cutoff)
    plt.show()

    # Perform SSI
    freqs, zeta, modes, _, _, status = oma.ssi.SSICOV(acc2,
                                                      dt=1 / Fs,
                                                      Ts=1,
                                                      ord_min=ord_min,
                                                      ord_max=ord_max,
                                                      limits=limits)

    # stabilization diag
    fig, ax = oma.ssi.stabilization_diag(freqs, status, cutoff)
    plt.show()

    # Perform SSI
    freqs, zeta, modes, _, _, status = oma.ssi.SSICOV(acc3[:, :2],
                                                      dt=1 / Fs,
                                                      Ts=1,
                                                      ord_min=ord_min,
                                                      ord_max=ord_max,
                                                      limits=limits)

    # stabilization diag
    fig, ax = oma.ssi.stabilization_diag(freqs, status, cutoff)
    plt.show()
