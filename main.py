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
    path = "Data/TiflisBruecke/"
    n_rov = 2
    n_ref = 1
    ref_channel = 0
    rov_channel = [1, 2]
    ref_position = [0, 0]

    # Cutoff frequency (band of interest)
    cutoff = 25

    # measurement duration
    t_end = 500

    # SSI-Parameters
    # Specify limits
    f_lim = 0.01  # Pole stability (frequency)
    z_lim = 0.04  # Pole stability (damping)
    mac_lim = 0.05  # Mode stability (MAC-Value)
    limits = [f_lim, z_lim, mac_lim]

    # block-rows
    ord_max = 50
    ord_min = 10

    # '''Peak Picking Procedure on SV-diagram of the whole dataset'''
    # # import data
    # acc, Fs = oma.merge_data(path=path,
    #                          fs=Fs,
    #                          n_rov=n_rov,
    #                          n_ref=n_ref,
    #                          ref_channel=ref_channel,
    #                          rov_channel=rov_channel,
    #                          ref_pos=ref_position,
    #                          t_meas=t_end,
    #                          detrend=True,
    #                          cutoff=cutoff * 4,
    #                          downsample=False)
    #
    # # SSI
    # # Perform SSI
    # freqs, zeta, modes, _, _, status = oma.ssi.SSICOV(acc,
    #                                                   dt=1 / Fs,
    #                                                   Ts=0.8,
    #                                                   ord_min=ord_min,
    #                                                   ord_max=ord_max,
    #                                                   limits=limits)
    #
    # # Temporarily Save Results from SSI to improve Debugging speed
    # with open('freqs.pkl', 'wb') as f:
    #     pickle.dump(freqs, f)
    # with open('modes.pkl', 'wb') as f:
    #     pickle.dump(modes, f)
    # with open('zeta.pkl', 'wb') as f:
    #     pickle.dump(zeta, f)
    # with open('status.pkl', 'wb') as f:
    #     pickle.dump(status, f)

    # Reload mat files with stored lists
    with open('freqs.pkl', 'rb') as f:
        freqs = pickle.load(f)
    with open('modes.pkl', 'rb') as f:
        modes = pickle.load(f)
    with open('zeta.pkl', 'rb') as f:
        zeta = pickle.load(f)
    with open('status.pkl', 'rb') as f:
        status = pickle.load(f)

    # the mode shapes themselves must be np.arrays
    for i in range(len(modes)):
        for j in range(len(modes[i])):
            modes[i][j] = np.array(modes[i][j])
    # stabilization diag
    fig, ax = oma.ssi.stabilization_diag(freqs, status, cutoff)
    plt.show()

    # Extract modal parameters only using results stable in all aspects
    ranges = [[1.8 - 0.5, 1.8 + 0.5],
              [6.25 - 0.5, 6.25 + 0.5],
              [7.1 - 0.5, 7.1 + 0.5],
              [13.25 - 0.5, 13.25 + 0.5],
              [16.9 - 0.5, 16.9 + 0.5]]
    nPeaks = len(ranges)
    fS, zetaS, _ = oma.ssi.ssi_extract(freqs, zeta, modes, status, ranges)

    # Print Damping and natural frequencies
    print("Natural Frequencies [Hz]:")
    print(fS)
    print("Damping [%]:")
    print([x * 100 for x in zetaS])

    # Extract the mode shape from each dataset separately
    fS, zetaS, modeS = oma.modal_extract_ssi(path=path,
                                             Fs=Fs,
                                             n_rov=n_rov,
                                             n_ref=n_ref,
                                             ref_channel=ref_channel,
                                             rov_channel=rov_channel,
                                             ref_pos=ref_position,
                                             t_meas=t_end,
                                             fPeaks=[1.8, 6.25, 7.1, 13.25, 16.99],
                                             limits=limits,
                                             ord_min=ord_min,
                                             ord_max=ord_max,
                                             d_ord=1,
                                             plot=True,
                                             mode_extract=True,
                                             cutoff=cutoff,
                                             Ts=0.8)

    # 2d-Plot mode shapes
    for i in range(nPeaks):
        plt.plot(np.real(modeS[i]), label="Mode: " + str(i + 1))
    plt.legend()
    plt.show()

    # 3d-Plot mode shapes
    discretization = loadmat('Discretizations/TiflisBruecke.mat')
    N = discretization['N']
    E = discretization['E']

    for i in range(nPeaks):
        mode = np.zeros(len(modeS[i]) + 4, dtype=np.complex_)
        mode[2:-2] = modeS[i]
        oma.animate_modeshape(N,
                              E + 1,
                              mode_shape=mode,
                              f_n=fS[i],
                              zeta_n=zetaS[i],
                              directory="Animations/TiflisSSI/",
                              mode_nr=i,
                              plot=True)
