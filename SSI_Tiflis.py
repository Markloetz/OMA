import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy
from OMA import OMA_Module as oma

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
    cutoff = 50

    # measurement duration
    t_end = 500

    # SSI-Parameters
    f_lim = 0.02        # Pole stability (frequency)
    z_lim = 0.05        # Pole stability (damping)
    mac_lim = 0.02      # Mode stability (MAC-Value)
    limits = [f_lim, z_lim, mac_lim]
    ord_min = 0
    ord_max = 60

    '''Extract modal damping by averaging over the damping values of each dataset'''
    fPeaks = [1.85, 6.36, 7.09, 13.30, 16.95]
    nPeaks = len(fPeaks)
    wn, zeta, PHI = oma.modal_extract_ssi(path=path,
                                          Fs=Fs,
                                          n_rov=n_rov,
                                          n_ref=n_ref,
                                          ref_channel=ref_channel,
                                          rov_channel=rov_channel,
                                          ref_pos=ref_position,
                                          t_meas=t_end,
                                          fPeaks=fPeaks,
                                          limits=limits,
                                          ord_min=ord_min,
                                          ord_max=ord_max,
                                          d_ord=1,
                                          plot=False,
                                          cutoff=cutoff,
                                          Ts=1)
    # MPC-Calculations
    MPC = []
    for i in range(nPeaks):
        MPC.append(oma.mpc(PHI[i, :].real, PHI[i, :].imag))

    # Print Damping and natural frequencies
    print("Natural Frequencies [Hz]:")
    print(wn)
    print("Damping [%]:")
    print(zeta * 100)
    print("Modal Phase Collinearity:")
    print(MPC)

    # 2d-Plot modeshapes
    for i in range(nPeaks):
        plt.plot(np.real(PHI[i, :]), label="Mode: " + str(i + 1))
    plt.legend()
    plt.show()

    '''Standardabweichung für die Dämpfung angeben'''

    # 3d-Plot mode shapes
    discretization = scipy.io.loadmat('Discretizations/TiflisBruecke.mat')
    N = discretization['N']
    E = discretization['E']

    for i in range(nPeaks):
        mode = np.zeros(PHI.shape[1] + 4, dtype=np.complex_)
        mode[2:-2] = PHI[i, :]
        title = "Mode " + str(i + 1) + " at " + str(round(wn[i] / 2 / np.pi, 2)) + "Hz (" + str(
            round(zeta[i] * 100, 2)) + "%)"
        oma.animate_modeshape(N,
                              E + 1,
                              mode_shape=mode.real,
                              f_n=wn[i],
                              zeta_n=zeta[i],
                              mpc=MPC[i],
                              directory="Animations/Tiflis/",
                              mode_nr=i,
                              plot=True)
