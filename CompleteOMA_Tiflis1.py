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
    path = "Data/TiflisBruecke2/"
    n_rov = 2
    n_ref = 2
    ref_channel = [0, 3]
    rov_channel = [1, 2]
    ref_position = [0, 0]

    # Cutoff frequency (band of interest)
    cutoff = 25

    # measurement duration
    t_end = 1000

    # Threshold for MAC
    mac_threshold = 0.99

    # Welch's Method Parameters
    window = 'hann'
    n_seg = 100
    overlap = 0.5
    zero_padding = False

    '''Peak Picking Procedure on SV-diagram of the whole dataset'''
    # import data
    acc, Fs = oma.merge_data(path=path,
                             fs=Fs,
                             n_rov=n_rov,
                             n_ref=n_ref,
                             ref_channel=ref_channel,
                             rov_channel=rov_channel,
                             ref_pos=ref_position,
                             t_meas=t_end,
                             detrend=True,
                             cutoff=cutoff,
                             downsample=False)

    # Build CPSD-Matrix from acceleration data
    mCPSD, vf = oma.fdd.cpsd_matrix(data=acc,
                                    fs=Fs,
                                    zero_padding=zero_padding,
                                    n_seg=n_seg,
                                    window=window,
                                    overlap=overlap)

    # SVD of CPSD-matrix @ each frequency
    S, U, S2, U2 = oma.fdd.sv_decomp(mCPSD)

    # harmonic filtering is active
    f_harmonic = oma.fdd.harmonic_est(data=acc, delta_f=0.1, f_max=cutoff, fs=Fs, plot=True)
    S = oma.fdd.eliminate_harmonic(vf, S, f_harmonic[1:-1])

    ''' Perform SSI '''
    # Reload mat files with stored lists
    with open('freqs2.pkl', 'rb') as f:
        freqs = pickle.load(f)
    with open('status2.pkl', 'rb') as f:
        status = pickle.load(f)

    fPeaks, Peaks, nPeaks = oma.ssi.peak_picking_ssi(x=vf,
                                                     y=20*np.log10(S),
                                                     freqs=freqs,
                                                     label=status,
                                                     cutoff=cutoff)

    '''Extract modal damping by averaging over the damping values of each dataset'''
    # Scaling the mode shapes
    wn, zeta, PHI = oma.modal_extract(path=path,
                                      Fs=Fs,
                                      n_rov=n_rov,
                                      n_ref=n_ref,
                                      ref_channel=ref_channel,
                                      ref_pos=ref_position,
                                      t_meas=t_end,
                                      fPeaks=fPeaks,
                                      window=window,
                                      overlap=overlap,
                                      n_seg=n_seg,
                                      zeropadding=zero_padding,
                                      mac_threshold=mac_threshold,
                                      plot=False)

    # MPC-Calculations
    MPC = []
    for i in range(nPeaks):
        MPC.append(oma.mpc(PHI[i, :].real, PHI[i, :].imag))

    # Print Damping and natural frequencies
    print("Natural Frequencies [Hz]:")
    print(wn / 2 / np.pi)
    print("Damping [%]:")
    print(zeta * 100)
    print("Modal Phase Collinearity:")
    print(MPC)

    # additional step in mode shape scaling
    # PHI = oma.mode_shape_normalize(PHI, [0, 1], 2)

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
                              mode_shape=mode,
                              f_n=wn[i] / 2 / np.pi,
                              zeta_n=zeta[i],
                              mpc=MPC[i],
                              directory="Animations/Tiflis_2/",
                              mode_nr=i,
                              plot=True)
