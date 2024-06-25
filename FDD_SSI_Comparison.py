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

    # SSI-Parameters
    # Specify limits
    f_lim = 0.01  # Pole stability (frequency)
    z_lim = 0.1  # Pole stability (damping)
    mac_lim = 0.2  # Mode stability (MAC-Value)
    z_max = 0.1  # Maximum damping value
    limits = [f_lim, z_lim, mac_lim, z_max]

    # block-rows
    br = 4
    ord_max = br * 12
    ord_min = 6
    d_ord = 2

    # Welch's Method Parameters
    window = 'hann'
    n_seg = 120
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

    # SSI
    # Perform SSI algorithm
    freqs, zeta, modes, A, C = oma.ssi.ssi_proc(acc,
                                                fs=Fs,
                                                ord_min=6,
                                                ord_max=2 * acc.shape[1],
                                                d_ord=d_ord,
                                                method='DataDriven')

    # Calculate stable poles
    freqs_stable, zeta_stable, modes_stable, order_stable = oma.ssi.stabilization_calc(freqs, zeta, modes, limits)

    # Peak Picking
    fPeaks, Peaks, nPeaks = oma.ssi.peak_picking_ssi(x=vf,
                                                     y=20 * np.log10(S),
                                                     freqs=freqs_stable,
                                                     order=order_stable,
                                                     cutoff=cutoff,
                                                     plot='all')

    '''Extract modal damping by averaging over the damping values of each dataset'''
    '''
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
                                      plot=True)'''

    '''Same for SSI'''
    wn, zeta, PHI = oma.ssi_extract(path=path,
                                    Fs=Fs,
                                    n_rov=n_rov,
                                    n_ref=n_ref,
                                    ref_channel=ref_channel,
                                    ref_pos=ref_position,
                                    t_meas=t_end,
                                    fPeaks=fPeaks,
                                    limits=limits,
                                    ord_min=ord_min,
                                    ord_max=ord_max,
                                    d_ord=d_ord,
                                    plot=True,
                                    cutoff=cutoff)

    # Print Damping and natural frequencies
    print("Natural Frequencies [Hz]:")
    print(wn / 2 / np.pi)
    print("Damping [%]:")
    print(zeta * 100)

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
                              directory="Animations/Tiflis_2/",
                              mode_nr=i,
                              plot=True)
