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
    z_lim = 0.04  # Pole stability (damping)
    mac_lim = 0.05  # Mode stability (MAC-Value)
    z_max = 0.1  # Maximum damping value
    limits = [f_lim, z_lim, mac_lim, z_max]

    # block-rows
    ord_max = 50
    ord_min = 10
    d_ord = 1

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
                             cutoff=cutoff*2,
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
    data, _ = oma.import_data("Data/TiflisBruecke2/Data_190624_pos_r1_09_10_r2.mat",
                              time=t_end,
                              plot=False,
                              fs=Fs,
                              detrend=True,
                              downsample=False,
                              cutoff=cutoff)
    freqs, zeta, modes, A, C = oma.ssi.ssi_proc(acc,
                                                fs=Fs,
                                                ord_min=ord_min,
                                                ord_max=ord_max,
                                                d_ord=d_ord,
                                                method='CovarianceDriven',
                                                Ts=0.8)

    # Calculate stable poles
    freqs_stable, zeta_stable, modes_stable, order_stable = oma.ssi.stabilization_calc(freqs, zeta, modes, limits)

    # Peak Picking
    fPeaks, Peaks, nPeaks = oma.ssi.peak_picking_ssi(x=vf,
                                                     y=20 * np.log10(S),
                                                     freqs=freqs_stable,
                                                     order=order_stable,
                                                     cutoff=cutoff,
                                                     plot='all')
    # Peak-picking
    # fPeaks, Peaks, nPeaks = oma.fdd.peak_picking(vf, 20 * np.log10(S), 20 * np.log10(S2), n_sval=1, cutoff=cutoff)

    '''Extract modal damping by averaging over the damping values of each dataset'''

    wn_fdd, zeta_fdd, PHI_fdd = oma.modal_extract(path=path,
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

    '''Same for SSI'''
    wn_ssi, zeta_ssi, _ = oma.modal_extract_ssi(path=path,
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
                                                plot=False,
                                                mode_extract=False,
                                                cutoff=cutoff)

    # Print Damping and natural frequencies
    print("FDD-Results: ")
    print("Natural Frequencies [Hz]:")
    print(wn_fdd / 2 / np.pi)
    print("Damping [%]:")
    print(zeta_fdd * 100)

    print("SSI-Results: ")
    print("Natural Frequencies [Hz]:")
    print(wn_ssi)
    print("Damping [%]:")
    print(zeta_ssi * 100)

    # additional step in mode shape scaling
    # PHI = oma.mode_shape_normalize(PHI, [0, 1], 2)

    # 2d-Plot modeshapes
    for i in range(nPeaks):
        plt.plot(np.real(PHI_fdd[i, :]), label="Mode: " + str(i + 1))
    plt.legend()
    plt.show()

    '''Standardabweichung für die Dämpfung angeben'''

    # 3d-Plot mode shapes
    discretization = scipy.io.loadmat('Discretizations/TiflisBruecke.mat')
    N = discretization['N']
    E = discretization['E']

    for i in range(nPeaks):
        mode = np.zeros(PHI_fdd.shape[1] + 4, dtype=np.complex_)
        mode[2:-2] = PHI_fdd[i, :]
        title = "Mode " + str(i + 1) + " at " + str(round(wn_fdd[i] / 2 / np.pi, 2)) + "Hz (" + str(
            round(zeta_fdd[i] * 100, 2)) + "%)"
        oma.animate_modeshape(N,
                              E + 1,
                              mode_shape=mode.real,
                              f_n=wn_fdd[i] / 2 / np.pi,
                              zeta_n=zeta_fdd[i],
                              directory="Animations/Tiflis_2/",
                              mode_nr=i,
                              plot=True)
