import pickle
import numpy as np
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
    ref_position = None

    # Nodes and Elements of Bridge
    discretization = scipy.io.loadmat('Discretizations/TiflisBruecke.mat')

    # Cutoff frequency (band of interest)
    cutoff = 25

    # measurement duration
    t_end = 500

    # Threshold for MAC
    mac_threshold = 0.99

    # Welch's Method Parameters
    window = 'hann'
    n_seg = 50
    overlap = 0.5

    # SSI-Parameters
    f_lim = 0.01  # Pole stability (frequency)
    z_lim = 0.05  # Pole stability (damping)
    mac_lim = 0.015  # Mode stability (MAC-Value)
    limits = [f_lim, z_lim, mac_lim]
    ord_min = 5
    ord_max = 60
    Ts = 1.8

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
                                    n_seg=n_seg,
                                    window=window,
                                    overlap=overlap)

    # SVD of CPSD-matrix @ each frequency
    S, U, S2, U2 = oma.fdd.sv_decomp(mCPSD)

    ''' Perform SSI '''
    # Reload mat files with stored lists from the SSI on the complete measurement Data
    with open('Data/SSI_Data/freqsTiflis1.pkl', 'rb') as f:
        freqs = pickle.load(f)
    with open('Data/SSI_Data/statusTiflis1.pkl', 'rb') as f:
        status = pickle.load(f)

    fPeaks, Peaks, nPeaks = oma.ssi.peak_picking_ssi(x=vf,
                                                     y=20 * np.log10(S),
                                                     freqs=freqs,
                                                     label=status,
                                                     ord_min=ord_min,
                                                     cutoff=cutoff)

    '''Extract modal damping by averaging over the damping values of each dataset'''
    # FDD
    wn_fdd, zeta_fdd, PHI_fdd, s_dev_fn_fdd, s_dev_zeta_fdd = oma.modal_extract_fdd(path=path,
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
                                                                                    mac_threshold=mac_threshold,
                                                                                    plot=False)
    # MPC-Calculations FDD
    MPC_fdd = []
    for i in range(nPeaks):
        MPC_fdd.append(oma.mpc(PHI_fdd[i, :].real, PHI_fdd[i, :].imag))
    # Print Damping and natural frequencies
    print("Natural Frequencies [Hz]:")
    print(wn_fdd)
    print("...with a standard deviation over all datasets of:")
    print(s_dev_fn_fdd)
    print("Damping [%]:")
    print(zeta_fdd * 100)
    print("...with a standard deviation over all datasets of:")
    print(s_dev_zeta_fdd)
    print("Modal Phase Collinearity:")
    print(MPC_fdd)

    # SSI
    wn_ssi, zeta_ssi, PHI_ssi, s_dev_fn_ssi, s_dev_zeta_ssi = oma.modal_extract_ssi(path=path,
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
                                                                                    plot=False,
                                                                                    cutoff=cutoff,
                                                                                    Ts=Ts)
    # MPC-Calculations SSI
    MPC_ssi = []
    for i in range(nPeaks):
        MPC_ssi.append(oma.mpc(PHI_ssi[i, :].real, PHI_ssi[i, :].imag))

    # Print Damping and natural frequencies
    print("Natural Frequencies [Hz]:")
    print(wn_ssi)
    print("...with a standard deviation over all datasets of:")
    print(s_dev_fn_ssi)
    print("Damping [%]:")
    print(zeta_ssi * 100)
    print("...with a standard deviation over all datasets of:")
    print(s_dev_zeta_ssi)
    print("Modal Phase Collinearity:")
    print(MPC_ssi)

    # Compare the two results using the MAC-Matrix
    oma.plot_mac_matrix(PHI_ssi, PHI_fdd, wn_ssi, wn_fdd)

    # Load custom mesh node data
    ema_results = scipy.io.loadmat('ComparisonData/EMA_modes.mat')
    ema_modes = ema_results['mode'].T
    ema_freqs = [1.87, 6.07, 6.89, 13.34, 16.93]
    nPeaks = len(ema_freqs)

    # Compare the two results using the MAC-Matrix
    oma.plot_mac_matrix(PHI_ssi, ema_modes, wn_ssi, ema_freqs)

    # 3d-Plot all Mode shapes from the FDD
    N = discretization['N']
    E = discretization['E']
    for i in range(nPeaks):
        mode = np.zeros(PHI_fdd.shape[1] + 4, dtype=np.complex_)
        mode[2:-2] = PHI_fdd[i, :]
        oma.animate_modeshape(N,
                              E + 1,
                              mode_shape=mode.real,
                              f_n=wn_fdd[i],
                              zeta_n=zeta_fdd[i],
                              mpc=MPC_fdd[i],
                              directory="Animations/Tiflis1_FDD/",
                              mode_nr=i,
                              plot=True)

    # 3d-Plot all Mode shapes from the SSI
    for i in range(nPeaks):
        mode = np.zeros(PHI_ssi.shape[1] + 4, dtype=np.complex_)
        mode[2:-2] = PHI_ssi[i, :]
        oma.animate_modeshape(N,
                              E + 1,
                              mode_shape=mode.real,
                              f_n=wn_ssi[i],
                              zeta_n=zeta_ssi[i],
                              mpc=MPC_ssi[i],
                              directory="Animations/Tiflis1_SSI/",
                              mode_nr=i,
                              plot=True)


