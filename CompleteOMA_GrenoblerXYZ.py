import numpy as np
import scipy
from OMA import OMA_Module as oma
import pickle

if __name__ == '__main__':

    '''Specify Parameters for OMA'''
    # Specify Sampling frequency
    Fs = 2048

    # Path of Measurement Files and other specifications
    path = "Data/GrenoblerBrueckeXYZ/"
    discretization = scipy.io.loadmat('Discretizations/GrenoblerBruecke.mat')
    n_rov = 2
    n_ref = 0
    ref_channel = None
    rov_channel = [0, 1]
    ref_position = None

    # Cutoff frequency (band of interest)
    cutoff = 55

    # Decide if harmonic estimation needs to be used
    filt = False

    # measurement duration
    t_end = 1200

    # Threshold for MAC
    mac_threshold = 0.98

    # Welch's Method Parameters
    window = 'hann'
    n_seg = 100
    overlap = 0.5

    # SSI-Parameters
    ord_min = 20
    delta_f = 0.2

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
                             cutoff=cutoff * 4,
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
    with open('Data/SSI_Data/freqsGrenoblerXYZ.pkl', 'rb') as f:
        freqs = pickle.load(f)
    with open('Data/SSI_Data/statusGrenoblerXYZ.pkl', 'rb') as f:
        status = pickle.load(f)
    with open('Data/SSI_Data/modesGrenoblerXYZ.pkl', 'rb') as f:
        modes = pickle.load(f)
    with open('Data/SSI_Data/zetaGrenoblerXYZ.pkl', 'rb') as f:
        zeta = pickle.load(f)

    fPeaks, Peaks, nPeaks = oma.ssi.peak_picking_ssi(x=vf,
                                                     y=20 * np.log10(S),
                                                     freqs=freqs,
                                                     label=status,
                                                     ord_min=ord_min,
                                                     cutoff=cutoff)

    '''Extract modal parameters by averaging over each dataset'''
    # FDD
    wn_fdd, zeta_fdd, PHI_fdd, _, _ = oma.modal_extract_fdd(path=path,
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
                                                            plot=True)

    # MPC-Calculations FDD
    MPC_fdd = []
    for i in range(nPeaks):
        MPC_fdd.append(oma.mpc(PHI_fdd[i, :].real, PHI_fdd[i, :].imag))
    # Print Damping and natural frequencies
    print("Natural Frequencies [Hz]:")
    print(wn_fdd)
    print("Damping [%]:")
    print(zeta_fdd * 100)
    print("Modal Phase Collinearity:")
    print(MPC_fdd)

    # SSI
    f_rel = []
    for i in range(nPeaks):
        f_rel.append([fPeaks[i] - delta_f, fPeaks[i] + delta_f])
    wn_ssi, zeta_ssi, PHI_ssi = oma.ssi.ssi_extract(freqs=freqs,
                                                    zeta=zeta,
                                                    modes=modes,
                                                    ranges=f_rel,
                                                    label=status)
    wn_ssi = np.array(wn_ssi)
    zeta_ssi = np.array(zeta_ssi)
    PHI_ssi = np.array(PHI_ssi, dtype=np.complex_).T

    # MPC-Calculations SSI
    MPC_ssi = []
    for i in range(nPeaks):
        MPC_ssi.append(oma.mpc(PHI_ssi[i, :].real, PHI_ssi[i, :].imag))

    # Print Damping and natural frequencies
    print("Natural Frequencies [Hz]:")
    print(wn_ssi)
    print("Damping [%]:")
    print(zeta_ssi * 100)
    print("Modal Phase Collinearity:")
    print(MPC_ssi)

    # Compare the two results using the MAC-Matrix
    # oma.plot_mac_matrix(PHI_ssi, PHI_fdd, wn_ssi, wn_fdd)

    # 3d-Plot all Mode shapes from the FDD
    N = discretization['N']
    E = discretization['E']
    for i in range(nPeaks):
        modeX = np.zeros(PHI_ssi.shape[0]//3 + 4, dtype=np.complex_)
        modeX[2:-2] = PHI_ssi[0:2, i]
        modeY = np.zeros(PHI_ssi.shape[0]//3 + 4, dtype=np.complex_)
        modeY[2:-2] = PHI_ssi[2:4, i]
        modeZ = np.zeros(PHI_ssi.shape[0]//3 + 4, dtype=np.complex_)
        modeZ[2:-2] = PHI_ssi[4:-1, i]
        oma.animate_modeshape_triax(N,
                                    E+1,
                                    mode_x=modeY,
                                    mode_y=modeX,
                                    mode_z=modeZ,
                                    f_n=wn_ssi[i],
                                    zeta_n=zeta_ssi[i],
                                    mpc=MPC_ssi[i],
                                    directory="Animations/Grenobler_FDD/",
                                    mode_nr=i,
                                    plot=True)
