import numpy as np
import matplotlib.pyplot as plt
import FDD_Module as fdd
import scipy

# Implement Filter-Script: Detrending (Highpass Filter) -> Check for Gaussian Distribution in Segments -> Page225

if __name__ == '__main__':
    # Specify Sampling frequency
    Fs = 2048

    # Threshold for MAC
    mac_threshold = 0.85

    # import data (and plot)
    acc, Fs = fdd.import_data(filename='Data/ShakerOMA/acc_data_01_09_12_33_harmonic_35Hz.csv',
                              plot=False,
                              fs=Fs,
                              time=180,
                              detrend=True,
                              downsample=False,
                              gausscheck=False,
                              cutoff=100)
    # find harmonic influences
    # f_harmonic = fdd.harmonic_est(data=acc, delta_f=1, f_max=100, fs=Fs, threshold=30)

    # Build CPSD-Matrix from acceleration data
    mCPSD, vf = fdd.cpsd_matrix(data=acc,
                                fs=Fs,
                                zero_padding=True)

    # SVD of CPSD-matrix @ each frequency
    S, U, S2, U2 = fdd.sv_decomp(mCPSD)

    # Peak-picking
    fPeaks, Peaks, nPeaks = fdd.peak_picking(vf, 20 * np.log10(S), 20 * np.log10(S2), n_sval=1)

    # extract mode shape at each peak
    _, mPHI = U.shape
    PHI = np.zeros((nPeaks, mPHI), dtype=np.complex_)
    for i in range(nPeaks):
        PHI[i, :] = U[np.where(vf == fPeaks[i]), :]
    # EFDD-Procedure
    # calculate mac value @ each frequency for each peak
    nMAC, _ = S.shape
    mac_vec = np.zeros((nMAC, nPeaks), dtype=np.complex_)
    for i in range(nPeaks):
        for j in range(nMAC):
            mac = fdd.mac_calc(PHI[i, :], U[j, :])
            if mac.real < mac_threshold:
                mac_vec[j, i] = 0
            else:
                mac_vec[j, i] = mac

    # Filter the SDOFs
    # Find non-zero indices
    fSDOF = np.full((nMAC, nPeaks), np.nan)
    sSDOF = np.full((nMAC, nPeaks), np.nan)
    for i in range(nPeaks):
        indSDOF = fdd.find_widest_range(mac_vec[:, i].real, np.where(vf == fPeaks[i])[0])
        fSDOF[indSDOF, i] = vf[indSDOF]
        sSDOF[indSDOF, i] = S[indSDOF, 0]

    # Plotting the singular values
    for i in range(nPeaks):
        fSDOF_temp_1 = fSDOF[:, i]
        sSDOF_temp_1 = sSDOF[:, i]
        fSDOF_temp_2 = fSDOF_temp_1[~np.isnan(fSDOF_temp_1)]
        sSDOF_temp_2 = sSDOF_temp_1[~np.isnan(sSDOF_temp_1)]
        color = ((nPeaks - i) / nPeaks, (i + 1) / nPeaks, 0.5, 1)
        plt.plot(fSDOF_temp_2, sSDOF_temp_2, color=color)
    plt.plot(fPeaks, Peaks, marker='o', linestyle='none')
    plt.xlabel('Frequency')
    plt.ylabel('Singular Values')
    plt.title('Singular Value Plot')
    plt.grid(True)
    plt.show()

    # Fitting SDOF in frequency domain
    wn = np.zeros((nPeaks, 1))
    zeta = np.zeros((nPeaks, 1))
    for i in range(nPeaks):
        # wn[i, :], zeta[i, :] = fdd.sdof_frf_fit(sSDOF[:, i], fSDOF[:, i], fPeaks[i])
        # wn[i, :], zeta[i, :], _ = fdd.sdof_cf(fSDOF[:, i], sSDOF[:, i])
        # wn[i, :], zeta[i, :] = fdd.sdof_half_power(fSDOF[:, i], sSDOF[:, i], fPeaks[i])
        wn[i, :], zeta[i, :] = fdd.sdof_time_domain_fit(sSDOF[:, i], vf, Fs, n_skip=0, n_peaks=30)
    # Print Damping and natural frequencies
    print(wn / 2 / np.pi)
    print(zeta)

    # Plot Fitted SDOF-Bell Functions
    # fdd.plot_fit(fSDOF, sSDOF, wn, zeta)
    for i in range(nPeaks):
        ms = fdd.modeshape_scaling(PHI[i, :].real)
        plt.plot(ms, label="Mode: "+str(i+1))
    plt.legend()
    plt.show
    # Plot mode shapes
    mode = np.zeros((nPeaks, 38))
    for i in range(nPeaks):
        mode_locs = np.array([0, 8, 11, 32])
        mode[i, mode_locs] = PHI[i, :].real
    discretization = scipy.io.loadmat('PlateHoleDiscretization.mat')
    N = discretization['N']
    E = discretization['E']
    for i in range(nPeaks):
        fdd.plot_modeshape(N, E, mode[i, :])
