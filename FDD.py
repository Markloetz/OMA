import numpy as np
import matplotlib.pyplot as plt
import scipy
from OMA import OMA_Module as oma

if __name__ == '__main__':
    # Specify Sampling frequency
    Fs = 2048

    # Cutoff frequency (band of interest)
    cutoff = 200
    # measurement duration
    t_end = 500

    # Threshold for MAC
    mac_threshold = 0.85

    # Decide if harmonic filtering is active
    filt = False

    # Decide if the modes need to be scaled (and where to find the data for scaling)
    scaling = False
    path = "Data/Platte/"

    # Welch's Method Parameters
    window = 'hann'
    n_seg = 100
    overlap = 0.5
    zero_padding = False

    # import data (and plot)
    acc, Fs = oma.import_data(filename="Data/Platte/Data_120624_pos_35_36_01_15.mat",
                              plot=False,
                              fs=Fs,
                              time=t_end,
                              detrend=True,
                              downsample=False,
                              cutoff=cutoff)

    # Build CPSD-Matrix from acceleration data
    mCPSD, vf = oma.fdd.cpsd_matrix(data=acc,
                                    fs=Fs,
                                    zero_padding=zero_padding,
                                    n_seg=n_seg,
                                    window=window,
                                    overlap=overlap)

    # SVD of CPSD-matrix @ each frequency
    S, U, S2, U2 = oma.fdd.sv_decomp(mCPSD)

    # Eliminate harmonic frequency bands (cut out harmonic peaks and interpolate)
    if filt:
        f_harmonic = oma.fdd.harmonic_est(data=acc, delta_f=0.25, f_max=cutoff, fs=Fs, plot=True)
        S = oma.fdd.eliminate_harmonic(vf, S, f_harmonic)

    # Peak-picking
    fPeaks, Peaks, nPeaks = oma.fdd.peak_picking(vf, 20 * np.log10(S), 20 * np.log10(S2), n_sval=1, cutoff=cutoff)

    # extract mode shape at each peak
    _, mPHI = U.shape
    PHI = np.zeros((nPeaks, mPHI), dtype=np.complex_)
    for i in range(nPeaks):
        PHI[i, :] = U[np.where(vf == fPeaks[i]), :]

    # EFDD-Procedure
    # calculate mac value @ each frequency for each peak
    nMAC, _ = S.shape
    mac_vec = np.zeros((nMAC, nPeaks), dtype=np.complex_)
    # average modeshape
    PHI_avg = np.zeros((nPeaks, mPHI), dtype=np.complex_)
    for i in range(nPeaks):
        PHI_correlated = np.zeros((nMAC, mPHI), dtype=np.complex_)
        for j in range(nMAC):
            mac = oma.fdd.mac_calc(PHI[i, :], U[j, :])
            if mac.real < mac_threshold:
                mac_vec[j, i] = 0
            else:
                mac_vec[j, i] = mac
                PHI_correlated[j, :] = U[j, :]
        PHI_avg[i, :] = np.mean(PHI_correlated[np.where(PHI_correlated != 0)], axis=0)
    # PHI = PHI_avg

    # Filter the SDOFs
    # Find non-zero indices
    fSDOF = np.full((nMAC, nPeaks), np.nan)
    sSDOF = np.full((nMAC, nPeaks), np.nan)
    for i in range(nPeaks):
        indSDOF = oma.fdd.find_widest_range(mac_vec[:, i].real, np.where(vf == fPeaks[i])[0])
        fSDOF[indSDOF, i] = vf[indSDOF]
        sSDOF[indSDOF, i] = S[indSDOF, 0]

    # Plotting the singular values
    for i in range(nPeaks):
        fSDOF_temp_1 = fSDOF[:, i]
        sSDOF_temp_1 = sSDOF[:, i]
        fSDOF_temp_2 = fSDOF_temp_1[~np.isnan(fSDOF_temp_1)]
        sSDOF_temp_2 = sSDOF_temp_1[~np.isnan(sSDOF_temp_1)]
        color = ((nPeaks - i) / nPeaks, (i + 1) / nPeaks, 0.5, 1)
        plt.plot(fSDOF_temp_2, 20 * np.log10(sSDOF_temp_2), color=color)
    plt.plot(fPeaks, 20 * np.log10(Peaks), marker='o', linestyle='none')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Singular Values (dB)')
    plt.title('Singular Values of SDOF Equivalents')
    plt.grid(True)
    plt.show()

    if scaling:
        # Scaling the mode shapes
        alpha = oma.modescale(path=path,
                              Fs=Fs,
                              n_rov=2,
                              n_ref=1,
                              ref_channel=[1, 2],
                              t_meas=t_end,
                              fPeaks=fPeaks,
                              window=window,
                              overlap=overlap,
                              n_seg=n_seg,
                              zeropadding=zero_padding)
        # Normalize PHI
        PHI = PHI * alpha
        for i in range(nPeaks):
            PHI[i, :] = PHI[i, :].real / np.max(np.abs(PHI[i, :].real))
            print(PHI[i, :])
    # Fitting SDOF in frequency domain
    wn = np.zeros((nPeaks, 1))
    zeta = np.zeros((nPeaks, 1))
    for i in range(nPeaks):
        wn[i, :], zeta[i, :] = oma.fdd.sdof_time_domain_fit(sSDOF[:, i], vf, n_skip=0, n_peaks=30, plot=False)
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

    # 3d-Plot mode shapes
    discretization = scipy.io.loadmat('Discretizations/TiflisBruecke.mat')
    N = discretization['N']
    E = discretization['E']

    # for i in range(nPeaks):
    #     mode = np.zeros(PHI.shape[1] + 4)
    #     mode[2:-2] = PHI[i, :].real
    #     oma.animate_modeshape(N,
    #                           E + 1,
    #                           mode_shape=mode,
    #                           title="Mode " + str(i + 1) + " at " + str(round(wn[i][0] / 2 / np.pi, 2)) + "Hz")
