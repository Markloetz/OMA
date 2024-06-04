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
    mac_threshold = 0.7

    # Decide if harmonic filtering is active
    filt = False

    # Decide if the modes need to be scaled (and where to find the data for scaling)
    scaling = True
    path = "Data/SDL_Floor/"

    # Welch's Method Parameters
    window = 'hann'
    n_seg = 100
    overlap = 0.5
    zero_padding = False

    # import data (and plot)
    acc, Fs = oma.import_data(filename="Data/SDL_FloorTotal.mat",
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
    for i in range(nPeaks):
        for j in range(nMAC):
            mac = oma.fdd.mac_calc(PHI[i, :], U[j, :])
            if mac.real < mac_threshold:
                mac_vec[j, i] = 0
            else:
                mac_vec[j, i] = mac

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
    plt.xlabel('Frequency')
    plt.ylabel('Singular Values')
    plt.title('Singular Value Plot')
    plt.grid(True)
    plt.show()

    # Scaling the modeshapes
    alpha = oma.modescale(path=path,
                          Fs=Fs,
                          n_rov=2,
                          n_ref=1,
                          ref_channel=2,
                          t_meas=t_end,
                          fPeaks=fPeaks,
                          Peaks=Peaks,
                          window=window,
                          overlap=overlap,
                          n_seg=n_seg,
                          zeropadding=zero_padding)
    if not scaling:
        alpha = np.ones(alpha.shape)
    PHI = PHI * alpha

    # Fitting SDOF in frequency domain
    wn = np.zeros((nPeaks, 1))
    zeta = np.zeros((nPeaks, 1))
    for i in range(nPeaks):
        wn[i, :], zeta[i, :] = oma.fdd.sdof_time_domain_fit(sSDOF[:, i], vf, Fs, n_skip=0, n_peaks=30)
    # Print Damping and natural frequencies
    print("Natural Frequencies [Hz]:")
    print(wn / 2 / np.pi)
    print("Damping [%]:")
    print(zeta*100)

    # 2d-Plot modeshapes
    ms = oma.modeshape_scaling(np.abs(PHI))
    for i in range(nPeaks):
        plt.plot(np.real(PHI[i, :]), label="Mode: " + str(i + 1))
    plt.legend()
    plt.show()

    # 3d-Plot mode shapes
    discretization = scipy.io.loadmat('Discretizations/SDL_Floor.mat')
    N = discretization['N']
    E = discretization['E']
    for i in range(nPeaks):
        oma.plot_modeshape(N,
                           E + 1,
                           PHI[i, :].real,
                           title="Mode " + str(i+1) + " at " + str(round(wn[i, :][0] / 2 / np.pi, 2)) + "Hz")
