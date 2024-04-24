import numpy as np
import matplotlib.pyplot as plt
import FDD_Module as fdd


if __name__ == '__main__':
    # Specify Sampling frequency
    Fs = 1000

    # Threshold for MAC
    mac_threshold = 0.85

    # import data (and plot)
    acc = fdd.import_data('MDOF_Data.csv', False, Fs, 300)

    # Build CPSD-Matrix from acceleration data
    mCPSD, vf = fdd.cpsd_matrix(acc, Fs)

    # SVD of CPSD-matrix @ each frequency
    S, U, S2, U2, S3, U3 = fdd.sv_decomp(mCPSD)

    # Peak-picking
    fPeaks, Peaks, nPeaks = fdd.peak_picking(vf, S, S2, S3, Fs)

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
        fSDOF[:len(indSDOF), i] = vf[indSDOF]
        sSDOF[:len(indSDOF), i] = S[indSDOF, 0]

    # Plotting the singular values
    plt.plot(fPeaks, Peaks, marker='o', linestyle='none')
    for i in range(nPeaks):
        fSDOF_temp_1 = fSDOF[:, i]
        sSDOF_temp_1 = sSDOF[:, i]
        fSDOF_temp_2 = fSDOF[~np.isnan(fSDOF_temp_1)]
        sSDOF_temp_2 = sSDOF[~np.isnan(sSDOF_temp_1)]
        plt.plot(fSDOF_temp_2, sSDOF_temp_2)
    plt.xlabel('Frequency')
    plt.ylabel('Singular Values')
    plt.title('Singular Value Plot')
    plt.grid(True)
    plt.show()

    # Fitting SDOF in frequency domain
    wn = np.zeros((nPeaks, 1))
    zeta = np.zeros((nPeaks, 1))
    for i in range(nPeaks):
        wn[i, :], zeta[i, :] = fdd.sdof_frf_fit(sSDOF[:, i], fSDOF[:, i])

    # Print Damping and natural frequencies
    print(wn/2/np.pi)
    print(zeta)

    # Plot Fitted SDOF-Bell-Functions
    # Determine the number of rows and columns
    if nPeaks != 0:
        num_rows = (nPeaks + 1) // 2
        num_cols = 2 if nPeaks > 1 else 1
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 8))
        for i in range(nPeaks):
            # Frequency vector
            freq_start = fSDOF[~np.isnan(fSDOF[:, i])][0][i]
            freq_end = fSDOF[~np.isnan(fSDOF[:, i])][-1][i]
            freq_vec = np.linspace(freq_start, freq_end, 1000)
            sSDOF_fit = fdd.sdof_frf(freq_vec, 1, (wn[i, :]**2), zeta[i, :])
            scaling_factor = max(sSDOF[:, i])/max(sSDOF_fit)
            if num_cols != 1:
                axs[i // num_cols, i % num_cols].plot(fSDOF[:, i], sSDOF[:, i].real)
                axs[i // num_cols, i % num_cols].plot(freq_vec, sSDOF_fit*scaling_factor)
                axs[i // num_cols, i % num_cols].set_title(f'SDOF-Fit {i + 1}')
            else:
                axs.plot(fSDOF[:, i], sSDOF[:, i].real)
                axs.plot(freq_vec, sSDOF_fit*scaling_factor)
                axs.set_title(f'SDOF-Fit {i + 1}')

        # Adjust layout and log scale axis
        plt.tight_layout()

        # Show the plot
        plt.show()
