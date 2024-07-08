import matplotlib.pyplot as plt
import numpy as np
import scipy
from OMA import OMA_Module as oma

if __name__ == '__main__':

    '''Specify Parameters for OMA'''
    # Specify Sampling frequency
    Fs = 2048

    # Path of Measurement Files and other specifications
    path = "Data/GrenoblerBruecke/"
    n_rov = 2
    n_ref = None
    ref_channel = None
    rov_channel = [0, 1]
    ref_position = None

    # Cutoff frequency (band of interest)
    cutoff = 30

    # measurement duration
    t_end = 1200

    # Threshold for MAC
    mac_threshold = 0.99

    # Decide if harmonic filtering is active
    filt = False

    # Welch's Method Parameters
    window = 'hann'
    n_seg = 100
    overlap = 0.5

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

    # Eliminate harmonic frequency bands (cut out harmonic peaks and interpolate)
    if filt:
        f_harmonic = oma.fdd.harmonic_est(data=acc, delta_f=0.5, f_max=cutoff, fs=Fs, plot=True)
        S = oma.fdd.eliminate_harmonic(vf, 20 * np.log10(S), f_harmonic[1:-1], cutoff=cutoff)

    # Peak-picking
    fPeaks, Peaks, nPeaks = oma.fdd.peak_picking(vf, 20 * np.log10(S), 20 * np.log10(S2), n_sval=2, cutoff=cutoff)

    '''Extract modal damping by averaging over the damping values of each dataset'''
    # Scaling the mode shapes
    fn, zeta, PHI, _, _ = oma.modal_extract_fdd(path=path,
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

    # MPC-Calculations
    MPC = []
    for i in range(nPeaks):
        MPC.append(oma.mpc(PHI[i, :].real, PHI[i, :].imag))

    # Print Damping and natural frequencies
    print("Natural Frequencies [Hz]:")
    print(fn)
    print("Damping [%]:")
    print(zeta * 100)
    print("Modal Phase Collinearity:")
    print(MPC)

    # 2d-Plot mode shapes
    for i in range(nPeaks):
        plt.plot(PHI[i, :].real, label=f"Mode{i+1}")
    plt.legend()
    plt.show()
