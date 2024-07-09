import numpy as np
from OMA import OMA_Module as oma

if __name__ == '__main__':

    '''Specify Parameters for OMA'''
    # Specify Sampling frequency
    Fs = 2048

    # Path of Measurement Files and other specifications
    path = "Data/GrenoblerBruecke/z-data/"
    n_rov = 2
    n_ref = 0
    ref_channel = None
    rov_channel = [0, 1]
    ref_position = None

    # Cutoff frequency (band of interest)
    cutoff = 25

    # Decide if harmonic estimation needs to be used
    filt = False

    # measurement duration
    t_end = 1200

    # Threshold for MAC
    mac_threshold = 0.98

    # Welch's Method Parameters
    window = 'hann'
    n_seg = 125
    overlap = 0.5

    # SSI-Parameters
    f_lim = 0.01  # Pole stability (frequency)
    z_lim = 0.05  # Pole stability (damping)
    mac_lim = 0.01  # Mode stability (MAC-Value)
    limits = [f_lim, z_lim, mac_lim]
    Ts = 2          # Time lag of covariance matrix [s]
    ord_min = 20
    ord_max = 60

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

    # Filter the harmonics
    if filt:
        f_harmonic = oma.fdd.harmonic_est(data=acc, delta_f=0.1, f_max=cutoff, fs=Fs, plot=True)
        S = oma.fdd.eliminate_harmonic(vf, 20 * np.log10(S), f_harmonic[1:-1], cutoff=cutoff)

    # SSI
    freqs, zeta, modes, _, _, status = oma.ssi.SSICOV(acc,
                                                      dt=1 / Fs,
                                                      Ts=Ts,
                                                      ord_min=ord_min,
                                                      ord_max=ord_max,
                                                      limits=limits)

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
    wn_ssi, zeta_ssi, PHI_ssi, _, _ = oma.modal_extract_ssi(path=path,
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
                                                            plot=True,
                                                            cutoff=cutoff,
                                                            Ts=Ts,
                                                            delta_f=0.2)
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
    oma.plot_mac_matrix(PHI_ssi, PHI_fdd, wn_ssi, wn_fdd)

