import numpy as np
import matplotlib.pyplot as plt
import scipy
from OMA import OMA_Module as oma

if __name__ == '__main__':

    '''Specify Parameters for OMA'''
    # Specify Sampling frequency
    Fs = 2048

    # Filename
    path = "Data/Platte_Harmonic/"

    # Cutoff frequency (band of interest)
    cutoff = 100

    # measurement duration
    t_end = 180

    # Threshold for MAC
    mac_threshold = 0.99

    # SSI-Parameters
    # Specify limits
    f_lim = 0.01  # Pole stability (frequency)
    z_lim = 0.02  # Pole stability (damping)
    mac_lim = 0.015  # Mode stability (MAC-Value)
    limits = [f_lim, z_lim, mac_lim]

    # block-rows
    ord_max = 20
    ord_min = 0

    # Welch's Method Parameters
    window = 'hann'
    n_seg = 50
    overlap = 0.5
    zero_padding = False

    # import data
    acc, Fs = oma.merge_data(path=path,
                             fs=Fs,
                             n_rov=4,
                             n_ref=0,
                             ref_channel=0,
                             rov_channel=[0, 1, 2, 3],
                             ref_pos=0,
                             t_meas=t_end,
                             detrend=True,
                             cutoff=cutoff,
                             downsample=False)

    ''' Peak Picking Procedure on SV-diagram of the whole dataset '''
    # Build CPSD-Matrix from acceleration data
    mCPSD, vf = oma.fdd.cpsd_matrix(data=acc,
                                    fs=Fs,
                                    zero_padding=zero_padding,
                                    n_seg=n_seg,
                                    window=window,
                                    overlap=overlap)

    # SVD of CPSD-matrix @ each frequency
    S, U, S2, U2 = oma.fdd.sv_decomp(mCPSD)

    # Harmonic Filtering
    # f_harmonic = oma.fdd.harmonic_est(data=acc, delta_f=0.25, f_max=cutoff, fs=Fs, plot=True)
    # S = oma.fdd.eliminate_harmonic(vf, 20*np.log10(S), f_harmonic[1:-1])

    ''' SSI '''
    freqs, zeta, modes, _, _, status = oma.ssi.SSICOV(acc,
                                                      dt=1 / Fs,
                                                      Ts=0.8,
                                                      ord_min=ord_min,
                                                      ord_max=ord_max,
                                                      limits=limits)
    # Peak Picking
    fPeaks, Peaks, nPeaks = oma.ssi.peak_picking_ssi(x=vf,
                                                     y=20 * np.log10(S),
                                                     freqs=freqs,
                                                     cutoff=cutoff,
                                                     label=status)

    # extract mode shape at each peak to get the modal coherence plot
    oma.modal_coherence_plot(f=vf,
                             s=20*np.log10(S),
                             u=U,
                             f_peaks=fPeaks,
                             cutoff=cutoff)

    # Extract modal parameters from FDD
    wnfdd, zetafdd, PHIfdd = oma.modal_extract(path=path,
                                               Fs=Fs,
                                               n_rov=4,
                                               n_ref=0,
                                               ref_channel=0,
                                               ref_pos=[0, 1, 2, 3],
                                               t_meas=t_end,
                                               fPeaks=fPeaks,
                                               window=window,
                                               overlap=overlap,
                                               n_seg=n_seg,
                                               zeropadding=zero_padding,
                                               mac_threshold=mac_threshold,
                                               plot=False)

    # Print Damping and natural frequencies
    print("FDD-Results: ")
    print("Natural Frequencies [Hz]:")
    print(wnfdd / 2 / np.pi)
    print("Damping [%]:")
    print(zetafdd * 100)