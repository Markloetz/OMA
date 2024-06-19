import numpy as np
import matplotlib.pyplot as plt
import scipy
from OMA import OMA_Module as oma

if __name__ == '__main__':

    '''Specify Parameters for OMA'''
    # Specify Sampling frequency
    Fs = 2048

    # Filename and Path
    path = "Data/Platte/"
    filename = "Data/PlatteTotal.mat"

    # Cutoff frequency (band of interest)
    cutoff = 200

    # measurement duration
    t_end = 500

    # Threshold for MAC
    mac_threshold = 0.95

    # Decide if harmonic filtering is active
    filt = False

    # Welch's Method Parameters
    window = 'hann'
    n_seg = 100
    overlap = 0.5
    zero_padding = False

    '''Peak Picking Procedure on SV-diagram of the whole dataset'''
    # import data (and plot)
    acc, Fs = oma.import_data(filename=filename,
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

    '''Extract modal damping by averaging over the damping values of each dataset'''
    '''Average the fitted frequencies'''
    '''Merge and Scale the Mode Shapes'''
    wn, zeta, PHI = oma.modal_extract(path=path,
                                      Fs=Fs,
                                      n_rov=2,
                                      n_ref=2,
                                      ref_channel=[2, 3],
                                      ref_pos=[1, 15],
                                      t_meas=t_end,
                                      fPeaks=fPeaks,
                                      window=window,
                                      overlap=overlap,
                                      n_seg=n_seg,
                                      zeropadding=zero_padding,
                                      mac_threshold=mac_threshold,
                                      plot=False)

    # Print Damping and natural frequencies
    print("Natural Frequencies [Hz]:")
    print(wn / 2 / np.pi)
    print("Damping [%]:")
    print(zeta * 100)

    # 2d-Plot modeshapes
    for i in range(nPeaks):
        plt.plot(np.real(PHI[i, :]), label="Mode: " + str(i + 1))
    plt.legend()
    plt.show()

    # 3d-Plot mode shapes
    discretization = scipy.io.loadmat('Discretizations/PlateHoleDiscretization.mat')
    N = discretization['N']
    E = discretization['E']

    for i in range(nPeaks):
        mode = PHI[i, :].real
        oma.animate_modeshape(N,
                              E,
                              mode_shape=mode,
                              f_n=wn[i] / 2 / np.pi,
                              zeta_n=zeta[i],
                              directory="Animations/Plate/",
                              mode_nr=i+1,
                              plot=True)
