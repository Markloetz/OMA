import numpy as np
import scipy
from matplotlib import pyplot as plt

from OMA import OMA_Module as oma

if __name__ == '__main__':
    # Specify Sampling frequency
    Fs = 2048

    # Cutoff frequency
    cutoff = 25

    # Specify limits
    f_lim = 0.01  # Pole stability (frequency)
    z_lim = 0.05  # Pole stability (damping)
    mac_lim = 0.9  # Mode stability (MAC-Value)
    z_max = 0.1  # Maximum damping value
    limits = [f_lim, z_lim, mac_lim, z_max]

    # block-rows
    ord_max = 70
    ord_min = 5
    d_ord = 1

    # import data (and plot)
    acc, Fs = oma.import_data(filename='Data/TiflisTotal_2.mat',
                              plot=False,
                              fs=Fs,
                              time=1000,
                              detrend=True,
                              downsample=True,
                              cutoff=cutoff)
    # Perform SSI algorithm
    freqs, zeta, modes, A, C = oma.ssi.ssi_proc(acc,
                                                fs=Fs,
                                                ord_min=ord_min,
                                                ord_max=ord_max,
                                                d_ord=d_ord,
                                                method='CovarianceDriven')

    # Calculate stable poles
    freqs_stable, zeta_stable, modes_stable, order_stable = oma.ssi.stabilization_calc(freqs, zeta, modes, limits)

    # Plot Stabilization Diagram
    _, ax = oma.ssi.stabilization_diag(freqs_stable, order_stable*d_ord, cutoff, plot='all')
    plt.show()

    # Extract modal parameters at relevant frequencies
    f_rel = [[12.5, 13.5], [16.5, 18.5]]
    f_n, z_n, m_n = oma.ssi.ssi_extract(f_rel, freqs_stable[2], zeta_stable[2], modes_stable[2])
    print("Natural Frequencies: ")
    print(f_n)
    print("Modal Damping: ")
    print(z_n)

    # 3d-Plot mode shapes
    discretization = scipy.io.loadmat('Discretizations/TiflisBruecke.mat')
    N = discretization['N']
    E = discretization['E']

    for i in range(len(f_rel)):
        mode = np.zeros(len(m_n[i]) + 4, dtype=np.complex_)
        mode[2:-2] = m_n[i]
        oma.animate_modeshape(N,
                              E + 1,
                              mode_shape=mode,
                              f_n=f_n[i] / 2 / np.pi,
                              zeta_n=zeta[i],
                              directory="Animations/Tiflis_2/",
                              mode_nr=i,
                              plot=True)

