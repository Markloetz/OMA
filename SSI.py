import numpy as np
import scipy

from OMA import OMA_Module as oma

if __name__ == '__main__':
    # Specify Sampling frequency
    Fs = 2048

    # Cutoff frequency
    cutoff = 50

    # Specify limits
    f_lim = 0.05  # Pole stability (frequency)
    z_lim = 0.05  # Pole stability (damping)
    mac_lim = 0.3  # Mode stability (MAC-Value)
    z_max = 0.05  # Maximum damping value
    limits = [f_lim, z_lim, mac_lim, z_max]

    # block-rows
    br = 6
    ord_max = br * 12
    ord_min = 0
    d_ord = 2

    # import data (and plot)
    acc, Fs = oma.import_data(filename='Data/TiflisTotal.mat',
                              plot=False,
                              fs=Fs,
                              time=500,
                              detrend=True,
                              downsample=False,
                              cutoff=cutoff)

    # Perform SSI algorithm
    freqs, zeta, modes, A, C = oma.ssi.ssi_proc(acc,
                                                fs=Fs,
                                                ord_min=ord_min,
                                                ord_max=ord_max,
                                                d_ord=d_ord)

    # Create averaged response function

    # Calculate stable poles
    freqs_stable, zeta_stable, modes_stable, order_stable = oma.ssi.stabilization_calc(freqs, zeta, modes, limits)

    # Plot Stabilization Diagram
    oma.ssi.stabilization_diag(freqs_stable, order_stable, cutoff, plot='FDM')

    # Extract modal parameters at relevant frequencies
    f_rel = [[12.5, 13.5], [17.5, 18.5]]
    f_n, z_n, m_n = oma.ssi.ssi_extract(f_rel, freqs_stable[0], zeta_stable[0], modes_stable[0])
    print("Natural Frequencies: ")
    print(f_n)
    print("Modal Damping: ")
    print(z_n)

    # 3d-Plot mode shapes
    discretization = scipy.io.loadmat('Discretizations/TiflisBruecke.mat')
    N = discretization['N']
    E = discretization['E']

    for i in range(len(f_rel)):
        mode = np.zeros(len(m_n[i]) + 4)
        mode[2:-2] = m_n[i].real
        oma.animate_modeshape(N,
                              E + 1,
                              mode_shape=mode,
                              title="Mode " + str(i + 1) + " at " + str(round(f_n[i], 2)) + "Hz")

