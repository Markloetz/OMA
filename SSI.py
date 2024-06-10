import numpy as np

from OMA import OMA_Module as oma

if __name__ == '__main__':
    # Specify Sampling frequency
    Fs = 2048

    # Cutoff frequency
    cutoff = 50

    # Specify limits
    f_lim = 0.05  # Pole stability (frequency)
    z_lim = 0.05  # Pole stability (damping)
    mac_lim = 0.1 # Mode stability (MAC-Value)
    z_max = 0.10  # Maximum damping value
    limits = [f_lim, z_lim, mac_lim, z_max]

    # block-rows
    br = 8
    ord_max = br*12
    ord_min = 5
    d_ord = 2

    # import data (and plot)
    acc, Fs = oma.import_data(filename='Data/TiflisTotal_2.mat',
                              plot=False,
                              fs=Fs,
                              time=500,
                              detrend=True,
                              downsample=False,
                              cutoff=cutoff)

    # Perform SSI algorithm
    freqs, zeta, modes = oma.ssi.ssi_proc(acc,
                                          fs=Fs,
                                          ord_min=ord_min,
                                          ord_max=ord_max,
                                          d_ord=d_ord)

    # Calculate stable poles
    freqs_stable, zeta_stable, modes_stable, order_stable = oma.ssi.stabilization_calc(freqs, zeta, modes, limits)

    # Plot Stabilization Diagram
    oma.ssi.stabilization_diag(freqs_stable, order_stable, cutoff)

    # Extract modal parameters at relevant frequencies
    f_rel = [[1.5, 2], [6, 6.4], [13, 13.6], [16.9, 17.1]]
    f_n, z_n, ms_n = oma.ssi.ssi_extract(f_rel, freqs_stable[0], zeta_stable[0], modes_stable[0])
