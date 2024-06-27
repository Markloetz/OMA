from matplotlib import pyplot as plt

from OMA import OMA_Module as oma
import numpy as np
from OMA import OMA_Module as oma

if __name__ == '__main__':
    '''Specify Parameters for OMA'''
    # Specify Sampling frequency
    Fs = 2048

    # Path of Measurement Files and other specifications
    path = "Data/TiflisBruecke/"
    n_rov = 2
    n_ref = 1
    ref_channel = 0
    rov_channel = [1, 2]
    ref_position = 0

    # Cutoff frequency (band of interest)
    cutoff = 25

    # measurement duration
    t_end = 500

    # SSI-Parameters
    # Specify limits
    f_lim = 0.01  # Pole stability (frequency)
    z_lim = 0.04  # Pole stability (damping)
    mac_lim = 0.05  # Mode stability (MAC-Value)
    limits = [f_lim, z_lim, mac_lim]

    # block-rows
    ord_max = 50
    ord_min = 10

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
                             cutoff=cutoff * 2,
                             downsample=False)

    # SSI
    # Perform SSI
    freqs, zeta, modes, orders, mac, status = oma.ssi.SSICOV(acc,
                                                             dt=1 / Fs,
                                                             Ts=0.2,
                                                             ord_min=ord_min,
                                                             ord_max=ord_max,
                                                             limits=limits)

    print(freqs)
    print(status)
    print(orders)


    # stabilization diag
    fig, ax = oma.ssi.stabilization_diag(freqs, orders, status, cutoff)
    plt.show()
