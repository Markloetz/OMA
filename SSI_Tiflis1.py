import pickle
import scipy
from matplotlib import pyplot as plt

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
    ref_position = None

    # Nodes and Elements of Bridge
    discretization = scipy.io.loadmat('Discretizations/TiflisBruecke.mat')

    # Cutoff frequency (band of interest)
    cutoff = 25

    # measurement duration
    t_end = 500

    # SSI-Parameters
    f_lim = 0.01  # Pole stability (frequency)
    z_lim = 0.05  # Pole stability (damping)
    mac_lim = 0.015  # Mode stability (MAC-Value)
    limits = [f_lim, z_lim, mac_lim]
    ord_min = 5
    ord_max = 60
    Ts = 0.8

    '''SVD Procedure on SV-diagram of the whole dataset'''
    acc, Fs = oma.merge_data(path=path,
                             fs=Fs,
                             n_rov=n_rov,
                             n_ref=n_ref,
                             ref_channel=ref_channel,
                             rov_channel=rov_channel,
                             ref_pos=ref_position,
                             t_meas=t_end,
                             detrend=True,
                             cutoff=cutoff * 4,
                             downsample=False)

    freqs, zeta, modes, _, _, status = oma.ssi.SSICOV(acc,
                                                      dt=1 / Fs,
                                                      Ts=Ts,
                                                      ord_min=ord_min,
                                                      ord_max=ord_max,
                                                      limits=limits)

    oma.ssi.stabilization_diag(freqs=freqs, label=status, order_min=ord_min, cutoff=cutoff)
    plt.show()

    # Save Results from SSI
    with open('Data/SSI_Data/freqsTiflis1.pkl', 'wb') as f:
        pickle.dump(freqs, f)
    with open('Data/SSI_Data/modesTiflis1.pkl', 'wb') as f:
        pickle.dump(modes, f)
    with open('Data/SSI_Data/zetaTiflis1.pkl', 'wb') as f:
        pickle.dump(zeta, f)
    with open('Data/SSI_Data/statusTiflis1.pkl', 'wb') as f:
        pickle.dump(status, f)
