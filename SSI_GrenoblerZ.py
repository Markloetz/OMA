import pickle
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
    cutoff = 55

    # measurement duration
    t_end = 1200

    # SSI-Parameters
    f_lim = 0.01  # Pole stability (frequency)
    z_lim = 0.05  # Pole stability (damping)
    mac_lim = 0.015  # Mode stability (MAC-Value)
    limits = [f_lim, z_lim, mac_lim]
    ord_min = 5
    ord_max = 60
    Ts = 8

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

    # Save Results from SSI
    with open('Data/SSI_Data/freqsGrenoblerZ.pkl', 'wb') as f:
        pickle.dump(freqs, f)
    with open('Data/SSI_Data/modesGrenoblerZ.pkl', 'wb') as f:
        pickle.dump(modes, f)
    with open('Data/SSI_Data/zetaGrenoblerZ.pkl', 'wb') as f:
        pickle.dump(zeta, f)
    with open('Data/SSI_Data/statusGrenoblerZ.pkl', 'wb') as f:
        pickle.dump(status, f)