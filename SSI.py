from OMA import OMA_Module as oma


if __name__ == '__main__':
    # Specify Sampling frequency
    Fs = 2048

    # Specify limits
    f_lim = 0.01        # Pole stability (frequency)
    z_lim = 0.05        # Pole stability (damping)
    mac_lim = 0.15      # Mode stability (MAC-Value)
    z_max = 0.20        # Maximum damping value
    limits = [f_lim, z_lim, mac_lim, z_max]

    # import data (and plot)
    acc, Fs = oma.import_data(filename='Data/DataPlateHarmonicInfluence/acc_data_01_09_12_33_harmonic_22_5Hz.csv',
                              plot=False,
                              fs=Fs,
                              time=180,
                              detrend=True,
                              downsample=False,
                              cutoff=1000)

    # generate modal parameters for the stabilization diagram by iteration through the amount of block rows used in SSI
    freqs = []
    zeta = []
    modes = []
    for br in range(acc.shape[1], 30):
        f, z, m = oma.ssi.ssi_proc(acc,
                                   fs=Fs,
                                   br=br,
                                   limits=limits)
        freqs.append(f)
        zeta.append(z)
        modes.append(m)

    # stabilization diagram from modal parameters
