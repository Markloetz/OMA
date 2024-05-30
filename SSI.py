import numpy as np

from OMA import OMA_Module as oma

if __name__ == '__main__':
    # Specify Sampling frequency
    Fs = 2048

    # Specify limits
    f_lim = 0.01  # Pole stability (frequency)
    z_lim = 0.05  # Pole stability (damping)
    mac_lim = 0.15  # Mode stability (MAC-Value)
    z_max = 0.20  # Maximum damping value
    limits = [f_lim, z_lim, mac_lim, z_max]

    # block-rows
    br_max = 25
    br_min = 5
    d_br = 5

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
    for br in range(br_min, br_max, d_br):
        print(str(br) + "/" + str((br_max - br_min)))
        f, z, m = oma.ssi.ssi_proc(acc,
                                   fs=Fs,
                                   br=br)
        freqs.append(f)
        zeta.append(z)
        modes.append(m)

    # stabilization diagram from modal parameters
    for i in range(len(freqs)):
        for j in range(len(freqs[i])):
            if i > 0:
                # Find closest freqency to current one
                pole_idx = np.argmin(np.abs(freqs[i][j] - freqs[i - 1]))
                f_old = freqs[i - 1][pole_idx]
                f_cur = freqs[i][j]
                # same for damping and modes
                z_old = zeta[i - 1][pole_idx]
                z_cur = zeta[i][j]
                m_old = modes[i - 1][:, pole_idx]
                m_cur = modes[i][:, j]

                # Store frequencies fulfilling certain contitions in seperate Lists
                # stable in frequency, damping and mode shape
                if np.abs(f_old - f_cur) / f_cur <= limits[0] and \
                        np.abs(z_old - z_cur) / z_cur <= limits[1] and \
                        oma.ssi.mac_calc(m_old, m_cur) <= (1 - limits[2]):
                    print(f_cur)
                    print(z_cur)
                    print(m_cur)
                # stable in frequency and damping
                elif np.abs(f_old - f_cur) / f_cur <= limits[0] and \
                        np.abs(z_old - z_cur) / z_cur <= limits[1]:
                    print('_')
                # stable in frequency:
                elif np.abs(f_old - f_cur) / f_cur <= limits[0]:
                    print('#')
                else:
                    print('--')
