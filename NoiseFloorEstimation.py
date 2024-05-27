import numpy as np
from OMA import OMA_Module as oma
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Specify Sampling frequency
    Fs = 2048

    # import data (and plot)
    acc, Fs = oma.import_data(filename='Data/NoiseFloorEstimation/acc_data_270524_4507.csv',
                              plot=True,
                              fs=Fs,
                              time=180,
                              detrend=True,
                              downsample=False,
                              cutoff=1000)

    # Calculate necessary aut- and CrossSpectralPowerDensities
    mCPSD, vf = oma.fdd.cpsd_matrix(data=acc,
                                    fs=Fs,
                                    zero_padding=True)

    # The autospectral densities are positioned on the diagonal
    S11 = mCPSD[0, 0, :]
    S22 = mCPSD[1, 1, :]
    S12 = mCPSD[0, 1, :]

    # Noise
    Snn = np.sqrt(S11 * S22) - np.abs(S12)

    # Plot results
    plt.plot(vf, 20 * np.log(S11.real), label='Spectral Density Sensor 1')
    plt.plot(vf, 20 * np.log(S22.real), label='Spectral Density Sensor 2')
    plt.plot(vf, 20 * np.log(Snn.real), label='Spectral Density Noise')
    plt.xlim([0, 200])
    plt.legend()
    plt.show()
