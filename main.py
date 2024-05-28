from OMA import OMA_Module as oma

if __name__ == "__main__":
    # Specify Sampling frequency
    Fs = 2048

    # import data (and plot)
    acc, Fs = oma.import_data(filename='Data/DataPlateHarmonicInfluence/acc_data_01_09_12_33.csv',
                              plot=False,
                              fs=Fs,
                              time=0.5,
                              detrend=True,
                              downsample=False,
                              cutoff=100)

    H = oma.ssi.block_hankel_matrix(data=acc,
                                    br=30)
