# Rowing reference processing
from OMA import OMA_Module as oma

if __name__ == '__main__':
    # Specify Sampling frequency
    Fs = 100

    # Threshold for MAC
    mac_threshold = 0.85

    # import data (and plot)
    acc, Fs = oma.import_data('Data/Accelerations.csv', False, Fs, 300, detrend=True, downsample=True)

    # RRNPS plot
    oma.mps(acc, Fs)
