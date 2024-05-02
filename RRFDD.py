# Rowing reference processing
import FDD_Module as fdd

if __name__ == '__main__':
    # Specify Sampling frequency
    Fs = 100

    # Threshold for MAC
    mac_threshold = 0.85

    # import data (and plot)
    acc = fdd.import_data('Accelerations.csv', False, Fs, 300, detrend=True, downsample=True)

    # RRNPS plot
    fdd.mps(acc, Fs)
