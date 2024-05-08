import FDD_Module as fdd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    Fs = 1000
    signal, _ = fdd.import_data(filename='Hammer.csv',
                                plot=True,
                                fs=Fs,
                                time=60,
                                detrend=False,
                                downsample=False)

    # Generate some sample data
    data = signal[:, 0]

    signal = data
    # Perform FFT
    fft_output = np.fft.fft(signal)
    spec = fft_output[:(len(fft_output)//2)]
    frequencies = np.fft.fftfreq(len(signal), 1 / Fs)
    f = frequencies[:(len(frequencies) // 2)]

    # Plot the FFT
    plt.figure(figsize=(10, 4))
    plt.plot(f, np.abs(spec))
    plt.title('FFT of Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()
