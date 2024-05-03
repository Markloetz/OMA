from scipy.signal import butter, filtfilt, freqz
import numpy as np
import matplotlib.pyplot as plt

def generate_bandlimited_white_noise(duration, sample_rate, cut_freq=400, ramp_duration=2, plot=True):
    num_samples = int(duration * sample_rate)

    # Generate white noise
    white_noise = np.random.standard_normal(num_samples)
    white_noise = white_noise/max(np.abs(white_noise))

    # Create Lowpass filter
    nyquist = 0.5 * sample_rate
    if cut_freq >= nyquist:
        cut_freq = nyquist-1
    b, a = butter(4, cut_freq, btype='low', fs=sample_rate, analog=False)

    # Apply filter to white noise
    filtered_noise = filtfilt(b, a, white_noise)

    # plot frequency response, if asked to

    # Plot the frequency response.
    if plot:
        w, h = freqz(b, a, fs=sample_rate, worN=8000)
        plt.subplot(2, 1, 1)
        plt.plot(w, np.abs(h), 'b')
        plt.plot(cut_freq, 0.5 * np.sqrt(2), 'ko')
        plt.axvline(cut_freq, color='k')
        plt.xlim(0, 0.5 * sample_rate)
        plt.title("Lowpass Filter Frequency Response")
        plt.xlabel('Frequency [Hz]')
        plt.grid()
        plt.show()

    # Apply ramp up and ramp down
    ramp_up_samples = int(ramp_duration * sample_rate)
    ramp_down_samples = int(ramp_duration * sample_rate)
    ramp_up = np.linspace(0, 1, ramp_up_samples)
    ramp_down = np.linspace(1, 0, ramp_down_samples)

    amplitude_ramp = np.ones(num_samples)
    amplitude_ramp[:ramp_up_samples] = ramp_up
    amplitude_ramp[-ramp_down_samples:] = ramp_down

    return filtered_noise * amplitude_ramp


if __name__ == "__main__":
    duration = 30
    Fs = 1000
    data = generate_bandlimited_white_noise(duration=duration, sample_rate=Fs)
    time = np.linspace(0,duration, Fs*duration)
    plt.plot(time, data)
    plt.show()

