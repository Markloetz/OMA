import numpy as np
import matplotlib.pyplot as plt
import nidaqmx
from nidaqmx.constants import (AcquisitionType, AccelSensitivityUnits, AccelUnits)
from scipy.signal import butter, filtfilt
import nidaqmx.stream_writers


def generate_bandlimited_white_noise(duration, sample_rate, cut_low=5, cut_high=500, ramp_duration=2):
    num_samples = int(duration * sample_rate)

    # Generate white noise
    white_noise = np.random.standard_normal(num_samples)
    white_noise = white_noise / max(np.abs(white_noise))

    # Create Lowpass filter
    nyquist = 0.5 * sample_rate
    if cut_high >= nyquist:
        cut_high = nyquist - 1
    b, a = butter(4, [cut_low, cut_high], btype='bandpass', fs=sample_rate, analog=False)

    # Apply filter to white noise
    filtered_noise = filtfilt(b, a, white_noise)

    # Apply ramp up and ramp down
    ramp_up_samples = int(ramp_duration * sample_rate)
    ramp_down_samples = int(ramp_duration * sample_rate)
    ramp_up = np.linspace(0, 1, ramp_up_samples)
    ramp_down = np.linspace(1, 0, ramp_down_samples)

    amplitude_ramp = np.ones(num_samples)
    amplitude_ramp[:ramp_up_samples] = ramp_up
    amplitude_ramp[-ramp_down_samples:] = ramp_down

    return filtered_noise * amplitude_ramp


def sine_gen(duration, sample_rate, freq):
    t = np.linspace(0, duration, duration * sample_rate)
    sine = np.sin(2 * np.pi * freq * t)
    # Apply ramp up and ramp down
    ramp_up_samples = int(2 * sample_rate)
    ramp_down_samples = int(2 * sample_rate)
    ramp_up = np.linspace(0, 1, ramp_up_samples)
    ramp_down = np.linspace(1, 0, ramp_down_samples)

    amplitude_ramp = np.ones(int(duration * sample_rate))
    amplitude_ramp[:ramp_up_samples] = ramp_up
    amplitude_ramp[-ramp_down_samples:] = ramp_down
    return sine * amplitude_ramp


def daq_oma_shaker(device_in, device_out, channels, duration, fs, acc_sensitivities,
                   harmonic_freq):
    with nidaqmx.Task() as acc_task, nidaqmx.Task() as out_task:
        # Configure IEPE task
        for channel in channels:
            acc_task.ai_channels.add_ai_accel_chan(f"{device_in}/ai{channel}",
                                                   sensitivity=1,
                                                   units=AccelUnits.G,
                                                   current_excit_val=0.002,
                                                   sensitivity_units=AccelSensitivityUnits.MILLIVOLTS_PER_G)
        acc_task.timing.cfg_samp_clk_timing(rate=fs,
                                            sample_mode=AcquisitionType.FINITE,
                                            samps_per_chan=(duration + 4) * fs)
        # Configure Shaker task
        out_task.ao_channels.add_ao_voltage_chan(f"{device_out}/ao0")
        out_task.timing.cfg_samp_clk_timing(rate=fs,
                                            sample_mode=AcquisitionType.FINITE,
                                            samps_per_chan=(duration + 4) * fs)
        writer = nidaqmx.stream_writers.AnalogSingleChannelWriter(out_task.out_stream, auto_start=True)

        # Generate white noise vector
        white_noise = generate_bandlimited_white_noise(duration=duration + 4,
                                                       sample_rate=fs,
                                                       cut_low=5,
                                                       cut_high=150)
        sine = sine_gen(duration + 4, sample_rate=fs, freq=harmonic_freq)

        # If user specifies harmonic frequency then a harmonic signal is generated...
        # otherwise the output is white noise
        # Specify additional option like overlays later...
        if harmonic_freq <= 0:
            out_data = white_noise
        else:
            out_data = sine

        # Start Analog Output
        print('DAQ Start')
        writer.write_many_sample(out_data * 5, timeout=duration + 4)

        # record accelerations
        acc_task.start()
        acc_data = acc_task.read(number_of_samples_per_channel=(duration + 4) * fs, timeout=duration + 4)

        # Stop everything
        out_task.stop()
        acc_task.stop()
        print('DAQ Stop')

        # Convert to numpy array
        acc_data = np.array(acc_data)
        acc_data = acc_data.transpose()
        acc_data_out = np.zeros((duration * fs, len(channels)))
        for channel in channels:
            acc_data_out[:, channel] = acc_data[(2 * fs):-(2 * fs), channel] / 1000 / acc_sensitivities[channel]
    return acc_data_out


def daq_oma(device_in, channels, duration, fs, acc_sensitivities):
    with nidaqmx.Task() as acc_task:
        # Configure IEPE task
        for channel in channels:
            acc_task.ai_channels.add_ai_accel_chan(f"{device_in}/ai{channel}",
                                                   sensitivity=1,
                                                   units=AccelUnits.G,
                                                   current_excit_val=0.002,
                                                   sensitivity_units=AccelSensitivityUnits.MILLIVOLTS_PER_G)
        acc_task.timing.cfg_samp_clk_timing(rate=fs,
                                            sample_mode=AcquisitionType.FINITE,
                                            samps_per_chan=(duration + 4) * fs)

        # record accelerations
        acc_task.start()
        print('DAQ Start')
        acc_data = acc_task.read(number_of_samples_per_channel=(duration + 4) * fs, timeout=duration + 4)

        # Stop everything
        acc_task.stop()
        print('DAQ Stop')

        # Convert to numpy array
        acc_data = np.array(acc_data)
        acc_data = acc_data.transpose()
        acc_data_out = np.zeros((duration * fs, len(channels)))
        for channel in channels:
            acc_data_out[:, channel] = acc_data[(2 * fs):-(2 * fs), channel] / 1000 / \
                                       acc_sensitivities[channel]
    return acc_data_out


def plot_data(data, sample_rate):
    time = np.arange(data.shape[0]) / sample_rate
    fig, ax = plt.subplots(data.shape[1], 1, sharex=True)

    for i in range(data.shape[1]):
        ax[i].plot(time, data[:, i])
        ax[i].set_ylabel(f"Channel {i + 1}")

    ax[-1].set_xlabel("Time (s)")
    plt.show()


def save_to_csv(data, filename):
    print('Writing data to '+filename+'...')
    np.savetxt(filename, data, delimiter=',', header='', comments='')
    print('Data stored succesfully!')
