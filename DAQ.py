import numpy as np
import matplotlib.pyplot as plt
import nidaqmx
from nidaqmx.constants import (AcquisitionType, AccelSensitivityUnits, AccelUnits, FuncGenType)
from scipy.signal import butter, filtfilt


def generate_bandlimited_white_noise(duration, sample_rate, cut_freq=500, ramp_duration=2):
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

    # Apply ramp up and ramp down
    ramp_up_samples = int(ramp_duration * sample_rate)
    ramp_down_samples = int(ramp_duration * sample_rate)
    ramp_up = np.linspace(0, 1, ramp_up_samples)
    ramp_down = np.linspace(1, 0, ramp_down_samples)

    amplitude_ramp = np.ones(num_samples)
    amplitude_ramp[:ramp_up_samples] = ramp_up
    amplitude_ramp[-ramp_down_samples:] = ramp_down

    return filtered_noise * amplitude_ramp


def daq_oma(device_in, device_out, device_force, channels, duration, fs, acc_sensitivities, force_sensitivity):
    with nidaqmx.Task() as acc_task, nidaqmx.Task() as out_task, nidaqmx.Task() as force_task:
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
        out_task.timing.cfg_samp_clk_timing(rate=fs)

        # Configure Force measurement
        force_task.ai_channels.add_ai_force_iepe_chan(f"{device_force}/ai0",
                                                      sensitivity=1,
                                                      current_excit_val=0.002)
        force_task.timing.cfg_samp_clk_timing(rate=fs,
                                              sample_mode=AcquisitionType.FINITE,
                                              samps_per_chan=(duration + 4) * fs)

        # Generate white noise vector
        white_noise = generate_bandlimited_white_noise(duration=duration + 4, sample_rate=fs)

        # Start Analog Output
        out_task.write(white_noise * 10, timeout=duration + 4)
        out_task.start()

        # record accelerations
        acc_task.start()
        acc_data = acc_task.read(number_of_samples_per_channel=(duration + 4) * fs, timeout=duration + 4)

        # record accelerations
        force_task.start()
        force_data = force_task.read(number_of_samples_per_channel=(duration + 4) * fs, timeout=duration + 4)

        # Stop everything
        out_task.stop()
        acc_task.stop()
        force_task.stop()

        # Convert to numpy array
        acc_data = np.array(acc_data)
        force_data = np.array(force_data)
        acc_data = acc_data.transpose()
        force_data = force_data.transpose()
        force_data_out = force_data[(2*sample_rate):-(2*sample_rate)] / force_sensitivity
        acc_data_out = np.zeros((duration*sample_rate, len(channels)))
        for channel in channels:
            acc_data_out[:, channel] = acc_data[(2*sample_rate):-(2*sample_rate), channel] / acc_sensitivities[channel]
    return acc_data_out, force_data_out


def plot_data(data, sample_rate):
    time = np.arange(data.shape[0]) / sample_rate
    fig, ax = plt.subplots(data.shape[1], 1, sharex=True)

    for i in range(data.shape[1]):
        ax[i].plot(time, data[:, i])
        ax[i].set_ylabel(f"Channel {i + 1}")

    ax[-1].set_xlabel("Time (s)")
    plt.show()


def save_to_csv(data, filename):
    np.savetxt(filename, data, delimiter=',', header='', comments='')


if __name__ == "__main__":
    device_in = "cDAQ9189-1CDF2BFMod2"
    device_out = "cDAQ9189-1CDF2BFMod1"
    device_force = "cDAQ9189-1CDF2BFMod3"
    channels = [0, 1]
    sensitivities = [1039.0, 1058.2]  # mv/ms^-2
    force_sensitivity = 1
    duration = 30
    sample_rate = 1000
    csv_filename = "test_data.csv"

    acc_data, force_data = daq_oma(device_in=device_in,
                                   device_out=device_out,
                                   device_force=device_force,
                                   channels=channels,
                                   duration=duration,
                                   fs=sample_rate,
                                   acc_sensitivities=sensitivities,
                                   force_sensitivity=force_sensitivity)
    save_to_csv(acc_data, csv_filename)
    plot_data(acc_data, sample_rate)
