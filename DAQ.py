import numpy as np
import matplotlib.pyplot as plt
import nidaqmx
from nidaqmx.constants import (AcquisitionType, AccelSensitivityUnits, AccelUnits)
import csv


def record_iepe_data(device, channels, duration, fs, sensitivities):
    with nidaqmx.Task() as task:
        for channel in channels:
            task.ai_channels.add_ai_accel_chan(f"{device}/ai{channel}",
                                               sensitivity=1,
                                               units=AccelUnits.G,
                                               current_excit_val=0.002,
                                               sensitivity_units=AccelSensitivityUnits.MILLIVOLTS_PER_G)
        task.timing.cfg_samp_clk_timing(rate=fs,
                                        sample_mode=AcquisitionType.FINITE,
                                        samps_per_chan=duration * fs)
        # record data
        task.start()
        data = task.read(number_of_samples_per_channel=duration * fs, timeout=duration)
        task.stop()

        # Convert to numpy array
        data = np.array(data)
        data = data.transpose()
        for channel in channels:
            data[:, channel] = data[:, channel]/sensitivities[channel]
    return data


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
    device = "cDAQ1Mod1"
    channels = [0, 1]
    sensitivities = [1039.0, 1058.2] # mv/ms^-2
    duration = 300
    sample_rate = 1000
    csv_filename = "test_data.csv"

    data = record_iepe_data(device=device,
                            channels=channels,
                            duration=duration,
                            fs=sample_rate,
                            sensitivities=sensitivities)
    save_to_csv(data, csv_filename)
    plot_data(data, sample_rate)
