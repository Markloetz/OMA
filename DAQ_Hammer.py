import numpy as np
import matplotlib.pyplot as plt
import nidaqmx
from nidaqmx.constants import (AcquisitionType, AccelSensitivityUnits, AccelUnits)
import nidaqmx.stream_writers


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
        acc_data_out = np.zeros((duration * sample_rate, len(channels)))
        for channel in channels:
            acc_data_out[:, channel] = acc_data[(2 * sample_rate):-(2 * sample_rate), channel] / 1000 / \
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
    print('Saving Data...')
    np.savetxt(filename, data, delimiter=',', header='', comments='')
    print('Data Saved')


if __name__ == "__main__":
    device_in = "cDAQ9189-1CDF2BFMod2"
    channels = [0, 1, 2]
    sensitivities = [1.008, 1.060, 1.036]  # mv/ms^-2
    duration = 60
    sample_rate = 1000
    csv_filename = "Data/CompleteOMA_Plate/acc_data_130524_38.csv"

    acc_data = daq_oma(device_in=device_in,
                       channels=channels,
                       duration=duration,
                       fs=sample_rate,
                       acc_sensitivities=sensitivities)
    save_to_csv(acc_data, csv_filename)
    plot_data(acc_data, sample_rate)

    # Notes for initial test
    # Meas 1: 33-12 // Meas 2: 33-35 // Meas 3: 33-16
    # Rearrange to 35-16-12 for the mode shapes to make sense

    # Notes for complete OMA
    # Referrence Sensor Positions are: 1, 33
