import numpy as np
import matplotlib.pyplot as plt
import nidaqmx
import csv


def record_iepe_data(device, channels, duration, sample_rate):
    with nidaqmx.Task() as task:
        for channel in channels:
            task.ai_channels.add_ai_accel_chan(
                f"{device}/ai{channel}", terminal_config=nidaqmx.constants.TerminalConfiguration.DIFFERENTIAL)

        task.timing.cfg_samp_clk_timing(sample_rate, sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)

        num_samples = int(sample_rate * duration)
        data = np.zeros((len(channels), num_samples))

        task.start()

        for i in range(num_samples):
            data[:, i] = task.read(number_of_samples_per_channel=len(channels))

        task.stop()

    return data


def plot_data(data, sample_rate):
    time = np.arange(len(data[0])) / sample_rate
    fig, ax = plt.subplots(len(data), 1, sharex=True)

    for i in range(len(data)):
        ax[i].plot(time, data[i])
        ax[i].set_ylabel(f"Channel {i+1}")

    ax[-1].set_xlabel("Time (s)")
    plt.show()


def save_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Time (s)"] + [f"Channel {i+1}" for i in range(len(data))])
        for i in range(len(data[0])):
            writer.writerow([i/sample_rate] + [data[j, i] for j in range(len(data))])


if __name__ == "__main__":
    device = "Dev1"  # Modify this according to your device name
    channels = [0, 1]  # Modify this according to your channel numbers
    duration = 10  # Duration of data recording in seconds
    sample_rate = 1000  # Sampling rate in Hz
    csv_filename = "accelerometer_data.csv"  # Name of the CSV file to save data

    data = record_iepe_data(device, channels, duration, sample_rate)
    save_to_csv(data, csv_filename)
    plot_data(data, sample_rate)