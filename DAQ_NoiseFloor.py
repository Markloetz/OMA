import DAQ_Module as daq

if __name__ == "__main__":
    device_in = "cDAQ2Mod1"
    channels = [0, 1]
    sensitivities = [100.93, 100.73]  # mv/ms^-2
    duration = 180
    sample_rate = 2048
    csv_filename = "Data/NoiseFloorEstimation/acc_data_240524.csv"

    acc_data = daq.daq_oma(device_in=device_in,
                           channels=channels,
                           duration=duration,
                           fs=sample_rate,
                           acc_sensitivities=sensitivities)
    daq.save_to_csv(acc_data, csv_filename)
    daq.plot_data(acc_data, sample_rate)
