from DAQ_Module import DAQ_Module as daq

if __name__ == "__main__":
    device_in = "cDAQ1Mod1"
    channels = [0, 1]
    sensitivities = [100.93, 100.73]  # mv/ms^-2
    duration = 5
    sample_rate = 2048
    mat_filename = "Data/testData.mat"

    acc_data = daq.daq_oma(device_in=device_in,
                           channels=channels,
                           duration=duration,
                           fs=sample_rate,
                           acc_sensitivities=sensitivities)
    daq.save_to_mat(acc_data, mat_filename)
    daq.plot_data(acc_data, sample_rate)
