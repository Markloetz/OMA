from DAQ_Module import DAQ_Module as daq

if __name__ == "__main__":
    device_in = "cDAQ9189-1CDF2BFMod2"
    channels = [0, 1, 2]
    sensitivities = [1.008, 1.060, 1.036]  # mv/ms^-2
    duration = 60
    sample_rate = 1000
    mat_filename = "Data/CompleteOMA_Plate/acc_data_130524_38.mat"

    acc_data = daq.daq_oma_hammer(device_in=device_in,
                                  channels=channels,
                                  duration=duration,
                                  fs=sample_rate,
                                  acc_sensitivities=sensitivities)
    daq.save_to_mat(acc_data, mat_filename)
    daq.plot_data(acc_data, sample_rate)
