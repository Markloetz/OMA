from DAQ_Module import DAQ_Module as daq

if __name__ == "__main__":
    device_in = "cDAQ9189-1CDF2BFMod2"
    device_out = "cDAQ9189-1CDF2BFMod1"
    channels = [0, 1, 2, 3]
    sensitivities = [1.016, 1.036, 1.060, 1.008]  # mv/ms^-2
    duration = 180
    sample_rate = 2048
    mat_filename = "Data/ShakerOMA/acc_data_01_09_12_33_harmonic_22.5Hz.mat"

    acc_data = daq.daq_oma_shaker(device_in=device_in,
                                  device_out=device_out,
                                  channels=channels,
                                  duration=duration,
                                  fs=sample_rate,
                                  acc_sensitivities=sensitivities,
                                  harmonic_freq=22.5)
    daq.save_to_mat(acc_data, mat_filename)
    daq.plot_data(acc_data, sample_rate)
