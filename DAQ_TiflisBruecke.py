from DAQ_Module import DAQ_Module as daq

if __name__ == "__main__":
    device_in = "cDAQ4Mod1"
    channels = [0, 1, 2, 3]    # Channels 2 and 3 are roving sensors while sensors 1 and 4 is used for reference
    sensitivities = [95.487, 100.93, 100.73, 94.299]  # mv/ms^-2
    duration = 1000        # Measurement length allows for the SDs to have an error of 10 %
    sample_rate = 2048      # Hz
    mat_filename = "Data/TiflisBruecke2/Data_190624_pos_r1_11_12_r2.mat"
    acc_data = daq.daq_oma(device_in=device_in,
                           channels=channels,
                           duration=duration,
                           fs=sample_rate,
                           acc_sensitivities=sensitivities)
    daq.save_to_mat(acc_data, mat_filename)
    daq.plot_data(acc_data, sample_rate)
