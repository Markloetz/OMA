from DAQ_Module import DAQ_Module as daq

if __name__ == "__main__":
    device_1 = "cDAQ3Mod1"
    device_2 = "cDAQ3Mod2"
    channels_1 = [0, 1, 2, 3]
    channels_2 = [0, 1]
    sensitivities = [98, 102.4, 97.3, 102.9, 103.9, 101.4]  # mv/ms^-2
    # x1, x2, y1, y2, z1, z2 -> 1 Stromabwärts -> z-> auf-ab x-> Gegenstrom
    duration = 1200        # Measurement length
    sample_rate = 2048      # Hz
    mat_filename = "Data/GrenoblerBruecke/Data_090724.mat"
    acc_data = daq.daq_oma_2dev(device1=device_1,
                                device2=device_2,
                                channels_1=channels_1,
                                channels_2=channels_2,
                                duration=duration,
                                fs=sample_rate,
                                acc_sensitivities=sensitivities)
    daq.save_to_mat(acc_data, mat_filename)
    daq.plot_data(acc_data, sample_rate)
