from OMA import OMA_Module as oma
import numpy as np
import glob
import os
from DAQ_Module import DAQ_Module as daq


if __name__ == '__main__':
    path = "Data/TiflisBruecke/"
    Fs = 2048  # [Hz]
    t_meas = 500  # [s]
    n_rov = 2  # number of roving sensors
    n_ref = 1  # The reference signal is allways at the n_rov+1st column
    ref_channel = 0
    ref_pos = [0, 0]

    # import data and store in one large array
    # preallocate
    n_files = len([name for name in os.listdir(path)])
    data = np.zeros((int(t_meas * Fs), n_files * (n_rov + n_ref)))
    for i, filename in enumerate(glob.glob(os.path.join(path, '*.mat'))):
        data_temp, _ = oma.import_data(filename=filename,
                                       plot=False,
                                       fs=Fs,
                                       time=t_meas,
                                       detrend=True,
                                       downsample=False,
                                       cutoff=Fs // 2)
        # for j in range(data_temp.shape[1]):
        #     data_temp[:, j] = exclude_windows(data_temp[:, j], data_temp[:, j].std()*3, Fs*2)
        data[:, i * (n_rov + n_ref):(i + 1) * (n_rov + n_ref)] = data_temp

    # Fill the merged data array
    # preallocate
    data_out = np.zeros((data.shape[0], n_rov * n_files))
    j = 0
    for i, _ in enumerate(data.T):
        if n_ref == 2:
            cond = (i + 1) % (n_rov + n_ref) != 0 and (i + 1) % (n_rov + n_ref) != 3
        else:
            cond = (i + 1) % (n_rov + n_ref) != 0
        if cond:
            print("out(" + str(j) + ") = data(" + str(i+1) + ")")
            data_out[:, j] = data[:, i+1]
            j = j + 1

    # Check if reference sensor(s) need to be merged into the complete dataset
    if np.mean(ref_pos) > 0:
        for i, pos in enumerate(ref_pos):
            data_out = np.insert(data_out, pos - 1 + i, data[:, ref_channel[i]], axis=1)

    print(data_out.shape)
    daq.save_to_mat(data_out, "Data/TiflisTotal.mat")
