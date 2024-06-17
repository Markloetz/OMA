from OMA import OMA_Module as oma
import numpy as np
import glob
import os
from DAQ_Module import DAQ_Module as daq

if __name__ == '__main__':
    path = "Data/Platte/"
    Fs = 2048  # [Hz]
    t_meas = 500  # [s]
    n_rov = 2  # number of roving sensors
    n_ref = 2  # The reference signal is allways at the n_rov+1st column
    ref_channel = [2, 3]
    ref_pos = [1, 15]

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
            print("out(" + str(j) + ") = data(" + str(i) + ")")
            data_out[:, j] = data[:, i]
            j = j + 1

    # Check if reference sensor(s) need to be merged into the complete dataset
    if np.mean(ref_pos) > 0:
        for i, pos in enumerate(ref_pos):
            print("out(" + str(pos - 1) + ") = data(" + str(ref_channel[i]) + ")")
            data_out = np.insert(data_out, pos - 1, data[:, ref_channel[i]], axis=1)

    daq.save_to_mat(data_out, "Data/PlatteTotal.mat")

