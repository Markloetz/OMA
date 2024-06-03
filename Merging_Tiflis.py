from OMA import OMA_Module as oma
import numpy as np
import glob
import os
from DAQ_Module import DAQ_Module as daq

if __name__ == '__main__':
    path = "Data/SDL_Floor/"
    Fs = 2048  # [Hz]
    t_meas = 500  # [s]
    n_rov = 2  # number of roving sensors
    n_ref = 1  # The reference signal is allways at the n_rov+1st column
    ref_channel = 2

    # import data and store in one large array
    # preallocate
    n_files = len([name for name in os.listdir(path)])
    data = np.zeros((int(t_meas * Fs), n_files * (n_rov + n_ref)))
    for i, filename in enumerate(glob.glob(os.path.join(path, '*.mat'))):
        data[:, i * (n_rov + n_ref):(i + 1) * (n_rov + n_ref)], _ = oma.import_data(filename=filename,
                                                                                    plot=False,
                                                                                    fs=Fs,
                                                                                    time=t_meas,
                                                                                    detrend=False,
                                                                                    downsample=False,
                                                                                    cutoff=Fs//2)
    
    # Fill the merged data array
    # preallocate
    data_out = np.zeros((data.shape[0], n_rov*n_files))
    j = 0
    for i, col in enumerate(data.T):
        if (i + 1) % (n_rov+n_ref) != 0:
            data_out[:, j] = col
            j = j+1

    daq.save_to_mat(data_out, "Data/SDL_FloorTotal.mat")
