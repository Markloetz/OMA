from OMA import OMA_Module as oma
import numpy as np
import glob
import os
from DAQ_Module import DAQ_Module as daq


def exclude_windows(data, threshold, window_length):
    # Find indices where data exceeds the threshold
    peak_indices = np.where(np.abs(data) > threshold)[0]

    # Initialize an empty list to store the data to be retained
    exclude_mask = np.zeros_like(data, dtype=bool)

    for idx in peak_indices:
        # Determine the range to exclude, ensuring we do not go out of bounds
        start_idx = idx
        end_idx = min(idx + window_length, len(data))
        exclude_mask[start_idx:end_idx] = True

    # Invert the mask to find segments to retain
    include_mask = ~exclude_mask
    retained_data = data[include_mask]

    # Calculate the amount of data needed to fill the gaps
    required_length = len(data) - len(retained_data)

    # If there is data to fill, repeat the retained data to fill the gaps
    if required_length > 0:
        fill_data = np.resize(retained_data, required_length)
        final_data = np.concatenate([retained_data, fill_data])
    else:
        final_data = retained_data

    # Ensure the final data is the same length as the original
    final_data = final_data[:len(data)]

    return final_data


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
        # for j in range(data_temp.shape[1]):
        #     data_temp[:, j] = exclude_windows(data_temp[:, j], data_temp[:, j].std()*3, Fs*2)
        data[:, i * (n_rov + n_ref):(i + 1) * (n_rov + n_ref)] = data_temp

    # Fill the merged data array
    # preallocate
    data_out = np.zeros((data.shape[0], n_rov * n_files))
    j = 0
    for i, col in enumerate(data.T):
        cond = False
        if n_ref == 2:
            cond = (i + 1) % (n_rov + n_ref) != 0 and (i + 1) % (n_rov + n_ref) != 3
        elif n_ref == 1:
            cond = (i + 1) % (n_rov + n_ref) != 0
        if cond:
            # print("out(" + str(j) + ") = data(" + str(i) + ")")
            data_out[:, j] = col
            j = j + 1

    for i, pos in enumerate(ref_pos):
        data_out = np.insert(data_out, pos - 1 + i, data[:, ref_channel[i]], axis=1)

    print(data_out.shape)
    daq.save_to_mat(data_out, "Data/Platte_Total.mat")
