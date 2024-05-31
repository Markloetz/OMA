import csv
import glob
import numpy as np
import os
from OMA import OMA_Module as oma


def save_to_csv(data, filename):
    np.savetxt(filename, data, delimiter=',', header='', comments='')


if __name__ == '__main__':
    path = 'Data/MCI_Measurement_Room493/EinfeldTest/'
    te = 300  # measurement time [s]
    Fs = 2048  # sampling frequency [Hz]
    n_files = 2  # number of files in directory []
    n_ref = 1  # number of reference sensors []
    n_rov = 1  # number of roving sensors []
    ref_pos = 1  # reference positions []
    # Import data
    i = 0
    scaling = True
    data = np.zeros((int(te * Fs), n_files * (n_rov + n_ref)))
    for filename in glob.glob(os.path.join(path, '*.csv')):
        with open(os.path.join(os.getcwd(), filename), 'r') as f:
            reader = csv.reader(f)
            data_list = list(reader)
            data[:, (i * (n_rov + n_ref)):(i * (n_rov + n_ref) + (n_rov + n_ref))] = np.array(data_list, dtype=float)
        i = i + 1
    # delete i -> it will for sure be used again
    del i

    # calculate and plot spectral densities of reference signals to scale them accordingly
    modal_max = np.ones((n_files, 1))
    fPeaks = []
    Peaks = []
    nPeaks = 0
    if scaling:
        for i in range(n_files):
            mCPSD, vf = oma.fdd.cpsd_matrix(data=data[:, (i * (n_rov + n_ref)):(i * (n_rov + n_ref) + (n_rov + n_ref))],
                                            fs=Fs,
                                            zero_padding=True)

            # SVD of CPSD-matrix @ each frequency
            S, U, S2, U2 = oma.fdd.sv_decomp(mCPSD)

            # Peak-picking only one time:
            if i == 0:
                freqs, vals, nPeaks = oma.fdd.peak_picking(vf, S, S2, n_sval=1)
                fPeaks.append(freqs)
                Peaks.append(vals)

            # extract reference mode shape at each peak
            _, mPHI = U.shape
            PHI = np.zeros((nPeaks, mPHI), dtype=np.complex_)
            for j in range(nPeaks):
                PHI[j, :] = U[np.where(vf == fPeaks[0][j]), :]
            # mean of the mode shape amplitudes is used for scaling (change when results are shit)
            modal_max[i] = np.mean(np.abs(PHI[:, :n_ref]))

    # scaling and merging data
    data_final = np.zeros((data.shape[0], n_rov*n_files+n_ref))
    print(data_final.shape)
    data_final[:, ref_pos-1] = data[:, 0]
    for i in range(n_files):
        data_final[:, i*n_rov+n_ref] = data[:, (i * (n_rov + n_ref) + n_rov)] * modal_max[0]/modal_max[i]
    # Store data
    save_to_csv(data_final, 'Data/acc_data_300524_total.csv')
