import csv
import glob
import numpy as np
import os
import FDD_Module as fdd


def save_to_csv(data, filename):
    np.savetxt(filename, data, delimiter=',', header='', comments='')


if __name__ == '__main__':
    path = 'Data/HammerTest/'
    te = 60             # measurement time [s]
    Fs = 1000           # sampling frequency [Hz]
    n_files = 3         # number of files in directory []
    # Import data
    i = 0
    data = np.zeros((te*Fs, n_files*2))
    for filename in glob.glob(os.path.join(path, '*.csv')):
        with open(os.path.join(os.getcwd(), filename), 'r') as f:
            reader = csv.reader(f)
            data_list = list(reader)
            data[:, (i*2):(i*2+2)] = np.array(data_list, dtype=float)
        i = i+1
    # delete i -> it will for sure be used again
    del i

    # calculate and plot spectral densities of reference signals to scale them accordingly
    modal_max = np.zeros((n_files, 1))
    for i in range(n_files):
        mCPSD, vf = fdd.cpsd_matrix(data=data[:, (i*2):(i*2+2)],
                                    fs=Fs,
                                    zero_padding=False)
        # SVD of CPSD-matrix @ each frequency
        S, U, S2, U2 = fdd.sv_decomp(mCPSD)

        # Peak-picking
        fPeaks, Peaks, nPeaks = fdd.peak_picking(vf, S, S2, Fs, n_sval=1)

        # extract mode shape at each peak
        _, mPHI = U.shape
        PHI = np.zeros((nPeaks, mPHI), dtype=np.complex_)
        for j in range(nPeaks):
            PHI[j, :] = U[np.where(vf == fPeaks[j]), :]
        modal_max[i] = np.abs(PHI[0, 0])

    # scaling and merging data
    data_final = np.zeros((data.shape[0], data.shape[1]//2+1))
    data_final[:, 0] = data[:, 0]
    for i in range(n_files):
        scaling = modal_max[0]/modal_max[i]
        data_final[:, i+1] = data[:, ((i+1)*2)-1] * scaling

    save_to_csv(data_final, 'Data/acc_data_080524_total.csv')



