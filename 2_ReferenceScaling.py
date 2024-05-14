import csv
import glob
import numpy as np
import os
import FDD_Module as fdd


def save_to_csv(data, filename):
    np.savetxt(filename, data, delimiter=',', header='', comments='')


if __name__ == '__main__':
    path = 'Data/CompleteOMA_Plate/'
    te = 60                                 # measurement time [s]
    Fs = 1000                               # sampling frequency [Hz]
    n_files = 36                            # number of files in directory []
    n_ref = 2                               # number of reference sensors []
    n_rov = 1                               # number of roving sensors []
    ref_pos = np.array([1, 33])             # reference positions []
    # Import data
    i = 0
    data = np.zeros((te*Fs, n_files*(n_rov+n_ref)))
    for filename in glob.glob(os.path.join(path, '*.csv')):
        with open(os.path.join(os.getcwd(), filename), 'r') as f:
            reader = csv.reader(f)
            data_list = list(reader)
            data[:, (i * (n_rov+n_ref)):(i * (n_rov+n_ref) + (n_rov+n_ref))] = np.array(data_list, dtype=float)
        i = i+1
    # delete i -> it will for sure be used again
    del i

    # calculate and plot spectral densities of reference signals to scale them accordingly
    modal_max = np.zeros((n_files, 1))
    fPeaks = []
    Peaks = []
    nPeaks = 0
    for i in range(n_files):
        mCPSD, vf = fdd.cpsd_matrix(data=data[:, (i * (n_rov+n_ref)):(i * (n_rov+n_ref) + (n_rov+n_ref))],
                                    fs=Fs,
                                    zero_padding=True)
        # SVD of CPSD-matrix @ each frequency
        S, U, S2, U2 = fdd.sv_decomp(mCPSD)

        # Peak-picking only one time:
        if i == 0:
            freqs, vals, nPeaks = fdd.peak_picking(vf, S, S2, n_sval=1)
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
    data_final = np.zeros((data.shape[0], data.shape[1]//(n_rov+n_ref)+n_ref))
    data_final[:, (ref_pos-1)] = data[:, :n_ref]
    for i in range(n_files):
        scaling = modal_max[0]/modal_max[i]
        if i not in (ref_pos-1):
            data_final[:, i:i+n_rov] = data[:, (i*(n_ref+n_rov)-n_rov):(i*(n_ref+n_rov))] * scaling

    # Store data
    save_to_csv(data_final, 'Data/acc_data_140524_total.csv')