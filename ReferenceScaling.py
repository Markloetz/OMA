import csv
import glob
import numpy as np
import os
import FDD_Module as fdd


def save_to_csv(data, filename):
    np.savetxt(filename, data, delimiter=',', header='', comments='')


if __name__ == '__main__':
    path = 'Data/ShakerOMA/'
    te = 60.001                                 # measurement time [s]
    Fs = 1000                               # sampling frequency [Hz]
    n_files = 18                            # number of files in directory []
    n_ref = 2                               # number of reference sensors []
    n_rov = 2                               # number of roving sensors []
    ref_pos = np.array([1, 33])             # reference positions []
    # Import data
    i = 0
    scaling = False
    data = np.zeros((int(te*Fs), n_files*(n_rov+n_ref)))
    for filename in glob.glob(os.path.join(path, '*.csv')):
        with open(os.path.join(os.getcwd(), filename), 'r') as f:
            reader = csv.reader(f)
            data_list = list(reader)
            data[:, (i * (n_rov+n_ref)):(i * (n_rov+n_ref) + (n_rov+n_ref))] = np.array(data_list, dtype=float)
        i = i+1
    # delete i -> it will for sure be used again
    del i

    # calculate and plot spectral densities of reference signals to scale them accordingly
    modal_max = np.ones((n_files, 1))
    fPeaks = []
    Peaks = []
    nPeaks = 0
    if scaling:
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
    if n_rov == 1:
        data_final = np.zeros((data.shape[0], data.shape[1]//(n_rov+n_ref)+n_ref))
        data_final[:, (ref_pos-1)] = data[:, :n_ref]
        for i in range(n_files):
            scaling = modal_max[0]/modal_max[i]
            if i not in (ref_pos-1):
                data_final[:, i:i+n_rov] = data[:, (i*(n_ref+n_rov)-n_rov):(i*(n_ref+n_rov))] * scaling
    else:
        data_final = np.zeros((data.shape[0], (data.shape[1]//(n_rov+n_ref)*n_ref)+n_ref))
        for i in range(n_files):
            scaling = modal_max[0]/modal_max[i]
            if i <= (ref_pos[1]-1)//2:
                data_final[:, (i*n_rov-1):(i*n_rov-1)+n_rov] = (data[:, (i*(n_ref+n_rov)-n_rov):(i*(n_ref+n_rov))]
                                                                * scaling)
            else:
                data_final[:, (i * n_rov):(i * n_rov) + n_rov] = data[:, (i * (n_ref + n_rov) - n_rov):(
                            i * (n_ref + n_rov))] * scaling
        data_final[:, ref_pos-1] = data[:, :n_ref] * scaling
        data_final[:, ref_pos[1]] = data[:, (n_ref+n_rov)*int((ref_pos[1]+1)/2)-(n_ref+n_rov+1)] * scaling
        data_final[:, -2:-1] = data[:, -2:-1] * scaling
        data_final[:, -1] = data[:, -1] * scaling
    data_final = np.delete(data_final, -1, 0)
    print(data_final[-1, :])
    print(data_final.shape)
    # Store data
    save_to_csv(data_final, 'Data/acc_data_170524_total.csv')