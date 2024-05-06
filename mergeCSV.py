import numpy as np
import FDD_Module as fdd


def save_to_csv(data, filename):
    np.savetxt(filename, data, delimiter=',', header='', comments='')

if __name__ == "__main__":
    Fs = 1000
    acc1, _ = fdd.import_data(filename='acc_data_050524_12.csv',
                              plot=False,
                              fs=Fs,
                              time=60,
                              detrend=False,
                              downsample=False)
    acc2, _ = fdd.import_data(filename='acc_data_050524_34.csv',
                              plot=False,
                              fs=Fs,
                              time=60,
                              detrend=False,
                              downsample=False)
    acc3, _ = fdd.import_data(filename='acc_data_050524_56.csv',
                              plot=False,
                              fs=Fs,
                              time=60,
                              detrend=False,
                              downsample=False)
    acc4, _ = fdd.import_data(filename='acc_data_050524_78.csv',
                              plot=False,
                              fs=Fs,
                              time=60,
                              detrend=False,
                              downsample=False)
    acc5, _ = fdd.import_data(filename='acc_data_050524_910.csv',
                              plot=False,
                              fs=Fs,
                              time=60,
                              detrend=False,
                              downsample=False)
    acc6, _ = fdd.import_data(filename='acc_data_050524_1112.csv',
                              plot=False,
                              fs=Fs,
                              time=60,
                              detrend=False,
                              downsample=False)
    acc7, _ = fdd.import_data(filename='acc_data_050524_1314.csv',
                              plot=False,
                              fs=Fs,
                              time=60,
                              detrend=False,
                              downsample=False)
    acc8, _ = fdd.import_data(filename='acc_data_050524_1516.csv',
                              plot=False,
                              fs=Fs,
                              time=60,
                              detrend=False,
                              downsample=False)
    acc9, _ = fdd.import_data(filename='acc_data_050524_1718.csv',
                              plot=False,
                              fs=Fs,
                              time=60,
                              detrend=False,
                              downsample=False)

    acc_tot = np.zeros((acc1.shape[0], 18))

    acc_tot[:, :2] = acc1
    acc_tot[:, 2:4] = acc2
    acc_tot[:, 4:6] = acc3
    acc_tot[:, 6:8] = acc4
    acc_tot[:, 8:10] = acc5
    acc_tot[:, 10:12] = acc6
    acc_tot[:, 12:14] = acc7
    acc_tot[:, 14:16] = acc8
    acc_tot[:, 16:18] = acc9

    save_to_csv(acc_tot, "acc_total_060524.csv")

