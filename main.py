# from matplotlib import pyplot as plt
# import numpy as np
# from scipy.io import loadmat
# from OMA import OMA_Module as oma
# import pickle
#
# if __name__ == '__main__':
#     '''Specify Parameters for OMA'''
#     # Specify Sampling frequency
#     Fs = 2048
#
#     # Path of Measurement Files and other specifications
#     path = "Data/TiflisBruecke2/"
#     n_rov = 2
#     n_ref = 2
#     ref_channel = [0, 3]
#     rov_channel = [1, 2]
#     ref_position = [0, 0]
#
#     # Cutoff frequency (band of interest)
#     cutoff = 25
#
#     # measurement duration
#     t_end = 1000
#
#     # SSI-Parameters
#     # Specify limits
#     f_lim = 0.01        # Pole stability (frequency)
#     z_lim = 0.02        # Pole stability (damping)
#     mac_lim = 0.015      # Mode stability (MAC-Value)
#     limits = [f_lim, z_lim, mac_lim]
#
#     # block-rows
#     ord_max = 50
#     ord_min = 10
#
#     '''Peak Picking Procedure on SV-diagram of the whole dataset'''
#     # import data
#     acc, Fs = oma.merge_data(path=path,
#                              fs=Fs,
#                              n_rov=n_rov,
#                              n_ref=n_ref,
#                              ref_channel=ref_channel,
#                              rov_channel=rov_channel,
#                              ref_pos=ref_position,
#                              t_meas=t_end,
#                              detrend=True,
#                              cutoff=cutoff * 4,
#                              downsample=False)
#
#     # SSI
#     # Perform SSI
#     freqs, zeta, modes, _, _, status = oma.ssi.SSICOV(acc,
#                                                       dt=1 / Fs,
#                                                       Ts=0.8,
#                                                       ord_min=ord_min,
#                                                       ord_max=ord_max,
#                                                       limits=limits)
#
#     # Temporarily Save Results from SSI to improve Debugging speed
#     with open('freqs2.pkl', 'wb') as f:
#         pickle.dump(freqs, f)
#     with open('modes2.pkl', 'wb') as f:
#         pickle.dump(modes, f)
#     with open('zeta2.pkl', 'wb') as f:
#         pickle.dump(zeta, f)
#     with open('status2.pkl', 'wb') as f:
#         pickle.dump(status, f)

    # # Reload mat files with stored lists
    # with open('freqs2.pkl', 'rb') as f:
    #     freqs = pickle.load(f)
    # with open('modes2.pkl', 'rb') as f:
    #     modes = pickle.load(f)
    # with open('zeta2.pkl', 'rb') as f:
    #     zeta = pickle.load(f)
    # with open('status2.pkl', 'rb') as f:
    #     status = pickle.load(f)
    #
    # ranges = [[1.8 - 0.5, 1.8 + 0.5],
    #           [5 - 0.5, 5 + 0.5],
    #           [6.25 - 0.5, 6.25 + 0.5],
    #           [13.25 - 0.5, 13.25 + 0.5],
    #           [18 - 1, 18 + 1]]
    # nPeaks = len(ranges)
    #
    # # stabilization diag
    # fig, ax = oma.ssi.stabilization_diag(freqs, status, cutoff)
    # for j in range(nPeaks):
    #     ax.axvspan(ranges[j][0], ranges[j][1], color='red', alpha=0.3)
    # plt.show()
    #
    # # Extract modal parameters only using results stable in all aspects
    # fS, zetaS, _ = oma.ssi.ssi_extract(freqs, zeta, modes, status, ranges)
    # temp = loadmat('alpha.mat')
    # alpha = temp['alpha']
    #
    # # Print Damping and natural frequencies
    # print("Natural Frequencies [Hz]:")
    # print(fS)
    # print("Damping [%]:")
    # print([x * 100 for x in zetaS])
    #
    # # Extract the mode shape from each dataset separately
    # fS, zetaS, modeS = oma.modal_extract_ssi(path=path,
    #                                          Fs=Fs,
    #                                          n_rov=n_rov,
    #                                          n_ref=n_ref,
    #                                          ref_channel=ref_channel,
    #                                          rov_channel=rov_channel,
    #                                          ref_pos=ref_position,
    #                                          t_meas=t_end,
    #                                          fPeaks=fS,
    #                                          limits=limits,
    #                                          ord_min=ord_min,
    #                                          ord_max=ord_max,
    #                                          d_ord=1,
    #                                          plot=True,
    #                                          cutoff=cutoff,
    #                                          Ts=0.4)
    #
    # # Print Damping and natural frequencies
    # print("Natural Frequencies [Hz]:")
    # print(fS)
    # print("Damping [%]:")
    # print([x * 100 for x in zetaS])
    #
    # # 2d-Plot mode shapes
    # for i in range(nPeaks):
    #     plt.plot(np.real(modeS[i]), label="Mode: " + str(i + 1))
    # plt.legend()
    # plt.show()
    #
    # # 3d-Plot mode shapes
    # discretization = loadmat('Discretizations/TiflisBruecke.mat')
    # N = discretization['N']
    # E = discretization['E']
    #
    # for i in range(nPeaks):
    #     mode = np.zeros(len(modeS[i]) + 4, dtype=np.complex_)
    #     mode[2:-2] = modeS[i]
    #     oma.animate_modeshape(N,
    #                           E + 1,
    #                           mode_shape=mode,
    #                           f_n=fS[i],
    #                           zeta_n=zetaS[i],
    #                           directory="Animations/TiflisSSI/",
    #                           mode_nr=i,
    #                           plot=True)



import numpy as np
import matplotlib.pyplot as plt

# Define example values for idx_low, idx_high, f_low, f_high, s_low, s_high
idx_low = 0
idx_high = 100
f_low = 0
f_high = 10
s_low = 5
s_high = 3

# Initialize the s array
s = np.zeros((idx_high, 1))

# Calculate the number of interpolation points
n_interpolate = idx_high - idx_low

# Generate linear space directly between f_low and f_high
f_space = np.linspace(f_low, f_high, n_interpolate)

# Map linear space between s_low and s_high
seg_int = s_low + (f_space - f_low) * (s_high - s_low) / (f_high - f_low)

# Assign the interpolated segment to the desired range in s
s[idx_low:idx_high, 0] = seg_int

# Plotting the results to verify linear interpolation
plt.plot(s, label='Interpolated Data')
plt.fill_between(np.arange(idx_low, idx_high), s.flatten(), alpha=0.3)
plt.legend()
plt.show()
