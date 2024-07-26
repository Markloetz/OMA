import numpy as np
from OMA import OMA_Module as oma
from scipy.io import loadmat

if __name__ == "__main__":

    # Load custom mesh node data
    ema_results = loadmat('ComparisonData/EMA_modes.mat')
    ema_modes = ema_results['mode'].T
    ema_freqs = [1.87, 6.07, 6.89, 13.34, 16.93]
    nPeaks = len(ema_freqs)


    # Check  mode shapes
    discretization = loadmat('Discretizations/TiflisBruecke.mat')
    N = discretization['N']
    E = discretization['E']

    # for i in range(nPeaks):
    #     mode = np.zeros(ema_modes.shape[1] + 4, dtype=np.complex_)
    #     mode[2:-2] = ema_modes[i, :]
    #     oma.animate_modeshape(N,
    #                           E + 1,
    #                           mode_shape=mode,
    #                           f_n=ema_freqs[i],
    #                           zeta_n=0,
    #                           mpc=0,
    #                           directory="Animations/Tiflis_EMA/",
    #                           mode_nr=i,
    #                           plot=False)


