import numpy as np
import scipy
import FDD_Module as fdd

if __name__ == "__main__":
    mode = np.array([0, 0.5, 0, 0.5, 1, 0.3, 0.1, 0.2, 0, 1, 0, 1, 0.5, 0.7, 0.8, 1, 0.7, 1, 0.9, 1, 0.3, 0.2, 0.3, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0])

    discretization = scipy.io.loadmat('PlateHoleDiscretization.mat')
    N = discretization['N']
    E = discretization['E']
    fdd.plot_modeshape(N, E, mode)

