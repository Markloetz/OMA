import scipy
import numpy as np
from OMA import OMA_Module as oma

if __name__ == "__main__":
    # 3d-Plot mode shapes
    discretization = scipy.io.loadmat('Discretizations/TiflisBruecke.mat')
    N = discretization['N']
    E = discretization['E']
    mode = np.cos(np.pi*np.linspace(0, 1, len(N))) + 1j * 0.5 * np.sin(np.pi*np.linspace(0, 1, len(N)))
    oma.animate_modeshape(N=N,
                          E=E + 1,
                          f_n=5,
                          mode_shape=mode,
                          zeta_n=0.5,
                          directory="Animations/Test/",
                          mode_nr=1,
                          plot=True,
                          mpc = 100)
