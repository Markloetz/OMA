from OMA import OMA_Module as oma
import numpy as np
import scipy

if __name__ == "__main__":
    # 3d-Plot mode shapes
    discretization = scipy.io.loadmat('Discretizations/TiflisBruecke.mat')
    N = discretization['N']
    E = discretization['E']
    mode = np.sin(np.pi*np.linspace(0, 1, len(N)))
    oma.animate_modeshape(N,
                          E + 1,
                          mode_shape=mode,
                          f_n=5,
                          zeta_n=0.5,
                          directory="Animations/Tiflis_1/",
                          mode_nr=1,
                          plot=True)
