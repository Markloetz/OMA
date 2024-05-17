import numpy as np
import scipy
import FDD_Module as fdd

if __name__ == "__main__":
    maxima = np.array([0, 1, 2, 3, 4, 5])
    minima = np.array([6, 5, 4, 3, 2, 1, 0])

    print("Before:")
    print(maxima)
    print(minima)

    if len(maxima) > len(minima):
        maxima = maxima[:-1]
    elif len(maxima) < len(minima):
        minima = minima[:-1]

    print("After:")
    print(maxima)
    print(minima)

    minmax = np.array((minima, maxima))
    print("Fitted into one array:")
    print(minmax)
    minmax = np.ravel(minmax, order='F')

    print("Flattened:")
    print(minmax)

    # Alternative:
    del minmax
    minmax = np.zeros((len(minima)*2))
    for i in range(len(minima)-1):
        minmax[i*2] = minima[i]
        minmax[(i+1)*2] = maxima[i]
    print(minmax)