import numpy as np
import scipy
import matplotlib.pyplot as plt

if __name__ == '__main__':
    filename = "Discretizations/TiflisBruecke.mat"

    # main Parameters for mesh generation
    length = 42.8
    width = 5.75
    n_rows = 8
    n_cols = 2
    polygon_shape = 4

    # create node coordinates
    nodes = np.zeros((n_rows * n_cols, n_cols))
    dist_l = length / (n_rows - 1)
    dist_w = width / (n_cols - 1)
    k = 0
    for i in range(n_rows):
        for j in range(n_cols):
            nodes[k, :] = np.array([dist_l * i, dist_w * j])
            k = k + 1
    # Center the origin
    nodes[:, 0] = nodes[:, 0] - length / 2
    nodes[:, 1] = nodes[:, 1] - width / 2

    # create triangular mesh
    # Remove duplicate points
    nodes = np.unique(nodes, axis=0)

    # Perform Delaunay triangulation on the normalized points
    tri = scipy.spatial.Delaunay(nodes)
    elements = tri.simplices
    elements = np.sort(elements, axis=1)
    idx = np.argsort(np.sum(elements, axis=1), axis=0)
    elements_sort = elements
    for i in range(3):
        elements_sort[:, i] = np.take_along_axis(elements[:, i], idx, axis=0)
    elements = elements_sort
    
    # Optional: Visualize the mesh
    plt.triplot(nodes[:, 0], nodes[:, 1], elements)
    for i in range(nodes.shape[0]):
        plt.plot(nodes[i, 0], nodes[i, 1], marker="$"+str(i+1)+"$", markersize=10, color="red")
    plt.show()

    print(nodes)

    # store as .mat file in discretizations directory
    scipy.io.savemat(filename, {'N': nodes, 'E': elements})
