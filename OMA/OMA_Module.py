from OMA import FDD_Module as fdd
from OMA import SSI_Module as ssi
import numpy as np
import csv
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm, colors, tri, animation
import glob
import os


def import_data(filename, plot, fs, time, detrend, downsample, cutoff=1000):
    # notify user
    print("Data import started...")
    # load data
    fileending = filename[-3:]
    if fileending == 'csv' or fileending == 'txt':
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
        data = np.array(data, dtype=float)
    elif fileending == 'mat':
        # Load the .mat file
        loaded_data = scipy.io.loadmat(filename)
        # Extract the array from the loaded data
        data = loaded_data['acc']
    else:
        return -1
    # time vector for plot
    if time < (data.shape[0] / fs):
        data = data[:int(fs * time), :]
        t_vec = np.linspace(0, time, int(time * fs))
    else:
        t_vec = np.linspace(0, data.shape[0] / fs, data.shape[0])

    # Some data processing
    n_rows, n_cols = data.shape
    # Detrending
    if detrend:
        for i in range(n_cols):
            data[:, i] = scipy.signal.detrend(data[:, i])

    # # Normalizing data:
    # for i in range(n_cols):
    #     data[:, i] = data[:, i]/data[:, i].std()

    # Downsampling
    fs_new = fs
    if downsample:
        q = 2
        data_new = np.zeros((n_rows // q, n_cols))
        for i in range(n_cols):
            data_new[:, i] = scipy.signal.decimate(data[:, i], q=q)
        fs_new = fs // q
        data = data_new

    # Apply filter to data
    nyquist = np.floor(fs_new / 2) - 1
    if cutoff >= nyquist:
        cutoff = nyquist
    b, a = scipy.signal.butter(4, cutoff, btype='low', fs=fs_new, analog=False)
    for i in range(n_cols):
        data[:, i] = scipy.signal.filtfilt(b, a, data[:, i])
    # notify user
    print("Data import ended...")

    # Plot data
    if plot:
        # Plotting the Data
        for i in range(n_cols):
            plt.plot(t_vec, data[:, i], label="Modal Point"+str(i+1))
        plt.xlabel('Time')
        plt.ylabel('Acceleration')
        plt.title('Raw Data')
        plt.legend()
        plt.grid(True)
        plt.show()

    # return data
    return data, fs_new


# MergedPowerSpectrum
def mps(data, fs):
    # dimensions
    n_rows, n_cols = data.shape

    if n_cols <= 2:
        raise ValueError()
    # MPS calculations
    window = 'hamming'
    n_per_seg = np.floor(n_rows / 8)  # divide into 8 segments
    n_overlap = np.floor(0.5 * n_per_seg)  # Matlab uses zero overlap

    # preallocate cpsd-matrix and frequency vector
    n_fft = int(n_per_seg / 2 + 1)  # limit the amount of fft datapoints to increase speed
    mps_mat = np.zeros((n_fft, n_cols), dtype=np.complex_)
    f = np.zeros((n_fft, 1))
    max_vec = np.zeros((n_cols, 1))

    # The first two mps entries are just the auto spectral densities
    for i in range(2):
        f, mps_mat[:, i] = scipy.signal.csd(data[:, i],
                                            data[:, i],
                                            fs=fs,
                                            nperseg=n_per_seg,
                                            noverlap=n_overlap,
                                            window=window)
    for i in range(2, n_cols):
        _, f_temp_i = scipy.signal.csd(data[:, i],
                                       data[:, i],
                                       fs=fs,
                                       nperseg=n_per_seg,
                                       noverlap=n_overlap,
                                       window=window)
        _, f_temp_i_1 = scipy.signal.csd(data[:, i - 1],
                                         data[:, i - 1],
                                         fs=fs,
                                         nperseg=n_per_seg,
                                         noverlap=n_overlap,
                                         window=window)
        mps_term_1 = np.divide(f_temp_i, f_temp_i_1)
        mps_mat[:, i] = mps_term_1 * mps_mat[:, i - 1]
        max_vec[i] = np.max(mps_mat[:, i].real)
    mps_mat = mps_mat / np.max(max_vec)
    # 3d-plot mps
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot each z-vector as a line at different y positions
    _, n_mps = mps_mat.shape
    for i in range(n_mps):
        ax.plot(f, np.full_like(f, i), mps_mat[:, i].real)

    # Set labels and title
    ax.set_xlabel('f/Hz')
    ax.set_ylabel('Position/m')
    ax.set_zlabel('Spectral Density')
    ax.set_title('RRNPS')

    # Add legend
    # plt.legend()

    plt.show()


def animate_modeshape(N, E, mode_shape, title):
    # Functions
    # create a custom symmetrical colormap
    def symmetrical_colormap(cmap):
        # this defined the roughness of the colormap, 128 fine
        n = 128

        # get the list of color from colormap
        colors_r = cmap(np.linspace(0, 1, n))  # take the standard colormap # 'right-part'
        colors_l = colors_r[::-1]  # take the first list of color and flip the order # "left-part"

        # combine them and build a new colormap
        color = np.vstack((colors_l, colors_r))
        mymap = colors.LinearSegmentedColormap.from_list('symmetric_jet', color)
        return mymap

    # function to set axis of 3d plot equal
    def set_axes_equal(ax):
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    # Pre-Animation Calculations (Initial Data)
    # scale mode shapes according to the size of the object
    x_diff = np.max(N[:, 0]) - np.min(N[:, 0])
    y_diff = np.max(N[:, 1]) - np.min(N[:, 1])
    longest_dim = np.max([x_diff, y_diff])
    mode_shape = mode_shape / np.max(np.abs(mode_shape)) * (longest_dim / 15)
    # Write the mode shape (z-coordinates) into the node vector
    N_temp = np.zeros((N.shape[0], N.shape[1] + 1))
    N_temp[:, 2] = mode_shape
    N_temp[:, :2] = N
    N = N_temp

    # Colormaps, etc...
    # Make the norm
    norm = colors.Normalize(vmin=-np.max(N[:, 2])*1.5, vmax=np.max(N[:, 2])*1.5, clip=False)
    # Create symmetric colormap
    myMap = symmetrical_colormap(cm.jet)
    # create a transparent colormap
    # Define discrete colors with alpha
    color = [
        (1.0, 0.0, 0.0, 0.0),  # Fully transparent red
        (0.0, 1.0, 0.0, 0.0),  # Semi-transparent green
        (0.0, 0.0, 1.0, 0.0)  # Fully opaque blue
    ]
    cm_transparent = colors.ListedColormap(color)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Title
    ax.set_title(title)

    # Initial Plot for each element face with interpolated color based on displacement
    x = []
    y = []
    z = []
    for e, element in enumerate(E):
        # Get the coordinates of the nodes for this element
        nodes = np.zeros((3, 3))
        for i, node_idx in enumerate(element):
            nodes[i, :] = N[node_idx - 1, :]
        # Extract x, y, z coordinates of the nodes
        x.append(nodes[:, 0])
        y.append(nodes[:, 1])
        z.append(nodes[:, 2])

        # refine mesh for interpolated colormapping
        triang = tri.Triangulation(x[e], y[e])
        refiner = tri.UniformTriRefiner(triang)
        interpolator = tri.LinearTriInterpolator(triang, z[e])
        new, new_z = refiner.refine_field(z, interpolator, subdiv=3)

        # Plot the polygon
        ax.plot_trisurf(new.x, new.y, new_z, cmap=myMap, norm=norm, alpha=1, linewidth=0)
        ax.plot_trisurf(x[e], y[e], z[e], triangles=[[0, 1, 2]], cmap=cm_transparent, linewidth=1, edgecolor='black')

    # Set plot limits
    ax.set_xlim(np.min(N[:, 0]), np.max(N[:, 0]))
    ax.set_ylim(np.min(N[:, 1]), np.max(N[:, 1]))
    ax.set_zlim(np.min(N[:, 2]), np.max(N[:, 2]))

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set axis equal
    set_axes_equal(ax)

    # Update frame of animation
    def update(frame):
        ax.clear()
        for i in range(len(E)):
            _z = z[i]*np.cos(np.pi/5*frame)
            # refine mesh for interpolated colormapping
            triang = tri.Triangulation(x[i], y[i])
            refiner = tri.UniformTriRefiner(triang)
            interpolator = tri.LinearTriInterpolator(triang, _z)
            new, new_z = refiner.refine_field(z, interpolator, subdiv=3)

            # Plot the polygon
            ax.plot_trisurf(new.x, new.y, new_z, cmap=myMap, norm=norm, alpha=1, linewidth=0)
            ax.plot_trisurf(x[i], y[i], _z, triangles=[[0, 1, 2]], cmap=cm_transparent, linewidth=1,
                            edgecolor='black')
        # Set plot limits
        ax.set_xlim(np.min(N[:, 0]), np.max(N[:, 0]))
        ax.set_ylim(np.min(N[:, 1]), np.max(N[:, 1]))
        ax.set_zlim(np.min(N[:, 2]), np.max(N[:, 2]))

        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Title
        ax.set_title(title)

        # Set axis equal
        set_axes_equal(ax)

    # animation
    _ = animation.FuncAnimation(fig=fig, func=update, interval=50,  frames=20)

    plt.show()


def modescale(path, Fs, n_rov, n_ref, ref_channel, t_meas, fPeaks, window, overlap, n_seg, zeropadding):
    # import data and store in one large array
    # preallocate
    n_files = len([name for name in os.listdir(path)])
    data = np.zeros((int(t_meas * Fs), n_files * (n_rov + n_ref)))
    for i, filename in enumerate(glob.glob(os.path.join(path, '*.mat'))):
        data[:, i * (n_rov + n_ref):(i + 1) * (n_rov + n_ref)], _ = import_data(filename=filename,
                                                                                plot=False,
                                                                                fs=Fs,
                                                                                time=t_meas,
                                                                                detrend=False,
                                                                                downsample=False,
                                                                                cutoff=Fs // 2)
        # data_ref = data[:, i * (n_rov + n_ref):(i + 1) * (n_rov + n_ref)]
        # data_ref = data_ref[:, ref_channel]
        # std_ref = data_ref.std()
        # data[:, i * (n_rov + n_ref):(i + 1) * (n_rov + n_ref)]/std_ref
    # calculate and plot spectral densities of reference signals to scale them accordingly
    nPeaks = len(fPeaks)
    ref_modes = np.zeros((n_files, nPeaks), dtype=np.complex_)
    alpha = np.zeros((n_files * n_rov, nPeaks), dtype=np.complex_)
    # loop over each dataset
    for i in range(n_files):
        mCPSD, vf = fdd.cpsd_matrix(data=data[:, i * (n_rov + n_ref):(i + 1) * (n_rov + n_ref)],
                                    fs=Fs,
                                    zero_padding=zeropadding,
                                    n_seg=n_seg,
                                    window=window,
                                    overlap=overlap)
        # SVD of CPSD-matrix @ each frequency
        S, U, S2, U2 = fdd.sv_decomp(mCPSD)

        # extract mode shapes at selected peaks for each dataset
        _, mPHI = U.shape
        PHI = np.zeros((nPeaks, mPHI), dtype=np.complex_)
        for j in range(nPeaks):
            PHI[j, :] = U[np.where(vf == fPeaks[j]), :]

        # reference mode for each natural frequency
        ref_modes[i, :] = PHI[:, ref_channel]

    # calculate scaling factor alpha for each dataset and each peak
    for i in range(n_files):
        for j in range(nPeaks):
            # modal amplitude of dataset i and frequency j
            amp = ref_modes[i, j]
            # reference amplitude from dataset 0 and frequency j
            amp_ref = ref_modes[0, j]
            # scaling factor
            alpha[i * n_rov:i * n_rov + n_rov, j] = amp / amp_ref
    return np.real(alpha.T)
