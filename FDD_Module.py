import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import cm, colors, tri
import scipy


# Global Variables
class SliderValClass:
    slider_val = 0


# Functions
def import_data(filename, plot, fs, time, detrend, downsample, gausscheck, cutoff=1000):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    data = np.array(data, dtype=float)
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
    b, a = scipy.signal.butter(4, cutoff, btype='low', fs=fs, analog=False)
    for i in range(n_cols):
        data[:, i] = scipy.signal.filtfilt(b, a, data[:, i])

    # Plot data
    if plot:
        # Plotting the Data
        for i in range(n_cols):
            plt.plot(t_vec, data[:, i])
        plt.xlabel('Time')
        plt.ylabel('Acceleration')
        plt.title('Raw Data')
        plt.grid(True)
        plt.show()

    if gausscheck:
        # split measurement into 10 segments
        n_seg = 10
        alpha = 0.05
        for i in range(n_cols):
            for j in range(n_seg):
                data_temp = data[j * n_rows // n_seg:(j * n_rows // n_seg + n_rows // n_seg), i]
                # check for normality of the signal with shapiro-wilk-test
                _, p_val = scipy.stats.shapiro(data_temp)
                if p_val > alpha:
                    print(p_val)

    # return data
    return data, fs_new


def harmonic_est(data, delta_f, f_max, fs, plot=True):
    # get dimensions
    n_rows, n_cols = data.shape
    # Normalize data (zero mean, unit variance)
    data_norm = (data - data.mean(axis=0)) / data.std(axis=0)
    # Run bandpass filter over each frequency and check for kurtosis
    n_filt = int(f_max // delta_f)
    # filter parameters for each frequency
    kurtosis = np.zeros((n_cols, 1))
    kurtosis_mean = np.zeros((n_filt, 1))
    for j in range(5, n_filt):
        b = np.zeros((n_filt, 5))
        a = np.zeros((n_filt, 5))
        b[j, :], a[j, :] = scipy.signal.butter(2, [j * delta_f - delta_f / 2, j * delta_f + delta_f / 2],
                                               btype='bandpass', fs=fs, analog=False)
        for i in range(n_cols):
            data_filt = scipy.signal.filtfilt(b[j, :], a[j, :], data_norm[:, i])
            # only use second half of data due to filter behaviour
            data_filt = data_filt[len(data_filt) // 2:-1]
            # plt.plot(data_filt)
            # plt.show()
            # calculate kurtosis
            kurtosis[i] = scipy.stats.kurtosis(data_filt, fisher=True)
        kurtosis_mean[j] = np.mean(kurtosis)

    # find platykurtic frequencies and sort them
    f_bad = []
    kurtosis_mag = []
    for i in range(1, n_filt):
        if kurtosis_mean[i] < 0:
            f_bad.append((i + 1) * delta_f)
            kurtosis_mag.append(kurtosis_mean[i][0])
    f_out = [x for _, x in sorted(zip(kurtosis_mag, f_bad))]
    # plot relevant kurtosis mean and frequency range
    if plot:
        plt.plot(kurtosis_mean)
        plt.show()

    return f_out


def cpsd_matrix(data, fs, zero_padding=True):
    # get dimensions
    n_rows, n_cols = data.shape

    # Zero padding
    if zero_padding:
        n_padding = 2
        buffer = np.zeros((n_rows * n_padding, n_cols))
        buffer[:n_rows, :] = data
        data = buffer
        n_rows = n_rows * n_padding

    # CSPD-Parameters (PyOMA) -> Use a mix between matlab default and pyoma
    # df = fs / n_rows * 2
    # n_per_seg = int(fs / df)
    # n_overlap = np.floor(n_per_seg*0.5)
    # window = 'hann'
    # CSPD-Parameters (Matlab-Style) -> very good for fitting
    window = 'hamming'
    n_seg = 18
    n_per_seg = np.floor(n_rows / n_seg)  # divide into 8 segments
    n_overlap = np.floor(0 * n_per_seg)  # Matlab uses zero overlap

    # preallocate cpsd-matrix and frequency vector
    n_fft = int(n_per_seg / 2 + 1)  # limit the amount of fft datapoints to increase speed
    cpsd = np.zeros((n_cols, n_cols, n_fft), dtype=np.complex_)
    f = np.zeros((n_fft, 1))

    # Build cpsd-matrix
    for i in range(n_cols):
        for j in range(n_cols):
            f, sd = scipy.signal.csd(data[:, i],
                                     data[:, j],
                                     fs=fs,
                                     nperseg=n_per_seg,
                                     noverlap=n_overlap,
                                     window=window)
            cpsd[i, j, :] = sd

    # return cpsd-matrix
    return cpsd, f


def sv_decomp(mat):
    # get dimensions
    n_cols, _, n_rows = mat.shape

    # preallocate singular values and mode shapes
    s1 = np.zeros((n_rows, 1))
    u1 = np.zeros((n_rows, n_cols), dtype=complex)
    s2 = np.zeros((n_rows, 1))
    u2 = np.zeros((n_rows, n_cols), dtype=complex)

    # SVD
    for i in range(n_rows):
        u, s, _ = np.linalg.svd(mat[:, :, i])
        u1[i, :] = u[:, 0].transpose()
        s1[i, :] = s[0]
        u2[i, :] = u[:, 1].transpose()
        s2[i, :] = s[1]

    # return function outputs
    return s1, u1, s2, u2


def prominence_adjust(x, y):
    # Adjusting peak-prominence with slider
    min_prominence = abs(max(y) / 100)
    max_prominence = abs(max(y))
    # Create the plot
    figure, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)  # Adjust bottom to make space for the slider

    # Plot the initial data
    locs, _ = scipy.signal.find_peaks(y, prominence=(min_prominence, None), distance=np.ceil(len(x) / 1000))
    y_data = y[locs]
    x_data = x[locs]
    line, = ax.plot(x_data, y_data, 'bo')

    # Add a slider
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Peak Prominence', min_prominence, max_prominence, valinit=min_prominence)

    # Update Plot
    def update(val):
        SliderValClass.slider_val = val
        locs_, _ = scipy.signal.find_peaks(y, prominence=(SliderValClass.slider_val, None),
                                           distance=np.ceil(len(x) / 1000))
        y_data_current = y[locs_]
        x_data_current = x[locs_]
        line.set_xdata(x_data_current)
        line.set_ydata(y_data_current)
        figure.canvas.draw_idle()

    slider.on_changed(update)
    ax.plot(x, y)
    ax.set_xlim([0, 100])
    plt.show()

    return SliderValClass.slider_val


def peak_picking(x, y, y2, n_sval=1):
    y = y.ravel()
    y2 = y2.ravel()
    x = x.ravel()

    # get prominence
    locs, _ = scipy.signal.find_peaks(y, prominence=(prominence_adjust(x, y), None), distance=np.ceil(len(x) / 1000))
    y_data = y[locs]
    x_data = x[locs]
    # Peak Picking
    # Create a figure and axis
    figure, ax = plt.subplots()
    ax.set_xlim([0, 100])

    # Store the selected points
    selected_points = {'x': [], 'y': []}

    # Function to calculate distance between two points
    def distance(x1, y1, x2, y2):
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Function to handle click events
    def onclick(event):
        if event.button == 1:  # Left mouse button
            x, y = event.xdata, event.ydata

            # Find the nearest blue point
            distances = [distance(x, y, xi, yi) for xi, yi in zip(x_data, y_data)]
            nearest_index = np.argmin(distances)
            nearest_x, nearest_y = x_data[nearest_index], y_data[nearest_index]

            # Store the selected point
            selected_points['x'].append(nearest_x)
            selected_points['y'].append(nearest_y)
            ax.plot(nearest_x, nearest_y, 'ro')  # Plot the selected point in red
            plt.draw()

    # Connect the onclick function to the figure
    _ = figure.canvas.mpl_connect('button_press_event', onclick)

    # Plot the blue data points
    ax.plot(x, y)
    if max(y2) <= 0.0001 * max(y):
        scaling = 1
    else:
        scaling = max(y) / max(y2) / 2
    if n_sval > 1:
        ax.plot(x, (y2 * scaling), linewidth=0.7, color='black')
    ax.plot(x_data, y_data, 'bo')  # Plot the data points in blue
    # vertical lines at suspected harmonic frequencies
    # for i in range(x_vert.shape[0]):
    #     ax.axvline(x_vert[i], color=[0, 0, 0])
    ax.set_title('Click to select points')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Show the plot
    plt.show()

    # remove multiple entries at same spot
    for i in range(1, len(selected_points['x'])):
        if selected_points['x'][i] == selected_points['x'][i - 1]:
            del selected_points['x'][i]
            del selected_points['y'][i]

    # Store number of selected points
    n_points = len(selected_points['x'])
    y_out = np.array(selected_points['y'])
    return selected_points['x'], 10 ** (y_out / 20), n_points


def mac_calc(phi, u):
    # calculates mac value between phi and u
    return (np.abs(phi.conj().T @ u) ** 2) / ((phi.conj().T @ phi) * (u.conj().T @ u))


def find_widest_range(array, center_indices):
    array = array.flatten()
    groups = []
    group_indices = []

    current_group = []
    current_group_index = []

    for i, val in enumerate(array):
        if val != 0:
            current_group.append(val)
            current_group_index.append(i)
        elif current_group:
            groups.append(current_group)
            group_indices.append(current_group_index)
            current_group = []
            current_group_index = []

    # If the last group extends to the end of the array
    if current_group:
        groups.append(current_group)
        group_indices.append(current_group_index)

    # find length of the groups and determine if group is positioned round the specified point
    lengths = np.zeros((len(group_indices)))
    for i in range(len(group_indices)):
        if set(center_indices).issubset(set(group_indices[i])):
            lengths[i] = len(group_indices[i])

    # extract indices with maximum length
    max_length_ind = np.argmax(lengths)
    out = group_indices[max_length_ind]
    return out


def sdof_frf(f, omega_n, zeta):
    omega = 2 * np.pi * f
    # h = (1 / (omega_n ** 2)) * ((omega_n**2)/(omega_n**2 - omega**2 + 1j*(2*zeta*omega*omega_n)))
    h = 1 / (-omega ** 2 + 2j * zeta * omega * omega_n + omega_n ** 2)
    return np.abs(h) ** 2


def sdof_half_power(f, y, fn):
    # removing nan
    f = f[~np.isnan(f)]
    y = y[~np.isnan(y)]

    # Applying the half power method to estimeate the damping coefficients
    # extract peak value and the half power value
    y_wn = y[np.where(f == fn)]
    threshold = 0.707 * y_wn
    # find the range north of the threshold
    ind_range = np.where(y >= threshold)[0]
    ind_high = ind_range[-1]
    ind_low = ind_range[0]
    delta_f = f[ind_high] - f[ind_low]
    zeta_est = delta_f / 2 / fn
    if zeta_est == 0:
        zeta_est = 0.001
    return fn * 2 * np.pi, zeta_est


def sdof_frf_fit(y, f, fn):
    y = y[~np.isnan(y)]
    f = f[~np.isnan(f)]

    # rearrange arrays
    y = y.ravel()
    f = f.ravel()

    # calculate initial guess with half power method
    initial_guess = sdof_half_power(f, y, fn)

    # Perform optimization
    popt = scipy.optimize.curve_fit(sdof_frf, f, y, p0=initial_guess, bounds=([0, 0], [1000, 1]))[0]

    # Extract optimized parameters
    omega_n_optimized, zeta_optimized = popt

    return omega_n_optimized, zeta_optimized


def sdof_time_domain_fit(y, f, fs, n_skip=3, n_peaks=30):
    y[np.isnan(y)] = 0

    # rearrange arrays
    y = y.ravel()
    f = f.ravel()

    # inverse fft to get autocorrelation function
    sdof_corr = np.fft.ifft(y, n=len(y)*20, axis=0, norm='ortho').real
    df = np.mean(np.diff(f))        # frequency resolution of the spectrum
    dt = 1/len(f)/df                # sampling time
    t = np.linspace(0, len(f)*dt, len(sdof_corr))

    # normalize and cut correlation
    sdof_corr = sdof_corr.real / np.max(sdof_corr.real)
    sdof_corr = sdof_corr[:len(sdof_corr) // 2]
    t = t[:len(t) // 2]
    # find zero crossing indices (with sign changes ... similar to pyOMA)
    sign = np.diff(np.sign(sdof_corr))
    zero_crossing_idx = np.where(sign)[0]
    # find maxima and minima between the zero crossings (peaks/valleys)
    maxima = [np.max(sdof_corr[zero_crossing_idx[i]:zero_crossing_idx[i + 2]])
              for i in range(0, len(zero_crossing_idx) - 2, 2)]
    minima = [np.min(sdof_corr[zero_crossing_idx[i]:zero_crossing_idx[i + 2]])
              for i in range(0, len(zero_crossing_idx) - 2, 2)]
    # match the lengths of the arrays to be able to fit them in a single array
    if len(maxima) > len(minima):
        maxima = maxima[:-1]
    elif len(maxima) < len(minima):
        minima = minima[:-1]
    # indices of minima and maxima
    maxima_idx = np.zeros((len(maxima)), dtype=int)
    minima_idx = np.zeros((len(minima)), dtype=int)
    for i in range(len(minima)):
        maxima_idx[i] = np.where(sdof_corr == maxima[i])[0][0]
        minima_idx[i] = np.where(sdof_corr == minima[i])[0][0]
    # Fit maxima and minima in single array and flatten
    minmax = np.array((minima, maxima))
    minmax = np.ravel(minmax, order='F')
    # Fit maxima and minima indices in single array and flatten
    minmax_idx = np.array((minima_idx, maxima_idx))
    minmax_idx = np.ravel(minmax_idx, order='F')
    # Remove first and last peaks with n_skip (peaks to skip) and n_peaks (number of peaks to use in fit)
    if (n_skip + n_peaks) >= len(minmax):
        n_peaks = len(minmax) - n_skip
    minmax_fit = np.array([minmax[_a] for _a in range(n_skip, n_skip + n_peaks)])
    minmax_fit_idx = np.array([minmax_idx[_a] for _a in range(n_skip, n_skip + n_peaks)])
    # plot the minima and maxima over the free decay
    plt.plot(t, sdof_corr)
    plt.plot(t[minmax_fit_idx], minmax_fit)
    plt.grid(visible=True, which='minor')
    plt.show()
    # natural frequency estimation
    fn_est = 1 / np.mean(np.diff(t[minmax_fit_idx]) * 2)
    # Fit damping ratio
    delta = np.array([2 * np.log(np.abs(minmax[0]) / np.abs(minmax[_i])) for _i in range(len(minmax_fit))])

    def fun(x, m):
        return m * x

    m, _ = scipy.optimize.curve_fit(fun, np.arange(len(minmax_fit)), delta)
    zeta_fit = m / np.sqrt(4 * np.pi ** 2 + m ** 2)
    fn_fit = fn_est / np.sqrt(1 - zeta_fit ** 2)

    return fn_fit * 2 * np.pi, zeta_fit


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


def plot_fit(fSDOF, sSDOF, wn, zeta):
    # Plot Fitted SDOF-Bell-Functions
    # Determine the number of rows and columns
    _, nPeaks = fSDOF.shape
    if nPeaks != 0:
        num_rows = (nPeaks + 1) // 2
        num_cols = 2 if nPeaks > 1 else 1
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 8))
        for i in range(nPeaks):
            # Frequency vector
            freq_start = fSDOF[~np.isnan(fSDOF[:, i])][0][i]
            freq_end = fSDOF[~np.isnan(fSDOF[:, i])][-1][i]
            freq_vec = np.linspace(freq_start, freq_end, 1000)
            sSDOF_fit = sdof_frf(freq_vec, wn[i, :], zeta[i, :])
            scaling_factor = max(sSDOF[:, i]) / max(sSDOF_fit)
            if num_cols != 1:
                axs[i // num_cols, i % num_cols].plot(fSDOF[:, i], sSDOF[:, i].real)
                axs[i // num_cols, i % num_cols].plot(freq_vec, sSDOF_fit * scaling_factor)  # *scaling_factor
                axs[i // num_cols, i % num_cols].set_title(f'SDOF-Fit {i + 1}')
            else:
                axs.plot(fSDOF[:, i], sSDOF[:, i].real)
                axs.plot(freq_vec, sSDOF_fit * scaling_factor)
                axs.set_title(f'SDOF-Fit {i + 1}')

        # Adjust layout and log scale axis
        plt.tight_layout()

        # Show the plot
        plt.show()


def plot_modeshape(N, E, mode_shape):
    # scale mode shapes according to the size of the object
    x_diff = np.max(N[:, 0]) - np.min(N[:, 0])
    y_diff = np.max(N[:, 1]) - np.min(N[:, 1])
    longest_dim = np.max([x_diff, y_diff])
    mode_shape = mode_shape / np.max(np.abs(mode_shape)) * (longest_dim / 12)

    # Write the mode shape (z-coordinates) into the node vector
    N_temp = np.zeros((N.shape[0], N.shape[1] + 1))
    N_temp[:, 2] = np.abs(mode_shape)
    N_temp[:, :2] = N
    N = N_temp

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

    def set_axes_equal(ax):
        """
        Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc.

        Input
          ax: a matplotlib axis, e.g., as output from plt.gca().
        """

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

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Make the norm
    norm = colors.Normalize(vmin=np.min(N[:, 2]), vmax=np.max(N[:, 2]), clip=False)
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

    # Plot each element face with interpolated color based on displacement
    for element in E:
        # Get the coordinates of the nodes for this element
        nodes = np.zeros((3, 3))
        i = 0
        for node_idx in element:
            nodes[i, :] = N[node_idx - 1, :]
            i = i + 1
        # Extract x, y, z coordinates of the nodes
        x, y, z = nodes[:, 0], nodes[:, 1], nodes[:, 2]

        # refine mesh for interpolated colormapping
        triang = tri.Triangulation(x, y)
        refiner = tri.UniformTriRefiner(triang)
        interpolator = tri.LinearTriInterpolator(triang, z)
        new, new_z = refiner.refine_field(z, interpolator, subdiv=4)

        # Plot the polygon
        ax.plot_trisurf(new.x, new.y, new_z, cmap=myMap, norm=norm, alpha=1, linewidth=0)
        ax.plot_trisurf(x, y, z, triangles=[[0, 1, 2]], cmap=cm_transparent, linewidth=1, edgecolor='black')

    # Set plot limits
    ax.set_xlim(np.min(N[:, 0]), np.max(N[:, 0]))
    ax.set_ylim(np.min(N[:, 1]), np.max(N[:, 1]))
    ax.set_zlim(np.min(N[:, 2]), np.max(N[:, 2]))

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    set_axes_equal(ax)
    plt.show()


def modeshape_scaling(ms):
    max_ms = np.max(np.abs(ms))
    ms_scaled = ms / max_ms
    return ms_scaled-ms_scaled[0]
