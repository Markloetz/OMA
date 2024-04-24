import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import scipy


# Functions
def import_data(filename, plot, fs, time):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    data = np.array(data, dtype=float)
    if time < (data.shape[0] / fs):
        data = data[:(fs * time), :]
        t_vec = np.linspace(0, time, time * fs)
    else:
        t_vec = np.linspace(0, data.shape[0] / fs, data.shape[0])
    # Some data processing
    n_rows, n_cols = data.shape
    for i in range(n_cols):
        data[:, i] = data[:, i] - np.mean(data[:, i])

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

    # return data
    return data


def cpsd_matrix(data, fs, zero_padding=True):
    # get dimensions
    n_rows, n_cols = data.shape

    # Zero padding
    if zero_padding:
        n_padding = 4
        buffer = np.zeros((n_rows * n_padding, n_cols))
        buffer[:n_rows, :] = data
        data = buffer
        n_rows = n_rows * n_padding

    # CSPD-Parameters (PyOMA) -> not ideal for EFDD, especially fitting
    # df = fs/n_rows
    # n_per_seg = int(fs/df)
    # n_overlap = np.floor(n_per_seg*0.5)
    # window = 'hann'
    # CSPD-Parameters (Matlab-Style) -> very good for fitting
    window = 'hamming'
    n_per_seg = np.floor(n_rows / 8)  # divide into 8 segments
    n_overlap = 0  # Matlab uses zero overlap

    # preallocate cpsd-matrix and frequency vector
    n_fft = int(n_per_seg / 2 + 1)  # limit the amount of fft datapoints to increase speed
    cpsd = np.zeros((n_cols, n_cols, n_fft), dtype=np.complex_)
    f = np.zeros((n_fft, 1))

    # Build cpsd-matrix
    for i in range(n_cols):
        for j in range(n_cols):
            f, cpsd[i, j, :] = scipy.signal.csd(data[:, i],
                                                data[:, j],
                                                fs=fs,
                                                nperseg=n_per_seg,
                                                noverlap=n_overlap,
                                                window=window)

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
    s3 = np.zeros((n_rows, 1))
    u3 = np.zeros((n_rows, n_cols), dtype=complex)

    # SVD
    for i in range(n_rows):
        u, s, _ = np.linalg.svd(mat[:, :, i])
        u1[i, :] = u[:, 0].transpose()
        s1[i, :] = s[0]
        u2[i, :] = u[:, 1].transpose()
        s2[i, :] = s[1]
        u2[i, :] = u[:, 2].transpose()
        s2[i, :] = s[2]

    # return function outputs
    return s1, u1, s2, u2, s3, u3


def peak_picking(x, y, y2, y3, fs):
    y = y.ravel()
    y2 = y2.ravel()
    y3 = y3.ravel()
    x = x.ravel()
    min_prominence = 0
    max_prominence = 10
    prominence = (min_prominence, None)

    # Adjusting peak-prominence with slider
    # Create the plot
    figure1, ax1 = plt.subplots()
    plt.subplots_adjust(bottom=0.25)  # Adjust bottom to make space for the slider

    # Plot the initial data
    locs, _ = scipy.signal.find_peaks(y, prominence=prominence, distance=fs/2)
    y_data = y[locs]
    x_data = x[locs]
    # ax1.plot(x, y)
    line, = ax1.plot(x_data, y_data, 'bo')

    # Add a slider
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Peak Prominence', min_prominence, max_prominence, valinit=min_prominence)

    # Update Function
    def update(val):
        locs_, _ = scipy.signal.find_peaks(y, prominence=(slider.val, None), distance=fs/2)
        y_data_current = y[locs_]
        x_data_current = x[locs_]
        line.set_xdata(x_data_current)
        line.set_ydata(y_data_current)
        figure1.canvas.draw_idle()

    slider.on_changed(update)
    ax1.plot(x, y)
    # Update x and y values
    locs, _ = scipy.signal.find_peaks(y, prominence=(slider.val, None), distance=fs/2)
    y_data = y[locs]
    x_data = x[locs]
    plt.show()

    # Peak Picking
    # Create a figure and axis
    figure2, ax2 = plt.subplots()

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
            ax2.plot(nearest_x, nearest_y, 'ro')  # Plot the selected point in red
            plt.draw()

    # Connect the onclick function to the figure
    _ = figure2.canvas.mpl_connect('button_press_event', onclick)

    # Plot the blue data points
    ax2.plot(x, y)  # Plot the data points in blue
    if max(y2) <= 0.0001 * max(y):
        scaling1 = 1
    else:
        scaling1 = max(y) / max(y2) / 2
    ax2.plot(x, (y2 * scaling1), linewidth=0.7, color='black')
    if max(y3) <= 0.0001 * max(y):
        scaling2 = 1
    else:
        scaling2 = max(y) / max(y3) / 2
    ax2.plot(x, (y3 * scaling2), linewidth=0.7, color='black')
    ax2.plot(x_data, y_data, 'bo')  # Plot the data points in blue
    ax2.set_title('Click to select points')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')

    # Show the plot
    plt.show()

    # remove multiple entries at same spot
    for i in range(1, len(selected_points['x'])):
        if selected_points['x'][i] == selected_points['x'][i - 1]:
            del selected_points['x'][i]
            del selected_points['y'][i]

    # Store number of selected points
    n_points = len(selected_points['x'])
    return selected_points['x'], selected_points['y'], n_points


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
    return group_indices[max_length_ind]


def sdof_frf(f, m, k, zeta):
    omega = 2 * np.pi * f
    h = 1 / (-m * omega ** 2 + 1j * 2 * np.pi * f * zeta * m + k)
    return np.abs(h)


def sdof_frf_fit(y, f):
    y = y[~np.isnan(y)]
    f = f[~np.isnan(f)]

    # rearrange arrays
    y = y.ravel()
    f = f.ravel()

    # Initial guess for parameters (m, k, zeta)
    initial_guess = (1.0, 1.0, 0.01)

    # Perform optimization
    popt = scipy.optimize.curve_fit(sdof_frf, f, y, p0=initial_guess, bounds=([0, 0, 0.001], [1000, 1000, 1000]))[0]

    # Extract optimized parameters
    mass_optimized, stiffness_optimised, zeta_optimized = popt
    omega_n_optimized = np.sqrt(stiffness_optimised / mass_optimized)

    return omega_n_optimized, zeta_optimized
