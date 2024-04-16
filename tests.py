import numpy as np
import csv
import matplotlib.pyplot as plt
import scipy

# Global Variables
global x0


# Functions
def import_data(sFilename, bPlot, fs, time):
    with open(sFilename, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    data = np.array(data, dtype=float)
    data = data[:(fs * time), :]

    # Some data processing
    n_rows, n_cols = data.shape
    for i in range(n_cols):
        data[:, i] = data[:, i] - np.mean(data[:, i])

    # Plot data
    if bPlot:
        # Plotting the Data
        plt.plot(data[:, 0])
        plt.plot(data[:, 1])
        plt.plot(data[:, 2])
        plt.xlabel('Time')
        plt.ylabel('Acceleration')
        plt.title('Raw Data')
        plt.grid(True)
        plt.show()

    # return data
    return data


def cpsd_matrix(data, fs):
    # get dimensions
    n_rows, n_cols = data.shape

    # preallocate cpsd-matrix and frequency vector
    n_fft = int(n_rows * 5)  # limit the amount of fft datapoints to increase speed
    cpsd = np.zeros((n_cols, n_cols, n_fft), dtype=np.complex_)
    f = np.zeros((n_fft, 1))

    # Build cpsd-matrix
    for i in range(n_cols):
        for j in range(n_cols):
            f, cpsd[i, j, :] = scipy.signal.csd(data[:, i],
                                                data[:, j],
                                                fs=fs,
                                                detrend=False,
                                                window='hamming',
                                                nfft=n_fft * 2 - 1)

    # return cpsd-matrix
    return cpsd, f


def sv_decomp(mat):
    # get dimensions
    n_cols, _, n_rows = mat.shape

    # preallocate singular values and mode shapes
    s1 = np.zeros((n_rows, 1))
    u1 = np.zeros((n_rows, n_cols), dtype=complex)

    # SVD
    for i in range(n_rows):
        u, s, _ = np.linalg.svd(mat[:, :, i])
        u1[i, :] = u[:, 0].transpose()
        s1[i, :] = s[0]

    # return function outputs
    return s1, u1


def peak_picking(x, y):
    y = y.ravel()
    x = x.ravel()
    locs, _ = scipy.signal.find_peaks(y)
    y_data = y[locs]
    x_data = x[locs]

    # Create a figure and axis
    fig, ax = plt.subplots()

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
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    # Plot the blue data points
    ax.plot(x, y)  # Plot the data points in blue
    ax.plot(x_data, y_data, 'bo')  # Plot the data points in blue
    ax.set_title('Click to select points')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Show the plot
    plt.show()

    # Store number of selected points
    n_points = len(selected_points['x'])

    return selected_points['x'], selected_points['y'], n_points


def mac_calc(phi, u):
    # calculates mac value between phi and u
    return (np.abs(phi.conj().T @ u) ** 2) / ((phi.conj().T @ phi) * (u.conj().T @ u))


def find_widest_range(array):
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

    # find length of the groups
    lengths = np.zeros((len(group_indices)))
    for i in range(len(group_indices)):
        lengths[i] = len(group_indices[i])

    # extract indices with maximum length
    max_length_ind = np.argmax(lengths)
    return group_indices[max_length_ind]


def sdof_frf(f, m, k, zeta):
    omega = 2 * np.pi * f
    h = 1 / (-m * omega**2 + 1j * 2 * np.pi * f * zeta * m + k)
    return np.abs(h)


def sdof_frf_fit(y, f):
    y = y[~np.isnan(y)]
    f = f[~np.isnan(f)]

    # rearrange arrays
    y = y.ravel()
    f = f.ravel()

    # Initial guess for parameters (m, k, zeta)
    initial_guess = (1.0, 1.0, 0.1)

    # Perform optimization
    popt = scipy.optimize.curve_fit(sdof_frf, f, y, p0=initial_guess, bounds=(0, [1000, 1000, 1000]))[0]

    # Extract optimized parameters
    mass_optimized, stiffness_optimised, zeta_optimized = popt
    omega_n_optimized = np.sqrt(stiffness_optimised/mass_optimized)

    return omega_n_optimized, zeta_optimized


if __name__ == '__main__':
    # Specify Sampling frequency
    Fs = 1000

    # import data (and plot)
    acc = import_data('MDOF_Data.csv', False, Fs, 20)

    # Build CPSD-Matrix from acceleration data
    mCPSD, vf = cpsd_matrix(acc, Fs)

    # SVD of CPSD-matrix @ each frequency
    S, U = sv_decomp(mCPSD)

    # Peak-picking (automated for this case)
    # nPeaks = 3  # maximum number of expected peaks (use number of sensors)
    # fPeaks, Peaks = aut_peak(vf, S, nPeaks)
    fPeaks, Peaks, nPeaks = peak_picking(vf, S)

    # extract mode shape at each peak
    _, mPHI = U.shape
    PHI = np.zeros((nPeaks, mPHI), dtype=np.complex_)
    for i in range(nPeaks):
        PHI[i, :] = U[np.where(vf == fPeaks[i]), :]

    # EFDD-Procedure
    # calculate mac value @ each frequency for each peak
    nMAC, _ = S.shape
    mac_vec = np.zeros((nMAC, nPeaks), dtype=np.complex_)
    for i in range(nPeaks):
        for j in range(nMAC):
            mac = mac_calc(PHI[i, :], U[j, :])
            if mac.real < 0.85:
                mac_vec[j, i] = 0
            else:
                mac_vec[j, i] = mac

    # Filter the SDOFs
    # Find non-zero indices
    fSDOF = np.full((nMAC, nPeaks), np.nan)
    sSDOF = np.full((nMAC, nPeaks), np.nan)
    for i in range(nPeaks):
        indSDOF = find_widest_range(mac_vec[:, i].real)
        fSDOF[:len(indSDOF), i] = vf[indSDOF]
        sSDOF[:len(indSDOF), i] = S[indSDOF, 0]

    # Plotting the singular values
    plt.plot(fPeaks, Peaks, marker='o', linestyle='none')
    for i in range(nPeaks):
        fSDOF_temp_1 = fSDOF[:, i]
        sSDOF_temp_1 = sSDOF[:, i]
        fSDOF_temp_2 = fSDOF[~np.isnan(fSDOF_temp_1)]
        sSDOF_temp_2 = sSDOF[~np.isnan(sSDOF_temp_1)]
        plt.plot(fSDOF_temp_2, sSDOF_temp_2)
    plt.xlabel('Frequency')
    plt.ylabel('Singular Values')
    plt.title('Singular Value Plot')
    plt.grid(True)
    plt.show()

    # # Fitting SDOF in frequency domain
    wn = np.zeros((nPeaks, 1))
    zeta = np.zeros((nPeaks, 1))
    for i in range(nPeaks):
        wn[i, :], zeta[i, :] = sdof_frf_fit(sSDOF[:, i], fSDOF[:, i])

    # Plot Fitted SDOF-Bell-Functions
    # Determine the number of rows and columns
    num_rows = (nPeaks + 1) // 2
    num_cols = 2 if nPeaks > 1 else 1
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    for i in range(nPeaks):
        # Frequency vector
        freq = np.linspace(0, (wn[i, :]+100), 1000)
        sSDOF_fit = sdof_frf(freq, 1, (wn[i, :]**2), zeta[i, :])
        scaling_factor = max(sSDOF[:, i])/max(sSDOF_fit)
        axs[i // num_cols, i % num_cols].plot(fSDOF[:, i], sSDOF[:, i].real)
        axs[i // num_cols, i % num_cols].plot(freq, sSDOF_fit*scaling_factor)
        axs[i // num_cols, i % num_cols].set_title(f'Subplot {i + 1}')

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()

    print(wn/2/np.pi)
    print(zeta)

