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


def aut_peak(x, y, n):
    # find all peaks
    y = y.ravel()
    locs, _ = scipy.signal.find_peaks(y)
    vals = y[locs]
    # Sort peak values and get corresponding peak locations
    sorted_indices = (-vals).argsort()  # Sort indices in descending order
    sorted_locs = locs[sorted_indices]
    sorted_vals = vals[sorted_indices]

    # output n number of peaks and their corresponding x value
    out1 = x[sorted_locs]
    return out1[:n], sorted_vals[:n]


def aut_peak(x, y, n):
    # find all peaks
    y = y.ravel()
    locs, _ = scipy.signal.find_peaks(y)
    vals = y[locs]
    # Sort peak values and get corresponding peak locations
    sorted_indices = (-vals).argsort()  # Sort indices in descending order
    sorted_locs = locs[sorted_indices]
    sorted_vals = vals[sorted_indices]

    # output n number of peaks and their corresponding x value
    out1 = x[sorted_locs]
    return out1[:n], sorted_vals[:n]


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


def find_widest_range(arr):
    nonzero_indices = np.where(arr != 0)

    # Initialize variables to store the start and end indices of the current non-zero range
    start_index = 0
    end_index = 0

    # Initialize variables to store the length and indices of the longest non-zero range
    max_length = 0
    max_start_index = 0
    max_end_index = 0

    # Iterate through the non-zero indices
    for i in range(1, len(nonzero_indices)):
        # If the current index is contiguous with the previous index
        if nonzero_indices[i] == nonzero_indices[i - 1] + 1:
            # Update the end index of the current non-zero range
            end_index = i
        else:
            # Calculate the length of the current non-zero range
            length = end_index - start_index + 1
            # If the current range is longer than the longest range found so far, update the max indices
            if length > max_length:
                max_length = length
                max_start_index = start_index
                max_end_index = end_index
            # Update the start and end indices for the next non-zero range
            start_index = i
            end_index = i

    # Calculate the indices of the largest non-zero range
    return nonzero_indices[max_start_index:max_end_index + 1]


def sdof_func(t, omega_n, zeta_):
    # Calculate acceleration
    x_dotdot = x0 * omega_n ** 2 * np.exp(-zeta_ * omega_n * t) * np.cos(omega_n * np.sqrt(1 - zeta_ ** 2) * t) - 2 \
               * zeta_ * x0 * omega_n * np.exp(-zeta_ * omega_n * t) * np.sin(omega_n * np.sqrt(1 - zeta_ ** 2) * t)
    return x_dotdot


def sdof_fit(y, t, omega_0):
    y = y.ravel()
    t = t.ravel()

    # Initial guess for parameters
    initial_guess = [omega_0, 0.4]

    # Perform optimization
    popt = scipy.optimize.curve_fit(sdof_func, t, y, p0=initial_guess)[0]

    # Extract optimized parameters
    omega_n_optimized, zeta_optimized = popt

    return omega_n_optimized, zeta_optimized


if __name__ == '__main__':
    # Specify Sampling frequency
    Fs = 1000

    # import data (and plot)
    acc = import_data('MDOF_Data.csv', False, Fs, 5)

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
        indSDOF = find_widest_range(mac_vec[:, i])
        fSDOF[indSDOF, i] = vf[indSDOF]
        sSDOF[indSDOF, i] = S[indSDOF, 0]  # :len(indSDOF[0])
        fSDOF[:indSDOF[0][0], i] = vf[:indSDOF[0][0]]
        sSDOF[:indSDOF[0][0], i] = 0
        fSDOF[indSDOF[0][-1]:len(vf), i] = vf[indSDOF[0][-1]:len(vf)]
        sSDOF[indSDOF[0][-1]:len(vf), i] = 0

    # Plotting the singular values
    plt.plot(vf, S)
    plt.plot(fPeaks, Peaks, marker='o', linestyle='none')
    for i in range(nPeaks):
        plt.plot(fSDOF[:, i], sSDOF[:, i])
    plt.xlabel('Frequency')
    plt.ylabel('Singular Values')
    plt.title('Singular Value Plot')
    plt.grid(True)
    plt.show()

    # re-transforming SDOFs into time domain
    ySDOF = np.full(sSDOF.shape, np.nan, dtype=np.complex_)
    tSDOF = np.full(sSDOF.shape, np.nan)
    for i in range(nPeaks):
        y_temp_1 = sSDOF[:, i]
        y_temp_2 = y_temp_1[~np.isnan(y_temp_1)]
        y_temp_3 = np.concatenate((y_temp_2, np.flip(y_temp_2)[1:-1]))
        ySDOF[:len(y_temp_2), i] = np.fft.ifft(y_temp_3)[:len(y_temp_2)]
        tSDOF[:len(y_temp_2), i] = np.arange(0, len(y_temp_2)) / Fs

    # Fitting SDOF
    wn = np.zeros((nPeaks, 1))
    zeta = np.zeros((nPeaks, 1))
    A = np.zeros((nPeaks, 1))
    for i in range(nPeaks):
        y_temp_4 = ySDOF[:, i].real
        t_temp_4 = tSDOF[:, i].real
        y_temp_5 = y_temp_4[~np.isnan(y_temp_4)]
        t_temp_5 = t_temp_4[~np.isnan(t_temp_4)]
        x0 = max(y_temp_5)
        wn[i, :], zeta[i, :] = sdof_fit(y_temp_5, t_temp_5, fPeaks[i])

    print(wn)
    print(zeta)

    # Plot Decay of Accelerations of SDOF equivalents and fitted version
    # Determine the number of rows and columns
    num_rows = (nPeaks + 1) // 2
    num_cols = 2 if nPeaks > 1 else 1
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    for i in range(nPeaks):
        x0 = max(ySDOF[:, i].real)
        ySDOF_fit = sdof_func(tSDOF[:, i], wn[i, :], zeta[i, :])
        axs[i // num_cols, i % num_cols].plot(tSDOF[:, i], ySDOF[:, i].real)
        axs[i // num_cols, i % num_cols].plot(tSDOF[:, i], ySDOF_fit)
        axs[i // num_cols, i % num_cols].set_title(f'Subplot {i + 1}')

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()
