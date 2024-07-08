import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import scipy

"""Classes"""


class SliderValClass:
    slider_val = 0


""" Functions """


def group_indices(indices):
    """group_indices(indices) returns groups of neighbouring indices..."""
    # Sort the indices to ensure they are in order
    indices = sorted(indices)
    # Initialize an empty list to store the results
    result = []
    # Initialize variables to track the start and end of a range
    start = indices[0]
    end = start
    for i in range(1, len(indices)):
        if indices[i] == end + 1:
            # If the current index is consecutive, update the end of the range
            end = indices[i]
        else:
            # If the current index is not consecutive, store the current range
            if start == end:
                result.append(start)
            else:
                result.append(np.arange(start, end + 1))
            # Reset the start and end for the new range
            start = indices[i]
            end = start
    # Append the last range or single index
    if start == end:
        result.append(start)
    else:
        result.append(np.arange(start, end + 1))
    return result


def find_nearest(array, value):
    """find_nearest(array, value) retrieves the array value, which is nearest to the specified value and returns
    it... """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def harmonic_est(data, delta_f, f_max, fs, plot=True):
    """harmonic_est(data, delta_f, f_max, fs, plot=True) applies the harmonic estimation technique from the paper
    Eliminating the Influence of Harmonic Components in Operational Modal Analysis from Jacobsen, Andersen, Brincker """
    # notify user
    print("Estimation of harmonic signals started...")
    # get dimensions
    n_rows, n_cols = data.shape
    # Normalize data (zero mean, unit variance)
    data_norm = (data - data.mean(axis=0)) / data.std(axis=0)
    # Run bandpass filter over each frequency and check for kurtosis
    n_filt = int(f_max // delta_f)
    f_axis = np.linspace(0, n_filt * delta_f, n_filt)
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
            # calculate kurtosis
            kurtosis[i] = scipy.stats.kurtosis(data_filt, fisher=True)
        kurtosis_mean[j] = np.mean(kurtosis)
        print(f"Filter at {j * delta_f} Hz")

    # find platykurtic frequencies and sort them
    idx_bad = []
    # detrend kurtosis
    # kurtosis_mean = scipy.signal.detrend(kurtosis_mean, type='linear')
    kurtosis_mean = kurtosis_mean - np.mean(kurtosis_mean)
    for i in range(1, kurtosis_mean.shape[0]):
        if kurtosis_mean[i] <= np.min(kurtosis_mean) / 2.1:
            idx_bad.append(i)

    # Store indices of "harmonic groups" in array
    harmonic_idx = group_indices(idx_bad)
    harmonic_f = []
    for i in range(len(harmonic_idx)):
        harmonic_f.append(harmonic_idx[i] * delta_f)
    # notify user
    print("Estimation of harmonic signals ended...")
    # plot relevant kurtosis mean and frequency range
    # vertival limits
    if plot:
        # Enable LaTeX rendering
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig, ax = plt.subplots()
        ax.set_ylim([-np.max(np.abs(kurtosis_mean)) * 1.5, np.max(np.abs(kurtosis_mean)) * 1.5])
        ax.set_xlim([0, f_max])
        ax.plot(f_axis, kurtosis_mean, color=(0, 0, 0), linewidth=1)
        ax.set_xlabel(r'$f$\,/Hz')
        ax.set_ylabel(r'$Kurtosis$')
        for i in range(len(harmonic_f)):
            if isinstance(harmonic_f[i], np.ndarray):
                ax.axvspan(harmonic_f[i][0], harmonic_f[i][-1], color='red', alpha=0.3)
            else:
                ax.vlines(harmonic_f[i], -np.max(np.abs(kurtosis_mean)) * 1.5, np.max(np.abs(kurtosis_mean)) * 1.5,
                          color='red', alpha=0.3)
        plt.show()

    return harmonic_f


def eliminate_harmonic(f, s, f_range, cutoff=100):
    """eliminate_harmonic(f, s, f_range, cutoff=100) cuts out identified frequency ranges and linearily interpolates
    them... """
    # Enable LaTeX rendering
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    figure, ax = plt.subplots()
    ax.set_xlim([0, cutoff])
    ax.set_xlabel(r'$f$\,/\,Hz')
    ax.set_ylabel(r'$Singular Values$\,/\,dB')
    ax.plot(f, s, color=(0, 0, 0), linewidth=1)
    # Adjust limits
    idx = np.where(f >= cutoff)[0][0]
    lim_low = np.min(s[:idx]) - (np.max(s[:idx]) - np.min(s[:idx])) * 0.1
    lim_high = np.max(s[:idx]) + (np.max(s[:idx]) - np.min(s[:idx])) * 0.1
    ax.set_ylim([lim_low, lim_high])
    for i in range(len(f_range)):
        if isinstance(f_range[i], np.ndarray):
            ax.axvspan(f_range[i][0], f_range[i][-1], color='red', alpha=0.3)
        else:
            ax.vlines(f_range[i], lim_low, lim_high,
                      color='red', alpha=0.3)
    plt.show()
    del figure, ax

    # calulate indices to eliminate and interpolate linearily between
    for i in range(len(f_range)):
        if isinstance(f_range[i], np.ndarray):
            idx_low = np.where(f == find_nearest(f, f_range[i][0]))[0] - 1
            idx_high = np.where(f == find_nearest(f, f_range[i][-1]))[0] + 1
            # interpolate
            idx_low = idx_low[0]
            idx_high = idx_high[0]
            f_low = f[idx_low]
            f_high = f[idx_high]
            s_low = s[idx_low]
            s_high = s[idx_high]
            n_interpolate = idx_high - idx_low
            f_space = np.linspace(f_low, f_high, n_interpolate)
            seg_int = s_low + (f_space - f_low) * (s_high - s_low) / (f_high - f_low)
            s[idx_low:idx_high, 0] = seg_int
        else:
            idx_low = np.where(f == find_nearest(f, f_range[i]))[0] - 1
            idx_high = np.where(f == find_nearest(f, f_range[i]))[0] + 1
            # interpolate
            idx_low = idx_low[0]
            idx_high = idx_high[0]
            f_low = f[idx_low]
            f_high = f[idx_high]
            s_low = s[idx_low]
            s_high = s[idx_high]
            n_interpolate = idx_high - idx_low
            seg_int = np.linspace(0, f_high - f_low, n_interpolate) * (s_high - s_low) / (f_high - f_low) + s_low
            s[idx_low:idx_high, 0] = seg_int

    # plot singular values w.o. harmonic peaks
    figure, ax = plt.subplots()
    # Enable LaTeX rendering
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax.set_xlim([0, cutoff])
    ax.set_ylim([lim_low, lim_high])
    ax.set_xlabel(r'$f$\,/\,Hz')
    ax.set_ylabel(r'$Singular Values$\,/\,dB')
    ax.plot(f, s, color=(0, 0, 0), linewidth=1)
    for i in range(len(f_range)):
        if isinstance(f_range[i], np.ndarray):
            ax.axvspan(f_range[i][0], f_range[i][-1], color='red', alpha=0.3)
        else:
            ax.vlines(f_range[i], lim_low, lim_high,
                      color='red', alpha=0.3)
    plt.show()

    return 10 ** (s / 20)


def cpsd_matrix(data, fs, n_seg=8, window='hamming', overlap=0.5):
    """cpsd_matrix(data, fs, zero_padding=True, n_seg=8, window='hamming', overlap=0.5) applies Welch's Method for
    estimating the SD Matrix needed for FDD... """
    # get dimensions
    n_rows, n_cols = data.shape

    # notify user
    print("CPSD calculations started...")

    # parameters for welch's method
    n_per_seg = np.floor(n_rows / n_seg)
    n_overlap = np.floor(overlap * n_per_seg)

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
    # notify user
    print("CPSD calculations ended...")
    # return cpsd-matrix
    return cpsd, f


def sv_decomp(mat):
    """sv_decomp(mat) performs the singular value decomposition on each SD-Matrix of a Spectrum and returns the first
    two Singular Values/Vectors """
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


def prominence_adjust(x, y, cutoff):
    """prominence_adjust(x, y, cutoff) is a UI-function to interactively change the peak prominence of the findpeaks
    function, to allow for the user to perform peak picking without too many available peaks... """
    # Enable LaTeX rendering
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    # Adjusting peak-prominence with slider
    min_prominence = 0
    max_prominence = abs(max(y))
    # Create the plot
    figure, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)  # Adjust bottom to make space for the slider

    # Plot the initial data
    locs, _ = scipy.signal.find_peaks(y, prominence=(min_prominence, None))
    y_data = y[locs]
    x_data = x[locs]
    line, = ax.plot(x_data, y_data, 'bo')

    # Adjust limits
    idx = np.where(x >= cutoff)[0][0]
    limlow = np.min(y[:idx]) - (np.max(y[:idx]) - np.min(y[:idx])) * 0.1
    limhigh = np.max(y[:idx]) + (np.max(y[:idx]) - np.min(y[:idx])) * 0.1
    ax.set_ylim([limlow, limhigh])

    # Add a slider
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Peak Prominence', min_prominence, max_prominence, valinit=min_prominence)

    # Update Plot
    def update(val):
        SliderValClass.slider_val = val
        locs_, _ = scipy.signal.find_peaks(y, prominence=(SliderValClass.slider_val, None))
        y_data_current = y[locs_]
        x_data_current = x[locs_]
        line.set_xdata(x_data_current)
        line.set_ydata(y_data_current)
        figure.canvas.draw_idle()

    slider.on_changed(update)
    ax.plot(x, y, color=(0, 0, 0), linewidth=1)
    ax.set_xlabel(r'$f$\,/\,Hz')
    ax.set_ylabel(r'$Singular Values$\,/\,dB')
    ax.set_xlim([0, cutoff])
    plt.show()

    return SliderValClass.slider_val


def peak_picking(x, y, y2, n_sval=1, cutoff=100):
    """peak_picking(x, y, y2, n_sval=1, cutoff=100) is a UI-function which allows the user to pick the peaks on a
    spectrum, which it then returns... """
    # Enable LaTeX rendering
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    y = y.ravel()
    y2 = y2.ravel()
    x = x.ravel()

    # get prominence
    locs, _ = scipy.signal.find_peaks(y,
                                      prominence=(prominence_adjust(x, y, cutoff=cutoff), None))
    y_data = y[locs]
    x_data = x[locs]
    # Peak Picking
    # Create a figure and axis
    figure, ax = plt.subplots()
    ax.set_xlim([0, cutoff])
    # Adjust limits
    idx = np.where(x >= cutoff)[0][0]
    limlow = np.min(y[:idx]) - (np.max(y[:idx]) - np.min(y[:idx])) * 0.1
    limhigh = np.max(y[:idx]) + (np.max(y[:idx]) - np.min(y[:idx])) * 0.1
    ax.set_ylim([limlow, limhigh])
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
    ax.plot(x, y, color=(0, 0, 0), linewidth=1, label=r'$S_1$')
    if n_sval > 1:
        ax.plot(x, y2, linewidth=1, color=(0.5, 0.5, 0.5), label=r'$S_2$')
    ax.plot(x_data, y_data, 'bo')  # Plot the data points in blue
    ax.set_title('Peak-Picking')
    ax.set_xlabel(r'$f$\,/\,Hz')
    ax.legend()
    ax.set_ylabel(r'$Singular Values$\,/\,dB')

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
    """mac_calc(phi, u) calculates mac value between phi and u..."""
    return (np.abs(phi.conj().T @ u) ** 2) / ((phi.conj().T @ phi) * (u.conj().T @ u))


def find_widest_range(array, center_indices):
    """find_widest_range(array, center_indices) finds the widest range of values in an array, positioned around a
    center point... """
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


def sdof_time_domain_fit(y, f, n_skip=4, n_peaks=30, plot=True):
    """sdof_time_domain_fit(y, f, n_skip=4, n_peaks=30, plot=True) performs the fitting of the logarithmic decay of
    an SDOF-Equivalent, as is usual for EFDD. This code is partially taken from dagghe's pyOMA ->
    https://github.com/dagghe/PyOMA """
    y[np.isnan(y)] = 0

    # rearrange arrays
    y = y.ravel()
    f = f.ravel()

    # inverse fft to get autocorrelation function
    sdof_corr = np.fft.ifft(y, n=len(y) * 20, axis=0, norm='ortho').real
    df = np.mean(np.diff(f))  # frequency resolution of the spectrum
    dt = 1 / len(f) / df  # sampling time
    t = np.linspace(0, len(f) * dt, len(sdof_corr))

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
    # natural frequency estimation
    fn_est = 1 / np.mean(np.diff(t[minmax_fit_idx]) * 2)
    # plot the minima and maxima over the free decay
    if plot:
        # Enable LaTeX rendering
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.plot(t, sdof_corr, color=(0, 0, 0), linewidth=1, label='Autocorrelation of SDOF bell at ' + str(fn_est) + ' Hz')
        # plt.plot(t[minmax_fit_idx], minmax_fit)
        plt.grid(visible=True, which='minor')
        plt.xlabel(r'$\tau$\,/s')
        plt.ylabel(r'$Normalized\,Amplitude$')
        plt.title('Autocorrelation of SDOF bell at ' + str(round(fn_est, 2)) + ' Hz')
        plt.show()
    # Fit damping ratio
    delta = np.array([2 * np.log(np.abs(minmax[0]) / np.abs(minmax[_i])) for _i in range(len(minmax_fit))])

    def fun(x, m):
        return m * x

    m, _ = scipy.optimize.curve_fit(fun, np.arange(len(minmax_fit)), delta)
    zeta_fit = m / np.sqrt(4 * np.pi ** 2 + m ** 2)
    fn_fit = fn_est / np.sqrt(1 - zeta_fit ** 2)

    if len(y) < 3:
        return np.nan, np.nan
    return fn_fit, zeta_fit


def mode_opt(f, s1, s2, u, f_peak, plot=False):
    """mode_opt(f, s1, s2, u, f_peak, plot=False) extracts the mode of a frequency range dominated by a single mode
    at the place, where the ratio between the forst and the secon SV is largest, since there the influence of
    orthogonality between the Singular vectors is minimized ;-)... """
    # Eliminate nan
    f = f[~np.isnan(f)]
    s1 = s1[~np.isnan(s1)]
    s2 = s2[~np.isnan(s2)]
    u = u[~np.isnan(u)]
    # Calculate ratios between first and second singular values
    s_ratio = s1 / s2
    max_ratio = np.max(s_ratio)
    u_out = u[np.where(s_ratio == max_ratio)]
    if plot:
        # Enable LaTeX rendering
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        # Plot first and second singular values
        plt.plot(f, 20 * np.log10(s1), color=(0.4, 0.4, 0.4), linewidth=0.7, label=r'$S_1$')
        plt.plot(f, 20 * np.log10(s2), color=(0, 0, 0), linestyle='--', linewidth=0.7, label=r'$S_2$')
        plt.plot(f, 20 * np.log10(s1/s2), color=(0, 0, 0), linewidth=1, label='$S_1/S_2$')
        plt.axvline(x=f_peak, color='black')
        plt.xlabel(r'$f$\,/Hz)')
        plt.ylabel(r'$Singular\,Values$\,/dB')
        plt.title('Singular Value Ratio')
        plt.legend()
        plt.show()
    return u_out
