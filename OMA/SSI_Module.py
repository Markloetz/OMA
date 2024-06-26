import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.signal import fftconvolve, correlate
from matplotlib.widgets import Slider


class SliderValClass:
    slider_val = 0


def NExT(x, dt, Ts, method=2):
    """
    Implements the Natural Excitation Technique to retrieve the Impulse Response Function (IRF)
    from the cross-correlation of the measured output y.

    Parameters:
    x (numpy.ndarray): time series of ambient vibrations (1D or 2D array)
    dt (float): Time step
    Ts (float): Duration of subsegments (T < dt * (len(y) - 1))
    method (int): 1 to use the fft without zero padding, 2 to use cross-correlation with zero padding (default: 2)

    Returns:
    tuple: (IRF, t)
        - IRF (numpy.ndarray): impulse response function
        - t (numpy.ndarray): time vector associated with the IRF
    """
    if len(x.shape) == 1:
        x = x[np.newaxis, :]  # Convert to 2D array with a single row if x is a 1D array

    Nxx, _ = x.shape

    # Maximal segment length fixed by T
    M = round(Ts / dt)

    if method == 1:
        IRF = np.zeros((Nxx, Nxx, M))
        for oo in range(Nxx):
            for jj in range(Nxx):
                y1 = np.fft.fft(x[oo, :])
                y2 = np.fft.fft(x[jj, :])
                h0 = np.fft.ifft(y1 * np.conj(y2))
                IRF[oo, jj, :] = h0[:M].real
        # Time vector t associated to the IRF
        t = np.arange(M) * dt
        if Nxx == 1:
            IRF = IRF.squeeze().T

    else:  # method == 2
        IRF = np.zeros((Nxx, Nxx, M + 1))
        for oo in range(Nxx):
            for jj in range(Nxx):
                dummy = correlate(x[oo, :], x[jj, :], mode='full', method='auto') / len(x[oo, :])
                mid = len(dummy) // 2
                IRF[oo, jj, :] = dummy[mid:mid + M + 1]
        if Nxx == 1:
            IRF = IRF.squeeze().T
        # Time vector t associated to the IRF
        t = dt * np.arange(0, M + 1)

    # Normalize the IRF
    if Nxx == 1:
        IRF = IRF / IRF[0]

    return IRF, t


# This code is a changed version of the code from pyOMA from dagghe (https://github.com/dagghe/PyOMA/tree/master) for
# the data driven SSI...
def mac_calc(phi, u):
    # calculates mac value between phi and u
    return (np.abs(phi.conj().T @ u) ** 2) / ((phi.conj().T @ phi) * (u.conj().T @ u))


def block_hankel_matrix(data, br):
    print("     Block-Hankel-Matrix calculations started...")
    # Get dimensions
    n_data, n_ch = data.shape
    # Construction of the BHM
    col_h = n_data - 2 * br + 1
    h = np.zeros((n_ch * 2 * br, col_h))
    data_t = data.T
    for i in range(2 * br):
        h[i * n_ch:((i + 1) * n_ch), :] = (1 / col_h ** 0.5) * data_t[:, i:i + col_h]
        # M = scipy.sparse.csr_matrix(h)
        # spy(M)
    print("     Block-Hankel-Matrix calculations ended...")
    return h


def qr_decomp(h):
    print("     QR-Decomposition started...")
    q, l = np.linalg.qr(h.T)
    print("     QR-Decomposition ended...")
    return q.T, l.T


def projection_mat(q, l, n_ch, br):
    a = n_ch * br
    b = n_ch

    l21 = l[a:a + b, :a]
    l22 = l[a:a + b, a:a + b]
    l31 = l[a + b:, :a]
    l32 = l[a + b:, a:a + b]

    q1 = q[:a, :]
    q2 = q[a:a + b, :]

    # Projection Matrix
    p_i = np.vstack((l21, l31)) @ q1
    # Shifted Projection Matrix
    p_i_1 = np.hstack((l31, l32)) @ np.vstack((q1, q2))
    # Output sequence
    y_i = np.hstack((l21, l22)) @ np.vstack((q1, q2))
    return p_i, p_i_1, y_i


def sv_decomp_ssi(p):
    print("     SVD of Projection Matrix started...")
    u, s, v_t = np.linalg.svd(p, full_matrices=False)
    s = np.sqrt(np.diag(s))
    print("     SVD of Projection Matrix ended...")
    return u, s, v_t


def toeplitz(data, fs, Ts=1):
    print("     Computation of the Toeplitz-Matrix...")
    h, _ = NExT(x=data.T, dt=1 / fs, Ts=Ts)
    N1 = round(h.shape[2] / 2) - 1
    M = h.shape[1]
    T1 = np.zeros((N1 * M, N1 * M))

    for oo in range(1, N1 + 1):
        for ll in range(1, N1 + 1):
            T1[(oo - 1) * M: oo * M, (ll - 1) * M: ll * M] = h[:, :, N1 + oo - ll]
    print("     Computation of the Toeplitz-Matrix complete!")
    # return Toeplitz Matrices
    return T1


def ssi_proc(data, fs, ord_min, ord_max, d_ord, method='CovarianceDriven'):
    print("Stochastic Subspace Identification started...")
    # Dimensions
    n_data, n_ch = data.shape
    br = ord_max // n_ch
    # Pre-allocations
    freqs = []
    zeta = []
    phi = []
    a_mat = 0
    c_mat = 0
    if method == 'DataDriven':

        # Calculate Block-Hankel-Matrix
        h = block_hankel_matrix(data=data,
                                br=br)

        # QR-factorization
        q, l = qr_decomp(h=h)

        # Projection matrix
        p_i, p_i_1, y_i = projection_mat(q=q,
                                         l=l,
                                         n_ch=n_ch,
                                         br=br)

        # Singular Value decomposition of projection Matrix
        u, s, v_t = sv_decomp_ssi(p=p_i)

        for i in range(ord_min + d_ord, ord_max + 1, d_ord):
            state = "     Order " + str(i) + "/" + str(ord_max)
            print(state)
            # Cut the results of the svd results according to the current order of the system
            u1 = u[:br * n_ch, :i]
            s1 = s[:i, :i]
            # System Matrix and Output Matrix calculations
            # Observability matrix
            obs = u1 @ s1
            # Split Matrix
            o1 = obs[:obs.shape[0] - n_ch, :]
            sp = np.linalg.pinv(obs) @ p_i  # kalman state sequence
            sp1 = np.linalg.pinv(o1) @ p_i_1  # shifted kalman state sequence
            ac = np.vstack((sp1, y_i)) @ np.linalg.pinv(sp)
            # System matrix A
            a_mat = ac[:sp1.shape[0]]
            # Output Influence Matrix C
            c_mat = ac[sp1.shape[0]:]
            # The eigenvalues of the system matrix determine the natural frequencies and the damping
            [mu, psi] = np.linalg.eig(a_mat)
            var_lambda = np.log(mu) * fs
            freqs.append(np.abs(var_lambda / 2 / np.pi))
            zeta.append(np.abs(np.real(var_lambda)) / np.abs(var_lambda))
            # The eigenvector together with the output matrix C determine the mode shapes
            phi.append(c_mat @ psi)  # each column contains one mode shape
    elif method == 'CovarianceDriven':
        # Calculate Toeplitz-Matrix
        tm = toeplitz(data=data,
                      Ts=1,
                      fs=fs)

        # Singular Value decomposition of projection Matrix
        u, s, v_t = sv_decomp_ssi(p=tm)

        for i in range(ord_min + d_ord, ord_max + 1, d_ord):
            state = "     Order " + str(i) + "/" + str(ord_max)
            print(state)
            # Cut the results of the svd results according to the current order of the system
            u1 = u[:br * n_ch, :i]
            s1 = s[:i, :i]
            # System Matrix and Output Matrix calculations
            # Observability matrix
            obs = u1 @ s1
            # Split Matrix
            o1 = obs[:obs.shape[0] - n_ch, :]
            o2 = obs[n_ch:, :]
            # System matrix A
            a_mat = np.linalg.pinv(o1) @ o2
            # Output Influence Matrix C
            c_mat = obs[:n_ch, :]
            # The eigenvalues of the system matrix determine the natural frequencies and the damping
            [mu, psi] = np.linalg.eig(a_mat)
            var_lambda = np.log(mu) * fs
            freqs.append(np.abs(var_lambda / 2 / np.pi))
            zeta.append(np.abs(np.real(var_lambda)) / np.abs(var_lambda))
            # The eigenvector together with the output matrix C determine the mode shapes
            phi.append(c_mat @ psi)  # each column contains one mode shape
    # Return modal parameters
    print("Stochastic Subspace Identification complete...")
    return freqs, zeta, phi, a_mat, c_mat


def stabilization_calc(freqs, zeta, modes, limits):
    # stable modal parameters
    freqs_stable_in_f = []
    zeta_stable_in_f = []
    modes_stable_in_f = []
    order_stable_in_f = []
    freqs_stable_in_f_d = []
    zeta_stable_in_f_d = []
    modes_stable_in_f_d = []
    order_stable_in_f_d = []
    freqs_stable_in_f_d_m = []
    zeta_stable_in_f_d_m = []
    modes_stable_in_f_d_m = []
    order_stable_in_f_d_m = []
    for i in range(len(freqs)):
        for j in range(len(freqs[i])):
            if i > 0:
                # Find the closest frequency to current one
                pole_idx = np.argmin(np.abs(freqs[i][j] - freqs[i - 1]))
                f_old = freqs[i - 1][pole_idx]
                f_cur = freqs[i][j]
                # same for damping and modes
                z_old = zeta[i - 1][pole_idx]
                z_cur = zeta[i][j]
                m_old = modes[i - 1][:, pole_idx]
                m_cur = modes[i][:, j]

                # Store frequencies fulfilling certain conditions in separate Lists
                # stable in frequency, damping and mode shape
                if np.abs(f_old - f_cur) / f_cur <= limits[0] and \
                        np.abs(z_old - z_cur) / z_cur <= limits[1] and \
                        mac_calc(m_old, m_cur) >= (1-limits[2]):
                    freqs_stable_in_f_d_m.append(f_cur)
                    zeta_stable_in_f_d_m.append(z_cur)
                    modes_stable_in_f_d_m.append(m_cur)
                    order_stable_in_f_d_m.append(i)
                # stable in frequency and damping
                elif np.abs(f_old - f_cur) / f_cur <= limits[0] and \
                        np.abs(z_old - z_cur) / z_cur <= limits[1]:
                    freqs_stable_in_f_d.append(f_cur)
                    zeta_stable_in_f_d.append(z_cur)
                    modes_stable_in_f_d.append(m_cur)
                    order_stable_in_f_d.append(i)
                # stable in frequency:
                elif np.abs(f_old - f_cur) / f_cur <= limits[0]:
                    freqs_stable_in_f.append(f_cur)
                    zeta_stable_in_f.append(z_cur)
                    modes_stable_in_f.append(m_cur)
                    order_stable_in_f.append(i)

    freqs_out = [freqs_stable_in_f_d_m, freqs_stable_in_f_d, freqs_stable_in_f]
    zeta_out = [zeta_stable_in_f_d_m, zeta_stable_in_f_d, zeta_stable_in_f]
    modes_out = [modes_stable_in_f_d_m, modes_stable_in_f_d, modes_stable_in_f]
    order_out = [order_stable_in_f_d_m, order_stable_in_f_d, order_stable_in_f]

    return freqs_out, zeta_out, modes_out, order_out


def stabilization_diag(freqs, order, cutoff, plot='FDM'):
    # Create a figure and axis object
    fig, ax = plt.subplots()

    handles = []
    if plot == 'FDM':
        for i, f in enumerate(freqs[0]):
            ax.scatter(f, [order[0][i]], marker='x', c='black')
        # Manually add a legend
        point0 = plt.Line2D([0], [0],
                            label='Stable in Frequency, Damping and Mode Shape',
                            marker='x',
                            color='black',
                            linestyle='')
        handles = [point0]
    elif plot == 'all':
        for i, f in enumerate(freqs[0]):
            ax.scatter(f, [order[0][i]], marker='x', c='black')
        # Manually add a legend
        point0 = plt.Line2D([0], [0],
                            label='Stable in Frequency, Damping and Mode Shape',
                            marker='x',
                            color='black',
                            linestyle='')
        for i, f in enumerate(freqs[1]):
            ax.scatter(f, [order[1][i]], marker='o', c='black', alpha=0.6)
        # Manually add a legend
        point1 = plt.Line2D([0], [0],
                            label='Stable in Frequency and Damping',
                            marker='o',
                            color='black',
                            alpha=0.6,
                            linestyle='')
        for i, f in enumerate(freqs[2]):
            ax.scatter(f, [order[2][i]], marker='.', c='black', alpha=0.3)
        # Manually add a legend
        point2 = plt.Line2D([0], [0],
                            label='Stable in Frequency',
                            marker='.',
                            color='black',
                            alpha=0.3,
                            linestyle='')
        handles = [point0, point1, point2]

    ax.set_xlim([0, cutoff])
    ax.set_xlabel("f (Hz)")
    ax.set_ylabel("Model Order")
    ax.grid(visible=True, which='both')
    ax.legend(handles=handles)

    # Return the axis object
    return fig, ax


def ssi_extract(ranges, freqs, zeta, modes):
    freqs_out = []
    zeta_out = []
    modes_out = []
    # iterate over frequency ranges
    for i, _range in enumerate(ranges):
        f_to_avg = []
        z_to_avg = []
        m_to_avg = []
        for j in range(len(freqs)):
            if _range[0] <= freqs[j] <= _range[1]:
                f_to_avg.append(freqs[j])
                z_to_avg.append(zeta[j])
                m_to_avg.append(modes[j])

        f_avg = np.mean(f_to_avg)
        z_avg = np.mean(z_to_avg)
        m_to_avg_arr = np.array(m_to_avg)
        m_avg = np.mean(m_to_avg_arr, axis=0)
        freqs_out.append(f_avg)
        zeta_out.append(z_avg)
        modes_out.append(m_avg)

    return freqs_out, zeta_out, modes_out


def prominence_adjust_ssi(x, y, freqs, order, cutoff, plot='all'):
    # Adjusting peak-prominence with slider
    min_prominence = 0
    max_prominence = abs(max(y))
    # Create the plot
    figure, ax1 = stabilization_diag(freqs=freqs,
                                     order=order,
                                     cutoff=cutoff,
                                     plot=plot)
    plt.subplots_adjust(bottom=0.25)
    ax = ax1.twinx()
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
    ax.plot(x, y)
    ax.set_xlabel('f (Hz)')
    ax.set_ylabel('Singular Values (dB)')
    ax.set_xlim([0, cutoff])
    # figure.tight_layout()
    plt.show()

    return SliderValClass.slider_val


def peak_picking_ssi(x, y, freqs, order, cutoff=100, plot='all'):
    y = y.ravel()
    x = x.ravel()

    # get prominence
    locs, _ = scipy.signal.find_peaks(y,
                                      prominence=(prominence_adjust_ssi(x=x,
                                                                        y=y,
                                                                        freqs=freqs,
                                                                        order=order,
                                                                        cutoff=cutoff,
                                                                        plot=plot),
                                                  None))
    y_data = y[locs]
    x_data = x[locs]
    # Peak Picking
    # Create a figure and axis
    figure, ax_1 = stabilization_diag(freqs, order, cutoff, plot=plot)
    ax = ax_1.twinx()
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
    ax.plot(x, y)
    ax.plot(x_data, y_data, 'bo')  # Plot the data points in blue
    ax.set_title('Click to select points')
    ax.set_xlabel('f (Hz)')
    ax.set_ylabel('Singular Values (dB)')
    figure.tight_layout()
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
