import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate, find_peaks
from matplotlib.widgets import Slider


class SliderValClass:
    slider_val = 0


'''Functions needed by the SSI_COV'''


def NExT(x, dt, Ts, method=2):
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
    print("     SVD started...")
    u, s, v_t = np.linalg.svd(p, full_matrices=False)
    s = np.sqrt(np.diag(s))
    print("     SVD ended...")
    return u, s, v_t


def toeplitz(data, fs, Ts=0.5):
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


def modalID(U, S, Nmodes, Nyy, dt):
    if Nmodes >= S.shape[0]:
        print(f'Warning: Nmodes is larger than the number of rows of S. Nmodes is reduced to {S.shape[0]}.')
        Nmodes = S.shape[0]

    obs = U[:, :Nmodes] @ np.sqrt(S[:Nmodes, :Nmodes])

    IndO = min(Nyy, obs.shape[0])
    C = obs[:IndO, :]
    jb = round(obs.shape[0] / IndO)

    A = np.linalg.pinv(obs[:IndO * (jb - 1), :]) @ obs[-IndO * (jb - 1):, :]
    Di, Vi = np.linalg.eig(A)

    mu = np.log(Di) / dt  # poles
    fn = np.abs(mu[1::2]) / (2 * np.pi)  # eigen-frequencies
    zeta = -np.real(mu[1::2]) / np.abs(mu[1::2])  # modal damping ratio
    phi = np.real(C @ Vi)  # mode shapes
    phi = phi[:, 1::2]

    return fn, zeta, phi


def stabilityCheck(fn0, zeta0, phi0, fn1, zeta1, phi1, eps_freq, eps_zeta, eps_MAC):
    fn, zeta, phi, MAC, stability_status = [], [], [], [], []

    for rr in range(len(fn0)):
        for jj in range(len(fn1)):
            stab_fn = errCheck(fn0[rr], fn1[jj], eps_freq)
            stab_zeta = errCheck(zeta0[rr], zeta1[jj], eps_zeta)
            stab_phi, dummyMAC = getMAC(phi0[:, rr], phi1[:, jj], eps_MAC)

            if stab_fn == 0:
                stabStatus = 0
            elif stab_fn == 1 and stab_phi == 1 and stab_zeta == 1:
                stabStatus = 1
            elif stab_fn == 1 and stab_zeta == 0 and stab_phi == 1:
                stabStatus = 2
            elif stab_fn == 1 and stab_zeta == 1 and stab_phi == 0:
                stabStatus = 3
            elif stab_fn == 1 and stab_zeta == 0 and stab_phi == 0:
                stabStatus = 4
            else:
                raise ValueError('Error: stability_status is undefined')

            fn.append(fn1[jj])
            zeta.append(zeta1[jj])
            phi.append(phi1[:, jj])
            MAC.append(dummyMAC)
            stability_status.append(stabStatus)

    indsort = np.argsort(fn)
    fn = np.array(fn)[indsort]
    zeta = np.array(zeta)[indsort]
    phi = np.array(phi)[indsort, :]
    MAC = np.array(MAC)[indsort]
    stability_status = np.array(stability_status)[indsort]

    return fn, zeta, phi, MAC, stability_status


def getMAC(phi1, phi2, eps_MAC):
    if phi1.shape[0] != phi2.shape[0]:
        raise ValueError('The mode shapes must have the same number of DOFs')

    MAC = (np.abs(np.conj(phi1).T @ phi2) ** 2) / ((np.conj(phi1).T @ phi1) * (np.conj(phi2).T @ phi2))
    stabMAC = errCheck(MAC, 1, eps_MAC)

    return stabMAC, MAC


def errCheck(x1, x2, eps):
    return int(np.abs((x1 - x2) / x1) < eps)


def getStablePoles(fn2, zeta2, phi2, MAC, stability_status):
    fnS, zetaS, phiS, MACS = [], [], [], []

    for oo in range(len(stability_status)):
        ind = (stability_status[oo] == 1) | (stability_status[oo] == 2)
        fnS.extend(fn2[oo][ind])
        zetaS.extend(zeta2[oo][ind])
        phiS.append(phi2[oo][:, ind])
        MACS.append(MAC[oo][ind])

    return fnS, zetaS, phiS, MACS


'''SSI-COV-Function -> Covariance Driven Stochastic Subspace ID'''


def SSICOV(y, dt, Ts, ord_min, ord_max, limits):
    # Get dimensions
    n_dat, n_ch = y.shape

    # Construct Toeplitz Matrix
    toep = toeplitz(data=y,
                    fs=1 / dt,
                    Ts=Ts)

    # Singular Value Decomposition of Teoplitz Matrix
    U, S, _ = sv_decomp_ssi(toep)

    # Iterate over all orders in descending fashion
    j = 1  # ascending iterator
    ord_cur = ord_max  # current order to store for stabilazation diagram
    # Initialize values to store
    fn2, zeta2, phi2, MAC, stability_status, order = [], [], [], [], [], []
    for i in range(ord_max, ord_min - 1, -1):
        if j == 1:
            fn0, zeta0, phi0 = modalID(U, S, i, n_ch, dt)
        else:
            fn1, zeta1, phi1 = modalID(U, S, i, n_ch, dt)
            a, b, c, d, e = stabilityCheck(fn0=fn0,
                                           zeta0=zeta0,
                                           phi0=phi0,
                                           fn1=fn1,
                                           zeta1=zeta1,
                                           phi1=phi1,
                                           eps_freq=limits[0],
                                           eps_zeta=limits[1],
                                           eps_MAC=limits[2])
            fn2.append(a)
            zeta2.append(b)
            phi2.append(c)
            MAC.append(d)
            stability_status.append(e)
            fn0, zeta0, phi0 = fn1, zeta1, phi1
        order.append(ord_cur)
        ord_cur -= 1
        j += 1

    stability_status = list(reversed(stability_status))
    fn2 = list(reversed(fn2))
    zeta2 = list(reversed(zeta2))
    phi2 = list(reversed(phi2))
    MAC = list(reversed(MAC))
    order = list(reversed(order))

    return fn2, zeta2, phi2, order, MAC, stability_status


def stabilization_diag(freqs, label, cutoff):
    # Create a figure and axis object
    fig, ax = plt.subplots()
    for order in range(len(freqs)):
        for i, f in enumerate(freqs[order]):
            if label[order][i] == 4:
                ax.scatter(f, order, marker='x', c='black')
            elif label[order][i] == 3:
                ax.scatter(f, order, marker='o', c='black', alpha=0.6)
            elif label[order][i] == 2:
                ax.scatter(f, order, marker='s', c='black', alpha=0.6)
            elif label[order][i] == 1:
                ax.scatter(f, order, marker='.', c='grey', alpha=0.6)
            else:
                ax.scatter(f, order, marker='.', c='grey', alpha=0.3)
    # Manually add a legend
    point4 = plt.Line2D([0], [0],
                        label='Stable in Frequency',
                        marker='x',
                        color='black',
                        linestyle='')
    point3 = plt.Line2D([0], [0],
                        label='Stable in Frequency and Damping',
                        marker='o',
                        color='black',
                        alpha=0.6,
                        linestyle='')
    point2 = plt.Line2D([0], [0],
                        label='Stable in Frequency and Mode Shape',
                        marker='s',
                        color='black',
                        alpha=0.6,
                        linestyle='')
    point1 = plt.Line2D([0], [0],
                        label='Stable Pole',
                        marker='.',
                        color='black',
                        alpha=0.6,
                        linestyle='')
    point0 = plt.Line2D([0], [0],
                        label='New Pole',
                        marker='.',
                        color='black',
                        alpha=0.3,
                        linestyle='')
    handles = [point0, point1, point2, point3, point4]
    ax.set_xlim([0, cutoff])
    ax.set_xlabel("f (Hz)")
    ax.set_ylabel("Model Order")
    ax.grid(visible=True, which='both')
    ax.legend(handles=handles)

    # Return the axis object
    return fig, ax


def ssi_extract(freqs, zeta, modes, label, ranges):
    # Initialize Output Arrays
    freqs_out = []
    zeta_out = []
    modes_out = []
    f_avg = 0
    z_avg = 0
    m_avg = 0
    for j, _range in enumerate(ranges):
        f_to_avg = []
        z_to_avg = []
        m_to_avg = []
        for order in range(len(freqs)):
            # Loop over each frequency
            for i, f in enumerate(freqs[order]):
                # Check if frequency is stable
                if label[order][i] == 1:
                    if _range[0] <= f <= _range[1]:
                        f_to_avg.append(f)
                        z_to_avg.append(zeta[order][i])
                        m_to_avg.append(modes[order][i])
        f_avg = np.mean(f_to_avg)
        z_avg = np.mean(z_to_avg)
        m_to_avg_arr = np.array(m_to_avg)
        m_avg = np.mean(m_to_avg_arr, axis=0)
        freqs_out.append(f_avg)
        zeta_out.append(z_avg)
        modes_out.append(m_avg)

    return freqs_out, zeta_out, modes_out


def prominence_adjust_ssi(x, y, freqs, label, cutoff):
    # Adjusting peak-prominence with slider
    min_prominence = 0
    max_prominence = abs(max(y))
    # Create the plot
    figure, ax1 = stabilization_diag(freqs=freqs,
                                     label=label,
                                     cutoff=cutoff)
    plt.subplots_adjust(bottom=0.25)
    ax = ax1.twinx()
    plt.subplots_adjust(bottom=0.25)  # Adjust bottom to make space for the slider

    # Plot the initial data
    locs, _ = find_peaks(y, prominence=(min_prominence, None))
    y_data = y[locs]
    x_data = x[locs]
    line, = ax.plot(x_data, y_data, 'bo')

    # Adjust limits
    idx = np.where(x >= cutoff)[0][0]
    limlow = np.min(y[:idx]) - (np.max(y[:idx]) - np.min(y[:idx])) * 0.1
    limhigh = np.max(y[:idx]) + (np.max(y[:idx]) - np.min(y[:idx])) * 0.1
    ax.set_ylim([limlow, limhigh])

    # Add a slider
    ax_slider = plt.axes((0.25, 0.1, 0.65, 0.03), facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Peak Prominence', min_prominence, max_prominence, valinit=min_prominence)

    # Update Plot
    def update(val):
        SliderValClass.slider_val = val
        locs_, _ = find_peaks(y, prominence=(SliderValClass.slider_val, None))
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


def peak_picking_ssi(x, y, freqs, label, cutoff=100, plot='all'):
    y = y.ravel()
    x = x.ravel()

    # get prominence
    locs, _ = find_peaks(y,
                         prominence=(prominence_adjust_ssi(x=x,
                                                           y=y,
                                                           freqs=freqs,
                                                           label=label,
                                                           cutoff=cutoff),
                                     None))
    y_data = y[locs]
    x_data = x[locs]
    # Peak Picking
    # Create a figure and axis
    figure, ax_1 = stabilization_diag(freqs, label, cutoff)
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
