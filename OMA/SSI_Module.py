import matplotlib.pyplot as plt
import numpy as np


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

    q1 = q[:a, :]
    q2 = q[a:a + b, :]

    # Projection Matrix
    p_i = np.vstack((l21, l31)) @ q1
    # Output sequence
    y_i = np.hstack((l21, l22)) @ np.vstack((q1, q2))
    return p_i, y_i


def sv_decomp_ssi(p):
    print("     SVD of Projection Matrix started...")
    u, s, v_t = np.linalg.svd(p, full_matrices=False)
    s = np.sqrt(np.diag(s))
    print("     SVD of Projection Matrix ended...")
    return u, s, v_t


def ssi_proc(data, fs, ord_min, ord_max, d_ord):
    print("Stochastic Subspace Identification started...")
    # Dimensions
    n_data, n_ch = data.shape
    br = ord_max // n_ch

    # Calculate Block-Hankel-Matrix
    h = block_hankel_matrix(data=data,
                            br=br)

    # QR-factorization
    q, l = qr_decomp(h=h)

    # Projection matrix
    p_i, y_i = projection_mat(q=q,
                              l=l,
                              n_ch=n_ch,
                              br=br)

    # Singular Value decomposition of projection Matrix
    u, s, v_t = sv_decomp_ssi(p=p_i)

    freqs = []
    zeta = []
    phi = []

    for i in range(ord_min + d_ord, ord_max + 1, d_ord):
        state = "     Order " + str(i) + "/" + str((ord_max - ord_min))
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
        # Output matrix C
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
    return freqs, zeta, phi


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
                        mac_calc(m_old, m_cur) <= (1 - limits[2]):
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
    # Plot Stabilization Diagram
    handles = []
    if plot == 'FDM':
        for i, f in enumerate(freqs[0]):
            plt.scatter(f, [order[0][i]], marker='x', c='black')
        # Manually add a legend
        point0 = plt.Line2D([0], [0],
                            label='Stable in Frequency, Damping and Mode Shape',
                            marker='x',
                            color='black',
                            linestyle='')
        handles = [point0]
    elif plot == 'all':
        for i, f in enumerate(freqs[0]):
            plt.scatter(f, [order[0][i]], marker='x', c='black')
        # Manually add a legend
        point0 = plt.Line2D([0], [0],
                            label='Stable in Frequency, Damping and Mode Shape',
                            marker='x',
                            color='black',
                            linestyle='')
        for i, f in enumerate(freqs[1]):
            plt.scatter(f, [order[1][i]], marker='o', c='black', alpha=0.6)
        # Manually add a legend
        point1 = plt.Line2D([0], [0],
                            label='Stable in Frequency and Damping',
                            marker='o',
                            color='black',
                            alpha=0.6,
                            linestyle='')
        for i, f in enumerate(freqs[2]):
            plt.scatter(f, [order[2][i]], marker='.', c='black', alpha=0.3)
        # Manually add a legend
        point2 = plt.Line2D([0], [0],
                            label='Stable in Frequency',
                            marker='.',
                            color='black',
                            alpha=0.3,
                            linestyle='')
        handles = [point0, point1, point2]

    plt.xlim([0, cutoff])
    plt.xlabel("f (Hz)")
    plt.ylabel("Model Order")
    plt.grid(visible=True, which='both')
    plt.legend(handles=handles)
    plt.show()


def ssi_extract(ranges, freqs, zeta, modes):
    for i, _range in enumerate(ranges):
        for j in range(len(freqs)):
            f_to_avg = []
            z_to_avg = []
            m_to_avg = []
            if _range[0] <= freqs[j] <= _range[1]:
                f_to_avg.append(freqs[j])
                z_to_avg.append(zeta[j])
                m_to_avg.append(modes[j])
            arr = np.array(m_to_avg)
            print(arr.shape)
            print('nloop')
    return 0, 0, 0
