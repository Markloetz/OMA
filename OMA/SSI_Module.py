import numpy as np


# This code is a changed version of the code from pyOMA from dagghe (https://github.com/dagghe/PyOMA/tree/master) for
# the data driven SSI... The approach of generating the stabilization diagram is changed to vary the block rows instead
# of the number singular values... This makes the code slower but easier to implement/understand
def mac_calc(phi, u):
    # calculates mac value between phi and u
    return (np.abs(phi.conj().T @ u) ** 2) / ((phi.conj().T @ phi) * (u.conj().T @ u))


def block_hankel_matrix(data, br):
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
    return h


def qr_decomp(h):
    q, l = np.linalg.qr(h.T)
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
    u, s, v_t = np.linalg.svd(p, full_matrices=False)
    s = np.sqrt(np.diag(s))
    return u, s, v_t


def ssi_proc(data, fs, br):
    print("Stochastic Subspace Identification started...")
    # Dimensions
    n_data, n_ch = data.shape

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

    # System Matrix and Output Matrix calculations
    # Observability matrix
    obs = u @ s
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
    freqs = np.abs(var_lambda / 2 / np.pi)
    zeta = np.abs(np.real(var_lambda)) / np.abs(var_lambda)
    # The eigenvector together with the output matrix C determine the mode shapes
    phi = c_mat @ psi  # each column contains one mode shape

    # Return modal parameters
    print("Stochastic Subspace Identification complete...")
    return freqs, zeta, phi
