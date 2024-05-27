import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)
import pandas as pd
import seaborn as sns
import mplcursors


# This code is a cleaned up version of the data driven SSI from pyOMA
def mac_calc(phi, u):
    # calculates mac value between phi and u
    return (np.abs(phi.conj().T @ u) ** 2) / ((phi.conj().T @ phi) * (u.conj().T @ u))


def block_hankel_matrix(data, br):
    # Get dimensions
    n_data, n_ch = data.shape
    # Construction of the BHM
    col_h = n_data - 2 * br + 1  #
    h = np.zeros((n_ch * 2 * br, col_h))
    for i in range(0, 2 * br):
        h[i * n_ch:((i + 1) * n_ch), :] = (1 / col_h ** 0.5) * data.T[:, i:i + col_h]

    return h


def lq_factorization(h):
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


def stabilization_diag(Fr, Fr_lab, Sm, br, max_order, max_freq=100, lim_s=0.05):
    _x = Fr.flatten(order='f')
    _y = np.array([_i // len(Fr) for _i in range(len(_x))])
    _l = Fr_lab.flatten(order='f')
    _d = Sm.flatten(order='f')
    # Creating a dataframe out of the flattened results
    df = pd.DataFrame(dict(Frequency=_x, Order=_y, Label=_l, Damp=_d))

    # =============================================================================
    # Reduced dataframe (without nans) where the modal info is saved
    df1 = df.copy()
    df1 = df1.dropna()
    emme = []
    # here I look for the index of the shape associated to a given pole
    for effe, order in zip(df1.Frequency, df1.Order):
        emme.append(np.nanargmin(abs(effe - Fr[:, order])))  # trovo l'indice
    # append the list of indexes to the dataframe
    emme = np.array(emme)
    df1['Emme'] = emme
    # =============================================================================
    df2 = df1.copy()
    # removing the poles that have damping exceding the limit value
    df2.Frequency = df2.Frequency.where(df2.Damp < lim_s)
    # removing the poles that have negative damping
    df2.Frequency = df2.Frequency.where(df2.Damp > 0)

    # Physical poles compare in pairs (complex + conjugate)
    # I look for the poles that DO NOT have a pair and I remove them from the dataframe
    df3 = df2.Frequency.drop_duplicates(keep=False)
    df2 = df2.where(~(df2.isin(df3)))  #
    df2 = df2.dropna()  # Dropping nans
    df2 = df2.drop_duplicates(subset='Frequency')  # removing conjugates

    # df4 = df4.where(df2.Order > ordmin).dropna() # Tengo solo i poli sopra ordmin
    # assigning colours to the labels
    _colors = {0: 'Red', 1: 'darkorange', 2: 'gold', 3: 'yellow', 4: 'Green'}

    fig1, ax1 = plt.subplots()
    ax1 = sns.scatterplot(x=df2['Frequency'], y=df2['Order'] * 2, hue=df2['Label'], palette=_colors)

    ax1.set_xlim(left=0, right=max_freq)
    ax1.set_ylim(bottom=0, top=max_order)
    ax1.xaxis.set_major_locator(MultipleLocator(max_freq / 10))
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax1.xaxis.set_minor_locator(MultipleLocator(max_freq / 100))
    ax1.set_title('''{0} - shift: {1}'''.format('Stabilization Diagram', br))
    ax1.set_xlabel('Frequency [Hz]')
    mplcursors.cursor()
    plt.show()


def ssi_proc(data, fs, br, lim=(0.01, 0.05, 0.02, 0.1)):
    # Dimensions
    n_data, n_ch = data.shape
    order = n_ch * br

    # Calculate Block-Hankel-Matrix
    h = block_hankel_matrix(data=data,
                            br=br)

    # LQ-factorization
    q, l = lq_factorization(h=h)

    # Projection matrix
    p_i, y_i = projection_mat(q=q,
                              l=l,
                              n_ch=n_ch,
                              br=br)

    # Singular Value decomposition of projection Matrix
    u, s, v_t = sv_decomp_ssi(p=p_i)

    # Loop to increase the order of the system (muss no schaun, warum man des macht)
    # frequency matrix
    freq = np.full((order, int(order / 2 + 1)), np.nan)
    # frequency labels
    freq_label = np.full((order, int(order / 2 + 1)), np.nan)
    # Damping ratios
    damping = np.full((order, int(order / 2 + 1)), np.nan)
    modes = []  # initialization of the matrix (list of arrays) that contains the mode shapes
    for i in range(0, int(order / 2 + 1)):
        modes.append(np.zeros((n_ch, i * 2)))

    for i in range(0, order + 1, 2):
        s11 = np.zeros((i, i))  # Inizializzo
        u11 = np.zeros((br * n_ch, i))  # Inizializzo
        v11 = np.zeros((i, br * n_ch))  # Inizializzo
        o1 = np.zeros((br * n_ch - n_ch, i))  # Inizializzo
        o2 = np.zeros((br * n_ch - n_ch, i))  # Inizializzo

        # Extraction of the submatrices for the increasing order of the system
        s11[:i, :i] = s[:i, :i]  #
        u11[:br * n_ch, :i] = u[:br * n_ch, :i]  #
        v11[:i, :br * n_ch] = v_t[:i, :br * n_ch]  #

        obs = u11 @ s11  # Observability matrix
        kal = np.linalg.pinv(obs) @ p_i  # Kalman filter state sequence

        o1[:, :] = obs[:obs.shape[0] - n_ch, :]
        o2[:, :] = obs[n_ch:, :]

        a_mat = np.linalg.pinv(o1) @ o2
        c_mat = obs[:n_ch, :]

        [eig_val, eig_vec] = np.linalg.eig(a_mat)
        _lambda = (np.log(eig_val)) * fs
        fr = abs(_lambda) / (2 * np.pi)  # Natural frequencies of the system
        zeta = -((np.real(_lambda)) / (abs(_lambda)))  # damping ratios
        fr[np.isnan(fr)] = 0

        # Complex mode shapes
        phi = c_mat @ eig_vec

        # we are increasing 2 orders at each step
        i_new = int((i - 0) / 2)

        freq[:len(fr), i_new] = fr  # save the frequencies
        damping[:len(fr), i_new] = zeta  # save the damping ratios
        modes[i_new] = phi  # save the mode shapes

        for idx, (_freq, _smor) in enumerate(zip(fr, zeta)):
            if i_new == 0 or i_new == 1:  # at the first iteration every pole is new
                freq_label[:len(fr), i_new] = 0  #

            else:
                # Find the index of the pole that minimize the difference with iteration(order) n-1
                ind2 = np.nanargmin(abs(_freq - freq[:, i_new - 1])
                                    - min(abs(_freq - freq[:, i_new - 1])))

                aMAC = mac_calc(phi[:, idx], modes[int(i_new - 1)][:, ind2])

                lim_f, lim_s, lim_ms, lim_s1 = lim[0], lim[1], lim[2], lim[3]
                cond1 = abs(_freq - freq[ind2, i_new - 1]) / _freq
                cond2 = abs(_smor - damping[ind2, i_new - 1]) / _smor
                cond3 = 1 - aMAC

                if cond1 < lim_f and cond2 < lim_s and cond3 < lim_ms:
                    freq_label[idx, i_new] = 4  # STABLE POLE

                elif cond1 < lim_f and cond3 < lim_ms:
                    freq_label[idx, i_new] = 3  # Stable for freq. and m.shape

                elif cond1 < lim_f and cond2 < lim_s:
                    freq_label[idx, i_new] = 2  # Stable for freq. and damp.

                elif cond1 < lim_f:
                    freq_label[idx, i_new] = 1  # Stable for freq.
                else:
                    freq_label[idx, i_new] = 0  # New or unstable pole

    stabilization_diag(Fr=freq,
                       Fr_lab=freq_label,
                       Sm=damping,
                       br=br,
                       max_order=order)