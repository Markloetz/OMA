from OMA import FDD_Module as fdd
from OMA import SSI_Module as ssi
import numpy as np
import csv
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm, colors, tri, animation
import glob
import os
import warnings

""" Functions """


def import_data(filename, plot, fs, time, detrend, downsample, cutoff=1000):
    """import_data(filename, plot, fs, time, detrend, downsample, cutoff=1000) imports data from .mat or .csv files.
    Additional filters or other preprocessing can be adjusted... """
    # notify user
    print("Data import started...")
    # load data
    fileending = filename[-3:]
    if fileending == 'csv' or fileending == 'txt':
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
        data = np.array(data, dtype=float)
    elif fileending == 'mat':
        # Load the .mat file
        loaded_data = scipy.io.loadmat(filename)
        # Extract the array from the loaded data
        data = loaded_data['acc']
    else:
        return -1
    # time vector for plot
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

    # # Normalizing data:
    # for i in range(n_cols):
    #     data[:, i] = data[:, i]/data[:, i].std()

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
    nyquist = np.floor(fs_new / 2) - 1
    if cutoff >= nyquist:
        cutoff = nyquist
    b, a = scipy.signal.butter(4, cutoff, btype='low', fs=fs_new, analog=False)
    for i in range(n_cols):
        data[:, i] = scipy.signal.filtfilt(b, a, data[:, i])
    # notify user
    print("Data import ended...")

    # Plot data
    if plot:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        # Plotting the first column of the Data
        plt.plot(t_vec, data[:, 1], label="Modal Point" + str(i + 1))
        plt.xlabel(r'$t$\,/s')
        plt.ylabel(r'$a$\,/ms^{-2}')
        plt.title('Acceleration Data')
        plt.legend()
        plt.grid(True)
        plt.show()

    # return data
    return data, fs_new


def merge_data(path, fs, n_rov, n_ref, ref_channel, rov_channel, ref_pos, t_meas, detrend, cutoff, downsample):
    """merge_data(path, fs, n_rov, n_ref, ref_channel, rov_channel, ref_pos, t_meas, detrend, cutoff, downsample)
    merges datasets in a directory of the type .mat. Only works with one and two reference sensors... """
    # import data and store in one large array
    # preallocate
    n_files = len([name for name in os.listdir(path)])
    data = np.zeros((int(t_meas * fs), n_files * (n_rov + n_ref)))
    for i, filename in enumerate(glob.glob(os.path.join(path, '*.mat'))):
        data_temp, fs = import_data(filename=filename,
                                    plot=False,
                                    fs=fs,
                                    time=t_meas,
                                    detrend=detrend,
                                    downsample=downsample,
                                    cutoff=cutoff)
        data[:, i * (n_rov + n_ref):(i + 1) * (n_rov + n_ref)] = data_temp

    # Fill the merged data array
    # preallocate
    data_out = np.zeros((data.shape[0], n_rov * n_files))
    for i in range(n_files):
        start_idx = i * (n_ref + n_rov)
        for j, ch in enumerate(rov_channel):
            print("out(" + str(i * n_rov + j) + ") = data(" + str(start_idx + ch) + ")")
            data_out[:, i * n_rov + j] = data[:, start_idx + ch]

    # Check if reference sensor(s) need to be merged into the complete dataset
    if ref_pos:
        for i, pos in enumerate(ref_pos):
            print("out(" + str(pos - 1) + ") = data(" + str(ref_channel[i]) + ")")
            data_out = np.insert(data_out, pos - 1, data[:, ref_channel[i]], axis=1)

    return data_out, fs


def modal_extract_fdd(path, Fs, n_rov, n_ref, ref_channel, ref_pos, t_meas, fPeaks, window, overlap, n_seg,
                      mac_threshold=0.95, plot=False):
    """ Extracts the modal parameters for each dataset within a directory, scales the modes, averages the frequencies
    and damping values. This modal_extract function is based on the EFDD-Method...."""
    # variables
    nPeaks = len(fPeaks)
    n_files = len([name for name in os.listdir(path)])
    # Preallocate arrays natural frequencies, damping and the reference/roving modes
    fn = np.zeros((n_files, nPeaks))
    zetan = np.zeros((n_files, nPeaks))
    ref_modes = np.zeros((n_files, nPeaks, n_ref), dtype=np.complex_)
    rov_modes = np.zeros((n_files, nPeaks, n_rov), dtype=np.complex_)

    # Import Data and do EFDD procedure for each dataset
    for i, filename in enumerate(glob.glob(os.path.join(path, '*.mat'))):
        data, _ = import_data(filename=filename,
                              plot=False,
                              fs=Fs,
                              time=t_meas,
                              detrend=False,
                              downsample=False,
                              cutoff=Fs // 2)

        mCPSD, vf = fdd.cpsd_matrix(data=data[:, :2],
                                    fs=Fs,
                                    n_seg=n_seg,
                                    window=window,
                                    overlap=overlap)

        # SVD of CPSD-matrix @ each frequency
        S, U, S2, U2 = fdd.sv_decomp(mCPSD)

        # extract mode shape at each peak
        _, mPHI = U.shape
        PHI = np.zeros((nPeaks, mPHI), dtype=np.complex_)
        for j in range(nPeaks):
            PHI[j, :] = U[np.where(vf == fPeaks[j]), :]

        # calculate mac value @ each frequency for each peak
        nMAC, _ = S.shape
        mac_vec = np.zeros((nMAC, nPeaks), dtype=np.complex_)
        for j in range(nPeaks):
            for k in range(nMAC):
                mac = fdd.mac_calc(PHI[j, :], U[k, :])
                if mac.real < mac_threshold:
                    mac_vec[k, j] = 0
                else:
                    mac_vec[k, j] = mac

        # Filter the SDOFs
        # Find non-zero indices
        fSDOF = np.full((nMAC, nPeaks), np.nan)
        sSDOF = np.full((nMAC, nPeaks), np.nan)
        sSDOF_2 = np.full((nMAC, nPeaks), np.nan)
        uSDOF = np.full((nMAC, nPeaks, n_rov + n_ref), np.nan, dtype=np.complex_)
        for j in range(nPeaks):
            indSDOF = fdd.find_widest_range(array=mac_vec[:, j].real,
                                            center_indices=np.where(vf == fPeaks[j])[0])
            fSDOF[indSDOF, j] = vf[indSDOF]
            sSDOF[indSDOF, j] = S[indSDOF, 0]
            sSDOF_2[indSDOF, j] = S2[indSDOF, 0]
            uSDOF[indSDOF, j] = U[indSDOF, :]
        if n_ref > 0:
            for j in range(nPeaks):
                # reference mode for each natural frequency (dim1 -> number of modal points; dim2 -> number of peaks)
                ref_modes[i, j] = PHI[j, ref_channel]
                # modes from the roving sensors (al modal displacements except the reference ones)
                rov_modes[i, j] = np.delete(PHI[j, :], ref_channel, axis=0)
        else:
            phi_out = PHI

        # Plotting the singular values
        if plot:
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            for j in range(nPeaks):
                fSDOF_temp = fSDOF[:, j][~np.isnan(fSDOF[:, j])]
                sSDOF_temp_1 = sSDOF[:, j][~np.isnan(sSDOF[:, j])]
                sSDOF_temp_2 = sSDOF_2[:, j][~np.isnan(sSDOF_2[:, j])]
                if nPeaks > 1:
                    color = ((0 + j * 0.8) / nPeaks, (0 + j * 0.8) / nPeaks, (0 + j * 0.8) / nPeaks, 1)
                else:
                    color = (0, 0, 0, 1)
                plt.plot(fSDOF_temp, 20 * np.log10(sSDOF_temp_1), color=color)
                plt.axvline(x=fPeaks[j], color='red', label="f = " + str(round(fPeaks[j])) + " Hz")
            plt.xlabel(r'$f$\,/Hz')
            plt.ylabel(r'$Singular\,Values$\,/dB')
            plt.title('Singular Values of SDOF Equivalents')
            plt.grid(True)
            plt.show()

        # Get the Damping Value by fitting the SDOFs in frequency domain
        for j in range(nPeaks):
            fn[i, j], zetan[i, j] = fdd.sdof_time_domain_fit(y=sSDOF[:, j],
                                                             f=vf,
                                                             n_skip=4,
                                                             n_peaks=30,
                                                             plot=False)

    # Average damping and scaling over all datasets
    s_dev_fn = np.std(fn, axis=0)
    s_dev_zeta = np.std(zetan, axis=0)
    fn_out = np.mean(fn, axis=0)
    zeta_out = np.mean(zetan, axis=0)

    if n_ref > 0:
        # decide which one of the reference sensors is used for the scaling
        ref_idx = []
        if n_ref > 1:
            for i in range(nPeaks):
                if np.sum(np.abs(ref_modes[:, i, 0])) > np.sum(np.abs(ref_modes[:, i, 1].real)):
                    ref_idx.append(0)
                else:
                    ref_idx.append(1)

        # Scale the mode shapes according to  the modal displacement of the reference coordinate
        alpha = np.zeros((n_files * n_rov, nPeaks), dtype=np.complex_)
        for i in range(n_files):
            for j in range(nPeaks):
                if n_ref > 1:
                    # modal amplitude of dataset i and frequency j
                    amp = ref_modes[i, j, :].reshape(n_ref, -1)
                    # reference amplitude from dataset 0 and frequency j
                    amp_ref = ref_modes[0, j, :].reshape(n_ref, -1)
                    alpha[i * n_rov:i * n_rov + n_rov, j] = amp[ref_idx[j], :] / amp_ref[ref_idx[j], :]
                else:
                    # modal amplitude of dataset i and frequency j
                    amp = ref_modes[i, j]
                    # reference amplitude from dataset 0 and frequency j
                    amp_ref = ref_modes[0, j]
                    # scaling factor
                    alpha[i * n_rov:i * n_rov + n_rov, j] = (amp / amp_ref)

        # Preallocate modeshape matrix
        phi_not_scaled = np.zeros((nPeaks, n_rov * n_files), dtype=np.complex_)

        # Rearrange roving modes in the order of measurement
        for i in range(nPeaks):
            phi_not_scaled[i, :] = rov_modes[:, i, :].flatten()
        if ref_pos:
            for j, pos in enumerate(ref_pos):
                # add reference modes to the not yet scaled roving modes
                phi_not_scaled = np.insert(phi_not_scaled, pos - 1, ref_modes[0, :, j], axis=1)
                # add scaling factor of 1 (none) at the positions of the reference sensors
                alpha = np.insert(alpha, pos - 1, np.ones(nPeaks), axis=0)
        phi_out = phi_not_scaled * alpha.T
        return fn_out, zeta_out, phi_out, s_dev_fn, s_dev_zeta
    else:
        return fn_out, zeta_out, phi_out, 0, 0


def modal_extract_ssi(path, Fs, n_rov, n_ref, ref_channel, rov_channel, ref_pos, t_meas, fPeaks, limits, ord_min,
                      ord_max, plot=False, cutoff=100, Ts=1, delta_f=0.5):
    """ Extracts the modal parameters for each dataset within a directory, scales the modes, averages the frequencies
    and damping values. This modal_extract function is based on the SSI-COV-Method..."""
    # Parameters
    nPeaks = len(fPeaks)
    n_files = len([name for name in os.listdir(path)])

    # Initializations
    fn = np.zeros((n_files, nPeaks))  # Store each identified natural frequency of each dataset
    zetan = np.zeros((n_files, nPeaks))  # store each identified damping ratio of each dataset
    ref_modes = np.zeros((n_files, nPeaks, n_ref), dtype=np.complex_)  # reference mode shapes are stored here
    rov_modes = np.zeros((n_files, nPeaks, n_rov), dtype=np.complex_)  # roving mode shapes are stored here

    # Frequency Ranges for the SSI_EXTRACT to use
    f_rel = []
    for j in range(nPeaks):
        f_rel.append([fPeaks[j] - delta_f, fPeaks[j] + delta_f])

    # Import Data and do EFDD procedure for each dataset
    for i, filename in enumerate(glob.glob(os.path.join(path, '*.mat'))):
        data, _ = import_data(filename=filename,
                              plot=False,
                              fs=Fs,
                              time=t_meas,
                              detrend=False,
                              downsample=False,
                              cutoff=cutoff)

        # SSI - Procedure
        # Perform SSI algorithm
        freqs, zeta, modes, _, _, status = ssi.SSICOV(data,
                                                      dt=1 / Fs,
                                                      Ts=Ts,
                                                      ord_min=ord_min,
                                                      ord_max=ord_max,
                                                      limits=limits)

        # Extract parameters and store in dedicated arrays
        freqs_extract, zeta_extract, modes_extract = ssi.ssi_extract(freqs, zeta, modes, status, f_rel)
        fn[i, :] = freqs_extract
        zetan[i, :] = zeta_extract
        if n_ref > 0:
            for j in range(nPeaks):
                mode_curr = np.array(modes_extract[j])
                ref_modes[i, j] = mode_curr[ref_channel]
                rov_modes[i, j] = mode_curr[rov_channel]
        else:
            phi_out = np.zeros((nPeaks, n_rov), dtype=np.complex_)
            for j in range(nPeaks):
                mode_curr = np.array(modes_extract[j])
                phi_out[j, :] = mode_curr
        # stabilization diagram
        if plot:
            fig, ax = ssi.stabilization_diag(freqs=freqs,
                                             label=status,
                                             cutoff=cutoff,
                                             order_min=ord_min)
            for j in range(nPeaks):
                ax.axvspan(f_rel[j][0], f_rel[j][1], color='red', alpha=0.3)
            plt.show()
        # End of Loop over Files .........................................................

    # I expect to see RuntimeWarnings in this block
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # Average damping and scaling over all datasets
        s_dev_fn = np.std(fn, axis=0)
        s_dev_zeta = np.std(zetan, axis=0)
        fn_out = np.nanmean(fn, axis=0)
        zeta_out = np.nanmean(zetan, axis=0)

    # Mode shape scaling .................................................................
    if n_ref > 0:
        # decide which one of the reference sensors is used for the scaling
        ref_idx = []
        if n_ref > 1:
            for i in range(nPeaks):
                if np.sum(np.abs(ref_modes[:, i, 0])) > np.sum(np.abs(ref_modes[:, i, 1].real)):
                    ref_idx.append(0)
                else:
                    ref_idx.append(1)

        # Scale the mode shapes according to  the modal displacement of the reference coordinate
        alpha = np.zeros((n_files * n_rov, nPeaks), dtype=np.complex_)
        for i in range(n_files):
            for j in range(nPeaks):
                if n_ref > 1:
                    amp = ref_modes[i, j, :].reshape(n_ref, -1)
                    amp_ref = ref_modes[0, j, :].reshape(n_ref, -1)
                    alpha[i * n_rov:i * n_rov + n_rov, j] = amp[ref_idx[j], :] / amp_ref[ref_idx[j], :]
                else:
                    amp = ref_modes[i, j]
                    amp_ref = ref_modes[0, j]
                    alpha[i * n_rov:i * n_rov + n_rov, j] = (amp / amp_ref)

        # Preallocate modeshape matrix
        phi_not_scaled = np.zeros((nPeaks, n_rov * n_files), dtype=np.complex_)

        # Rearrange roving modes in the order of measurement
        for i in range(nPeaks):
            phi_not_scaled[i, :] = rov_modes[:, i, :].flatten()
        if ref_pos:
            for j, pos in enumerate(ref_pos):
                # add reference modes to the not yet scaled roving modes
                phi_not_scaled = np.insert(phi_not_scaled, pos - 1, ref_modes[0, :, j], axis=1)
                # add scaling factor of 1 (none) at the positions of the reference sensors
                alpha = np.insert(alpha, pos - 1, np.ones(nPeaks), axis=0)
        phi_out = phi_not_scaled * alpha.T
        return fn_out, zeta_out, phi_out, s_dev_fn, s_dev_zeta
    else:
        return fn_out, zeta_out, phi_out, 0, 0


def animate_modeshape(N, E, f_n, zeta_n, mode_shape, mpc, directory, mode_nr, plot=True):
    """ Mode Shape animation function to create and store a gif of the specified mode shape and show it to the
    user... """

    # Create a custom symmetrical colormap
    def symmetrical_colormap(cmap):
        n = 128
        colors_r = cmap(np.linspace(0, 1, n))
        colors_l = colors_r[::-1]
        colors_ = np.vstack((colors_l, colors_r))
        my_map = colors.LinearSegmentedColormap.from_list('symmetric_jet', colors_)
        return my_map

    # Function to set axis of 3d plot equal
    def set_axes_equal(ax):
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    # Pre-Animation Calculations (Initial Data)
    if np.sum(mode_shape) == 0 or mode_shape.any() == np.nan:
        mode_shape = np.zeros(mode_shape.shape)
        mode_shape = mode_shape + 1
    x_diff = np.max(N[:, 0]) - np.min(N[:, 0])
    y_diff = np.max(N[:, 1]) - np.min(N[:, 1])
    longest_dim = np.max([x_diff, y_diff])
    mode_shape = mode_shape / np.max(np.abs(mode_shape)) * (longest_dim / 15)

    N_temp = np.zeros((N.shape[0], N.shape[1] + 1), dtype=np.complex_)
    N_temp[:, 2] = mode_shape
    N_temp[:, :2] = N
    N = N_temp

    x = []
    y = []
    z = []
    refined_x = []
    refined_y = []
    refined_z = []
    for e, element in enumerate(E):
        nodes = np.zeros((3, 3), dtype=np.complex_)
        for i, node_idx in enumerate(element):
            nodes[i, :] = N[node_idx - 1, :]
        x.append(nodes[:, 0])
        y.append(nodes[:, 1])
        z.append(nodes[:, 2])
    n_frames = 20
    for frame in range(n_frames):
        new_x = []
        new_y = []
        new_z = []
        for i in range(len(E)):
            _z = z[i].real * np.cos(np.pi / 5 * frame) + z[i].imag * np.sin(np.pi / 5 * frame)
            triang = tri.Triangulation(x[i].real, y[i].real)
            refiner = tri.UniformTriRefiner(triang)
            interpolator = tri.LinearTriInterpolator(triang, _z)
            new, new_z_temp = refiner.refine_field(_z, interpolator, subdiv=3)
            new_x.append(new.x)
            new_y.append(new.y)
            new_z.append(new_z_temp)
        new_x = np.array(new_x).flatten()
        new_y = np.array(new_y).flatten()
        new_z = np.array(new_z).flatten()
        refined_x.append(new_x)
        refined_y.append(new_y)
        refined_z.append(new_z)

    def update(frame, art0, art1):
        art0[0].remove()
        art1[0].remove()
        art0[0] = ax.plot_trisurf(refined_x[frame].real,
                                  refined_y[frame].real,
                                  refined_z[frame].real,
                                  cmap=myMap,
                                  norm=norm,
                                  alpha=1,
                                  linewidth=0)
        art1[0] = ax.plot_trisurf(N[:, 0].real,
                                  N[:, 1].real,
                                  N[:, 2].real * np.cos(np.pi / 5 * frame) + N[:, 2].imag * np.sin(np.pi / 5 * frame),
                                  triangles=E - 1,
                                  cmap=colors.ListedColormap([(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)]), linewidth=1,
                                  edgecolor='black')
        return art0[0], art1[0]

    fig = plt.figure()
    norm = colors.Normalize(vmin=-np.max(np.abs(N[:, 2].real)) * 1.3, vmax=np.max(np.abs(N[:, 2].real)) * 1.3,
                            clip=False)
    myMap = symmetrical_colormap(cm.jet)
    ax = fig.add_subplot(111, projection='3d')
    title = f"Mode {mode_nr + 1} at {round(f_n, 2)}Hz (Zeta = {round(zeta_n * 100, 2)}%; MPC = {round(mpc * 100, 2)})"
    ax.set_title(title)
    ax.set_xlim(np.min(N[:, 0].real), np.max(N[:, 0].real))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    set_axes_equal(ax)
    art0 = [ax.plot_trisurf(refined_x[0].real,
                            refined_y[0].real,
                            refined_z[0].real,
                            cmap=myMap,
                            norm=norm,
                            alpha=1,
                            linewidth=0)]
    art1 = [ax.plot_trisurf(N[:, 0].real,
                            N[:, 1].real,
                            N[:, 2].real * np.cos(0) + np.real(N[:, 2].imag) * np.sin(0),
                            triangles=E - 1,
                            cmap=colors.ListedColormap([(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)]), linewidth=1,
                            edgecolor='black')]

    ani = animation.FuncAnimation(fig=fig, func=update, fargs=(art0, art1), frames=n_frames, interval=100,
                                  blit=False)
    plt.show()
    # create directory if it doesn't exist:
    filename = f"{directory}mode_{round(f_n)}Hz.gif"
    if not os.path.exists(directory):
        os.makedirs(directory)
    # save the animation
    print("Saving Animation...")
    ani.save(filename, writer='pillow')
    print(f"Animation saved to {filename}!")

    if plot:
        # Enable LaTeX rendering
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        title = f"Mode {mode_nr + 1} at {round(f_n, 2)}Hz (Zeta = {round(zeta_n * 100, 2)}\,\%; MPC = {round(mpc * 100, 2)})"
        ax.set_title(title)
        ax.set_xlim(np.min(N[:, 0].real), np.max(N[:, 0].real))
        ax.set_xlabel(r'x')
        ax.set_ylabel(r'y')
        ax.set_zlabel(r'$Modal\,Displacement$')
        ax.set_title(title)
        set_axes_equal(ax)
        ax.plot_trisurf(N[:, 0].real,
                        N[:, 1].real,
                        N[:, 2].real * np.cos(0) + np.real(N[:, 2].imag) * np.sin(0),
                        triangles=E - 1,
                        cmap=colors.ListedColormap([(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)]), linewidth=1,
                        edgecolor='black')
        plt.show()


def mpc(phi_r, phi_i):
    """mpc(phi_r, phi_i) calculates the Modal Phase Collinearity of a mode shape, which is an indicator for its
    complexity... """
    # Calculate S_xx, S_yy, and S_xy
    S_xx = phi_r.T @ phi_r
    S_yy = phi_i.T @ phi_i
    S_xy = phi_r.T @ phi_i

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        m_p_c = ((S_xx - S_yy) ** 2 + 4 * S_xy ** 2) / ((S_xx + S_yy) ** 2)
    # Return the MPC
    return m_p_c


def modal_coherence_plot(f, s, u, f_peaks, cutoff):
    """modal_coherence_plot(f, s, u, f_peaks, cutoff) plots the modal coherence over the frequency range to show,
    which mode dominates where. This should help with peak picking on difficult data... """

    # Function still needs to be finished, it is propably not needed ata all

    n_peaks = len(f_peaks)
    n_data = len(f)
    # Calculate U at peaks
    _, m = u.shape
    u_peaks = np.zeros((n_peaks, m), dtype=np.complex_)
    for j in range(n_peaks):
        u_peaks[j, :] = u[np.where(f == f_peaks[j]), :]
    # Plot modal Coherence
    fig, ax1 = plt.subplots()
    ax1.set_title('Modal Coherence Indicator')
    ax1.set_xlim([0, cutoff])
    for i in range(n_peaks):
        if n_peaks > 1:
            color = ((n_peaks - i) / n_peaks, (i + 1) / n_peaks, (n_peaks - i) / n_peaks, 1)
        else:
            color = (0, 0, 0, 1)

        d = np.zeros(n_data, dtype=np.complex_)
        for j in range(n_data):
            u_h = np.conj(u[j, :]).T
            d[j] = u_h @ u_peaks[i, :]
        ax1.plot(f, d.real, color=color, label=f'ModalCoherenceIndicator for f_n = {round(f_peaks[i], 2)}')

    # Overlay Singular Values Plot with second y-axis
    ax1.legend()
    ax2 = ax1.twinx()
    ax2.plot(f, s, color='black')

    plt.show()


def plot_mac_matrix(phi_1, phi_2, wn_1, wn_2):
    """
    Compute the MAC matrix between two mode shapes and plot it in 3D.

    Parameters:
    phi_1, phi_2 : ndarray
        Mode shape vectors. Phi1 -> SSI-Reasult; Phi2 -> FDD Result
    natural_frequency : float
        Natural frequency.
    damping_ratio : float
        Damping ratio.
    MPC : float
        Some value MPC (not used in MAC calculation but passed as a parameter).
    """
    # Handle nan
    nan_idx_1 = np.where(np.isnan(wn_1))[0]
    nan_idx_2 = np.where(np.isnan(wn_2))[0]
    nan_idx = np.union1d(nan_idx_1, nan_idx_2)
    wn_1 = np.delete(wn_1, nan_idx)
    wn_2 = np.delete(wn_2, nan_idx)
    phi_1 = np.delete(phi_1, nan_idx, axis=0)
    phi_2 = np.delete(phi_2, nan_idx, axis=0)

    # Compute MAC matrix
    n_freq = phi_1.shape[0]
    MAC = np.zeros((n_freq, n_freq), dtype=np.complex_)
    for i in range(n_freq):
        for j in range(n_freq):
            MAC[i, j] = fdd.mac_calc(phi_1[i, :], phi_2[j, :])

    MAC = MAC.real
    # Plot the MAC matrix in 3D
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    # Colormap
    norm = colors.Normalize(0, 1.5)
    colors_ = plt.cm.Greys(norm(MAC))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y = np.meshgrid(range(n_freq), range(n_freq))
    x = x.flatten()
    y = y.flatten()
    z = np.zeros_like(x)
    dx = dy = 0.8
    dz = MAC.flatten()

    ax.bar3d(x, y, z, dx, dy, dz, shade=True, color=colors_.reshape(-1, 4))
    ax.invert_xaxis()
    # Customize axis labels
    x_labels = [f'{round(wn_1[i], 2)} Hz' for i in range(n_freq)]
    y_labels = [f'{round(wn_2[i], 2)} Hz' for i in range(n_freq)]

    ax.set_xticks(np.arange(n_freq))
    ax.set_yticks(np.arange(n_freq))
    ax.set_xticklabels(x_labels, rotation=90, ha='center')
    ax.set_yticklabels(y_labels, rotation=-45, ha='left')

    # Create a scalar mappable object for color mapping
    # mappable = cm.ScalarMappable(norm=norm, cmap='Greys')
    # mappable.set_array(MAC)
    # Add colorbar
    # cbar = plt.colorbar(mappable, ax=ax, shrink=0.6)
    # cbar.set_label('MAC Value')

    ax.set_zlabel('MAC Value')
    title = f"MAC-Matrix between SSI-COV Modes and FDD Modes"
    plt.title(title)
    plt.show()
