# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

# Correct
try:
    import hdf5plugin  # for P10, should be imported before h5py or PyTables
except ModuleNotFoundError:
    pass
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import tkinter as tk
from tkinter import filedialog
import sys
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.preprocessing.preprocessing_utils as pru
from bcdi.experiment.detector import create_detector
from bcdi.experiment.setup import Setup
import bcdi.utils.utilities as util


# Functions used in the gui

def correct_angles_detector(
    filename,
    direct_inplane,
    direct_outofplane,
    get_temperature,
    reflection,
    reference_spacing,
    reference_temperature,
    high_threshold,
    save_dir,
    scan,
    root_folder,
    sample_name,
    filtered_data,
    peak_method,
    normalize_flux,
    debug,
    beamline,
    actuators,
    is_series,
    custom_scan,
    custom_images,
    custom_monitor,
    custom_motors,
    rocking_angle,
    specfile_name,
    detector,
    roi_detector,
    hotpixels_file,
    flatfield_file,
    template_imagefile,
    beam_direction,
    sample_offsets,
    directbeam_x,
    directbeam_y,
    sdd,
    energy,
    GUI,
):
    """
    Calculate exact inplane and out-of-plane detector angles from the direct beam and Bragg peak positions,
    based on the beamline geometry.

    Supported beamlines: ESRF ID01, PETRAIII P10, SOLEIL SIXS, SOLEIL CRISTAL.

    For Pt samples it gives also an estimation of the temperature based on the thermal expansion.

    Input: direct beam and Bragg peak position, sample to detector distance, energy
    Output: corrected inplane, out-of-plane detector angles for the Bragg peak.
    """

    plt.ion()
    #######################
    # Initialize detector #
    #######################
    detector = create_detector(
        name=detector,
        template_imagefile=template_imagefile,
        roi=roi_detector,
    )

    ####################
    # Initialize setup #
    ####################
    setup = Setup(
        beamline=beamline,
        detector=detector,
        energy=energy,
        rocking_angle=rocking_angle,
        distance=sdd,
        beam_direction=beam_direction,
        custom_scan=custom_scan,
        custom_images=custom_images,
        custom_monitor=custom_monitor,
        custom_motors=custom_motors,
        sample_offsets=sample_offsets,
        actuators=actuators,
        is_series=is_series,
    )

    ########################################
    # Initialize the paths and the logfile #
    ########################################
    # initialize the paths
    setup.init_paths(
        sample_name=sample_name,
        scan_number=scan,
        root_folder=root_folder,
        save_dir=None,
        create_savedir=False,
        specfile_name=specfile_name,
        template_imagefile=template_imagefile,
        verbose=True,
    )

    logfile = setup.create_logfile(
        scan_number=scan, root_folder=root_folder, filename=detector.specfile
    )

    #################
    # load the data #
    #################
    flatfield = util.load_flatfield(flatfield_file)
    hotpix_array = util.load_hotpixels(hotpixels_file)

    if not filtered_data:
        data, _, monitor, frames_logical = pru.load_data(
            logfile=logfile,
            scan_number=scan,
            detector=detector,
            setup=setup,
            flatfield=flatfield,
            hotpixels=hotpix_array,
            normalize=normalize_flux,
            debugging=debug,
        )
    else:
        try:
            root = tk.Tk()
            root.withdraw()
        except tk.TclError:
            pass
        file_path = filedialog.askopenfilename(
            initialdir=detector.scandir + "pynxraw/",
            title="Select 3D data",
            filetypes=[("NPZ", "*.npz")],
        )
        data = np.load(file_path)["data"]
        data = data[detector.roi[0]: detector.roi[1],
                    detector.roi[2]: detector.roi[3]]
        frames_logical = np.ones(data.shape[0]).astype(
            int
        )  # use all frames from the filtered data
    numz, numy, numx = data.shape
    print("Shape of dataset: ", numz, numy, numx)

    ##############################################
    # apply photon threshold to remove hotpixels #
    ##############################################
    if high_threshold != 0:
        nb_thresholded = (data > high_threshold).sum()
        data[data > high_threshold] = 0
        print(
            f"Applying photon threshold, {nb_thresholded} high intensity pixels masked")

    ###############################
    # load releavant motor values #
    ###############################
    (
        tilt_values,
        setup.grazing_angle,
        setup.inplane_angle,
        setup.outofplane_angle,
    ) = setup.diffractometer.goniometer_values(
        logfile=logfile, scan_number=scan, setup=setup, frames_logical=frames_logical
    )
    setup.tilt_angle = (tilt_values[1:] - tilt_values[0:-1]).mean()

    nb_frames = len(tilt_values)
    if numz != nb_frames:
        print("The loaded data has not the same shape as the raw data")
        sys.exit()

    #######################
    # Find the Bragg peak #
    #######################
    z0, y0, x0 = pru.find_bragg(data, peak_method=peak_method)
    z0 = np.rint(z0).astype(int)
    y0 = np.rint(y0).astype(int)
    x0 = np.rint(x0).astype(int)

    print(f"Bragg peak at (z, y, x): {z0}, {y0}, {x0}")
    print(
        f"Bragg peak (full detector) at (z, y, x): {z0},"
        f" {y0+detector.roi[0]}, {x0+detector.roi[2]}"
    )

    ######################################################
    # calculate rocking curve and fit it to get the FWHM #
    ######################################################
    rocking_curve = np.zeros(nb_frames)
    if filtered_data == 0:  # take a small ROI to avoid parasitic peaks
        for idx in range(nb_frames):
            rocking_curve[idx] = data[idx, y0 -
                                      20: y0 + 20, x0 - 20: x0 + 20].sum()
        plot_title = "Rocking curve for a 40x40 pixels ROI"
    else:  # take the whole detector
        for idx in range(nb_frames):
            rocking_curve[idx] = data[idx, :, :].sum()
        plot_title = "Rocking curve (full detector)"
    z0 = np.unravel_index(rocking_curve.argmax(), rocking_curve.shape)[0]

    interpolation = interp1d(tilt_values, rocking_curve, kind="cubic")
    interp_points = 5 * nb_frames
    interp_tilt = np.linspace(
        tilt_values.min(), tilt_values.max(), interp_points)
    interp_curve = interpolation(interp_tilt)
    interp_fwhm = (
        len(np.argwhere(interp_curve >= interp_curve.max() / 2))
        * (tilt_values.max() - tilt_values.min())
        / (interp_points - 1)
    )
    print("FWHM by interpolation", str("{:.3f}".format(interp_fwhm)), "deg")

    # added plot
    plt.close()
    fig, (ax0, ax1) = plt.subplots(2, 1, sharex='col', figsize=(10, 5))
    ax0.plot(tilt_values, rocking_curve, '.')
    ax0.plot(interp_tilt, interp_curve)
    ax0.axvline(tilt_values[z0], color='r', alpha=0.7, linewidth=1)
    ax0.set_ylabel('Integrated intensity')
    ax0.legend(('data', 'interpolation'))
    ax0.set_title(plot_title)
    ax1.plot(tilt_values, np.log10(rocking_curve), '.')
    ax1.plot(interp_tilt, np.log10(interp_curve))
    ax1.axvline(tilt_values[z0], color='r', alpha=0.7, linewidth=1)

    ax1.set_xlabel('Rocking angle (deg)')
    ax1.set_ylabel('Log(integrated intensity)')
    ax0.legend(('data', 'interpolation'))
    plt.savefig(save_dir + "rocking_curve.png")
    plt.show()

    ##############################
    #  Calculate corrected angles #
    ##############################
    bragg_x = detector.roi[2] + x0  # convert it in full detector pixel
    bragg_y = detector.roi[0] + y0  # convert it in full detector pixel

    x_direct_0 = directbeam_x + setup.inplane_coeff * (
        direct_inplane * np.pi / 180 * sdd / detector.pixelsize_x
    )  # inplane_coeff is +1 or -1
    y_direct_0 = (
        directbeam_y
        - setup.outofplane_coeff
        * direct_outofplane
        * np.pi
        / 180
        * sdd
        / detector.pixelsize_y
    )  # outofplane_coeff is +1 or -1

    print(
        f"\nDirect beam at (gam={direct_inplane}, "
        f"del={direct_outofplane}) (X, Y): {directbeam_x}, {directbeam_y}"
    )
    print(
        f"Direct beam at (gam=0, del=0) (X, Y): ({x_direct_0:.2f}, {y_direct_0:.2f})")
    print(
        f"\nBragg peak at (gam={setup.inplane_angle}, "
        f"del={setup.outofplane_angle}) (X, Y): ({bragg_x:.2f}, {bragg_y:.2f})"
    )

    bragg_inplane = setup.inplane_angle + setup.inplane_coeff * (
        detector.pixelsize_x * (bragg_x - x_direct_0) / sdd * 180 / np.pi
    )  # inplane_coeff is +1 or -1
    bragg_outofplane = (
        setup.outofplane_angle
        - setup.outofplane_coeff
        * detector.pixelsize_y
        * (bragg_y - y_direct_0)
        / sdd
        * 180
        / np.pi
    )  # outofplane_coeff is +1 or -1

    print(
        f"\nBragg angles before correction (gam, del): ({setup.inplane_angle:.4f}, "
        f"{setup.outofplane_angle:.4f})"
    )
    print(
        f"Bragg angles after correction (gam, del): ({bragg_inplane:.4f}, "
        f"{bragg_outofplane:.4f})"
    )

    # update setup with the corrected detector angles
    setup.inplane_angle = bragg_inplane
    setup.outofplane_angle = bragg_outofplane

    print(f"\nGrazing angle(s) = {setup.grazing_angle} deg")
    print(f"Rocking step = {setup.tilt_angle:.5f} deg")

    ####################################
    # wavevector transfer calculations #
    ####################################
    kin = (
        2 * np.pi / setup.wavelength * np.asarray(beam_direction)
    )  # in lab frame z downstream, y vertical, x outboard
    kout = setup.exit_wavevector  # in lab.frame z downstream, y vertical, x outboard
    q = (kout - kin) / 1e10  # convert from 1/m to 1/angstrom
    qnorm = np.linalg.norm(q)
    dist_plane = 2 * np.pi / qnorm
    print(f"\nWavevector transfer of Bragg peak: {q}, Qnorm={qnorm:.4f}")
    print(f"Interplanar distance: {dist_plane:.6f} angstroms")

    if get_temperature:
        print("\nEstimating the temperature:")
        temperature = pu.bragg_temperature(
            spacing=dist_plane,
            reflection=reflection,
            spacing_ref=reference_spacing,
            temperature_ref=reference_temperature,
            use_q=False,
            material="Pt",
        )

    #########################
    # calculate voxel sizes #
    #########################
    #  update the detector angles in setup
    setup.inplane_angle = bragg_inplane
    setup.outofplane_angle = bragg_outofplane
    dz_realspace, dy_realspace, dx_realspace = setup.voxel_sizes(
        (nb_frames, numy, numx),
        tilt_angle=setup.tilt_angle,
        pixel_x=detector.pixelsize_x,
        pixel_y=detector.pixelsize_y,
        verbose=True,
    )

    #################################
    # plot image at Bragg condition #
    #################################
    plt.close()
    plt.imshow(np.log10(abs(data[int(round(z0)), :, :])), vmin=0, vmax=5)
    plt.title(f'Central slice at frame {int(np.rint(z0))}')
    plt.colorbar()

    plt.scatter(bragg_x, bragg_y, color='r', alpha=0.7, linewidth=1)
    plt.savefig(save_dir + "central_slice.png")
    plt.show()

    print("End of script \n")
    plt.close()

    # added script
    COM_rocking_curve = tilt_values[z0],
    detector_data_COM = abs(data[int(round(z0)), :, :]),
    try:
        metadata = {
            "tilt_values": tilt_values,
            "rocking_curve": rocking_curve,
            "interp_tilt": interp_tilt,
            "interp_curve": interp_curve,
            "COM_rocking_curve": tilt_values[z0],
            "detector_data_COM": abs(data[int(round(z0)), :, :]),
            "interp_fwhm": interp_fwhm,
            "temperature": temperature,
            "bragg_x": bragg_x,
            "bragg_y": bragg_y,
            "q": q,
            "qnorm": qnorm,
            "dist_plane": dist_plane,
            "bragg_inplane": bragg_inplane,
            "bragg_outofplane": bragg_outofplane,
        }
    except:
        try:
            metadata = {
                "tilt_values": tilt_values,
                "rocking_curve": rocking_curve,
                "interp_tilt": interp_tilt,
                "interp_curve": interp_curve,
                "COM_rocking_curve": tilt_values[z0],
                "detector_data_COM": abs(data[int(round(z0)), :, :]),
                "interp_fwhm": interp_fwhm,
                "bragg_x": bragg_x,
                "bragg_y": bragg_y,
                "q": q,
                "qnorm": qnorm,
                "dist_plane": dist_plane,
                "bragg_inplane": bragg_inplane,
                "bragg_outofplane": bragg_outofplane,
            }
        except Exception as e:
            raise e
    return metadata