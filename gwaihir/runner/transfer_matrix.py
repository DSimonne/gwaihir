# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

# Transfer matrix
import bcdi.graph.graph_utils as gu
from bcdi.experiment.detector import create_detector
from bcdi.experiment.setup import Setup
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.simulation.simulation_utils as simu
import bcdi.utils.image_registration as reg
import bcdi.utils.utilities as util
import bcdi.utils.validation as valid
import numpy as np
from numbers import Real, Integral
from collections.abc import Sequence


def compute_transformation_matrix(
    scan,
    original_size,
    reconstruction_file,
    keep_size,
    detector,
    template_imagefile,
    phasing_binning,
    comment,
    preprocessing_binning,
    pixel_size,
    tilt_angle,
    beamline,
    energy,
    outofplane_angle,
    inplane_angle,
    rocking_angle,
    sdd,
    sample_offsets,
    actuators,
    custom_scan,
    custom_motors,
    fix_voxel,
    ref_axis_q,
    sample_name,
    root_folder,
    save_dir,
    specfile_name,
    centering_method,
    # GUI=True
):

    #######################
    # Initialize detector #
    #######################
    detector = create_detector(
        name=detector,
        template_imagefile=template_imagefile,
        binning=phasing_binning,
        preprocessing_binning=preprocessing_binning,
        pixel_size=pixel_size,
    )

    ####################################
    # define the experimental geometry #
    ####################################
    # correct the tilt_angle for binning
    tilt_angle = tilt_angle * preprocessing_binning[0] * phasing_binning[0]
    setup = Setup(
        beamline=beamline,
        detector=detector,
        energy=energy,
        outofplane_angle=outofplane_angle,
        inplane_angle=inplane_angle,
        tilt_angle=tilt_angle,
        rocking_angle=rocking_angle,
        distance=sdd,
        sample_offsets=sample_offsets,
        actuators=actuators,
        custom_scan=custom_scan,
        custom_motors=custom_motors,
    )

    ########################################
    # Initialize the paths and the logfile #
    ########################################
    setup.init_paths(
        sample_name=sample_name,
        scan_number=scan,
        root_folder=root_folder,
        save_dir=save_dir,
        specfile_name=specfile_name,
        template_imagefile=template_imagefile,
    )

    logfile = setup.create_logfile(
        scan_number=scan, root_folder=root_folder, filename=detector.specfile
    )

    #########################################################
    # get the motor position of goniometer circles which    #
    # are below the rocking angle (e.g., chi for eta/omega) #
    #########################################################
    _, setup.grazing_angle, _, _ = setup.diffractometer.goniometer_values(
        logfile=logfile, scan_number=scan, setup=setup
    )

    axis_to_array_xyz = {
        "x": np.array([1, 0, 0]),
        "y": np.array([0, 1, 0]),
        "z": np.array([0, 0, 1]),
    }

    if reconstruction_file:
        file_path = reconstruction_file,

    if not reconstruction_file:
        try:
            root = tk.Tk()
            root.withdraw()
        except tk.TclError:
            pass
        file_path = filedialog.askopenfilenames(
            initialdir=detector.scandir,
            filetypes=[("HDF5", "*.h5"), ("NPZ", "*.npz"),
                       ("NPY", "*.npy"), ("CXI", "*.cxi")],
        )

    nbfiles = len(file_path)

    obj, extension = util.load_file(file_path[0])

    if extension == ".h5":
        comment = comment + "_mode"

    print("\n###############\nProcessing data\n###############")
    nz, ny, nx = obj.shape
    print("Initial data size: (", nz, ",", ny, ",", nx, ")")
    if len(original_size) == 0:
        original_size = obj.shape
    print("FFT size before accounting for phasing_binning", original_size)
    original_size = tuple(
        [
            original_size[index] // phasing_binning[index]
            for index in range(len(phasing_binning))
        ]
    )
    print("Binning used during phasing:", detector.binning)
    print("Padding back to original FFT size", original_size)
    obj = util.crop_pad(array=obj, output_shape=original_size)
    nz, ny, nx = obj.shape

    ###########################################################################
    # define range for orthogonalization and plotting - speed up calculations #
    ###########################################################################
    zrange, yrange, xrange = pu.find_datarange(
        array=obj, amplitude_threshold=0.05, keep_size=keep_size
    )

    numz = zrange * 2
    numy = yrange * 2
    numx = xrange * 2
    print(
        f"Data shape used for orthogonalization and plotting: ({numz}, {numy}, {numx})")

    ######################
    # centering of array #
    ######################
    if centering_method == "max":
        avg_obj = pu.center_max(avg_obj)
        # shift based on max value,
        # required if it spans across the edge of the array before COM
    elif centering_method == "com":
        avg_obj = pu.center_com(avg_obj)
    elif centering_method == "max_com":
        avg_obj = pu.center_max(avg_obj)
        avg_obj = pu.center_com(avg_obj)

    #########################################################
    # calculate q of the Bragg peak in the laboratory frame #
    #########################################################
    q_lab = (
        setup.q_laboratory
    )  # (1/A), in the laboratory frame z downstream, y vertical, x outboard
    qnorm = np.linalg.norm(q_lab)
    q_lab = q_lab / qnorm

    #######################
    #  orthogonalize data #
    #######################

    # Changes here
    avg_obj = np.zeros((numz, numy, numx))
    print("\nShape before orthogonalization", avg_obj.shape)

    # ortho_directspace, setup method
    verbose = True
    arrays = avg_obj
    q_com = np.array([q_lab[2], q_lab[1], q_lab[0]])
    initial_shape = original_size
    voxel_size = fix_voxel
    reference_axis = axis_to_array_xyz[ref_axis_q]
    fill_value = 0
    debugging = True
    title = "amplitude"
    input_shape = arrays.shape

    #########################################################
    # calculate the direct space voxel sizes in nm          #
    # based on the FFT window shape used in phase retrieval #
    #########################################################
    dz_realspace, dy_realspace, dx_realspace = setup.voxel_sizes(
        initial_shape,
        tilt_angle=abs(setup.tilt_angle),
        pixel_x=setup.detector.unbinned_pixel_size[1],
        pixel_y=setup.detector.unbinned_pixel_size[0],
    )
    if verbose:
        print(
            "Sampling in the laboratory frame (z, y, x): ",
            f"({dz_realspace:.2f} nm,"
            f" {dy_realspace:.2f} nm,"
            f" {dx_realspace:.2f} nm)",
        )

    if input_shape != initial_shape:
        # recalculate the tilt and pixel sizes to accomodate a shape change
        tilt = setup.tilt_angle * initial_shape[0] / input_shape[0]
        pixel_y = (
            setup.detector.unbinned_pixel_size[0] *
            initial_shape[1] / input_shape[1]
        )
        pixel_x = (
            setup.detector.unbinned_pixel_size[1] *
            initial_shape[2] / input_shape[2]
        )
        if verbose:
            print(
                "Tilt, pixel_y, pixel_x based on the shape of the cropped array:",
                f"({tilt:.4f} deg,"
                f" {pixel_y * 1e6:.2f} um,"
                f" {pixel_x * 1e6:.2f} um)",
            )

        # sanity check, the direct space voxel sizes
        # calculated below should be equal to the original ones
        dz_realspace, dy_realspace, dx_realspace = setup.voxel_sizes(
            input_shape, tilt_angle=abs(tilt), pixel_x=pixel_x, pixel_y=pixel_y
        )
        if verbose:
            print(
                "Sanity check, recalculated direct space voxel sizes (z, y, x): ",
                f"({dz_realspace:.2f} nm,"
                f" {dy_realspace:.2f} nm,"
                f" {dx_realspace:.2f} nm)",
            )
    else:
        tilt = setup.tilt_angle
        pixel_y = setup.detector.unbinned_pixel_size[0]
        pixel_x = setup.detector.unbinned_pixel_size[1]

    if not voxel_size:
        voxel_size = dz_realspace, dy_realspace, dx_realspace  # in nm
    else:
        if isinstance(voxel_size, Real):
            voxel_size = (voxel_size, voxel_size, voxel_size)
        if not isinstance(voxel_size, Sequence):
            raise TypeError(
                "voxel size should be a sequence of three positive numbers in nm"
            )
        if len(voxel_size) != 3 or any(val <= 0 for val in voxel_size):
            raise ValueError(
                "voxel_size should be a sequence of three positive numbers in nm"
            )

    ######################################################################
    # calculate the transformation matrix based on the beamline geometry #
    ######################################################################
    transfer_matrix = setup.transformation_matrix(
        array_shape=input_shape,
        tilt_angle=tilt,
        pixel_x=pixel_x,
        pixel_y=pixel_y,
        # direct_space=True,
        verbose=verbose,
    )

    ################################################################################
    # calculate the rotation matrix from the crystal frame to the laboratory frame #
    ################################################################################
    # (inverse rotation to have reference_axis along q)
    rotation_matrix = util.rotation_matrix_3d(
        axis_to_align=reference_axis, reference_axis=q_com /
        np.linalg.norm(q_com)
    )

    ################################################
    # calculate the full transfer matrix including #
    # the rotation into the crystal frame          #
    ################################################
    transfer_matrix = np.matmul(rotation_matrix, transfer_matrix)
    # transfer_matrix is the transformation matrix of the direct space coordinates
    # the spacing in the crystal frame is therefore given by the rows of the matrix
    d_along_x = np.linalg.norm(transfer_matrix[0, :])  # along x outboard
    d_along_y = np.linalg.norm(transfer_matrix[1, :])  # along y vertical up
    d_along_z = np.linalg.norm(transfer_matrix[2, :])  # along z downstream

    return transfer_matrix
