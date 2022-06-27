import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import operator as operator_lib
from datetime import datetime
import tables as tb
import h5py
import shutil
from numpy.fft import fftshift
from scipy.ndimage import center_of_mass
from shlex import quote
from IPython.display import display

# PyNX
try:
    from pynx.cdi import CDI
    from pynx.cdi.runner.id01 import params
    from pynx.utils.math import smaller_primes
    pynx_import = True
except ModuleNotFoundError:
    pynx_import = False

# gwaihir package
import gwaihir

# bcdi package
from bcdi.preprocessing import ReadNxs3 as rd
from bcdi.utils.utilities import bin_data


def initialize_postprocessing(
    self,
    unused_label_averaging,
    sort_method,
    correlation_threshold,
    unused_label_FFT,
    phasing_binning,
    original_size,
    preprocessing_binning,
    output_size,
    keep_size,
    fix_voxel,
    unused_label_disp_strain,
    data_frame,
    save_frame,
    ref_axis_q,
    isosurface_strain,
    skip_unwrap,
    strain_method,
    phase_offset,
    phase_offset_origin,
    offset_method,
    centering_method,
    unused_label_refraction,
    correct_refraction,
    optical_path_method,
    dispersion,
    absorption,
    threshold_unwrap_refraction,
    unused_label_options,
    simulation,
    invert_phase,
    flip_reconstruction,
    phase_ramp_removal,
    threshold_gradient,
    save_raw,
    save_support,
    save,
    debug,
    roll_modes,
    unused_label_data_vis,
    align_axis,
    ref_axis,
    axis_to_align,
    strain_range,
    phase_range,
    grey_background,
    tick_spacing,
    tick_direction,
    tick_length,
    tick_width,
    unused_label_average,
    averaging_space,
    threshold_avg,
    unused_label_apodize,
    apodize,
    apodization_window,
    half_width_avg_phase,
    apodization_mu,
    apodization_sigma,
    apodization_alpha,
    unused_label_strain,
    strain_folder,
    reconstruction_files,
    run_strain,
):
    """
    Interpolate the output of the phase retrieval into an orthonormal frame,
    and calculate the strain component along the direction of the
    experimental diffusion vector q.

    Input: complex amplitude array, output from a phase retrieval program.
    Output: data in an orthonormal frame (laboratory or crystal frame),
    amp_disp_strain array.The disp array should be divided by q to get the
    displacement (disp = -1*phase here).

    Laboratory frame: z downstream, y vertical, x outboard (CXI convention)
    Crystal reciprocal frame: qx downstream, qz vertical, qy outboard
    Detector convention: when out_of_plane angle=0   Y=-y , when in_plane
    angle=0   X=x

    In arrays, when plotting the first parameter is the row (vertical axis),
    and the second the column (horizontal axis). Therefore the data
    structure is data[qx, qz, qy] for reciprocal space, or data[z, y, x]
    for real space.

    Loading argument from strain tab widgets but also values of
    parameters used in preprocessing that are common.

    Parameters used when averaging several reconstruction:

    :param sort_method: e.g. "variance/mean"
     'mean_amplitude' or 'variance' or 'variance/mean' or 'volume',
     metric for averaging
    :param averaging_space: e.g. "reciprocal_space" TODO
     in which space to average, 'direct_space' or 'reciprocal_space'
    :param correlation_threshold: e.g. 0.90
     minimum correlation between two arrays to average them

    Parameters related to centering:

    :param centering_method: e.g. "max_com"
    'com' (center of mass), 'max', 'max_com' (max then com), 'do_nothing'
    :param roll_modes: e.g. [0, 0, 0]
    correct a roll of few pixels after the decomposition into modes in PyNX
    axis=(0, 1, 2)

    Parameters relative to the FFT window and voxel sizes:

    :param original_size: e.g. [150, 256, 500]
     size of the FFT array before binning. It will be modified to take
     into account binning during phasing automatically. Leave it to None
     if the shape did not change.
    :param phasing_binning: e.g. [1, 1, 1]
     binning factor applied during phase retrieval
    :param preprocessing_binning: e.g. [1, 2, 2]
     binning factors in each dimension used in preprocessing (not phase
     retrieval)
    :param output_size: e.g. [100, 100, 100]
     (z, y, x) Fix the size of the output array, leave None to use the
     object size
    :param keep_size: e.g. False
     True to keep the initial array size for orthogonalization (slower),
     it will be cropped otherwise
    :param fix_voxel: e.g. 10
     voxel size in nm for the interpolation during the geometrical
     transformation. If a single value is provided, the voxel size will be
     identical in all 3 directions. Set it to None to use the default voxel
     size (calculated from q values, it will be different in each
     dimension).

    Parameters related to the strain calculation:

    :param data_frame: e.g. "detector"
     in which frame is defined the input data, available options:

     - 'crystal' if the data was interpolated into the crystal frame using
       xrayutilities or (transformation matrix + align_q=True)
     - 'laboratory' if the data was interpolated into the laboratory frame
       using the transformation matrix (align_q: False)
     - 'detector' if the data is still in the detector frame

    :param ref_axis_q: e.g. "y"
     axis along which q will be aligned (data_frame= 'detector' or
     'laboratory') or is already aligned (data_frame='crystal')
    :param save_frame: e.g. "laboratory"
     in which frame should be saved the data, available options:

     - 'crystal' to save the data with q aligned along ref_axis_q
     - 'laboratory' to save the data in the laboratory frame (experimental
       geometry)
     - 'lab_flat_sample' to save the data in the laboratory frame, with
       all sample angles rotated back to 0. The rotations for 'laboratory'
       and 'lab_flat_sample' are realized after the strain calculation
       (which is always done in the crystal frame along ref_axis_q)

    :param isosurface_strain: e.g. 0.2
     threshold use for removing the outer layer (the strain is undefined
     at the exact surface voxel)
    :param skip_unwrap: e.g. False
     If 'skip_unwrap', it will not unwrap the phase. It can be used when there is a
     defect and phase unwrapping does not work well.
    :param strain_method: e.g. "default"
     how to calculate the strain, available options:

     - 'default': use the single value calculated from the gradient of
       the phase
     - 'defect': it will offset the phase in a loop and keep the smallest
       magnitude value for the strain.
       See: F. Hofmann et al. PhysRevMaterials 4, 013801 (2020)

    Parameters related to the refraction correction:

    :param correct_refraction: e.g. True
     True for correcting the phase shift due to refraction
    :param optical_path_method: e.g. "threshold"
     'threshold' or 'defect', if 'threshold' it uses isosurface_strain to
     define the support  for the optical path calculation, if 'defect'
     (holes) it tries to remove only outer layers even if the amplitude is
     lower than isosurface_strain inside the crystal
    :param dispersion: e.g. 5.0328e-05
     delta value used for refraction correction, for Pt:
     3.0761E-05 @ 10300eV, 5.0328E-05 @ 8170eV, 3.2880E-05 @ 9994eV,
     4.1184E-05 @ 8994eV, 5.2647E-05 @ 7994eV, 4.6353E-05 @ 8500eV
     Ge 1.4718E-05 @ 8keV
    :param absorption: e.g. 4.1969e-06
     beta value, for Pt:  2.0982E-06 @ 10300eV, 4.8341E-06 @ 8170eV,
     2.3486E-06 @ 9994eV, 3.4298E-06 @ 8994eV, 5.2245E-06 @ 7994eV,
     4.1969E-06 @ 8500eV
    :param threshold_unwrap_refraction: e.g. 0.05
     threshold used to calculate the optical path. The threshold for
     refraction correction should be low, to correct for an object larger
     than the real one, otherwise it messes up the phase

    Parameters related to the phase:

    :param simulation: e.g. False
     True if it is a simulation, the parameter invert_phase will be set
     to 0 (see below)
    :param invert_phase: e.g. True
    True for the displacement to have the right sign (FFT convention),
    it is False only for simulations
    :param flip_reconstruction: e.g. True
     True if you want to get the conjugate object
    :param phase_ramp_removal: e.g. "gradient"
     'gradient' or 'upsampling', 'gradient' is much faster
    :param threshold_gradient: e.g. 1.0
     upper threshold of the gradient of the phase, use for ramp removal
    :param phase_offset: e.g. 0
     manual offset to add to the phase, should be 0 in most cases
    :param phase_offset_origin: e.g. [12, 32, 65]
     the phase at this voxel will be set to phase_offset, leave None to
     use the default position computed using offset_method (see below)
    :param offset_method: e.g. "mean"
     'com' (center of mass) or 'mean', method for determining the phase
     offset origin

    Parameters related to data visualization:

    :param debug: e.g. False
     True to show all plots for debugging
    :param align_axis: e.g. False
     True to rotate the crystal to align axis_to_align along ref_axis for
     visualization. This is done after the calculation of the strain and
     has no effect on it.
    :param ref_axis: e.g. "y"
     it will align axis_to_align to that axis if align_axis is True
    :param axis_to_align: e.g. [-0.01166, 0.9573, -0.2887]
     axis to align with ref_axis in the order x y z (axis 2, axis 1,
     axis 0)
    :param strain_range: e.g. 0.001
     range of the colorbar for strain plots
    :param phase_range: e.g. 0.4
     range of the colorbar for phase plots
    :param grey_background: e.g. True
     True to set the background to grey in phase and strain plots
    :param tick_spacing: e.g. 50
     spacing between axis ticks in plots, in nm
    :param tick_direction: e.g. "inout"
     direction of the ticks in plots: 'out', 'in', 'inout'
    :param tick_length: e.g. 3
     length of the ticks in plots
    :param tick_width: e.g. 1
     width of the ticks in plots

    Parameters for averaging several reconstructed objects:

    :param averaging_space: e.g. "reciprocal_space"
     in which space to average, 'direct_space' or 'reciprocal_space'
    :param threshold_avg: e.g. 0.90
     minimum correlation between arrays for averaging

    Parameters for phase averaging or apodization:

    :param half_width_avg_phase: e.g. 0
     (width-1)/2 of the averaging window for the phase, 0 means no phase
     averaging
    :param apodize: e.g. False
     True to multiply the diffraction pattern by a filtering window
    :param apodization_window: e.g. "blackman"
     filtering window, multivariate 'normal' or 'tukey' or 'blackman'
    :param apodization_mu: e.g. [0.0, 0.0, 0.0]
     mu of the gaussian window
    :param apodization_sigma: e.g. [0.30, 0.30, 0.30]
     sigma of the gaussian window
    :param apodization_alpha: e.g. [1.0, 1.0, 1.0]
     shape parameter of the tukey window

    Parameters related to saving:

    :param save_rawdata: e.g. False
     True to save the amp-phase.vti before orthogonalization
    :param save_support: e.g. False
     True to save the non-orthogonal support for later phase retrieval
    :param save: e.g. True
     True to save amp.npz, phase.npz, strain.npz and vtk files
    """
    # Save parameter values
    # parameters used when averaging several reconstruction #
    self.Dataset.sort_method = sort_method
    self.Dataset.correlation_threshold = correlation_threshold
    # parameters relative to the FFT window and voxel sizes #
    self.Dataset.phasing_binning = phasing_binning
    self.Dataset.original_size = original_size
    self.Dataset.preprocessing_binning = preprocessing_binning
    self.Dataset.output_size = output_size
    self.Dataset.keep_size = keep_size
    self.Dataset.fix_voxel = fix_voxel
    # parameters related to displacement and strain calculation #
    self.Dataset.data_frame = data_frame
    self.Dataset.save_frame = save_frame
    self.Dataset.ref_axis_q = ref_axis_q
    self.Dataset.isosurface_strain = isosurface_strain
    self.Dataset.skip_unwrap = skip_unwrap
    self.Dataset.strain_method = strain_method
    self.Dataset.phase_offset = phase_offset
    self.Dataset.phase_offset_origin = phase_offset_origin
    self.Dataset.offset_method = offset_method
    self.Dataset.centering_method = centering_method
    # parameters related to the refraction correction
    self.Dataset.correct_refraction = correct_refraction
    self.Dataset.optical_path_method = optical_path_method
    self.Dataset.dispersion = dispersion
    self.Dataset.absorption = absorption
    self.Dataset.threshold_unwrap_refraction\
        = threshold_unwrap_refraction
    # options #
    self.Dataset.simulation = simulation
    self.Dataset.invert_phase = invert_phase
    self.Dataset.flip_reconstruction = flip_reconstruction
    self.Dataset.phase_ramp_removal = phase_ramp_removal
    self.Dataset.threshold_gradient = threshold_gradient
    self.Dataset.save_raw = save_raw
    self.Dataset.save_support = save_support
    self.Dataset.save = save
    self.Dataset.debug = debug
    self.Dataset.roll_modes = roll_modes
    # parameters related to data visualization #
    self.Dataset.align_axis = align_axis
    self.Dataset.ref_axis = ref_axis
    self.Dataset.axis_to_align = axis_to_align
    self.Dataset.strain_range = strain_range
    self.Dataset.phase_range = phase_range
    self.Dataset.grey_background = grey_background
    self.Dataset.tick_spacing = tick_spacing
    self.Dataset.tick_direction = tick_direction
    self.Dataset.tick_length = tick_length
    self.Dataset.tick_width = tick_width
    # parameters for averaging several reconstructed objects #
    self.Dataset.averaging_space = averaging_space
    self.Dataset.threshold_avg = threshold_avg
    # setup for phase averaging or apodization
    self.Dataset.half_width_avg_phase = half_width_avg_phase
    self.Dataset.apodize = apodize
    self.Dataset.apodization_window = apodization_window
    self.Dataset.apodization_mu = apodization_mu
    self.Dataset.apodization_sigma = apodization_sigma
    self.Dataset.apodization_alpha = apodization_alpha
    self.reconstruction_files = strain_folder + reconstruction_files

    if run_strain:
        # Save directory
        save_dir = f"{self.postprocessing_folder}/result_{self.Dataset.save_frame}/"

        # Disable all widgets until the end of the program
        for w in self._list_widgets_strain.children[:-1]:
            w.disabled = True

        for w in self._list_widgets_preprocessing.children[:-2]:
            w.disabled = True

        # Extract dict, list and tuple from strings
        list_parameters = [
            "original_size", "output_size", "axis_to_align",
            "apodization_mu", "apodization_sigma", "apodization_alpha"]

        tuple_parameters = [
            "phasing_binning", "preprocessing_binning",
            "phase_offset_origin", "roll_modes"]

        try:
            for p in list_parameters:
                if getattr(self.Dataset, p) == "":
                    setattr(self.Dataset, p, [])
                else:
                    setattr(self.Dataset, p, literal_eval(
                        getattr(self.Dataset, p)))
        except ValueError:
            gutil.hash_print(f"Wrong list syntax for {p}")

        try:
            for p in tuple_parameters:
                if getattr(self.Dataset, p) == "":
                    setattr(self.Dataset, p, ())
                else:
                    setattr(self.Dataset, p, literal_eval(
                        getattr(self.Dataset, p)))
        except ValueError:
            gutil.hash_print(f"Wrong tuple syntax for {p}")

        # Empty parameters are set to None (bcdi syntax)
        if self.Dataset.output_size == []:
            self.Dataset.output_size = None

        if self.Dataset.fix_voxel == 0:
            self.Dataset.fix_voxel = None

        if self.Dataset.phase_offset_origin == ():
            self.Dataset.phase_offset_origin = (None)

        # Check beamline for save folder
        try:
            # Change data_dir and root folder depending on beamline
            if self.Dataset.beamline == "SIXS_2019":
                root_folder = self.Dataset.root_folder
                data_dir = self.Dataset.data_dir

            elif self.Dataset.beamline == "P10":
                root_folder = self.Dataset.data_dir
                data_dir = None

            else:
                root_folder = self.Dataset.root_folder
                data_dir = self.Dataset.data_dir

        except AttributeError:
            for w in self._list_widgets_strain.children[:-1]:
                w.disabled = False

            for w in self._list_widgets_preprocessing.children[:-2]:
                w.disabled = False

            print("You need to initialize all the parameters with the \
                preprocess tab first, some parameters are used here such \
                as the energy, detector distance, ...""")
            return

        try:
            gutil.create_yaml_file(
                fname=f"{self.postprocessing_folder}/config_postprocessing.yml",
                scans=self.Dataset.scan,
                root_folder=root_folder,
                save_dir=save_dir,
                data_dir=data_dir,
                sample_name=self.Dataset.sample_name,
                comment=self.Dataset.comment,
                reconstruction_files=self.reconstruction_files,
                backend=self.matplotlib_backend,
                # parameters used when averaging several reconstruction #
                sort_method=self.Dataset.sort_method,
                averaging_space=self.Dataset.averaging_space,
                correlation_threshold=self.Dataset.correlation_threshold,
                # parameters related to centering #
                centering_method=self.Dataset.centering_method,
                roll_modes=self.Dataset.roll_modes,
                # parameters relative to the FFT window and voxel sizes #
                original_size=self.Dataset.original_size,
                phasing_binning=self.Dataset.phasing_binning,
                preprocessing_binning=self.Dataset.preprocessing_binning,
                output_size=self.Dataset.output_size,
                keep_size=self.Dataset.keep_size,
                fix_voxel=self.Dataset.fix_voxel,
                # parameters related to the strain calculation #
                data_frame=self.Dataset.data_frame,
                ref_axis_q=self.Dataset.ref_axis_q,
                save_frame=self.Dataset.save_frame,
                isosurface_strain=self.Dataset.isosurface_strain,
                skip_unwrap=self.Dataset.skip_unwrap,
                strain_method=self.Dataset.strain_method,
                # define beamline related parameters #
                beamline=self.Dataset.beamline,
                is_series=self.Dataset.is_series,
                actuators=self.Dataset.actuators,
                # setup for custom scans #
                custom_scan=self.Dataset.custom_scan,
                custom_images=self.Dataset.custom_images,
                custom_monitor=self.Dataset.custom_monitor,
                rocking_angle=self.Dataset.rocking_angle,
                detector_distance=self.Dataset.detector_distance,
                energy=self.Dataset.energy,
                beam_direction=self.Dataset.beam_direction,
                sample_offsets=self.Dataset.sample_offsets,
                tilt_angle=self.Dataset.tilt_angle,
                direct_beam=self.Dataset.direct_beam,
                dirbeam_detector_angles=self.Dataset.dirbeam_detector_angles,
                bragg_peak=self.Dataset.bragg_peak,
                outofplane_angle=self.Dataset.outofplane_angle,
                inplane_angle=self.Dataset.inplane_angle,
                specfile_name=self.Dataset.specfile_name,
                # detector related parameters #
                detector=self.Dataset.detector,
                roi_detector=self.Dataset.roi_detector,
                template_imagefile=self.Dataset.template_imagefile,
                # parameters related to the refraction correction #
                correct_refraction=self.Dataset.correct_refraction,
                optical_path_method=self.Dataset.optical_path_method,
                dispersion=self.Dataset.dispersion,
                absorption=self.Dataset.absorption,
                threshold_unwrap_refraction=self.Dataset.threshold_unwrap_refraction,
                # parameters related to the phase #
                simulation=self.Dataset.simulation,
                invert_phase=self.Dataset.invert_phase,
                flip_reconstruction=self.Dataset.flip_reconstruction,
                phase_ramp_removal=self.Dataset.phase_ramp_removal,
                threshold_gradient=self.Dataset.threshold_gradient,
                phase_offset=self.Dataset.phase_offset,
                phase_offset_origin=self.Dataset.phase_offset_origin,
                offset_method=self.Dataset.offset_method,
                # parameters related to data visualization #
                debug=self.Dataset.debug,
                align_axis=self.Dataset.align_axis,
                ref_axis=self.Dataset.ref_axis,
                axis_to_align=self.Dataset.axis_to_align,
                strain_range=self.Dataset.strain_range,
                phase_range=self.Dataset.phase_range,
                grey_background=self.Dataset.grey_background,
                tick_spacing=self.Dataset.tick_spacing,
                tick_direction=self.Dataset.tick_direction,
                tick_length=self.Dataset.tick_length,
                tick_width=self.Dataset.tick_width,
                # parameters for temperature estimation #
                # get_temperature=self.Dataset.get_temperature,
                # reflection=self.Dataset.reflection,
                # reference_spacing=self.Dataset.reference_spacing,
                # reference_temperature=self.Dataset.reference_temperature,
                # parameters for phase averaging or apodization #
                half_width_avg_phase=self.Dataset.half_width_avg_phase,
                apodize=self.Dataset.apodize,
                apodization_window=self.Dataset.apodization_window,
                apodization_mu=self.Dataset.apodization_mu,
                apodization_sigma=self.Dataset.apodization_sigma,
                apodization_alpha=self.Dataset.apodization_alpha,
                # parameters related to saving #
                save_rawdata=self.Dataset.save_rawdata,
                save_support=self.Dataset.save_support,
                save=self.Dataset.save,
            )
            # Run bcdi_postprocessing
            print(
                "\n###########################################"
                "#############################################"
            )
            print(f"Running: $ {self.path_scripts}/bcdi_strain.py")
            print(
                f"Config file: {self.postprocessing_folder}/config_postprocessing.yml")
            print(
                "\n###########################################"
                "#############################################"
            )

            # Load the config file
            config_file = self.postprocessing_folder + "/config_postprocessing.yml"
            parser = ConfigParser(config_file)
            args = parser.load_arguments()
            args["time"] = f"{datetime.now()}"

            # Run function
            run_postprocessing(prm=args)
            gutil.hash_print("End of script")

            # Get data from saved file
            phase_fieldname = "disp" if self.Dataset.invert_phase else "phase"

            files = sorted(
                glob.glob(
                    f"{self.postprocessing_folder}/**/"
                    f"S{self.Dataset.scan}_amp{phase_fieldname}"
                    f"strain*{self.Dataset.comment}.h5",
                    recursive=True),
                key=os.path.getmtime)
            self.strain_output_file = files[0]

            creation_time = datetime.fromtimestamp(
                os.path.getmtime(self.strain_output_file)
            ).strftime('%Y-%m-%d %H:%M:%S')

            print(
                "\n###########################################"
                "#############################################"
                f"\nResult file used to extract results saved in the .cxi file:"
                f"\n{self.strain_output_file}"
                f"\n\tCreated: {creation_time}"
                "\nMake sure it is the latest one!!"
                "\n###########################################"
                "#############################################"
            )

            print(
                "\n###########################################"
                "#############################################"
                "\nRemember to save your progress as a cxi file !"
                "\n###########################################"
                "#############################################"
            )

        except KeyboardInterrupt:
            gutil.hash_print("Strain analysis stopped by user ...")

        finally:
            # At the end of the function
            self._list_widgets_strain.children[-2].disabled = False

            # Refresh folders
            self.sub_directories_handler(change=self.Dataset.scan_folder)

            # PyNX folder, refresh values
            self._list_widgets_phase_retrieval.children[1].value\
                = self.preprocessing_folder
            self.pynx_folder_handler(change=self.preprocessing_folder)

            self.tab_data.children[1].value = self.preprocessing_folder
            self.plot_folder_handler(
                change=self.preprocessing_folder)

    if not run_strain:
        plt.close()
        for w in self._list_widgets_strain.children[:-1]:
            w.disabled = False

        for w in self._list_widgets_preprocessing.children[:-2]:
            w.disabled = False

        # Refresh folders
        self.sub_directories_handler(change=self.Dataset.scan_folder)

        # PyNX folder, refresh values
        self._list_widgets_phase_retrieval.children[1].value\
            = self.preprocessing_folder
        self.pynx_folder_handler(change=self.preprocessing_folder)

        # Plot folder, refresh values
        self.tab_data.children[1].value = self.preprocessing_folder
        self.plot_folder_handler(change=self.preprocessing_folder)

        gutil.hash_print("Cleared window.")
        clear_output(True)
