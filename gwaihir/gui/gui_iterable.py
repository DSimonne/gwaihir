#!/usr/bin/env python
# coding: utf-8

# In[ ]:

"""Regroups all the possible classes that can be used used as iterable in the
Interface class For now only Dataset.
"""

import numpy as np
import pandas as pd
import h5py
import shutil

from IPython.display import display
from datetime import datetime


class Dataset():
    """THE DATASETS CLASS IS MEANT TO BE READ VIA THE gwaihir.gui CLASS !!"""

    def __init__(self, scan, sample_name, data_dir, root_folder):
        """Initialiaze the Dataset class, some metadata can be associated as
        well.
        """
        self.scan = scan
        self.sample_name = sample_name
        self.data_dir = data_dir
        self.root_folder = root_folder

    def __repr__(self):
        return "Dataset {}{}.\n".format(
            self.sample_name,
            self.scan,
        )

    def __str__(self):
        return repr(self)

    def to_cxi(self, cxi_filename, reconstruction_filename=False):
        """
        Save all the parameters used in the data analysis with a specific
        architecture based on NeXuS.
        """
        final_data_path = f"{self.scan_folder}{self.sample_name}{self.scan}.cxi"
        shutil.copy(cxi_filename,
                    final_data_path,
                    )

        if isinstance(reconstruction_filename, str):
            with h5py.File(reconstruction_filename, "r") as reconstruction_file, \
                    h5py.File(final_data_path, "a") as final_file:
                # Real space data is already here from PyNX

                print("\nSaving phase retrieval output ...")

                # Copy image_1 from reconstruction to entry_1.image_2
                try:
                    reconstruction_file.copy(
                        '/entry_1/image_1/', final_file["entry_1"],
                        name="image_2")
                except RuntimeError:
                    del final_file["entry_1"]["image_2"]
                    reconstruction_file.copy(
                        '/entry_1/image_1/', final_file["entry_1"],
                        name="image_2")

                # Update params if reconstruction file results
                # from mode decomposition
                if reconstruction_filename.endswith(".h5"):
                    final_file["entry_1"]["image_2"].create_dataset(
                        "data_space", data="real")
                    final_file["entry_1"]["image_2"].create_dataset(
                        "data_type", data="electron density")

                # Update entry_1.image_2.support softlink
                if reconstruction_filename.endswith(".cxi"):
                    del final_file["entry_1"]["image_2"]["support"]
                    final_file["entry_1"]["image_2"]["support"] = h5py.SoftLink(
                        "/entry_1/image_2/mask")

                # Create entry_1.data_2 and create softlink to
                # entry_1.image_2.data
                try:
                    group = final_file["entry_1"].create_group(
                        "data_2")
                    group["data"] = h5py.SoftLink(
                        "/entry_1/image_2/data")
                except ValueError:
                    group = final_file["entry_1"]["data_2"]
                    del final_file["entry_1"]["data_2"]["data"]
                    group["data"] = h5py.SoftLink(
                        "/entry_1/image_2/data")
                # Assign correct type to entry_1.data_2
                final_file["entry_1"]["data_2"].attrs['NX_class'] = 'NXdata'
                final_file["entry_1"]["data_2"].attrs['signal'] = 'data'

                # Rename entry_1.image_2.process_1 to entry_1.image_2.process_2
                try:
                    final_file["entry_1"]["image_2"]["process_1"].move(
                        "/entry_1/image_2/process_1/",
                        "/entry_1/image_2/process_2/")
                except ValueError:
                    del final_file["entry_1"]["image_2"]["process_2"]
                    final_file["entry_1"]["image_2"]["process_1"].move(
                        "/entry_1/image_2/process_1/",
                        "/entry_1/image_2/process_2/")
                # Assign correct type to entry_1.image_2.process_2
                final_file["entry_1"]["image_2"]["process_2"].attrs['NX_class'] = 'NXprocess'

                # Move PyNX configuration
                try:
                    conf = final_file["entry_1"]["data_1"]["process_1"]["configuration"]
                    if reconstruction_filename.endswith(".h5"):
                        final_file.create_group(
                            "entry_1/image_2/process_2/configuration/")
                    for k in conf.keys():
                        final_file.move(
                            f"entry_1/data_1/process_1/configuration/{k}",
                            f"entry_1/image_2/process_2/configuration/{k}"
                        )

                    del final_file["entry_1"]["data_1"]["process_1"]
                except KeyError:
                    # Already moved or does not exist
                    pass
                final_file.move("entry_1/program_name",
                                "entry_1/image_2/process_2/program_name")

                # Delete entry_1.instrument_1 if exists
                try:
                    del final_file["entry_1"]["image_2"]["instrument_1"]
                except KeyError:
                    pass

                # Also copy mode data to entry_1.image_2.modes_percentage
                if reconstruction_filename.endswith(".h5"):
                    try:
                        reconstruction_file.copy(
                            '/entry_1/data_2/',
                            final_file["entry_1"]["image_2"],
                            name="modes_percentage")
                    except RuntimeError:
                        del final_file["entry_1"]["image_2"]["modes_percentage"]
                        reconstruction_file.copy(
                            '/entry_1/data_2/',
                            final_file["entry_1"]["image_2"],
                            name="modes_percentage")

        # Add GUI data
        with h5py.File(final_data_path, "a") as f:
            # Parameters
            try:
                data_3 = f.create_group("entry_1/data_3/")
                f["entry_1"]["data_3"].attrs['NX_class'] = 'NXdata'

            except ValueError:
                del f["entry_1"]["data_3"]
                data_3 = f.create_group("entry_1/data_3/")
                f["entry_1"]["data_3"].attrs['NX_class'] = 'NXdata'

            try:
                image_3 = f.create_group("entry_1/image_3")
                f["entry_1"]["image_3"].attrs['NX_class'] = 'NXdata'

            except ValueError:
                del f["entry_1"]["image_3"]
                image_3 = f.create_group("entry_1/image_3")
                f["entry_1"]["image_3"].attrs['NX_class'] = 'NXdata'

            parameters = f.create_group("entry_1/image_3/process_3/")
            parameters.attrs['NX_class'] = 'NXprocess'

            parameters.create_dataset(
                "ObjectName", data=f"Dataset_{self.sample_name}{self.scan}")

            # Preprocessing
            preprocessing = parameters.create_group("preprocessing")
            print(
                "\n###############################################################################\n")
            print("Saving parameters used in preprocessing ...")

            # Masking
            masking = preprocessing.create_group("masking")
            try:
                masking.create_dataset(
                    "flag_interact", data=self.flag_interact)
                masking.create_dataset(
                    "background_plot", data=self.background_plot)

            except AttributeError:
                print("Could not save masking parameters")

            # Cropping padding centering
            cropping_padding_centering = preprocessing.create_group(
                "cropping_padding_centering")
            try:
                cropping_padding_centering.create_dataset(
                    "centering_method", data=self.centering_method)
                cropping_padding_centering.create_dataset(
                    "fix_size", data=self.fix_size)
                cropping_padding_centering.create_dataset(
                    "center_fft", data=self.center_fft)
                cropping_padding_centering.create_dataset(
                    "pad_size", data=self.pad_size)
            except AttributeError:
                print("Could not save cropping padding centering parameters")

            # Intensity normalization
            intensity_normalization = preprocessing.create_group(
                "intensity_normalization")
            try:
                intensity_normalization.create_dataset(
                    "normalize_flux", data=self.normalize_flux)
            except AttributeError:
                print("Could not save intensity normalization parameters")

            # Data filtering
            data_filtering = preprocessing.create_group("data_filtering")
            try:
                data_filtering.create_dataset(
                    "mask_zero_event", data=self.mask_zero_event)
                data_filtering.create_dataset(
                    "median_filter", data=self.median_filter)
                data_filtering.create_dataset(
                    "median_filter_order", data=self.median_filter_order)
            except AttributeError:
                print("Could not save data filtering parameters")

            # Saving options
            saving_options = preprocessing.create_group("saving_options")
            try:
                saving_options.create_dataset(
                    "save_rawdata", data=self.save_rawdata)
                saving_options.create_dataset(
                    "save_to_npz", data=self.save_to_npz)
                saving_options.create_dataset(
                    "save_to_mat", data=self.save_to_mat)
                saving_options.create_dataset(
                    "save_to_vti", data=self.save_to_vti)
                saving_options.create_dataset(
                    "save_as_int", data=self.save_as_int)
            except AttributeError:
                print("Could not save saving options")

            # Reloading options
            reload_options = preprocessing.create_group("reload_options")
            try:
                reload_options.create_dataset(
                    "reload_previous", data=self.reload_previous)
                reload_options.create_dataset(
                    "reload_orthogonal", data=self.reload_orthogonal)
                reload_options.create_dataset(
                    "preprocessing_binning", data=self.preprocessing_binning)
            except AttributeError:
                print("Could not save reloading options")

            # Beamline
            beamline = preprocessing.create_group("beamline")
            try:
                beamline.create_dataset("beamline", data=self.beamline)
                beamline.create_dataset("actuators", data=str(self.actuators))
                beamline.create_dataset("is_series", data=self.is_series)
                beamline.create_dataset(
                    "specfile_name", data=str(self.specfile_name))
                beamline.create_dataset(
                    "rocking_angle", data=self.rocking_angle)
            except (AttributeError, TypeError):
                print("Could not save beamline parameters")

            try:
                beamline.create_dataset("custom_scan", data=self.custom_scan)
                beamline.create_dataset(
                    "custom_images", data=self.custom_images)
                beamline.create_dataset(
                    "custom_monitor", data=self.custom_monitor)
                beamline.create_dataset(
                    "custom_monitor", data=self.custom_motors)
            except (AttributeError, TypeError):
                print("Could not save custom parameters")

            # Detector
            detector = preprocessing.create_group("detector")
            try:
                detector.create_dataset("detector", data=self.detector)
                detector.create_dataset(
                    "photon_threshold", data=self.photon_threshold)
                detector.create_dataset(
                    "photon_filter", data=self.photon_filter)
                detector.create_dataset(
                    "background_file", data=str(self.background_file))
                detector.create_dataset(
                    "hotpixels_file", data=str(self.hotpixels_file))
                detector.create_dataset(
                    "flatfield_file", data=str(self.flatfield_file))
                detector.create_dataset(
                    "template_imagefile", data=str(self.template_imagefile))
            except AttributeError:
                print("Could not save detector parameters")

            try:
                detector.create_dataset("nb_pixel_x", data=self.nb_pixel_x)
                detector.create_dataset("nb_pixel_y", data=self.nb_pixel_y)
            except (TypeError, AttributeError):
                pass

            # Temperature estimation, not really used
            # temperature_estimation = parameters.create_group(
            #     "temperature_estimation")
            # try:
            #     temperature_estimation.create_dataset(
            #         "reflection", data=self.reflection)
            #     temperature_estimation.create_dataset(
            #         "reference_spacing", data=self.reference_spacing)
            #     temperature_estimation.create_dataset(
            #         "reference_temperature", data=self.reference_temperature)
            #     temperature_estimation["reference_temperature"].attrs['units'] = 'Celsius'

            # except AttributeError:
            #     print("Could not save temperature_estimation parameters")

            # try:
            #     temperature_estimation.create_dataset(
            #         "estimated_temperature", data=self.temperature)
            # except AttributeError:
            #     print("No estimated temperature")

            # Angles correction
            angles_corrections = parameters.create_group("angles_corrections")
            try:
                angles_corrections.create_dataset(
                    "tilt_values", data=self.tilt_values)
                angles_corrections.create_dataset(
                    "rocking_curve", data=self.rocking_curve)
                angles_corrections.create_dataset(
                    "interp_tilt", data=self.interp_tilt)
                angles_corrections.create_dataset(
                    "interp_curve", data=self.interp_curve)
                angles_corrections.create_dataset(
                    "COM_rocking_curve", data=self.COM_rocking_curve)
                angles_corrections.create_dataset(
                    "detector_data_COM", data=self.detector_data_COM)
                angles_corrections.create_dataset(
                    "interp_fwhm", data=self.interp_fwhm)
                angles_corrections.create_dataset("q", data=self.q)
                angles_corrections.create_dataset("qnorm", data=self.qnorm)
                angles_corrections.create_dataset(
                    "dist_plane", data=self.dist_plane)
                angles_corrections.create_dataset(
                    "bragg_inplane", data=self.bragg_inplane)
                angles_corrections.create_dataset(
                    "bragg_outofplane", data=self.bragg_outofplane)
            except AttributeError:
                print("Could not save setup parameters")

            # Orthogonalisation
            print(
                "\n###############################################################################\n")
            print("Saving orthogonalisation parameters ...")
            orthogonalisation = parameters.create_group("orthogonalisation")

            # Linearized transformation matrix
            linearized_transformation_matrix = orthogonalisation.create_group(
                "linearized_transformation_matrix")
            try:
                linearized_transformation_matrix.create_dataset(
                    "use_rawdata", data=self.use_rawdata)
                linearized_transformation_matrix.create_dataset(
                    "interpolation_method", data=self.interpolation_method)
                linearized_transformation_matrix.create_dataset(
                    "fill_value_mask", data=self.fill_value_mask)
                linearized_transformation_matrix.create_dataset(
                    "beam_direction", data=self.beam_direction)
                linearized_transformation_matrix.create_dataset(
                    "sample_offsets", data=self.sample_offsets)
                linearized_transformation_matrix.create_dataset(
                    "sdd", data=self.sdd)
                linearized_transformation_matrix["sdd"].attrs['units'] = 'm'
                linearized_transformation_matrix.create_dataset(
                    "energy", data=self.energy)
                linearized_transformation_matrix["energy"].attrs['units'] = 'keV'
                linearized_transformation_matrix.create_dataset(
                    "custom_motors", data=str(self.custom_motors))
            except AttributeError:
                print("Could not save linearized transformation matrix parameters")

            # xrayutilities
            xrayutilities = orthogonalisation.create_group("xrayutilities")
            try:
                xrayutilities.create_dataset("align_q", data=self.align_q)
                xrayutilities.create_dataset(
                    "ref_axis_q", data=self.ref_axis_q)
                xrayutilities.create_dataset(
                    "outofplane_angle", data=self.outofplane_angle)
                xrayutilities["outofplane_angle"].attrs['units'] = 'degrees'
                xrayutilities.create_dataset(
                    "inplane_angle", data=self.inplane_angle)
                xrayutilities["inplane_angle"].attrs['units'] = 'degrees'
                xrayutilities.create_dataset(
                    "tilt_angle", data=self.tilt_angle)
                xrayutilities["tilt_angle"].attrs['units'] = 'degrees'
                xrayutilities.create_dataset(
                    "sample_inplane", data=self.sample_inplane)
                xrayutilities.create_dataset(
                    "sample_outofplane", data=self.sample_outofplane)
                xrayutilities.create_dataset(
                    "offset_inplane", data=self.offset_inplane)
                xrayutilities.create_dataset("cch1", data=self.cch1)
                xrayutilities.create_dataset("cch2", data=self.cch2)
                xrayutilities.create_dataset(
                    "dirbeam_detector_angles", data=self.dirbeam_detector_angles)
                xrayutilities.create_dataset(
                    "direct_beam", data=self.direct_beam)
                xrayutilities.create_dataset("detrot", data=self.detrot)
                xrayutilities.create_dataset(
                    "tiltazimuth", data=self.tiltazimuth)
                xrayutilities.create_dataset(
                    "tilt_detector", data=self.tilt_detector)
            except AttributeError:
                print("Could not save xrayutilities parameters")

            # Tranformation matrix
            try:
                orthogonalisation.create_dataset(
                    "transfer_matrix", data=self.transfer_matrix)
            except AttributeError:
                print("Could not save transfer_matrix")

            # Postprocessing
            postprocessing = parameters.create_group("postprocessing")
            print(
                "\n###############################################################################\n")
            print("Saving parameters used in postprocessing ...")

            # Averaging reconstructions
            averaging_reconstructions = postprocessing.create_group(
                "averaging_reconstructions")
            try:
                averaging_reconstructions.create_dataset(
                    "sort_method", data=self.sort_method)
                averaging_reconstructions.create_dataset(
                    "correlation_threshold", data=self.correlation_threshold)
            except AttributeError:
                print("Could not save averaging reconstructions parameters")

            # FFT_window_voxel
            FFT_window_voxel = postprocessing.create_group("FFT_window_voxel")
            try:
                FFT_window_voxel.create_dataset(
                    "phasing_binning", data=self.phasing_binning)
                FFT_window_voxel.create_dataset(
                    "original_size", data=self.original_size)
                FFT_window_voxel.create_dataset(
                    "preprocessing_binning", data=self.preprocessing_binning)
                FFT_window_voxel.create_dataset(
                    "output_size", data=str(self.output_size))
                FFT_window_voxel.create_dataset(
                    "keep_size", data=self.keep_size)
                FFT_window_voxel.create_dataset(
                    "fix_voxel", data=str(self.fix_voxel))
            except AttributeError:
                print("Could not save angles_corrections parameters")

            # Displacement strain calculation
            displacement_strain_calculation = postprocessing.create_group(
                "displacement_strain_calculation")
            try:
                displacement_strain_calculation.create_dataset(
                    "data_frame", data=self.data_frame)
                displacement_strain_calculation.create_dataset(
                    "save_frame", data=self.save_frame)
                displacement_strain_calculation.create_dataset(
                    "ref_axis_q", data=self.ref_axis_q)
                displacement_strain_calculation.create_dataset(
                    "isosurface_strain", data=self.isosurface_strain)
                displacement_strain_calculation.create_dataset(
                    "strain_method", data=self.strain_method)
                displacement_strain_calculation.create_dataset(
                    "phase_offset", data=self.phase_offset)
                displacement_strain_calculation.create_dataset(
                    "phase_offset_origin", data=str(self.phase_offset_origin))
                displacement_strain_calculation.create_dataset(
                    "offset_method", data=self.offset_method)
                displacement_strain_calculation.create_dataset(
                    "centering_method", data=self.centering_method)
            except AttributeError:
                print("Could not save displacement & strain calculation parameters")

            # Refraction
            refraction = postprocessing.create_group("refraction")
            try:
                refraction.create_dataset(
                    "correct_refraction", data=self.correct_refraction)
                refraction.create_dataset(
                    "optical_path_method", data=self.optical_path_method)
                refraction.create_dataset("dispersion", data=self.dispersion)
                refraction.create_dataset("absorption", data=self.absorption)
                refraction.create_dataset(
                    "threshold_unwrap_refraction",
                    data=self.threshold_unwrap_refraction)
            except AttributeError:
                print("Could not save refraction parameters")

            # Options
            options = postprocessing.create_group("options")
            try:
                options.create_dataset("simulation", data=self.simulation)
                options.create_dataset("invert_phase", data=self.invert_phase)
                options.create_dataset(
                    "flip_reconstruction", data=self.flip_reconstruction)
                options.create_dataset(
                    "phase_ramp_removal", data=self.phase_ramp_removal)
                options.create_dataset(
                    "threshold_gradient", data=self.threshold_gradient)
                options.create_dataset("save_raw", data=self.save_raw)
                options.create_dataset("save_support", data=self.save_support)
                options.create_dataset("save", data=self.save)
                options.create_dataset("debug", data=self.debug)
                options.create_dataset("roll_modes", data=self.roll_modes)
            except AttributeError:
                print("Could not save postprocessing options")

            # Data visualisation
            data_visualisation = postprocessing.create_group(
                "data_visualisation")
            try:
                data_visualisation.create_dataset(
                    "align_axis", data=self.align_axis)
                data_visualisation.create_dataset(
                    "ref_axis", data=self.ref_axis)
                data_visualisation.create_dataset(
                    "axis_to_align", data=self.axis_to_align)
                data_visualisation.create_dataset(
                    "strain_range", data=self.strain_range)
                data_visualisation.create_dataset(
                    "phase_range", data=self.phase_range)
                data_visualisation.create_dataset(
                    "grey_background", data=self.grey_background)
                data_visualisation.create_dataset(
                    "tick_spacing", data=self.tick_spacing)
                data_visualisation.create_dataset(
                    "tick_direction", data=self.tick_direction)
                data_visualisation.create_dataset(
                    "tick_length", data=self.tick_length)
                data_visualisation.create_dataset(
                    "tick_width", data=self.tick_width)
            except AttributeError:
                print("Could not save data visualisation parameters")

            # Averaging reconstructed objects
            averaging_reconstructed_objects = postprocessing.create_group(
                "averaging_reconstructed_objects")
            try:
                averaging_reconstructed_objects.create_dataset(
                    "averaging_space", data=self.averaging_space)
            except AttributeError:
                print("Could not save averaging reconstructed objects parameters")

            # Phase averaging apodization
            phase_averaging_apodization = postprocessing.create_group(
                "phase_averaging_apodization")
            try:
                phase_averaging_apodization.create_dataset(
                    "apodize", data=self.apodize)
                phase_averaging_apodization.create_dataset(
                    "apodization_window", data=self.apodization_window)
                phase_averaging_apodization.create_dataset(
                    "half_width_avg_phase", data=self.half_width_avg_phase)
                phase_averaging_apodization.create_dataset(
                    "apodization_mu", data=self.apodization_mu)
                phase_averaging_apodization.create_dataset(
                    "apodization_sigma", data=self.apodization_sigma)
                phase_averaging_apodization.create_dataset(
                    "apodization_alpha", data=self.apodization_alpha)
            except AttributeError:
                print("Could not save phase averaging apodization parameters")

            # Save strain output
            try:
                image_3.create_dataset("strain_analysis_output_file",
                                       data=self.strain_output_file)

                with h5py.File(self.strain_output_file, "r") as fi:
                    image_3.create_dataset("amplitude",
                                           data=fi["output"]["amp"][:],
                                           chunks=True,
                                           shuffle=True,
                                           compression="gzip")

                    image_3.create_dataset("phase",
                                           data=fi["output"]["phase"][:],
                                           chunks=True,
                                           shuffle=True,
                                           compression="gzip")

                    image_3.create_dataset("bulk",
                                           data=fi["output"]["bulk"][:],
                                           chunks=True,
                                           shuffle=True,
                                           compression="gzip")

                    image_3.create_dataset("strain",
                                           data=fi["output"]["strain"][:],
                                           chunks=True,
                                           shuffle=True,
                                           compression="gzip")

                    image_3.create_dataset("voxel_sizes",
                                           data=fi["output"]["voxel_sizes"])

                    image_3.create_dataset("q_com",
                                           data=fi["output"]["q_com"])

                image_3.attrs['signal'] = 'phase'

            except AttributeError:
                print("Could not save strain output")

            # Create data_3 link
            try:
                f["entry_1"]["image_3"]["phase"]
                data_3["data"] = h5py.SoftLink(
                    "/entry_1/image_3/phase")
                data_3.attrs['signal'] = 'data'

            except KeyError:
                pass

        print(
            f"Saved file as {self.scan_folder}{self.sample_name}{self.scan}.cxi")
