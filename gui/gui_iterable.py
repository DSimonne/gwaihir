#!/usr/bin/env python
# coding: utf-8

# In[ ]:

"""
Regroups all the possible classes that can be used used as iterable in the Interface class
For now only Dataset
"""

try :
    import numpy as np
    import pandas as pd
    import time
    import h5py

    import glob
    import os

    import ipywidgets as widgets
    from ipywidgets import interact, Button, Layout, interactive, fixed
    from IPython.display import display, Markdown, Latex, clear_output

    from datetime import datetime
    import pickle

    import tables as tb

except ModuleNotFoundError:
    raise ModuleNotFoundError("""The following packages must be installed: numpy, pandas ipywidgets, iPython, thorondor and pytables.""")

class Dataset():
    """
    THE DATASETS CLASS IS MEANT TO BE READ VIA THE gwaihir.gui CLASS !!
    """

    def __init__(self, scans, sample_name, data_directory, root_folder):
        """Initialiaze the Dataset class, some metadata can be associated as well
        """

        self.scans = scans
        self.sample_name = sample_name
        self.data_directory = data_directory
        self.root_folder = root_folder

        # self.pickle()

    def pickle(self):
        """Use the pickle module to save the classes
        """
        try:
            with open(f"{self.saving_directory}/"+self.name.split("~")[0]+".pickle", 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        except PermissionError:
            print("""Permission denied, You cannot save this file because you are not its creator. The changes are updated for this session and you can still plot figures but once you exit the program, all changes will be erased.""")
            pass

    @staticmethod
    def unpickle(prompt):
        """Use the pickle module to load the classes
        """

        with open(f"{prompt}", 'rb') as f:
            return pickle.load(f)


    def __repr__(self):
        return "Dataset {}{}.\n".format(
                    self.sample_name, 
                    self.scans,
                    )
    
    def __str__(self):        
        return repr(self)


    def to_gwr(self):
        """
        Save all the parameters used in the data analysis with a specific architecture
        Alias for hdf5 file,
        Can be reloaded with the load_gwr() function 
        Always overwrites for now
        """

        # Create file
        with h5py.File(f"{self.scan_folder}{self.sample_name}{self.scans}.h5", mode="w") as f:

            # Init
            f.create_dataset("ObjectName", data = f"Dataset_{self.sample_name}{self.scans}")
            f.create_dataset("FileTimeStamp", data = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(time.time())))

            # Parameters
            parameters = f.create_group("parameters")


            ## Preprocessing
            preprocessing = parameters.create_group("preprocessing")

            ### Masking
            masking = preprocessing.create_group("masking")
            try:
                masking.create_dataset("flag_interact", data = self.flag_interact)
                masking.create_dataset("background_plot", data = self.background_plot)

            except AttributeError:
                print("Could not save masking parameters")

            ### Cropping padding centering
            cropping_padding_centering = preprocessing.create_group("cropping_padding_centering")
            try:
                cropping_padding_centering.create_dataset("centering", data = self.centering) 
                cropping_padding_centering.create_dataset("fix_bragg", data = self.fix_bragg) 
                cropping_padding_centering.create_dataset("fix_size", data = self.fix_size) 
                cropping_padding_centering.create_dataset("center_fft", data = self.center_fft) 
                cropping_padding_centering.create_dataset("pad_size", data = self.pad_size)
            except AttributeError:
                print("Could not save cropping padding centering parameters")

            ### Intensity normalization
            intensity_normalization = preprocessing.create_group("intensity_normalization")
            try:
                intensity_normalization.create_dataset("normalize_flux", data = self.normalize_flux) 
            except AttributeError:
                print("Could not save intensity normalization parameters")

            ### Data filtering
            data_filtering = preprocessing.create_group("data_filtering")
            try:
                data_filtering.create_dataset("mask_zero_event", data = self.mask_zero_event) 
                data_filtering.create_dataset("flag_medianfilter", data = self.flag_medianfilter) 
                data_filtering.create_dataset("medfilt_order", data = self.medfilt_order) 
                data_filtering.create_dataset("binning", data = self.binning)
            except AttributeError:
                print("Could not save data filtering parameters")

            ### Saving options
            saving_options = preprocessing.create_group("saving_options")
            try:
                saving_options.create_dataset("save_rawdata", data = self.save_rawdata) 
                saving_options.create_dataset("save_to_npz", data = self.save_to_npz) 
                saving_options.create_dataset("save_to_mat", data = self.save_to_mat) 
                saving_options.create_dataset("save_to_vti", data = self.save_to_vti) 
                saving_options.create_dataset("save_asint", data = self.save_asint)
            except AttributeError:
                print("Could not save saving options")

            ### Reloading options
            reload_options = preprocessing.create_group("reload_options")
            try:
                reload_options.create_dataset("reload_previous", data = self.reload_previous) 
                reload_options.create_dataset("reload_orthogonal", data = self.reload_orthogonal) 
                reload_options.create_dataset("preprocessing_binning", data = self.preprocessing_binning)
            except AttributeError:
                print("Could not save reloading options")

            ### Beamline
            beamline = preprocessing.create_group("beamline")
            try:
                beamline.create_dataset("beamline", data = self.beamline)
                beamline.create_dataset("actuators", data = str(self.actuators)) 
                beamline.create_dataset("is_series", data = self.is_series) 
                beamline.create_dataset("custom_scan", data = self.custom_scan) 
                beamline.create_dataset("custom_images", data = self.custom_images) 
                beamline.create_dataset("custom_monitor", data = self.custom_monitor) 
                beamline.create_dataset("specfile_name", data = str(self.specfile_name)) 
                beamline.create_dataset("rocking_angle", data = self.rocking_angle) 
                beamline.create_dataset("follow_bragg", data = self.follow_bragg)
            except AttributeError:
                print("Could not save beamline parameters")

            ### Detector
            detector = preprocessing.create_group("detector")
            try:
                detector.create_dataset("detector", data = self.detector)
                detector.create_dataset("photon_threshold", data = self.photon_threshold) 
                detector.create_dataset("photon_filter", data = self.photon_filter) 
                detector.create_dataset("background_file", data = str(self.background_file)) 
                detector.create_dataset("hotpixels_file", data = str(self.hotpixels_file)) 
                detector.create_dataset("flatfield_file", data = str(self.flatfield_file))
                detector.create_dataset("template_imagefile", data = str(self.template_imagefile))
            except AttributeError:
                print("Could not save detector parameters")

            try:
                detector.create_dataset("nb_pixel_x", data = self.nb_pixel_x) 
                detector.create_dataset("nb_pixel_y", data = self.nb_pixel_y)
            except (TypeError, AttributeError):
                pass

            ### Orthogonalisation
            orthogonalisation = preprocessing.create_group("orthogonalisation")

            #### Linearized transformation matrix
            linearized_transformation_matrix = orthogonalisation.create_group("linearized_transformation_matrix")
            try:
                linearized_transformation_matrix.create_dataset("use_rawdata", data = self.use_rawdata)
                linearized_transformation_matrix.create_dataset("interp_method", data = self.interp_method)
                linearized_transformation_matrix.create_dataset("fill_value_mask", data = self.fill_value_mask)
                linearized_transformation_matrix.create_dataset("beam_direction", data = self.beam_direction)
                linearized_transformation_matrix.create_dataset("sample_offsets", data = self.sample_offsets)
                linearized_transformation_matrix.create_dataset("sdd", data = self.sdd)
                linearized_transformation_matrix.create_dataset("energy", data = self.energy)
                linearized_transformation_matrix.create_dataset("custom_motors", data = str(self.custom_motors))
            except AttributeError:
                print("Could not save linearized transformation matrix parameters")

            ### xrayutilities
            xrayutilities = orthogonalisation.create_group("xrayutilities")
            try:
                xrayutilities.create_dataset("align_q", data = self.align_q) 
                xrayutilities.create_dataset("ref_axis_q", data = self.ref_axis_q) 
                xrayutilities.create_dataset("outofplane_angle", data = self.outofplane_angle) 
                xrayutilities.create_dataset("inplane_angle", data = self.inplane_angle) 
                xrayutilities.create_dataset("sample_inplane", data = self.sample_inplane) 
                xrayutilities.create_dataset("sample_outofplane", data = self.sample_outofplane) 
                xrayutilities.create_dataset("offset_inplane", data = self.offset_inplane) 
                xrayutilities.create_dataset("cch1", data = self.cch1) 
                xrayutilities.create_dataset("cch2", data = self.cch2) 
                xrayutilities.create_dataset("direct_inplane", data = self.direct_inplane) 
                xrayutilities.create_dataset("direct_outofplane", data = self.direct_outofplane) 
                xrayutilities.create_dataset("detrot", data = self.detrot) 
                xrayutilities.create_dataset("tiltazimuth", data = self.tiltazimuth) 
                xrayutilities.create_dataset("tilt", data = self.tilt)
            except AttributeError:
                print("Could not save xrayutilities parameters")


            ## Temperature estimation
            temperature_estimation = parameters.create_group("temperature_estimation")
            try:
                temperature_estimation.create_dataset("reflection", data = self.reflection)
                temperature_estimation.create_dataset("reference_spacing", data = self.reference_spacing)
                temperature_estimation.create_dataset("reference_temperature", data = self.reference_temperature)
            except AttributeError:
                print("Could not save temperature_estimation parameters")

            try:
                temperature_estimation.create_dataset("estimated_temperature", data = self.temperature)
            except AttributeError:
                print("No estimated temperature")

            ## Angles correction
            angles_corrections = parameters.create_group("angles_corrections")
            try:
                angles_corrections.create_dataset("tilt_values", data = self.tilt_values)
                angles_corrections.create_dataset("rocking_curve", data = self.rocking_curve)
                angles_corrections.create_dataset("interp_tilt", data = self.interp_tilt)
                angles_corrections.create_dataset("interp_curve", data = self.interp_curve)
                angles_corrections.create_dataset("COM_rocking_curve", data = self.COM_rocking_curve)
                angles_corrections.create_dataset("detector_data_COM", data = self.detector_data_COM)
                angles_corrections.create_dataset("interp_fwhm", data = self.interp_fwhm)
                angles_corrections.create_dataset("bragg_x", data = self.bragg_x) 
                angles_corrections.create_dataset("bragg_y", data = self.bragg_y)
                angles_corrections.create_dataset("q", data = self.q) 
                angles_corrections.create_dataset("qnorm", data = self.qnorm) 
                angles_corrections.create_dataset("dist_plane", data = self.dist_plane) 
                angles_corrections.create_dataset("bragg_inplane", data = self.bragg_inplane) 
                angles_corrections.create_dataset("bragg_outofplane", data = self.bragg_outofplane)
            except AttributeError:
                print("Could not save angles_corrections parameters")


            ## Postprocessing
            postprocessing = parameters.create_group("postprocessing")

            ### Averaging reconstructions
            averaging_reconstructions = postprocessing.create_group("averaging_reconstructions")
            try:
                averaging_reconstructions.create_dataset("sort_method", data = self.sort_method) 
                averaging_reconstructions.create_dataset("correlation_threshold", data = self.correlation_threshold)
            except AttributeError:
                print("Could not save averaging reconstructions parameters")


            ### FFT_window_voxel
            FFT_window_voxel = postprocessing.create_group("FFT_window_voxel")
            try:
                FFT_window_voxel.create_dataset("phasing_binning", data = self.phasing_binning) 
                FFT_window_voxel.create_dataset("original_size", data = self.original_size) 
                FFT_window_voxel.create_dataset("preprocessing_binning", data = self.preprocessing_binning) 
                FFT_window_voxel.create_dataset("output_size", data = str(self.output_size)) 
                FFT_window_voxel.create_dataset("keep_size", data = self.keep_size) 
                FFT_window_voxel.create_dataset("fix_voxel", data = str(self.fix_voxel))
            except AttributeError:
                print("Could not save angles_corrections parameters")


            ### Displacement strain calculation
            displacement_strain_calculation = postprocessing.create_group("displacement_strain_calculation")
            try:
                displacement_strain_calculation.create_dataset("data_frame", data = self.data_frame)
                displacement_strain_calculation.create_dataset("save_frame", data = self.save_frame)
                displacement_strain_calculation.create_dataset("ref_axis_q", data = self.ref_axis_q)
                displacement_strain_calculation.create_dataset("isosurface_strain", data = self.isosurface_strain)
                displacement_strain_calculation.create_dataset("strain_method", data = self.strain_method)
                displacement_strain_calculation.create_dataset("phase_offset", data = self.phase_offset)
                displacement_strain_calculation.create_dataset("phase_offset_origin", data = str(self.phase_offset_origin))
                displacement_strain_calculation.create_dataset("offset_method", data = self.offset_method)
                displacement_strain_calculation.create_dataset("centering_method", data = self.centering_method)
            except AttributeError:
                print("Could not save displacement & strain calculation parameters")

            ### Refraction
            refraction = postprocessing.create_group("refraction")
            try:
                refraction.create_dataset("correct_refraction", data = self.correct_refraction)
                refraction.create_dataset("optical_path_method", data = self.optical_path_method)
                refraction.create_dataset("dispersion", data = self.dispersion)
                refraction.create_dataset("absorption", data = self.absorption)
                refraction.create_dataset("threshold_unwrap_refraction", data = self.threshold_unwrap_refraction)
            except AttributeError:
                print("Could not save refraction parameters")

            ### Options
            options = postprocessing.create_group("options")
            try:
                options.create_dataset("simu_flag", data = self.simu_flag)
                options.create_dataset("invert_phase", data = self.invert_phase)
                options.create_dataset("flip_reconstruction", data = self.flip_reconstruction)
                options.create_dataset("phase_ramp_removal", data = self.phase_ramp_removal)
                options.create_dataset("threshold_gradient", data = self.threshold_gradient)
                options.create_dataset("save_raw", data = self.save_raw)
                options.create_dataset("save_support", data = self.save_support)
                options.create_dataset("save", data = self.save)
                options.create_dataset("debug", data = self.debug)
                options.create_dataset("roll_modes", data = self.roll_modes)
            except AttributeError:
                print("Could not save postprocessing options")


            ### Data visualisation
            data_visualisation = postprocessing.create_group("data_visualisation")
            try:
                data_visualisation.create_dataset("align_axis", data = self.align_axis)
                data_visualisation.create_dataset("ref_axis", data = self.ref_axis)
                data_visualisation.create_dataset("axis_to_align", data = self.axis_to_align)
                data_visualisation.create_dataset("strain_range", data = self.strain_range)
                data_visualisation.create_dataset("phase_range", data = self.phase_range)
                data_visualisation.create_dataset("grey_background", data = self.grey_background)
                data_visualisation.create_dataset("tick_spacing", data = self.tick_spacing)
                data_visualisation.create_dataset("tick_direction", data = self.tick_direction)
                data_visualisation.create_dataset("tick_length", data = self.tick_length)
                data_visualisation.create_dataset("tick_width", data = self.tick_width)
            except AttributeError:
                print("Could not save data visualisation parameters")


            ### Averaging reconstructed objects
            averaging_reconstructed_objects = postprocessing.create_group("averaging_reconstructed_objects")
            try:
                averaging_reconstructed_objects.create_dataset("avg_method", data = self.avg_method)
                averaging_reconstructed_objects.create_dataset("avg_threshold", data = self.avg_threshold)
            except AttributeError:
                print("Could not save averaging reconstructed objects parameters")


            ### Phase averaging apodization
            phase_averaging_apodization = postprocessing.create_group("phase_averaging_apodization")
            try:
                phase_averaging_apodization.create_dataset("apodize_flag", data = self.apodize_flag)
                phase_averaging_apodization.create_dataset("apodize_window", data = self.apodize_window)
                phase_averaging_apodization.create_dataset("hwidth", data = self.hwidth)
                phase_averaging_apodization.create_dataset("mu", data = self.mu)
                phase_averaging_apodization.create_dataset("sigma", data = self.sigma)
                phase_averaging_apodization.create_dataset("alpha", data = self.alpha)
            except AttributeError:
                print("Could not save phase averaging apodization parameters")


            ## Phase retrieval
            phase_retrieval = parameters.create_group("phase_retrieval")

            ### Support
            support = phase_retrieval.create_group("support")
            try:
                support.create_dataset("support_threshold", data = self.support_threshold)
                support.create_dataset("support_only_shrink", data = self.support_only_shrink)
                support.create_dataset("support_update_period", data = self.support_update_period)
                support.create_dataset("support_smooth_width", data = self.support_smooth_width)
                support.create_dataset("support_post_expand", data = self.support_post_expand)
            except AttributeError:
                print("Could not save support parameters")

            ### Data
            data = phase_retrieval.create_group("data")
            try:
                data.create_dataset("iobs", data = str(self.iobs))
                data.create_dataset("mask", data = str(self.mask))
                data.create_dataset("support", data = str(self.support))
                data.create_dataset("obj", data = str(self.obj))
                data.create_dataset("auto_center_resize", data = self.auto_center_resize)
                data.create_dataset("max_size", data = self.max_size)
            except AttributeError:
                print("Could not save data parameters")

            ### PSF
            PSF = phase_retrieval.create_group("PSF")
            try:
                PSF.create_dataset("psf", data = self.psf)
                PSF.create_dataset("psf_model", data = self.psf_model)
                PSF.create_dataset("fwhm", data = self.fwhm)
                PSF.create_dataset("eta", data = self.eta)
                PSF.create_dataset("update_psf", data = self.update_psf)
            except AttributeError:
                print("Could not save PSF parameters")

            ### Algorithms
            algorithms = phase_retrieval.create_group("algorithms")
            try:
                algorithms.create_dataset("use_operators", data = self.use_operators)
                algorithms.create_dataset("operator_chain", data = self.operator_chain)
                algorithms.create_dataset("nb_hio", data = self.nb_hio)
                algorithms.create_dataset("nb_raar", data = self.nb_raar)
                algorithms.create_dataset("nb_er", data = self.nb_er)
                algorithms.create_dataset("nb_ml", data = self.nb_ml)
                algorithms.create_dataset("nb_run", data = self.nb_run)
                algorithms.create_dataset("positivity", data = self.positivity)
                algorithms.create_dataset("beta", data = self.beta)
                algorithms.create_dataset("detwin", data = self.detwin)
                algorithms.create_dataset("rebin", data = self.rebin)
            except AttributeError:
                print("Could not save algorithms parameters")

            ### Instrument
            instrument = phase_retrieval.create_group("instrument")
            try:
                instrument.create_dataset("sdd", data = self.sdd)
                instrument.create_dataset("pixel_size_detector", data = self.pixel_size_detector)
            except AttributeError:
                print("Could not save instrument parameters")

            ### Filtering
            filtering = phase_retrieval.create_group("filtering")
            try:
                filtering.create_dataset("filter_criteria", data = self.filter_criteria)
                filtering.create_dataset("nb_run_keep", data = self.nb_run_keep)
            except AttributeError:
                print("Could not save filtering parameters")

            # Data
            data = f.create_group("data")

            ## Reciprocal space
            reciprocal_space = data.create_group("reciprocal_space")

            ### Save 3D coherent diffraction intensity
            try:
                reciprocal_space.create_dataset("data",
                                          data = np.load(self.iobs)["data"],
                                          chunks=True,
                                          shuffle=True,
                                          compression="gzip")
                reciprocal_space.create_dataset("mask",
                                          data = np.load(self.mask)["mask"],
                                          chunks=True,
                                          shuffle=True,
                                          compression="gzip")
            except:
                print("Could not save reciprocal space data and mask")

            try:
                reciprocal_space.create_dataset("q_final", data = self.q_final)
                reciprocal_space["detector_distance"] = h5py.SoftLink("/parameters/phase_retrieval/instrument/sdd")
                reciprocal_space["pixel_size_detector"] = h5py.SoftLink("/parameters/phase_retrieval/instrument/pixel_size_detector")
                reciprocal_space["energy"] = h5py.SoftLink("/parameters/preprocessing/orthogonalisation/linearized_transformation_matrix/energy")
                reciprocal_space.create_dataset("wavelength", data = self.wavelength)
            except:
                print("Could not save reciprocal space ")

            ## Real space
            real_space = data.create_group("real_space")

            ### Save raw electronic density
            try:
                with h5py.File(self.reconstruction_file, "r") as fi:
                    real_space.create_dataset("raw_electronic_density_file", data = self.reconstruction_file)
                    real_space.create_dataset("raw_electronic_density", 
                                                    data = fi["entry_1"]["image_1"]["data"][:],
                                                    chunks=True, 
                                                    shuffle=True,
                                                    compression="gzip")

            except:
                print("Could not save electronic density")

            ### Save strain output
            try:
                real_space.create_dataset("voxel_size", data = self.voxel_size)
                real_space.create_dataset("strain_analysis_output_file", data = self.strain_output_file)

                with h5py.File(self.strain_output_file, "r") as fi:
                            real_space.create_dataset("amplitude", 
                                                    data = fi["output"]["amp"][:],
                                                    chunks=True, 
                                                    shuffle=True,
                                                    compression="gzip")

                            real_space.create_dataset("phase", 
                                                    data = fi["output"]["phase"][:],
                                                    chunks=True, 
                                                    shuffle=True,
                                                    compression="gzip")

                            real_space.create_dataset("bulk", 
                                                    data = fi["output"]["bulk"][:],
                                                    chunks=True, 
                                                    shuffle=True,
                                                    compression="gzip")

                            real_space.create_dataset("strain", 
                                                    data = fi["output"]["strain"][:],
                                                    chunks=True, 
                                                    shuffle=True,
                                                    compression="gzip")
            except AttributeError:
                print("Could not save strain output")

        print(f"Saved file as {self.scan_folder}{self.sample_name}{self.scans}.h5")



    def load_gwr(self):
        """
        Load all the parameters used in the data analysis with a specific architecture
        Alias for hdf5 file,
        Can be written with the to_gwr() function 
        """

        # Create file
        with h5py.File(f"{self.scan_folder}{self.sample_name}{self.scans}.h5", mode="r") as f:

            # Init
            f.create_dataset("ObjectName", data = f"Dataset_{self.sample_name}{self.scans}")
            f.create_dataset("FileTimeStamp", data = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(time.time())))


            # Parameters
            parameters = f.create_group("parameters")


            ## Preprocessing
            preprocessing = parameters.create_group("preprocessing")

            ### Masking
            masking = preprocessing.create_group("masking")
            try:
                masking.create_dataset("flag_interact", data = self.flag_interact)
                masking.create_dataset("background_plot", data = self.background_plot)

            except AttributeError:
                print("Could not save masking parameters")

            ### Cropping padding centering
            cropping_padding_centering = preprocessing.create_group("cropping_padding_centering")
            try:
                cropping_padding_centering.create_dataset("centering", data = self.centering) 
                cropping_padding_centering.create_dataset("fix_bragg", data = self.fix_bragg) 
                cropping_padding_centering.create_dataset("fix_size", data = self.fix_size) 
                cropping_padding_centering.create_dataset("center_fft", data = self.center_fft) 
                cropping_padding_centering.create_dataset("pad_size", data = self.pad_size)
            except AttributeError:
                print("Could not save cropping padding centering parameters")

            ### Intensity normalization
            intensity_normalization = preprocessing.create_group("intensity_normalization")
            try:
                intensity_normalization.create_dataset("normalize_flux", data = self.normalize_flux) 
            except AttributeError:
                print("Could not save intensity normalization parameters")

            ### Data filtering
            data_filtering = preprocessing.create_group("data_filtering")
            try:
                data_filtering.create_dataset("mask_zero_event", data = self.mask_zero_event) 
                data_filtering.create_dataset("flag_medianfilter", data = self.flag_medianfilter) 
                data_filtering.create_dataset("medfilt_order", data = self.medfilt_order) 
                data_filtering.create_dataset("binning", data = self.binning)
            except AttributeError:
                print("Could not save data filtering parameters")

            ### Saving options
            saving_options = preprocessing.create_group("saving_options")
            try:
                saving_options.create_dataset("save_rawdata", data = self.save_rawdata) 
                saving_options.create_dataset("save_to_npz", data = self.save_to_npz) 
                saving_options.create_dataset("save_to_mat", data = self.save_to_mat) 
                saving_options.create_dataset("save_to_vti", data = self.save_to_vti) 
                saving_options.create_dataset("save_asint", data = self.save_asint)
            except AttributeError:
                print("Could not save saving options")

            ### Reloading options
            reload_options = preprocessing.create_group("reload_options")
            try:
                reload_options.create_dataset("reload_previous", data = self.reload_previous) 
                reload_options.create_dataset("reload_orthogonal", data = self.reload_orthogonal) 
                reload_options.create_dataset("preprocessing_binning", data = self.preprocessing_binning)
            except AttributeError:
                print("Could not save reloading options")

            ### Beamline
            beamline = preprocessing.create_group("beamline")
            try:
                beamline.create_dataset("beamline", data = self.beamline)
                beamline.create_dataset("actuators", data = str(self.actuators)) 
                beamline.create_dataset("is_series", data = self.is_series) 
                beamline.create_dataset("custom_scan", data = self.custom_scan) 
                beamline.create_dataset("custom_images", data = self.custom_images) 
                beamline.create_dataset("custom_monitor", data = self.custom_monitor) 
                beamline.create_dataset("specfile_name", data = str(self.specfile_name)) 
                beamline.create_dataset("rocking_angle", data = self.rocking_angle) 
                beamline.create_dataset("follow_bragg", data = self.follow_bragg)
            except AttributeError:
                print("Could not save beamline parameters")

            ### Detector
            detector = preprocessing.create_group("detector")
            try:
                detector.create_dataset("detector", data = self.detector)
                detector.create_dataset("photon_threshold", data = self.photon_threshold) 
                detector.create_dataset("photon_filter", data = self.photon_filter) 
                detector.create_dataset("background_file", data = str(self.background_file)) 
                detector.create_dataset("hotpixels_file", data = str(self.hotpixels_file)) 
                detector.create_dataset("flatfield_file", data = str(self.flatfield_file))
                detector.create_dataset("template_imagefile", data = str(self.template_imagefile))
            except AttributeError:
                print("Could not save detector parameters")

            try:
                detector.create_dataset("nb_pixel_x", data = self.nb_pixel_x) 
                detector.create_dataset("nb_pixel_y", data = self.nb_pixel_y)
            except (TypeError, AttributeError):
                pass

            ### Orthogonalisation
            orthogonalisation = preprocessing.create_group("orthogonalisation")

            #### Linearized transformation matrix
            linearized_transformation_matrix = orthogonalisation.create_group("linearized_transformation_matrix")
            try:
                linearized_transformation_matrix.create_dataset("use_rawdata", data = self.use_rawdata)
                linearized_transformation_matrix.create_dataset("interp_method", data = self.interp_method)
                linearized_transformation_matrix.create_dataset("fill_value_mask", data = self.fill_value_mask)
                linearized_transformation_matrix.create_dataset("beam_direction", data = self.beam_direction)
                linearized_transformation_matrix.create_dataset("sample_offsets", data = self.sample_offsets)
                linearized_transformation_matrix.create_dataset("sdd", data = self.sdd)
                linearized_transformation_matrix.create_dataset("energy", data = self.energy)
                linearized_transformation_matrix.create_dataset("custom_motors", data = str(self.custom_motors))
            except AttributeError:
                print("Could not save linearized transformation matrix parameters")

            ### xrayutilities
            xrayutilities = orthogonalisation.create_group("xrayutilities")
            try:
                xrayutilities.create_dataset("align_q", data = self.align_q) 
                xrayutilities.create_dataset("ref_axis_q", data = self.ref_axis_q) 
                xrayutilities.create_dataset("outofplane_angle", data = self.outofplane_angle) 
                xrayutilities.create_dataset("inplane_angle", data = self.inplane_angle) 
                xrayutilities.create_dataset("sample_inplane", data = self.sample_inplane) 
                xrayutilities.create_dataset("sample_outofplane", data = self.sample_outofplane) 
                xrayutilities.create_dataset("offset_inplane", data = self.offset_inplane) 
                xrayutilities.create_dataset("cch1", data = self.cch1) 
                xrayutilities.create_dataset("cch2", data = self.cch2) 
                xrayutilities.create_dataset("direct_inplane", data = self.direct_inplane) 
                xrayutilities.create_dataset("direct_outofplane", data = self.direct_outofplane) 
                xrayutilities.create_dataset("detrot", data = self.detrot) 
                xrayutilities.create_dataset("tiltazimuth", data = self.tiltazimuth) 
                xrayutilities.create_dataset("tilt", data = self.tilt)
            except AttributeError:
                print("Could not save xrayutilities parameters")


            ## Temperature estimation
            temperature_estimation = parameters.create_group("temperature_estimation")
            try:
                temperature_estimation.create_dataset("reflection", data = self.reflection)
                temperature_estimation.create_dataset("reference_spacing", data = self.reference_spacing)
                temperature_estimation.create_dataset("reference_temperature", data = self.reference_temperature)
            except AttributeError:
                print("Could not save temperature_estimation parameters")

            try:
                temperature_estimation.create_dataset("estimated_temperature", data = self.temperature)
            except AttributeError:
                print("No estimated temperature")

            ## Angles correction
            angles_corrections = parameters.create_group("angles_corrections")
            try:
                angles_corrections.create_dataset("tilt_values", data = self.tilt_values)
                angles_corrections.create_dataset("rocking_curve", data = self.rocking_curve)
                angles_corrections.create_dataset("interp_tilt", data = self.interp_tilt)
                angles_corrections.create_dataset("interp_curve", data = self.interp_curve)
                angles_corrections.create_dataset("COM_rocking_curve", data = self.COM_rocking_curve)
                angles_corrections.create_dataset("detector_data_COM", data = self.detector_data_COM)
                angles_corrections.create_dataset("interp_fwhm", data = self.interp_fwhm)
                angles_corrections.create_dataset("bragg_x", data = self.bragg_x) 
                angles_corrections.create_dataset("bragg_y", data = self.bragg_y)
                angles_corrections.create_dataset("q", data = self.q) 
                angles_corrections.create_dataset("qnorm", data = self.qnorm) 
                angles_corrections.create_dataset("dist_plane", data = self.dist_plane) 
                angles_corrections.create_dataset("bragg_inplane", data = self.bragg_inplane) 
                angles_corrections.create_dataset("bragg_outofplane", data = self.bragg_outofplane)
            except AttributeError:
                print("Could not save angles_corrections parameters")


            ## Postprocessing
            postprocessing = parameters.create_group("postprocessing")

            ### Averaging reconstructions
            averaging_reconstructions = postprocessing.create_group("averaging_reconstructions")
            try:
                averaging_reconstructions.create_dataset("sort_method", data = self.sort_method) 
                averaging_reconstructions.create_dataset("correlation_threshold", data = self.correlation_threshold)
            except AttributeError:
                print("Could not save averaging reconstructions parameters")


            ### FFT_window_voxel
            FFT_window_voxel = postprocessing.create_group("FFT_window_voxel")
            try:
                FFT_window_voxel.create_dataset("phasing_binning", data = self.phasing_binning) 
                FFT_window_voxel.create_dataset("original_size", data = self.original_size) 
                FFT_window_voxel.create_dataset("preprocessing_binning", data = self.preprocessing_binning) 
                FFT_window_voxel.create_dataset("output_size", data = str(self.output_size)) 
                FFT_window_voxel.create_dataset("keep_size", data = self.keep_size) 
                FFT_window_voxel.create_dataset("fix_voxel", data = str(self.fix_voxel))
            except AttributeError:
                print("Could not save angles_corrections parameters")


            ### Displacement strain calculation
            displacement_strain_calculation = postprocessing.create_group("displacement_strain_calculation")
            try:
                displacement_strain_calculation.create_dataset("data_frame", data = self.data_frame)
                displacement_strain_calculation.create_dataset("save_frame", data = self.save_frame)
                displacement_strain_calculation.create_dataset("ref_axis_q", data = self.ref_axis_q)
                displacement_strain_calculation.create_dataset("isosurface_strain", data = self.isosurface_strain)
                displacement_strain_calculation.create_dataset("strain_method", data = self.strain_method)
                displacement_strain_calculation.create_dataset("phase_offset", data = self.phase_offset)
                displacement_strain_calculation.create_dataset("phase_offset_origin", data = str(self.phase_offset_origin))
                displacement_strain_calculation.create_dataset("offset_method", data = self.offset_method)
                displacement_strain_calculation.create_dataset("centering_method", data = self.centering_method)
            except AttributeError:
                print("Could not save displacement & strain calculation parameters")

            ### Refraction
            refraction = postprocessing.create_group("refraction")
            try:
                refraction.create_dataset("correct_refraction", data = self.correct_refraction)
                refraction.create_dataset("optical_path_method", data = self.optical_path_method)
                refraction.create_dataset("dispersion", data = self.dispersion)
                refraction.create_dataset("absorption", data = self.absorption)
                refraction.create_dataset("threshold_unwrap_refraction", data = self.threshold_unwrap_refraction)
            except AttributeError:
                print("Could not save refraction parameters")

            ### Options
            options = postprocessing.create_group("options")
            try:
                options.create_dataset("simu_flag", data = self.simu_flag)
                options.create_dataset("invert_phase", data = self.invert_phase)
                options.create_dataset("flip_reconstruction", data = self.flip_reconstruction)
                options.create_dataset("phase_ramp_removal", data = self.phase_ramp_removal)
                options.create_dataset("threshold_gradient", data = self.threshold_gradient)
                options.create_dataset("save_raw", data = self.save_raw)
                options.create_dataset("save_support", data = self.save_support)
                options.create_dataset("save", data = self.save)
                options.create_dataset("debug", data = self.debug)
                options.create_dataset("roll_modes", data = self.roll_modes)
            except AttributeError:
                print("Could not save postprocessing options")


            ### Data visualisation
            data_visualisation = postprocessing.create_group("data_visualisation")
            try:
                data_visualisation.create_dataset("align_axis", data = self.align_axis)
                data_visualisation.create_dataset("ref_axis", data = self.ref_axis)
                data_visualisation.create_dataset("axis_to_align", data = self.axis_to_align)
                data_visualisation.create_dataset("strain_range", data = self.strain_range)
                data_visualisation.create_dataset("phase_range", data = self.phase_range)
                data_visualisation.create_dataset("grey_background", data = self.grey_background)
                data_visualisation.create_dataset("tick_spacing", data = self.tick_spacing)
                data_visualisation.create_dataset("tick_direction", data = self.tick_direction)
                data_visualisation.create_dataset("tick_length", data = self.tick_length)
                data_visualisation.create_dataset("tick_width", data = self.tick_width)
            except AttributeError:
                print("Could not save data visualisation parameters")


            ### Averaging reconstructed objects
            averaging_reconstructed_objects = postprocessing.create_group("averaging_reconstructed_objects")
            try:
                averaging_reconstructed_objects.create_dataset("avg_method", data = self.avg_method)
                averaging_reconstructed_objects.create_dataset("avg_threshold", data = self.avg_threshold)
            except AttributeError:
                print("Could not save averaging reconstructed objects parameters")


            ### Phase averaging apodization
            phase_averaging_apodization = postprocessing.create_group("phase_averaging_apodization")
            try:
                phase_averaging_apodization.create_dataset("apodize_flag", data = self.apodize_flag)
                phase_averaging_apodization.create_dataset("apodize_window", data = self.apodize_window)
                phase_averaging_apodization.create_dataset("hwidth", data = self.hwidth)
                phase_averaging_apodization.create_dataset("mu", data = self.mu)
                phase_averaging_apodization.create_dataset("sigma", data = self.sigma)
                phase_averaging_apodization.create_dataset("alpha", data = self.alpha)
            except AttributeError:
                print("Could not save phase averaging apodization parameters")


            ## Phase retrieval
            phase_retrieval = parameters.create_group("phase_retrieval")

            ### Support
            support = phase_retrieval.create_group("support")
            try:
                support.create_dataset("support_threshold", data = self.support_threshold)
                support.create_dataset("support_only_shrink", data = self.support_only_shrink)
                support.create_dataset("support_update_period", data = self.support_update_period)
                support.create_dataset("support_smooth_width", data = self.support_smooth_width)
                support.create_dataset("support_post_expand", data = self.support_post_expand)
            except AttributeError:
                print("Could not save support parameters")

            ### Data
            data = phase_retrieval.create_group("data")
            try:
                data.create_dataset("iobs", data = str(self.iobs))
                data.create_dataset("mask", data = str(self.mask))
                data.create_dataset("support", data = str(self.support))
                data.create_dataset("obj", data = str(self.obj))
                data.create_dataset("auto_center_resize", data = self.auto_center_resize)
                data.create_dataset("max_size", data = self.max_size)
            except AttributeError:
                print("Could not save data parameters")

            ### PSF
            PSF = phase_retrieval.create_group("PSF")
            try:
                PSF.create_dataset("psf", data = self.psf)
                PSF.create_dataset("psf_model", data = self.psf_model)
                PSF.create_dataset("fwhm", data = self.fwhm)
                PSF.create_dataset("eta", data = self.eta)
                PSF.create_dataset("update_psf", data = self.update_psf)
            except AttributeError:
                print("Could not save PSF parameters")

            ### Algorithms
            algorithms = phase_retrieval.create_group("algorithms")
            try:
                algorithms.create_dataset("use_operators", data = self.use_operators)
                algorithms.create_dataset("operator_chain", data = self.operator_chain)
                algorithms.create_dataset("nb_hio", data = self.nb_hio)
                algorithms.create_dataset("nb_raar", data = self.nb_raar)
                algorithms.create_dataset("nb_er", data = self.nb_er)
                algorithms.create_dataset("nb_ml", data = self.nb_ml)
                algorithms.create_dataset("nb_run", data = self.nb_run)
                algorithms.create_dataset("positivity", data = self.positivity)
                algorithms.create_dataset("beta", data = self.beta)
                algorithms.create_dataset("detwin", data = self.detwin)
                algorithms.create_dataset("rebin", data = self.rebin)
            except AttributeError:
                print("Could not save algorithms parameters")

            ### Instrument
            instrument = phase_retrieval.create_group("instrument")
            try:
                instrument.create_dataset("sdd", data = self.sdd)
                instrument.create_dataset("pixel_size_detector", data = self.pixel_size_detector)
            except AttributeError:
                print("Could not save instrument parameters")

            ### Filtering
            filtering = phase_retrieval.create_group("filtering")
            try:
                filtering.create_dataset("filter_criteria", data = self.filter_criteria)
                filtering.create_dataset("nb_run_keep", data = self.nb_run_keep)
            except AttributeError:
                print("Could not save filtering parameters")

            # Data
            data = f.create_group("data")

            ## Reciprocal space
            reciprocal_space = data.create_group("reciprocal_space")

            ### Save 3D coherent diffraction intensity
            try:
                reciprocal_space.create_dataset("data",
                                          data = np.load(self.iobs)["data"],
                                          chunks=True,
                                          shuffle=True,
                                          compression="gzip")
                reciprocal_space.create_dataset("mask",
                                          data = np.load(self.mask)["mask"],
                                          chunks=True,
                                          shuffle=True,
                                          compression="gzip")
            except:
                print("Could not save reciprocal space data and mask")

            try:
                reciprocal_space.create_dataset("q_final", data = self.q_final)
                reciprocal_space["detector_distance"] = h5py.SoftLink("/parameters/phase_retrieval/instrument/sdd")
                reciprocal_space["pixel_size_detector"] = h5py.SoftLink("/parameters/phase_retrieval/instrument/pixel_size_detector")
                reciprocal_space["energy"] = h5py.SoftLink("/parameters/preprocessing/orthogonalisation/linearized_transformation_matrix/energy")
                reciprocal_space.create_dataset("wavelength", data = self.wavelength)
            except:
                print("Could not save reciprocal space ")

            ### Facets

            ## Real space
            real_space = data.create_group("real_space")

            ### Save raw electronic density
            try:
                with h5py.File(self.reconstruction_file, "r") as fi:
                    real_space.create_dataset("raw_electronic_density_file", data = self.reconstruction_file)
                    real_space.create_dataset("raw_electronic_density", 
                                                    data = fi["entry_1"]["image_1"]["data"][:],
                                                    chunks=True, 
                                                    shuffle=True,
                                                    compression="gzip")

            except:
                print("Could not save electronic density")

            ### Save strain output
            try:
                real_space.create_dataset("voxel_size", data = self.voxel_size)
                real_space.create_dataset("strain_analysis_output_file", data = self.strain_output_file)

                with h5py.File(self.strain_output_file, "r") as fi:
                            real_space.create_dataset("amplitude", 
                                                    data = fi["output"]["amp"][:],
                                                    chunks=True, 
                                                    shuffle=True,
                                                    compression="gzip")

                            real_space.create_dataset("phase", 
                                                    data = fi["output"]["phase"][:],
                                                    chunks=True, 
                                                    shuffle=True,
                                                    compression="gzip")

                            real_space.create_dataset("bulk", 
                                                    data = fi["output"]["bulk"][:],
                                                    chunks=True, 
                                                    shuffle=True,
                                                    compression="gzip")

                            real_space.create_dataset("strain", 
                                                    data = fi["output"]["strain"][:],
                                                    chunks=True, 
                                                    shuffle=True,
                                                    compression="gzip")
            except AttributeError:
                print("Could not save strain output")

        print(f"Saved file as {self.scan_folder}{self.sample_name}{self.scans}.h5")

