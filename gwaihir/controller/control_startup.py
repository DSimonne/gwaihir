import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import tables as tb
import h5py
import shutil
from IPython.display import display


def initialize_directories(
    unused_label_scan,
    interface,
    sample_name,
    scan,
    data_dir,
    root_folder,
    comment,
    debug,
    matplotlib_backend,
    run_dir_init,
):
    """
    Mandatory to run before any other step

    :param sample_name: e.g. "S"
     str of sample names (usually string in front of the scan number in the
     folder name).
    :param scan: e.g. 11
     scan number
    :param data_dir: e.g. None
     use this to override the beamline default search path for the data
    :param root_folder: folder of the experiment, where all scans are stored
    :param comment: string use in filenames when saving
    :param debug: e.g. False. True to see extra plots to help with debugging
    :param matplotlib_backend: e.g. "Qt5Agg"
     Backend used in script, change to "Agg" to make sure the figures are
     saved, not compatible with interactive masking. Other possibilities
     are 'module://matplotlib_inline.backend_inline' default value is
     "Qt5Agg"

    return:
        Dataset instance
    """
    if run_dir_init:
        # Create Dataset Class
        Dataset = gui_iterable.Dataset(
            scan=scan,
            sample_name=sample_name,
            data_dir=data_dir,
            root_folder=root_folder,
        )

        # Start to assign attributes
        Dataset.comment = comment
        Dataset.debug = debug
        Dataset.scan_name = Dataset.sample_name + str(Dataset.scan)

        # Backend used for plotting
        matplotlib_backend = matplotlib_backend

        # Assign folders
        Dataset.scan_folder = Dataset.root_folder + Dataset.scan_name + "/"
        preprocessing_folder = Dataset.scan_folder + "preprocessing/"
        postprocessing_folder = Dataset.scan_folder + "postprocessing/"

        # Update the directory structure
        print("Updating directories ...")
        init_directories(
            scan_name=Dataset.scan_name,
            root_folder=Dataset.root_folder,
        )

        # Try and find SixS data, will also rotate the data if needed
        template_imagefile, Dataset.data_dir, Dataset.path_to_nxs_data = find_and_copy_raw_data(
            scan=Dataset.scan,
            sample_name=Dataset.sample_name,
            root_folder=Dataset.root_folder,
            data_dir=Dataset.data_dir,
        )

        # Save template_imagefile in GUI
        if template_imagefile != "":
            interface.TabPreprocess.window[42].value\
                = template_imagefile

        # Refresh folders
        interface.sub_directories_handler(change=Dataset.scan_folder)

        # PyNX folder, refresh values
        interface._list_widgets_phase_retrieval.children[1].value\
            = preprocessing_folder
        interface.pynx_folder_handler(change=preprocessing_folder)

        # Plot folder, refresh values
        interface.tab_data.children[1].value = preprocessing_folder
        interface.plot_folder_handler(change=preprocessing_folder)

        # Strain folder, refresh values
        interface._list_widgets_strain.children[-4].value = preprocessing_folder
        interface.strain_folder_handler(change=preprocessing_folder)

        # Facet folder, refresh values
        interface.tab_facet.children[1].value = postprocessing_folder
        interface.vtk_file_handler(change=postprocessing_folder)

        return Dataset, matplotlib_backend, preprocessing_folder, postprocessing_folder

    elif not run_dir_init:
        print("Cleared window.")
        clear_output(True)

        return None, None, None, None


def save_dataset():

    # Only allow to save data if PyNX is imported to avoid errors
    if pynx_import:
        # Button to save data
        button_save_as_cxi = Button(
            description="Save work as .cxi file",
            continuous_update=False,
            button_style='',
            layout=Layout(width='40%'),
            style={'description_width': 'initial'},
            icon='step-forward')

        display(button_save_as_cxi)

        @ button_save_as_cxi.on_click
        def action_button_save_as_cxi(selfbutton):
            """Create button to save Dataset object as .cxi file."""
            clear_output(True)
            display(button_save_as_cxi)
            print("Saving data ...")

            try:
                # Reciprocal space data
                # Define path to .cxi file that will contain the
                # preprocessed data, created thanks to PyNX.
                cxi_filename = "{}/preprocessing/{}.cxi".format(
                    Dataset.scan_folder,
                    Dataset.iobs.split("/")[-1].split(".")[0]
                )

                # Check if this file already exists or not
                if not os.path.isfile(cxi_filename):
                    print(
                        "Saving diffraction data and mask selected in the PyNX tab..."
                    )

                    # Define cxi file with the data selected
                    # in the phase retrieval tab and save as cxi
                    cdi = initialize_cdi_operator(
                        iobs=Dataset.iobs,
                        mask=Dataset.mask,
                        support=Dataset.support,
                        obj=Dataset.obj,
                        rebin=Dataset.rebin,
                        auto_center_resize=Dataset.auto_center_resize,
                        max_size=Dataset.max_size,
                        wavelength=Dataset.wavelength,
                        pixel_size_detector=Dataset.pixel_size_detector,
                        detector_distance=Dataset.detector_distance,
                    )

                    save_cdi_operator_as_cxi(
                        gwaihir_dataset=Dataset,
                        cdi_operator=cdi,
                        path_to_cxi=cxi_filename,
                    )

                # Real space data
                # Path to final file
                final_cxi_filename = "{}{}{}.cxi".format(
                    Dataset.scan_folder,
                    Dataset.sample_name,
                    Dataset.scan,
                )

                Dataset.to_cxi(
                    raw_data_cxi_filename=cxi_filename,
                    final_cxi_filename=final_cxi_filename,
                    reconstruction_filename=interface.reconstruction_files,
                    strain_output_file=interface.strain_output_file
                )

            except (AttributeError, UnboundLocalError):
                print(
                    "Could not save reciprocal space data, select the"
                    "intensity and the mask files in the phase"
                    "retrieval tab first"
                )

            # Facets analysis output
            try:
                print("Saving Facets class data")
                interface.Facets.to_hdf5(
                    f"{Dataset.scan_folder}{Dataset.scan_name}.cxi")
            except AttributeError:
                print(
                    "Could not save facet extraction data, "
                    "run the analysis in the `Facets` tab first."
                )


def init_directories(
    scan_name,
    root_folder,
):
    """
    Create/touch following folders necessary for workflow:
        scan_folder: root_folder + scan_name + "/"
        preprocessing_folder: scan_folder + "preprocessing/"
        postprocessing_folder: scan_folder + "postprocessing/"
        data_folder: scan_folder + "data/"
        postprocessing_folder + "result_crystal/""
        postprocessing_folder + "result_lab_flat_sample/""
        postprocessing_folder + "result_laboratory/""

    :param scan_name: str, scan name, e.g. 'S1322'
    :param root_folder: root folder of the experiment
    """
    # Assign scan folder
    scan_folder = root_folder + "/" + scan_name + "/"
    print("Scan folder:", scan_folder)

    # Assign preprocessing folder
    preprocessing_folder = scan_folder + "preprocessing/"

    # Assign postprocessing folder
    postprocessing_folder = scan_folder + "postprocessing/"

    # Assign data folder
    data_folder = scan_folder + "data/"

    # Create final directory, if not yet existing
    if not os.path.isdir(root_folder):
        print(root_folder)
        full_path = ""
        for d in root_folder.split("/"):
            full_path += d + "/"
            try:
                os.mkdir(full_path)
            except (FileExistsError, PermissionError):
                pass

    # Scan directory
    try:
        os.mkdir(f"{scan_folder}")
        print(f"\tCreated {scan_folder}")
    except (FileExistsError, PermissionError):
        print(f"\t{scan_folder} exists")

    # /data directory
    try:
        os.mkdir(f"{data_folder}")
        print(f"\tCreated {data_folder}")
    except (FileExistsError, PermissionError):
        print(f"\t{data_folder} exists")

    # /preprocessing directory
    try:
        os.mkdir(f"{preprocessing_folder}")
        print(f"\tCreated {preprocessing_folder}")
    except (FileExistsError, PermissionError):
        print(f"\t{preprocessing_folder} exists")

    # /postprocessing directory
    try:
        os.mkdir(f"{postprocessing_folder}")
        print(f"\tCreated {postprocessing_folder}")
    except (FileExistsError, PermissionError):
        print(f"\t{postprocessing_folder} exists")

    # Subfolders to avoid bog
    for d in [
        "result_crystal",
        "result_lab_flat_sample",
        "result_laboratory"
    ]:
        try:
            os.mkdir(f"{postprocessing_folder}{d}")
            print(f"\tCreated {preprocessing_folder}{d}")
        except (FileExistsError, PermissionError):
            pass


def rotate_sixs_data(
    path_to_nxs_data
):
    """
    Python script to rotate the data when using the vertical configuration.
    Should work on a copy of the data !! Never use the OG data !!

    :param path_to_nxs_data: absolute path to nexus file
    """
    # Define save folder
    save_folder = os.path.dirname(path_to_nxs_data)

    # Check if already rotated
    with h5py.File(path_to_nxs_data, "a") as f:
        try:
            f.create_dataset("rotation", data=True)
            data_already_rotated = False
        except (ValueError, RuntimeError):
            data_already_rotated = f['rotation'][...]

    if not data_already_rotated:
        hash_print("Rotating SIXS data ...")
        with tb.open_file(path_to_nxs_data, "a") as f:
            # Get data
            try:
                # Omega scan
                data_og = f.root.com.scan_data.data_02[:]
                index = 2
                if np.ndim(data_og) is 1:
                    data_og = f.root.com.scan_data.data_10[:]
                    index = 10
                # Mu scan
                print("Calling merlin the enchanter in SBS...")
                scan_type = "SBS"
            except tb.NoSuchNodeError:
                try:
                    data_og = f.root.com.scan_data.self_image[:]
                    print("Calling merlin the enchanter in FLY...")
                    scan_type = "FLY"
                except tb.NoSuchNodeError:
                    print("This data does not result from Merlin :/")

            # Just an index for plotting schemes
            half = int(data_og.shape[0] / 2)

            # Transpose and flip lr data
            data = np.transpose(data_og, axes=(0, 2, 1))
            for idx in range(data.shape[0]):
                tmp = data[idx, :, :]
                data[idx, :, :] = np.fliplr(tmp)
            print("Data well rotated by 90°.")

            print("Saving example figures...", end="\n\n")
            plt.figure(figsize=(16, 9))
            plt.imshow(data_og[half, :, :], vmax=10)
            plt.xlabel('Delta')
            plt.ylabel('Gamma')
            plt.tight_layout()
            plt.savefig(save_folder + "/data_before_rotation.png")
            plt.close()

            plt.figure(figsize=(16, 9))
            plt.imshow(data[half, :, :], vmax=10)
            plt.xlabel('Gamma')
            plt.ylabel('Delta')
            plt.tight_layout()
            plt.savefig(save_folder + "/data_after_rotation.png")
            plt.close()

            # Overwrite data in copied file
            try:
                if scan_type is "SBS" and index is 2:
                    f.root.com.scan_data.data_02[:] = data
                elif scan_type is "SBS" and index is 10:
                    f.root.com.scan_data.data_10[:] = data
                elif scan_type is "FLY":
                    f.root.com.scan_data.test_image[:] = data
            except tb.NoSuchNodeError:
                print("Could not overwrite data ><")

    else:
        hash_print("Data already rotated ...")


def find_and_copy_raw_data(
    scan,
    sample_name,
    root_folder,
    data_dir,
):
    """
    If a file is found:
        - template_imagefile parameter updated to match it for bcdi scripts
        - a copy of the raw data file is saved in scan_folder + "data/"
        - data_dir parameter is changed to scan_folder + "data/" to work with
          the copy of the raw data file
    This method allows us not to work with the original data of SixS, since we
    need to rotate the data when working with the vertical configuration.

    :param scan: int, scan number
    :param sample_name: str, sample name, e.g. 'S'
    :param root_folder: root folder of the experiment
    :param data_dir: directory with all the raw data

    returns:
    :template_imagefile: empty string if no file or string updated to match the
     file found
    :data_dir: updated
    :param path_to_sixs_data: absolute path to nexus file to have metadata
     access
    """
    # Assign scan folder
    scan_folder = root_folder + "/" + sample_name + str(scan) + "/"
    path_to_nxs_data = ""
    template_imagefile = ""

    # Get path_to_nxs_data from data in data_dir
    try:
        # Try and find a mu scan
        path_to_nxs_data = glob.glob(f"{data_dir}*mu*{scan}*")[0]
    except IndexError:
        try:
            # Try and find an omega scan
            path_to_nxs_data = glob.glob(f"{data_dir}*omega*{scan}*")[0]
        except IndexError:
            print("Could not find data, please specify template.")

    # Get template_imagefile from path_to_nxs_data
    if path_to_nxs_data != "":
        try:
            print("File path:", path_to_nxs_data)
            template_imagefile = os.path.basename(path_to_nxs_data).split(
                "%05d" % scan)[0] + "%05d.nxs"
            print(f"File template: {template_imagefile}\n\n")

        except (IndexError, AttributeError):
            pass

        # Move data file to scan_folder + "data/"
        try:
            shutil.copy2(path_to_nxs_data, scan_folder + "data/")
            print(f"Copied {path_to_nxs_data} to {data_dir}")

            # Change data_dir, only if copy successful
            data_dir = scan_folder + "data/"

            # Change path_to_nxs_data, only if copy successful
            path_to_nxs_data = data_dir + os.path.basename(path_to_nxs_data)

            # Rotate the data
            rotate_sixs_data(path_to_nxs_data)

        except (FileExistsError, PermissionError, shutil.SameFileError):
            print(f"File exists in {scan_folder}data/")

            # Change data_dir, since data already copied
            data_dir = scan_folder + "data/"

            # Change path_to_nxs_data, since data already copied
            path_to_nxs_data = data_dir + os.path.basename(path_to_nxs_data)

            # Rotate the data
            rotate_sixs_data(path_to_nxs_data)

        except (AttributeError, FileNotFoundError):
            print("Could not move the data file.")
            pass

    return template_imagefile, data_dir, path_to_nxs_data
