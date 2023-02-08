import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import glob
import os
import h5py
import shutil
from IPython.display import display, clear_output
import ipywidgets as widgets

import gwaihir.dataset as gd

try:
    from gwaihir.controller.control_phase_retrieval import save_cdi_operator_as_cxi, \
        initialize_cdi_operator
    pynx_import_success = True
except ModuleNotFoundError:
    pynx_import_success = False


def init_startup_tab(
    interface,
    unused_label_scan,
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
    Mandatory to run before any other step, updates the directories and
    the creates the Dataset instance

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
    """
    if run_dir_init:
        # Create Dataset Class
        Dataset = gd.Dataset(
            scan=scan,
            sample_name=sample_name,
            data_dir=data_dir,
            root_folder=root_folder,
        )

        # Save Dataset as interface attribute
        interface.Dataset = Dataset

        # Start to assign attributes
        Dataset.comment = comment
        Dataset.debug = debug
        Dataset.scan_name = Dataset.sample_name + str(Dataset.scan)

        # Save backend used for plotting
        interface.matplotlib_backend = matplotlib_backend

        # Update the directory structure
        print("Updating directories ...")
        Dataset.scan_folder, interface.preprocessing_folder, interface.postprocessing_folder = init_directories(
            scan_name=Dataset.scan_name,
            root_folder=Dataset.root_folder,
        )

        # Try and find SixS data
        template_imagefile, Dataset.data_dir, Dataset.path_to_nxs_data = find_and_copy_raw_data(
            scan=Dataset.scan,
            scan_folder=Dataset.scan_folder,
            data_dir=Dataset.data_dir,
        )

        # Save template_imagefile in GUI
        if template_imagefile != "":
            interface.TabDetector.template_imagefile.value\
                = template_imagefile

        # Refresh folders
        interface.root_folder_handler(
            change=Dataset.scan_folder
        )

        # PyNX folder
        interface.TabPhaseRetrieval.parent_folder.value\
            = interface.preprocessing_folder
        interface.TabPhaseRetrieval.pynx_folder_handler(
            change=interface.preprocessing_folder
        )

        # Plot folder
        interface.TabPlotData.parent_folder.value = interface.preprocessing_folder
        interface.TabPlotData.plot_folder_handler(
            change=interface.preprocessing_folder
        )

        # Strain folder, refresh values
        interface.TabPostprocess.strain_folder.value = interface.preprocessing_folder
        interface.TabPostprocess.strain_folder_handler(
            change=interface.preprocessing_folder
        )

        # Facet folder, refresh values
        interface.TabFacet.parent_folder.value = interface.postprocessing_folder
        interface.TabFacet.vtk_file_handler(
            change=interface.postprocessing_folder)

        # Allow to save cxi files if pynx is imported
        if pynx_import_success:
            # Button to save data
            button_save_as_cxi = widgets.Button(
                description="Save work as .cxi file",
                continuous_update=False,
                button_style='',
                layout=widgets.Layout(width='40%'),
                style={'description_width': 'initial'},
                icon='step-forward')

            display(button_save_as_cxi)

            @ button_save_as_cxi.on_click
            def action_button_save_as_cxi(selfbutton):
                """Create button to save Dataset object as .cxi file."""
                clear_output(True)
                display(button_save_as_cxi)
                print("Saving data ...")
                save_data_analysis_workflow(interface.Dataset)

    elif not run_dir_init:
        print("Cleared window.")
        clear_output(True)


def save_data_analysis_workflow(Dataset):
    """Save the Dataset instance as a `.cxi` file."""
    # Path to the final .cxi file
    final_cxi_file = "{}{}.cxi".format(
        Dataset.scan_folder,
        Dataset.scan_name,
    )

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

    if cdi is None:
        raise TypeError("Could not initiliaze the cdi object.")

    # Path to the .cxi file with the raw data
    raw_data_cxi_file = "{}/preprocessing/{}".format(
        Dataset.scan_folder,
        Dataset.iobs.split("/")[-1].replace(".npz", ".cxi"),
    )

    # Check if the raw data file was already created during phase retrieval
    if not os.path.isfile(raw_data_cxi_file):
        save_cdi_operator_as_cxi(
            gwaihir_dataset=Dataset,
            cdi_operator=cdi,
            path_to_cxi=raw_data_cxi_file,
        )

    # Save all the data in a single .cxi file
    Dataset.to_cxi(
        raw_data_cxi_file=raw_data_cxi_file,
        final_cxi_file=final_cxi_file,
    )

    # Save the Facets analysis output data as well
    print(
        "\n#######################################"
        "########################################\n")
    try:
        Dataset.Facets.to_hdf5(
            f"{Dataset.scan_folder}{Dataset.scan_name}.cxi")
        print("Saved Facets class data")
    except AttributeError:
        print(
            "Could not append Facet data, "
            "run the analysis in the `Facets` tab first."
        )
    print(
        "\n#######################################"
        "########################################\n")


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

    return: scan_folder, preprocessing_folder, postprocessing_folder
    """
    # Assign scan folder
    scan_folder = root_folder + scan_name + "/"
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

    return scan_folder, preprocessing_folder, postprocessing_folder


def find_and_copy_raw_data(
    scan,
    scan_folder,
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
    :param scan_folder: folder in which the data is stored
    :param data_dir: directory with all the raw data

    returns:
    :template_imagefile: empty string if no file or string updated to match the
     file found
    :data_dir: updated
    :param path_to_sixs_data: absolute path to nexus file to have metadata
     access
    """
    path_to_nxs_data = ""
    template_imagefile = ""

    # Get path_to_nxs_data from data in data_dir
    try:
        # Try and find a mu scan
        path_to_nxs_data = glob.glob(f"{data_dir}*mu*{scan}*.nxs")[0]
    except IndexError:
        try:
            # Try and find an omega scan
            path_to_nxs_data = glob.glob(f"{data_dir}*omega*{scan}*.nxs")[0]
        except IndexError:
            print("Could not find data, please specify template.")

    # Get template_imagefile from path_to_nxs_data
    if path_to_nxs_data != "":
        try:
            print("File path:", path_to_nxs_data)
            template_imagefile = os.path.basename(path_to_nxs_data).split(
                "%05d" % scan)[0] + "%05d.nxs"  # Does not work at crystal
            print(f"File template: {template_imagefile}\n\n")

        except (IndexError, AttributeError):
            pass

        # Move data file to scan_folder + "data/"
        try:
            if not os.path.isfile(scan_folder + "data/" + os.path.basename(
                path_to_nxs_data)
            ):
                shutil.copy2(path_to_nxs_data, scan_folder + "data/")
                print(f"Copied {path_to_nxs_data} to {scan_folder}data/")
            else:
                print("{} already exists in {}data/".format(
                    os.path.basename(path_to_nxs_data),
                    scan_folder
                )
                )

            # Change data_dir, only if copy successful
            data_dir = scan_folder + "data/"

            # Change path_to_nxs_data, only if copy successful
            path_to_nxs_data = data_dir + os.path.basename(path_to_nxs_data)

        except (AttributeError, FileNotFoundError):
            print("Could not move the data file.")

    return template_imagefile, data_dir, path_to_nxs_data
