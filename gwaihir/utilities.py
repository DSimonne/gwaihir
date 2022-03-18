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

# Widgets
import ipywidgets as widgets
from ipywidgets import interact, Button, Layout, interactive, fixed
from IPython.display import display, Markdown, Latex, clear_output, Image

# gwaihir package
import gwaihir
from gwaihir.gui import gui_iterable

# bcdi package
from bcdi.preprocessing import ReadNxs3 as rd


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


def rotate_sixs_data(path_to_sixs_data):
    """
    Python script to rotate the data when using the vertical configuration.
    Should work on a copy of the data !! Never use the OG data !!

    :param path_to_sixs_data: absolute path to nexus file
    """
    # Define save folder
    save_folder = os.path.dirname(path_to_sixs_data)

    # Check if already rotated
    with h5py.File(path_to_sixs_data, "a") as f:
        try:
            f.create_dataset("rotation", data=True)
            data_already_rotated = False
        except (ValueError, RuntimeError):
            data_already_rotated = f['rotation'][...]

    if not data_already_rotated:
        hash_print("Rotating SIXS data ...")
        with tb.open_file(path_to_sixs_data, "a") as f:
            # Get data
            try:
                # if rocking_angle == "omega":
                data_og = f.root.com.scan_data.data_02[:]
                index = 2
                if np.ndim(data_og) == 1:
                    data_og = f.root.com.scan_data.data_10[:]
                    index = 10
                # elif rocking_angle == "mu":
                #     data_og = f.root.com.scan_data.merlin_image[:]
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
                if scan_type == "SBS" and index == 2:
                    f.root.com.scan_data.data_02[:] = data
                elif scan_type == "SBS" and index == 10:
                    f.root.com.scan_data.data_10[:] = data
                elif scan_type == "FLY":
                    f.root.com.scan_data.test_image[:] = data
            except tb.NoSuchNodeError:
                print("Could not overwrite data ><")

    else:
        hash_print("Data already rotated ...")


def find_move_sixs_data(
    scan,
    scan_name,
    root_folder,
    data_dir,
):
    """
    If a file is indeed found:
        template_imagefile parameter updated to match it
        copy of the file is saved in scan_folder + "data/"
        data_dir parameter is changed to scan_folder + "data/" to work with the
        copy of the original data file
    This method allows us not to work with the original data of SixS, since we
    need to rotate the data when working with the vertical configuration.

    :param scan: scan number
    :param scan_name: str, scan name, e.g. 'S1322'
    :param root_folder: root folder of the experiment
    :param data_dir: original data directory

    returns:
    :template_imagefile: empty string if no file or string updated to match the
     file found
    :data_dir: updated
    :path_to_sixs_data: empty string if no file or full path to file found
    """

    # Assign scan folder
    scan_folder = root_folder + "/" + scan_name + "/"
    path_to_sixs_data = ""
    template_imagefile = ""

    # Get path_to_sixs_data from data in data_dir
    try:
        # Try and find a mu scan
        path_to_sixs_data = glob.glob(f"{data_dir}*mu*{scan}*")[0]
    except IndexError:
        try:
            # Try and find an omega scan
            path_to_sixs_data = glob.glob(f"{data_dir}*omega*{scan}*")[0]
        except IndexError:
            print("Could not find data, please specify template.")

    # Get template_imagefile from path_to_sixs_data
    if path_to_sixs_data != "":
        try:
            print("File path:", path_to_sixs_data)
            template_imagefile = os.path.basename(path_to_sixs_data).split(
                "%05d" % scan)[0] + "%05d.nxs"
            print(f"File template: {template_imagefile}\n\n")

        except (IndexError, AttributeError):
            pass

        # Move data file to scan_folder + "data/"
        try:
            shutil.copy2(path_to_sixs_data, scan_folder + "data/")
            print(f"Copied {path_to_sixs_data} to {data_dir}")

            # Change data_dir, only if copy successful
            data_dir = scan_folder + "data/"

            # Change path_to_sixs_data, only if copy successful
            path_to_sixs_data = data_dir + os.path.basename(path_to_sixs_data)

            # Rotate the data
            rotate_sixs_data(path_to_sixs_data)

        except (FileExistsError, PermissionError, shutil.SameFileError):
            print(f"File exists in {scan_folder}data/")

            # Change data_dir, only if copy successful
            data_dir = scan_folder + "data/"

            # Change path_to_sixs_data, only if copy successful
            path_to_sixs_data = data_dir + os.path.basename(path_to_sixs_data)

            # Rotate the data
            rotate_sixs_data(path_to_sixs_data)

        except (AttributeError, FileNotFoundError):
            print("Could not move the data file.")
            pass

    return template_imagefile, data_dir


def filter_reconstructions(
    folder,
    nb_run_keep,
    nb_run=None,
    filter_criteria="LLK"
):
    """
    Filter the phase retrieval output depending on a given parameter,
    for now only LLK and standard deviation are available. This allows the
    user to run a lot of reconstructions but to then automatically keep the
    "best" ones, according to this parameter. filter_criteria can take the
    values "LLK" or "standard_deviation" If you filter based on both, the
    function will filter nb_run_keep/2 files by the first criteria, and the
    remaining files by the second criteria.

    The parameters are specified in the phase retrieval tab, and
    their values saved through self.initialize_phase_retrieval()

    .param folder: parent folder to cxi files
    :param nb_run_keep: number of best run results to keep in the end,
     according to filter_criteria.
    :param nb_run: number of times to run the optimization, if None, equal
     to nb of files detected
    :param filter_criteria: default "LLK"
     criteria onto which the best solutions will be chosen
     possible values are ("standard_deviation", "LLK",
     "standard_deviation_LLK", "LLK_standard_deviation")
    """

    # Sorting functions depending on filtering criteria
    def filter_by_std(cxi_files, nb_run_keep):
        """Use the standard deviation of the reconstructed object as
        filtering criteria.

        The lowest standard deviations are best.
        """
        # Keep filtering criteria of reconstruction modules in dictionnary
        filtering_criteria_value = {}

        print(
            "\n#########################################################################################"
        )
        print("Computing standard deviation of object modulus for scans:")
        for filename in cxi_files:
            print(f"\t{os.path.basename(filename)}")
            with tb.open_file(filename, "r") as f:
                data = f.root.entry_1.image_1.data[:]
                amp = np.abs(data)
                # Skip values near 0
                meaningful_data = amp[amp > 0.05 * amp.max()]
                filtering_criteria_value[filename] = np.std(amp)

        # Sort files
        sorted_dict = sorted(
            filtering_criteria_value.items(),
            key=operator_lib.itemgetter(1)
        )

        # Remove files
        print("\nRemoving scans:")
        for filename, filtering_criteria_value in sorted_dict[nb_run_keep:]:
            print(f"\t{os.path.basename(filename)}")
            os.remove(filename)
        print(
            "#########################################################################################\n"
        )

    def filter_by_LLK(cxi_files, nb_run_keep):
        """Use the free log-likelihood values of the reconstructed object
        as filtering criteria.

        The lowest standard deviations are best. See PyNX for
        details
        """
        # Keep filtering criteria of reconstruction modules in dictionnary
        filtering_criteria_value = {}

        print(
            "\n#########################################################################################"
        )
        print("Extracting LLK value (poisson statistics) for scans:")
        for filename in cxi_files:
            print(f"\t{os.path.basename(filename)}")
            with tb.open_file(filename, "r") as f:
                llk = f.root.entry_1.image_1.process_1.\
                    results.llk_poisson[...]
                filtering_criteria_value[filename] = llk

        # Sort files
        sorted_dict = sorted(
            filtering_criteria_value.items(),
            key=operator_lib.itemgetter(1)
        )

        # Remove files
        print("\nRemoving scans:")
        for filename, filtering_criteria_value in sorted_dict[nb_run_keep:]:
            print(f"\t{os.path.basename(filename)}")
            os.remove(filename)
        print(
            "#########################################################################################\n"
        )

    # Main function supporting different cases
    try:
        print(
            "\n#########################################################################################"
        )
        print(f"Iterating on files matching:")
        print(f"\t{folder}/*LLK*.cxi")
        cxi_files = sorted(glob.glob(f"{folder}/result_scan*LLK*.cxi"))
        print(
            "#########################################################################################\n"
        )

        if cxi_files == []:
            print(f"No *LLK*.cxi files in {folder}/result_scan*LLK*.cxi")

        else:
            # only standard_deviation
            if filter_criteria == "standard_deviation":
                filter_by_std(cxi_files, nb_run_keep)

            # only LLK
            elif filter_criteria == "LLK":
                filter_by_LLK(cxi_files, nb_run_keep)

            # standard_deviation then LLK
            elif filter_criteria == "standard_deviation_LLK":
                if nb_run == None:
                    nb_run = len(cxi_files)

                filter_by_std(cxi_files, nb_run_keep +
                              (nb_run - nb_run_keep) // 2)

                hash_print("Iterating on remaining files.")

                cxi_files = sorted(
                    glob.glob(f"{folder}/result_scan*LLK*.cxi"))

                if cxi_files == []:
                    print(
                        f"No *LLK*.cxi files remaining in \
                        {folder}/result_scan*LLK*.cxi")
                else:
                    filter_by_LLK(cxi_files, nb_run_keep)

            # LLK then standard_deviation
            elif filter_criteria == "LLK_standard_deviation":
                if nb_run == None:
                    nb_run = len(cxi_files)

                filter_by_LLK(cxi_files, nb_run_keep +
                              (nb_run - nb_run_keep) // 2)

                hash_print("Iterating on remaining files.")

                cxi_files = sorted(
                    glob.glob(f"{folder}/result_scan*LLK*.cxi"))

                if cxi_files == []:
                    print(
                        f"No *LLK*.cxi files remaining in \
                        {folder}/result_scan*LLK*.cxi")
                else:
                    filter_by_std(cxi_files, nb_run_keep)

            else:
                hash_print("No filtering")
    except KeyboardInterrupt:
        hash_print("File filtering stopped by user ...")


def extract_metadata(
    scan_nb,
    metadata_file,
    gwaihir_dataset=None,
    metadata_csv_file=None,
):
    """
    Extract meaningful data from bcdi script output files and saves them
    in a csv file as well as in the Dataset object to allow comparison.

    :param scan_nb: int, nb of scan, used for indexing in csv file.
    :param metadata_file: absolute path to metadata file (.h5) created by
     bcdi.preprocessing_BCDI.py script
    :param gwaihir_dataset: Dataset object in which the metadata is saved,
     optionnal
    :param metadata_csv_file: csv file in which the metadata is saved. If
     None, defaulted to gwaihir_dataset.root_folder + "/metadata.csv"
    """

    # Open file
    with tb.open_file(metadata_file, "r") as f:

        # Save metadata in a pd.DataFrame
        temp_df = pd.DataFrame([[
            scan_nb,
            f.root.output.q[...][0],
            f.root.output.q[...][1],
            f.root.output.q[...][2],
            f.root.output.qnorm[...],
            f.root.output.dist_plane[...],
            f.root.output.bragg_inplane[...],
            f.root.output.bragg_outofplane[...],
            f.root.output.bragg_peak[...],
        ]],
            columns=[
                "scan",
                "qx",
                "qy",
                "qz",
                "q_norm",
                "d_hkl",
                "inplane_angle",
                "out_of_plane_angle",
                "bragg_peak",
        ])

        # Extra metadata that is not always computed
        try:
            temp_df["COM_rocking_curve"] = f.root.output.COM_rocking_curve[...]
            temp_df["interp_fwhm"] = f.root.output.interp_fwhm[...]

            tilt_angle = np.round(
                np.mean(f.root.output.tilt_values[...][1:]
                        - f.root.output.tilt_values[...][:-1]),
                4)
            temp_df["tilt_angle"] = tilt_angle

        except tb.NoSuchNodeError:
            # No angle correction during preprocess
            pass

        # Save metadata in the Dataset object
        if isinstance(gwaihir_dataset, gwaihir.gui.gui_iterable.Dataset):

            gwaihir_dataset.bragg_peak = f.root.output.bragg_peak[...]
            gwaihir_dataset.q = f.root.output.q[...]
            gwaihir_dataset.qnorm = f.root.output.qnorm[...]
            gwaihir_dataset.dist_plane = f.root.output.dist_plane[...]
            gwaihir_dataset.bragg_inplane = f.root.output.bragg_inplane[...]
            gwaihir_dataset.bragg_outofplane = f.root.output.bragg_outofplane[...]

            # Extra metadata that is not always computed
            try:
                gwaihir_dataset.tilt_values = f.root.output.tilt_values[...]
                gwaihir_dataset.rocking_curve = f.root.output.rocking_curve[...]
                gwaihir_dataset.interp_tilt = f.root.output.interp_tilt[...]
                gwaihir_dataset.interp_curve = f.root.output.interp_curve[...]
                gwaihir_dataset.detector_data_COM = f.root.output.detector_data_COM[...]
                gwaihir_dataset.COM_rocking_curve = f.root.output.COM_rocking_curve[...]
                gwaihir_dataset.interp_fwhm = f.root.output.interp_fwhm[...]
                gwaihir_dataset.tilt_angle = tilt_angle

            except tb.NoSuchNodeError:
                # No angle correction during preprocess
                pass

            # Extra metadata for SixS to save in df
            if gwaihir_dataset.beamline == "SIXS_2019":
                data = rd.DataSet(gwaihir_dataset.path_to_sixs_data)
                try:
                    temp_df["x"] = data.x[0]
                    temp_df["y"] = data.y[0]
                    temp_df["z"] = data.z[0]
                    temp_df["mu"] = data.mu[0]
                    temp_df["delta"] = data.delta[0]
                    temp_df["omega"] = data.omega[0]
                    temp_df["gamma"] = data.gamma[0]
                    temp_df["gamma-mu"] = data.gamma[0] - data.mu[0]
                    temp_df["step_size"] = (
                        data.mu[-1] - data.mu[-0]) / len(data.mu)
                    temp_df["integration_time"] = data.integration_time[0]
                    temp_df["steps"] = len(data.integration_time)
                except AttributeError:
                    print("Could not extract metadata from SixS file")

    # Save in a csv file
    try:
        # Load old file
        df = pd.read_csv(metadata_csv_file)

        # Replace old data linked to this scan
        indices = df[df['scan'] == scan_nb].index
        df.drop(indices, inplace=True)
        result = pd.concat([df, temp_df])

        # Save
        display(result.head())
        result.to_csv(metadata_csv_file, index=False)
        hash_print(f"Saved logs in {metadata_csv_file}")

    except (FileNotFoundError, ValueError):
        # Create file
        metadata_csv_file = os.getcwd() + "/metadata.csv"

        # Save
        display(temp_df.head())
        temp_df.to_csv(metadata_csv_file, index=False)
        hash_print(f"Saved logs in {metadata_csv_file}")


def create_yaml_file(fname, **kwargs):
    """
    Create yaml file storing all keywords arguments given in input.
    Used for bcdi scripts.

    :param fname: path to created yaml file
    :param kwargs: kwargs to store in file
    """
    config_file = []

    for k, v in kwargs.items():
        if isinstance(v, str):
            config_file.append(f"{k}: \"{v}\"")
        elif isinstance(v, tuple):
            if v:
                config_file.append(f"{k}: {list(v)}")
            else:
                config_file.append(f"{k}: None")
        elif isinstance(v, np.ndarray):
            config_file.append(f"{k}: {list(v)}")
        elif isinstance(v, list):
            if v:
                config_file.append(f"{k}: {v}")
            else:
                config_file.append(f"{k}: None")
        else:
            config_file.append(f"{k}: {v}")

    file = os.path.basename(fname)
    directory = fname.strip(file)
    # Create directory
    if not os.path.isdir(directory):
        full_path = ""
        for d in directory.split("/"):
            full_path += d + "/"
            try:
                os.mkdir(full_path)
            except (FileExistsError, PermissionError):
                pass

    with open(fname, "w") as v:
        for line in config_file:
            v.write(line + "\n")


def hash_print(
    string_to_print,
    hash_line_before=True,
    hash_line_after=True,
    new_line_before=True,
    new_line_after=True
):
    """Print string with hashtag lines before and after"""
    if new_line_before:
        print()
    hash_line = "#" * len(string_to_print)
    if hash_line_before:
        print(hash_line)
    print(string_to_print)
    if hash_line_after:
        print(hash_line)

    if new_line_after:
        print()
