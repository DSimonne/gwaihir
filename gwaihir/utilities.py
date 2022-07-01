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
                    # Mu scan
                    data_og = f.root.com.scan_data.data_10[:]
                    index = 10
                print("Calling Merlin the enchanter in SBS...")
                scan_type = "SBS"
            except tb.NoSuchNodeError:
                try:
                    data_og = f.root.com.scan_data.self_image[:]
                    print("Calling Merlin the enchanter in FLY...")
                    scan_type = "FLY1"
                except tb.NoSuchNodeError:
                    try:
                        data_og = f.root.com.scan_data.test_image[:]
                        print("Calling Merlin the enchanter in FLY...")
                        scan_type = "FLY2"
                    except:
                        raise TypeError("Unsupported data")

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
                elif scan_type is "FLY1":
                    f.root.com.scan_data.self_image[:] = data
                elif scan_type is "FLY2":
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
            "\n###################"
            "#####################"
            "#####################"
            "#####################"
        )
        print("Computing standard deviation of object modulus for scans:")
        for filename in cxi_files:
            print(f"\t{os.path.basename(filename)}")
            with tb.open_file(filename, "r") as f:
                data = f.root.entry_1.image_1.data[:]
                amp = np.abs(data)
                # Skip values near 0
                meaningful_data = amp[amp > 0.05 * amp.max()]
                filtering_criteria_value[filename] = np.std(
                    meaningful_data
                )

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
            "#####################"
            "#####################"
            "#####################"
            "###################\n"
        )

    def filter_by_LLK(cxi_files, nb_run_keep):
        """
        Use the free log-likelihood values of the reconstructed object
        as filtering criteria.

        The lowest standard deviations are best. See PyNX for
        details
        """
        # Keep filtering criteria of reconstruction modules in dictionnary
        filtering_criteria_value = {}

        print(
            "\n###################"
            "#####################"
            "#####################"
            "#####################"
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
            "#####################"
            "#####################"
            "#####################"
            "###################\n"
        )

    # Main function supporting different cases
    try:
        print(
            "\n###################"
            "#####################"
            "#####################"
            "#####################"
        )
        print("Iterating on files matching:")
        print(f"\t{folder}/*LLK*.cxi")
        cxi_files = sorted(glob.glob(f"{folder}/result_scan*LLK*.cxi"))
        print(
            "#####################"
            "#####################"
            "#####################"
            "###################\n"
        )

        if cxi_files == []:
            print(f"No *LLK*.cxi files in {folder}/result_scan*LLK*.cxi")

        else:
            # only standard_deviation
            if filter_criteria is "standard_deviation":
                filter_by_std(cxi_files, nb_run_keep)

            # only LLK
            elif filter_criteria is "LLK":
                filter_by_LLK(cxi_files, nb_run_keep)

            # standard_deviation then LLK
            elif filter_criteria is "standard_deviation_LLK":
                if nb_run is None:
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
            elif filter_criteria is "LLK_standard_deviation":
                if nb_run is None:
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
    :param metadata_csv_file: csv file in which the metadata is saved.
     If None, defaulted to os.getcwd() + "/metadata.csv"
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
            if gwaihir_dataset.beamline is "SIXS_2019":
                data = rd.DataSet(gwaihir_dataset.path_to_nxs_data)
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


def initialize_cdi_operator(
    iobs,
    mask=None,
    support=None,
    obj=None,
    rebin=(1, 1, 1),
    auto_center_resize=False,
    max_size=None,
    wavelength=None,
    pixel_size_detector=None,
    detector_distance=None,
):
    """
    Initialize the cdi operator by processing the possible inputs:
        - iobs
        - mask
        - support
        - obj
    Will also crop and center the data if specified.

    :param iobs: path to npz or npy that stores this array
    :param mask: path to npz or npy that stores this array
    :param support: path to npz or npy that stores this array
    :param obj: path to npz or npy that stores this array
    :param rebin: e.g. (1, 1, 1), applied to all the arrays
    :param auto_center_resize:
    :param max_size:
    :param wavelength:
    :param pixel_size_detector:
    :param detector_distance:

    return: cdi operator
    """
    if iobs in ("", None) or not os.path.isfile(iobs):
        # Dataset.iobs = None
        iobs = None
        print("At least iobs must exist.")
        return None  # stop function directly

    if iobs.endswith(".npy"):
        iobs = np.load(iobs)
        print("\tCXI input: loading data")
    elif iobs.endswith(".npz"):
        try:
            iobs = np.load(iobs)["data"]
            print("\tCXI input: loading data")
        except KeyError:
            print("\t\"data\" key does not exist.")
            raise KeyboardInterrupt

    if rebin != (1, 1, 1):
        iobs = bin_data(iobs, rebin)

    # fft shift
    iobs = fftshift(iobs)

    if mask not in ("", None) or not os.path.isfile(mask):
        if mask.endswith(".npy"):
            mask = np.load(mask).astype(np.int8)
            nb = mask.sum()
            print("\tCXI input: loading mask, with %d pixels masked (%6.3f%%)" % (
                nb, nb * 100 / mask.size))
        elif mask.endswith(".npz"):
            try:
                mask = np.load(mask)[
                    "mask"].astype(np.int8)
                nb = mask.sum()
                print("\tCXI input: loading mask, with %d pixels masked (%6.3f%%)" % (
                    nb, nb * 100 / mask.size))
            except KeyError:
                print("\t\"mask\" key does not exist.")

        if rebin != (1, 1, 1):
            mask = bin_data(mask, rebin)

        # fft shift
        mask = fftshift(mask)

    else:
        mask = None

    print(support)
    if support not in ("", None) or not os.path.isfile(support):
        if support.endswith(".npy"):
            support = np.load(support)
            print("\tCXI input: loading support")
        elif support.endswith(".npz"):
            try:
                support = np.load(support)["data"]
                print("\tCXI input: loading support")
            except (FileNotFoundError, ValueError):
                print("\tFile not supported or does not exist.")
            except KeyError:
                print("\t\"data\" key does not exist.")
                try:
                    support = np.load(support)["support"]
                    print("\tCXI input: loading support")
                except KeyError:
                    print("\t\"support\" key does not exist.")
                    try:
                        support = np.load(support)["obj"]
                        print("\tCXI input: loading support")
                    except KeyError:
                        print(
                            "\t\"obj\" key does not exist."
                            "\t--> Could not load support array."
                        )

        if rebin != (1, 1, 1):
            support = bin_data(support, rebin)

        # fft shift
        try:
            support = fftshift(support)
        except ValueError:
            support = None

    else:
        support = None

    print(obj)
    if obj not in ("", None) or not os.path.isfile(obj):
        if obj.endswith(".npy"):
            obj = np.load(obj)
            print("\tCXI input: loading object")
        elif obj.endswith(".npz"):
            try:
                obj = np.load(obj)["data"]
                print("\tCXI input: loading object")
            except KeyError:
                print("\t\"data\" key does not exist.")

        if rebin != (1, 1, 1):
            obj = bin_data(obj, rebin)

        # fft shift
        try:
            obj = fftshift(obj)
        except ValueError:
            obj = None

    else:
        obj = None

    # Center and crop data
    if auto_center_resize:
        if iobs.ndim is 3:
            nz0, ny0, nx0 = iobs.shape

            # Find center of mass
            z0, y0, x0 = center_of_mass(iobs)
            print("Center of mass at:", z0, y0, x0)
            iz0, iy0, ix0 = int(round(z0)), int(
                round(y0)), int(round(x0))

            # Max symmetrical box around center of mass
            nx = 2 * min(ix0, nx0 - ix0)
            ny = 2 * min(iy0, ny0 - iy0)
            nz = 2 * min(iz0, nz0 - iz0)

            if max_size is not None:
                nx = min(nx, max_size)
                ny = min(ny, max_size)
                nz = min(nz, max_size)

            # Crop data to fulfill FFT size requirements
            nz1, ny1, nx1 = smaller_primes(
                (nz, ny, nx), maxprime=7, required_dividers=(2,))

            print("Centering & reshaping data: (%d, %d, %d) -> \
                (%d, %d, %d)" % (nz0, ny0, nx0, nz1, ny1, nx1))
            iobs = iobs[
                iz0 - nz1 // 2:iz0 + nz1 // 2,
                iy0 - ny1 // 2:iy0 + ny1 // 2,
                ix0 - nx1 // 2:ix0 + nx1 // 2]
            if mask is not None:
                mask = mask[
                    iz0 - nz1 // 2:iz0 + nz1 // 2,
                    iy0 - ny1 // 2:iy0 + ny1 // 2,
                    ix0 - nx1 // 2:ix0 + nx1 // 2]
                print("Centering & reshaping mask: (%d, %d, %d) -> \
                    (%d, %d, %d)" % (nz0, ny0, nx0, nz1, ny1, nx1))

        else:
            ny0, nx0 = iobs.shape

            # Find center of mass
            y0, x0 = center_of_mass(iobs)
            iy0, ix0 = int(round(y0)), int(round(x0))
            print("Center of mass (rounded) at:", iy0, ix0)

            # Max symmetrical box around center of mass
            nx = 2 * min(ix0, nx0 - ix0)
            ny = 2 * min(iy0, ny0 - iy0)
            if max_size is not None:
                nx = min(nx, max_size)
                ny = min(ny, max_size)
                nz = min(nz, max_size)

            # Crop data to fulfill FFT size requirements
            ny1, nx1 = smaller_primes(
                (ny, nx), maxprime=7, required_dividers=(2,))

            print("Centering & reshaping data: (%d, %d) -> (%d, %d)" %
                  (ny0, nx0, ny1, nx1))
            iobs = iobs[iy0 - ny1 // 2:iy0 + ny1 //
                        2, ix0 - nx1 // 2:ix0 + nx1 // 2]

            if mask is not None:
                mask = mask[iy0 - ny1 // 2:iy0 + ny1 //
                            2, ix0 - nx1 // 2:ix0 + nx1 // 2]

    # Create cdi object with data and mask, load the main parameters
    cdi = CDI(
        iobs,
        support=support,
        obj=obj,
        mask=mask,
        wavelength=wavelength,
        pixel_size_detector=pixel_size_detector,
        detector_distance=detector_distance,
    )

    return cdi


def save_cdi_operator_as_cxi(
    gwaihir_dataset,
    cdi_operator,
    path_to_cxi,
):
    """
    We need to create a dictionnary with the parameters to save in the
    cxi file.

    :param cdi_operator: cdi object
     created with PyNX
    :param path_to_cxi: path to future cxi data
     Below are parameters that are saved in the cxi file
        - filename: the file name to save the data to
        - iobs: the observed intensity
        - wavelength: the wavelength of the experiment (in meters)
        - detector_distance: the detector distance (in meters)
        - pixel_size_detector: the pixel size of the detector (in meters)
        - mask: the mask indicating valid (=0) and bad pixels (>0)
        - sample_name: optional, the sample name
        - experiment_id: the string identifying the experiment, e.g.:
          'HC1234: Siemens star calibration tests'
        - instrument: the string identifying the instrument, e.g.:
         'ESRF id10'
        - iobs_is_fft_shifted: if true, input iobs (and mask if any)
        have their origin in (0,0[,0]) and will be shifted back to
        centered-versions before being saved.
        - process_parameters: a dictionary of parameters which will
          be saved as a NXcollection

    :return: Nothing, a CXI file is created.
    """
    cdi_parameters = params
    cdi_parameters["data"] = gwaihir_dataset.iobs
    cdi_parameters["wavelength"] = gwaihir_dataset.wavelength
    cdi_parameters["detector_distance"] = gwaihir_dataset.detector_distance
    cdi_parameters["pixel_size_detector"] = gwaihir_dataset.pixel_size_detector
    cdi_parameters["wavelength"] = gwaihir_dataset.wavelength
    cdi_parameters["verbose"] = gwaihir_dataset.verbose
    cdi_parameters["live_plot"] = gwaihir_dataset.live_plot
    # cdi_parameters["gpu"] = gwaihir_dataset.gpu
    cdi_parameters["auto_center_resize"] = gwaihir_dataset.auto_center_resize
    # cdi_parameters["roi_user"] = gwaihir_dataset.roi_user
    # cdi_parameters["roi_final"] = gwaihir_dataset.roi_final
    cdi_parameters["nb_run"] = gwaihir_dataset.nb_run
    cdi_parameters["max_size"] = gwaihir_dataset.max_size
    # cdi_parameters["data2cxi"] = gwaihir_dataset.data2cxi
    cdi_parameters["output_format"] = "cxi"
    cdi_parameters["mask"] = gwaihir_dataset.mask
    cdi_parameters["support"] = gwaihir_dataset.support
    # cdi_parameters["support_autocorrelation_threshold"]\
    # = gwaihir_dataset.support_autocorrelation_threshold
    cdi_parameters["support_only_shrink"] = gwaihir_dataset.support_only_shrink
    cdi_parameters["object"] = gwaihir_dataset.obj
    cdi_parameters["support_update_period"] = gwaihir_dataset.support_update_period
    cdi_parameters["support_smooth_width_begin"] = gwaihir_dataset.support_smooth_width[0]
    cdi_parameters["support_smooth_width_end"] = gwaihir_dataset.support_smooth_width[1]
    # cdi_parameters["support_smooth_width_relax_n"] = \
    # gwaihir_dataset.support_smooth_width_relax_n
    # cdi_parameters["support_size"] = gwaihir_dataset.support_size
    cdi_parameters["support_threshold"] = gwaihir_dataset.support_threshold
    cdi_parameters["positivity"] = gwaihir_dataset.positivity
    cdi_parameters["beta"] = gwaihir_dataset.beta
    cdi_parameters["crop_output"] = 0
    cdi_parameters["rebin"] = gwaihir_dataset.rebin
    # cdi_parameters["support_update_border_n"] \
    # = gwaihir_dataset.support_update_border_n
    # cdi_parameters["support_threshold_method"] \
    # = gwaihir_dataset.support_threshold_method
    cdi_parameters["support_post_expand"] = gwaihir_dataset.support_post_expand
    cdi_parameters["psf"] = gwaihir_dataset.psf
    # cdi_parameters["note"] = gwaihir_dataset.note
    try:
        cdi_parameters["instrument"] = gwaihir_dataset.beamline
    except AttributeError:
        cdi_parameters["instrument"] = None
    cdi_parameters["sample_name"] = gwaihir_dataset.sample_name
    # cdi_parameters["fig_num"] = gwaihir_dataset.fig_num
    # cdi_parameters["algorithm"] = gwaihir_dataset.algorithm
    cdi_parameters["zero_mask"] = "auto"
    cdi_parameters["nb_run_keep"] = gwaihir_dataset.nb_run_keep
    # cdi_parameters["save"] = gwaihir_dataset.save
    # cdi_parameters["gps_inertia"] = gwaihir_dataset.gps_inertia
    # cdi_parameters["gps_t"] = gwaihir_dataset.gps_t
    # cdi_parameters["gps_s"] = gwaihir_dataset.gps_s
    # cdi_parameters["gps_sigma_f"] = gwaihir_dataset.gps_sigma_f
    # cdi_parameters["gps_sigma_o"] = gwaihir_dataset.gps_sigma_o
    # cdi_parameters["iobs_saturation"] = gwaihir_dataset.iobs_saturation
    # cdi_parameters["free_pixel_mask"] = gwaihir_dataset.free_pixel_mask
    # cdi_parameters["support_formula"] = gwaihir_dataset.support_formula
    # cdi_parameters["mpi"] = "run"
    # cdi_parameters["mask_interp"] = gwaihir_dataset.mask_interp
    # cdi_parameters["confidence_interval_factor_mask_min"] \
    # = gwaihir_dataset.confidence_interval_factor_mask_min
    # cdi_parameters["confidence_interval_factor_mask_max"] \
    # = gwaihir_dataset.confidence_interval_factor_mask_max
    # cdi_parameters["save_plot"] = gwaihir_dataset.save_plot
    # cdi_parameters["support_fraction_min"] \
    # = gwaihir_dataset.support_fraction_min
    # cdi_parameters["support_fraction_max"] \
    # = gwaihir_dataset.support_fraction_max
    # cdi_parameters["support_threshold_auto_tune_factor"] \
    # = gwaihir_dataset.support_threshold_auto_tune_factor
    # cdi_parameters["nb_run_keep_max_obj2_out"] \
    # = gwaihir_dataset.nb_run_keep_max_obj2_out
    # cdi_parameters["flatfield"] = gwaihir_dataset.flatfield
    # cdi_parameters["psf_filter"] = gwaihir_dataset.psf_filter
    cdi_parameters["detwin"] = gwaihir_dataset.detwin
    cdi_parameters["nb_raar"] = gwaihir_dataset.nb_raar
    cdi_parameters["nb_hio"] = gwaihir_dataset.nb_hio
    cdi_parameters["nb_er"] = gwaihir_dataset.nb_er
    cdi_parameters["nb_ml"] = gwaihir_dataset.nb_ml
    try:
        cdi_parameters["specfile"] = gwaihir_dataset.specfile_name
    except AttributeError:
        pass
    # cdi_parameters["imgcounter"] = gwaihir_dataset.imgcounter
    # cdi_parameters["imgname"] = gwaihir_dataset.imgname
    cdi_parameters["scan"] = gwaihir_dataset.scan

    print(
        "\nSaving phase retrieval parameters selected "
        "in the PyNX tab in the cxi file ..."
    )
    cdi_operator.save_data_cxi(
        filename=path_to_cxi,
        process_parameters=cdi_parameters,
    )


def create_yaml_file(
    fname,
    **kwargs
):
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

    # Save in file
    if fname.endswith('.yaml') or fname.endswith('.yml'):
        with open(fname, "w") as v:
            for line in config_file:
                v.write(line + "\n")
    else:
        raise FileError("Parameter fname must end with .yaml or .yml")


def list_reconstructions(
    folder,
    scan_name
):
    """List all cxi files in the folder and sort by creation time"""
    cxi_file_list = [f for f in sorted(
        glob.glob(folder + "*.cxi"),
        key=os.path.getmtime,
        reverse=True,
    ) if not os.path.basename(f).startswith(scan_name)]

    print(
        "################################################"
        "################################################"
    )
    for j, f in enumerate(cxi_file_list):
        file_timestamp = datetime.fromtimestamp(
            os.path.getmtime(f)).strftime('%Y-%m-%d %H:%M:%S')
        print(
            f"File: {os.path.basename(f)}"
            f"\n\tCreated: {file_timestamp}"
        )
        if j != len(cxi_file_list)-1:
            print("")
        else:
            print(
                "################################################"
                "################################################"
            )


def run_modes_decomposition(
    path_scripts,
    folder,
):
    """
    Decomposes several phase retrieval solutions into modes, saves only
    the first mode to save space.

    :param path_scripts: absolute path to script containing
     folder
    :param folder: path to folder in which are stored
     the .cxi files, all files corresponding to
     *LLK* pattern are loaded
    """
    try:
        print(
            "\n###########################################"
            "#############################################"
            f"\nUsing {path_scripts}/pynx-cdi-analysis"
            f"\nUsing {folder}/*LLK* files."
            f"\nRunning: $ pynx-cdi-analysis *LLK* modes=1"
            f"\nOutput in {folder}/modes_gui.h5"
            "\n###########################################"
            "#############################################"
        )
        os.system(
            "{}/pynx-cdi-analysis {}/*LLK* modes=1 modes_output={}/modes_gui.h5".format(
                quote(path_scripts),
                quote(folder),
                quote(folder),
            )
        )
    except KeyboardInterrupt:
        hash_print("Decomposition into modes stopped by user...")


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
