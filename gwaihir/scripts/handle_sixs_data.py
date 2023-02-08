import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import glob
import os
import h5py
import shutil
from IPython.display import display, clear_output
import ipywidgets as widgets


def rotate_sixs_data(path_to_nxs_data):
    """
    Python script to rotate the data when using the vertical configuration.
    Should work on a copy of the data !! Never use the OG data !!
    Use for experiment performed in 2021 with the borrowed Merlin.

    :param path_to_nxs_data: absolute path to nexus file
    """
    # Define save folder
    save_folder = os.path.dirname(path_to_nxs_data)

    # Check if already rotated
    with h5py.File(path_to_nxs_data, "a") as f:
        try:
            data_already_rotated = f['rotation'][...]
        except (ValueError, RuntimeError, KeyError):
            data_already_rotated = False

    # Find 3D array key
    three_d_data_keys = []
    with h5py.File(path_to_nxs_data, "a") as f:
        try:
            for key in f['com']['scan_data'].keys():
                shape = f['com']['scan_data'][key].shape
                if len(shape) == 3:
                    three_d_data_keys.append(key)

            if not data_already_rotated:
                print("Rotating SIXS data ...")
                # Get data
                if len(three_d_data_keys) == 1:
                    good_data_key = three_d_data_keys[0]
                    print(f"Found 3D array for key: {good_data_key}")
                    data_og = f['com']['scan_data'][good_data_key][...]

                else:  # There are multiple 3D arrays
                    for key in three_d_data_keys:
                        data = f['com']['scan_data'][key][...]

                        # We know that the Merlin detector array shape
                        # should be either 512 or 515
                        # we remove 3D arrays from other detectors
                        if not data_og.shape[1] in (512, 515) \
                                and data_og.shape[2] in (512, 515):
                            three_d_data_keys.remove(key)

                    if len(three_d_data_keys) == 1:
                        good_data_key = three_d_data_keys[0]
                        print(f"Found 3D array for key: {good_data_key}")
                        data_og = f['com']['scan_data'][good_data_key][...]
                    else:
                        raise IndexError("Could not find 3D array")

                # Save that we rotated the data
                f.create_dataset("rotation", data=True)

                # Just an index for plotting schemes
                half = int(data_og.shape[0] / 2)

                # Transpose and flip lr data
                data = np.transpose(data_og, axes=(0, 2, 1))
                for idx in range(data.shape[0]):
                    tmp = data[idx, :, :]
                    data[idx, :, :] = np.fliplr(tmp)
                print("Data well rotated by 90°.")

                # Find bad frames
                sum_along_rc = data.sum(axis=(1, 2))

                bad_frames = []
                for j, summed_frame in enumerate(sum_along_rc):
                    if j == 0:
                        if 100 * sum_along_rc[1] < summed_frame:
                            bad_frames.append(j)
                    elif j == len(sum_along_rc)-1:
                        if 100 * sum_along_rc[-2] < summed_frame:
                            bad_frames.append(j)
                    else:
                        if 100 * sum_along_rc[j-1] < summed_frame \
                                and 100 * sum_along_rc[j+1] < summed_frame:
                            bad_frames.append(j)

                # Mask bad frames
                for j in bad_frames:
                    print("Masked frame", j)
                    data[j] = np.zeros((data.shape[1], data.shape[2]))

                    # Put one random pixel to 1, so that bcdi
                    # does not skip the frame since it is boggy for now
                    data[j, random.randint(0, data.shape[1]),
                         random.randint(0, data.shape[1])] = 1

                # Overwrite data in copied file
                f['com']['scan_data'][good_data_key][...] = data

                # Plot data
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

            else:
                print("Data already rotated ...")

        except (ValueError, RuntimeError, KeyError):
            pass  # Not sixs data


def flip_sixs_data(path_to_nxs_data):
    """
    Flips the data in the nexus file

    For coco's beamtime in June 2022
    Make sure to use this function on a copy of the original data directory,
    prior to executing gwaihir and bcdi.
    """
    # Define save folder
    save_folder = os.path.dirname(path_to_nxs_data)

    # Check if already flipped
    with h5py.File(path_to_nxs_data, "a") as f:
        try:
            data_already_flipped = f['flip'][...]
        except (ValueError, RuntimeError, KeyError):
            data_already_flipped = False

    # Find 3D array key
    three_d_data_keys = []
    with h5py.File(path_to_nxs_data, "a") as f:
        try:
            for key in f['com']['scan_data'].keys():
                shape = f['com']['scan_data'][key].shape
                if len(shape) == 3:
                    three_d_data_keys.append(key)

            if not data_already_flipped:
                print("Flipping SIXS data ...")
                # Get data
                if len(three_d_data_keys) == 1:
                    good_data_key = three_d_data_keys[0]
                    print(f"Found 3D array for key: {good_data_key}")
                    data_og = f['com']['scan_data'][good_data_key][...]

                else:  # There are multiple 3D arrays
                    for key in three_d_data_keys:
                        data = f['com']['scan_data'][key][...]

                        # We know that the Merlin detector array shape
                        # should be either 512 or 515
                        # we remove 3D arrays from other detectors
                        if not data_og.shape[1] in (512, 515) \
                                and data_og.shape[2] in (512, 515):
                            three_d_data_keys.remove(key)

                    if len(three_d_data_keys) == 1:
                        good_data_key = three_d_data_keys[0]
                        print(f"Found 3D array for key: {good_data_key}")
                        data_og = f['com']['scan_data'][good_data_key][...]
                    else:
                        raise IndexError("Could not find 3D array")

                # Save that we flipped the data
                f.create_dataset("flip", data=True)

                # Just an index for plotting schemes
                half = int(data_og.shape[0] / 2)

                # Flip data
                data = np.flip(data_og, (1, 2))
                print("Data well flipped.")

                # Find bad frames
                sum_along_rc = data.sum(axis=(1, 2))

                bad_frames = []
                for j, summed_frame in enumerate(sum_along_rc):
                    if j == 0:
                        if 100 * sum_along_rc[1] < summed_frame:
                            bad_frames.append(j)
                    elif j == len(sum_along_rc)-1:
                        if 100 * sum_along_rc[-2] < summed_frame:
                            bad_frames.append(j)
                    else:
                        if 100 * sum_along_rc[j-1] < summed_frame \
                                and 100 * sum_along_rc[j+1] < summed_frame:
                            bad_frames.append(j)

                # Mask bad frames
                for j in bad_frames:
                    print("Masked frame", j)
                    data[j] = np.zeros((data.shape[1], data.shape[2]))

                    # Put one random pixel to 1, so that bcdi
                    # does not skip the frame since it is boggy for now
                    data[j, random.randint(0, data.shape[1]),
                         random.randint(0, data.shape[1])] = 1

                # Overwrite data in copied file
                f['com']['scan_data'][good_data_key][...] = data

                # Plot data
                print("Saving example figures...", end="\n\n")
                plt.figure(figsize=(16, 9))
                plt.imshow(data_og[half, :, :], vmax=10)
                plt.xlabel('Delta')
                plt.ylabel('Gamma')
                plt.tight_layout()
                plt.savefig(save_folder + "/data_before_flip.png")
                plt.close()

                plt.figure(figsize=(16, 9))
                plt.imshow(data[half, :, :], vmax=10)
                plt.xlabel('Gamma')
                plt.ylabel('Delta')
                plt.tight_layout()
                plt.savefig(save_folder + "/data_after_flip.png")
                plt.close()

            else:
                print("Data already flipped ...")

        except (ValueError, RuntimeError, KeyError):
            pass  # Not sixs data
