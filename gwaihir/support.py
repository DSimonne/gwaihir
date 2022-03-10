import glob
import numpy as np
import tables as tb

from scipy.ndimage import gaussian_filter
from IPython.display import display, Markdown, Latex, clear_output

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import gwaihir.plot as plot


class SupportTools():
    """Class that regroups the methods used to create/extract/optimize support
    in BCDI.
    """

    def __init__(
        self,
        path_to_data=None,
        path_to_support=None,
        saving_directory=None
    ):
        """Initialize the class with either a reconstructed object or the
        support of the reconstructed object.

        :param path_to_data: path to reconstructed object file
        :param path_to_support: path to support of reconstructed object file
        :param saving_directory: where the figures will be saved
        """
        self.path_to_data = path_to_data
        self.path_to_support = path_to_support

        if saving_directory is None:
            try:
                self.saving_directory = self.path_to_data.replace(
                    self.path_to_data.split("/")[-1], "")
            except AttributeError:
                try:
                    self.saving_directory = self.path_to_support.replace(
                        self.path_to_support.split("/")[-1], "")
                except AttributeError:
                    raise AttributeError("Please provide a saving_directory.")
        else:
            self.saving_directory = saving_directory

    def extract_support(self, compute=True):
        """Extract support as a 3D array of 0 et 1 from a reconstruction.

        :param compute: True to run function
        """
        # Work on cxi files
        if compute:
            if self.path_to_data.endswith(".cxi"):
                try:
                    with tb.open_file(self.path_to_data, "r") as f:

                        # Since .cxi files follow a specific architectture,
                        # we know where the mask is.
                        support = f.root.entry_1.image_1.mask[:]

                        # Save support
                        print(
                            "\n##########################################################################################"
                        )
                        np.savez_compressed(self.saving_directory +
                                            "extracted_support.npz", support=support)
                        print(f"Saved support in {self.saving_directory} as:")
                        print(f"\textracted_support.npz")
                        print(
                            "##########################################################################################\n"
                        )
                        plot.plot_3d_slices(support, log="interact")
                except tb.NoSuchNodeError:
                    hash_print("Data type not supported")

            # elif self.self.path_to_data.endswith(".npz"):

            else:
                hash_print("Data type not supported")

        else:
            clear_output(True)
            hash_print("Set compute to true to continue")

    def gaussian_convolution(self, sigma, threshold, compute=True):
        """Apply a gaussian convolution to the support, to avoid having holes
        inside.

        :param sigma: parameter used in scipy.ndimage.gaussian_filter
        :param threshold: threshold above which we define the support
         between 1 and 0, 1 being the maximum intensity
        :param compute: True to run function
        """
        if compute:
            try:
                old_support = np.load(self.path_to_support)["support"]
            except KeyError:
                try:
                    old_support = np.load(self.path_to_support)["data"]
                except KeyError:
                    print("Could not load 'data' or 'support' array from \
                        file.")
            except ValueError:
                hash_print("Data type not supported")

            try:
                bigdata = 100 * old_support
                conv_support = np.where(gaussian_filter(
                    bigdata, sigma) > threshold, 1, 0)
                print(
                    "\n##########################################################################################"
                )
                np.savez_compressed(
                    f"{self.saving_directory}filter_sig{sigma}_t{threshold}",
                    oldsupport=old_support, support=conv_support)

                print(f"Saved support in {self.saving_directory} as:")
                print(f"\tfilter_sig{sigma}_t{threshold}")
                print(
                    "##########################################################################################\n"
                )

                plot.plot_3d_slices(conv_support, log="interact")

            except UnboundLocalError:
                pass

        else:
            clear_output(True)
            hash_print("Set compute to true to continue")

    def compute_support(self, threshold, compute=True):
        """Create support from data, based on maximum value of electronic
        density module, a threshold is applied.

        :param compute: True to run function
        """
        if compute:
            try:
                with tb.open_file(self.path_to_data, "r") as f:
                    # Since .cxi files follow a specific architecture,
                    # we know where our data is
                    if self.path_to_data.endswith(".cxi"):
                        electronic_density = f.root.entry_1.data_1.data[:]

                    elif self.path_to_data.endswith(".h5"):
                        # Take first mode
                        electronic_density = f.root.entry_1.data_1.data[:][0]

                    print(
                        "\n##########################################################################################"
                    )
                    print("Shape of real space complex electronic density array:")
                    print(f"\t{np.shape(electronic_density)}")

                    # Find max value in image, we work with the module
                    amp = np.abs(electronic_density)
                    print(f"\tMaximum value in amplitude array: {amp.max()}")

                    # Define support based on max value and threshold
                    support = np.where(amp < threshold * amp.max(), 0, 1)

                    # Check % occupied by the support
                    rocc = np.where(support == 1)
                    rnocc = np.where(support == 0)

                    print("Percentage of 3D array occupied by support:")
                    print(f"\t{np.shape(rocc)[1] / np.shape(rnocc)[1]}")

                    # Save support
                    np.savez_compressed(self.saving_directory +
                                        "computed_support.npz", support=support)
                    print(f"Saved support in {self.saving_directory} as:")
                    print(f"\tcomputed_support.npz")
                    print(
                        "##########################################################################################\n"
                    )

                    plot.plot_3d_slices(support, log="interact")
            except tb.HDF5ExtError:
                hash_print("Data type not supported")

        else:
            clear_output(True)
            hash_print("Set compute to true to continue")


def hash_print(
    string_to_print,
    hash_line_before=True,
    hash_line_after=True,
    new_line_before=True,
    new_line_after=True
):
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
