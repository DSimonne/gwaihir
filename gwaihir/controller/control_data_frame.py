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


def display_data_frame(
    interface,
    unused_label_logs,
    parent_folder,
    csv_file,
    show_logs
):
    """
    Loads exterior .csv file and displays it in the GUI.

    :param parent_folder: all .csv files in the parent_folder subsirectories
     will be shown in the dropdown list.
    :param csv_file: path to csv file
    :param show_logs: True to display dataframe
    """
    # Load data
    if show_logs in ("load_csv", "load_field_data"):
        interface.tab_data_frame.children[1].disabled = True
        try:
            # csv data
            if show_logs == "load_csv":
                try:
                    logs = pd.read_csv(csv_file)
                except ValueError:
                    gutil.hash_print("Data type not supported.")

            # field data taken from facet analysis
            elif show_logs == "load_field_data":
                logs = interface.Facets.field_data.copy()

            @ interact(
                cols=widgets.SelectMultiple(
                    options=list(logs.columns),
                    value=list(logs.columns)[:],
                    rows=10,
                    style={'description_width': 'initial'},
                    layout=Layout(width='90%'),
                    description='Select multiple columns with \
                    Ctrl + click:',
                )
            )
            def pick_columns(cols):
                display(logs[list(cols)])

        except (FileNotFoundError, UnboundLocalError):
            gutil.hash_print("Wrong path")
        except AttributeError:
            print(
                "You need to run the facet analysis in the dedicated tab first."
                "\nThen this function will load the resulting DataFrame."
            )

    else:
        interface.tab_data_frame.children[1].disabled = False
        interface.csv_file_handler(parent_folder)
        clear_output(True)
