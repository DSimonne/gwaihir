import gwaihir

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import os
import shutil
from ast import literal_eval
import operator as operator_lib
import getpass

# import lmfit
# from lmfit import minimize, Parameters, Parameter
# from lmfit.models import LinearModel, ConstantModel, QuadraticModel, PolynomialModel, StepModel
# from lmfit.models import GaussianModel, LorentzianModel, SplitLorentzianModel, VoigtModel, PseudoVoigtModel
# from lmfit.models import MoffatModel, Pearson7Model, StudentsTModel, BreitWignerModel, LognormalModel, ExponentialGaussianModel, SkewedGaussianModel, SkewedVoigtModel, DonaichModel
# import corner
# import numdifftools
# from scipy.stats import chisquare

import ipywidgets as widgets
from ipywidgets import interact, Button, Layout, interactive, fixed
from IPython.display import display, Markdown, Latex, clear_output

# from scipy import interpolate
# from scipy import optimize, signal
# from scipy import sparse

from datetime import datetime
import pickle
import inspect
import time
# import warnings

import tables as tb

# gwaihir package
from gwaihir import facet_analysis
from gwaihir import plot
from gwaihir.sixs import ReadNxs4 as rd
from gwaihir.runner import preprocess, correct_angles, strain
from gwaihir.gui import gui_iterable
from gwaihir.support import SupportTools

# PyNX package
import h5py
from numpy.fft import fftshift
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage import gaussian_filter

# bcdi package
from bcdi.utils.utilities import bin_data

try:
    # This imports all necessary operators. GPU will be auto-selected
    print("Importing pynx ...")
    from pynx.cdi import *
    from pynx.cdi.widgets import CDIViewer
    from pynx.utils.math import smaller_primes
    pynx_import = True
except:
    pynx_import = False
    print("Could not load PyNX, the phase retrieval tab will be disabled")


class Interface(object):
    """This  class is a Graphical User Interface (gui) that is meant to be used to process important amount of XAS datasets that focus on the same energy range and absoption edge.
        Initializing the procedure in a jupyter notebook:
            _ gui = gui.Interface(); One will have to write the name of the data folder in which all his datasets are saved.
        This class makes extensive use of the ipywidgets and is thus meant to be used with a jupyter notebook.
        Additional informations are provided in the "ReadMe" tab of the gui.
        The necessary Python packages are : numpy, pandas, matplotlib, glob, errno, os, shutil, ipywidgets, IPython, scipy, datetime, importlib, pickle, lmfit
        lmfit, xlrd, corner and inspect.
    """

    def __init__(self):
        """All the widgets for the gui are defined here. 
        """
        super(Interface, self).__init__()

        self.work_dir = os.getcwd()
        self.path_package = inspect.getfile(gwaihir).split("__")[0]
        self.path_scripts = self.path_package.split("/lib/python")[0]+"/bin"

        # Get user name
        try:
            self.user_name = getpass.getuser()

            print("Login used for slurm:", self.user_name)
            print("If wrong login, please change self.user_name attribute")
        except:
            print("Could not get user name, please create self.user_name attribute for batch jobs")

        # Widgets for initialization 
        self._list_widgets_init = interactive(self.initialize_directories,
            ### Define scan related parameters
            label_scan = widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>Define working directory and scan number", # 
                style = {'description_width': 'initial'},
                layout = Layout(width='90%', height = "35px")),

            sample_name = widgets.Text(
                value = "S",
                placeholder = "",
                description = 'Sample Name',
                disabled = False,
                continuous_update = False,
                layout = Layout(width='45%'),
                style = {'description_width': 'initial'}),

            scans = widgets.BoundedIntText(
                value = "01415",
                description = 'Scan nb:',
                min = 0,
                max = 9999999,
                disabled = False,
                continuous_update = False,
                layout = Layout(width='45%'),
                style = {'description_width': 'initial'}),

            data_directory = widgets.Text(
                value = os.getcwd() + "/data_dir/",
                placeholder = "Path to data directory",
                description = 'Data directory',
                disabled = False,
                continuous_update = False,
                layout = Layout(width='90%'),
                style = {'description_width': 'initial'}),

            final_directory = widgets.Text(
                value = os.getcwd() + "/TestGui/",
                placeholder = "Path to target directory (parent to all scan directories)",
                description = 'Target directory',
                disabled = False,
                continuous_update = False,
                layout = Layout(width='90%'),
                style = {'description_width': 'initial'}),

            user_comment = widgets.Text(
                value = "",
                description = 'Comment',
                disabled = False,
                continuous_update = False,
                layout = Layout(width='90%'),
                placeholder = "Comment regarding Dataset...",
                style = {'description_width': 'initial'}),

            debug = widgets.Checkbox(
                value = False,
                description = 'Debug',
                disabled = False,
                tooltip = 'True to interact with plots, False to close it automatically',
                indent = False,
                continuous_update = False,
                style = {'description_width': 'initial'}),

            run_dir_init = widgets.ToggleButton(
                value = False,
                description = 'Initialize directories ...',
                disabled = False,
                button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                icon = 'step-forward',
                layout = Layout(width='45%'),
                style = {'description_width': 'initial'}),

            reload_previous_data = widgets.ToggleButton(
                value = False,
                description = 'Reload previous data (.gwr) from target directory ...',
                disabled = False,
                button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                icon = 'step-forward',
                layout = Layout(width='45%'),
                style = {'description_width': 'initial'}),
            )
        self._list_widgets_init.children[7].observe(self.init_handler, names = "value")
        self._list_widgets_init.children[8].observe(self.init_handler, names = "value")

        self.tab_init = widgets.VBox([
            self._list_widgets_init.children[0],
            widgets.HBox(self._list_widgets_init.children[1:3]),
            self._list_widgets_init.children[3],
            self._list_widgets_init.children[4],
            self._list_widgets_init.children[5],
            self._list_widgets_init.children[6],
            widgets.HBox(self._list_widgets_init.children[7:9]),
            self._list_widgets_init.children[-1],
            ])

        # Widgets for preprocessing
        self._list_widgets_preprocessing = interactive(self.initialize_parameters,
            ### Define beamline related parameters
            label_beamline = widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>Parameters specific to the beamline", # 
                style = {'description_width': 'initial'},
                layout = Layout(width='90%', height = "35px")),

            beamline = widgets.Dropdown(
                options = ['ID01', 'SIXS_2018', 'SIXS_2019', 'CRISTAL', 'P10', 'NANOMAX', '34ID'],
                value = "SIXS_2019",
                description = 'Beamline',
                continuous_update = False,
                disabled = True,
                tooltip = "Name of the beamline, used for data loading and normalization by monitor",
                style = {'description_width': 'initial'}),

            actuators = widgets.Text(
                value = "{}",
                placeholder = "{}",
                continuous_update = False,
                description = 'Actuators',
                tooltip = "Optional dictionary that can be used to define the entries corresponding to actuators in data files (useful at CRISTAL where the location of data keeps changing)",
                readout = True,
                style = {'description_width': 'initial'},
                disabled = True),

            is_series = widgets.Checkbox(
                value = False,
                description = 'Is series (P10)',
                disabled = True,
                # button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                continuous_update = False,
                tooltip = 'specific to series measurement at P10',
                icon = 'check'),

            custom_scan = widgets.Checkbox(
                value = False,
                description = 'Custom scan',
                continuous_update = False,
                disabled = True,
                indent = False,
                # button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                tooltip = 'set it to True for a stack of images acquired without scan, e.g. with ct in a macro, or when there is no spec/log file available',
                icon = 'check'),

            custom_images = widgets.IntText(
                value = 3, # np.arange(11353, 11453, 1)  # list of image numbers for the custom_scan
                description='Custom images',
                continuous_update = False,
                disabled = True,
                style = {'description_width': 'initial'}),

            custom_monitor = widgets.IntText(
                value = 51, # np.ones(51),  # monitor values for normalization for the custom_scan
                description='Custom monitor',
                continuous_update = False,
                disabled = True,
                style = {'description_width': 'initial'}),

            specfile_name = widgets.Text(
                placeholder = "alias_dict_2019.txt",
                value = "",
                description = 'Specfile name',
                disabled = True,
                continuous_update = False,
                layout = Layout(width='90%'),
                style = {'description_width': 'initial'}),

            rocking_angle = widgets.Dropdown(
                options = ['inplane', 'outofplane', 'energy'],
                value = "inplane",
                continuous_update = False,
                description = 'Rocking angle',
                disabled = True,
                tooltip = "Name of the beamline, used for data loading and normalization by monitor",
                layout = Layout(height = "50px"),
                style = {'description_width': 'initial'}),

            follow_bragg = widgets.Checkbox(
                value = False,
                description = 'Follow bragg',
                disabled = True,
                continuous_update = False,
                # button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                tooltip = 'Only for energy scans, set to True if the detector was also scanned to follow the Bragg peak',
                layout = Layout(height = "50px"),
                icon = 'check'),


            ### Parameters used in masking
            label_masking = widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>Parameters used in masking", # 
                style = {'description_width': 'initial'},
                layout = Layout(width='90%', height = "35px")),

            flag_interact = widgets.Checkbox(
                value = False,
                description = 'Manual masking',
                continuous_update = False,
                disabled = True,
                indent = False,
                # button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                tooltip = 'True to interact with plots and manually mask points',
                layout = Layout(height = "50px"),
                icon = 'check'),

            background_plot = widgets.FloatText(
                value = 0.5,
                step = 0.01,
                max = 1,
                min = 0,
                continuous_update = False,
                description = 'Background plot:',
                layout = Layout(width='30%', height = "50px"),
                tooltip = "In level of grey in [0,1], 0 being dark. For visual comfort during masking",
                readout = True,
                style = {'description_width': 'initial'},
                disabled = True),


            ### Parameters related to data cropping/padding/centering
            label_centering = widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>Parameters related to data cropping/padding/centering</p>", # 
                style = {'description_width': 'initial'},
                layout = Layout(width='90%', height = "35px")),

            centering = widgets.Dropdown(
                options = ["max", "com", "manual"],
                value = "max",
                description = 'Centering of Bragg peak method:',
                continuous_update = False,
                disabled = True,
                layout = Layout(width='45%'),
                tooltip = "Bragg peak determination: 'max' or 'com', 'max' is better usually. It will be overridden by 'fix_bragg' if not empty",
                style = {'description_width': 'initial'}),

            fix_bragg = widgets.Text(
                placeholder = "[z_bragg, y_bragg, x_bragg]",
                description = 'Bragg peak position', # fix the Bragg peak position [z_bragg, y_bragg, x_bragg] considering the full detector
                disabled = True, # It is useful if hotpixels or intense aliens. Leave it [] otherwise.
                continuous_update = False,
                layout = Layout(width='45%'),
                style = {'description_width': 'initial'}),

            fix_size = widgets.Text(
                placeholder = "[zstart, zstop, ystart, ystop, xstart, xstop]",
                description = 'Fix array size', # crop the array to predefined size considering the full detector, leave it to [] otherwise [zstart, zstop, ystart, ystop, xstart, xstop]. ROI will be defaulted to []
                disabled = True,
                continuous_update = False,
                layout = Layout(width='45%'),
                style = {'description_width': 'initial'}),  

            center_fft = widgets.Dropdown(
                options = ['crop_sym_ZYX','crop_asym_ZYX','pad_asym_Z_crop_sym_YX', 'pad_sym_Z_crop_asym_YX','pad_sym_Z', 'pad_asym_Z', 'pad_sym_ZYX','pad_asym_ZYX', 'skip'],
                value = "crop_sym_ZYX",
                description = 'Center FFT',
                continuous_update = False,
                layout = Layout(height = "50px"),
                disabled = True,
                style = {'description_width': 'initial'}),

            pad_size = widgets.Text(
                placeholder = "[256, 512, 512]",
                description = 'Array size after padding', # used in 'pad_sym_Z_crop_sym_YX', 'pad_sym_Z', 'pad_sym_ZYX'
                disabled = True,
                continuous_update = False,
                layout = Layout(width='50%', height = "50px"),
                style = {'description_width': 'initial'}), 

            ### Parameters used in intensity normalization
            normalize_flux = widgets.Dropdown(
                options = ["skip", "monitor"],
                value = "skip",
                description = 'Normalize flux',
                disabled = True,
                continuous_update = False,
                layout = Layout(height = "50px"),
                # button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                tooltip = 'Monitor to normalize the intensity by the default monitor values, skip to do nothing',
                # icon = 'check',
                style = {'description_width': 'initial'}),


            ### Parameters for data filtering
            label_filtering = widgets.HTML(
                description="""<p style='font-weight: bold;font-size:1.2em'>Parameters for data filtering</p>""", 
                style = {'description_width': 'initial'},
                layout = Layout(width='90%', height = "35px")),

            mask_zero_event = widgets.Checkbox(
                value = False,
                description = 'Mask zero event',
                disabled = True,
                continuous_update = False,
                indent = False,
                # button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                tooltip = 'Mask pixels where the sum along the rocking curve is zero - may be dead pixels',
                icon = 'check'),

            flag_medianfilter = widgets.Dropdown(
                options = ['skip','median','interp_isolated', 'mask_isolated'],
                value = "skip",
                description = 'Flag median filter',
                continuous_update = False,
                disabled = True,
                tooltip = "set to 'median' for applying med2filter [3,3], set to 'interp_isolated' to interpolate isolated empty pixels based on 'medfilt_order' parameter, set to 'mask_isolated' it will mask isolated empty pixels, set to 'skip' will skip filtering",
                style = {'description_width': 'initial'}),

            medfilt_order = widgets.IntText(
                value = 7,
                description='Med filter order:',
                disabled = True,
                continuous_update = False,
                tooltip = "for custom median filter, number of pixels with intensity surrounding the empty pixel",
                style = {'description_width': 'initial'}),

            binning = widgets.Text(
                value = "(1, 1, 1)",
                placeholder = "(1, 1, 1)",
                description = 'Binning for phasing',
                disabled = True,
                continuous_update = False,
                layout = Layout(width='20%', height = "50px"),
                style = {'description_width': 'initial'},
                tooltip = "binning that will be used for phasing (stacking dimension, detector vertical axis, detector horizontal axis)"),

            ### Parameters used when reloading processed data
            label_reload = widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>Parameters used when reloading processed data</p>", # 
                style = {'description_width': 'initial'},
                layout = Layout(width='90%', height = "35px")),

            reload_previous = widgets.Checkbox(
                value = False,
                description = 'Reload previous',
                continuous_update = False,
                disabled = True,
                indent = False,
                layout = Layout(height = "50px"),
                # button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                tooltip = 'True to resume a previous masking (load data and mask)',
                icon = 'check'),

            reload_orthogonal = widgets.Checkbox(
                value = False,
                description = 'Reload orthogonal',
                continuous_update = False,
                disabled = True,
                indent = False,
                layout = Layout(height = "50px"),
                # button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                tooltip = 'True if the reloaded data is already intepolated in an orthonormal frame',
                icon = 'check'),

            preprocessing_binning = widgets.Text(
                value = "(1, 1, 1)",
                placeholder = "(1, 1, 1)", # binning factors in each dimension of the binned data to be reloaded
                description = 'Binning used in data to be reloaded',
                disabled = True,
                continuous_update = False,
                layout = Layout(width='30%', height = "50px"),
                style = {'description_width': 'initial'},
                tooltip = "binning that will be used for phasing (stacking dimension, detector vertical axis, detector horizontal axis)"),

            ### Saving options
            label_saving = widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>Parameters used when saving the data</p>", # 
                style = {'description_width': 'initial'},
                layout = Layout(width='90%', height = "35px")),

            save_rawdata = widgets.Checkbox(
                value = False,
                description = 'Save raw data',
                disabled = True,
                continuous_update = False,
                indent = False,
                layout = Layout(width = "20%", height = "50px"),
                # button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                tooltip = 'Save also the raw data when use_rawdata is False',
                icon = 'check'),

            save_to_npz = widgets.Checkbox(
                value = True,
                description = 'Save to npz',
                disabled = True,
                continuous_update = False,
                indent = False,
                layout = Layout(width = "20%", height = "50px"),
                # button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                tooltip = 'True to save the processed data in npz format',
                icon = 'check'),

            save_to_mat = widgets.Checkbox(
                value = False,
                description = 'Save to mat',
                disabled = True,
                continuous_update = False,
                indent = False,
                layout = Layout(width = "20%", height = "50px"),
                # button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                tooltip = 'True to save also in .mat format',
                icon = 'check'),

            save_to_vti = widgets.Checkbox(
                value = False,
                description = 'Save to vti',
                continuous_update = False,
                disabled = True,
                # button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                indent = False,
                layout = Layout(width = "20%", height = "50px"),
                tooltip = 'Save the orthogonalized diffraction pattern to VTK file',
                icon = 'check'),

            save_asint = widgets.Checkbox(
                value = False,
                description = 'Save as integers',
                continuous_update = False,
                disabled = True,
                # button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                indent = False,
                layout = Layout(width = "20%", height = "50px"),
                tooltip = 'if True, the result will be saved as an array of integers (save space)',
                icon = 'check'),

            ### Detector related parameters
            label_detector = widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>Parameters related to the detector used</p>", # 
                style = {'description_width': 'initial'},
                layout = Layout(width='90%', height = "35px")),

            detector = widgets.Dropdown(
                options = ["Eiger2M", "Maxipix", "Eiger4M", "Merlin", "Timepix"],
                value = "Merlin",
                description = 'Detector',
                continuous_update = False,
                disabled = True,
                style = {'description_width': 'initial'}),

            roi_detector = widgets.Text(
                placeholder = "[low_y_bound, high_y_bound, low_x_bound, high_x_bound]",
                description = 'Fix roi area, will overwrite cropping parameters', # fix the Bragg peak position [z_bragg, y_bragg, x_bragg] considering the full detector
                disabled = True, # It is useful if hotpixels or intense aliens. Leave it [] otherwise.
                continuous_update = False,
                layout = Layout(width='90%'),
                style = {'description_width': 'initial'}),

            photon_threshold = widgets.IntText(
                value = 0,
                description = 'Photon Threshold:',
                disabled = True,
                continuous_update = False,
                tooltip = "data[data < photon_threshold] = 0",
                style = {'description_width': 'initial'}),

            photon_filter = widgets.Dropdown(
                options = ['loading','postprocessing'],
                value = "loading",
                continuous_update = False,
                description = 'Photon filter',
                disabled = True,
                tooltip = "When the photon threshold should be applied, if 'loading', it is applied before binning; if 'postprocessing', it is applied at the end of the script before saving",
                style = {'description_width': 'initial'}),

            background_file = widgets.Text(
                value = "",
                placeholder = f"{self.work_dir}/background.npz'",
                description = 'Background file',
                disabled = True,
                continuous_update = False,
                layout = Layout(width='90%'),
                style = {'description_width': 'initial'}),

            flatfield_file = widgets.Text(
                value = "",
                placeholder = f"{self.work_dir}/flatfield_maxipix_8kev.npz",
                description = 'Flatfield file',
                disabled = True,
                continuous_update = False,
                layout = Layout(width='90%'),
                style = {'description_width': 'initial'}),

            hotpixels_file = widgets.Text(
                value = f"{self.work_dir}/SIXS_June_2021/reconstructions/analysis/mask_merlin_better_flipped.npy",
                placeholder = f"{self.work_dir}/mask_merlin.npz",
                description = 'Hotpixels file',
                disabled = True,
                continuous_update = False,
                layout = Layout(width='90%'),
                style = {'description_width': 'initial'}),

            template_imagefile = widgets.Text(
                value = 'Pt_ascan_mu_%05d.nxs',
                description = 'Template imagefile',
                disabled = True,
                tooltip = """Template for ID01: 'data_mpx4_%05d.edf.gz' or 'align_eiger2M_%05d.edf.gz'; Template for SIXS_2018: 'align.spec_ascan_mu_%05d.nxs';
                            Template for SIXS_2019: 'spare_ascan_mu_%05d.nxs';
                            Template for Cristal: 'S%d.nxs';
                            Template for P10: '_master.h5'; 
                            Template for NANOMAX: '%06d.h5'; 
                            Template for 34ID: 'Sample%dC_ES_data_51_256_256.npz'""",
                layout = Layout(width='90%'),
                style = {'description_width': 'initial'}),

            nb_pixel_x = widgets.IntText(
                description = 'Nb pixel x',
                disabled = True,
                continuous_update = False,
                tooltip = "fix to declare a known detector but with less pixels",
                style = {'description_width': 'initial'}),

            nb_pixel_y = widgets.IntText(
                description = 'Nb pixel y',
                disabled = True,
                continuous_update = False,
                tooltip = "fix to declare a known detector but with less pixels",
                style = {'description_width': 'initial'}),

            ### Define parameters below if you want to orthogonalize the data before phasing
            label_ortho = widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>Parameters to define the data orthogonalization</p>", # 
                style = {'description_width': 'initial'},
                layout = Layout(width='90%', height = "35px")),

            use_rawdata = widgets.Checkbox(
                value = False,
                continuous_update = False,
                description = 'Orthogonalize data',
                disabled = True,
                indent = False,
                # button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                tooltip = 'False for using data gridded in laboratory frame/ True for using data in detector frame',
                icon = 'check'),

            interp_method = widgets.Dropdown(
                options = ['linearization','xrayutilities'],
                value = "linearization",
                continuous_update = False,
                description = 'Interpolation method',
                disabled = True,
                # tooltip = "",
                style = {'description_width': 'initial'}),

            fill_value_mask = widgets.Dropdown(
                options = [0, 1],
                value = 0,
                description = 'Fill value mask',
                continuous_update = False,
                disabled = True,
                tooltip = "It will define how the pixels outside of the data range are processed during the interpolation. Because of the large number of masked pixels, phase retrieval converges better if the pixels are not masked (0 intensity imposed). The data is by default set to 0 outside of the defined range.",
                style = {'description_width': 'initial'}),

            beam_direction = widgets.Text(
                value = "(1, 0, 0)",
                placeholder = "(1, 0, 0)",
                description = 'Beam direction in lab. frame',
                disabled = True,
                continuous_update = False,
                layout = Layout(width='50%'),
                style = {'description_width': 'initial'},
                tooltip = "Beam direction in the laboratory frame (downstream, vertical up, outboard), beam along z"),

            sample_offsets = widgets.Text(
                value = "(0, 0)",
                placeholder = "(0, 0, 90, 0)",
                description = 'Sample offsets',
                disabled = True,
                continuous_update = False,
                layout = Layout(width='25%'),
                style = {'description_width': 'initial'},
                tooltip = """Tuple of offsets in degrees of the sample for each sample circle (outer first). 
                            Convention: the sample offsets will be subtracted to the motor values"""),

            sdd = widgets.FloatText(
                value = 1.18,
                step = 0.01,
                description = 'Sample Detector Dist. (m):',
                continuous_update = False,
                disabled = True,
                tooltip = "sample to detector distance in m",
                style = {'description_width': 'initial'}),

            energy = widgets.IntText(
                value = 8500,
                description = 'X-ray energy in eV',
                continuous_update = False,
                disabled = True,
                layout = Layout(height = "50px"),
                style = {'description_width': 'initial'}),

            custom_motors = widgets.Text(
                value = "{}",
                placeholder = "{}",
                description = 'Custom motors',
                disabled = True,
                continuous_update = False,
                layout = Layout(width='90%', height = "50px"),
                style = {'description_width': 'initial'},
                tooltip = "Use this to declare motor positions"),
            # {"mu": 0, "phi": -15.98, "chi": 90, "theta": 0, "delta": -0.5685, "gamma": 33.3147}
            # use this to declare motor positions if there is not log file
            # example: {"eta": np.linspace(16.989, 18.989, num=100, endpoint=False), "phi": 0, "nu": -0.75, "delta": 36.65}
            # ID01: eta, phi, nu, delta
            # CRISTAL: mgomega, gamma, delta
            # SIXS: beta, mu, gamma, delta
            # P10: om, phi, chi, mu, gamma, delta
            # NANOMAX: theta, phi, gamma, delta, energy, radius
            # 34ID: mu, phi (incident angle), chi, theta (inplane), delta (inplane), gamma (outofplane)


            ### Parameters for xrayutilities to orthogonalize the data before phasing
            label_xru = widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>Parameters used in xrayutilities to orthogonalize the data before phasing (initialize the directories before)</p>", # 
                style = {'description_width': 'initial'},
                layout = Layout(width='90%', height = "35px")),

            #xrayutilities uses the xyz crystal frame: for incident angle = 0, x is downstream, y outboard, and z vertical up
            align_q = widgets.Checkbox(
                value = True,
                description = 'Align q',
                continuous_update = False,
                disabled = True,
                indent = False,
                layout = Layout(width='20%'),
                # button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                tooltip = """used only when interp_method is 'linearization', if True it rotates the crystal to align q along one axis of the array""",
                icon = 'check'),

            ref_axis_q = widgets.Dropdown(
                options = ["x", "y", "z"],
                value = "y",
                description = 'Ref axis q',
                disabled = True,
                continuous_update = False,
                layout = Layout(width='20%'),
                tooltip = "q will be aligned along that axis",
                style = {'description_width': 'initial'}),

            outofplane_angle = widgets.FloatText(
                value = 0,
                step = 0.01,
                description = 'Outofplane angle',
                continuous_update = False,
                disabled = True,
                layout = Layout(width='25%'),
                style = {'description_width': 'initial'}),

            inplane_angle = widgets.FloatText(
                value = 0,
                step = 0.01,
                description = 'Inplane angle',
                continuous_update = False,
                disabled = True,
                layout = Layout(width='25%'),
                style = {'description_width': 'initial'}),

            sample_inplane = widgets.Text(
                value = "(1, 0, 0)",
                placeholder = "(1, 0, 0)",
                description = 'Sample inplane',
                disabled = True,
                continuous_update = False,
                layout = Layout(width='20%'),
                style = {'description_width': 'initial'},
                tooltip = "Sample inplane reference direction along the beam at 0 angles"),

            sample_outofplane = widgets.Text(
                value = "(0, 0, 1)",
                placeholder = "(0, 0, 1)",
                description = 'Sample outofplane',
                disabled = True,
                continuous_update = False,
                layout = Layout(width='20%'),
                style = {'description_width': 'initial'},
                tooltip = "Surface normal of the sample at 0 angles"),

            offset_inplane = widgets.FloatText(
                value = 0,
                step = 0.01,
                description = 'Offset inplane',
                disabled = True,
                continuous_update = False,
                layout = Layout(width='20%'),
                style = {'description_width': 'initial'},
                tooltip = "Outer detector angle offset, not important if you use raw data"),

            cch1 = widgets.IntText(
                value = 271,
                description = 'cch1',
                continuous_update = False,
                disabled = True,
                layout = Layout(width='15%'),
                tooltip = "cch1 parameter from xrayutilities 2D detector calibration, vertical",
                style = {'description_width': 'initial'}),

            cch2 = widgets.IntText(
                value = 213,
                description = 'cch2',
                continuous_update = False,
                disabled = True,
                layout = Layout(width='15%'),
                tooltip = "cch2 parameter from xrayutilities 2D detector calibration, horizontal",
                style = {'description_width': 'initial'}),

            direct_inplane = widgets.FloatText(
                value = 0,
                step = 0.01,
                min = 0,
                max = 360,
                continuous_update = False,
                description = 'Direct inplane angle:', # detector angle in deg(rotation around y vertical up, typically gamma), corrected
                layout = Layout(width='30%'),
                readout = True,
                style = {'description_width': 'initial'},
                disabled = True),

            direct_outofplane = widgets.FloatText(
                value = 0,
                step = 0.01,
                min = 0,
                max = 360,
                continuous_update = False,
                description = 'Direct outofplane angle:',# detector angle in deg (rotation around x outboard, typically delta), corrected
                layout = Layout(width='30%'),
                readout = True,
                style = {'description_width': 'initial'},
                disabled = True),

            detrot = widgets.FloatText(
                value = 0,
                step = 0.01,
                description = 'Detector rotation',
                disabled = True,
                continuous_update = False,
                layout = Layout(width='20%'),
                style = {'description_width': 'initial'},
                tooltip = "Detrot parameter from xrayutilities 2D detector calibration"),

            tiltazimuth = widgets.FloatText(
                value = 360,
                step = 0.01,
                description = 'Tilt azimuth',
                disabled = True,
                continuous_update = False,
                layout = Layout(width='15%'),
                style = {'description_width': 'initial'},
                tooltip = "tiltazimuth parameter from xrayutilities 2D detector calibration"),

            tilt = widgets.FloatText(
                value = 0,
                step = 0.01,
                description = 'Tilt',
                disabled = True,
                continuous_update = False,
                layout = Layout(width='15%'),
                style = {'description_width': 'initial'},
                tooltip = "tilt parameter from xrayutilities 2D detector calibration"),

            # Run preprocess
            label_preprocess = widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>Click below to run the data processing before phasing</p>", # 
                style = {'description_width': 'initial'},
                layout = Layout(width='90%', height = "35px")),

            init_para = widgets.ToggleButton(
                value = False,
                description = 'Initialize parameters ...',
                disabled = True,
                continuous_update = False,
                button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                layout = Layout(width='40%'),
                style = {'description_width': 'initial'},
                icon = 'fast-forward')
            )
        self._list_widgets_preprocessing.children[1].observe(self.beamline_handler, names = "value")
        self._list_widgets_preprocessing.children[8].observe(self.energy_scan_handler, names = "value")
        self._list_widgets_preprocessing.children[14].observe(self.bragg_peak_centering_handler, names = "value")
        self._list_widgets_preprocessing.children[26].observe(self.reload_data_handler, names = "value")
        self._list_widgets_preprocessing.children[47].observe(self.interpolation_handler, names = "value")
        self._list_widgets_preprocessing.children[-2].observe(self.preprocess_handler, names = "value")

        self.tab_detector = widgets.VBox([
            self._list_widgets_preprocessing.children[35],
            self._list_widgets_preprocessing.children[36],
            self._list_widgets_preprocessing.children[37],
            widgets.HBox(self._list_widgets_preprocessing.children[38:40]),
            self._list_widgets_preprocessing.children[40],
            self._list_widgets_preprocessing.children[41],
            self._list_widgets_preprocessing.children[42],
            self._list_widgets_preprocessing.children[43],
            widgets.HBox(self._list_widgets_preprocessing.children[44:46]),
            ])

        self.tab_ortho = widgets.VBox([
            self._list_widgets_preprocessing.children[46],
            self._list_widgets_preprocessing.children[47],
            widgets.HBox(self._list_widgets_preprocessing.children[48:50]),
            self._list_widgets_preprocessing.children[50],
            self._list_widgets_preprocessing.children[51],
            self._list_widgets_preprocessing.children[52],
            widgets.HBox(self._list_widgets_preprocessing.children[53:55]),
            self._list_widgets_preprocessing.children[55],
            widgets.HBox(self._list_widgets_preprocessing.children[56:58]),
            widgets.HBox(self._list_widgets_preprocessing.children[58:61]),
            widgets.HBox(self._list_widgets_preprocessing.children[61:63]),
            widgets.HBox(self._list_widgets_preprocessing.children[63:67]),
            widgets.HBox(self._list_widgets_preprocessing.children[67:70]),
            ])

        self.tab_beamline = widgets.VBox([
            self._list_widgets_preprocessing.children[0],
            self._list_widgets_preprocessing.children[1],
            widgets.HBox(self._list_widgets_preprocessing.children[2:4]),
            widgets.HBox(self._list_widgets_preprocessing.children[4:7]),
            self._list_widgets_preprocessing.children[7],
            widgets.HBox(self._list_widgets_preprocessing.children[8:10]),
            self._list_widgets_preprocessing.children[10],
            widgets.HBox(self._list_widgets_preprocessing.children[11:13]),
            ])

        self.tab_reduction = widgets.VBox([
            self._list_widgets_preprocessing.children[13],
            widgets.HBox(self._list_widgets_preprocessing.children[14:16]),
            self._list_widgets_preprocessing.children[16],
            widgets.HBox(self._list_widgets_preprocessing.children[17:20]),
            self._list_widgets_preprocessing.children[20],
            widgets.HBox(self._list_widgets_preprocessing.children[21:24]),
            self._list_widgets_preprocessing.children[24],
            ])

        self.tab_save_load = widgets.VBox([
            self._list_widgets_preprocessing.children[25],
            widgets.HBox(self._list_widgets_preprocessing.children[26:29]),
            self._list_widgets_preprocessing.children[29],
            widgets.HBox(self._list_widgets_preprocessing.children[30:35]),
            ])

        self.tab_preprocess = widgets.VBox([
            self.tab_beamline,
            self.tab_reduction,
            self.tab_save_load,
            self._list_widgets_preprocessing.children[-3],
            self._list_widgets_preprocessing.children[-2],
            self._list_widgets_preprocessing.children[-1]
            ])


        # Widgets for angles correction 
        self._list_widgets_correct = interactive(self.correct_angles,
            label_correct = widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>Find the real values for the Bragg peak angles, needs correct xru parameters.", # 
                style = {'description_width': 'initial'},
                layout = Layout(width='90%', height = "35px")),

            csv_file = widgets.Text(
                value = os.getcwd() + "/metadata.csv",
                placeholder = "Path to csv file",
                description = 'Csv file',
                disabled = True,
                continuous_update = False,
                layout = Layout(width='90%'),
                style = {'description_width': 'initial'}),

            temp_bool = widgets.Checkbox(
                value = False,
                description = 'Estimate the temperature (Pt only)',
                disabled = True,
                # button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                tooltip = 'Click to estimate the mean temperature of the sample from the Bragg peak angles',
                # icon = 'check',
                layout = Layout(width='40%'),
                style = {'description_width': 'initial'}),

            reflection = widgets.Text(
                value = "[1, 1, 1]",
                placeholder = "[1, 1, 1]",
                description = 'Reflection',
                disabled = True,
                continuous_update = False,
                layout = Layout(width='30%'),
                style = {'description_width': 'initial'},
                tooltip = "Sample inplane reference direction along the beam at 0 angles"),

            reference_spacing = widgets.FloatText(
                value = 2.269545,
                step = 0.000001,
                min = 0,
                max = 100,
                description = 'Interplanar distance (A):',
                layout = Layout(width='30%'),
                readout = True,
                style = {'description_width': 'initial'},
                disabled = True),

            reference_temperature = widgets.FloatText(
                value = 298.15,
                step = 0.01,
                min = 0,
                max = 2000,
                description = 'Reference temperature:',
                layout = Layout(width='30%'),
                readout = True,
                style = {'description_width': 'initial'},
                disabled = True),

            angles_bool = widgets.ToggleButton(
                value = False,
                description = 'Correct angles',
                disabled = True,
                button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                tooltip = 'Click to correct the Bragg peak angles',
                icon = 'fast-forward',
                layout = Layout(width='40%'),
                style = {'description_width': 'initial'}),
            )
        self._list_widgets_correct.children[2].observe(self.temp_handler, names = "value")
        self._list_widgets_correct.children[-2].observe(self.correct_angles_handler, names = "value")
        self.tab_correct = widgets.VBox([
            self._list_widgets_correct.children[0],
            self._list_widgets_correct.children[1],
            self._list_widgets_correct.children[2],
            widgets.HBox(self._list_widgets_correct.children[3:6]),
            self._list_widgets_correct.children[6],
            self._list_widgets_correct.children[-1],
            ])

        # Widgets for strain
        self._list_widgets_strain = interactive(self.strain_gui,
            label_averaging = widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>Parameters used when averaging several reconstruction", # 
                style = {'description_width': 'initial'},
                layout = Layout(width='90%', height = "35px")),

            sort_method = widgets.Dropdown(
                options = ['mean_amplitude', 'variance', 'variance/mean', 'volume'],
                value = "variance/mean",
                description = 'Sorting method',
                disabled = False,
                style = {'description_width': 'initial'}),

            correlation_threshold = widgets.FloatText(
                value = 0.9,
                step = 0.01,
                max = 1,
                min = 0,
                continuous_update = False,
                description = 'Correlation threshold:',
                style = {'description_width': 'initial'},
                disabled = False),

            label_FFT = widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>Parameters relative to the FFT window and voxel sizes", # 
                style = {'description_width': 'initial'},
                layout = Layout(width='90%', height = "35px")),

            original_size = widgets.Text(
                placeholder = "[256, 512, 512]",
                description = 'Size of the FFT array before binning',
                disabled = False,
                layout = Layout(width='45%'),
                continuous_update = False,
                style = {'description_width': 'initial'}),

            phasing_binning = widgets.Text(
                value = "(1, 1, 1)",
                placeholder = "(1, 1, 1)",
                description = 'Binning factor used in phase retrieval',
                disabled = False,
                continuous_update = False,
                layout = Layout(width='45%'),
                style = {'description_width': 'initial'},
                ),

            preprocessing_binning = widgets.Text(
                value = "(1, 1, 1)",
                placeholder = "(1, 1, 1)",
                description = 'Binning factors used in preprocessing',
                disabled = False,
                continuous_update = False,
                layout = Layout(width='45%'),
                style = {'description_width': 'initial'},
                ),

            output_size = widgets.Text(
                placeholder = "[256, 512, 512]",
                description = 'Output size',
                disabled = False,
                continuous_update = False,
                style = {'description_width': 'initial'}),

            keep_size = widgets.Checkbox(
                value = False,
                description = 'Keep the initial array size for orthogonalization (slower)',# True to keep the initial array size for orthogonalization (slower), it will be cropped otherwise
                disabled = False,
                layout = Layout(width='45%'),
                # icon = 'check',
                style = {'description_width': 'initial'}),

            fix_voxel = widgets.BoundedIntText(
                placeholder = "10",
                description = 'Fix voxel size, put 0 to set free:',
                min = 0,
                max = 9999999,
                disabled = False,
                continuous_update = False,
                style = {'description_width': 'initial'}),

            label_disp_strain = widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>Parameters related to displacement and strain calculation", # 
                style = {'description_width': 'initial'},
                layout = Layout(width='90%', height = "35px")),

            data_frame = widgets.ToggleButtons(
                options = ['detector', 'crystal', "laboratory"],
                value = "detector",
                description = 'Data frame',
                tooltips = [
                    "If the data is still in the detector frame",
                    "If the data was interpolated into the crystal frame using (xrayutilities) or (transformation matrix + align_q=True)", 
                    "If the data was interpolated into the laboratory frame using the transformation matrix (align_q = False)"
                    ],
                disabled = False,
                style = {'description_width': 'initial'}),

            ref_axis_q = widgets.Dropdown(
                options = ["x", "y", "z"],
                value = "y",
                description = 'Ref axis q',
                disabled = False,
                continuous_update = False,
                layout = Layout(width='15%'),
                tooltip = "q will be aligned along that axis",
                style = {'description_width': 'initial'}),

            save_frame = widgets.ToggleButtons(
                options = ['crystal', 'laboratory', "lab_flat_sample"],
                value = "lab_flat_sample",
                description = 'Final frame',
                tooltips = [
                    "Save the data with q aligned along ref_axis_q",
                    "Save the data in the laboratory frame (experimental geometry)", 
                    "Save the data in the laboratory frame, with all sample angles rotated back to 0"
                    ],
                # rotations for 'laboratory' and 'lab_flat_sample' are realized after the strain calculation
                # (which is done in the crystal frame along ref_axis_q)
                disabled = False,
                style = {'description_width': 'initial'}),

            isosurface_strain = widgets.FloatText(
                value = 0.3,
                step = 0.01,
                max = 1,
                min = 0,
                continuous_update = False,
                description = 'Isosurface strain:',
                tooltip = "Threshold use for removing the outer layer (strain is undefined at the exact surface voxel)",
                readout = True,
                layout = Layout(width='20%'),
                style = {'description_width': 'initial'},
                disabled = False),

            strain_method = widgets.ToggleButtons(
                options = ['default', 'defect'],
                value = "default",
                description = 'Strain method',
                tooltips = [
                    "",
                    "Will offset the phase in a loop and keep the smallest magnitude value for the strain. See: F. Hofmann et al. PhysRevMaterials 4, 013801 (2020)"
                    ],
                disabled = False,
                style = {'description_width': 'initial'}),

            phase_offset = widgets.FloatText(
                value = 0,
                step = 0.01,
                min = 0,
                max = 360,
                continuous_update = False,
                description = 'Phase offset:',
                layout = Layout(width='15%'),
                readout = True,
                style = {'description_width': 'initial'},
                disabled = False),

            phase_offset_origin = widgets.Text(
                placeholder = "(x, y, z), leave None for automatic.",
                description = 'Phase offset origin',
                disabled = False,
                continuous_update = False,
                layout = Layout(width='40%'),
                style = {'description_width': 'initial'},
                ),

            offset_method = widgets.Dropdown(
                options = ["COM", "mean"],
                value = "mean",
                description = 'Offset method:',
                continuous_update = False,
                disabled = False,
                layout = Layout(width='20%'),
                style = {'description_width': 'initial'}),

            centering_method = widgets.Dropdown(
                options = ["COM", "max", "max_COM"],
                value = "max_COM",
                description = 'Centering method:',
                continuous_update = False,
                disabled = False,
                layout = Layout(width='25%'),
                style = {'description_width': 'initial'}),

            label_refraction = widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>Parameters related to the refraction correction", # 
                style = {'description_width': 'initial'},
                layout = Layout(width='90%', height = "35px")),

            correct_refraction = widgets.Checkbox(
                value = False,
                description = 'Correct refraction',
                disabled = False,
                # icon = 'check',
                style = {'description_width': 'initial'}
                ),
            
            optical_path_method = widgets.ToggleButtons(
                options = ['threshold', 'defect'],
                value = "threshold",
                description = 'Optical path method',
                tooltips = [
                    "Uses isosurface_strain to define the support  for the optical path calculation",
                    "Tries to remove only outer layers even if the amplitude is lower than isosurface_strain inside the crystal"
                    ],
                disabled = True,
                style = {'description_width': 'initial'}),
            
            dispersion = widgets.FloatText(
                value = 0.000050328,
                continuous_update = False,
                description = 'Dispersion (delta):',
                readout = True,
                style = {'description_width': 'initial'},
                disabled = True),
            
            absorption = widgets.FloatText(
                value = 0.000050328,
                continuous_update = False,
                description = 'Absorption (beta):',
                readout = True,
                style = {'description_width': 'initial'},
                disabled = True),
            
            threshold_unwrap_refraction = widgets.FloatText(
                value = 0.05,
                # threshold used to calculate the optical path
                # the threshold for refraction/absorption corrections should be low, to correct for an object larger than the real one,
                # otherwise it messes up the phase
                step = 0.01,
                continuous_update = False,
                description = 'Threshold unwrap refraction:',
                readout = True,
                style = {'description_width': 'initial'},
                disabled = True),
            
            label_options = widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>Options", # 
                style = {'description_width': 'initial'},
                layout = Layout(width='90%', height = "35px")),
            
            simu_flag = widgets.Checkbox(
                value = False,
                description = 'Simulated data',
                disabled = False,
                # button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                # icon = 'check',
                layout = Layout(width='33%'),
                style = {'description_width': 'initial'}
                ),
            
            invert_phase = widgets.Checkbox(
                value = True, # False only for simulations
                description = 'Invert phase', # True for the displacement to have the right sign (FFT convention), False only for simulations
                disabled = False,
                # button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                # icon = 'check',
                layout = Layout(width='33%'),
                style = {'description_width': 'initial'}
                ),
            
            flip_reconstruction = widgets.Checkbox(
                value = False,
                description = 'Get conjugated object',  # True if you want to get the conjugate object
                disabled = False,
                # button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                # icon = 'check',
                layout = Layout(width='33%'),
                style = {'description_width': 'initial'}
                ),
            
            phase_ramp_removal = widgets.Dropdown(
                options = ["gradient", "upsampling"],
                value = "gradient",
                description = 'Phase ramp removal:',
                continuous_update = False,
                disabled = False,
                style = {'description_width': 'initial'}),
            
            threshold_gradient = widgets.FloatText(
                value = 1.0,
                step = 0.01,
                continuous_update = False,
                description = 'Upper threshold gradient:',# upper threshold of the gradient of the phase, use for ramp removal
                readout = True,
                style = {'description_width': 'initial'},
                disabled = False),
            
            save_raw = widgets.Checkbox(
                value = False,
                description = 'Save raw data', # True to save the amp-phase.vti before orthogonalizatio
                disabled = False,
                # button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                # icon = 'check',
                style = {'description_width': 'initial'}
                ),
            
            save_support = widgets.Checkbox(
                value = False,
                description = 'Save support',# True to save the non-orthogonal support for later phase retrieval
                disabled = False,
                # button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                # icon = 'check',
                style = {'description_width': 'initial'}
                ),
            
            save = widgets.Checkbox(
                value = True,
                description = 'Save output',  # True to save amp.npz, phase.npz, strain.npz and vtk files
                disabled = False,
                # button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                # icon = 'check',
                style = {'description_width': 'initial'}
                ),
            
            debug = widgets.Checkbox(
                value = False,
                description = 'Debug', # set to True to show all plots for debugging
                disabled = False,
                # icon = 'check',
                style = {'description_width': 'initial'}
                ),
            
            roll_modes = widgets.Text(
                value = "(0, 0, 0)",
                placeholder = "(0, 0, 0)",
                description = 'Roll modes', # axis=(0, 1, 2), correct a roll of few pixels after the decomposition into modes in PyNX
                disabled = False,
                continuous_update = False,
                layout = Layout(width='30%'),
                style = {'description_width': 'initial'},
                ),
            
            label_data_vis = widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>Parameters related to data visualization", # 
                style = {'description_width': 'initial'},
                layout = Layout(width='90%', height = "35px")),
            
            align_axis = widgets.Checkbox(
                value = False,
                description = 'Align axis', # if True rotates the crystal to align axis_to_align along ref_axis after the calculation of the strain
                disabled = False,
                # button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                # icon = 'check',
                style = {'description_width': 'initial'}
                ),
            
            ref_axis = widgets.Dropdown(
                options = ["x", "y", "z"],
                value = "y",
                description = 'Ref axis for align axis',  # will align axis_to_align to that axis
                disabled = False,
                continuous_update = False,
                layout = Layout(width='20%'),
                tooltip = "q will be aligned along that axis",
                style = {'description_width': 'initial'}),
            
            axis_to_align = widgets.Text(
                value = "[0.0, 0.0, 0.0]",
                placeholder = "[0.0, 0.0, 0.0]",
                description = 'Axis to align for ref axis', # axis to align with ref_axis in the order x y z (axis 2, axis 1, axis 0)
                disabled = False,
                continuous_update = False,
                style = {'description_width': 'initial'}),
            
            strain_range = widgets.FloatText(
                value = 0.002,
                step = 0.00001,
                continuous_update = False,
                description = 'Strain range:',
                readout = True,
                style = {'description_width': 'initial'},
                disabled = False),
            
            phase_range = widgets.FloatText(
                value = np.round(np.pi, 3),
                step = 0.001,
                continuous_update = False,
                description = 'Phase range:',
                readout = True,
                style = {'description_width': 'initial'},
                disabled = False),
            
            grey_background = widgets.Checkbox(
                value = True,
                description = 'Grey background in plots',  # True to set the background to grey in phase and strain plots
                disabled = False,
                # button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                # icon = 'check',
                layout = Layout(width='25%'),
                style = {'description_width': 'initial'}
                ),
            
            tick_spacing =  widgets.BoundedIntText(
                value = "100",
                description = 'Tick spacing:',
                min = 0,
                max = 5000,
                layout = Layout(width='25%'),
                disabled = False,
                continuous_update = False,
                style = {'description_width': 'initial'}),
            
            tick_direction = widgets.Dropdown(
                options = ["out", "in", "inout"],
                value = "inout",
                description = 'Tick direction:',
                layout = Layout(width='25%'),
                continuous_update = False,
                disabled = False,
                style = {'description_width': 'initial'}),
            
            tick_length =  widgets.BoundedIntText(
                value = "3",
                description = 'Tick length:',
                min = 0,
                max = 50,
                disabled = False,
                continuous_update = False,
                layout = Layout(width='20%'),
                style = {'description_width': 'initial'}),
            
            tick_width =  widgets.BoundedIntText(
                value = "1",
                description = 'Tick width:',
                min = 0,
                max = 10,
                disabled = False,
                continuous_update = False,
                layout = Layout(width='45%'),
                style = {'description_width': 'initial'}),
            
            label_average = widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>Parameters for averaging several reconstructed objects", # 
                style = {'description_width': 'initial'},
                layout = Layout(width='90%', height = "35px")),
            
            avg_method = widgets.Dropdown(
                options = ["reciprocal_space", "real_space"],
                value = "reciprocal_space",
                description = 'Average method:',
                continuous_update = False,
                disabled = False,
                style = {'description_width': 'initial'}),
            
            avg_threshold = widgets.FloatText(
                value = 0.90,
                step = 0.01,
                continuous_update = False,
                description = 'Average threshold:',  # minimum correlation within reconstructed object for averaging
                readout = True,
                style = {'description_width': 'initial'},
                disabled = False),
            
            label_apodize = widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>Setup for phase averaging or apodization", # 
                style = {'description_width': 'initial'},
                layout = Layout(width='90%', height = "35px")),
                        
            apodize_flag = widgets.Checkbox(
                value = True,
                description = 'Multiply diffraction pattern by filtering window',
                disabled = False,
                # button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                # icon = 'check',
                style = {'description_width': 'initial'}
                ),
            
            apodize_window = widgets.Dropdown(
                options = ["normal", "tukey", "blackman"],
                value = "blackman",
                description = 'Filtering window',
                disabled = False,
                continuous_update = False,
                style = {'description_width': 'initial'}),
            
            hwidth = widgets.BoundedIntText(
                value = 1,
                continuous_update = False,
                description = 'Width of apodizing window:',
                readout = True,
                style = {'description_width': 'initial'},
                disabled = False),

            mu = widgets.Text(
                value = "[0.0, 0.0, 0.0]",
                placeholder = "[0.0, 0.0, 0.0]",
                description = 'Mu of gaussian window',
                disabled = False,
                continuous_update = False,
                style = {'description_width': 'initial'}),
            
            sigma = widgets.Text(
                value = "[0.30, 0.30, 0.30]",
                placeholder = "[0.30, 0.30, 0.30]",
                description = 'Sigma of gaussian window',
                disabled = False,
                continuous_update = False,
                style = {'description_width': 'initial'}),
            
            alpha = widgets.Text(
                value = "[1.0, 1.0, 1.0]",
                placeholder = "[1.0, 1.0, 1.0]",
                description = 'Alpha of gaussian window',
                disabled = False,
                continuous_update = False,
                style = {'description_width': 'initial'}),

            label_strain = widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>Path to file", # 
                style = {'description_width': 'initial'},
                layout = Layout(width='90%', height = "35px")),

            folder_strain = widgets.Text(
                value = os.getcwd(),
                placeholder = os.getcwd(),
                description = 'Data folder:',
                disabled = False,
                continuous_update = False,
                layout = Layout(width='90%'),
                style = {'description_width': 'initial'}),

            reconstruction_file = widgets.Dropdown(
                options = sorted(glob.glob(os.getcwd() + "/*.h5") + glob.glob(os.getcwd() + "/*.cxi") + glob.glob(os.getcwd() + "/*.npy") + glob.glob(os.getcwd() + "/*.npz")),
                description = 'Compatible file list',
                disabled = False,
                layout = Layout(width='90%'),
                style = {'description_width': 'initial'}),

            run_strain = widgets.ToggleButton(
                value = False,
                description = 'Run strain analysis',
                disabled = False,
                button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                icon = 'fast-forward',
                layout = Layout(width='40%'),
                style = {'description_width': 'initial'}),
                )
        self._list_widgets_strain.children[-4].observe(self.folder_strain_handler, names = "value")
        self.tab_strain = widgets.VBox([
            self._list_widgets_strain.children[0],
            widgets.HBox(self._list_widgets_strain.children[1:3]),
            self._list_widgets_strain.children[3],
            widgets.HBox(self._list_widgets_strain.children[4:6]),
            widgets.HBox(self._list_widgets_strain.children[6:8]),
            widgets.HBox(self._list_widgets_strain.children[8:10]),
            self._list_widgets_strain.children[10],
            widgets.HBox(self._list_widgets_strain.children[11:13]),
            widgets.HBox(self._list_widgets_strain.children[13:16]),
            widgets.HBox(self._list_widgets_strain.children[16:20]),
            self._list_widgets_strain.children[20],
            widgets.HBox(self._list_widgets_strain.children[21:23]),
            widgets.HBox(self._list_widgets_strain.children[23:26]),
            self._list_widgets_strain.children[26],
            widgets.HBox(self._list_widgets_strain.children[27:30]),
            widgets.HBox(self._list_widgets_strain.children[30:32]),
            widgets.HBox(self._list_widgets_strain.children[32:35]),
            widgets.HBox(self._list_widgets_strain.children[35:37]),
            self._list_widgets_strain.children[37],
            widgets.HBox(self._list_widgets_strain.children[38:41]),
            widgets.HBox(self._list_widgets_strain.children[41:43]),
            self._list_widgets_strain.children[43],
            widgets.HBox(self._list_widgets_strain.children[44:48]),
            self._list_widgets_strain.children[48],
            widgets.HBox(self._list_widgets_strain.children[49:51]),
            self._list_widgets_strain.children[51],
            widgets.HBox(self._list_widgets_strain.children[52:55]),
            widgets.HBox(self._list_widgets_strain.children[55:58]),
            self._list_widgets_strain.children[-4],
            self._list_widgets_strain.children[-3],
            self._list_widgets_strain.children[-2],
            self._list_widgets_strain.children[-1],
            ])

        # Widgets for logs
        self.tab_logs = interactive(self.display_logs,
            label_logs = widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>Loads csv file and displays it in the gui, saved as self.logs", # 
                style = {'description_width': 'initial'},
                layout = Layout(width='90%', height = "35px")),

            csv_file = widgets.Text(
                value = os.getcwd() + "/metadata.csv",
                placeholder = "Path to csv file",
                description = 'Csv file',
                disabled = False,
                continuous_update = False,
                layout = Layout(width='90%'),
                style = {'description_width': 'initial'}),

            show_logs = widgets.ToggleButton(
                value = False,
                description = 'Show logs',
                disabled = False,
                button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                icon = 'fast-forward',
                layout = Layout(width='40%'),
                style = {'description_width': 'initial'}),
            )

        # Widgets for plotting
        self.tab_data = interactive(self.load_data,
            label_plot = widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>Loads data files (.cxi or npz/npy) and displays it in the gui", # 
                style = {'description_width': 'initial'},
                layout = Layout(width='90%', height = "35px")),

            folder = widgets.Text(
                value = os.getcwd(),
                placeholder = os.getcwd(),
                description = 'Data folder:',
                disabled = False,
                continuous_update = False,
                layout = Layout(width='90%'),
                style = {'description_width': 'initial'}),

            file_list = widgets.Dropdown(
                options = sorted(glob.glob(os.getcwd() + "/*.npz") + glob.glob(os.getcwd() + "/*.cxi") + glob.glob(os.getcwd() + "/*.h5")) + [""],
                description = 'Compatible file list',
                disabled = False,
                layout = Layout(width='90%'),
                style = {'description_width': 'initial'}),

            data_use = widgets.ToggleButtons(
                options = [
                    ("Clear output", False),
                    ('2D plotting', "two_d_plot"),
                    ("Create support", "create_support"),
                    ("Extract support", "extract_support"),
                    ("Smooth support", "smooth_support"),
                    ("3D plotting", "three_d_plot")
                    ],
                value = False,
                description = 'Load data',
                tooltips = [
                    "Clear the output and unload data from gui, saves RAM",
                    "Load data and present plotting options", 
                    "Load data and present support creation options"
                    ],
                disabled = False,
                button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                icon = 'fast-forward',
                layout = Layout(width='90%'),
                style = {'description_width': 'initial'}),
            )
        self.tab_data.children[1].observe(self.folder_plot_handler, names = "value")

        # Widgets for PyNX
        self._list_widgets_pynx = interactive(self.init_pynx,
                    label_data = widgets.HTML(
                        description="<p style='font-weight: bold;font-size:1.2em'>Data files",
                        style = {'description_width': 'initial'},
                        layout = Layout(width='90%', height = "35px")),

                    folder = widgets.Text(
                        value = os.getcwd(),
                        placeholder = os.getcwd(),
                        description = 'Data folder:',
                        disabled = False,
                        continuous_update = False,
                        layout = Layout(width='90%'),
                        style = {'description_width': 'initial'}),

                    iobs = widgets.Dropdown(
                        options = sorted(glob.glob(os.getcwd() + "*.npz")) + [""],
                        description = 'Dataset',
                        disabled = False,
                        layout = Layout(width='90%'),
                        style = {'description_width': 'initial'}),

                    mask = widgets.Dropdown(
                        options = sorted(glob.glob(os.getcwd() + "*.npz")) + [""],
                        description = 'Mask',
                        disabled = False,
                        layout = Layout(width='90%'),
                        style = {'description_width': 'initial'}),

                    support  = widgets.Dropdown(
                        options = sorted(glob.glob(os.getcwd() + "*.npz")) + [""],
                        value = "",
                        description = 'Support',
                        disabled = False,
                        layout = Layout(width='90%'),
                        style = {'description_width': 'initial'}),

                    obj = widgets.Dropdown(
                        options = sorted(glob.glob(os.getcwd() + "*.npz")) + [""],
                        value = "",
                        description = 'Object',
                        disabled = False,
                        layout = Layout(width='90%'),
                        style = {'description_width': 'initial'}),

                    auto_center_resize = widgets.Checkbox(
                        value = False,
                        description = 'Auto center and resize',
                        continuous_update = False,
                        disabled = False,
                        indent = False,
                        layout = Layout(height = "50px"),
                        icon = 'check'),

                    max_size  = widgets.BoundedIntText(
                        value = 256,
                        step = 1,
                        min = 0,
                        max = 1000,
                        layout = Layout(height = "50px", width = "40%"),
                        continuous_update = False,
                        description = 'Maximum array size for cropping:',
                        readout = True,
                        style = {'description_width': 'initial'},
                        disabled = False),

                    label_support = widgets.HTML(
                        description="<p style='font-weight: bold;font-size:1.2em'>Support parameters",
                        style = {'description_width': 'initial'},
                        layout = Layout(width='90%', height = "35px")),

                    support_threshold = widgets.Text(
                        value = "(0.23, 0.30)",
                        placeholder = "(0.23, 0.30)",
                        description = 'Support threshold',
                        disabled = False,
                        layout = Layout(height = "50px", width = "40%"),
                        continuous_update = False,
                        style = {'description_width': 'initial'}),

                    support_only_shrink = widgets.Checkbox(
                        value = False,
                        description = 'Support only shrink',
                        continuous_update = False,
                        disabled = False,
                        indent = False,
                        layout = Layout(height = "50px", width = "15%"),
                        icon = 'check'),

                    support_update_period  = widgets.BoundedIntText(
                        value = 20,
                        step = 5,
                        layout = Layout(height = "50px", width = "25%"),
                        continuous_update = False,
                        description = 'Support update period:',
                        readout = True,
                        style = {'description_width': 'initial'},
                        disabled = False),

                    support_smooth_width =  widgets.Text(
                        value = "(2, 1, 600)",
                        placeholder = "(2, 1, 600)",
                        description = 'Support smooth width',
                        disabled = False,
                        layout = Layout(height = "50px", width = "35%"),
                        continuous_update = False,
                        style = {'description_width': 'initial'}),

                    support_post_expand = widgets.Text(
                        value = "(1, -2, 1)",
                        placeholder = "(1, -2, 1)",
                        description = 'Support post expand',
                        disabled = False,
                        layout = Layout(height = "50px", width = "35%"),
                        continuous_update = False,
                        style = {'description_width': 'initial'}),
                                
                    label_psf = widgets.HTML(
                        description="<p style='font-weight: bold;font-size:1.2em'>Point spread function parameters",
                        style = {'description_width': 'initial'},
                        layout = Layout(width='90%', height = "35px")),

                    psf = widgets.Checkbox(
                        value = False,
                        description = 'Use point spread function:',
                        continuous_update = False,
                        disabled = False,
                        indent = False,
                        layout = Layout(height = "50px"),
                        icon = 'check'),

                    psf_model = widgets.Dropdown(
                        options = ["gaussian", "lorentzian", "pseudo-voigt"],
                        value = "gaussian",
                        description = 'PSF peak shape',
                        continuous_update = False,
                        disabled = True,
                        style = {'description_width': 'initial'}),

                    fwhm = widgets.FloatText(
                        value = 1,
                        step = 0.01,
                        min = 0,
                        continuous_update = False,
                        description = "FWHM:",
                        layout = Layout(width='15%', height = "50px"),
                        readout = True,
                        style = {'description_width': 'initial'},
                        disabled = True),

                    eta = widgets.FloatText(
                        value = 0.05,
                        step = 0.01,
                        max = 1,
                        min = 0,
                        continuous_update = False,
                        description = 'Eta:',
                        layout = Layout(width='15%', height = "50px"),
                        readout = True,
                        style = {'description_width': 'initial'},
                        disabled = True),
                    
                    update_psf = widgets.BoundedIntText(
                        value = 20,
                        step = 5,
                        continuous_update = False,
                        description = 'Update PSF every:',
                        readout = True,
                        style = {'description_width': 'initial'},
                        disabled = True),
                                
                    label_algo = widgets.HTML(
                        description="<p style='font-weight: bold;font-size:1.2em'>Iterative algorithms parameters",
                        style = {'description_width': 'initial'},
                        layout = Layout(width='90%', height = "35px")),
                    
                    use_operators = widgets.Checkbox(
                        value = False,
                        description = 'Use operator chain:',
                        continuous_update = False,
                        disabled = False,
                        indent = False,
                        layout = Layout(height = "35px", width = "15%"),
                        icon = 'check'),
                    
                    operator_chain = widgets.Text(
                        value = "",
                        placeholder = "",
                        description = 'Operator chain',
                        layout = Layout(height = "35px", width = "70%"),
                        disabled = True,
                        continuous_update = False,
                        style = {'description_width': 'initial'}),
                    
                    nb_raar = widgets.BoundedIntText(
                        value = 1000,
                        min = 0,
                        max = 9999,
                        step = 10,
                        continuous_update = False,
                        description = 'Nb of RAAR:',
                        layout = Layout(height = "35px", width = "20%"),
                        readout = True,
                        style = {'description_width': 'initial'},
                        disabled = False),
                    
                    nb_hio = widgets.BoundedIntText(
                        value = 400,
                        min = 0,
                        max = 9999,
                        step = 10,
                        continuous_update = False,
                        description = 'Nb of HIO:',
                        layout = Layout(height = "35px", width = "20%"),
                        readout = True,
                        style = {'description_width': 'initial'},
                        disabled = False),
                    
                    nb_er = widgets.BoundedIntText(
                        value = 300,
                        min = 0,
                        max = 9999,
                        step = 10,
                        continuous_update = False,
                        description = 'Nb of ER:',
                        layout = Layout(height = "35px", width = "20%"),
                        readout = True,
                        style = {'description_width': 'initial'},
                        disabled = False),
                    
                    nb_ml = widgets.BoundedIntText(
                        value = 0,
                        min = 0,
                        max = 9999,
                        step = 10,
                        continuous_update = False,
                        description = 'Nb of ML:',
                        layout = Layout(height = "35px", width = "20%"),
                        readout = True,
                        style = {'description_width': 'initial'},
                        disabled = False),
                    
                    nb_run = widgets.BoundedIntText(
                        value = 30,
                        continuous_update = False,
                        description = 'Number of run:',
                        layout = Layout(height = "50px"),
                        readout = True,
                        style = {'description_width': 'initial'},
                        disabled = False),

                    label_filtering = widgets.HTML(
                        description="<p style='font-weight: bold;font-size:1.2em'>Filtering criteria for reconstructions",
                        style = {'description_width': 'initial'},
                        layout = Layout(width='90%', height = "35px")),
                    
                    filter_criteria = widgets.Dropdown(
                        options = [
                            ("No filtering", "no_filtering"),
                            ("Standard deviation", "standard_deviation"),
                            ("Log-likelihood (LLK)", "LLK"),
                            ("LLK > Standard deviation", "LLK_standard_deviation"),
                            # ("Standard deviation > LLK", "standard_deviation_LLK"),
                            ],
                        value = "LLK_standard_deviation", 
                        description = 'Filtering criteria',
                        disabled = False,
                        layout = Layout(width='90%'),
                        style = {'description_width': 'initial'}),
                    
                    nb_run_keep = widgets.BoundedIntText(
                        value = 10,
                        continuous_update = False,
                        description = 'Number of run to keep:',
                        layout = Layout(height = "50px"),
                        readout = True,
                        style = {'description_width': 'initial'},
                        disabled = False),
                                
                    label_options = widgets.HTML(
                        description="<p style='font-weight: bold;font-size:1.2em'>Options",
                        style = {'description_width': 'initial'},
                        layout = Layout(width='90%', height = "35px")),
                    
                    live_plot = widgets.BoundedIntText(
                        value = 200,
                        step = 10,
                        max = 500,
                        min = 0,
                        continuous_update = False,
                        description = 'Plot every:',
                        readout = True,
                        layout = Layout(height = "50px", width = "20%"),
                        style = {'description_width': 'initial'},
                        disabled = False),
                    
                    positivity = widgets.Checkbox(
                        value = False,
                        description = 'Force positivity',
                        continuous_update = False,
                        disabled = False,
                        indent = False,
                        style = {'description_width': 'initial'},
                        layout = Layout(height = "50px", width = "20%"),
                        icon = 'check'),
                    
                    beta = widgets.FloatText(
                        value = 0.9,
                        step = 0.01,
                        max = 1,
                        min = 0,
                        continuous_update = False,
                        description = 'Beta parameter for RAAR and HIO:',
                        layout = Layout(width='35%', height = "50px"),
                        readout = True,
                        style = {'description_width': 'initial'},
                        disabled = False),
                    
                    detwin = widgets.Checkbox(
                        value = False,
                        description = 'Detwinning',
                        continuous_update = False,
                        disabled = False,
                        indent = False,
                        style = {'description_width': 'initial'},
                        layout = Layout(height = "50px", width = "15%"),
                        icon = 'check'),
                    
                    rebin = widgets.Text(
                        value = "(1, 1, 1)",
                        placeholder = "(1, 1, 1)",
                        description = 'Rebin',
                        layout = Layout(height = "50px"),
                        disabled = False,
                        continuous_update = False,
                        style = {'description_width': 'initial'}),
                    
                    verbose = widgets.BoundedIntText(
                        value = 100,
                        min = 10,
                        max = 300,
                        continuous_update = False,
                        description = 'Verbose:',
                        layout = Layout(height = "50px"),
                        readout = True,
                        style = {'description_width': 'initial'},
                        disabled = False),
                    
                    pixel_size_detector = widgets.BoundedIntText(
                        value = 55,
                        continuous_update = False,
                        description = 'Pixel size of detector (um):',
                        layout = Layout(height = "50px"),
                        readout = True,
                        style = {'description_width': 'initial'},
                        disabled = False),

                    label_phase_retrieval = widgets.HTML(
                        description="<p style='font-weight: bold;font-size:1.2em'>Click below to run the phase retrieval</p>", # 
                        style = {'description_width': 'initial'},
                        layout = Layout(width='90%', height = "35px")),

                    run_phase_retrieval = widgets.ToggleButtons(
                        options = [
                        ('No phase retrieval', False), 
                        ('Run batch job', "batch"), 
                        ("Run script locally", "local_script"), 
                        ("Use operators", "operators"),
                        ],
                        value = False,
                        tooltips = [
                            "Click to be able to change parameters",
                            "Collect parameters to run a job on slurm, will automatically apply a std deviation filter and run modes decomposition, freed the kernel",
                            "Run script on jupyter notebook environment, uses notebook kernel, will be performed in background also but more slowly, good if you cannot use jobs.",
                            "Use operators on local environment, if using PSF, it is activated after 50\% of RAAR cycles"
                            ],
                        description = 'Run phase retrieval ...',
                        disabled = False,
                        continuous_update = False,
                        button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                        layout = Layout(width='100%', height = "50px"),
                        style = {'description_width': 'initial'},
                        icon = 'fast-forward'),

                    label_run_pynx_tools = widgets.HTML(
                        description="<p style='font-weight: bold;font-size:1.2em'>Click below to use a phase retrieval tool</p>", # 
                        style = {'description_width': 'initial'},
                        layout = Layout(width='90%', height = "35px")),

                    run_pynx_tools = widgets.ToggleButtons(
                        options = [
                        ('No tool running', False),
                        ("Modes decomposition","modes"), 
                        ("Filter reconstructions","filter")
                        ],
                        value = False,
                        tooltips = [
                            "Click to be able to change parameters",
                            "Run modes decomposition in data folder, selects *LLK*.cxi files",
                            "Filter reconstructions"
                            ],
                        description = "Choose analysis:",
                        disabled = False,
                        continuous_update = False,
                        button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                        layout = Layout(width='100%', height = "50px"),
                        style = {'description_width': 'initial'},
                        icon = 'fast-forward')
                   )
        self._list_widgets_pynx.children[1].observe(self.folder_pynx_handler, names = "value")
        self._list_widgets_pynx.children[15].observe(self.pynx_psf_handler, names = "value")
        self._list_widgets_pynx.children[16].observe(self.pynx_peak_shape_handler, names = "value")
        self._list_widgets_pynx.children[21].observe(self.pynx_operator_handler, names = "value")
        self._list_widgets_pynx.children[-4].observe(self.run_pynx_handler, names = "value")
        self._list_widgets_pynx.children[-2].observe(self.run_pynx_handler, names = "value")
        self.tab_pynx = widgets.VBox([
            widgets.VBox(self._list_widgets_pynx.children[:6]),
            widgets.HBox(self._list_widgets_pynx.children[6:8]),
            self._list_widgets_pynx.children[8],
            widgets.HBox(self._list_widgets_pynx.children[9:11]),
            widgets.HBox(self._list_widgets_pynx.children[11:14]),
            self._list_widgets_pynx.children[14],
            widgets.HBox(self._list_widgets_pynx.children[15:19]),
            self._list_widgets_pynx.children[19],
            self._list_widgets_pynx.children[20],
            widgets.HBox(self._list_widgets_pynx.children[21:23]),
            widgets.HBox(self._list_widgets_pynx.children[23:27]),
            self._list_widgets_pynx.children[27],
            self._list_widgets_pynx.children[31],
            widgets.HBox(self._list_widgets_pynx.children[32:36]),
            widgets.HBox(self._list_widgets_pynx.children[36:39]),
            self._list_widgets_pynx.children[28],
            widgets.HBox(self._list_widgets_pynx.children[29:31]),
            self._list_widgets_pynx.children[-5],
            self._list_widgets_pynx.children[-4],
            self._list_widgets_pynx.children[-3],
            self._list_widgets_pynx.children[-2],
            self._list_widgets_pynx.children[-1],
            ])

        # Widgets for facet analysis
        self.tab_facet = interactive(self.facet_analysis,
            label_facet = widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>Extract facet specific data from vtk file", # 
                style = {'description_width': 'initial'},
                layout = Layout(width='90%', height = "35px")),

            facet_folder = widgets.Text(
                value = os.getcwd() + f"/postprocessing/",
                placeholder = os.getcwd() + f"/postprocessing/",
                description = 'Facet data (.vtk) folder:',
                disabled = False,
                continuous_update = False,
                layout = Layout(width='90%'),
                style = {'description_width': 'initial'}),

            facet_filename = widgets.Dropdown(
                options = sorted(glob.glob(os.getcwd() + f"/postprocessing/")) + [""],
                description = 'Vtk data',
                disabled = False,
                layout = Layout(width='90%'),
                style = {'description_width': 'initial'}),

            load_data = widgets.ToggleButton(
                value = False,
                description = 'Load vtk data',
                disabled = False,
                button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                icon = 'fast-forward',
                layout = Layout(width='40%'),
                style = {'description_width': 'initial'}),
            )
        self.tab_facet.children[1].observe(self.folder_facet_handler, names = "value")

        # Widgets for readme tab
        self.tab_readme = interactive(self.display_readme, 
                                contents = widgets.ToggleButtons(
                                    options = ['Preprocessing', 'Phase retrieval', 'Postprocessing', "Facet analysis"],
                                    value = 'Phase retrieval',
                                    description = 'Show info about:',
                                    disabled = False,
                                    tooltips = [
                                            'Nothing is shown', 'Insight in the functions used for preprocessing',
                                            'Insight in the functions used for phase retrieval',
                                            'Insight in the functions used for postprocessing'
                                            'Insight in the functions used for facet analysis'
                                            ],
                                    style = {'description_width': 'initial'}))

        # Create the final window
        if pynx_import:
            self.window = widgets.Tab(
                            children=[
                                self.tab_init,
                                self.tab_detector,
                                self.tab_ortho, 
                                self.tab_preprocess,
                                self.tab_correct, 
                                self.tab_logs,
                                self.tab_pynx,
                                self.tab_strain,
                                self.tab_data,
                                self.tab_facet,
                                self.tab_readme,
                            ])
            self.window.set_title(0, 'Scan detail')
            self.window.set_title(1, 'Detector')
            self.window.set_title(2, 'Setup')
            self.window.set_title(3, "Preprocess")
            self.window.set_title(4, 'Correct')
            self.window.set_title(5, 'Logs')
            self.window.set_title(6, 'PyNX')
            self.window.set_title(7, 'Strain')
            self.window.set_title(8, 'Plot data')
            self.window.set_title(9, 'Facets')
            self.window.set_title(10, 'Readme')

            display(self.window)

        elif not pynx_import:
            self.window = widgets.Tab(
                            children=[
                                self.tab_init,
                                self.tab_detector,
                                self.tab_ortho, 
                                self.tab_preprocess,
                                self.tab_correct, 
                                self.tab_logs,
                                # self.tab_pynx,
                                self.tab_strain,
                                self.tab_data,
                                self.tab_facet,
                                self.tab_readme,
                            ])
            self.window.set_title(0, 'Scan detail')
            self.window.set_title(1, 'Detector')
            self.window.set_title(2, 'Setup')
            self.window.set_title(3, "Preprocess")
            self.window.set_title(4, 'Correct')
            self.window.set_title(5, 'Logs')
            # self.window.set_title(6, 'PyNX')
            self.window.set_title(6, 'Strain')
            self.window.set_title(7, 'Plot data')
            self.window.set_title(8, 'Facets')
            self.window.set_title(9, 'Readme')

            display(self.window)

    # Widgets interactive functions
    def initialize_directories(self,
        label_scan,
        sample_name,
        scans,
        data_directory,
        final_directory,
        user_comment,
        debug,
        run_dir_init,
        reload_previous_data,
        ):
        """
        Function to move file from datadir to folder where it will be used by preprocess.bcdi

        Also moves all the notebooks needed for data analysis, and a pynx_run.txt file with all the parameters for phase retrieval,
        initialized for this Dataset.

        Mandatory to run before any other step
        """

        if run_dir_init:
            # Save as attributes for use in future widgets

            # Transform string of list into python list object if multiple scans
            # if scans.startswith("["): # Should not happen in the gui
            #     self.Dataset.scans = ast.literal_eval(scans)
                
            # else:
            #     self.Dataset.scans = [scans]# 
            
            # Create Dataset attribute (class from other module)
            self.Dataset = gui_iterable.Dataset(scans = scans, sample_name = sample_name, data_directory = data_directory, root_folder = final_directory)

            self.Dataset.user_comment = user_comment
            self.Dataset.debug = debug

            # Scan folder
            self.Dataset.scan_folder = self.Dataset.root_folder + f"S{scans}/"
            print("Scan folder:", self.Dataset.scan_folder)
            self.tab_facet.children[1].value = self.Dataset.scan_folder + f"postprocessing/{self.Dataset.scans}_fa.vtk"
            self.tab_data.children[1].value = self.Dataset.scan_folder + f"pynxraw/"
            self._list_widgets_strain.children[-4].value = self.Dataset.scan_folder + f"pynxraw/"
            self._list_widgets_pynx.children[1].value = self.Dataset.scan_folder + f"pynxraw/"

            # Filename for SIXS, should be temporary
            try:
                try:
                    self.Dataset.path_to_data = glob.glob(f"{self.Dataset.data_directory}*mu*{self.Dataset.scans}*")[0]
                    print("File path:", self.Dataset.path_to_data)
                except IndexError:
                        self.Dataset.path_to_data = glob.glob(f"{self.Dataset.data_directory}*omega*{self.Dataset.scans}*")[0]
                        print("Omega scan") 

                # If rotated before
                self.Dataset.template_imagefile = self.Dataset.path_to_data.split("%05d"%self.Dataset.scans)[0]+"%05d.nxs" 
                print("File template:", self.Dataset.template_imagefile, end = "\n\n")

                # Save file name
                self._list_widgets_preprocessing.children[43].value = self.Dataset.template_imagefile.split("/")[-1]

            except IndexError:
                self.Dataset.template_imagefile = ""
                self.Dataset.path_to_data = ""

            # Data folder
            self.Dataset.data_folder = self.Dataset.scan_folder + "data/" # folder of the experiment, where all scans are stored

            # Create final directory is not yet existing
            if not os.path.isdir(self.Dataset.root_folder):
                print(self.Dataset.root_folder)
                full_path = ""
                for d in self.Dataset.root_folder.split("/"):
                    full_path += d + "/"
                    try:
                        os.mkdir(full_path)
                    except FileExistsError:
                        pass

            print(f"Updating directories ...")

            # Scan directory
            try:
                os.mkdir(f"{self.Dataset.root_folder}S{self.Dataset.scans}")
                print(f"Created {self.Dataset.root_folder}S{self.Dataset.scans}")
            except FileExistsError:
                print(f"{self.Dataset.root_folder}S{self.Dataset.scans} exists")
                pass

            # /data directory
            try:
                os.mkdir(f"{self.Dataset.root_folder}S{self.Dataset.scans}/data")
                print(f"Created {self.Dataset.root_folder}S{self.Dataset.scans}/data")
            except FileExistsError:
                print(f"{self.Dataset.root_folder}S{self.Dataset.scans}/data exists")
                pass

            # /pynxraw directory
            try:
                os.mkdir(f"{self.Dataset.root_folder}S{self.Dataset.scans}/pynxraw")
                print(f"Created {self.Dataset.root_folder}S{self.Dataset.scans}/pynxraw")
            except FileExistsError:
                print(f"{self.Dataset.root_folder}S{self.Dataset.scans}/pynxraw exists")
                pass

            # /postprocessing directory
            try:
                os.mkdir(f"{self.Dataset.root_folder}S{self.Dataset.scans}/postprocessing")
                print(f"Created {self.Dataset.root_folder}S{self.Dataset.scans}/postprocessing", end = "\n\n")
            except FileExistsError:
                print(f"{self.Dataset.root_folder}S{self.Dataset.scans}/postprocessing exists", end = "\n\n")
                pass

            # move data file
            try:
                shutil.copy2(self.Dataset.path_to_data, f"{self.Dataset.root_folder}S{self.Dataset.scans}/data")
                print(f"Copied {self.Dataset.path_to_data} to {self.Dataset.root_folder}S{self.Dataset.scans}/data")
            except (FileExistsError,shutil.SameFileError):
                print(f"{self.Dataset.root_folder}S{self.Dataset.scans}/data/{self.Dataset.path_to_data} exists")
            except (AttributeError, FileNotFoundError):
                pass

            # move pynx_run.txt file
            # try:
            #     shutil.copy(f"{self.path_package}bcdi/pynx_run.txt", f"{self.Dataset.root_folder}S{self.Dataset.scans}/pynxraw")
            #     print(f"Copied pynx_run.txt to {self.Dataset.root_folder}S{self.Dataset.scans}/pynxraw")
            # except FileExistsError:
            #     print(f"{self.Dataset.root_folder}S{self.Dataset.scans}/pynxraw/pynx_run.txt exists")
            #     pass

            # Move notebooks
            if not os.path.exists(f"{self.Dataset.root_folder}S{self.Dataset.scans}/pynxraw/PhasingNotebook.ipynb"):
                shutil.copy(f"{self.path_package}bcdi/data_files/PhasingNotebook.ipynb", f"{self.Dataset.root_folder}S{self.Dataset.scans}/pynxraw")
                print(f"Copied PhasingNotebook.ipynb to {self.Dataset.root_folder}S{self.Dataset.scans}/pynxraw")
            else:
                print(f"{self.Dataset.root_folder}S{self.Dataset.scans}/pynxraw/PhasingNotebook.ipynb exists")

            if not os.path.exists(f"{self.Dataset.root_folder}S{self.Dataset.scans}/postprocessing/CompareFacetsEvolution.ipynb"):
                shutil.copy(f"{self.path_package}bcdi/data_files/CompareFacetsEvolution.ipynb", f"{self.Dataset.root_folder}S{self.Dataset.scans}/postprocessing")
                print(f"Copied CompareFacetsEvolution.ipynb to {self.Dataset.root_folder}S{self.Dataset.scans}/postprocessing")
            else:
                print(f"{self.Dataset.root_folder}S{self.Dataset.scans}/postprocessing/CompareFacetsEvolution.ipynb exists")

            self._list_widgets_pynx.children[1].value = self.Dataset.scan_folder + f"pynxraw/"
            self.folder_pynx_handler(change = self._list_widgets_pynx.children[1].value)

            self.tab_data.children[1].value = self.Dataset.scan_folder + f"pynxraw/"
            self.folder_plot_handler(change = self.tab_data.children[1].value)

            self._list_widgets_strain.children[-4].value = self.Dataset.scan_folder + f"pynxraw/"
            self.folder_strain_handler(change = self._list_widgets_strain.children[-4].value)
 
            self.tab_facet.children[1].value = self.Dataset.scan_folder + f"postprocessing/"
            self.folder_facet_handler(change = self.tab_facet.children[1].value)

            # Button to save data
            button_save_as_gwr = Button(
                description = "Save work as .gwr file",
                continuous_update = False,
                button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                layout = Layout(width='40%'),
                style = {'description_width': 'initial'},
                icon = 'fast-forward')
            display(button_save_as_gwr)

            @button_save_as_gwr.on_click
            def action_button_save_as_gwr(selfbutton):
                clear_output(True)
                display(button_save_as_gwr)
                print("Saving data, takes some time ...")
                self.Dataset.to_gwr()

        elif reload_previous_data:
            # Reload previous data that was saved as .gwr file, initialize all related widgets values, authorize all functions 
            print("plop")

        elif not run_dir_init and not reload_previous_data:
            clear_output(True)


    def initialize_parameters(self,
        label_beamline,
        beamline,
        actuators, 
        is_series, 
        custom_scan, 
        custom_images, 
        custom_monitor, 
        specfile_name, 
        rocking_angle, 
        follow_bragg,
        label_masking, 
        flag_interact, 
        background_plot,
        label_centering, 
        centering, 
        fix_bragg, 
        fix_size, 
        center_fft, 
        pad_size,
        normalize_flux, 
        label_filtering, 
        mask_zero_event, 
        flag_medianfilter, 
        medfilt_order, 
        binning,
        label_reload, 
        reload_previous, 
        reload_orthogonal, 
        preprocessing_binning,
        label_saving, 
        save_rawdata, 
        save_to_npz, 
        save_to_mat, 
        save_to_vti, 
        save_asint,
        label_detector, 
        detector, 
        roi_detector,
        photon_threshold, 
        photon_filter, 
        background_file, 
        hotpixels_file, 
        flatfield_file,
        template_imagefile,
        nb_pixel_x, 
        nb_pixel_y,
        label_ortho, 
        use_rawdata, 
        interp_method, 
        fill_value_mask, 
        beam_direction, 
        sample_offsets, 
        sdd, 
        energy, 
        custom_motors,
        label_xru, 
        align_q, 
        ref_axis_q, 
        outofplane_angle, 
        inplane_angle, 
        sample_inplane, 
        sample_outofplane, 
        offset_inplane, 
        cch1, 
        cch2, 
        direct_inplane, 
        direct_outofplane, 
        detrot, 
        tiltazimuth, 
        tilt,
        label_preprocess, 
        init_para
        ):
        """
        Initialize the parameters of using in script taken from bcdi package, necessary for preprocessing, correction and strain

        Will also run preprocessing of the data (cropping, masking, compiling files into .npz files, ...)
        """
        if init_para:
            # Disable all widgets until the end of the program, will update automatticaly after
            for w in self._list_widgets_init.children[:-1]:
                w.disabled = True

            for w in self._list_widgets_preprocessing.children[:-2]:
                w.disabled = True

            for w in self._list_widgets_correct.children[:-2]:
                w.disabled = True

            # Save parameter values as attributes
            self.Dataset.binning = binning
            self.Dataset.flag_interact = flag_interact
            self.Dataset.background_plot = str(background_plot)
            if centering == "manual": # will be overridden
                self.Dataset.centering = "max"
            else:
                self.Dataset.centering = centering                
            self.Dataset.fix_bragg = fix_bragg
            self.Dataset.fix_size = fix_size
            self.Dataset.center_fft = center_fft
            self.Dataset.pad_size = pad_size
            self.Dataset.normalize_flux = normalize_flux
            self.Dataset.mask_zero_event = mask_zero_event
            self.Dataset.flag_medianfilter = flag_medianfilter
            self.Dataset.medfilt_order = medfilt_order
            self.Dataset.reload_previous = reload_previous
            self.Dataset.reload_orthogonal = reload_orthogonal
            self.Dataset.preprocessing_binning = preprocessing_binning
            self.Dataset.save_rawdata = save_rawdata
            self.Dataset.save_to_npz = save_to_npz
            self.Dataset.save_to_mat = save_to_mat
            self.Dataset.save_to_vti = save_to_vti
            self.Dataset.save_asint = save_asint
            self.Dataset.beamline = beamline
            self.Dataset.actuators = actuators
            self.Dataset.is_series = is_series
            self.Dataset.custom_scan = custom_scan
            self.Dataset.custom_images = custom_images
            self.Dataset.custom_monitor = custom_monitor
            self.Dataset.rocking_angle = rocking_angle
            self.Dataset.follow_bragg = follow_bragg
            self.Dataset.specfile_name = specfile_name
            self.Dataset.detector = detector
            self.Dataset.roi_detector = roi_detector
            self.Dataset.photon_threshold = photon_threshold
            self.Dataset.photon_filter = photon_filter
            self.Dataset.background_file = background_file
            self.Dataset.hotpixels_file = hotpixels_file
            self.Dataset.flatfield_file = flatfield_file
            self.Dataset.template_imagefile = template_imagefile
            self.Dataset.nb_pixel_x = nb_pixel_x
            self.Dataset.nb_pixel_y = nb_pixel_y
            self.Dataset.use_rawdata = not use_rawdata
            self.Dataset.interp_method = interp_method
            self.Dataset.fill_value_mask = fill_value_mask
            self.Dataset.beam_direction = beam_direction
            self.Dataset.sample_offsets = sample_offsets
            self.Dataset.sdd = sdd
            self.Dataset.energy = energy
            self.Dataset.custom_motors = custom_motors
            self.Dataset.align_q = align_q
            self.Dataset.ref_axis_q = ref_axis_q
            self.Dataset.outofplane_angle = outofplane_angle
            self.Dataset.inplane_angle = inplane_angle
            self.Dataset.sample_inplane = sample_inplane
            self.Dataset.sample_outofplane = sample_outofplane
            self.Dataset.offset_inplane = offset_inplane
            self.Dataset.cch1 = cch1
            self.Dataset.cch2 = cch2
            self.Dataset.direct_inplane = direct_inplane
            self.Dataset.direct_outofplane = direct_outofplane
            self.Dataset.detrot = detrot
            self.Dataset.tiltazimuth = tiltazimuth
            self.Dataset.tilt = tilt
            self.Dataset.data_dirname = None

            # Extract dict, list and tuple from strings
            list_parameters = ["fix_bragg", "fix_size", "pad_size", "roi_detector"]

            tuple_parameters = ["binning", "preprocessing_binning", "beam_direction", "sample_offsets", "sample_inplane", "sample_outofplane"]

            dict_parameters = ["actuators", "custom_motors"]

            try:
                for p in list_parameters:
                    if getattr(self.Dataset, p) == "":
                        setattr(self.Dataset, p, [])
                    else:
                        setattr(self.Dataset, p, literal_eval(getattr(self.Dataset, p)))
                    # print(f"{p}:", getattr(self.Dataset, p))
            except ValueError:
                print(f"Wrong list syntax for {p}")

            try:
                for p in tuple_parameters:
                    if getattr(self.Dataset, p) == "":
                        setattr(self.Dataset, p, ())
                    else:
                        setattr(self.Dataset, p, literal_eval(getattr(self.Dataset, p)))
                    # print(f"{p}:", getattr(self.Dataset, p))
            except ValueError:
                print(f"Wrong tuple syntax for {p}")

            try:
                for p in dict_parameters:
                    if getattr(self.Dataset, p) == "":
                        setattr(self.Dataset, p, None) # or {}
                    else:
                        if literal_eval(getattr(self.Dataset, p)) == {}:
                            setattr(self.Dataset, p, None)
                        else:
                            setattr(self.Dataset, p, literal_eval(getattr(self.Dataset, p)))
                    # print(f"{p}:", getattr(self.Dataset, p))
            except ValueError:
                print(f"Wrong dict syntax for {p}")

            # Empty parameters are set to None (bcdi syntax)
            if self.Dataset.data_dirname == "":
                self.Dataset.data_dirname = None

            if self.Dataset.background_file == "":
                self.Dataset.background_file = None

            if self.Dataset.hotpixels_file == "":
                self.Dataset.hotpixels_file = None

            if self.Dataset.flatfield_file == "":
                self.Dataset.flatfield_file = None

            if self.Dataset.specfile_name == "":
                self.Dataset.specfile_name = None
                
            if self.Dataset.nb_pixel_x == 0:
                self.Dataset.nb_pixel_x = None

            if self.Dataset.nb_pixel_y == 0:
                self.Dataset.nb_pixel_y = None


            # self.Dataset.roi_detector = [self.Dataset.y_bragg - 160, self.Dataset.y_bragg + 160, self.Dataset.x_bragg - 160, self.Dataset.x_bragg + 160]
            # self.Dataset.roi_detector = None
            # [Vstart, Vstop, Hstart, Hstop]
            # leave it as [] to use the full detector. Use with center_fft='skip' if you want this exact size.

            self.Dataset.linearity_func = None
            print("Parameters initialized...")

            button_run_preprocess = Button(
                description = "Run data preprocessing...",
                continuous_update = False,
                button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                layout = Layout(width='40%'),
                style = {'description_width': 'initial'},
                icon = 'fast-forward')
            display(button_run_preprocess)

            @button_run_preprocess.on_click
            def action_button_run_preprocess(selfbutton):

                # Check is SIXS data, in that case rotate
                if self.Dataset.beamline == "SIXS_2019":
                    self.rotate_sixs_data()
                    root_folder = self.Dataset.root_folder
                    save_dir = None
                    
                if self.Dataset.beamline == "ID01":
                    root_folder = self.Dataset.data_directory
                    save_dir = self.Dataset.root_folder + f"S{self.Dataset.scans}/pynxraw/"

                # On lance BCDI
                preprocess.preprocess_bcdi(
                    scans = self.Dataset.scans,
                    sample_name = self.Dataset.sample_name,
                    root_folder = root_folder,
                    save_dir = save_dir,
                    data_dirname = self.Dataset.data_dirname,
                    user_comment = self.Dataset.user_comment,
                    debug = self.Dataset.debug,
                    binning = self.Dataset.binning,
                    flag_interact = self.Dataset.flag_interact,
                    background_plot = self.Dataset.background_plot,
                    centering = self.Dataset.centering,
                    fix_bragg = self.Dataset.fix_bragg,
                    fix_size = self.Dataset.fix_size,
                    center_fft = self.Dataset.center_fft,
                    pad_size = self.Dataset.pad_size,
                    normalize_flux = self.Dataset.normalize_flux,
                    mask_zero_event = self.Dataset.mask_zero_event,
                    flag_medianfilter = self.Dataset.flag_medianfilter,
                    medfilt_order = self.Dataset.medfilt_order,
                    reload_previous = self.Dataset.reload_previous,
                    reload_orthogonal = self.Dataset.reload_orthogonal,
                    preprocessing_binning = self.Dataset.preprocessing_binning,
                    save_rawdata = self.Dataset.save_rawdata,
                    save_to_npz = self.Dataset.save_to_npz,
                    save_to_mat = self.Dataset.save_to_mat,
                    save_to_vti = self.Dataset.save_to_vti,
                    save_asint = self.Dataset.save_asint,
                    beamline = self.Dataset.beamline,
                    actuators = self.Dataset.actuators,
                    is_series = self.Dataset.is_series,
                    custom_scan = self.Dataset.custom_scan,
                    custom_images = self.Dataset.custom_images,
                    custom_monitor = self.Dataset.custom_monitor,
                    rocking_angle = self.Dataset.rocking_angle,
                    follow_bragg = self.Dataset.follow_bragg,
                    specfile_name = self.Dataset.specfile_name,
                    detector = self.Dataset.detector,
                    linearity_func = self.Dataset.linearity_func,
                    roi_detector = self.Dataset.roi_detector,
                    photon_threshold = self.Dataset.photon_threshold,
                    photon_filter = self.Dataset.photon_filter,
                    background_file = self.Dataset.background_file,
                    hotpixels_file = self.Dataset.hotpixels_file,
                    flatfield_file = self.Dataset.flatfield_file,
                    template_imagefile = self.Dataset.template_imagefile,
                    nb_pixel_x = self.Dataset.nb_pixel_x,
                    nb_pixel_y = self.Dataset.nb_pixel_y,
                    use_rawdata = self.Dataset.use_rawdata,
                    interp_method = self.Dataset.interp_method,
                    fill_value_mask = self.Dataset.fill_value_mask,
                    beam_direction = self.Dataset.beam_direction,
                    sample_offsets = self.Dataset.sample_offsets,
                    sdd = self.Dataset.sdd,
                    energy = self.Dataset.energy,
                    custom_motors = self.Dataset.custom_motors,
                    align_q = self.Dataset.align_q,
                    ref_axis_q = self.Dataset.ref_axis_q,
                    outofplane_angle = self.Dataset.outofplane_angle,
                    inplane_angle = self.Dataset.inplane_angle,
                    sample_inplane = self.Dataset.sample_inplane,
                    sample_outofplane = self.Dataset.sample_outofplane,
                    offset_inplane = self.Dataset.offset_inplane,
                    cch1 = self.Dataset.cch1,
                    cch2 = self.Dataset.cch2,
                    detrot = self.Dataset.detrot,
                    tiltazimuth = self.Dataset.tiltazimuth,
                    tilt = self.Dataset.tilt,
                    GUI=True,
                )

                # Scan folder, refresh
                self._list_widgets_pynx.children[1].value = self.Dataset.scan_folder + f"pynxraw/"
                self.folder_pynx_handler(change = self._list_widgets_pynx.children[1].value)

                self.tab_data.children[1].value = self.Dataset.scan_folder + f"pynxraw/"
                self.folder_plot_handler(change = self.tab_data.children[1].value)

                self._list_widgets_strain.children[-4].value = self.Dataset.scan_folder + f"pynxraw/"
                self.folder_strain_handler(change = self._list_widgets_strain.children[-4].value)
     
                self.tab_facet.children[1].value = self.Dataset.scan_folder + f"postprocessing/"
                self.folder_facet_handler(change = self.tab_facet.children[1].value)


        if not init_para:
            clear_output(True)


    def correct_angles(self,
        label_correct,
        csv_file,
        temp_bool,
        reflection,
        reference_spacing,
        reference_temperature,
        angles_bool,
        ):
        """
        Use this script to extract and save the rocking curve as well as the detector image at the rocking curve's COM
        Will correct the values of the inplane and outofplane angles corresponding to the COM of the Bragg peak, values used then to compute q_hkl
        """
        if angles_bool:
            # # Disable all widgets until the end of the program, will update automatticaly after, no need here because quite fast
            # for w in self._list_widgets_init.children[:-1]:
            #     w.disabled = True

            # for w in self._list_widgets_preprocessing.children[:-1]:
            #     w.disabled = True

            # for w in self._list_widgets_correct.children[:-2]:
            #     w.disabled = True

            # Save parameter values as attributes
            self.Dataset.csv_file = csv_file
            self.Dataset.angles_bool = angles_bool
            self.Dataset.temp_bool = temp_bool
            self.Dataset.reference_spacing = reference_spacing
            self.Dataset.reference_temperature = reference_temperature

            try:
                self.Dataset.reflection = np.array(literal_eval(reflection))
            except ValueError:
                print(f"Wrong list syntax for refelction")

            try:
                # Check beamline for save folder
                if self.Dataset.beamline == "SIXS_2019":
                    root_folder = self.Dataset.root_folder
                    
                elif self.Dataset.beamline == "ID01":
                    root_folder = self.Dataset.data_directory
                
                save_dir = f"{self.Dataset.root_folder}S{self.Dataset.scans}/postprocessing/corrections/"
                    
                # Create final directory is not yet existing
                if not os.path.isdir(save_dir):
                    full_path = ""
                    for d in save_dir.split("/"):
                        full_path += d + "/"
                        try:
                            os.mkdir(full_path)
                        except FileExistsError:
                            pass
            except AttributeError:
                print("Make sure you initialize the parameters by running the data preprocessing...")
                return

            try:
                # On lance la correction
                metadata = correct_angles.correct_angles_detector(
                    filename = self.Dataset.path_to_data,
                    direct_inplane = self.Dataset.direct_inplane,
                    direct_outofplane = self.Dataset.direct_outofplane,
                    get_temperature = self.Dataset.temp_bool,
                    reflection = self.Dataset.reflection,
                    reference_spacing = self.Dataset.reference_spacing, 
                    reference_temperature = self.Dataset.reference_temperature,
                    high_threshold = 1000000,  
                    save_dir = save_dir,
                    scan = self.Dataset.scans,
                    root_folder = root_folder,
                    sample_name = self.Dataset.sample_name,
                    filtered_data = False,
                    peak_method = self.Dataset.centering,
                    normalize_flux = self.Dataset.normalize_flux,
                    debug = self.Dataset.debug,
                    beamline = self.Dataset.beamline,
                    actuators = self.Dataset.actuators,
                    is_series = self.Dataset.is_series,
                    custom_scan = self.Dataset.custom_scan,
                    custom_images = self.Dataset.custom_images,
                    custom_monitor = self.Dataset.custom_monitor,
                    custom_motors = self.Dataset.custom_motors,
                    rocking_angle = self.Dataset.rocking_angle,
                    specfile_name = self.Dataset.specfile_name,
                    detector = self.Dataset.detector,
                    roi_detector = self.Dataset.roi_detector,
                    hotpixels_file = self.Dataset.hotpixels_file, 
                    flatfield_file = self.Dataset.flatfield_file,
                    template_imagefile = self.Dataset.template_imagefile,
                    beam_direction = self.Dataset.beam_direction,
                    sample_offsets = self.Dataset.sample_offsets,
                    directbeam_x = self.Dataset.cch1,
                    directbeam_y = self.Dataset.cch2,
                    sdd = self.Dataset.sdd,
                    energy = self.Dataset.energy,
                    GUI=True
                )

                # Save metadata
                for keys, values in metadata.items():
                    setattr(self.Dataset, keys, values)

                self.extract_metadata()

                # Save corrected angles in the widgets
                print("Saving corrected angles values...")
                self._list_widgets_preprocessing.children[58].value = self.Dataset.bragg_outofplane
                self._list_widgets_preprocessing.children[59].value = self.Dataset.bragg_inplane
                self.Dataset.tilt_angle = np.round(np.mean(self.Dataset.tilt_values[1:] - self.Dataset.tilt_values[:-1]), 4)

            # except ValueError:
            #     print("Inplane or outofplane ?")

            # except TypeError:
            #     print("Make sure you initialize the parameters by running the data preprocessing...")

            except Exception as e:
                raise e

        if not angles_bool:
            clear_output(True)


    def init_pynx(self,
        label_data,
        folder,
        iobs,
        mask,
        support,
        obj,
        auto_center_resize,
        max_size,
        label_support,
        support_threshold,
        support_only_shrink,
        support_update_period,
        support_smooth_width,
        support_post_expand,
        label_psf,
        psf,
        psf_model,
        fwhm,
        eta,
        update_psf,
        label_algo,
        use_operators,
        operator_chain,
        nb_hio,
        nb_raar,
        nb_er,
        nb_ml,
        nb_run,
        label_filtering,
        filter_criteria,
        nb_run_keep,
        label_options,
        live_plot,
        # zero_mask,
        # crop_output,
        positivity,
        beta,
        detwin,
        rebin,
        verbose,
        pixel_size_detector,
        label_phase_retrieval,
        run_phase_retrieval,
        label_run_pynx_tools,
        run_pynx_tools,
        ):
        """
        Get parameters from widgets and run phase retrieval
        """

        self.Dataset.folder = folder
        self.Dataset.iobs = iobs
        self.Dataset.mask = mask
        self.Dataset.support = support
        self.Dataset.obj = obj
        self.Dataset.auto_center_resize = auto_center_resize
        self.Dataset.max_size = max_size
        self.Dataset.support_threshold = support_threshold
        self.Dataset.support_only_shrink = support_only_shrink
        self.Dataset.support_update_period = support_update_period
        self.Dataset.support_smooth_width = support_smooth_width
        self.Dataset.support_post_expand = support_post_expand
        self.Dataset.psf = psf
        self.Dataset.psf_model = psf_model
        self.Dataset.fwhm = fwhm
        self.Dataset.eta = eta
        self.Dataset.update_psf = update_psf
        self.Dataset.use_operators = use_operators
        self.Dataset.operator_chain = operator_chain
        self.Dataset.nb_raar = nb_raar
        self.Dataset.nb_hio = nb_hio
        self.Dataset.nb_er = nb_er
        self.Dataset.nb_ml = nb_ml
        self.Dataset.nb_run = nb_run
        self.Dataset.filter_criteria = filter_criteria
        self.Dataset.nb_run_keep = nb_run_keep
        self.Dataset.live_plot = live_plot
        # self.Dataset.zero_mask = zero_mask
        # self.Dataset.crop_output = crop_output
        self.Dataset.positivity = positivity
        self.Dataset.beta = beta
        self.Dataset.detwin = detwin
        self.Dataset.rebin = rebin
        self.Dataset.verbose = verbose
        self.Dataset.pixel_size_detector = np.round(pixel_size_detector*1e-6, 6)
        self.run_phase_retrieval = run_phase_retrieval
        self.run_pynx_tools = run_pynx_tools

        # Extract dict, list and tuple from strings
        self.Dataset.support_threshold = literal_eval(self.Dataset.support_threshold)
        self.Dataset.support_smooth_width = literal_eval(self.Dataset.support_smooth_width)
        self.Dataset.support_post_expand = literal_eval(self.Dataset.support_post_expand)
        self.Dataset.rebin = literal_eval(self.Dataset.rebin)

        # "" strings should be None
        if self.Dataset.iobs == "":
            self.Dataset.iobs = None
            iobs = None
            
        if self.Dataset.mask == "":
            self.Dataset.mask = None
            mask = None
      
        if self.Dataset.support == "":
            self.Dataset.support = None
            support = None

        if self.Dataset.obj == "":
            self.Dataset.obj = None
            obj = None

        if self.Dataset.live_plot == 0:
            self.Dataset.live_plot = False
        
        print("Scan n°", self.Dataset.scans)

        self.Dataset.energy = self._list_widgets_preprocessing.children[53].value
        self.Dataset.wavelength = 1.2399*1e-6 / self.Dataset.energy
        self.Dataset.sdd = self._list_widgets_preprocessing.children[52].value

        print("CXI input: Energy = %8.2feV" % self.Dataset.energy)
        print(f"CXI input: Wavelength = {self.Dataset.wavelength*1e10} A")
        print("CXI input: detector distance = %8.2fm" % self.Dataset.sdd)
        print(f"CXI input: detector pixel size = {self.Dataset.pixel_size_detector}")

        # PyNX arguments text files
        self.Dataset.pynx_parameter_gui_file = self.Dataset.scan_folder + '/pynxraw/pynx_run_gui.txt'
        self.Dataset.pynx_parameter_cli_file = self.Dataset.scan_folder + '/pynxraw/pynx_run.txt'

        # Phase retrieval
        if self.run_phase_retrieval and not self.run_pynx_tools:
            if self.run_phase_retrieval == "batch" or self.run_phase_retrieval == "local_script":
                self.text_file = []
                self.Dataset.live_plot = False
                
                # Load files
                self.text_file.append("# Parameters\n")
                for file, parameter in [(self.Dataset.iobs, "data"), (self.Dataset.mask, "mask"), (self.Dataset.obj, "object")]:
                    if file:
                        self.text_file.append(f"{parameter} = \"{file}\"\n")
                        
                if support:
                    self.text_file += [
                    f"support = \"{self.Dataset.support}\"\n",
                    '\n']
                # else no support, just don't write it

                # Other support parameters
                self.text_file += [
                    f'support_threshold = {str(self.Dataset.support_threshold).replace("(", "").replace(")", "").replace(" ", "")}\n',
                    f'support_only_shrink = {self.Dataset.support_only_shrink}\n',
                    f'support_update_period = {self.Dataset.support_update_period}\n',
                    f'support_smooth_width_begin = {self.Dataset.support_smooth_width[0]}\n',
                    f'support_smooth_width_end = {self.Dataset.support_smooth_width[1]}\n',
                    f'support_post_expand = {self.Dataset.support_post_expand}\n'
                    '\n',
                    ]

                # PSF
                if self.Dataset.psf:
                    if self.Dataset.psf_model != "pseudo-voigt":
                        self.text_file.append(f"psf = \"{self.Dataset.psf_model},{self.Dataset.fwhm}\"\n")
                        
                    if self.Dataset.psf_model == "pseudo-voigt":
                        self.text_file.append(f"psf = \"{self.Dataset.psf_model},{self.Dataset.fwhm},{self.Dataset.eta}\"\n")
                # no PSF, just don't write anything

                # Filtering the reconstructions
                if self.Dataset.filter_criteria == "LLK":
                    nb_keep_LLK = self.Dataset.nb_run_keep
                    nb_keep_std = False

                elif self.Dataset.filter_criteria == "std":
                    nb_keep_LLK = self.Dataset.nb_run
                    nb_keep_std = self.Dataset.nb_run_keep

                elif self.Dataset.filter_criteria == "LLK_standard_deviation":
                    nb_keep_LLK = self.Dataset.nb_run_keep + (self.Dataset.nb_run - self.Dataset.nb_run_keep) // 2
                    nb_keep_std = self.Dataset.nb_run_keep
          
                # Other parameters
                self.text_file += [
                    'data2cxi = True\n',
                    f'auto_center_resize = {self.Dataset.auto_center_resize}\n',
                    '\n',
                    f'nb_raar = {self.Dataset.nb_raar}\n',
                    f'nb_hio = {self.Dataset.nb_hio}\n',
                    f'nb_er = {self.Dataset.nb_er}\n',
                    f'nb_ml = {self.Dataset.nb_ml}\n',
                    '\n',
                    f'nb_run = {self.Dataset.nb_run}\n',
                    f'nb_run_keep = {nb_keep_LLK}\n',
                    '\n',
                    f'# max_size = {self.Dataset.max_size}\n',
                    'zero_mask = auto # masked pixels will start from imposed 0 and then let free\n',
                    'crop_output= 0 # set to 0 to avoid cropping the output in the .cxi\n',
                    "mask_interp=8,2\n"
                    "confidence_interval_factor_mask=0.5,1.2\n"
                    '\n',
                    f'positivity = {self.Dataset.positivity}\n',
                    f'beta = {self.Dataset.beta}\n',
                    f'detwin = {self.Dataset.detwin}\n',
                    f'rebin = {str(self.Dataset.rebin).replace("(", "").replace(")", "").replace(" ", "")}\n',
                    '\n',
                    '# Generic parameters\n',
                    f'detector_distance = {self.Dataset.sdd}\n',
                    f'pixel_size_detector = {self.Dataset.pixel_size_detector}\n',
                    f'wavelength = {self.Dataset.wavelength}\n',
                    f'verbose = {self.Dataset.verbose}\n',
                    "output_format= 'cxi'\n",
                    f'live_plot = {self.Dataset.live_plot}\n',
                    "mpi=run\n",
                ]
                
                with open(self.Dataset.pynx_parameter_gui_file, "w") as v:
                    for line in self.text_file:
                        v.write(line)
                        
                print(f"Saved parameters in {self.Dataset.pynx_parameter_gui_file}")
                
                if self.run_phase_retrieval == "batch":
                    """
                    Runs modes directly and saves all data in an "all" subdir, filter based on LLK
                    """
                    print(f"\nRunning {self.path_scripts}/run_slurm_job.sh --reconstruct gui --username {self.user_name} --path {self.Dataset.scan_folder}pynxraw --filtering {nb_keep_std} --modes true")
                    print("\nSolution filtering and modes decomposition are automatically applied at the end of the batch job.\n")
                    os.system(f"{self.path_scripts}/run_slurm_job.sh --reconstruct gui --username {self.user_name} --path {self.Dataset.scan_folder}pynxraw --filtering {nb_keep_std} --modes true")
                
                elif self.run_phase_retrieval == "local_script":
                    try:
                        print(f"\nRunning {self.path_scripts}/pynx-id01cdi.py pynx_run_gui.txt 2>&1 | tee README_pynx_local_script.md &", end="\n\n")
                        os.system(f"cd {self.Dataset.scan_folder}pynxraw; {self.path_scripts}/pynx-id01cdi.py pynx_run_gui.txt 2>&1 | tee README_pynx_local_script.md &")
                    except KeyboardInterrupt:
                        print("Phase retrieval stopped by user ...")

            elif self.run_phase_retrieval == "operators":
                # Extract data
                print("Log likelihood is updated every 50 iterations.")

                self.Dataset.calc_llk = 50 # for now

                if self.Dataset.iobs:
                    if self.Dataset.iobs.endswith(".npy"):
                        iobs = np.load(self.Dataset.iobs)
                        print("CXI input: loading data")
                    elif self.Dataset.iobs.endswith(".npz"):
                        try:
                            iobs = np.load(self.Dataset.iobs)["data"]
                            print("CXI input: loading data")
                        except:
                            print("Could not load 'data' array from npz file")

                    if self.Dataset.rebin != (1,1,1):
                        try:
                            iobs = bin_data(iobs, self.Dataset.rebin)
                        except Exception as e:
                            print("Could not bin data")

                    # fft shift
                    iobs = fftshift(iobs)

                if self.Dataset.mask:
                    if self.Dataset.mask.endswith(".npy"):
                        mask = np.load(self.Dataset.mask).astype(np.int8)
                        nb = mask.sum()
                        print("CXI input: loading mask, with %d pixels masked (%6.3f%%)" % (nb, nb * 100 / mask.size))
                    elif self.Dataset.mask.endswith(".npz"):
                        try:
                            mask = np.load(self.Dataset.mask)["mask"].astype(np.int8)
                            nb = mask.sum()
                            print("CXI input: loading mask, with %d pixels masked (%6.3f%%)" % (nb, nb * 100 / mask.size))
                        except:
                            print("Could not load 'mask' array from npz file")            

                    if self.Dataset.rebin != (1,1,1):
                        try:
                            mask = bin_data(mask, self.Dataset.rebin)
                        except Exception as e:
                            print("Could not bin data")

                    # fft shift
                    mask = fftshift(mask)

                if self.Dataset.support:
                    if self.Dataset.support.endswith(".npy"):
                        support = np.load(self.Dataset.support)
                        print("CXI input: loading support")
                    elif self.Dataset.support.endswith(".npz"):
                        try:
                            support = np.load(self.Dataset.support)["data"]
                            print("CXI input: loading support")
                        except:
                            # print("Could not load 'data' array from npz file")
                            try:
                                support = np.load(self.Dataset.support)["support"]
                                print("CXI input: loading support")
                            except:
                                # print("Could not load 'support' array from npz file")
                                try:
                                    support = np.load(self.Dataset.support)["obj"]
                                    print("CXI input: loading support")
                                except:
                                    print("Could not load support")

                    if self.Dataset.rebin != (1,1,1):
                        try:
                            support = bin_data(support, self.Dataset.rebin)
                        except Exception as e:
                            print("Could not bin data")

                    # fft shift
                    support = fftshift(support)

                if self.Dataset.obj:
                    if self.Dataset.obj.endswith(".npy"):
                        obj = np.load(self.Dataset.obj)
                        print("CXI input: loading object")
                    elif self.Dataset.obj.endswith(".npz"):
                        try:
                            obj = np.load(self.Dataset.obj)["data"]
                            print("CXI input: loading object")
                        except:
                            print("Could not load 'data' array from npz file")

                    if self.Dataset.rebin != (1,1,1):
                        try:
                            obj = bin_data(obj, self.Dataset.rebin)
                        except Exception as e:
                            print("Could not bin data")

                    # fft shift
                    obj = fftshift(obj)

                # Center and crop data
                if self.Dataset.auto_center_resize:
                    if iobs.ndim == 3:
                        nz0, ny0, nx0 = iobs.shape

                        # Find center of mass
                        z0, y0, x0 = center_of_mass(iobs)
                        print("Center of mass at:", z0, y0, x0)
                        iz0, iy0, ix0 = int(round(z0)), int(round(y0)), int(round(x0))

                        # Max symmetrical box around center of mass
                        nx = 2 * min(ix0, nx0 - ix0)
                        ny = 2 * min(iy0, ny0 - iy0)
                        nz = 2 * min(iz0, nz0 - iz0)

                        if self.Dataset.max_size is not None:
                            nx = min(nx, self.Dataset.max_size)
                            ny = min(ny, self.Dataset.max_size)
                            nz = min(nz, self.Dataset.max_size)

                        # Crop data to fulfill FFT size requirements
                        nz1, ny1, nx1 = smaller_primes((nz, ny, nx), maxprime=7, required_dividers=(2,))

                        print("Centering & reshaping data: (%d, %d, %d) -> (%d, %d, %d)" % (nz0, ny0, nx0, nz1, ny1, nx1))
                        iobs = iobs[iz0 - nz1 // 2:iz0 + nz1 // 2, iy0 - ny1 // 2:iy0 + ny1 // 2,
                                    ix0 - nx1 // 2:ix0 + nx1 // 2]
                        if mask is not None:
                            mask = mask[iz0 - nz1 // 2:iz0 + nz1 // 2, iy0 - ny1 // 2:iy0 + ny1 // 2,
                                        ix0 - nx1 // 2:ix0 + nx1 // 2]
                            print("Centering & reshaping mask: (%d, %d, %d) -> (%d, %d, %d)" % (nz0, ny0, nx0, nz1, ny1, nx1))

                    else:
                        ny0, nx0 = iobs.shape

                        # Find center of mass
                        y0, x0 = center_of_mass(iobs)
                        iy0, ix0 = int(round(y0)), int(round(x0))
                        print("Center of mass (rounded) at:", iy0, ix0)

                        # Max symmetrical box around center of mass
                        nx = 2 * min(ix0, nx0 - ix0)
                        ny = 2 * min(iy0, ny0 - iy0)
                        if self.Dataset.max_size is not None:
                            nx = min(nx, self.Dataset.max_size)
                            ny = min(ny, self.Dataset.max_size)
                            nz = min(nz, self.Dataset.max_size)

                        # Crop data to fulfill FFT size requirements
                        ny1, nx1 = smaller_primes((ny, nx), maxprime=7, required_dividers=(2,))

                        print("Centering & reshaping data: (%d, %d) -> (%d, %d)" % (ny0, nx0, ny1, nx1))
                        iobs = iobs[iy0 - ny1 // 2:iy0 + ny1 // 2, ix0 - nx1 // 2:ix0 + nx1 // 2]

                        if mask is not None:
                            mask = mask[iy0 - ny1 // 2:iy0 + ny1 // 2, ix0 - nx1 // 2:ix0 + nx1 // 2]
                
                print("\n#############################################################################################################\n")    
                
                try:
                    # Run phase retrieval for nb_run
                    for i in range(self.Dataset.nb_run):
                        print(f"Run {i}")
                        if i>4:
                            print("\nStopping liveplot to go faster\n")
                            self.Dataset.live_plot = False

                        # Create cdi object with data and mask, load the main parameters
                        cdi = CDI(iobs,
                                  support = support,
                                  obj = obj,
                                  mask = mask,
                                  wavelength = self.Dataset.wavelength,
                                  pixel_size_detector = self.Dataset.pixel_size_detector,
                                  detector_distance = self.Dataset.sdd,
                                 )

                        # Save diffraction pattern
                        if i==0:
                            cxi_filename = "{}{}{}/pynxraw/{}.cxi".format(
                                                                        self.Dataset.root_folder,
                                                                        self.Dataset.sample_name,
                                                                        self.Dataset.scans,
                                                                        self.Dataset.iobs.split("/")[-1].split(".")[0]
                                                                        )
                            if not os.path.exists(cxi_filename):
                                cdi.save_data_cxi(
                                    filename = cxi_filename,
                                    sample_name = "",
                                    experiment_id = "",
                                    instrument = ""
                                    )

                        # Change support threshold for supports update
                        if isinstance(self.Dataset.support_threshold, float):
                            self.Dataset.threshold_relative = self.Dataset.support_threshold
                        elif isinstance(self.Dataset.support_threshold, tuple):
                            self.Dataset.threshold_relative = np.random.uniform(self.Dataset.support_threshold[0], self.Dataset.support_threshold[1])
                        print(f"Threshold: {self.Dataset.threshold_relative}")

                        # Create support object
                        sup = SupportUpdate(
                            threshold_relative = self.Dataset.threshold_relative,
                            smooth_width = self.Dataset.support_smooth_width, 
                            force_shrink = self.Dataset.support_only_shrink,
                            method='rms', 
                            post_expand = self.Dataset.support_post_expand,
                            )

                        # Initialize the free pixels for LLK
                        # cdi = InitFreePixels() * cdi

                        # Initialize the support with autocorrelation, if no support given
                        if not self.Dataset.support:
                            sup_init = "autocorrelation"
                            if isinstance(self.Dataset.live_plot, int):
                                if i>4:
                                    cdi = ScaleObj() * AutoCorrelationSupport(
                                        threshold = 0.1, # extra argument
                                        verbose = True) * cdi

                                else:
                                    cdi = ShowCDI() * ScaleObj() * AutoCorrelationSupport(
                                        threshold = 0.1, # extra argument
                                        verbose = True) * cdi

                            else:
                                cdi = ScaleObj() * AutoCorrelationSupport(
                                    threshold = 0.1, # extra argument
                                    verbose = True) * cdi
                        else:
                            sup_init = "support"

                        # Begin with HIO cycles without PSF and with support updates
                        try:
                            # update_psf = 0 probably enough but not sure
                            if self.Dataset.psf:
                                if self.Dataset.support_update_period == 0:
                                    cdi = HIO(
                                            beta = self.Dataset.beta, 
                                            calc_llk = self.Dataset.calc_llk, 
                                            show_cdi = self.Dataset.live_plot
                                            )** self.Dataset.nb_hio * cdi
                                    cdi = RAAR(
                                            beta = self.Dataset.beta, 
                                            calc_llk = self.Dataset.calc_llk, 
                                            show_cdi = self.Dataset.live_plot
                                            )** (self.Dataset.nb_raar//2) * cdi

                                    # PSF is introduced at 66% of HIO and RAAR
                                    if psf_model != "pseudo-voigt":
                                        cdi = InitPSF(
                                                model = self.Dataset.psf_model,
                                                fwhm = self.Dataset.fwhm,
                                                ) * cdi
                                        
                                    elif psf_model == "pseudo-voigt":
                                        cdi = InitPSF(
                                                model = self.Dataset.psf_model,
                                                fwhm = self.Dataset.fwhm,
                                                eta = self.Dataset.eta,
                                                ) * cdi
                                        
                                    cdi = RAAR(
                                            beta = self.Dataset.beta, 
                                            calc_llk = self.Dataset.calc_llk, 
                                            show_cdi = self.Dataset.live_plot, 
                                            update_psf = self.Dataset.update_psf
                                            )** (self.Dataset.nb_raar//2) * cdi
                                    cdi = ER(
                                            calc_llk = self.Dataset.calc_llk, 
                                            show_cdi = self.Dataset.live_plot,
                                            update_psf = self.Dataset.update_psf
                                            )** self.Dataset.nb_er * cdi

                                else:
                                    hio_power = self.Dataset.nb_hio//self.Dataset.support_update_period
                                    raar_power = (self.Dataset.nb_raar//2)//self.Dataset.support_update_period
                                    er_power = self.Dataset.nb_er//self.Dataset.support_update_period

                                    cdi = (sup * HIO(
                                                    beta = self.Dataset.beta, 
                                                    calc_llk = self.Dataset.calc_llk, 
                                                    show_cdi = self.Dataset.live_plot
                                                    )**self.Dataset.support_update_period
                                            )** hio_power * cdi
                                    cdi = (sup * RAAR(
                                                    beta = self.Dataset.beta, 
                                                    calc_llk = self.Dataset.calc_llk, 
                                                    show_cdi = self.Dataset.live_plot
                                                    )**self.Dataset.support_update_period
                                            )** raar_power * cdi

                                    # PSF is introduced at 66% of HIO and RAAR so from cycle n°924
                                    if psf_model != "pseudo-voigt":
                                        cdi = InitPSF(
                                                model = self.Dataset.psf_model,
                                                fwhm = self.Dataset.fwhm,
                                                ) * cdi
                                        
                                    elif psf_model == "pseudo-voigt":
                                        cdi = InitPSF(
                                                model = self.Dataset.psf_model,
                                                fwhm = self.Dataset.fwhm,
                                                eta = self.Dataset.eta,
                                                ) * cdi
                                        
                                    cdi = (sup * RAAR(
                                                    beta = self.Dataset.beta, 
                                                    calc_llk = self.Dataset.calc_llk, 
                                                    show_cdi = self.Dataset.live_plot, 
                                                    update_psf = self.Dataset.update_psf
                                                    )**self.Dataset.support_update_period
                                            )** raar_power * cdi
                                    cdi = (sup * ER(
                                                    calc_llk = self.Dataset.calc_llk, 
                                                    show_cdi = self.Dataset.live_plot,
                                                    update_psf = self.Dataset.update_psf
                                                    )**self.Dataset.support_update_period
                                            )** er_power * cdi

                            if not self.Dataset.psf:
                                if self.Dataset.support_update_period == 0:

                                    cdi = HIO(
                                            beta = self.Dataset.beta, 
                                            calc_llk = self.Dataset.calc_llk, 
                                            show_cdi = self.Dataset.live_plot
                                            )** self.Dataset.nb_hio * cdi
                                    cdi = RAAR(
                                            beta = self.Dataset.beta, 
                                            calc_llk = self.Dataset.calc_llk, 
                                            show_cdi = self.Dataset.live_plot
                                            )** self.Dataset.nb_raar * cdi
                                    cdi = ER(
                                            calc_llk = self.Dataset.calc_llk, 
                                            show_cdi = self.Dataset.live_plot
                                            )** self.Dataset.nb_er * cdi

                                else:
                                    hio_power = self.Dataset.nb_hio//self.Dataset.support_update_period
                                    raar_power = self.Dataset.nb_raar//self.Dataset.support_update_period
                                    er_power = self.Dataset.nb_er//self.Dataset.support_update_period

                                    cdi = (sup * HIO(
                                                    beta = self.Dataset.beta, 
                                                    calc_llk = self.Dataset.calc_llk, 
                                                    show_cdi = self.Dataset.live_plot
                                                    )**self.Dataset.support_update_period
                                            )** hio_power * cdi
                                    cdi = (sup * RAAR(
                                                    beta = self.Dataset.beta, 
                                                    calc_llk = self.Dataset.calc_llk, 
                                                    show_cdi = self.Dataset.live_plot
                                                    )**self.Dataset.support_update_period
                                            )** raar_power * cdi
                                    cdi = (sup * ER(
                                                    calc_llk = self.Dataset.calc_llk, 
                                                    show_cdi = self.Dataset.live_plot
                                                    )**self.Dataset.support_update_period
                                            )** er_power * cdi


                            cdi.save_obj_cxi("{}/result_scan_{}_run_{}_LLK_{:.4}_support_threshold_{:.4}_shape_{}_{}_{}_{}.cxi".format(
                                                                                                            self.Dataset.folder,
                                                                                                            self.Dataset.scans,
                                                                                                            i,
                                                                                                            cdi.get_llk()[0],
                                                                                                            self.Dataset.threshold_relative,
                                                                                                            iobs.shape[0],
                                                                                                            iobs.shape[1],
                                                                                                            iobs.shape[2],
                                                                                                            sup_init,
                                                                                                            )
                                            )

                            print("Saved as result_scan_{}_run_{}_LLK_{:.4}_support_threshold_{:.4}_shape_{}_{}_{}_{}.cxi".format(
                                                                                                            self.Dataset.scans,
                                                                                                            i,
                                                                                                            cdi.get_llk()[0],
                                                                                                            self.Dataset.threshold_relative,
                                                                                                            iobs.shape[0],
                                                                                                            iobs.shape[1],
                                                                                                            iobs.shape[2],
                                                                                                            sup_init,
                                                                                                            )
                                            )
                        except SupportTooLarge:
                            print("Threshold value probably too low, support too large too continue")
                            pass

                        print("\n#############################################################################################################\n")    
                
                    # If filter, filter data
                    if self.Dataset.filter_criteria:
                        self.filter_reconstructions(self.Dataset.folder, self.Dataset.nb_run, self.Dataset.nb_run_keep, self.Dataset.filter_criteria)

                except KeyboardInterrupt:
                    clear_output(True)
                    print("Phase retrieval stopped by user ...")

        # Tools
        if self.run_pynx_tools and not self.run_phase_retrieval:
            if self.run_pynx_tools == "modes":
                self.run_modes_decomposition(self.Dataset.folder)

            elif self.run_pynx_tools == "filter":
                self.filter_reconstructions(self.Dataset.folder, self.Dataset.nb_run, self.Dataset.nb_run_keep, self.Dataset.filter_criteria)

        # Clean output
        if not self.run_phase_retrieval and not self.run_pynx_tools:
            clear_output(True)


    @staticmethod
    def filter_reconstructions(
        folder, 
        nb_run, 
        nb_keep, 
        filter_criteria
        ):
        """
        Filter the phase retrieval output depending on a given parameter, for now only LLK and standard deviation are available.
        This allows the user to run a lot of reconstructions but to then automatically keep the "best" ones, according to this parameter.
        filter_criteria can take the values "LLK" or "standard_deviation"
        If you take filter based on both, the function will filter nb_keep/2 files by the first criteria, and the remaining files by the second criteria 
        """

        # Different functions depending in the criteria
        def filter_by_std(cxi_files, nb_keep):
            # Keep filtering criteria of reconstruction modules in dictionnary
            filtering_criteria_value = {}

            for filename in cxi_files:
                print("Computing standard deviation of object modulus for ", filename)
                with tb.open_file(filename, "r") as f:
                    data = f.root.entry_1.image_1.data[:]
                    filtering_criteria_value[filename] = np.std(np.abs(data))
            
            sorted_dict = sorted(filtering_criteria_value.items(), key=operator_lib.itemgetter(1))

            for f, filtering_criteria_value in sorted_dict[nb_keep:]:
                print(f"Removed scan {f}")
                os.remove(f)

        def filter_by_LLK(cxi_files, nb_keep):
            # Keep filtering criteria of reconstruction modules in dictionnary
            filtering_criteria_value = {}
            
            for filename in cxi_files:
                print("Extracting llk value for poisson statistics for ", filename)
                with tb.open_file(filename, "r") as f:
                    llk = f.root.entry_1.image_1.process_1.results.llk_poisson[...]
                    filtering_criteria_value[filename] = llk

            sorted_dict = sorted(filtering_criteria_value.items(), key=operator_lib.itemgetter(1))

            for f, filtering_criteria_value in sorted_dict[nb_keep:]:
                print(f"Removed scan {f}")
                os.remove(f)

        # Main function, different cases
        try:
            print(f"Iterating on files corresponding to {folder}/*LLK*.cxi")
            cxi_files = sorted(glob.glob(f"{folder}/result_scan*LLK*.cxi"))

            if cxi_files == []:
                print(f"No *LLK*.cxi files in {folder}/result_scan*LLK*.cxi")

            else:
                if filter_criteria == "standard_deviation":
                    filter_by_std(cxi_files, nb_keep)
                    
                elif filter_criteria == "LLK":
                    filter_by_LLK(cxi_files, nb_keep)

                elif filter_criteria == "standard_deviation_LLK":
                    filter_by_std(cxi_files, nb_keep + (nb_run - nb_keep) // 2)

                    print(f"Iterating on remaining files in {folder}/*LLK*.cxi")
                    cxi_files = sorted(glob.glob(f"{folder}/result_scan*LLK*.cxi"))

                    if cxi_files == []:
                        print(f"No *LLK*.cxi files remaining in {folder}/result_scan*LLK*.cxi")
                    else:
                        filter_by_LLK(cxi_files, nb_keep)     

                elif filter_criteria == "LLK_standard_deviation":
                    filter_by_LLK(cxi_files, nb_keep + (nb_run - nb_keep) // 2)

                    print(f"Iterating on remaining files in {folder}/*LLK*.cxi")
                    cxi_files = sorted(glob.glob(f"{folder}/result_scan*LLK*.cxi"))

                    if cxi_files == []:
                        print(f"No *LLK*.cxi files remaining in {folder}/result_scan*LLK*.cxi")
                    else:
                        filter_by_std(cxi_files, nb_keep)

                else:
                    print("No filtering")
        except KeyboardInterrupt:
            print("cxi files filtering stopped by user ...")

    def run_modes_decomposition(self,
        folder
        ):
        """
        Run a decomposition into modes of the phase retrieval solutions, saves only the first mode
        """
        try:
            print(f"Using {self.path_scripts}/pynx-cdi-analysis.py (py38-stable environment)")
            print(f"Using {folder}/*LLK* files.")
            print("Running pynx-cdi-analysis.py *LLK* modes=1")
            print(f"Output in {folder}/modes_gui.h5")
            os.system(f"{self.path_scripts}/pynx-cdi-analysis.py {folder}/*LLK* modes=1 modes_output={folder}/modes_gui.h5")
        except KeyboardInterrupt:
            print("Decomposition into modes stopped by user...")


    def strain_gui(self,
        label_averaging,
        sort_method, 
        correlation_threshold,
        label_FFT,
        phasing_binning, 
        original_size, 
        preprocessing_binning, 
        output_size, 
        keep_size, 
        fix_voxel,
        label_disp_strain,
        data_frame,
        save_frame,
        ref_axis_q,
        isosurface_strain,
        strain_method,
        phase_offset,
        phase_offset_origin,
        offset_method,
        centering_method,
        label_refraction,
        correct_refraction,
        optical_path_method,
        dispersion,
        absorption,
        threshold_unwrap_refraction,
        label_options,
        simu_flag,
        invert_phase,
        flip_reconstruction,
        phase_ramp_removal,
        threshold_gradient,
        save_raw,
        save_support,
        save,
        debug,
        roll_modes,
        label_data_vis,
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
        label_average,
        avg_method,
        avg_threshold,
        label_apodize,
        apodize_flag,
        apodize_window,
        hwidth,
        mu,
        sigma,
        alpha,
        label_strain,
        folder_strain,
        reconstruction_file,
        run_strain,
        ):
        """
        Loading argument from strain tab widgets but also values of parameters used in preprocessing that are common
        Runs postprocessing script from bcdi package to extract the strain from the reconstructed phase. 
        Also plots images depending on the given isosurface.
        """
        
        if run_strain:
            # Disable all widgets until the end of the program, will update automatticaly after
            for w in self._list_widgets_strain.children[:-1]:
                w.disabled = True

            for w in self._list_widgets_preprocessing.children[:-2]:
                w.disabled = True

            for w in self._list_widgets_correct.children[:-1]:
                w.disabled = True

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
            self.Dataset.strain_method = strain_method
            self.Dataset.phase_offset = phase_offset
            self.Dataset.phase_offset_origin = phase_offset_origin
            self.Dataset.offset_method = offset_method
            self.Dataset.centering_method = centering_method
            # pixel_size,
            # parameters related to the refraction correction # 
            self.Dataset.correct_refraction = correct_refraction
            self.Dataset.optical_path_method = optical_path_method
            self.Dataset.dispersion = dispersion
            self.Dataset.absorption = absorption
            self.Dataset.threshold_unwrap_refraction = threshold_unwrap_refraction
            # options #
            self.Dataset.simu_flag = simu_flag
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
            self.Dataset.avg_method = avg_method
            self.Dataset.avg_threshold = avg_threshold
            # setup for phase averaging or apodization
            self.Dataset.hwidth = hwidth
            self.Dataset.apodize_flag = apodize_flag
            self.Dataset.apodize_window = apodize_window
            self.Dataset.mu = mu
            self.Dataset.sigma = sigma
            self.Dataset.alpha = alpha
            self.Dataset.reconstruction_file = reconstruction_file

            # Extract dict, list and tuple from strings
            list_parameters = ["original_size", "output_size", "axis_to_align", "mu", "sigma", "alpha"]

            tuple_parameters = ["phasing_binning", "preprocessing_binning", "phase_offset_origin", "roll_modes"]

            try:
                for p in list_parameters:
                    if getattr(self.Dataset, p) == "":
                        setattr(self.Dataset, p, [])
                    else:
                        setattr(self.Dataset, p, literal_eval(getattr(self.Dataset, p)))
                    # print(f"{p}:", getattr(self.Dataset, p))
            except ValueError:
                print(f"Wrong list syntax for {p}")

            try:
                for p in tuple_parameters:
                    if getattr(self.Dataset, p) == "":
                        setattr(self.Dataset, p, ())
                    else:
                        setattr(self.Dataset, p, literal_eval(getattr(self.Dataset, p)))
                    # print(f"{p}:", getattr(self.Dataset, p))
            except ValueError:
                print(f"Wrong tuple syntax for {p}")


            # Empty parameters are set to None (bcdi syntax)
            if self.Dataset.output_size == []:
                self.Dataset.output_size = None

            if self.Dataset.fix_voxel == 0:
                self.Dataset.fix_voxel = None

            if self.Dataset.phase_offset_origin == ():
                self.Dataset.phase_offset_origin = (None)

            self.Dataset.pixel_size = None

            # Check beamline for save folder
            try:
                if self.Dataset.beamline == "SIXS_2019":
                    root_folder = self.Dataset.root_folder
                
                if self.Dataset.beamline == "ID01":
                    root_folder = self.Dataset.data_directory
                
                save_dir = f"{self.Dataset.root_folder}S{self.Dataset.scans}/result_{self.Dataset.save_frame}/"
            except AttributeError:
                for w in self._list_widgets_strain.children[:-1]:
                    w.disabled = False

                for w in self._list_widgets_preprocessing.children[:-2]:
                    w.disabled = False

                for w in self._list_widgets_correct.children[:-1]:
                    w.disabled = False

                print("You need to initialize all the parameters with the preprocess tab first, some parameters are used here such as the energy, detector distance, ...""")
                return
                    
            # Create final directory is not yet existing
            if not os.path.isdir(save_dir):
                full_path = ""
                for d in save_dir.split("/"):
                    full_path += d + "/"
                    try:
                        os.mkdir(full_path)
                    except FileExistsError:
                        pass
                
            try:
                self.Dataset.strain_output_file, self.Dataset.voxel_size, self.Dataset.q_final = strain.strain_bcdi(
                    scan = self.Dataset.scans, 
                    root_folder = root_folder,
                    save_dir = save_dir,
                    data_dirname = self.Dataset.data_dirname,
                    sample_name = self.Dataset.sample_name, 
                    comment = self.Dataset.user_comment, 
                    sort_method = self.Dataset.sort_method, 
                    correlation_threshold = self.Dataset.correlation_threshold,
                    original_size = self.Dataset.original_size, 
                    phasing_binning = self.Dataset.phasing_binning, 
                    preprocessing_binning = self.Dataset.preprocessing_binning, 
                    output_size = self.Dataset.output_size, 
                    keep_size = self.Dataset.keep_size, 
                    fix_voxel = self.Dataset.fix_voxel,
                    data_frame = self.Dataset.data_frame,
                    ref_axis_q = self.Dataset.ref_axis_q,
                    save_frame = self.Dataset.save_frame,
                    isosurface_strain = self.Dataset.isosurface_strain,
                    strain_method = self.Dataset.strain_method,
                    phase_offset = self.Dataset.phase_offset,
                    phase_offset_origin = self.Dataset.phase_offset_origin,
                    offset_method = self.Dataset.offset_method,
                    centering_method = self.Dataset.centering_method,
                    beamline = self.Dataset.beamline,
                    actuators = self.Dataset.actuators,
                    rocking_angle = self.Dataset.rocking_angle,
                    sdd = self.Dataset.sdd,
                    energy = self.Dataset.energy,
                    beam_direction = self.Dataset.beam_direction,
                    outofplane_angle = self.Dataset.outofplane_angle,
                    inplane_angle = self.Dataset.inplane_angle,
                    tilt_angle = self.Dataset.tilt_angle,
                    sample_offsets = self.Dataset.sample_offsets,
                    specfile_name = self.Dataset.specfile_name,
                    custom_scan = self.Dataset.custom_scan,
                    custom_motors = self.Dataset.custom_motors,
                    detector = self.Dataset.detector,
                    nb_pixel_x = self.Dataset.nb_pixel_x,
                    nb_pixel_y = self.Dataset.nb_pixel_y,
                    pixel_size = self.Dataset.pixel_size,
                    template_imagefile = self.Dataset.template_imagefile,
                    correct_refraction = self.Dataset.correct_refraction,
                    optical_path_method = self.Dataset.optical_path_method,
                    dispersion = self.Dataset.dispersion,
                    absorption = self.Dataset.absorption,
                    threshold_unwrap_refraction = self.Dataset.threshold_unwrap_refraction,
                    simu_flag = self.Dataset.simu_flag,
                    invert_phase = self.Dataset.invert_phase,
                    flip_reconstruction = self.Dataset.flip_reconstruction,
                    phase_ramp_removal = self.Dataset.phase_ramp_removal,
                    threshold_gradient = self.Dataset.threshold_gradient,
                    save_raw = self.Dataset.save_raw,
                    save_support = self.Dataset.save_support,
                    save = self.Dataset.save,
                    debug = self.Dataset.debug,
                    roll_modes = self.Dataset.roll_modes,
                    align_axis = self.Dataset.align_axis,
                    ref_axis = self.Dataset.ref_axis,
                    axis_to_align = self.Dataset.axis_to_align,
                    strain_range = self.Dataset.strain_range,
                    phase_range = self.Dataset.phase_range,
                    grey_background = self.Dataset.grey_background,
                    tick_spacing = self.Dataset.tick_spacing,
                    tick_direction = self.Dataset.tick_direction,
                    tick_length = self.Dataset.tick_length,
                    tick_width = self.Dataset.tick_width,
                    get_temperature = self.Dataset.temp_bool,
                    reflection = self.Dataset.reflection,
                    reference_spacing = self.Dataset.reference_spacing,
                    reference_temperature = self.Dataset.reference_temperature,
                    avg_method = self.Dataset.avg_method,
                    avg_threshold = self.Dataset.avg_threshold,
                    hwidth = self.Dataset.hwidth,
                    apodize_flag = self.Dataset.apodize_flag,
                    apodize_window = self.Dataset.apodize_window,
                    mu = self.Dataset.mu,
                    sigma = self.Dataset.sigma,
                    alpha = self.Dataset.alpha,
                    reconstruction_file = self.Dataset.reconstruction_file,
                    GUI=True,
                )

            except AttributeError:
                print("Run angles correction first, the values of the inplane and outofplane angles are the bragg peak center of mass will be automatically computed.")
            except KeyboardInterrupt:
                print("Strain analysis stopped by user ...")

            finally:
                # At the end of the function 
                self._list_widgets_strain.children[-2].disabled = False

                self.tab_data.children[1].value = self.Dataset.scan_folder + f"pynxraw/"
                self.folder_plot_handler(change = self.tab_data.children[1].value)

        if not run_strain:
            for w in self._list_widgets_strain.children[:-1]:
                w.disabled = False

            for w in self._list_widgets_preprocessing.children[:-2]:
                w.disabled = False

            for w in self._list_widgets_correct.children[:-1]:
                w.disabled = False

            clear_output(True)


    def facet_analysis(self,
        label_facet,
        facet_folder,
        facet_filename,
        load_data,
        ):
        """
        Allows one to:
            load a vtk file (previously created in paraview via theFacetAnalyser plugin)
            realign the particle by assigning a vector to 2 of its facets
            extract information from each facet
        """
        # plt.switch_backend(
        #     'module://ipykernel.pylab.backend_inline'
        # )
        if load_data:
            # Disable text widget to avoid bugs
            self.tab_facet.children[1].disabled = True

            self.Dataset.facet_filename = facet_filename
            fn = self.Dataset.facet_filename.split("/")[-1]
            pathdir = self.Dataset.facet_filename.replace(fn, "")

            try:
                self.Facets = facet_analysis.Facets(filename = fn, pathdir = pathdir)
                print("Facets object saved as self.Facets, call help(self.Facets) for more details.")
            except:
                return "Wrong file."

            # Button to rotate data
            button_rotate = Button(
                description = "Work on facet data",
                continuous_update = False,
                button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                layout = Layout(width='40%'),
                style = {'description_width': 'initial'},
                icon = 'fast-forward')

            # Button to view data
            button_view_particle = Button(
                description = "View particle",
                continuous_update = False,
                button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                layout = Layout(width='40%'),
                style = {'description_width': 'initial'},
                icon = 'fast-forward')

            # Common button as widget
            buttons_facets = widgets.HBox([button_rotate, button_view_particle])

            @button_rotate.on_click
            def action_button_rotate(selfbutton):
                clear_output(True)
                display(buttons_facets)

                # Run interactive function
                @interact(
                    facet_a_id = widgets.Dropdown(
                        options = [i+1 for i in range(self.Facets.nb_facets)],
                        value = 1,
                        description = 'Facet a id:',
                        disabled = False,
                        continuous_update = True,
                        layout = Layout(width='45%'),
                        style = {'description_width': 'initial'}),
                    facet_b_id = widgets.Dropdown(
                        options = [i+1 for i in range(self.Facets.nb_facets)],
                        value = 2,
                        description = 'Facet b id:',
                        disabled = False,
                        continuous_update = True,
                        layout = Layout(width='45%'),
                        style = {'description_width': 'initial'}),
                    u0 = widgets.Text(
                        value = "[1, 1, 1]",
                        placeholder = "[1, 1, 1]",
                        description = 'Vector perpendicular to facet a:',
                        disabled = False,
                        continuous_update = False,
                        # layout = Layout(width='20%'),
                        style = {'description_width': 'initial'},),
                    v0 = widgets.Text(
                        value = "[1, -1, 0]",
                        placeholder = "[1, -1, 0]",
                        description = 'Vector perpendicular to facet b:',
                        disabled = False,
                        continuous_update = False,
                        # layout = Layout(width='20%'),
                        style = {'description_width': 'initial'},),
                    w0 = widgets.Text(
                        value = "[1, 1, -2]",
                        placeholder = "[1, 1, -2]",
                        description = 'Cross product of u0 and v0:',
                        disabled = False,
                        continuous_update = False,
                        # layout = Layout(width='20%'),
                        style = {'description_width': 'initial'},),
                    hkl_reference = widgets.Text(
                        value = "[1, 1, 1]",
                        placeholder = "[1, 1, 1]",
                        description = 'Reference for interplanar angles:',
                        disabled = False,
                        continuous_update = False,
                        # layout = Layout(width='20%'),
                        style = {'description_width': 'initial'},),
                    view = widgets.Text(
                        value = "[90, 0]",
                        placeholder = "[90, 0]",
                        description = 'Elevation and azimuth of the axes in degrees:',
                        disabled = False,
                        continuous_update = False,
                        layout = Layout(width='70%'),
                        style = {'description_width': 'initial'},),
                    )
                def fix_facets(
                    facet_a_id,
                    facet_b_id,
                    u0,
                    v0,
                    w0,
                    hkl_reference,
                    view,
                    ):
                    """Function to interactively visualize the two facets tht will be chosen, to also help pick two vectors"""

                    # Save parameters value
                    self.Facets.facet_a_id = facet_a_id
                    self.Facets.facet_b_id = facet_b_id
                    self.Facets.u0 = u0
                    self.Facets.v0 = v0
                    self.Facets.w0 = w0
                    self.Facets.hkl_reference = hkl_reference
                    self.Facets.view = view

                    # Extract list from strings
                    list_parameters = ["u0", "v0", "w0", "hkl_reference", "view"]
                    try:
                        for p in list_parameters:
                            if getattr(self.Facets, p) == "":
                                setattr(self.Facets, p, [])
                            else:
                                setattr(self.Facets, p, literal_eval(getattr(self.Facets, p)))
                            # print(f"{p}:", getattr(self.Dataset, p))
                    except ValueError:
                        print(f"Wrong list syntax for {p}")

                    # Plot the chosen facet to help the user to pick the facets he wants to use to orient the particule
                    self.Facets.extract_facet(facet_id = self.Facets.facet_a_id, plot = True, view = self.Facets.view, output = False, save = False)
                    self.Facets.extract_facet(facet_id = self.Facets.facet_b_id, plot = True, view = self.Facets.view, output = False, save = False)

                    display(Markdown("""# Field data"""))
                    display(self.Facets.field_data)

                    button_fix_facets = Button(
                        description = "Fix parameters and extract data.",
                        layout = Layout(width='50%', height='35px'))
                    display(button_fix_facets)

                    @button_fix_facets.on_click
                    def action_button_fix_facets(selfbutton):
                        "Fix facets to compute the new rotation matrix and launch the data extraction"

                        clear_output(True)

                        display(button_fix_facets)

                        display(Markdown("""# Computing the rotation matrix"""))

                        # Take those facets' vectors
                        u = np.array([self.Facets.field_data.n0[self.Facets.facet_a_id], self.Facets.field_data.n1[self.Facets.facet_a_id], self.Facets.field_data.n2[self.Facets.facet_a_id]])
                        v = np.array([self.Facets.field_data.n0[self.Facets.facet_b_id], self.Facets.field_data.n1[self.Facets.facet_b_id], self.Facets.field_data.n2[self.Facets.facet_b_id]])

                        self.Facets.set_rotation_matrix(
                            u0 = self.Facets.u0 / np.linalg.norm(self.Facets.u0),
                            v0 = self.Facets.v0 / np.linalg.norm(self.Facets.v0),
                            w0 = self.Facets.w0 / np.linalg.norm(self.Facets.w0),
                            u = u,
                            v = v,
                            )

                        self.Facets.rotate_particle()

                        display(Markdown("""# Computing interplanar angles from reference"""))
                        print(f"Used reference: {self.Facets.hkl_reference}")
                        self.Facets.fixed_reference(hkl_reference = self.Facets.hkl_reference)

                        display(Markdown("""# Strain values for each surface voxel and averaged per facet"""))
                        self.Facets.plot_strain(view = self.Facets.view)

                        display(Markdown("""# Displacement values for each surface voxel and averaged per facet"""))
                        self.Facets.plot_displacement(view = self.Facets.view)

                        display(Markdown("""# Evolution curves"""))
                        self.Facets.evolution_curves()

                        # Also save edges and corners data
                        self.Facets.save_edges_corners_data()

                        display(Markdown("""# Field data"""))
                        display(self.Facets.field_data)

                        button_save_facet_data = Button(
                            description = "Save data",
                            layout = Layout(width='50%', height='35px'))
                        display(button_save_facet_data)

                        @button_save_facet_data.on_click
                        def action_button_save_facet_data(selfbutton):
                            """Save data ..."""
                            try:
                                # Create subfolder
                                try:
                                    os.mkdir(f"{self.Dataset.root_folder}S{self.Dataset.scans}/postprocessing/facets_analysis/")
                                    print(f"Created {self.Dataset.root_folder}S{self.Dataset.scans}/postprocessing/facets_analysis/")
                                except FileExistsError:
                                    print(f"{self.Dataset.root_folder}S{self.Dataset.scans}/postprocessing/facets_analysis/ exists")
                                    pass

                                # Save data
                                self.Facets.save_data(f"{self.Dataset.root_folder}{self.Dataset.sample_name}{self.Dataset.scans}/postprocessing/facets_analysis/field_data_{self.Dataset.scans}.csv")
                                print(f"Saved field data as {self.Dataset.root_folder}{self.Dataset.sample_name}{self.Dataset.scans}/postprocessing/facets_analysis/field_data_{self.Dataset.scans}.csv")

                                self.Facets.to_hdf5(f"{self.Dataset.scan_folder}{self.Dataset.sample_name}{self.Dataset.scans}.h5")
                                print(f"Saved Facets class attributes in {self.Dataset.scan_folder}{self.Dataset.sample_name}{self.Dataset.scans}.h5")
                            except AttributeError:
                                print("Initialize the directories first to save the figures and data ...")


            @button_view_particle.on_click
            def action_button_view_particle(selfbutton):
                clear_output(True)
                display(buttons_facets)

                # Display visualisation window of facet class
                display(self.Facets.window)


            # Display button
            display(buttons_facets)

        if not load_data:
            self.tab_facet.children[1].disabled = False
            clear_output(True)


    def display_readme(self,
        contents):
        """Docs about different steps in data analysis workflow"""

        if contents == "Preprocessing":
            clear_output(True)
            display(Markdown("""
                Prepare experimental data for Bragg CDI phasing: crop/pad, center, mask, normalize and
                filter the data.
                Beamlines currently supported: ESRF ID01, SOLEIL CRISTAL, SOLEIL SIXS, PETRAIII P10 and
                APS 34ID-C.
                Output: data and mask as numpy .npz or Matlab .mat 3D arrays for phasing
                File structure should be (e.g. scan 1):
                specfile, hotpixels file and flatfield file in:    /rootdir/
                data in:                                           /rootdir/S1/data/
                output files saved in:   /rootdir/S1/pynxraw/ or /rootdir/S1/pynx/ depending on the
                'use_rawdata' option
                """))


        if contents == "Phase retrieval":
            clear_output(True)
            display(Markdown("""
                ## Pynx parameters
                `data=data.npy`: name of the data file including the 3D observed intensity.
                               recognized formats include .npy, .npz (if several arrays are included iobs, 
                               should be named 'data' or 'iobs'), .tif or .tiff 
                               (assumes a multiframe tiff image), or .cxi (hdf5).
                               [mandatory unless another beamline-specific method to import data is used]

                `detector_distance=0.7`: detector distance in meters

                `pixel_size_detector=55e-6`: pixel size of the supplied data (detector pixel size)

                `wavelength=1.5e-10`: wavelength in meters

                `verbose=20`: the run will print and optionally plot every 'verbose' cycles

                `live_plot`: if used as keyword (or live_plot=True in a parameters file), a live plot 
                           will be shown  every 'verbose' cycle. If an integer number N is given, 
                           display every N cycle.

                `gpu=Titan`: name of the gpu to use [optional, by default the fastest available will be used]

                `auto_center_resize`: if used (command-line keyword) or =True, the input data will be centered 
                                    and cropped  so that the size of the array is compatible with the (GPU) 
                                    FFT library used. If 'roi' is used, centering is based on ROI. 
                                    [default=False]

                `roi=0,120,0,235,0,270`: set the region-of-interest to be used, with min,max given along each 
                                       dimension in python fashion (min included, max excluded), for the 2 or 3
                                       axis. ROI coordinates should be indicated before any rebin is done.
                                       Note that if 'auto_center_resize' is used, the ROI may still be shrunk
                                       to obtain an array size compatible with the FFT library used. Similarly 
                                       it will be shrunk if 'max_size' is used but ROI size is larger.
                                       Other example: roi=0,-1,300-356,300+256,500-256,500+256
                                       [default=None]

                `nb_run=1`: number of times to run the optimization [default=1]

                `nb_run_keep`: number of best run results to keep, according to likelihood statistics. This is only useful
                             associated with nb_run [default: keep all run results]

                `data2cxi`: if used as keyword (or data2cxi=True in a parameters file), convert the original 
                          data to CXI(HDF5)  format. Will be saved to file 'data.cxi', or if a data file
                          has been supplied (e.g. data57.npz), to the same file with extension .cxi.

                `output_format='cxi'`: choose the output format for the final object and support.
                                     Other possible choice: 'npz', 'none'
                                     [Default='cxi']

                `note`='This Dataset was measure... Citation: Journal of coherent imaging (2018), 729...':
                     Optional text note which will be saved as a note in the output CXI file 
                     (and also for data2cxi).

                `instrument='ESRF-idxx'`: the name of the beamline/instrument used for data collection
                                        [default: depends on the script actually called]

                `sample_name='GaN nanowire'`: optional name for the sample

                `mask=zero`: mask for the diffraction data. If 'zero', all pixels with iobs <= 0 will be masked.
                          If 'negative', all pixels with iobs < 0 will be masked. 
                          If 'maxipix', the maxipix gaps will be masked.
                          Other possibilities: give a filename for the mask file and  import mask from .npy, .npz, .edf, .mat.
                          (the first available array will be used if multiple are present) file.
                          Pixels = 0 are valid, > 0 are masked. If the mask is 2D
                          and the data 3D, the mask is repeated for all frames along the first dimension (depth).
                          [default=None, no mask]

                `iobs_saturation=1e6`: saturation value for the observed intensity. Any pixel above this intensity will be masked
                                     [default: no saturation value]

                `zero_mask`: by default masked pixels are free and keep the calculated intensities during HIO, RAAR, ER and CF cycles.
                           Setting this flag will force all masked pixels to zero intensity. This can be more stable with a large 
                           number of masked pixels and low intensity diffraction data.
                           If a value is supplied the following options can be used:
                           zero_mask=0: masked pixels are free and keep the calculated complex amplitudes
                           zero_mask=1: masked pixels are set to zero
                           zero_mask=auto: this is only meaningful when using a 'standard' algorithm below. The masked pixels will
                                           be set to zero during the first 60% of the HIO/RAAR cycles, and will be free during the 
                                           last 40% and ER, ML ones.

                `object=obj.npy`: starting object. Import object from .npy, .npz, .mat (the first available array 
                      will  be used if multiple are present), or CXI file.
                      [default=None, object will be defined as random values inside the support area]

                `support=sup.npy`: starting support. Import support from .npy, .npz, .edf, .mat (the first 
                          available array will be used if multiple are present) file.  Pixels > 0 are in
                          the support, 0 outside. if 'auto', support will be estimated using the intensity
                          auto-correlation. If 'circle' or 'square', the suppport will be initialized to a 
                          circle (sphere in 3d), or a square (cube).
                          [default='auto', support will be defined otherwise]

                `support_size=50`: size (radius or half-size) for the initial support, to be used in 
                                 combination with 'support_type'. The size is given in pixel units.
                                 Alternatively one value can be given for each dimension, i.e. 
                                 support_size=20,40 for 2D data, and support_size=20,40,60 for 3D data. 
                                 This will result in an initial support which is a rectangle/parallelepiped
                                 or ellipsis/ellipsoid. 
                                 [if not given, this will trigger the use of auto-correlation 
                                  to estimate the initial support]

                `support_autocorrelation_threshold=0.1`: if no support is given, it will be estimated 
                                                       from the intensity autocorrelation, with this relative 
                                                       threshold.
                                                       [default value: 0.1]

                `support_threshold=0.25`: threshold for the support update. Alternatively two values can be given, and the threshold
                                        will be randomly chosen in the interval given by two values: support_threshold=0.20,0.28.
                                        This is mostly useful in combination with nb_run.
                                        [default=0.25]

                `support_threshold_method=max`: method used to determine the absolute threshold for the 
                                              support update. Either:'max' or 'average' (the default) values,
                                              taken over the support area/volume, after smoothing

                `support_only_shrink`: if set or support_only_shrink=True, the support will only shrink 
                                     (default: the support can grow again)

                `support_smooth_width_begin=2`

                `support_smooth_width_end=0.25`: during support update, the object amplitude is convoluted by a
                                               gaussian with a size
                                               (sigma) exponentially decreasing from support_smooth_width_begin
                                               to support_smooth_width_end from the first to the last RAAR or 
                                               HIO cycle.
                                               [default values: 2 and 0.5]

                `support_smooth_width_relax_n`: the number of cycles over which the support smooth width will
                                              exponentially decrease from support_smooth_width_begin to 
                                              support_smooth_width_end, and then stay constant. 
                                              This is ignored if nb_hio, nb_raar, nb_er are used, 
                                              and the number of cycles used
                                              is the total number of HIO+RAAR cycles [default:500]

                `support_post_expand=1`: after the support has been updated using a threshold,  it can be shrunk 
                                       or expanded by a few pixels, either one or multiple times, e.g. in order
                                       to 'clean' the support:
                                       - support_post_expand=1 will expand the support by 1 pixel
                                       - support_post_expand=-1 will shrink the support by 1 pixel
                                       - support_post_expand=-1,1 will shrink and then expand the support 
                                         by 1 pixel
                                       - support_post_expand=-2,3 will shrink and then expand the support 
                                         by 2 and 3 pixels
                                       - support_post_expand=2,-4,2 will expand/shrink/expand the support 
                                         by 2, 4 and 2 pixels
                                       - etc..
                                       [default=None, no shrinking or expansion]

                `support_update_border_n`: if > 0, the only pixels affected by the support updated lie within +/- N pixels around                          the outer border of the support.

                `positivity`: if set or positivity=True, the algorithms will be biased towards a real, positive
                            object. Object is still complex-valued, but random start will begin with real 
                            values. [default=False]

                `beta=0.9`: beta value for the HIO/RAAR algorithm [default=0.9]

                `crop_output=0`: if 1 (the default), the output data will be cropped around the final
                               support plus 'crop_output' pixels. If 0, no cropping is performed.

                `rebin=2`: the experimental data can be rebinned (i.e. a group of n x n (x n) pixels is
                         replaced by a single one whose intensity is equal to the sum of all the pixels).
                         Both iobs and mask (if any) will be rebinned, but the support (if any) should
                         correspond to the new size. The supplied pixel_size_detector should correspond
                         to the original size. The rebin factor can also be supplied as one value per
                         dimension, e.g. "rebin=4,1,2".
                         [default: no rebin]

                `max_size=256`: maximum size for the array used for analysis, along all dimensions. The data
                              will be cropped to this value after centering. [default: no maximum size]

                `user_config*=*`: this can be used to store a custom configuration parameter which will be ignored by the 
                                algorithm, but will be stored among configuration parameters in the CXI file (data and output).
                                e.g.: user_config_temperature=268K  user_config_comment="Vibrations during measurement" etc...

                ### ALGORITHMS: standard version, using RAAR, then HIO, then ER and ML

                `nb_raar=600`: number of relaxed averaged alternating reflections cycles, which the 
                             algorithm will use first. During RAAR and HIO, the support is updated regularly

                `nb_hio=0`: number of hybrid input/output cycles, which the algorithm will use after RAAR. 
                            During RAAR and HIO, the support is updated regularly

                `nb_er=200`: number of error reduction cycles, performed after HIO, without support update

                `nb_ml=20`: number of maximum-likelihood conjugate gradient to perform after ER

                `detwin`: if set (command-line) or if detwin=True (parameters file), 10 cycles will be performed
                        at 25% of the total number of RAAR or HIO cycles, with a support cut in half to bias
                        towards one twin image

                `support_update_period=50`: during RAAR/HIO, update support every N cycles.
                                          If 0, support is never updated.



                ### ALGORITHMS: customized version 

                `algorithm="ER**50,(Sup*ER**5*HIO**50)**10"`: give a specific sequence of algorithms and/or 
                          parameters to be  used for the optimisation (note: this string is case-insensitive).
                #### Important: 
                1. when supplied from the command line, there should be NO SPACE in the expression ! And if there are parenthesis in the expression, quotes are required around the algorithm string
                2. the string and operators are applied from right to left

                #### Valid changes of individual parameters include (see detailed description above):
                `positivity` = 0 or 1

                `support_only_shrink` = 0 or 1

                `beta` = 0.7

                `live_plot` = 0 (no display) or an integer number N to trigger plotting every N cycle

                `support_update_period` = 0 (no update) or a positivie integer number

                `support_smooth_width_begin` = 2.0

                `support_smooth_width_end` = 0.5

                `support_smooth_width_relax_n` = 500

                `support_threshold` = 0.25

                `support_threshold_method`=max or average

                `support_post_expand`=-1#2 (in this case the commas are replaced by # for parsing)

                `zero_mask` = 0 or 1

                `verbose`=20

                `fig_num`=1: change the figure number for plotting


                #### Valid basic operators include:

                `ER`: Error Reduction

                `HIO`: Hybrid Input/Output

                `RAAR`: Relaxed Averaged Alternating Reflections

                `DetwinHIO`: HIO with a half-support (along first dimension)

                `DetwinHIO1`: HIO with a half-support (along second dimension)

                `DetwinHIO2`: HIO with a half-support (along third dimension)

                `DetwinRAAR`: RAAR with a half-support (along first dimension)

                `DetwinRAAR1`: RAAR with a half-support (along second dimension)

                `DetwinRAAR2`: RAAR with a half-support (along third dimension)

                `CF`: Charge Flipping

                `ML`: Maximum Likelihood conjugate gradient (incompatible with partial coherence PSF)

                `PSF` or `EstimatePSF`: calculate partial coherence point-spread function 
                                    with 50 cycles of Richardson-Lucy
                                    
                `Sup` or `SupportUpdate`: update the support according to the support_* parameters

                `ShowCDI`: display of the object and calculated/observed intensity. This can be used
                         to trigger this plot at specific steps, instead of regularly using 
                         live_plot=N. This is thus best used using live_plot=0
                         

                Examples of algorithm strings, where steps are separated with commas (and NO SPACE!),
                and are applied from right to left. Operations in a given step will be applied
                mathematically, also from right to left, and `**N` means repeating N tymes (N cycles) 
                the  operation on the left of the exponent:

                `algorithm=HIO` : single HIO cycle
                `algorithm=ER**100` : 100 cycles of HIO
                `algorithm=ER**50,HIO**100` : 100 cycles of HIO, followed by 50 cycles of ER
                `algorithm=ER**50*HIO**100` : 100 cycles of HIO, followed by 50 cycles of ER
                `algorithm="ER**50,(Sup*ER**5*HIO**50)**10"`: 10 times [50 HIO + 5 ER + Support update], followed by 50 ER
                `algorithm="ER**50,verbose=1,(Sup*ER**5*HIO**50)**10,verbose=100,HIO**100"`: change the periodicity of verbose output
                `algorithm="ER**50,(Sup*ER**5*HIO**50)**10,support_post_expand=1, (Sup*ER**5*HIO**50)**10,support_post_expand=-1#2,HIO**100"`: same but change the post-expand (wrap) method
                `algorithm="ER**50,(Sup*PSF*ER**5*HIO**50)**5,(Sup*ER**5*HIO**50)**10,HIO**100"`: activate partial correlation after a first series of algorithms
                `algorithm="ER**50,(Sup*PSF*HIO**50)**4,(Sup*HIO**50)**8"`: typical algorithm steps with partial coherence
                `algorithm="ER**50,(Sup*HIO**50)**4,(Sup*HIO**50)**4,positivity=0,(Sup*HIO**50)**8,positivity=1"`: same as previous but starting with positivity constraint, removed at the end.

                **Default**: use nb_raar, nb_hio, nb_er and nb_ml to perform the sequence of algorithms]     

                `save=all`: either 'final' or 'all' this keyword will activate saving after each optimisation 
                          step (comma-separated) of the algorithm in any given run [default=final]

                #### Script to perform a CDI reconstruction of data from id01@ESRF.
                command-line/file parameters arguments: (all keywords are case-insensitive):

                `specfile=/some/dir/to/specfile.spec`: path to specfile [mandatory, unless data= is used instead]

                `scan=56`: scan number in specfile [mandatory].
                         Alternatively a list or range of scans can be given:
                            scan=12,23,45 or scan="range(12,25)" (note the quotes)

                `imgcounter=mpx4inr`: spec counter name for image number
                                    [default='auto', will use either 'mpx4inr' or 'ei2mint']

                `imgname=/dir/to/images/prefix%05d.edf.gz`: images location with prefix 
                        [default: will be extracted from the ULIMA_mpx4 entry in the spec scan header]

                #### Specific defaults for this script:
                auto_center_resize = True

                detwin = True

                nb_raar = 600

                nb_hio = 0

                nb_er = 200

                nb_ml = 0

                support_size = None

                zero_mask = auto
                """
                ))


        if contents == "Postprocessing":
            clear_output(True)
            display(Markdown("""
                Interpolate the output of the phase retrieval into an orthonormal frame,
                and calculate the strain component along the direction of the experimental diffusion
                vector q.
                Input: complex amplitude array, output from a phase retrieval program.
                Output: data in an orthonormal frame (laboratory or crystal frame), amp_disp_strain
                array.The disp array should be divided by q to get the displacement (disp = -1*phase
                here).
                Laboratory frame: z downstream, y vertical, x outboard (CXI convention)
                Crystal reciprocal frame: qx downstream, qz vertical, qy outboard
                Detector convention: when out_of_plane angle=0   Y=-y , when in_plane angle=0   X=x
                In arrays, when plotting the first parameter is the row (vertical axis), and the
                second the column (horizontal axis). Therefore the data structure is data[qx, qz,
                qy] for reciprocal space, or data[z, y, x] for real space
                """))


        if contents == "Facet analysis":
            clear_output(True)
            display(Markdown("""
                Import and stores data output of facet analyzer plugin for further analysis.
                Extract strain and displacements at facets, and retrieves the correct facet normals based on a user input.
                Needs a vtk file extracted from the FacetAnalyser plugin of ParaView (see gwaihir repository for further info)
                acknowledgements: mrichard@esrf.fr & maxime.dupraz@esrf.fr
                original tutorial: http://forrestbao.blogspot.com/2011/12/reading-vtk-files-in-python-via-python.html

                vtk file should have been saved in in Sxxxx/postprocessing
                Analysis output in Sxxxx/postprocessing/facet_analysis

                Several plotting options are attributes of this class, feel free to change them (cmap, strain_range, disp_range_avg, disp_range, strain_range_avg, comment, title_fontsize, axes_fontsize, legend_fontsize, ticks_fontsize)
                """))


    def display_logs(self,
        label_logs,
        csv_file,
        show_logs
        ):
        """
        Loads exterior .csv file and displays it in the gui
        """
        self.csv_file = csv_file

        if show_logs:
            self.tab_logs.children[1].disabled = True
            self.logs = pd.read_csv(self.csv_file)
            display(self.logs)
            print("\n################################################################################################################\n")
            print("For a more detailed analysis, please proceed as follows")
            print("import pandas as pd")
            print(f"df = pd.read_csv({self.csv_file})\n")
            print("You can then work on the `df` dataframe as you please.")
            print("\n################################################################################################################\n")

        else:
            self.tab_logs.children[1].disabled = False
            clear_output(True)


    def load_data(self,
        label_plot,
        folder,
        file_list,
        data_use,
        ):
        """
        Allows the user to plot an array (1D, 2D or 3D) from npz, npy or .cxi files.
        """

        if data_use == "two_d_plot":
            # Disable widgets 
           for w in self.tab_data.children[:-2]:
                w.disabled = True

           # Plot data
           plot.Plotter(file_list)

        elif data_use == "create_support":
            # Disable widgets 
            for w in self.tab_data.children[:-2]:
                w.disabled = True
                
            # Initialize class
            sup = SupportTools(path_to_data = file_list)

            # Interactive function to loadt threshold value
            window_support = interactive(sup.compute_support,
                threshold = widgets.FloatText(
                                value = 0.05,
                                step = 0.001,
                                max = 1,
                                min = 0.001,
                                continuous_update = False,
                                description = 'Threshold:',
                                readout = True,
                                layout = Layout(width='20%'),
                                style = {'description_width': 'initial'},
                                disabled = False),
                compute =  widgets.ToggleButton(
                                value = False,
                                description = 'Compute support ...',
                                disabled = False,
                                button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                                icon = 'step-forward',
                                layout = Layout(width='45%'),
                                style = {'description_width': 'initial'})
                )

            def support_handler(change):
                """Handles changes on the widget used for the initialization"""

                if not change.new:
                    window_support.children[0].disabled = False

                if change.new:
                    window_support.children[0].disabled = True

            window_support.children[1].observe(support_handler, names = "value")

            display(window_support)

        elif data_use == "extract_support":
            # Disable widgets 
            for w in self.tab_data.children[:-2]:
                w.disabled = True
                
            # Initialize class
            sup = SupportTools(path_to_data = file_list)

            # Extract the support from the data file and save it as npz
            sup.extract_support()

        elif data_use == "smooth_support":
            # Disable widgets 
            for w in self.tab_data.children[:-2]:
                w.disabled = True
                
            # Initialize class
            sup = SupportTools(path_to_support = file_list)

            # Interactive function to loadt threshold value
            window_support = interactive(sup.gaussian_convolution,
                sigma = widgets.FloatText(
                                value = 0.05,
                                step = 0.001,
                                max = 1,
                                min = 0.001,
                                continuous_update = False,
                                description = 'Sigma:',
                                readout = True,
                                layout = Layout(width='20%'),
                                style = {'description_width': 'initial'}),
                threshold = widgets.FloatText(
                                value = 0.05,
                                step = 0.001,
                                max = 1,
                                min = 0.001,
                                continuous_update = False,
                                description = 'Threshold:',
                                readout = True,
                                layout = Layout(width='20%'),
                                style = {'description_width': 'initial'}),
                compute =  widgets.ToggleButton(
                                value = False,
                                description = 'Compute support ...',
                                button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                                icon = 'step-forward',
                                layout = Layout(width='45%'),
                                style = {'description_width': 'initial'})
                )

            def support_handler(change):
                """Handles changes on the widget used for the initialization"""

                if not change.new:
                    window_support.children[0].disabled = False

                if change.new:
                    window_support.children[0].disabled = True

            window_support.children[1].observe(support_handler, names = "value")

            display(window_support)

        # elif data_use == "plot_vtk":
        #     # Disable widgets 
        #     for w in self.tab_data.children[:-2]:
        #         w.disabled = True
                
        elif data_use == "three_d_plot":
            for w in self.tab_data.children[:-2]:
                w.disabled = True
            w = CDIViewer(file_list)
            display(w)

        elif data_use == False:
            for w in self.tab_data.children[:-2]:
                w.disabled = False
            clear_output(True)


    # Non widgets functions
    def rotate_sixs_data(self):
        """
        Python script to rotate the data for vertical configuration
                Arg 1: Path of target directory (before /S{scan} ... )
                Arg 2: Scan(s) number, list or single value

        Only for SIXS data in the vertical MED configuration
        """

        print("Rotating SIXS data ...")
        with tb.open_file(self.Dataset.path_to_data, "a") as f:
            # Get data
            try:
                # if rocking_angle == "omega":
                data_og = f.root.com.scan_data.data_02[:]
                index = 2
                if np.ndim(data_og)==1:
                    data_og = f.root.com.scan_data.data_10[:]
                    index = 10
                # elif rocking_angle == "mu":
                #     data_og = f.root.com.scan_data.merlin_image[:]
                print("Calling merlin the enchanter in SBS...")
                self.Dataset.scan_type = "SBS"
            except:
                try:
                    data_og = f.root.com.scan_data.self_image[:]
                    print("Calling merlin the enchanter in FLY...")
                    self.Dataset.scan_type = "FLY"
                except:
                    print("This data does not result from Merlin :/")

            # Just an index for plotting schemes
            half = int(data_og.shape[0]/2)

            # Rotate data
            data = np.transpose(data_og, axes=(0, 2, 1))
            for idx in range(data.shape[0]):
                tmp = data[idx, :, :]
                data[idx, :, :] = np.fliplr(tmp)
            print("Data well rotated by 90°.")  

            print("Saving example figures...", end="\n\n")
            plt.figure(figsize = (16, 9))
            plt.imshow(data_og[half, :, :], vmax = 10)
            plt.xlabel('Delta')
            plt.ylabel('Gamma')
            plt.tight_layout()
            plt.savefig(self.Dataset.root_folder + self.Dataset.sample_name + str(self.Dataset.scans) + "/data/data_before_rotation.png")
            plt.close()

            plt.figure(figsize = (16, 9))        
            plt.imshow(data[half, :, :], vmax = 10)
            plt.xlabel('Gamma')
            plt.ylabel('Delta')
            plt.tight_layout()
            plt.savefig(self.Dataset.root_folder + self.Dataset.sample_name + str(self.Dataset.scans) + "/data/data_after_rotation.png")
            plt.close()

            # Overwrite data in copied file
            try:
                if self.Dataset.scan_type == "SBS" and index == 2:
                    f.root.com.scan_data.data_02[:] = data
                elif self.Dataset.scan_type == "SBS" and index == 10:
                    f.root.com.scan_data.data_10[:] = data
                elif self.Dataset.scan_type == "FLY":
                    f.root.com.scan_data.test_image[:] = data
            except:
                print("Could not overwrite data ><")


    def extract_metadata(self):
        """
        Needs Dataset to be corrected beforehand
        Extract meaningful data and saves them in a csv file to allow comparison
        """

        # Save rocking curve data
        np.savez(f"{self.Dataset.scan_folder}postprocessing/interpolated_rocking_curve.npz",
            tilt_values = self.Dataset.tilt_values,
            rocking_curve = self.Dataset.rocking_curve,
            interp_tilt = self.Dataset.interp_tilt,
            interp_curve = self.Dataset.interp_curve,
            )

        # Save in a csv file
        try:
            if self.Dataset.beamline == "SIXS_2019":
                # Load Dataset, quite slow 
                data = rd.DataSet(self.Dataset.path_to_data)

                ## Add new data
                temp_df = pd.DataFrame([[
                    self.Dataset.scans,
                    self.Dataset.q[0], self.Dataset.q[1], self.Dataset.q[2], self.Dataset.qnorm, self.Dataset.dist_plane,
                    self.Dataset.bragg_inplane, self.Dataset.bragg_outofplane,
                    self.Dataset.bragg_x, self.Dataset.bragg_y,
                    data.x[0], data.y[0], data.z[0], data.mu[0], data.delta[0], data.omega[0],
                    data.gamma[0], data.gamma[0] - data.mu[0], 
                    (data.mu[-1] - data.mu[-0]) / len(data.mu), data.integration_time[0], len(data.integration_time), 
                    self.Dataset.interp_fwhm, self.Dataset.COM_rocking_curve,
                    data.ssl3hg[0], data.ssl3vg[0], 
                    # data.ssl1hg[0], data.ssl1vg[0]
                    ]],
                    columns = [
                        "scan",
                        "qx", "qy", "qz", "q_norm", "d_hkl", 
                        "inplane_angle", "out_of_plane_angle",
                        "bragg_x", "bragg_y",
                        "x", "y", "z", "mu", "delta", "omega",
                        "gamma", 'gamma-mu',
                        "step size", "integration time", "steps", 
                        "FWHM", "COM_rocking_curve",
                        "ssl3hg", "ssl3vg", 
                    #     "ssl1hg", "ssl1vg", 
                    ])
            else:
                ## Add new data
                temp_df = pd.DataFrame([[
                    self.Dataset.scans,
                    self.Dataset.q[0], self.Dataset.q[1], self.Dataset.q[2], self.Dataset.qnorm, self.Dataset.dist_plane,
                    self.Dataset.bragg_inplane, self.Dataset.bragg_outofplane,
                    self.Dataset.bragg_x, self.Dataset.bragg_y,
                    self.Dataset.interp_fwhm, self.Dataset.COM_rocking_curve,
                    ]],
                    columns = [
                        "scan",
                        "qx", "qy", "qz", "q_norm", "d_hkl", 
                        "inplane_angle", "out_of_plane_angle",
                        "bragg_x", "bragg_y",
                        "FWHM", "COM_rocking_curve",
                    ])

            # Load all the logs
            try:
                df = pd.read_csv(self.csv_file)

                # Replace old data linked to this scan, no problem if this row does not exist yet
                indices = df[df['scan'] == self.Dataset.scans].index
                df.drop(indices , inplace=True)

                result = pd.concat([df, temp_df])

            except FileNotFoundError:
                result = temp_df

            # Save
            result.to_csv(self.csv_file, index = False)
            print(f"Saved logs in {self.csv_file}")

        except Exception as e:
            raise e
            # print("Could not extract metadata from Dataset ...")


    # Below are handlers
    def init_handler(self, change):
        """Handles changes on the widget used for the initialization"""

        if not change.new:
            for w in self._list_widgets_init.children[:7]:
                w.disabled = False

            for w in self._list_widgets_preprocessing.children[:-1]:
                w.disabled = True

        if change.new:
            for w in self._list_widgets_init.children[:7]:
                w.disabled = True

            for w in self._list_widgets_preprocessing.children[:-1]:
                w.disabled = False

            self.beamline_handler(change = self._list_widgets_preprocessing.children[1].value)
            self.energy_scan_handler(change = self._list_widgets_preprocessing.children[8].value)
            self.bragg_peak_centering_handler(change = self._list_widgets_preprocessing.children[14].value)
            self.reload_data_handler(change = self._list_widgets_preprocessing.children[26].value)
            self.interpolation_handler(change = self._list_widgets_preprocessing.children[47].value)

    def beamline_handler(self, change):
        "Handles changes on the widget used for the initialization"
        try:
            if change.new in ["SIXS_2019", "ID01"]:
                for w in self._list_widgets_preprocessing.children[2:7]:
                    w.disabled = True

            if change.new not in ["SIXS_2019", "ID01"]:
                for w in self._list_widgets_preprocessing.children[2:7]:
                    w.disabled = False
        except AttributeError:
            if change in ["SIXS_2019", "ID01"]:
                for w in self._list_widgets_preprocessing.children[2:7]:
                    w.disabled = True

            if change not in ["SIXS_2019", "ID01"]:
                for w in self._list_widgets_preprocessing.children[2:7]:
                    w.disabled = False   

    def energy_scan_handler(self, change):
        "Handles changes related to energy scans"
        try:
            if change.new == "energy":
                self._list_widgets_preprocessing.children[9].disabled = False

            if change.new != "energy":
                self._list_widgets_preprocessing.children[9].disabled = True
        except AttributeError:
            if change == "energy":
                self._list_widgets_preprocessing.children[9].disabled = False

            if change != "energy":
                self._list_widgets_preprocessing.children[9].disabled = True

    def bragg_peak_centering_handler(self, change):
        "Handles changes related to the centering of the Bragg peak"
        try:
            if change.new == "manual":
                self._list_widgets_preprocessing.children[15].disabled = False

            if change.new != "manual":
                self._list_widgets_preprocessing.children[15].disabled = True

        except AttributeError:
            if change == "manual":
                self._list_widgets_preprocessing.children[15].disabled = False

            if change != "manual":
                self._list_widgets_preprocessing.children[15].disabled = True

    def reload_data_handler(self, change):
        "Handles changes related to data reloading"
        try:
            if change.new:
                for w in self._list_widgets_preprocessing.children[27:29]:
                    w.disabled = False

            if not change.new:
                for w in self._list_widgets_preprocessing.children[27:29]:
                    w.disabled = True

        except AttributeError:
            if change:
                for w in self._list_widgets_preprocessing.children[27:29]:
                    w.disabled = False

            if not change:
                for w in self._list_widgets_preprocessing.children[27:29]:
                    w.disabled = True

    def interpolation_handler(self, change):
        "Handles changes related to data interpolation"
        try:
            if change.new:
                for w in self._list_widgets_preprocessing.children[48:70]:
                    w.disabled = False

            if not change.new:
                for w in self._list_widgets_preprocessing.children[48:70]:
                    w.disabled = True
        except AttributeError:
            if change:
                for w in self._list_widgets_preprocessing.children[48:70]:
                    w.disabled = False

            if not change:
                for w in self._list_widgets_preprocessing.children[48:70]:
                    w.disabled = True

    def preprocess_handler(self, change):
        "Handles changes on the widget used for the initialization"
        try:
            if not change.new:
                self._list_widgets_init.children[7].disabled = False

                for w in self._list_widgets_preprocessing.children[:-2]:
                    w.disabled = False

                for w in self._list_widgets_correct.children[:-1]:
                    w.disabled = True

                self.beamline_handler(change = self._list_widgets_preprocessing.children[1].value)
                self.energy_scan_handler(change = self._list_widgets_preprocessing.children[8].value)
                self.bragg_peak_centering_handler(change = self._list_widgets_preprocessing.children[14].value)
                self.reload_data_handler(change = self._list_widgets_preprocessing.children[26].value)
                self.interpolation_handler(change = self._list_widgets_preprocessing.children[47].value)

            if change.new:
                self._list_widgets_init.children[7].disabled = True

                for w in self._list_widgets_preprocessing.children[:-2]:
                    w.disabled = True

                for w in self._list_widgets_correct.children[:-1]:
                    w.disabled = False

                self.temp_handler(change = self._list_widgets_correct.children[2].value)

        except:
            if not change:
                self._list_widgets_init.children[7].disabled = False

                for w in self._list_widgets_preprocessing.children[:-2]:
                    w.disabled = False

                for w in self._list_widgets_correct.children[:3]:
                    w.disabled = True

                self.beamline_handler(change = self._list_widgets_preprocessing.children[1].value)
                self.energy_scan_handler(change = self._list_widgets_preprocessing.children[8].value)
                self.bragg_peak_centering_handler(change = self._list_widgets_preprocessing.children[14].value)
                self.reload_data_handler(change = self._list_widgets_preprocessing.children[26].value)
                self.interpolation_handler(change = self._list_widgets_preprocessing.children[47].value)

            if change:
                self._list_widgets_init.children[7].disabled = True

                for w in self._list_widgets_preprocessing.children[:-2]:
                    w.disabled = True

                for w in self._list_widgets_correct.children[:-1]:
                    w.disabled = False

                self.temp_handler(change = self._list_widgets_correct.children[2].value)

    def temp_handler(self, change):
        "Handles changes related to data interpolation"
        try:
            if change.new:
                for w in self._list_widgets_correct.children[3:6]:
                    w.disabled = False

            if not change.new:
                for w in self._list_widgets_correct.children[3:6]:
                    w.disabled = True
        except:
            if change:
                for w in self._list_widgets_correct.children[3:6]:
                    w.disabled = False

            if not change:
                for w in self._list_widgets_correct.children[3:6]:
                    w.disabled = True

    def correct_angles_handler(self, change):
        "Handles changes related to data interpolation"
        try:
            if change.new:
                for w in self._list_widgets_correct.children[:-2]:
                    w.disabled = True

            if not change.new:
                for w in self._list_widgets_correct.children[:-2]:
                    w.disabled = False

                self.temp_handler(change = self._list_widgets_correct.children[2].value)

        except AttributeError:
            if change:
                for w in self._list_widgets_correct.children[:-2]:
                    w.disabled = True

            if not change:
                for w in self._list_widgets_correct.children[:-2]:
                    w.disabled = False

                self.temp_handler(change = self._list_widgets_correct.children[2].value)

    def folder_pynx_handler(self, change):
        """Handles changes on the widget used to load a data file"""
        try:
            list_all_npz = sorted(glob.glob(change.new + "/*.npz"))
            list_probable_iobs_files = sorted(glob.glob(change.new + "/*_pynx_align*.npz"))
            list_probable_mask_files = sorted(glob.glob(change.new + "/*maskpynx*.npz"))

            for f in list_probable_iobs_files:
                try:
                    list_all_npz.remove(f)
                except ValueError:
                    # not in list
                    pass
            sorted_iobs_list = list_probable_iobs_files + list_all_npz + [""]

            list_all_npz = sorted(glob.glob(change.new + "/*.npz"))
            for f in list_probable_mask_files:
                try:
                    list_all_npz.remove(f)
                except ValueError:
                    # not in list
                    pass
            sorted_mask_list = list_probable_mask_files + list_all_npz + [""]

            self._list_widgets_pynx.children[2].options = sorted_iobs_list # iobs list
            self._list_widgets_pynx.children[3].options = sorted_mask_list # mask list
            self._list_widgets_pynx.children[4].options = [""] + sorted(glob.glob(change.new + "/*.npz")) # support list
            self._list_widgets_pynx.children[5].options = [""] + sorted(glob.glob(change.new + "/*.npz")) # obj list

        except AttributeError:
            list_all_npz = sorted(glob.glob(change + "/*.npz"))
            list_probable_iobs_files = sorted(glob.glob(change + "/*_pynx_align*.npz"))
            list_probable_mask_files = sorted(glob.glob(change + "/*maskpynx*.npz"))

            for f in list_probable_iobs_files:
                try:
                    list_all_npz.remove(f)
                except ValueError:
                    # not in list
                    pass
            sorted_iobs_list = list_probable_iobs_files + list_all_npz + [""]

            list_all_npz = sorted(glob.glob(change + "/*.npz"))
            for f in list_probable_mask_files:
                try:
                    list_all_npz.remove(f)
                except ValueError:
                    # not in list
                    pass
            sorted_mask_list = list_probable_mask_files + list_all_npz + [""]

            self._list_widgets_pynx.children[2].options = sorted_iobs_list # iobs list
            self._list_widgets_pynx.children[3].options = sorted_mask_list # mask list
            self._list_widgets_pynx.children[4].options = [""] + sorted(glob.glob(change + "/*.npz")) # support list
            self._list_widgets_pynx.children[5].options = [""] + sorted(glob.glob(change + "/*.npz")) # obj list

    def pynx_psf_handler(self, change):
        "Handles changes related to the psf"
        try:
            if change.new:
                for w in self._list_widgets_pynx.children[16:20]:
                    w.disabled = False

            if not change.new:
                for w in self._list_widgets_pynx.children[16:20]:
                    w.disabled = True

        except:
            if change:
                for w in self._list_widgets_pynx.children[16:20]:
                    w.disabled = False

            if not change:
                for w in self._list_widgets_pynx.children[16:20]:
                    w.disabled = True

        self.pynx_peak_shape_handler(change = self._list_widgets_pynx.children[16].value)
        self.pynx_operator_handler(change = self._list_widgets_pynx.children[21].value)

    def pynx_peak_shape_handler(self, change):
        "Handles changes related to psf the peak shape"
        try:
            if change.new != "pseudo-voigt":
                self._list_widgets_pynx.children[18].disabled = True

            if change.new == "pseudo-voigt":
                self._list_widgets_pynx.children[18].disabled = False

        except:
            if change != "pseudo-voigt":
                self._list_widgets_pynx.children[18].disabled = True

            if change == "pseudo-voigt":
                self._list_widgets_pynx.children[18].disabled = False

    def pynx_operator_handler(self, change):
        "Handles changes related to operator algorithm instructions"
        try:
            if change.new:
                self._list_widgets_pynx.children[22].disabled = False

            if not change.new:
                self._list_widgets_pynx.children[22].disabled = True

        except:
            if change:
                self._list_widgets_pynx.children[22].disabled = False

            if not change:
                self._list_widgets_pynx.children[22].disabled = True

    def run_pynx_handler(self, change):
        if change.new:
            for w in self._list_widgets_pynx.children[:-2]:
                w.disabled = True
            self._list_widgets_pynx.children[-4].disabled = False

        elif not change.new:
            for w in self._list_widgets_pynx.children[:-2]:
                w.disabled = False

            self.pynx_psf_handler(change = self._list_widgets_pynx.children[15].value)

    def folder_strain_handler(self, change):
        """Handles changes on the widget used to load a data file"""
        try:
            self._list_widgets_strain.children[-3].options = sorted(glob.glob(change.new + "/*.h5") + glob.glob(change.new + "/*.cxi") + glob.glob(change.new + "/*.npy") + glob.glob(change.new + "/*.npz")) + [""]
        except:
            self._list_widgets_strain.children[-3].options = sorted(glob.glob(change + "/*.h5") + glob.glob(change + "/*.cxi") + glob.glob(change + "/*.npy") + glob.glob(change + "/*.npz")) + [""]

    def folder_plot_handler(self, change):
        """Handles changes on the widget used to load a data file"""
        try:
            self.tab_data.children[2].options = sorted(glob.glob(change.new + "/*.npz") + glob.glob(change.new + "/*.cxi") + glob.glob(change.new + "/*.h5")) + [""]

        except AttributeError:
            self.tab_data.children[2].options = sorted(glob.glob(change + "/*.npz") + glob.glob(change + "/*.cxi") + glob.glob(change + "/*.h5")) + [""]

    def folder_facet_handler(self, change):
        """Handles changes on the widget used to load a data file"""
        try:
            self.tab_facet.children[2].options = sorted(glob.glob(change.new + "/*.vtk")) + [""]

        except AttributeError:
            self.tab_facet.children[2].options = sorted(glob.glob(change + "/*.vtk")) + [""]
