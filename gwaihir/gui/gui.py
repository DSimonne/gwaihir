import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from shlex import quote
import shutil
from ast import literal_eval
import operator as operator_lib
import getpass
from datetime import datetime
import inspect
import time
import tables as tb
from h5glance import H5Glance

# Widgets
import ipywidgets as widgets
from ipywidgets import interact, Button, Layout, interactive
from IPython.display import display, Markdown, clear_output, Image

# gwaihir package
import gwaihir
from gwaihir import plot, support
from gwaihir.gui import gui_iterable
import gwaihir.utilities as gutil

# bcdi package
from bcdi.utils.utilities import bin_data
from bcdi.postprocessing import facet_analysis
from bcdi.preprocessing import ReadNxs3 as rd
from bcdi.preprocessing.preprocessing_runner import run as run_preprocessing
from bcdi.postprocessing.postprocessing_runner import run as run_postprocessing
from bcdi.utils.parser import ConfigParser
import argparse

# PyNX package
import h5py
from numpy.fft import fftshift
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage import gaussian_filter
try:
    # This imports all necessary operators. GPU will be auto-selected
    print("Importing pynx ...")
    from pynx.cdi import CDI, SupportUpdate, ScaleObj, AutoCorrelationSupport,\
        InitPSF, ShowCDI, HIO, RAAR, ER, SupportTooLarge
    from pynx.cdi.runner.id01 import params
    from pynx.utils.math import smaller_primes
    pynx_import = True
except ModuleNotFoundError:
    pynx_import = False
    print(
        "Could not load PyNX, the phase retrieval tab will be disabled.\n"
        "Make sure you have the right version of PyNX installed.")


class Interface:
    """
    This class is a Graphical User Interface (GUI).

    It makes extensive use of the ipywidgets and is thus meant to be
    used with a jupyter notebook. Additional informations are provided
    in the "ReadMe" tab of the GUI.
    """

    def __init__(self, plot_only=False):
        """
        All the widgets for the GUI are defined here. They are regrouped in
        a few tabs that design the GUI, the tabs are: tab_init tab_detector
        tab_setup tab_preprocess tab_data_frame tab_pynx tab_strain
        tab_data tab_facet tab_readme.

        Also defines:
            path_scripts: path to folder in which bcdi script are stored
            user_name: user_name used to login to slurm if working
                on the ESRF cluster

        :param plot_only: True to only work with the plotting tab
        """
        super(Interface, self).__init__()

        # Plotting tool
        self.plot_only = plot_only

        if not plot_only:
            # Get path to scripts folder
            self.path_package = inspect.getfile(gwaihir).split("__")[0]
            self.path_scripts = self.path_package.split(
                "/lib/python")[0] + "/bin"
            print(
                f"Using `{self.path_scripts}`\n"
                "as absolute path to scripts containing folder.\n"
                "This should be correct if gwaihir was installed in an environment.\n"
                "Otherwise change self.path_scripts attribute to the correct folder.\n"
            )

            # Get user name
            try:
                self.user_name = getpass.getuser()

                print(
                    f"Login used for batch jobs: {self.user_name}\n"
                    "If wrong login, please change self.user_name attribute")
            except Exception as e:
                print(
                    "Could not get user name, please create self.user_name \
                    attribute for jobs")
                raise e

        # Initialize future attributes
        self.Dataset = None
        self.cxi_filename = None
        self.run_phase_retrieval = False
        self.run_pynx_tools = False
        self.text_file = None
        self.params = None
        self.Facets = None
        self.preprocessing_folder = None
        self.postprocessing_folder = None

        # Widgets for initialization
        self._list_widgets_init_dir = interactive(
            self.initialize_directories,
            # Define scan related
            # parameters
            unused_label_scan=widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>\
                Define working directory and scan number",
                style={
                    'description_width': 'initial'},
                layout=Layout(width='90%', height="35px")),

            sample_name=widgets.Text(
                value="S",
                placeholder="",
                description='Sample Name',
                continuous_update=False,
                layout=Layout(
                    width='45%'),
                style={'description_width': 'initial'}),

            scan=widgets.BoundedIntText(
                value="01415",
                description='Scan nb:',
                min=0,
                max=9999999,
                continuous_update=False,
                layout=Layout(
                    width='45%'),
                style={'description_width': 'initial'}),

            data_dir=widgets.Text(
                value=os.getcwd() + "/data_dir/",
                placeholder="Path to data directory",
                description='Data directory',
                continuous_update=False,
                layout=Layout(
                    width='90%'),
                style={'description_width': 'initial'}),

            root_folder=widgets.Text(
                value=os.getcwd() + "/TestGui/",
                placeholder="Root folder (parent to all scan directories)",
                description='Target directory (root_folder)',
                continuous_update=False,
                layout=Layout(
                    width='90%'),
                style={'description_width': 'initial'}),

            comment=widgets.Text(
                value="",
                description='Comment',
                continuous_update=False,
                layout=Layout(
                    width='90%'),
                placeholder="Comment regarding Dataset...",
                style={'description_width': 'initial'}),

            debug=widgets.Checkbox(
                value=False,
                description='Debug scripts',
                tooltip='True to interact with plots, False to close it \
                automatically',
                indent=False,
                continuous_update=False,
                style={'description_width': 'initial'}),

            matplotlib_backend=widgets.Dropdown(
                options=[('Agg - No plots (faster)', 'Agg'), ('Qt5Agg - Interactive plots', 'Qt5Agg'),
                         ("ipympl- Plots in notebook output", "module://matplotlib_inline.backend_inline")],
                value="module://matplotlib_inline.backend_inline",
                description='Matplotlib backend for scripts:',
                continuous_update=False,
                layout=Layout(
                    width='60%'),
                # tooltip="Name of the beamline, used for data loading and \
                # normalization by monitor",
                style={'description_width': 'initial'}),

            run_dir_init=widgets.ToggleButton(
                value=False,
                description='Initialize directories ...',
                button_style='',
                icon='step-forward',
                layout=Layout(
                    width='45%'),
                style={'description_width': 'initial'}),
        )
        self._list_widgets_init_dir.children[8].observe(
            self.init_handler, names="value")
        self._list_widgets_init_dir.children[4].observe(
            self.sub_directories_handler, names="value")
        # Organize into vertical and horizontal widgets boxes
        self.tab_init = widgets.VBox([
            self._list_widgets_init_dir.children[0],
            widgets.HBox(self._list_widgets_init_dir.children[1:3]),
            self._list_widgets_init_dir.children[3],
            self._list_widgets_init_dir.children[4],
            self._list_widgets_init_dir.children[5],
            self._list_widgets_init_dir.children[6],
            self._list_widgets_init_dir.children[7],
            self._list_widgets_init_dir.children[8],
            self._list_widgets_init_dir.children[-1],
        ])

        # Widgets for preprocessing, all in a single list because of
        # interactive fct
        self._list_widgets_preprocessing = interactive(
            self.initialize_preprocessing,
            # Define beamline related parameters
            unused_label_beamline=widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>\
                Parameters specific to the beamline",
                style={
                    'description_width': 'initial'},
                layout=Layout(width='90%', height="35px")),

            beamline=widgets.Dropdown(
                options=['ID01', 'SIXS_2018', 'SIXS_2019',
                         'CRISTAL', 'P10', 'NANOMAX', '34ID',
                         "ID01BLISS"],
                value="SIXS_2019",
                description='Beamline',
                continuous_update=False,
                disabled=True,
                tooltip="Name of the beamline, used for data loading and \
                normalization by monitor",
                style={'description_width': 'initial'}),

            actuators=widgets.Text(
                value="{}",
                placeholder="{}",
                continuous_update=False,
                description='Actuators',
                tooltip="Optional dictionary that can be used to define the \
                entries corresponding to actuators in data files (useful at \
                CRISTAL where the location of data keeps changing)",
                readout=True,
                style={
                    'description_width': 'initial'},
                disabled=True),

            is_series=widgets.Checkbox(
                value=False,
                description='Is series (P10)',
                disabled=True,
                # button_style = '',
                # # 'success',
                # 'info', 'warning',
                # 'danger' or ''
                continuous_update=False,
                tooltip='specific to series measurement at P10',
                icon='check'),

            custom_scan=widgets.Checkbox(
                value=False,
                description='Custom scan',
                continuous_update=False,
                disabled=True,
                indent=False,
                tooltip='set it to True for a stack of images acquired without\
                 scan, e.g. with ct in a macro, or when there is no spec/log \
                 file available',
                icon='check'),

            custom_images=widgets.Text(
                value="[]",
                description='Custom images',
                continuous_update=False,
                disabled=True,
                style={'description_width': 'initial'}),

            custom_monitor=widgets.IntText(
                value=0,
                description='Custom monitor',
                continuous_update=False,
                disabled=True,
                style={'description_width': 'initial'}),

            specfile_name=widgets.Text(
                placeholder="alias_dict_2019.txt",
                value="",
                description='Specfile name',
                disabled=True,
                continuous_update=False,
                layout=Layout(
                    width='90%'),
                style={'description_width': 'initial'}),

            rocking_angle=widgets.Dropdown(
                options=[
                    'inplane', 'outofplane'],
                value="inplane",
                continuous_update=False,
                description='Rocking angle',
                disabled=True,
                tooltip="Name of the beamline, used for data loading and \
                normalization by monitor",
                layout=Layout(
                    height="50px"),
                style={'description_width': 'initial'}),


            # Parameters used in masking
            unused_label_masking=widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>\
                Parameters used in masking",
                style={
                    'description_width': 'initial'},
                layout=Layout(width='90%', height="35px")),

            flag_interact=widgets.Checkbox(
                value=False,
                description='Manual masking',
                continuous_update=False,
                disabled=True,
                indent=False,
                tooltip='True to interact with plots and manually mask points',
                layout=Layout(
                    height="50px"),
                icon='check'),

            background_plot=widgets.FloatText(
                value=0.5,
                step=0.01,
                max=1,
                min=0,
                continuous_update=False,
                description='Background plot:',
                layout=Layout(
                    width='30%', height="50px"),
                tooltip="In level of grey in [0,1], 0 being dark. For visual \
                comfort during masking",
                readout=True,
                style={
                    'description_width': 'initial'},
                disabled=True),


            # Parameters related to data cropping/padding/centering
            unused_label_centering=widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>\
                Parameters related to data cropping/padding/centering</p>",
                style={
                    'description_width': 'initial'},
                layout=Layout(width='90%', height="35px")),

            centering_method=widgets.Dropdown(
                options=[
                    "max", "com", "manual"],
                value="max",
                description='Centering of Bragg peak method:',
                continuous_update=False,
                disabled=True,
                layout=Layout(
                    width='45%'),
                tooltip="Bragg peak determination: 'max' or 'com', 'max' is \
                better usually. It will be overridden by 'bragg_peak' if \
                not empty",
                style={'description_width': 'initial'}),

            bragg_peak=widgets.Text(
                placeholder="[z_bragg, y_bragg, x_bragg]",
                description='Bragg peak position',
                disabled=True,
                continuous_update=False,
                layout=Layout(
                    width='45%'),
                style={'description_width': 'initial'}),

            fix_size=widgets.Text(
                placeholder="[zstart, zstop, ystart, ystop, xstart, xstop]",
                description='Fix array size',
                disabled=True,
                continuous_update=False,
                layout=Layout(
                    width='45%'),
                style={'description_width': 'initial'}),

            center_fft=widgets.Dropdown(
                options=[
                    'crop_sym_ZYX', 'crop_asym_ZYX', 'pad_asym_Z_crop_sym_YX',
                    'pad_sym_Z_crop_asym_YX', 'pad_sym_Z', 'pad_asym_Z',
                    'pad_sym_ZYX', 'pad_asym_ZYX', 'skip'],
                value="crop_sym_ZYX",
                description='Center FFT',
                continuous_update=False,
                layout=Layout(
                    height="50px"),
                disabled=True,
                style={'description_width': 'initial'}),

            pad_size=widgets.Text(
                placeholder="[256, 512, 512]",
                description='Array size after padding',
                disabled=True,
                continuous_update=False,
                layout=Layout(
                    width='50%', height="50px"),
                style={'description_width': 'initial'}),

            # Parameters used in intensity normalization
            normalize_flux=widgets.Dropdown(
                options=[
                    "skip", "monitor"],
                value="skip",
                description='Normalize flux',
                disabled=True,
                continuous_update=False,
                layout=Layout(
                    height="50px"),
                tooltip='Monitor to normalize the intensity by the default \
                monitor values, skip to do nothing',
                style={'description_width': 'initial'}),


            # Parameters for data filtering
            unused_label_filtering=widgets.HTML(
                description="""<p style='font-weight: bold;font-size:1.2em'>\
                Parameters for data filtering</p>""",
                style={
                    'description_width': 'initial'},
                layout=Layout(width='90%', height="35px")),

            mask_zero_event=widgets.Checkbox(
                value=False,
                description='Mask zero event',
                disabled=True,
                continuous_update=False,
                indent=False,
                tooltip='Mask pixels where the sum along the rocking curve is \
                zero - may be dead pixels',
                icon='check'),

            median_filter=widgets.Dropdown(
                options=[
                    'skip', 'median', 'interp_isolated', 'mask_isolated'],
                value="skip",
                description='Flag median filter',
                continuous_update=False,
                disabled=True,
                tooltip="set to 'median' for applying med2filter [3,3], set to \
                'interp_isolated' to interpolate isolated empty pixels based on\
                 'median_filter_order' parameter, set to 'mask_isolated' it \
                 will mask isolated empty pixels, set to 'skip' will skip \
                 filtering",
                style={'description_width': 'initial'}),

            median_filter_order=widgets.IntText(
                value=7,
                description='Med filter order:',
                disabled=True,
                continuous_update=False,
                tooltip="for custom median filter, number of pixels with \
                intensity surrounding the empty pixel",
                style={'description_width': 'initial'}),

            phasing_binning=widgets.Text(
                value="(1, 1, 1)",
                placeholder="(1, 1, 1)",
                description='Binning for phasing',
                disabled=True,
                continuous_update=False,
                layout=Layout(
                    width='20%', height="50px"),
                style={
                    'description_width': 'initial'},
                tooltip="binning that will be used for phasing (stacking \
                dimension, detector vertical axis, detector horizontal axis)"),

            # Parameters used when reloading processed data
            unused_label_reload=widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>\
                Parameters used when reloading processed data</p>",
                style={
                    'description_width': 'initial'},
                layout=Layout(width='90%', height="35px")),

            reload_previous=widgets.Checkbox(
                value=False,
                description='Reload previous',
                continuous_update=False,
                disabled=True,
                indent=False,
                layout=Layout(
                    height="50px"),
                tooltip='True to resume a previous masking (load data\
                and mask)',
                icon='check'),

            reload_orthogonal=widgets.Checkbox(
                value=False,
                description='Reload orthogonal',
                continuous_update=False,
                disabled=True,
                indent=False,
                layout=Layout(
                    height="50px"),
                tooltip='True if the reloaded data is already intepolated \
                in an orthonormal frame',
                icon='check'),

            preprocessing_binning=widgets.Text(
                value="(1, 1, 1)",
                placeholder="(1, 1, 1)",
                description='Binning used in data to be reloaded',
                disabled=True,
                continuous_update=False,
                layout=Layout(
                    width='30%', height="50px"),
                style={
                    'description_width': 'initial'},
                tooltip="binning that will be used for phasing (stacking \
                dimension, detector vertical axis, detector horizontal axis)"),

            # Saving options
            unused_label_saving=widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>\
                Parameters used when saving the data</p>",
                style={
                    'description_width': 'initial'},
                layout=Layout(width='90%', height="35px")),

            save_rawdata=widgets.Checkbox(
                value=False,
                description='Save raw data',
                disabled=True,
                continuous_update=False,
                indent=False,
                layout=Layout(
                    width="15%", height="50px"),
                tooltip='Save also the raw data when use_rawdata is False',
                icon='check'),

            save_to_npz=widgets.Checkbox(
                value=True,
                description='Save to npz',
                disabled=True,
                continuous_update=False,
                indent=False,
                layout=Layout(
                    width="15%", height="50px"),
                tooltip='True to save the processed data in npz format',
                icon='check'),

            save_to_mat=widgets.Checkbox(
                value=False,
                description='Save to mat',
                disabled=True,
                continuous_update=False,
                indent=False,
                layout=Layout(
                    width="15%", height="50px"),
                tooltip='True to save also in .mat format',
                icon='check'),

            save_to_vti=widgets.Checkbox(
                value=False,
                description='Save to vti',
                continuous_update=False,
                disabled=True,
                indent=False,
                layout=Layout(
                    width="15%", height="50px"),
                tooltip='Save the orthogonalized diffraction pattern to \
                VTK file',
                icon='check'),

            save_as_int=widgets.Checkbox(
                value=False,
                description='Save as int',
                continuous_update=False,
                disabled=True,
                indent=False,
                layout=Layout(
                    width="15%", height="50px"),
                tooltip='if True, the result will be saved as an array of \
                integers (save space)',
                icon='check'),

            # Detector related parameters
            unused_label_detector=widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>\
                Parameters related to the detector used</p>",
                style={
                    'description_width': 'initial'},
                layout=Layout(width='90%', height="35px")),

            detector=widgets.Dropdown(
                options=[
                    "Eiger2M", "Maxipix", "Eiger4M", "Merlin", "Timepix"],
                value="Merlin",
                description='Detector',
                continuous_update=False,
                disabled=True,
                style={'description_width': 'initial'}),

            roi_detector=widgets.Text(
                placeholder="""[low_y_bound, high_y_bound, low_x_bound, high_x_bound]""",
                description='Fix roi area, will overwrite cropping parameters',
                disabled=True,
                continuous_update=False,
                layout=Layout(
                    width='90%'),
                style={'description_width': 'initial'}),

            photon_threshold=widgets.IntText(
                value=0,
                description='Photon Threshold:',
                disabled=True,
                continuous_update=False,
                tooltip="data[data < photon_threshold] = 0",
                style={'description_width': 'initial'}),

            photon_filter=widgets.Dropdown(
                options=[
                    'loading', 'postprocessing'],
                value="loading",
                continuous_update=False,
                description='Photon filter',
                disabled=True,
                tooltip="When the photon threshold should be applied, if \
                'loading', it is applied before binning; if 'postprocessing',\
                 it is applied at the end of the script before saving",
                style={'description_width': 'initial'}),

            background_file=widgets.Text(
                value="",
                placeholder=f"{os.getcwd()}/background.npz'",
                description='Background file',
                disabled=True,
                continuous_update=False,
                layout=Layout(
                    width='90%'),
                style={'description_width': 'initial'}),

            flatfield_file=widgets.Text(
                value="",
                placeholder=f"{os.getcwd()}/flatfield_maxipix_8kev.npz",
                description='Flatfield file',
                disabled=True,
                continuous_update=False,
                layout=Layout(
                    width='90%'),
                style={'description_width': 'initial'}),

            hotpixels_file=widgets.Text(
                value="",
                placeholder=f"{os.getcwd()}/mask_merlin.npz",
                description='Hotpixels file',
                disabled=True,
                continuous_update=False,
                layout=Layout(
                    width='90%'),
                style={'description_width': 'initial'}),

            template_imagefile=widgets.Text(
                value='Pt_ascan_mu_%05d.nxs',
                description='Template imagefile',
                disabled=True,
                layout=Layout(
                    width='90%'),
                style={'description_width': 'initial'}),

            # Define parameters below if you want to orthogonalize the data
            # before phasing
            unused_label_ortho=widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>\
                Parameters to define the data orthogonalization</p>",
                style={
                    'description_width': 'initial'},
                layout=Layout(width='90%', height="35px")),

            use_rawdata=widgets.Checkbox(
                value=False,
                continuous_update=False,
                description='Orthogonalize data',
                disabled=True,
                indent=False,
                # button_style = '',
                # # 'success',
                # 'info', 'warning',
                # 'danger' or ''
                tooltip='False for using data gridded in laboratory frame/ \
                True for using data in detector frame',
                icon='check'),

            interpolation_method=widgets.Dropdown(
                options=[
                    'linearization', 'xrayutilities'],
                value="linearization",
                continuous_update=False,
                description='Interpolation method',
                disabled=True,
                # tooltip = "",
                style={'description_width': 'initial'}),

            fill_value_mask=widgets.Dropdown(
                options=[0, 1],
                value=0,
                description='Fill value mask',
                continuous_update=False,
                disabled=True,
                tooltip="It will define how the pixels outside of the data \
                range are processed during the interpolation. Because of the \
                large number of masked pixels, phase retrieval converges better\
                 if the pixels are not masked (0 intensity imposed). The data \
                 is by default set to 0 outside of the defined range.",
                style={'description_width': 'initial'}),

            beam_direction=widgets.Text(
                value="(1, 0, 0)",
                placeholder="(1, 0, 0)",
                description='Beam direction in lab. frame',
                disabled=True,
                continuous_update=False,
                layout=Layout(
                    width='50%'),
                style={
                    'description_width': 'initial'},
                tooltip="Beam direction in the laboratory frame (downstream, \
                vertical up, outboard), beam along z"),

            sample_offsets=widgets.Text(
                value="(0, 0)",
                placeholder="(0, 0, 90, 0)",
                description='Sample offsets',
                disabled=True,
                continuous_update=False,
                layout=Layout(
                    width='25%'),
                style={
                    'description_width': 'initial'},
                tooltip="""Tuple of offsets in degrees of the sample for each \
                sample circle (outer first). Convention: the sample offsets \
                will be subtracted to the motor values"""),

            sdd=widgets.FloatText(
                value=1.18,
                step=0.01,
                description='Sample detector distance (m):',
                continuous_update=False,
                disabled=True,
                tooltip="sample to detector distance in m",
                style={'description_width': 'initial'}),

            energy=widgets.IntText(
                value=8500,
                description='X-ray energy in eV',
                continuous_update=False,
                disabled=True,
                layout=Layout(
                    height="50px"),
                style={'description_width': 'initial'}),

            custom_motors=widgets.Text(
                value="{}",
                placeholder="{}",
                description='Custom motors',
                disabled=True,
                continuous_update=False,
                layout=Layout(
                    width='90%', height="50px"),
                style={
                    'description_width': 'initial'},
                tooltip="Use this to declare motor positions"),

            # Parameters for xrayutilities to orthogonalize the data
            # before phasing
            unused_label_xru=widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>\
                Parameters used in xrayutilities to orthogonalize the data \
                before phasing (initialize the directories before)</p>",
                style={
                    'description_width': 'initial'},
                layout=Layout(width='90%', height="35px")),

            align_q=widgets.Checkbox(
                value=True,
                description='Align q',
                continuous_update=False,
                disabled=True,
                indent=False,
                layout=Layout(
                    width='20%'),
                tooltip="""Used only when interpolation_method is \
                'linearization', if True it rotates the crystal to align q \
                along one axis of the array""",
                icon='check'),

            ref_axis_q=widgets.Dropdown(
                options=[
                    "x", "y", "z"],
                value="y",
                description='Ref axis q',
                disabled=True,
                continuous_update=False,
                layout=Layout(
                    width='20%'),
                tooltip="q will be aligned along that axis",
                style={'description_width': 'initial'}),

            direct_beam=widgets.Text(
                value="[250, 250]",
                placeholder="[250, 250]",
                description='Direct beam position (V, H)',
                disabled=True,
                continuous_update=False,
                layout=Layout(
                    width='45%'),
                style={'description_width': 'initial'}),

            dirbeam_detector_angles=widgets.Text(
                value="[0, 0]",
                placeholder="[0, 0]",
                description='Direct beam angles (°)',
                disabled=True,
                continuous_update=False,
                layout=Layout(
                    width='45%'),
                style={'description_width': 'initial'}),

            outofplane_angle=widgets.FloatText(
                value=0,
                step=0.01,
                description='Outofplane angle',
                continuous_update=False,
                disabled=True,
                layout=Layout(
                    width='25%'),
                style={'description_width': 'initial'}),

            inplane_angle=widgets.FloatText(
                value=0,
                step=0.01,
                description='Inplane angle',
                continuous_update=False,
                disabled=True,
                layout=Layout(
                    width='25%'),
                style={'description_width': 'initial'}),

            tilt_angle=widgets.FloatText(
                value=0,
                step=0.0001,
                description='Tilt angle',
                continuous_update=False,
                disabled=True,
                layout=Layout(
                    width='25%'),
                style={'description_width': 'initial'}),
            sample_inplane=widgets.Text(
                value="(1, 0, 0)",
                placeholder="(1, 0, 0)",
                description='Sample inplane',
                disabled=True,
                continuous_update=False,
                layout=Layout(
                    width='20%'),
                style={
                    'description_width': 'initial'},
                tooltip="Sample inplane reference direction along the beam at \
                0 angles"),

            sample_outofplane=widgets.Text(
                value="(0, 0, 1)",
                placeholder="(0, 0, 1)",
                description='Sample outofplane',
                disabled=True,
                continuous_update=False,
                layout=Layout(
                    width='20%'),
                style={
                    'description_width': 'initial'},
                tooltip="Surface normal of the sample at 0 angles"),

            offset_inplane=widgets.FloatText(
                value=0,
                step=0.01,
                description='Offset inplane',
                disabled=True,
                continuous_update=False,
                layout=Layout(
                    width='20%'),
                style={
                    'description_width': 'initial'},
                tooltip="Outer detector angle offset, not important if you \
                use raw data"),

            cch1=widgets.IntText(
                value=271,
                description='cch1',
                continuous_update=False,
                disabled=True,
                layout=Layout(
                    width='15%'),
                tooltip="cch1 parameter from xrayutilities 2D detector \
                calibration, vertical",
                style={'description_width': 'initial'}),

            cch2=widgets.IntText(
                value=213,
                description='cch2',
                continuous_update=False,
                disabled=True,
                layout=Layout(
                    width='15%'),
                tooltip="cch2 parameter from xrayutilities 2D detector \
                calibration, horizontal",
                style={'description_width': 'initial'}),

            detrot=widgets.FloatText(
                value=0,
                step=0.01,
                description='Detector rotation',
                disabled=True,
                continuous_update=False,
                layout=Layout(
                    width='20%'),
                style={
                    'description_width': 'initial'},
                tooltip="Detrot parameter from xrayutilities 2D detector \
                calibration"),

            tiltazimuth=widgets.FloatText(
                value=360,
                step=0.01,
                description='Tilt azimuth',
                disabled=True,
                continuous_update=False,
                layout=Layout(
                    width='15%'),
                style={
                    'description_width': 'initial'},
                tooltip="tiltazimuth parameter from xrayutilities 2D detector\
                 calibration"),

            tilt_detector=widgets.FloatText(
                value=0,
                step=0.01,
                description='Tilt detector',
                disabled=True,
                continuous_update=False,
                layout=Layout(
                    width='15%'),
                style={
                    'description_width': 'initial'},
                tooltip="tilt parameter from xrayutilities 2D detector \
                calibration"),

            # Run preprocess
            unused_label_preprocess=widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>\
                Click below to run the data processing before phasing</p>",
                style={
                    'description_width': 'initial'},
                layout=Layout(width='90%', height="35px")),

            init_para=widgets.ToggleButton(
                value=False,
                description='Initialize parameters ...',
                disabled=True,
                continuous_update=False,
                button_style='',
                layout=Layout(
                    width='40%'),
                style={
                    'description_width': 'initial'},
                icon='fast-forward')
        )
        self._list_widgets_preprocessing.children[1].observe(
            self.beamline_handler, names="value")
        self._list_widgets_preprocessing.children[13].observe(
            self.bragg_peak_centering_handler, names="value")
        self._list_widgets_preprocessing.children[25].observe(
            self.reload_data_handler, names="value")
        # self._list_widgets_preprocessing.children[44].observe(
        #     self.orthogonalisation_handler, names="value")
        self._list_widgets_preprocessing.children[-2].observe(
            self.preprocess_handler, names="value")

        # Parameters specific to the beamline
        self.tab_beamline = widgets.VBox([
            self._list_widgets_preprocessing.children[0],
            self._list_widgets_preprocessing.children[1],
            widgets.HBox(self._list_widgets_preprocessing.children[2:4]),
            widgets.HBox(self._list_widgets_preprocessing.children[4:7]),
            self._list_widgets_preprocessing.children[7],
            self._list_widgets_preprocessing.children[8],
            self._list_widgets_preprocessing.children[9],
            widgets.HBox(self._list_widgets_preprocessing.children[10:12]),
        ])

        # Parameters related to data cropping/padding/centering
        self.tab_reduction = widgets.VBox([
            self._list_widgets_preprocessing.children[12],
            widgets.HBox(self._list_widgets_preprocessing.children[13:15]),
            self._list_widgets_preprocessing.children[15],
            widgets.HBox(self._list_widgets_preprocessing.children[16:19]),
            self._list_widgets_preprocessing.children[19],
            widgets.HBox(self._list_widgets_preprocessing.children[20:23]),
            self._list_widgets_preprocessing.children[23],
        ])

        # Parameters used when reloading processed data
        self.tab_save_load = widgets.VBox([
            self._list_widgets_preprocessing.children[24],
            widgets.HBox(self._list_widgets_preprocessing.children[25:28]),
            self._list_widgets_preprocessing.children[28],
            widgets.HBox(self._list_widgets_preprocessing.children[29:34]),
        ])

        # Parameters related to the detector used
        self.tab_detector = widgets.VBox([
            self._list_widgets_preprocessing.children[34],
            self._list_widgets_preprocessing.children[35],
            self._list_widgets_preprocessing.children[36],
            widgets.HBox(self._list_widgets_preprocessing.children[37:39]),
            self._list_widgets_preprocessing.children[39],
            self._list_widgets_preprocessing.children[40],
            self._list_widgets_preprocessing.children[41],
            self._list_widgets_preprocessing.children[42],
        ])

        # Parameters to define the data orthogonalization
        self.tab_setup = widgets.VBox([
            self._list_widgets_preprocessing.children[43],
            self._list_widgets_preprocessing.children[44],
            widgets.HBox(self._list_widgets_preprocessing.children[45:47]),
            self._list_widgets_preprocessing.children[47],
            self._list_widgets_preprocessing.children[48],
            self._list_widgets_preprocessing.children[49],
            widgets.HBox(self._list_widgets_preprocessing.children[50:52]),
            self._list_widgets_preprocessing.children[52],
            widgets.HBox(self._list_widgets_preprocessing.children[53:55]),
            widgets.HBox(self._list_widgets_preprocessing.children[55:57]),
            widgets.HBox(self._list_widgets_preprocessing.children[57:60]),
            widgets.HBox(self._list_widgets_preprocessing.children[60:63]),
            widgets.HBox(self._list_widgets_preprocessing.children[63:68]),
        ])

        # Group all preprocess tabs into a single one, besides detector and
        # setup parameter
        self.tab_preprocess = widgets.VBox([
            self.tab_beamline,
            self.tab_reduction,
            self.tab_save_load,
            self._list_widgets_preprocessing.children[-3],
            self._list_widgets_preprocessing.children[-2],
            self._list_widgets_preprocessing.children[-1]
        ])

        # Widgets for PyNX
        self._list_widgets_phase_retrieval = interactive(
            self.initialize_phase_retrieval,
            unused_label_data=widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>\
                Data files",
                style={
                    'description_width': 'initial'},
                layout=Layout(width='90%', height="35px")),

            parent_folder=widgets.Dropdown(
                options=[x[0] + "/" for x in os.walk(os.getcwd())],
                value=os.getcwd() + "/",
                placeholder=os.getcwd() + "/",
                description='Parent folder:',
                continuous_update=False,
                layout=Layout(width='90%'),
                style={'description_width': 'initial'}),

            iobs=widgets.Dropdown(
                options=[""]
                + sorted([os.path.basename(f) for f in
                          glob.glob(os.getcwd() + "*.npz")],
                         key=os.path.getmtime),
                description='Dataset',
                layout=Layout(width='90%'),
                style={'description_width': 'initial'}),

            mask=widgets.Dropdown(
                options=[""]
                + sorted([os.path.basename(f) for f in
                          glob.glob(os.getcwd() + "*.npz")],
                         key=os.path.getmtime),
                description='Mask',
                layout=Layout(width='90%'),
                style={'description_width': 'initial'}),

            support=widgets.Dropdown(
                options=[""]
                + sorted([os.path.basename(f) for f in
                          glob.glob(os.getcwd() + "*.npz")],
                         key=os.path.getmtime),
                value="",
                description='Support',
                layout=Layout(width='90%'),
                style={'description_width': 'initial'}),

            obj=widgets.Dropdown(
                options=[""]
                + sorted([os.path.basename(f) for f in
                          glob.glob(os.getcwd() + "*.npz")],
                         key=os.path.getmtime),
                value="",
                description='Object',
                layout=Layout(width='90%'),
                style={'description_width': 'initial'}),

            auto_center_resize=widgets.Checkbox(
                value=False,
                description='Auto center and resize',
                continuous_update=False,
                indent=False,
                layout=Layout(height="50px"),
                icon='check'),

            max_size=widgets.BoundedIntText(
                value=256,
                step=1,
                min=0,
                max=1000,
                layout=Layout(
                    height="50px", width="40%"),
                continuous_update=False,
                description='Maximum array size for cropping:',
                readout=True,
                style={
                    'description_width': 'initial'},
                disabled=False),

            unused_label_support=widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>\
                Support parameters",
                style={
                    'description_width': 'initial'},
                layout=Layout(width='90%', height="35px")),

            support_threshold=widgets.Text(
                value="(0.23, 0.30)",
                placeholder="(0.23, 0.30)",
                description='Support threshold',
                layout=Layout(
                    height="50px", width="40%"),
                continuous_update=False,
                style={'description_width': 'initial'}),

            support_only_shrink=widgets.Checkbox(
                value=False,
                description='Support only shrink',
                continuous_update=False,
                indent=False,
                layout=Layout(
                    height="50px", width="15%"),
                icon='check'),

            support_update_period=widgets.BoundedIntText(
                value=20,
                step=5,
                layout=Layout(
                    height="50px", width="25%"),
                continuous_update=False,
                description='Support update period:',
                readout=True,
                style={
                    'description_width': 'initial'},
                disabled=False),

            support_smooth_width=widgets.Text(
                value="(2, 1, 600)",
                placeholder="(2, 1, 600)",
                description='Support smooth width',
                layout=Layout(
                    height="50px", width="35%"),
                continuous_update=False,
                style={'description_width': 'initial'}),

            support_post_expand=widgets.Text(
                value="(1, -2, 1)",
                placeholder="(1, -2, 1)",
                description='Support post expand',
                layout=Layout(
                    height="50px", width="35%"),
                continuous_update=False,
                style={'description_width': 'initial'}),

            unused_label_psf=widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>\
                Point spread function parameters",
                style={
                    'description_width': 'initial'},
                layout=Layout(width='90%', height="35px")),

            psf=widgets.Checkbox(
                value=True,
                description='Use point spread function:',
                continuous_update=False,
                indent=False,
                layout=Layout(height="50px"),
                icon='check'),

            psf_model=widgets.Dropdown(
                options=[
                    "gaussian", "lorentzian", "pseudo-voigt"],
                value="pseudo-voigt",
                description='PSF peak shape',
                continuous_update=False,
                style={'description_width': 'initial'}),

            fwhm=widgets.FloatText(
                value=0.5,
                step=0.01,
                min=0,
                continuous_update=False,
                description="FWHM:",
                layout=Layout(
                    width='15%', height="50px"),
                style={
                    'description_width': 'initial'}),

            eta=widgets.FloatText(
                value=0.05,
                step=0.01,
                max=1,
                min=0,
                continuous_update=False,
                description='Eta:',
                layout=Layout(
                    width='15%', height="50px"),
                readout=True,
                style={
                    'description_width': 'initial'}),

            update_psf=widgets.BoundedIntText(
                value=20,
                step=5,
                continuous_update=False,
                description='Update PSF every:',
                readout=True,
                style={
                    'description_width': 'initial'}),

            unused_label_algo=widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>\
                Iterative algorithms parameters",
                style={
                    'description_width': 'initial'},
                layout=Layout(width='90%', height="35px")),

            nb_raar=widgets.BoundedIntText(
                value=1000,
                min=0,
                max=9999,
                step=10,
                continuous_update=False,
                description='Nb of RAAR:',
                layout=Layout(
                    height="35px", width="20%"),
                readout=True,
                style={
                    'description_width': 'initial'},
                disabled=False),

            nb_hio=widgets.BoundedIntText(
                value=400,
                min=0,
                max=9999,
                step=10,
                continuous_update=False,
                description='Nb of HIO:',
                layout=Layout(
                    height="35px", width="20%"),
                readout=True,
                style={
                    'description_width': 'initial'},
                disabled=False),

            nb_er=widgets.BoundedIntText(
                value=300,
                min=0,
                max=9999,
                step=10,
                continuous_update=False,
                description='Nb of ER:',
                layout=Layout(
                    height="35px", width="20%"),
                readout=True,
                style={
                    'description_width': 'initial'},
                disabled=False),

            nb_ml=widgets.BoundedIntText(
                value=0,
                min=0,
                max=9999,
                step=10,
                continuous_update=False,
                description='Nb of ML:',
                layout=Layout(
                    height="35px", width="20%"),
                readout=True,
                style={
                    'description_width': 'initial'},
                disabled=False),

            nb_run=widgets.BoundedIntText(
                value=30,
                continuous_update=False,
                description='Number of run:',
                layout=Layout(height="50px"),
                readout=True,
                style={
                    'description_width': 'initial'},
                disabled=False),

            unused_label_filtering=widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>\
                Filtering criteria for reconstructions",
                style={
                    'description_width': 'initial'},
                layout=Layout(width='90%', height="35px")),

            filter_criteria=widgets.Dropdown(
                options=[
                    ("No filtering",
                     "no_filtering"),
                    ("Standard deviation",
                        "standard_deviation"),
                    ("Log-likelihood (LLK)", "LLK"),
                    ("LLK > Standard deviation",
                        "LLK_standard_deviation"),
                    # ("Standard deviation > LLK", "standard_deviation_LLK"),
                ],
                value="LLK_standard_deviation",
                description='Filtering criteria',
                layout=Layout(width='90%'),
                style={'description_width': 'initial'}),

            nb_run_keep=widgets.BoundedIntText(
                value=10,
                continuous_update=False,
                description='Number of run to keep:',
                layout=Layout(height="50px"),
                readout=True,
                style={
                    'description_width': 'initial'},
                disabled=False),

            unused_label_options=widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>\
                Options",
                style={
                    'description_width': 'initial'},
                layout=Layout(width='90%', height="35px")),

            live_plot=widgets.BoundedIntText(
                value=200,
                step=10,
                max=500,
                min=0,
                continuous_update=False,
                description='Plot every:',
                readout=True,
                layout=Layout(
                    height="50px", width="20%"),
                style={
                    'description_width': 'initial'},
                disabled=False),

            positivity=widgets.Checkbox(
                value=False,
                description='Force positivity',
                continuous_update=False,
                indent=False,
                style={
                    'description_width': 'initial'},
                layout=Layout(
                    height="50px", width="20%"),
                icon='check'),

            beta=widgets.FloatText(
                value=0.9,
                step=0.01,
                max=1,
                min=0,
                continuous_update=False,
                description='Beta parameter for RAAR and HIO:',
                layout=Layout(
                    width='35%', height="50px"),
                readout=True,
                style={
                    'description_width': 'initial'},
                disabled=False),

            detwin=widgets.Checkbox(
                value=False,
                description='Detwinning',
                continuous_update=False,
                indent=False,
                style={
                    'description_width': 'initial'},
                layout=Layout(
                    height="50px", width="15%"),
                icon='check'),

            rebin=widgets.Text(
                value="(1, 1, 1)",
                placeholder="(1, 1, 1)",
                description='Rebin',
                layout=Layout(height="50px"),
                continuous_update=False,
                style={'description_width': 'initial'}),

            verbose=widgets.BoundedIntText(
                value=100,
                min=10,
                max=300,
                continuous_update=False,
                description='Verbose:',
                layout=Layout(height="50px"),
                readout=True,
                style={
                    'description_width': 'initial'},
                disabled=False),

            pixel_size_detector=widgets.BoundedIntText(
                value=55,
                continuous_update=False,
                description='Pixel size of detector (um):',
                layout=Layout(height="50px"),
                readout=True,
                style={
                    'description_width': 'initial'},
                disabled=False),

            unused_label_phase_retrieval=widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>\
                Click below to run the phase retrieval</p>",
                style={
                    'description_width': 'initial'},
                layout=Layout(width='90%', height="35px")),

            run_phase_retrieval=widgets.ToggleButtons(
                options=[
                    ('No phase retrieval', False),
                    ('Run batch job', "batch"),
                    ("Run script locally",
                        "local_script"),
                    ("Use operators",
                        "operators"),
                ],
                value=False,
                tooltips=[
                    "Click to be able to change parameters",
                    "Collect parameters to run a job on slurm, will \
                    automatically apply a std deviation filter and run modes \
                    decomposition, freed the kernel",
                    "Run script on jupyter notebook environment, uses notebook\
                     kernel, will be performed in background also but more \
                     slowly, good if you cannot use jobs.",
                    r"Use operators on local environment, if using PSF, it is \
                    activated after 50\% of RAAR cycles"
                ],
                description='Run phase retrieval ...',
                continuous_update=False,
                button_style='',
                layout=Layout(
                    width='100%', height="50px"),
                style={
                    'description_width': 'initial'},
                icon='fast-forward'),

            unused_label_run_pynx_tools=widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>\
                Click below to use a phase retrieval tool</p>",
                style={
                    'description_width': 'initial'},
                layout=Layout(width='90%', height="35px")),

            run_pynx_tools=widgets.ToggleButtons(
                options=[
                    ('No tool running', False),
                    ("Modes decomposition",
                        "modes"),
                    ("Filter reconstructions",
                        "filter")
                ],
                value=False,
                tooltips=[
                    "Click to be able to change parameters",
                    "Run modes decomposition in data folder, selects *LLK*.cxi\
                     files",
                    "Filter reconstructions"
                ],
                description="Choose analysis:",
                continuous_update=False,
                button_style='',
                layout=Layout(
                    width='100%', height="50px"),
                style={
                    'description_width': 'initial'},
                icon='fast-forward')
        )
        self._list_widgets_phase_retrieval.children[1].observe(
            self.pynx_folder_handler, names="value")
        self._list_widgets_phase_retrieval.children[15].observe(
            self.pynx_psf_handler, names="value")
        self._list_widgets_phase_retrieval.children[16].observe(
            self.pynx_peak_shape_handler, names="value")
        self._list_widgets_phase_retrieval.children[-4].observe(
            self.run_pynx_handler, names="value")
        self._list_widgets_phase_retrieval.children[-2].observe(
            self.run_pynx_handler, names="value")
        self.tab_pynx = widgets.VBox([
            widgets.VBox(self._list_widgets_phase_retrieval.children[:6]),
            widgets.HBox(self._list_widgets_phase_retrieval.children[6:8]),
            self._list_widgets_phase_retrieval.children[8],
            widgets.HBox(self._list_widgets_phase_retrieval.children[9:11]),
            widgets.HBox(self._list_widgets_phase_retrieval.children[11:14]),
            self._list_widgets_phase_retrieval.children[14],
            widgets.HBox(self._list_widgets_phase_retrieval.children[15:19]),
            self._list_widgets_phase_retrieval.children[19],
            self._list_widgets_phase_retrieval.children[20],
            widgets.HBox(self._list_widgets_phase_retrieval.children[21:25]),
            self._list_widgets_phase_retrieval.children[25],
            self._list_widgets_phase_retrieval.children[29],
            widgets.HBox(self._list_widgets_phase_retrieval.children[30:34]),
            widgets.HBox(self._list_widgets_phase_retrieval.children[34:37]),
            self._list_widgets_phase_retrieval.children[26],
            widgets.HBox(self._list_widgets_phase_retrieval.children[27:29]),
            self._list_widgets_phase_retrieval.children[-5],
            self._list_widgets_phase_retrieval.children[-4],
            self._list_widgets_phase_retrieval.children[-3],
            self._list_widgets_phase_retrieval.children[-2],
            self._list_widgets_phase_retrieval.children[-1],
        ])

        # Widgets for logs
        self.tab_data_frame = interactive(
            self.display_data_frame,
            unused_label_logs=widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>\
                Loads csv file and displays it in the GUI",
                style={'description_width': 'initial'},
                layout=Layout(width='90%', height="35px")),

            parent_folder=widgets.Dropdown(
                options=[x[0] + "/" for x in os.walk(os.getcwd())],
                value=os.getcwd() + "/",
                placeholder=os.getcwd() + "/",
                description='Parent folder:',
                continuous_update=False,
                layout=Layout(width='90%'),
                style={'description_width': 'initial'}),

            csv_file=widgets.Dropdown(
                options=sorted(glob.glob(os.getcwd()+"*.csv"),
                               key=os.path.getmtime),
                description='csv file in subdirectories:',
                continuous_update=False,
                layout=Layout(width='90%'),
                style={'description_width': 'initial'}),

            show_logs=widgets.ToggleButtons(
                options=[
                    ("Clear/ Reload folder", False),
                    ('Load .csv file', "load_csv"),
                    ("Load facets data ", "load_field_data"),
                ],
                value=False,
                description='Load dataframe',
                button_style='',
                layout=Layout(width='90%'),
                style={'description_width': 'initial'}),
        )
        self.tab_data_frame.children[1].observe(
            self.csv_file_handler, names="value")

        # Widgets for strain
        self._list_widgets_strain = interactive(
            self.initialize_postprocessing,
            unused_label_averaging=widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>\
                Parameters used when averaging several reconstruction",
                style={
                    'description_width': 'initial'},
                layout=Layout(width='90%', height="35px")),

            sort_method=widgets.Dropdown(
                options=['mean_amplitude', 'variance',
                         'variance/mean', 'volume'],
                value="variance/mean",
                description='Sorting method',
                style={'description_width': 'initial'}),

            correlation_threshold=widgets.FloatText(
                value=0.9,
                step=0.01,
                max=1,
                min=0,
                continuous_update=False,
                description='Correlation threshold:',
                style={
                    'description_width': 'initial'},
                disabled=False),

            unused_label_FFT=widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>\
                Parameters relative to the FFT window and voxel sizes",
                style={
                    'description_width': 'initial'},
                layout=Layout(width='90%', height="35px")),

            original_size=widgets.Text(
                placeholder="[256, 512, 512]",
                description='FFT shape before PyNX binning in PyNX',
                layout=Layout(width='45%'),
                continuous_update=False,
                style={'description_width': 'initial'}),

            phasing_binning=widgets.Text(
                value="(1, 1, 1)",
                placeholder="(1, 1, 1)",
                description='Binning factor used in phase retrieval',
                continuous_update=False,
                layout=Layout(width='45%'),
                style={
                    'description_width': 'initial'},
            ),

            preprocessing_binning=widgets.Text(
                value="(1, 1, 1)",
                placeholder="(1, 1, 1)",
                description='Binning factors used in preprocessing',
                continuous_update=False,
                layout=Layout(width='45%'),
                style={
                    'description_width': 'initial'},
            ),

            output_size=widgets.Text(
                placeholder="[256, 512, 512]",
                description='Output size',
                continuous_update=False,
                style={'description_width': 'initial'}),

            keep_size=widgets.Checkbox(
                value=False,
                description='Keep the initial array size for orthogonalization\
                 (slower)',
                layout=Layout(width='45%'),
                # icon = 'check',
                style={'description_width': 'initial'}),

            fix_voxel=widgets.BoundedIntText(
                placeholder="10",
                description='Fix voxel size, put 0 to set free:',
                min=0,
                max=9999999,
                continuous_update=False,
                style={'description_width': 'initial'}),

            # link to setup

            unused_label_disp_strain=widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>\
                Parameters related to displacement and strain calculation",
                style={
                    'description_width': 'initial'},
                layout=Layout(width='90%', height="35px")),

            data_frame=widgets.ToggleButtons(
                options=[
                    'detector', 'crystal', "laboratory"],
                value="detector",
                description='Data frame',
                tooltips=[
                    "If the data is still in the detector frame",
                    "If the data was interpolated into the crystal frame using\
                     (xrayutilities) or (transformation matrix + align_q=True)",
                    "If the data was interpolated into the laboratory frame \
                    using the transformation matrix (align_q = False)"
                ],
                style={'description_width': 'initial'}),

            ref_axis_q=widgets.Dropdown(
                options=["x", "y", "z"],
                value="y",
                description='Ref axis q',
                continuous_update=False,
                layout=Layout(width='15%'),
                tooltip="q will be aligned along that axis",
                style={'description_width': 'initial'}),

            save_frame=widgets.ToggleButtons(
                options=[
                    'crystal', 'laboratory', "lab_flat_sample"],
                value="lab_flat_sample",
                description='Final frame',
                tooltips=[
                    "Save the data with q aligned along ref_axis_q",
                    "Save the data in the laboratory frame (experimental \
                    geometry)",
                    "Save the data in the laboratory frame, with all sample\
                     angles rotated back to 0"
                ],
                style={'description_width': 'initial'}),

            isosurface_strain=widgets.FloatText(
                value=0.3,
                step=0.01,
                max=1,
                min=0,
                continuous_update=False,
                description='Isosurface strain:',
                tooltip="Threshold use for removing the outer layer (strain is\
                 undefined at the exact surface voxel)",
                readout=True,
                layout=Layout(width='20%'),
                style={
                    'description_width': 'initial'},
                disabled=False),

            strain_method=widgets.ToggleButtons(
                options=[
                    'default', 'defect'],
                value="default",
                description='Strain method',
                tooltips=[
                    "",
                    "Will offset the phase in a loop and keep the smallest \
                    magnitude value for the strain. See: F. Hofmann et al. \
                    PhysRevMaterials 4, 013801 (2020)"
                ],
                style={'description_width': 'initial'}),

            phase_offset=widgets.FloatText(
                value=0,
                step=0.01,
                min=0,
                max=360,
                continuous_update=False,
                description='Phase offset:',
                layout=Layout(width='15%'),
                readout=True,
                style={
                    'description_width': 'initial'},
                disabled=False),

            phase_offset_origin=widgets.Text(
                placeholder="(x, y, z), leave None for automatic.",
                description='Phase offset origin',
                continuous_update=False,
                layout=Layout(width='40%'),
                style={
                    'description_width': 'initial'},
            ),

            offset_method=widgets.Dropdown(
                options=["COM", "mean"],
                value="mean",
                description='Offset method:',
                continuous_update=False,
                layout=Layout(width='20%'),
                style={'description_width': 'initial'}),

            centering_method=widgets.Dropdown(
                options=[
                    "COM", "max", "max_com"],
                value="max_com",
                description='Centering method:',
                continuous_update=False,
                layout=Layout(width='25%'),
                style={'description_width': 'initial'}),

            unused_label_refraction=widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>\
                Parameters related to the refraction correction",
                style={
                    'description_width': 'initial'},
                layout=Layout(width='90%', height="35px")),

            correct_refraction=widgets.Checkbox(
                value=False,
                description='Correct refraction',
                # icon = 'check',
                style={
                    'description_width': 'initial'}
            ),

            optical_path_method=widgets.ToggleButtons(
                options=[
                    'threshold', 'defect'],
                value="threshold",
                description='Optical path method',
                tooltips=[
                    "Uses isosurface_strain to define the support for the \
                    optical path calculation",
                    "Tries to remove only outer layers even if the amplitude \
                    is lower than isosurface_strain inside the crystal"
                ],
                disabled=True,
                style={'description_width': 'initial'}),

            dispersion=widgets.FloatText(
                value=0.000050328,
                continuous_update=False,
                description='Dispersion (delta):',
                readout=True,
                style={
                    'description_width': 'initial'},
                disabled=True),

            absorption=widgets.FloatText(
                value=0.000050328,
                continuous_update=False,
                description='Absorption (beta):',
                readout=True,
                style={
                    'description_width': 'initial'},
                disabled=True),

            threshold_unwrap_refraction=widgets.FloatText(
                value=0.05,
                step=0.01,
                continuous_update=False,
                description='Threshold unwrap refraction:',
                readout=True,
                style={
                    'description_width': 'initial'},
                disabled=True),

            unused_label_options=widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>\
                Options",
                style={
                    'description_width': 'initial'},
                layout=Layout(width='90%', height="35px")),

            simulation=widgets.Checkbox(
                value=False,
                description='Simulated data',
                layout=Layout(width='33%'),
                style={
                    'description_width': 'initial'}
            ),

            invert_phase=widgets.Checkbox(
                value=True,
                description='Invert phase',
                layout=Layout(width='33%'),
                style={
                    'description_width': 'initial'}
            ),

            flip_reconstruction=widgets.Checkbox(
                value=False,
                description='Get conjugated object',
                layout=Layout(width='33%'),
                style={
                    'description_width': 'initial'}
            ),

            phase_ramp_removal=widgets.Dropdown(
                options=[
                    "gradient", "upsampling"],
                value="gradient",
                description='Phase ramp removal:',
                continuous_update=False,
                style={'description_width': 'initial'}),

            threshold_gradient=widgets.FloatText(
                value=1.0,
                step=0.01,
                continuous_update=False,
                description='Upper threshold gradient:',
                readout=True,
                style={
                    'description_width': 'initial'},
                disabled=False),

            save_raw=widgets.Checkbox(
                value=False,
                description='Save raw data',
                style={
                    'description_width': 'initial'}
            ),

            save_support=widgets.Checkbox(
                value=False,
                description='Save support',
                style={
                    'description_width': 'initial'}
            ),

            save=widgets.Checkbox(
                value=True,
                description='Save output',
                style={
                    'description_width': 'initial'}
            ),

            debug=widgets.Checkbox(
                value=False,
                description='Debug',
                style={
                    'description_width': 'initial'}
            ),

            roll_modes=widgets.Text(
                value="(0, 0, 0)",
                placeholder="(0, 0, 0)",
                description='Roll modes',
                continuous_update=False,
                layout=Layout(width='30%'),
                style={
                    'description_width': 'initial'},
            ),

            unused_label_data_vis=widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>\
                Parameters related to data visualization",
                style={
                    'description_width': 'initial'},
                layout=Layout(width='90%', height="35px")),

            align_axis=widgets.Checkbox(
                value=False,
                description='Align axis',
                style={
                    'description_width': 'initial'}
            ),

            ref_axis=widgets.Dropdown(
                options=["x", "y", "z"],
                value="y",
                description='Ref axis for align axis',
                continuous_update=False,
                layout=Layout(width='20%'),
                tooltip="q will be aligned along that axis",
                style={'description_width': 'initial'}),

            axis_to_align=widgets.Text(
                value="[0.0, 0.0, 0.0]",
                placeholder="[0.0, 0.0, 0.0]",
                description='Axis to align for ref axis',
                continuous_update=False,
                style={'description_width': 'initial'}),

            strain_range=widgets.FloatText(
                value=0.002,
                step=0.00001,
                continuous_update=False,
                description='Strain range:',
                readout=True,
                style={
                    'description_width': 'initial'},
                disabled=False),

            phase_range=widgets.FloatText(
                value=np.round(np.pi, 3),
                step=0.001,
                continuous_update=False,
                description='Phase range:',
                readout=True,
                style={
                    'description_width': 'initial'},
                disabled=False),

            grey_background=widgets.Checkbox(
                value=True,
                description='Grey background in plots',
                layout=Layout(width='25%'),
                style={
                    'description_width': 'initial'}
            ),

            tick_spacing=widgets.BoundedIntText(
                value="100",
                description='Tick spacing:',
                min=0,
                max=5000,
                layout=Layout(width='25%'),
                continuous_update=False,
                style={'description_width': 'initial'}),

            tick_direction=widgets.Dropdown(
                options=[
                    "out", "in", "inout"],
                value="inout",
                description='Tick direction:',
                layout=Layout(width='25%'),
                continuous_update=False,
                style={'description_width': 'initial'}),

            tick_length=widgets.BoundedIntText(
                value="3",
                description='Tick length:',
                min=0,
                max=50,
                continuous_update=False,
                layout=Layout(width='20%'),
                style={'description_width': 'initial'}),

            tick_width=widgets.BoundedIntText(
                value="1",
                description='Tick width:',
                min=0,
                max=10,
                continuous_update=False,
                layout=Layout(width='45%'),
                style={'description_width': 'initial'}),

            unused_label_average=widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>\
                Parameters for averaging several reconstructed objects",
                style={
                    'description_width': 'initial'},
                layout=Layout(width='90%', height="35px")),

            averaging_space=widgets.Dropdown(
                options=[
                    "reciprocal_space", "real_space"],
                value="reciprocal_space",
                description='Average method:',
                continuous_update=False,
                style={'description_width': 'initial'}),

            threshold_avg=widgets.FloatText(
                value=0.90,
                step=0.01,
                continuous_update=False,
                description='Average threshold:',
                readout=True,
                style={
                    'description_width': 'initial'},
                disabled=False),

            unused_label_apodize=widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>\
                Setup for phase averaging or apodization",
                style={
                    'description_width': 'initial'},
                layout=Layout(width='90%', height="35px")),

            apodize=widgets.Checkbox(
                value=True,
                description='Multiply diffraction pattern by filtering window',
                style={
                    'description_width': 'initial'}
            ),

            apodization_window=widgets.Dropdown(
                options=[
                    "normal", "tukey", "blackman"],
                value="blackman",
                description='Filtering window',
                continuous_update=False,
                style={'description_width': 'initial'}),

            half_width_avg_phase=widgets.BoundedIntText(
                value=1,
                continuous_update=False,
                description='Width of apodizing window:',
                readout=True,
                style={
                    'description_width': 'initial'},
                disabled=False),

            apodization_mu=widgets.Text(
                value="[0.0, 0.0, 0.0]",
                placeholder="[0.0, 0.0, 0.0]",
                description='Mu of gaussian window',
                continuous_update=False,
                style={'description_width': 'initial'}),

            apodization_sigma=widgets.Text(
                value="[0.30, 0.30, 0.30]",
                placeholder="[0.30, 0.30, 0.30]",
                description='Sigma of gaussian window',
                continuous_update=False,
                style={'description_width': 'initial'}),

            apodization_alpha=widgets.Text(
                value="[1.0, 1.0, 1.0]",
                placeholder="[1.0, 1.0, 1.0]",
                description='Alpha of gaussian window',
                continuous_update=False,
                style={'description_width': 'initial'}),

            unused_label_strain=widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>\
                Path to file",
                style={
                    'description_width': 'initial'},
                layout=Layout(width='90%', height="35px")),

            strain_folder=widgets.Dropdown(
                options=[x[0] + "/" for x in os.walk(os.getcwd())],
                value=os.getcwd() + "/",
                placeholder=os.getcwd() + "/",
                description='Data folder:',
                continuous_update=False,
                layout=Layout(width='90%'),
                style={'description_width': 'initial'}),

            reconstruction_file=widgets.Dropdown(
                options=[""]\
                + [os.path.basename(f) for f in sorted(
                    glob.glob(os.getcwd() + "/*.h5")
                    + glob.glob(os.getcwd() + "/*.cxi")
                    + glob.glob(os.getcwd() + "/*.npy")
                    + glob.glob(os.getcwd() + "/*.npz"),
                    key=os.path.getmtime)],
                description='Compatible file list',
                layout=Layout(width='90%'),
                style={'description_width': 'initial'}),

            run_strain=widgets.ToggleButtons(
                options=[
                    ("Clear/ Reload folder", False),
                    ("Run postprocessing", "run"),
                ],
                value=False,
                description='Run strain analysis',
                button_style='',
                icon='fast-forward',
                layout=Layout(width='90%'),
                style={'description_width': 'initial'}),
        )
        self._list_widgets_strain.children[-4].observe(
            self.strain_folder_handler, names="value")
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

        self.tab_data = interactive(
            self.load_data,
            unused_label_plot=widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>\
                Loads data files and displays it in the GUI",
                style={'description_width': 'initial'},
                layout=Layout(width='90%', height="35px")),

            folder=widgets.Dropdown(
                options=[x[0] + "/" for x in os.walk(os.getcwd())],
                value=os.getcwd() + "/",
                placeholder=os.getcwd() + "/",
                description='Data folder:',
                continuous_update=False,
                layout=Layout(width='90%'),
                style={'description_width': 'initial'}),

            filename=widgets.SelectMultiple(
                options=[""]
                + [os.path.basename(f) for f in sorted(
                    glob.glob(os.getcwd() + "/*.npy")
                    + glob.glob(os.getcwd() + "/*.npz")
                    + glob.glob(os.getcwd() + "/*.cxi")
                    + glob.glob(os.getcwd() + "/*.h5")
                    + glob.glob(os.getcwd() + "/*.png"),
                    key=os.path.getmtime)],
                rows=10,
                description='Compatible file list',
                layout=Layout(width='90%'),
                style={'description_width': 'initial'}),

            cmap=widgets.Dropdown(
                options=plt.colormaps(),
                value="jet",
                description="Color map:",
                continuous_update=False,
                layout=Layout(width='90%'),
                style={'description_width': 'initial'}),

            data_use=widgets.ToggleButtons(
                options=[
                    ("Clear/ Reload folder", False),
                    ('2D plot', "2D"),
                    ("Plot slices", "slices"),
                    ("3D plot", "3D"),
                    ("Create support", "create_support"),
                    ("Extract support", "extract_support"),
                    ("Smooth support", "smooth_support"),
                    ("Display .png image", "show_image"),
                    ("Display hdf5 tree", "hf_glance"),
                    ("Delete selected files", "delete")
                ],
                value=False,
                description='Load data',
                tooltips=[
                    "Clear the output and unload data from GUI, saves RAM",
                    "Load data and plot data slice interactively",
                    "Load data and plot data slices for each dimension in \
                    its middle",
                    "Load data and plot 3D data interactively",
                    "Load data and allow for the creation of a support \
                    interactively",
                    "Load data and allow for the creation of a support \
                    automatically",
                    "Load support and smooth its boundaries",
                    "Delete selected files, careful !!"
                ],
                button_style='',
                icon='fast-forward',
                layout=Layout(width='90%'),
                style={'description_width': 'initial'}),
        )
        self.tab_data.children[1].observe(
            self.plot_folder_handler, names="value")

        # Widgets for facet analysis
        self.tab_facet = interactive(
            self.init_facet_analysis,
            unused_label_facet=widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>\
                 Extract facet specific data from vtk file",
                style={
                    'description_width': 'initial'},
                layout=Layout(width='90%', height="35px")),

            parent_folder=widgets.Dropdown(
                options=[x[0] + "/" for x in os.walk(os.getcwd())],
                value=os.getcwd() + "/",
                placeholder=os.getcwd() + "/",
                description='Parent folder:',
                continuous_update=False,
                layout=Layout(width='90%'),
                style={'description_width': 'initial'}),

            vtk_file=widgets.Dropdown(
                options=sorted(glob.glob(os.getcwd()+"*.vtk"),
                               key=os.path.getmtime),
                description='vtk file in subdirectories:',
                layout=Layout(width='90%'),
                style={'description_width': 'initial'}),

            load_data=widgets.ToggleButtons(
                options=[
                    ("Clear/ Reload folder", False),
                    ('Load .vtk file', "load_csv"),
                ],
                value=False,
                description='Load vtk data',
                button_style='',
                layout=Layout(width='90%'),
                style={'description_width': 'initial'}),
        )
        self.tab_facet.children[1].observe(
            self.vtk_file_handler, names="value")

        # Widgets for readme tab
        self.tab_readme = interactive(
            self.display_readme,
            contents=widgets.ToggleButtons(
                options=[
                    "GUI",
                    'Preprocessing', 'Phase retrieval',
                    'Postprocessing', "Facet analysis"],
                value="GUI",
                description='Show info about:',
                tooltips=[
                    'Basic informations',
                    'Insight in the functions used for preprocessing',
                    'Insight in the functions used for phase retrieval',
                    'Insight in the functions used for postprocessing'
                    'Insight in the functions used for facet analysis'
                ],
                style={'description_width': 'initial'}))

        # Create the final window
        # Ignore phase retrieval tab if PyNX could not be imported
        if pynx_import and not plot_only:
            self.window = widgets.Tab(
                children=[
                    self.tab_init,
                    self.tab_detector,
                    self.tab_setup,
                    self.tab_preprocess,
                    self.tab_data_frame,
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
            self.window.set_title(4, 'Logs')
            self.window.set_title(5, 'Phase retrieval')
            self.window.set_title(6, 'Postprocess')
            self.window.set_title(7, 'Handle data')
            self.window.set_title(8, 'Facets')
            self.window.set_title(9, 'Readme')

        elif not pynx_import and not plot_only:
            self.window = widgets.Tab(
                children=[
                    self.tab_init,
                    self.tab_detector,
                    self.tab_setup,
                    self.tab_preprocess,
                    self.tab_data_frame,
                    self.tab_strain,
                    self.tab_data,
                    self.tab_facet,
                    self.tab_readme,
                ])
            self.window.set_title(0, 'Scan detail')
            self.window.set_title(1, 'Detector')
            self.window.set_title(2, 'Setup')
            self.window.set_title(3, "Preprocess")
            self.window.set_title(4, 'Logs')
            self.window.set_title(5, 'Strain')
            self.window.set_title(6, 'Plot data')
            self.window.set_title(7, 'Facets')
            self.window.set_title(8, 'Readme')

        elif plot_only:
            self.window = widgets.Tab(
                children=[
                    self.tab_data,
                ])
            self.window.set_title(0, 'Plot data')

        # Display the final window
        display(self.window)

    # Widgets interactive functions

    def initialize_directories(
        self,
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
        """
        if run_dir_init:
            # Create Dataset attribute
            self.Dataset = gui_iterable.Dataset(
                scan=scan, sample_name=sample_name,
                data_dir=data_dir, root_folder=root_folder)

            # Start to assign attributes
            self.Dataset.comment = comment
            self.Dataset.debug = debug
            self.Dataset.scan_name = self.Dataset.sample_name + \
                str(self.Dataset.scan)

            # Backend used for plotting
            self.matplotlib_backend = matplotlib_backend

            # Assign scan folder
            self.Dataset.scan_folder = self.Dataset.root_folder \
                + self.Dataset.scan_name + "/"
            print("Scan folder:", self.Dataset.scan_folder)

            # Assign preprocessing folder
            self.preprocessing_folder = self.Dataset.scan_folder \
                + "preprocessing/"

            # Assign postprocessing folder
            self.postprocessing_folder = self.Dataset.scan_folder \
                + "postprocessing/"

            # Update the directory structure
            gutil.hash_print("Updating directories ...")
            gutil.init_directories(
                scan_name=self.Dataset.scan_name,
                root_folder=self.Dataset.root_folder,
            )

            # Try and find SixS data, will also rotate the data
            template_imagefile, self.Dataset.data_dir = gutil.find_move_sixs_data(
                scan=self.Dataset.scan,
                scan_name=self.Dataset.scan_name,
                root_folder=self.Dataset.root_folder,
                data_dir=self.Dataset.data_dir,
            )

            # Save template_imagefile in GUI
            if template_imagefile != "":
                self._list_widgets_preprocessing.children[42].value\
                    = template_imagefile

            # Refresh folders
            self.sub_directories_handler(change=self.Dataset.scan_folder)

            # Refresh csv file
            self.tab_data_frame.children[1].options\
                = [x[0] + "/" for x in os.walk(self.Dataset.root_folder)]
            self.tab_data_frame.children[1].value = self.Dataset.root_folder+"/"
            self.csv_file_handler(self.Dataset.root_folder)

            # PyNX folder, refresh values
            self._list_widgets_phase_retrieval.children[1].value\
                = self.preprocessing_folder
            self.pynx_folder_handler(change=self.preprocessing_folder)

            # Plot folder, refresh values
            self.tab_data.children[1].value = self.preprocessing_folder
            self.plot_folder_handler(change=self.preprocessing_folder)

            # Strain folder, refresh values
            self._list_widgets_strain.children[-4].value\
                = self.preprocessing_folder
            self.strain_folder_handler(change=self.preprocessing_folder)

            # Facet folder, refresh values
            self.tab_facet.children[1].value = self.postprocessing_folder
            self.vtk_file_handler(change=self.postprocessing_folder)

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

                # Button to reload data
                button_reload_previous_data = Button(
                    description="Reload previous data (.cxi) from target \
                    directory ...",
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
                    display(buttons_init)
                    gutil.hash_print("Saving data, takes some time ...",
                                     hash_line_after=False)

                    try:
                        # Reciprocal space data
                        gutil.hash_print(
                            "Saving diffraction data and mask selected in the PyNX tab...")

                        # Define cxi operator
                        self.initialize_cdi_operator()

                        # Real space data
                        print(
                            "\n#########################################################################################\n"
                        )
                        if os.path.isfile(self.Dataset.reconstruction_file):
                            self.Dataset.to_cxi(
                                cxi_filename=self.cxi_filename,
                                reconstruction_filename=self.Dataset.reconstruction_file,
                            )
                        else:
                            self.Dataset.to_cxi(
                                cxi_filename=self.cxi_filename,
                                reconstruction_filename=False,
                            )

                        print(
                            "\n#########################################################################################\n"
                        )

                    except (AttributeError, UnboundLocalError):
                        print(
                            "Could not save reciprocal space data, select the\
                            \n intensity and the mask files in the phase\
                            \n retrieval tab first")

                    # Facets analysis output
                    try:
                        gutil.hash_print("Saving Facets class data",
                                         hash_line_after=False)
                        self.Facets.to_hdf5(
                            f"{self.Dataset.scan_folder}{self.Dataset.scan_name}.cxi")
                    except AttributeError:
                        gutil.hash_print(
                            "Could not save facets' data, run the analysis in the `Facets` tab first.", hash_line_after=False)

                    print(
                        "\n#########################################################################################\n"
                    )

        elif not run_dir_init:
            gutil.hash_print("Cleared window.")
            clear_output(True)

    # Preprocessing

    def initialize_preprocessing(
        self,
        unused_label_beamline,
        beamline,
        actuators,
        is_series,
        custom_scan,
        custom_images,
        custom_monitor,
        specfile_name,
        rocking_angle,
        unused_label_masking,
        flag_interact,
        background_plot,
        unused_label_centering,
        centering_method,
        bragg_peak,
        fix_size,
        center_fft,
        pad_size,
        normalize_flux,
        unused_label_filtering,
        mask_zero_event,
        median_filter,
        median_filter_order,
        phasing_binning,
        unused_label_reload,
        reload_previous,
        reload_orthogonal,
        preprocessing_binning,
        unused_label_saving,
        save_rawdata,
        save_to_npz,
        save_to_mat,
        save_to_vti,
        save_as_int,
        unused_label_detector,
        detector,
        # phasing_binning,
        # linearity_func
        # center_roi_x
        # center_roi_y
        roi_detector,
        # normalize_flux
        photon_threshold,
        photon_filter,
        # bin_during_loading todo
        # frames_pattern todo
        background_file,
        hotpixels_file,
        flatfield_file,
        template_imagefile,
        unused_label_ortho,
        use_rawdata,
        interpolation_method,
        fill_value_mask,
        beam_direction,
        sample_offsets,
        sdd,
        energy,
        custom_motors,
        unused_label_xru,
        align_q,
        ref_axis_q,
        direct_beam,
        dirbeam_detector_angles,
        # bragg_peak
        outofplane_angle,
        inplane_angle,
        tilt_angle,
        sample_inplane,
        sample_outofplane,
        offset_inplane,
        cch1,
        cch2,
        detrot,
        tiltazimuth,
        tilt_detector,
        unused_label_preprocess,
        init_para
    ):
        """
        Initialize the parameters used in bcdi_preprocess_BCDI.py.
        Necessary for preprocessing and postprocessing.

        If init_para is True, displays a button that allow
        the user to run the bcdi_preprocess_BCDI script

        All the parameters values are then saved in a yaml configuration file.

        Parameters used in the interactive masking GUI:

        :param flag_interact: e.g. True
         True to interact with plots, False to close it automatically
        :param background_plot: e.g. "0.5"
         background color for the GUI in level of grey in [0,1], 0 being dark.
         For visual comfort during interactive masking.
        :param backend: e.g. "Qt5Agg"
         Backend used in script, change to "Agg" to make sure the figures are
         saved, not compaticle with interactive masking. Other possibilities
         are 'module://matplotlib_inline.backend_inline'
         default value is "Qt5Agg"

        Parameters related to data cropping/padding/centering #

        :param centering_method: e.g. "max"
         Bragg peak determination: 'max' or 'com', 'max' is better usually.
         It will be overridden by 'fix_bragg' if not empty
        :param fix_size: e.g. [0, 256, 10, 240, 50, 350]
         crop the array to that predefined size considering the full detector.
         [zstart, zstop, ystart, ystop, xstart, xstop], ROI will be defaulted
         to [] if fix_size is provided. Leave None otherwise
        :param center_fft: e.g. "skip"
         how to crop/pad/center the data, available options: 'crop_sym_ZYX',
         'crop_asym_ZYX', 'pad_asym_Z_crop_sym_YX', 'pad_sym_Z_crop_asym_YX',
         'pad_sym_Z', 'pad_asym_Z', 'pad_sym_ZYX','pad_asym_ZYX' or 'skip'
        :param pad_size: e.g. [256, 512, 512]
         Use this to pad the array. Used in 'pad_sym_Z_crop_sym_YX',
         'pad_sym_Z' and 'pad_sym_ZYX'. Leave None otherwise.

        Parameters for data filtering

        :param mask_zero_event: e.g. False
         mask pixels where the sum along the rocking curve is zero may be dead
         pixels
        :param median_filter: e.g. "skip"
         which filter to apply, available filters:

         - 'median': to apply a med2filter [3,3]
         - 'interp_isolated': to interpolate isolated empty pixels based
            on 'medfilt_order' parameter
         - 'mask_isolated': mask isolated empty pixels
         - 'skip': skip filtering

        :param median_filter_order: e.g. 7
         minimum number of non-zero neighboring pixels to apply filtering

        Parameters used when reloading processed data

        :param reload_previous: e.g. False
         True to resume a previous masking (load data and mask)
        :param reload_orthogonal: e.g. False
         True if the reloaded data is already intepolated in an orthonormal
         frame
        :param preprocessing_binning: e.g. [1, 1, 1]
         binning factors in each dimension of the binned data to be reloaded

        Options for saving:

        :param save_rawdata: e.g. False
         True to save also the raw data when use_rawdata is False
        :param save_to_npz: e.g. True
         True to save the processed data in npz format
        :param save_to_mat: e.g. False
         True to save also in .mat format
        :param save_to_vti: e.g. False
         True to save the orthogonalized diffraction pattern to VTK file
        :param save_as_int: e.g. False
         True to save the result as an array of integers (save space)

        Parameters for the beamline:

        :param beamline: e.g. "ID01"
         name of the beamline, used for data loading and normalization by
         monitor
        :param actuators: e.g. {'rocking_angle': 'actuator_1_1'}
         optional dictionary that can be used to define the entries
         corresponding to actuators in data files (useful at CRISTAL where the
         location of data keeps changing, or to declare a non-standard monitor)
        :param is_series: e.g. True
         specific to series measurement at P10
        :param rocking_angle: e.g. "outofplane"
         "outofplane" for a sample rotation around x outboard, "inplane" for a
         sample rotation around y vertical up, "energy"
        :param specfile_name: e.g. "l5.spec"
         beamline-dependent parameter, use the following template:

         - template for ID01 and 34ID: name of the spec file if it is at the
         default location (in root_folder) or full path to the spec file
         - template for SIXS: full path of the alias dictionnary or None to use
          the one in the package folder
         - for P10, either None (if you are using the same directory structure
         as the beamline) or the full path to the .fio file
         - template for all other beamlines: None

        Parameters for custom scans:

        :param custom_scan: e.g. False
         True for a stack of images acquired without scan, e.g. with ct in a
         macro, or when there is no spec/log file available
        :param custom_images: list of image numbers for the custom_scan, None
         otherwise
        :param custom_monitor: list of monitor values for normalization for the
         custom_scan, None otherwise

        Parameters for the detector:

        :param detector: e.g. "Maxipix"
         name of the detector
        :param phasing_binning: e.g. [1, 2, 2]
         binning to apply to the data (stacking dimension, detector vertical
         axis, detector horizontal axis)
        :param linearity_func: name of the linearity correction for the
         detector, leave None otherwise.
        :param center_roi_x: e.g. 1577
         horizontal pixel number of the center of the ROI for data loading.
         Leave None to use the full detector.
        :param center_roi_y: e.g. 833
         vertical pixel number of the center of the ROI for data loading.
         Leave None to use the full detector.
        :param roi_detector: e.g.[0, 250, 10, 210]
         region of interest of the detector to load. If "x_bragg" or "y_bragg"
         are not None, it will consider that the current values in roi_detector
         define a window around the Bragg peak position and the final output
         will be: [y_bragg - roi_detector[0], y_bragg + roi_detector[1],
         x_bragg - roi_detector[2], x_bragg + roi_detector[3]]. Leave None to
         use the full detector. Use with center_fft='skip' if you want this
         exact size for the output.
        :param normalize_flux: e.g. "monitor"
         'monitor' to normalize the intensity by the default monitor values,
         'skip' to do nothing
        :param photon_threshold: e.g. 0
         voxels with a smaller intensity will be set to 0.
        :param photon_filter: e.g. "loading"
         'loading' or 'postprocessing', when the photon threshold should be
         applied. If 'loading', it is applied before binning;
         if 'postprocessing', it is applied at the end of the script before
         saving
        :param bin_during_loading: e.g. False
         True to bin during loading, faster
        :param frames_pattern:  list of int, of length data.shape[0].
         If frames_pattern is 0 at index, the frame at data[index] will be
         skipped, if 1 the frame will be added to the stack. Use this if you
         need to remove some frames and you know it in advance.
        :param background_file: non-empty file path or None
        :param hotpixels_file: non-empty file path or None
        :param flatfield_file: non-empty file path or None
        :param template_imagefile: e.g. "data_mpx4_%05d.edf.gz"
         use one of the following template:

         - template for ID01: 'data_mpx4_%05d.edf.gz' or
          'align_eiger2M_%05d.edf.gz'
         - template for SIXS_2018: 'align.spec_ascan_mu_%05d.nxs'
         - template for SIXS_2019: 'spare_ascan_mu_%05d.nxs'
         - template for Cristal: 'S%d.nxs'
         - template for P10: '_master.h5'
         - template for NANOMAX: '%06d.h5'
         - template for 34ID: 'Sample%dC_ES_data_51_256_256.npz'

        Parameters below if you want to orthogonalize the data before phasing:

        :param use_rawdata: e.g. True
         False for using data gridded in laboratory frame, True for using data
         in detector frame
        :param interpolation_method: e.g. "xrayutilities"
         'xrayutilities' or 'linearization'
        :param fill_value_mask: e.g. 0
         0 (not masked) or 1 (masked). It will define how the pixels outside of
         the data range are processed during the interpolation. Because of the
         large number of masked pixels, phase retrieval converges better if the
         pixels are not masked (0 intensity imposed). The data is by default set
         to 0 outside of the defined range.
        :param beam_direction: e.g. [1, 0, 0]
         beam direction in the laboratory frame (downstream, vertical up,
         outboard)
        :param sample_offsets: e.g. None
         tuple of offsets in degrees of the sample for each sample circle
         (outer first).
         convention: the sample offsets will be subtracted to the motor values.
         Leave None if there is no offset.
        :param sdd: e.g. 0.50678
         in m, sample to detector distance in m
        :param energy: e.g. 9000
         X-ray energy in eV, it can be a number or a list in case of
         energy scans.
        :param custom_motors: e.g. {"mu": 0, "phi": -15.98, "chi": 90,
         "theta": 0, "delta": -0.5685, "gamma": 33.3147}
         use this to declare motor positions if there is not log file,
         None otherwise

        Parameters when orthogonalizing the data before phasing  using the
        linearized transformation matrix:

        :param align_q: e.g. True
         if True it rotates the crystal to align q, along one axis of the
         array. It is used only when interp_method is 'linearization'
        :param ref_axis_q: e.g. "y"  # q will be aligned along that axis
        :param direct_beam: e.g. [125, 362]
         [vertical, horizontal], direct beam position on the unbinned, full detector
         measured with detector angles given by `dirbeam_detector_angles`. It will be used
         to calculate the real detector angles for the measured Bragg peak. Leave None for
         no correction.
        :param dirbeam_detector_angles: e.g. [1, 25]
         [outofplane, inplane] detector angles in degrees for the direct beam measurement.
         Leave None for no correction
        :param bragg_peak: e.g. [121, 321, 256]
         Bragg peak position [z_bragg, y_bragg, x_bragg] considering the unbinned full
         detector. If 'outofplane_angle' and 'inplane_angle' are None and the direct beam
         position is provided, it will be used to calculate the correct detector angles.
         It is useful if there are hotpixels or intense aliens. Leave None otherwise.
        :param outofplane_angle: e.g. 42.6093
         detector angle in deg (rotation around x outboard, typically delta),
         corrected for the direct beam position. Leave None to use the
         uncorrected position.
        :param inplane_angle: e.g. -0.5783
         detector angle in deg(rotation around y vertical up, typically gamma),
         corrected for the direct beam position. Leave None to use the
         uncorrected position.

        Parameters when orthogonalizing the data before phasing using
        xrayutilities. xrayutilities uses the xyz crystal frame (for zero
        incident angle x is downstream, y outboard, and z vertical up):

        :param sample_inplane: e.g. [1, 0, 0]
         sample inplane reference direction along the beam at 0 angles in
         xrayutilities frame
        :param sample_outofplane: e.g. [0, 0, 1]
         surface normal of the sample at 0 angles in xrayutilities frame
        :param offset_inplane: e.g. 0
         outer detector angle offset as determined by xrayutilities area
         detector initialization
        :param cch1: e.g. 208
         direct beam vertical position in the full unbinned detector for
         xrayutilities 2D detector calibration
        :param cch2: e.g. 154
         direct beam horizontal position in the full unbinned detector for
         xrayutilities 2D detector calibration
        :param detrot: e.g. 0
         detrot parameter from xrayutilities 2D detector calibration
        :param tiltazimuth: e.g. 360
         tiltazimuth parameter from xrayutilities 2D detector calibration
        :param tilt_detector: e.g. 0
         tilt parameter from xrayutilities 2D detector calibration
        """
        if init_para:
            # Disable all widgets until the end of the program, will update
            # automatticaly after
            for w in self._list_widgets_init_dir.children[:-1]:
                w.disabled = True

            for w in self._list_widgets_preprocessing.children[:-2]:
                w.disabled = True

            # Save parameter values as attributes
            self.Dataset.beamline = beamline
            self.Dataset.actuators = actuators
            self.Dataset.is_series = is_series
            self.Dataset.custom_scan = custom_scan
            self.Dataset.custom_images = custom_images
            self.Dataset.custom_monitor = custom_monitor
            self.Dataset.specfile_name = specfile_name
            self.Dataset.rocking_angle = rocking_angle
            self.Dataset.flag_interact = flag_interact
            self.Dataset.background_plot = str(background_plot)
            if centering_method == "manual":  # will be overridden
                self.Dataset.centering_method = "max"
            else:
                self.Dataset.centering_method = centering_method
            self.Dataset.bragg_peak = bragg_peak
            self.Dataset.fix_size = fix_size
            self.Dataset.center_fft = center_fft
            self.Dataset.pad_size = pad_size
            self.Dataset.mask_zero_event = mask_zero_event
            self.Dataset.median_filter = median_filter
            self.Dataset.median_filter_order = median_filter_order
            self.Dataset.reload_previous = reload_previous
            self.Dataset.reload_orthogonal = reload_orthogonal
            self.Dataset.preprocessing_binning = preprocessing_binning
            self.Dataset.save_rawdata = save_rawdata
            self.Dataset.save_to_npz = save_to_npz
            self.Dataset.save_to_mat = save_to_mat
            self.Dataset.save_to_vti = save_to_vti
            self.Dataset.save_as_int = save_as_int
            self.Dataset.detector = detector
            self.Dataset.phasing_binning = phasing_binning
            self.Dataset.linearity_func = None  # TODO
            self.Dataset.roi_detector = roi_detector
            self.Dataset.normalize_flux = normalize_flux
            self.Dataset.photon_threshold = photon_threshold
            self.Dataset.photon_filter = photon_filter
            self.Dataset.bin_during_loading = True  # TODO
            self.Dataset.frames_pattern = None  # TODO
            self.Dataset.background_file = background_file
            self.Dataset.hotpixels_file = hotpixels_file
            self.Dataset.flatfield_file = flatfield_file
            self.Dataset.template_imagefile = template_imagefile
            self.Dataset.use_rawdata = not use_rawdata
            self.Dataset.interpolation_method = interpolation_method
            self.Dataset.fill_value_mask = fill_value_mask
            self.Dataset.beam_direction = beam_direction
            self.Dataset.sample_offsets = sample_offsets
            self.Dataset.sdd = sdd
            self.Dataset.energy = energy
            self.Dataset.custom_motors = custom_motors
            self.Dataset.align_q = align_q
            self.Dataset.ref_axis_q = ref_axis_q
            self.Dataset.direct_beam = direct_beam
            self.Dataset.dirbeam_detector_angles = dirbeam_detector_angles
            # bragg_peak
            self.Dataset.outofplane_angle = outofplane_angle
            self.Dataset.inplane_angle = inplane_angle
            self.Dataset.tilt_angle = tilt_angle
            self.Dataset.sample_inplane = sample_inplane
            self.Dataset.sample_outofplane = sample_outofplane
            self.Dataset.offset_inplane = offset_inplane
            self.Dataset.cch1 = cch1
            self.Dataset.cch2 = cch2
            self.Dataset.detrot = detrot
            self.Dataset.tiltazimuth = tiltazimuth
            self.Dataset.tilt_detector = tilt_detector

            # Extract dict, list and tuple from strings
            list_parameters = ["bragg_peak", "custom_images",
                               "fix_size", "pad_size", "roi_detector",
                               "direct_beam", "dirbeam_detector_angles"]

            tuple_parameters = [
                "phasing_binning", "preprocessing_binning",  "beam_direction",
                "sample_offsets", "sample_inplane", "sample_outofplane"]

            dict_parameters = ["actuators", "custom_motors"]

            try:
                for p in list_parameters:
                    if getattr(self.Dataset, p) == "":
                        setattr(self.Dataset, p, [])
                    else:
                        setattr(self.Dataset, p, literal_eval(
                            getattr(self.Dataset, p)))
                    # print(f"{p}:", getattr(self.Dataset, p))
            except ValueError:
                gutil.hash_print(f"Wrong list syntax for {p}")

            try:
                for p in tuple_parameters:
                    if getattr(self.Dataset, p) == "":
                        setattr(self.Dataset, p, ())
                    else:
                        setattr(self.Dataset, p, literal_eval(
                            getattr(self.Dataset, p)))
                    # print(f"{p}:", getattr(self.Dataset, p))
            except ValueError:
                gutil.hash_print(f"Wrong tuple syntax for {p}")

            try:
                for p in dict_parameters:
                    if getattr(self.Dataset, p) == "":
                        setattr(self.Dataset, p, None)  # or {}
                    else:
                        if literal_eval(getattr(self.Dataset, p)) == {}:
                            setattr(self.Dataset, p, None)
                        else:
                            setattr(self.Dataset, p, literal_eval(
                                getattr(self.Dataset, p)))
                    # print(f"{p}:", getattr(self.Dataset, p))
            except ValueError:
                gutil.hash_print(f"Wrong dict syntax for {p}")

            # Set None if we are not using custom scans
            if not self.Dataset.custom_scan:
                self.Dataset.custom_images = None
                self.Dataset.custom_monitor = None

            # Empty parameters are set to None (bcdi syntax)
            if self.Dataset.background_file == "":
                self.Dataset.background_file = None

            if self.Dataset.hotpixels_file == "":
                self.Dataset.hotpixels_file = None

            if self.Dataset.flatfield_file == "":
                self.Dataset.flatfield_file = None

            if self.Dataset.specfile_name == "":
                self.Dataset.specfile_name = None

            button_run_preprocess = Button(
                description="Run data preprocessing...",
                continuous_update=False,
                button_style='',
                layout=Layout(width='40%'),
                style={'description_width': 'initial'},
                icon='fast-forward')
            display(button_run_preprocess)
            gutil.hash_print("Parameters initialized...")

            @ button_run_preprocess.on_click
            def action_button_run_preprocess(selfbutton):
                """Run preprocessing script"""
                # Clear output
                clear_output(True)
                display(button_run_preprocess)

                # Change data_dir and root folder depending on beamline
                if self.Dataset.beamline == "SIXS_2019":
                    data_dir = self.Dataset.data_dir

                elif self.Dataset.beamline == "P10":
                    data_dir = f"{self.Dataset.data_dir}{self.Dataset.sample_name}_{self.Dataset.scan:05d}/e4m/"

                elif self.Dataset.beamline in ("ID01", "ID01BLISS"):
                    data_dir = self.Dataset.data_dir

                # Create config file
                gutil.create_yaml_file(
                    fname=f"{self.preprocessing_folder}config_preprocessing.yml",
                    scans=self.Dataset.scan,
                    root_folder=self.Dataset.root_folder,
                    save_dir=self.preprocessing_folder,
                    data_dir=data_dir,
                    sample_name=self.Dataset.sample_name,
                    comment=self.Dataset.comment,
                    debug=self.Dataset.debug,
                    # parameters used in masking
                    flag_interact=self.Dataset.flag_interact,
                    background_plot=self.Dataset.background_plot,
                    backend=self.matplotlib_backend,
                    # parameters related to data cropping/padding/centering
                    centering_method=self.Dataset.centering_method,
                    fix_size=self.Dataset.fix_size,
                    center_fft=self.Dataset.center_fft,
                    pad_size=self.Dataset.pad_size,
                    # parameters for data filtering
                    mask_zero_event=self.Dataset.mask_zero_event,
                    median_filter=self.Dataset.median_filter,
                    median_filter_order=self.Dataset.median_filter_order,
                    # parameters used when reloading processed data
                    reload_previous=self.Dataset.reload_previous,
                    reload_orthogonal=self.Dataset.reload_orthogonal,
                    preprocessing_binning=self.Dataset.preprocessing_binning,
                    # saving options
                    save_rawdata=self.Dataset.save_rawdata,
                    save_to_npz=self.Dataset.save_to_npz,
                    save_to_mat=self.Dataset.save_to_mat,
                    save_to_vti=self.Dataset.save_to_vti,
                    save_as_int=self.Dataset.save_as_int,
                    # define beamline related parameters
                    beamline=self.Dataset.beamline,
                    actuators=self.Dataset.actuators,
                    is_series=self.Dataset.is_series,
                    rocking_angle=self.Dataset.rocking_angle,
                    specfile_name=self.Dataset.specfile_name,
                    # parameters for custom scans
                    custom_scan=self.Dataset.custom_scan,
                    custom_images=self.Dataset.custom_images,
                    custom_monitor=self.Dataset.custom_monitor,
                    # detector related parameters
                    detector=self.Dataset.detector,
                    phasing_binning=self.Dataset.phasing_binning,
                    linearity_func=self.Dataset.linearity_func,
                    # center_roi_x
                    # center_roi_y
                    roi_detector=self.Dataset.roi_detector,
                    normalize_flux=self.Dataset.normalize_flux,
                    photon_threshold=self.Dataset.photon_threshold,
                    photon_filter=self.Dataset.photon_filter,
                    bin_during_loading=self.Dataset.bin_during_loading,
                    frames_pattern=self.Dataset.frames_pattern,
                    background_file=self.Dataset.background_file,
                    hotpixels_file=self.Dataset.hotpixels_file,
                    flatfield_file=self.Dataset.flatfield_file,
                    template_imagefile=self.Dataset.template_imagefile,
                    # define parameters below if you want to orthogonalize the
                    # data before phasing
                    use_rawdata=self.Dataset.use_rawdata,
                    interpolation_method=self.Dataset.interpolation_method,
                    fill_value_mask=self.Dataset.fill_value_mask,
                    beam_direction=self.Dataset.beam_direction,
                    sample_offsets=self.Dataset.sample_offsets,
                    sdd=self.Dataset.sdd,
                    energy=self.Dataset.energy,
                    custom_motors=self.Dataset.custom_motors,
                    # parameters when orthogonalizing the data before
                    # phasing  using the linearized transformation matrix
                    align_q=self.Dataset.align_q,
                    ref_axis_q=self.Dataset.ref_axis_q,
                    direct_beam=self.Dataset.direct_beam,
                    dirbeam_detector_angles=self.Dataset.dirbeam_detector_angles,
                    bragg_peak=self.Dataset.bragg_peak,
                    outofplane_angle=self.Dataset.outofplane_angle,
                    inplane_angle=self.Dataset.inplane_angle,
                    tilt_angle=self.Dataset.tilt_angle,
                    # parameters when orthogonalizing the data before phasing
                    # using xrayutilities
                    sample_inplane=self.Dataset.sample_inplane,
                    sample_outofplane=self.Dataset.sample_outofplane,
                    offset_inplane=self.Dataset.offset_inplane,
                    cch1=self.Dataset.cch1,
                    cch2=self.Dataset.cch2,
                    detrot=self.Dataset.detrot,
                    tiltazimuth=self.Dataset.tiltazimuth,
                    tilt_detector=self.Dataset.tilt_detector,
                )

                # Run bcdi_preprocess
                print(
                    "\n#########################################################################################\n"
                )
                print(
                    f"Running: $ {self.path_scripts}/bcdi_preprocess_BCDI.py")
                print(
                    f"Config file: {self.preprocessing_folder}config_preprocessing.yml")
                print(
                    "\n#########################################################################################\n"
                )

                # Construct the argument parser
                ap = argparse.ArgumentParser()

                # Load the config file
                config_file = self.preprocessing_folder + "/config_preprocessing.yml"
                parser = ConfigParser(config_file)
                args = parser.load_arguments()
                args["time"] = f"{datetime.now()}"

                # Run function
                run_preprocessing(prm=args)
                gutil.hash_print("End of script")

                # Button to save metadata
                button_save_metadata = Button(
                    description="Save metadata",
                    continuous_update=False,
                    button_style='',
                    layout=Layout(width='40%'),
                    style={'description_width': 'initial'},
                    icon='fast-forward')

                @ button_save_metadata.on_click
                def action_button_save_metadata(selfbutton):
                    try:
                        # Get latest file
                        metadata_file = sorted(
                            glob.glob(
                                f"{self.preprocessing_folder}*preprocessing*.h5"),
                            key=os.path.getmtime)[-1]

                        gutil.extract_metadata(
                            scan_nb=self.Dataset.scan,
                            metadata_file=metadata_file,
                            gwaihir_dataset=self.Dataset,
                            metadata_csv_file=self.Dataset.root_folder
                            + "metadata.csv"
                        )
                    except (IndexError, TypeError):
                        gutil.hash_print(
                            f"Could not find any .h5 file in {self.preprocessing_folder}")

                    # PyNX folder, refresh
                    self._list_widgets_phase_retrieval.children[1].value\
                        = self.preprocessing_folder
                    self.pynx_folder_handler(change=self.preprocessing_folder)

                    # Plot folder, refresh
                    self.tab_data.children[1].value = self.preprocessing_folder
                    self.plot_folder_handler(change=self.preprocessing_folder)

                display(button_save_metadata)

                # Change window view
                # self.window.selected_index = 8

        if not init_para:
            plt.close()
            clear_output(True)
            gutil.hash_print("Cleared window.")

    # Phase retrieval

    def initialize_cdi_operator(self, save_as_cxi=True):
        """
        Initialize the cdi operator by processing the possible inputs:
            iobs, mask, support, obj
        Will also crop and center the data if specified.
        Loads phase retrieval tab parameters values.

        :param save_as_cxi: e.g. True
         Save the instanced cdi object as .cxi following the cxi convention

        return: cdi operator
        """
        if self.Dataset.iobs not in ("", None):
            if self.Dataset.iobs.endswith(".npy"):
                iobs = np.load(self.Dataset.iobs)
                print("\tCXI input: loading data")
            elif self.Dataset.iobs.endswith(".npz"):
                try:
                    iobs = np.load(self.Dataset.iobs)["data"]
                    print("\tCXI input: loading data")
                except KeyError:
                    print("\t\"data\" key does not exist.")

            if self.Dataset.rebin != (1, 1, 1):
                iobs = bin_data(iobs, self.Dataset.rebin)

            # fft shift
            iobs = fftshift(iobs)

        else:
            self.Dataset.iobs = None
            iobs = None

        if self.Dataset.mask not in ("", None, self.Dataset.parent_folder):
            if self.Dataset.mask.endswith(".npy"):
                mask = np.load(self.Dataset.mask).astype(np.int8)
                nb = mask.sum()
                print("\tCXI input: loading mask, with %d pixels masked (%6.3f%%)" % (
                    nb, nb * 100 / mask.size))
            elif self.Dataset.mask.endswith(".npz"):
                try:
                    mask = np.load(self.Dataset.mask)[
                        "mask"].astype(np.int8)
                    nb = mask.sum()
                    print("\tCXI input: loading mask, with %d pixels masked (%6.3f%%)" % (
                        nb, nb * 100 / mask.size))
                except KeyError:
                    print("\t\"mask\" key does not exist.")

            if self.Dataset.rebin != (1, 1, 1):
                mask = bin_data(mask, self.Dataset.rebin)

            # fft shift
            mask = fftshift(mask)

        else:
            self.Dataset.mask = None
            mask = None

        if self.Dataset.support not in ("", None, self.Dataset.parent_folder):
            if self.Dataset.support.endswith(".npy"):
                support = np.load(self.Dataset.support)
                print("\tCXI input: loading support")
            elif self.Dataset.support.endswith(".npz"):
                try:
                    support = np.load(self.Dataset.support)["data"]
                    print("\tCXI input: loading support")
                except (FileNotFoundError, ValueError):
                    print("\tFile not supported or does not exist.")
                except KeyError:
                    print("\t\"data\" key does not exist.")
                    try:
                        support = np.load(self.Dataset.support)["support"]
                        print("\tCXI input: loading support")
                    except KeyError:
                        print("\t\"support\" key does not exist.")
                        try:
                            support = np.load(self.Dataset.support)["obj"]
                            print("\tCXI input: loading support")
                        except KeyError:
                            print(
                                "\t\"obj\" key does not exist."
                                "\t--> Could not load support array."
                            )

            if self.Dataset.rebin != (1, 1, 1):
                support = bin_data(support, self.Dataset.rebin)

            # fft shift
            support = fftshift(support)

        else:
            self.Dataset.support = None
            support = None

        if self.Dataset.obj not in ("", None, self.Dataset.parent_folder):
            if self.Dataset.obj.endswith(".npy"):
                obj = np.load(self.Dataset.obj)
                print("\tCXI input: loading object")
            elif self.Dataset.obj.endswith(".npz"):
                try:
                    obj = np.load(self.Dataset.obj)["data"]
                    print("\tCXI input: loading object")
                except KeyError:
                    print("\t\"data\" key does not exist.")

            if self.Dataset.rebin != (1, 1, 1):
                obj = bin_data(obj, self.Dataset.rebin)

            # fft shift
            obj = fftshift(obj)

        else:
            self.Dataset.obj = None
            obj = None

        # Center and crop data
        if self.Dataset.auto_center_resize:
            if iobs.ndim == 3:
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

                if self.Dataset.max_size is not None:
                    nx = min(nx, self.Dataset.max_size)
                    ny = min(ny, self.Dataset.max_size)
                    nz = min(nz, self.Dataset.max_size)

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
                if self.Dataset.max_size is not None:
                    nx = min(nx, self.Dataset.max_size)
                    ny = min(ny, self.Dataset.max_size)
                    nz = min(nz, self.Dataset.max_size)

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
        cdi = CDI(iobs,
                  support=support,
                  obj=obj,
                  mask=mask,
                  wavelength=self.Dataset.wavelength,
                  pixel_size_detector=self.Dataset.pixel_size_detector,
                  detector_distance=self.Dataset.sdd,
                  )

        if save_as_cxi:
            # Save diffraction pattern
            self.cxi_filename = "{}/preprocessing/{}.cxi".format(
                self.Dataset.scan_folder,
                self.Dataset.iobs.split("/")[-1].split(".")[0]
            )
            self.save_as_cxi(cdi_operator=cdi, path_to_cxi=self.cxi_filename)

        return cdi

    def initialize_phase_retrieval(
        self,
        unused_label_data,
        parent_folder,
        iobs,
        mask,
        support,
        obj,
        auto_center_resize,
        max_size,
        unused_label_support,
        support_threshold,
        support_only_shrink,
        support_update_period,
        support_smooth_width,
        support_post_expand,
        unused_label_psf,
        psf,
        psf_model,
        fwhm,
        eta,
        update_psf,
        unused_label_algo,
        nb_hio,
        nb_raar,
        nb_er,
        nb_ml,
        nb_run,
        unused_label_filtering,
        filter_criteria,
        nb_run_keep,
        unused_label_options,
        live_plot,
        # zero_mask,
        # crop_output,
        positivity,
        beta,
        detwin,
        rebin,
        verbose,
        pixel_size_detector,
        unused_label_phase_retrieval,
        run_phase_retrieval,
        unused_label_run_pynx_tools,
        run_pynx_tools,
    ):
        """
        Get parameters values from widgets and run phase retrieval Possible
        to run phase retrieval via the CLI (with ot without MPI) Or directly in
        python using the operators.

        :param parent_folder: folder in which the raw data files are, and where the
         output will be saved
        :param iobs: 2D/3D observed diffraction data (intensity).
          Assumed to be corrected and following Poisson statistics, will be
          converted to float32. Dimensions should be divisible by 4 and have a
          prime factor decomposition up to 7. Internally, the following special
          values are used:
          * values<=-1e19 are masked. Among those, values in ]-1e38;-1e19] are
             estimated values, stored as -(iobs_est+1)*1e19, which can be used
             to make a loose amplitude projection.
            Values <=-1e38 are masked (no amplitude projection applied), just
            below the minimum float32 value
          * -1e19 < values <= 1 are observed but used as free pixels
            If the mask is not supplied, then it is assumed that the above
            special values are used.
        :param support: initial support in real space (1 = inside support,
         0 = outside)
        :param obj: initial object. If None, it should be initialised later.
        :param mask: mask for the diffraction data (0: valid pixel, >0: masked)
        :param auto_center_resize: if used (command-line keyword) or =True,
         the input data will be centered and cropped  so that the size of the
         array is compatible with the (GPU) FFT library used. If 'roi' is used,
         centering is based on ROI. [default=False]
        :param max_size=256: maximum size for the array used for analysis,
         along all dimensions. The data will be cropped to this value after
         centering. [default: no maximum size]
        :param support_threshold: must be between 0 and 1. Only points with
         object amplitude above a value equal to relative_threshold *
         reference_value are kept in the support.
         reference_value can either:
            - use the fact that when converged, the square norm of the object
            is equal to the number of recorded photons (normalized Fourier
            Transform). Then: reference_value = sqrt((abs(obj)**2).sum()/
            nb_points_support)
            - or use threshold_percentile (see below, very slow, deprecated)
        :param support_smooth_width: smooth the object amplitude using a
         gaussian of this width before calculating new support
         If this is a scalar, the smooth width is fixed to this value.
         If this is a 3-value tuple (or list or array), i.e. 'smooth_width=2,
         0.5,600', the smooth width will vary with the number of cycles
         recorded in the CDI object (as cdi.cycle), varying exponentially from
         the first to the second value over the number of cycles specified by
         the last value.
         With 'smooth_width=a,b,nb':
         - smooth_width = a * exp(-cdi.cycle/nb*log(b/a)) if cdi.cycle < nb
         - smooth_width = b if cdi.cycle >= nb
        :param support_only_shrink: if True, the support can only shrink
        :param method: either 'max' or 'average' or 'rms' (default), the
         threshold will be relative to either the maximum amplitude in the
         object, or the average or root-mean-square amplitude (computed inside
         support)
        :param support_post_expand=1: after the new support has been calculated,
        it can be processed using the SupportExpand operator, either one or
        multiple times, in order to 'clean' the support:
         - 'post_expand=1' will expand the support by 1 pixel
         - 'post_expand=-1' will shrink the support by 1 pixel
         - 'post_expand=(-1,1)' will shrink and then expand the support by
         1 pixel
         - 'post_expand=(-2,3)' will shrink and then expand the support by
         respectively 2 and 3 pixels
        :param psf: e.g. True
         whether or not to use the PSF, partial coherence point-spread function,
         estimated with 50 cycles of Richardson-Lucy
        :param psf_model: "lorentzian", "gaussian" or "pseudo-voigt", or None
         to deactivate
        :param fwhm: the full-width at half maximum, in pixels
        :param eta: the eta parameter for the pseudo-voigt
        :param update_psf: how often the psf is updated
        :param nb_raar: number of relaxed averaged alternating reflections
         cycles, which the algorithm will use first. During RAAR and HIO, the
         support is updated regularly
        :param nb_hio: number of hybrid input/output cycles, which the
         algorithm will use after RAAR. During RAAR and HIO, the support is
         updated regularly
        :param nb_er: number of error reduction cycles, performed after HIO,
         without support update
        :param nb_ml: number of maximum-likelihood conjugate gradient to
         perform after ER
        :param nb_run: number of times to run the optimization
        :param nb_run_keep: number of best run results to keep, according to
         filter_criteria.
        :param filter_criteria: e.g. "LLK"
            criteria onto which the best solutions will be chosen
        :param live_plot: a live plot will be displayed every N cycle
        :param beta: the beta value for the HIO operator
        :param positivity: True or False
        :param zero_mask: if True, masked pixels (iobs<-1e19) are forced to
         zero, otherwise the calculated complex amplitude is kept with an
         optional scale factor.
        :param detwin: if set (command-line) or if detwin=True (parameters
         file), 10 cycles will be performed at 25% of the total number of
         RAAR or HIO cycles, with a support cut in half to bias towards one
         twin image
        :param pixel_size_detector: detector pixel size (meters)
        :param wavelength: experiment wavelength (meters)
        :param detector_distance: detector distance (meters)
        """

        self.Dataset.parent_folder = parent_folder
        self.Dataset.iobs = parent_folder + iobs
        if mask != "":
            self.Dataset.mask = parent_folder + mask
        else:
            self.Dataset.mask = ""
        if support != "":
            self.Dataset.support = parent_folder + support
        else:
            self.Dataset.support = ""
        if obj != "":
            self.Dataset.obj = parent_folder + obj
        else:
            self.Dataset.obj = ""
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
        self.Dataset.nb_raar = nb_raar
        self.Dataset.nb_hio = nb_hio
        self.Dataset.nb_er = nb_er
        self.Dataset.nb_ml = nb_ml
        self.Dataset.nb_run = nb_run
        self.Dataset.filter_criteria = filter_criteria
        self.Dataset.nb_run_keep = nb_run_keep
        self.Dataset.live_plot = live_plot
        # To do
        # self.Dataset.zero_mask = zero_mask
        # self.Dataset.crop_output = crop_output
        self.Dataset.positivity = positivity
        self.Dataset.beta = beta
        self.Dataset.detwin = detwin
        self.Dataset.rebin = rebin
        self.Dataset.verbose = verbose
        self.Dataset.pixel_size_detector = np.round(
            pixel_size_detector * 1e-6, 6)
        self.run_phase_retrieval = run_phase_retrieval
        self.run_pynx_tools = run_pynx_tools

        # Extract dict, list and tuple from strings
        self.Dataset.support_threshold = literal_eval(
            self.Dataset.support_threshold)
        self.Dataset.support_smooth_width = literal_eval(
            self.Dataset.support_smooth_width)
        self.Dataset.support_post_expand = literal_eval(
            self.Dataset.support_post_expand)
        self.Dataset.rebin = literal_eval(self.Dataset.rebin)

        if self.Dataset.live_plot == 0:
            self.Dataset.live_plot = False

        print("Scan n°", self.Dataset.scan)

        self.Dataset.energy = self._list_widgets_preprocessing.children[50].value
        self.Dataset.wavelength = 1.2399 * 1e-6 / self.Dataset.energy
        self.Dataset.sdd = self._list_widgets_preprocessing.children[49].value

        print("\tCXI input: Energy = %8.2f eV" % self.Dataset.energy)
        print(f"\tCXI input: Wavelength = {self.Dataset.wavelength*1e10} A")
        print("\tCXI input: detector distance = %8.2f m" % self.Dataset.sdd)
        print(
            f"\tCXI input: detector pixel size = {self.Dataset.pixel_size_detector} m")

        # PyNX arguments text files
        self.Dataset.pynx_parameter_gui_file = self.preprocessing_folder\
            + "/pynx_run_gui.txt"
        self.Dataset.pynx_parameter_cli_file = self.preprocessing_folder\
            + "/pynx_run.txt"

        # Phase retrieval
        if self.run_phase_retrieval and not self.run_pynx_tools:
            if self.run_phase_retrieval in ("batch", "local_script"):
                # Create /gui_run/ directory
                try:
                    os.mkdir(
                        f"{self.preprocessing_folder}/gui_run/")
                    print(
                        f"Created {self.preprocessing_folder}/gui_run/", end="\n\n")
                except (FileExistsError, PermissionError):
                    print(
                        f"{self.preprocessing_folder}/gui_run/ exists", end="\n\n")

                self.text_file = []
                self.Dataset.live_plot = False

                # Load files
                self.text_file.append("# Parameters\n")
                for file, parameter in [
                        (self.Dataset.iobs, "data"),
                        (self.Dataset.mask, "mask"),
                        (self.Dataset.obj, "object")
                ]:
                    if file != "":
                        self.text_file.append(f"{parameter} = \"{file}\"\n")

                if support != "":
                    self.text_file += [
                        f"support = \"{self.Dataset.support}\"\n",
                        '\n']
                # else no support, just don't write it

                # Clean threshold syntax
                support_threshold = support_threshold.replace("(", "")
                support_threshold = support_threshold.replace(")", "")
                support_threshold = support_threshold.replace(" ", "")

                # Other support parameters
                self.text_file += [
                    f'support_threshold= {support_threshold}\n',
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
                        self.text_file.append(
                            f"psf = \"{self.Dataset.psf_model},{self.Dataset.fwhm}\"\n")

                    if self.Dataset.psf_model == "pseudo-voigt":
                        self.text_file.append(
                            f"psf = \"{self.Dataset.psf_model},{self.Dataset.fwhm},{self.Dataset.eta}\"\n")
                # no PSF, just don't write anything

                # Filtering the reconstructions
                if self.Dataset.filter_criteria == "LLK":
                    nb_run_keep_LLK = self.Dataset.nb_run_keep
                    nb_run_keep_std = False

                elif self.Dataset.filter_criteria == "std":
                    nb_run_keep_LLK = self.Dataset.nb_run
                    nb_run_keep_std = self.Dataset.nb_run_keep

                elif self.Dataset.filter_criteria == "LLK_standard_deviation":
                    nb_run_keep_LLK = self.Dataset.nb_run_keep + \
                        (self.Dataset.nb_run - self.Dataset.nb_run_keep) // 2
                    nb_run_keep_std = self.Dataset.nb_run_keep

                # Clean rebin syntax
                rebin = rebin.replace("(", "")
                rebin = rebin.replace(")", "")
                rebin = rebin.replace(" ", "")

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
                    f'nb_run_keep = {nb_run_keep_LLK}\n',
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
                    f'rebin = {rebin}\n',
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

                gutil.hash_print(
                    f"Saved parameters in: {self.Dataset.pynx_parameter_gui_file}")

                if self.run_phase_retrieval == "batch":
                    # Runs modes directly and saves all data in a "gui_run"
                    # subdir, filter based on LLK
                    print(
                        f"\nRunning: $ {self.path_scripts}/run_slurm_job.sh --reconstruct gui --username {self.user_name} --path {self.preprocessing_folder} --filtering {nb_run_keep_std} --modes true")
                    print(
                        "\nSolution filtering and modes decomposition are automatically applied at the end of the batch job.")
                    os.system(
                        "{}/run_slurm_job.sh \
                        --reconstruct gui \
                        --username {} \
                        --path {} \
                        --filtering {} \
                        --modes true".format(
                            quote(self.path_scripts),
                            quote(self.user_name),
                            quote(self.preprocessing_folder),
                            quote(str(nb_run_keep_std)),
                        )
                    )

                    # Copy Pynx parameter file in folder
                    shutil.copyfile(self.Dataset.pynx_parameter_gui_file,
                                    f"{self.preprocessing_folder}/gui_run/pynx_run_gui.txt")

                elif self.run_phase_retrieval == "local_script":
                    try:
                        print(
                            f"\nRunning: $ {self.path_scripts}/pynx-id01cdi.py pynx_run_gui.txt 2>&1 | tee README_pynx_local_script.md &",
                            end="\n\n")
                        os.system(
                            "cd {}; {}/pynx-id01cdi.py pynx_run_gui.txt 2>&1 | tee README_pynx_local_script.md &".format(
                                quote(self.preprocessing_folder),
                                quote(self.path_scripts),
                            )
                        )
                    except KeyboardInterrupt:
                        print("Phase retrieval stopped by user ...")

            elif self.run_phase_retrieval == "operators":
                # Extract data
                gutil.hash_print(
                    "Log likelihood is updated every 50 iterations.")
                self.Dataset.calc_llk = 50  # TODO

                # Keep a list of the resulting scans
                self.reconstruction_file_list = []

                try:
                    # Run phase retrieval for nb_run
                    for i in range(self.Dataset.nb_run):
                        print(
                            "\n#########################################################################################"
                        )
                        print(f"Run {i}")

                        # Initialise the cdi operator
                        cdi = self.initialize_cdi_operator()

                        if i > 4:
                            print("Stopping liveplot to go faster\n")
                            self.Dataset.live_plot = False

                        # Change support threshold for supports update
                        if isinstance(self.Dataset.support_threshold, float):
                            self.Dataset.threshold_relative\
                                = self.Dataset.support_threshold
                        elif isinstance(self.Dataset.support_threshold, tuple):
                            self.Dataset.threshold_relative = np.random.uniform(
                                self.Dataset.support_threshold[0],
                                self.Dataset.support_threshold[1]
                            )
                        print(f"Threshold: {self.Dataset.threshold_relative}")

                        # Create support object
                        sup = SupportUpdate(
                            threshold_relative=self.Dataset.threshold_relative,
                            smooth_width=self.Dataset.support_smooth_width,
                            force_shrink=self.Dataset.support_only_shrink,
                            method='rms',
                            post_expand=self.Dataset.support_post_expand,
                        )

                        # Initialize the free pixels for LLK
                        # cdi = InitFreePixels() * cdi

                        # Initialize the support with autocorrelation, if no
                        # support given
                        if not self.Dataset.support:
                            sup_init = "autocorrelation"
                            if isinstance(self.Dataset.live_plot, int):
                                if i > 4:
                                    cdi = ScaleObj() * AutoCorrelationSupport(
                                        threshold=0.1,  # extra argument
                                        verbose=True) * cdi

                                else:
                                    cdi = ShowCDI() * ScaleObj() \
                                        * AutoCorrelationSupport(
                                        threshold=0.1,  # extra argument
                                        verbose=True) * cdi

                            else:
                                cdi = ScaleObj() * AutoCorrelationSupport(
                                    threshold=0.1,  # extra argument
                                    verbose=True) * cdi
                        else:
                            sup_init = "support"

                        # Begin with HIO cycles without PSF and with support
                        # updates
                        try:
                            # update_psf = 0 probably enough but not sure
                            if self.Dataset.psf:
                                if self.Dataset.support_update_period == 0:
                                    cdi = HIO(
                                        beta=self.Dataset.beta,
                                        calc_llk=self.Dataset.calc_llk,
                                        show_cdi=self.Dataset.live_plot
                                    ) ** self.Dataset.nb_hio * cdi
                                    cdi = RAAR(
                                        beta=self.Dataset.beta,
                                        calc_llk=self.Dataset.calc_llk,
                                        show_cdi=self.Dataset.live_plot
                                    ) ** (self.Dataset.nb_raar // 2) * cdi

                                    # PSF is introduced at 66% of HIO and RAAR
                                    if psf_model != "pseudo-voigt":
                                        cdi = InitPSF(
                                            model=self.Dataset.psf_model,
                                            fwhm=self.Dataset.fwhm,
                                        ) * cdi

                                    elif psf_model == "pseudo-voigt":
                                        cdi = InitPSF(
                                            model=self.Dataset.psf_model,
                                            fwhm=self.Dataset.fwhm,
                                            eta=self.Dataset.eta,
                                        ) * cdi

                                    cdi = RAAR(
                                        beta=self.Dataset.beta,
                                        calc_llk=self.Dataset.calc_llk,
                                        show_cdi=self.Dataset.live_plot,
                                        update_psf=self.Dataset.update_psf
                                    ) ** (self.Dataset.nb_raar // 2) * cdi
                                    cdi = ER(
                                        calc_llk=self.Dataset.calc_llk,
                                        show_cdi=self.Dataset.live_plot,
                                        update_psf=self.Dataset.update_psf
                                    ) ** self.Dataset.nb_er * cdi

                                else:
                                    hio_power = self.Dataset.nb_hio \
                                        // self.Dataset.support_update_period
                                    raar_power = (
                                        self.Dataset.nb_raar // 2) \
                                        // self.Dataset.support_update_period
                                    er_power = self.Dataset.nb_er \
                                        // self.Dataset.support_update_period

                                    cdi = (sup * HIO(
                                        beta=self.Dataset.beta,
                                        calc_llk=self.Dataset.calc_llk,
                                        show_cdi=self.Dataset.live_plot
                                    )**self.Dataset.support_update_period
                                    ) ** hio_power * cdi
                                    cdi = (sup * RAAR(
                                        beta=self.Dataset.beta,
                                        calc_llk=self.Dataset.calc_llk,
                                        show_cdi=self.Dataset.live_plot
                                    )**self.Dataset.support_update_period
                                    ) ** raar_power * cdi

                                    # PSF is introduced at 66% of HIO and RAAR
                                    # so from cycle n°924
                                    if psf_model != "pseudo-voigt":
                                        cdi = InitPSF(
                                            model=self.Dataset.psf_model,
                                            fwhm=self.Dataset.fwhm,
                                        ) * cdi

                                    elif psf_model == "pseudo-voigt":
                                        cdi = InitPSF(
                                            model=self.Dataset.psf_model,
                                            fwhm=self.Dataset.fwhm,
                                            eta=self.Dataset.eta,
                                        ) * cdi

                                    cdi = (sup * RAAR(
                                        beta=self.Dataset.beta,
                                        calc_llk=self.Dataset.calc_llk,
                                        show_cdi=self.Dataset.live_plot,
                                        update_psf=self.Dataset.update_psf
                                    )**self.Dataset.support_update_period
                                    ) ** raar_power * cdi
                                    cdi = (sup * ER(
                                        calc_llk=self.Dataset.calc_llk,
                                        show_cdi=self.Dataset.live_plot,
                                        update_psf=self.Dataset.update_psf
                                    )**self.Dataset.support_update_period
                                    ) ** er_power * cdi

                            if not self.Dataset.psf:
                                if self.Dataset.support_update_period == 0:

                                    cdi = HIO(
                                        beta=self.Dataset.beta,
                                        calc_llk=self.Dataset.calc_llk,
                                        show_cdi=self.Dataset.live_plot
                                    ) ** self.Dataset.nb_hio * cdi
                                    cdi = RAAR(
                                        beta=self.Dataset.beta,
                                        calc_llk=self.Dataset.calc_llk,
                                        show_cdi=self.Dataset.live_plot
                                    ) ** self.Dataset.nb_raar * cdi
                                    cdi = ER(
                                        calc_llk=self.Dataset.calc_llk,
                                        show_cdi=self.Dataset.live_plot
                                    ) ** self.Dataset.nb_er * cdi

                                else:
                                    hio_power = self.Dataset.nb_hio \
                                        // self.Dataset.support_update_period
                                    raar_power = self.Dataset.nb_raar \
                                        // self.Dataset.support_update_period
                                    er_power = self.Dataset.nb_er \
                                        // self.Dataset.support_update_period

                                    cdi = (sup * HIO(
                                        beta=self.Dataset.beta,
                                        calc_llk=self.Dataset.calc_llk,
                                        show_cdi=self.Dataset.live_plot
                                    )**self.Dataset.support_update_period
                                    ) ** hio_power * cdi
                                    cdi = (sup * RAAR(
                                        beta=self.Dataset.beta,
                                        calc_llk=self.Dataset.calc_llk,
                                        show_cdi=self.Dataset.live_plot
                                    )**self.Dataset.support_update_period
                                    ) ** raar_power * cdi
                                    cdi = (sup * ER(
                                        calc_llk=self.Dataset.calc_llk,
                                        show_cdi=self.Dataset.live_plot
                                    )**self.Dataset.support_update_period
                                    ) ** er_power * cdi

                            fn = "{}/result_scan_{}_run_{}_LLK_{:.4}_support_threshold_{:.4}_shape_{}_{}_{}_{}.cxi".format(
                                self.Dataset.parent_folder,
                                self.Dataset.scan,
                                i,
                                cdi.get_llk()[0],
                                self.Dataset.threshold_relative,
                                cdi.iobs.shape[0],
                                cdi.iobs.shape[1],
                                cdi.iobs.shape[2],
                                sup_init,
                            )

                            self.reconstruction_file_list.append(fn)
                            cdi.save_obj_cxi(fn)
                            print(f"\nSaved as {fn}.")
                            print(
                                "#########################################################################################\n"
                            )

                        except SupportTooLarge:
                            print(
                                "Threshold value probably too low, support too large too continue")

                    # If filter, filter data
                    if self.Dataset.filter_criteria:
                        gutil.filter_reconstructions(
                            self.Dataset.parent_folder,
                            self.Dataset.nb_run,
                            self.Dataset.nb_run_keep,
                            self.Dataset.filter_criteria
                        )

                except KeyboardInterrupt:
                    clear_output(True)
                    gutil.hash_print("Phase retrieval stopped by user ...")

        # Modes decomposition and solution filtering
        if self.run_pynx_tools and not self.run_phase_retrieval:
            if self.run_pynx_tools == "modes":
                self.run_modes_decomposition(self.Dataset.parent_folder)

            elif self.run_pynx_tools == "filter":
                gutil.filter_reconstructions(
                    folder=self.Dataset.parent_folder,
                    nb_run=None,
                    nb_run_keep=self.Dataset.nb_run_keep,
                    filter_criteria=self.Dataset.filter_criteria
                )

        # Clean output
        if not self.run_phase_retrieval and not self.run_pynx_tools:
            gutil.hash_print("Cleared output.")
            clear_output(True)

            # Refresh folders
            self.sub_directories_handler(change=self.Dataset.scan_folder)

            # PyNX folder, refresh values
            # self._list_widgets_phase_retrieval.children[1].value\
            #     = self.preprocessing_folder
            # self.pynx_folder_handler(change=self.preprocessing_folder)

            # Plot folder, refresh values
            self.tab_data.children[1].value = self.preprocessing_folder
            self.plot_folder_handler(change=self.preprocessing_folder)

            # Strain folder, refresh values
            self._list_widgets_strain.children[-4].value\
                = self.preprocessing_folder
            self.strain_folder_handler(change=self.preprocessing_folder)

    def run_modes_decomposition(self, folder,):
        """
        Decomposes several phase retrieval solutions into modes, saves only
        the first mode to save space.

        :param folder: path to folder in which are stored
         the .cxi files, all files corresponding to
         *LLK* pattern are loaded
        """
        try:
            print(
                "\n#########################################################################################"
            )
            print(
                f"Using {self.path_scripts}/pynx-cdi-analysis.py\n"
                f"Using {folder}/*LLK* files.\n"
                "Running: $ pynx-cdi-analysis.py *LLK* modes=1\n"
                f"Output in {folder}/modes_gui.h5")
            print(
                "#########################################################################################\n"
            )
            os.system(
                "{}/pynx-cdi-analysis.py {}/*LLK* modes=1 modes_output={}/modes_gui.h5".format(
                    quote(self.path_scripts),
                    quote(folder),
                    quote(folder),
                )
            )
        except KeyboardInterrupt:
            gutil.hash_print("Decomposition into modes stopped by user...")

        finally:
            # Refresh folders
            self.sub_directories_handler(change=self.Dataset.scan_folder)

            # PyNX folder, refresh values
            self._list_widgets_phase_retrieval.children[1].value\
                = self.preprocessing_folder
            self.pynx_folder_handler(change=self.preprocessing_folder)

            self.tab_data.children[1].value = self.preprocessing_folder
            self.plot_folder_handler(
                change=self.preprocessing_folder)

    def save_as_cxi(self, cdi_operator, path_to_cxi):
        """
        We need to create a dictionnary with the parameters to save in the
        cxi file.

        :param cdi_operator: cdi object
         created with PyNX
        :param path_to_cxi: path to future
         cxi data
         Below are parameters that are saved in the cxi file
        :param filename: the file name to save the data to
        :param iobs: the observed intensity
        :param wavelength: the wavelength of the experiment (in meters)
        :param detector_distance: the detector distance (in meters)
        :param pixel_size_detector: the pixel size of the detector (in meters)
        :param mask: the mask indicating valid (=0) and bad pixels (>0)
        :param sample_name: optional, the sample name
        :param experiment_id: the string identifying the experiment, e.g.:
         'HC1234: Siemens star calibration tests'
        :param instrument: the string identifying the instrument, e.g.:
         'ESRF id10'
        :param iobs_is_fft_shifted: if true, input iobs (and mask if any)
         have their origin in (0,0[,0]) and will be shifted back to
         centered-versions before being saved.
        :param process_parameters: a dictionary of parameters which will
         be saved as a NXcollection
        :return: Nothing. a CXI file is created
        """
        self.params = params
        self.params["data"] = self.Dataset.iobs
        self.params["wavelength"] = self.Dataset.wavelength
        self.params["detector_distance"] = self.Dataset.sdd
        self.params["pixel_size_detector"] = self.Dataset.pixel_size_detector
        self.params["wavelength"] = self.Dataset.wavelength
        self.params["verbose"] = self.Dataset.verbose
        self.params["live_plot"] = self.Dataset.live_plot
        # self.params["gpu"] = self.Dataset.gpu
        self.params["auto_center_resize"] = self.Dataset.auto_center_resize
        # self.params["roi_user"] = self.Dataset.roi_user
        # self.params["roi_final"] = self.Dataset.roi_final
        self.params["nb_run"] = self.Dataset.nb_run
        self.params["max_size"] = self.Dataset.max_size
        # self.params["data2cxi"] = self.Dataset.data2cxi
        self.params["output_format"] = "cxi"
        self.params["mask"] = self.Dataset.mask
        self.params["support"] = self.Dataset.support
        # self.params["support_autocorrelation_threshold"]\
        # = self.Dataset.support_autocorrelation_threshold
        self.params["support_only_shrink"] = self.Dataset.support_only_shrink
        self.params["object"] = self.Dataset.obj
        self.params["support_update_period"] = self.Dataset.support_update_period
        self.params["support_smooth_width_begin"] = self.Dataset.support_smooth_width[0]
        self.params["support_smooth_width_end"] = self.Dataset.support_smooth_width[1]
        # self.params["support_smooth_width_relax_n"] = \
        # self.Dataset.support_smooth_width_relax_n
        # self.params["support_size"] = self.Dataset.support_size
        self.params["support_threshold"] = self.Dataset.support_threshold
        self.params["positivity"] = self.Dataset.positivity
        self.params["beta"] = self.Dataset.beta
        self.params["crop_output"] = 0
        self.params["rebin"] = self.Dataset.rebin
        # self.params["support_update_border_n"] \
        # = self.Dataset.support_update_border_n
        # self.params["support_threshold_method"] \
        # = self.Dataset.support_threshold_method
        self.params["support_post_expand"] = self.Dataset.support_post_expand
        self.params["psf"] = self.Dataset.psf
        # self.params["note"] = self.Dataset.note
        try:
            self.params["instrument"] = self.Dataset.beamline
        except AttributeError:
            self.params["instrument"] = None
        self.params["sample_name"] = self.Dataset.sample_name
        # self.params["fig_num"] = self.Dataset.fig_num
        # self.params["algorithm"] = self.Dataset.algorithm
        self.params["zero_mask"] = "auto"
        self.params["nb_run_keep"] = self.Dataset.nb_run_keep
        # self.params["save"] = self.Dataset.save
        # self.params["gps_inertia"] = self.Dataset.gps_inertia
        # self.params["gps_t"] = self.Dataset.gps_t
        # self.params["gps_s"] = self.Dataset.gps_s
        # self.params["gps_sigma_f"] = self.Dataset.gps_sigma_f
        # self.params["gps_sigma_o"] = self.Dataset.gps_sigma_o
        # self.params["iobs_saturation"] = self.Dataset.iobs_saturation
        # self.params["free_pixel_mask"] = self.Dataset.free_pixel_mask
        # self.params["support_formula"] = self.Dataset.support_formula
        # self.params["mpi"] = "run"
        # self.params["mask_interp"] = self.Dataset.mask_interp
        # self.params["confidence_interval_factor_mask_min"] \
        # = self.Dataset.confidence_interval_factor_mask_min
        # self.params["confidence_interval_factor_mask_max"] \
        # = self.Dataset.confidence_interval_factor_mask_max
        # self.params["save_plot"] = self.Dataset.save_plot
        # self.params["support_fraction_min"] \
        # = self.Dataset.support_fraction_min
        # self.params["support_fraction_max"] \
        # = self.Dataset.support_fraction_max
        # self.params["support_threshold_auto_tune_factor"] \
        # = self.Dataset.support_threshold_auto_tune_factor
        # self.params["nb_run_keep_max_obj2_out"] \
        # = self.Dataset.nb_run_keep_max_obj2_out
        # self.params["flatfield"] = self.Dataset.flatfield
        # self.params["psf_filter"] = self.Dataset.psf_filter
        self.params["detwin"] = self.Dataset.detwin
        self.params["nb_raar"] = self.Dataset.nb_raar
        self.params["nb_hio"] = self.Dataset.nb_hio
        self.params["nb_er"] = self.Dataset.nb_er
        self.params["nb_ml"] = self.Dataset.nb_ml
        try:
            self.params["specfile"] = self.Dataset.specfile_name
        except AttributeError:
            pass
        # self.params["imgcounter"] = self.Dataset.imgcounter
        # self.params["imgname"] = self.Dataset.imgname
        self.params["scan"] = self.Dataset.scan

        print("\nSaving phase retrieval parameters selected in the PyNX tab in the cxi file ...")
        cdi_operator.save_data_cxi(
            filename=path_to_cxi,
            process_parameters=self.params,
        )

    # Postprocessing

    def initialize_postprocessing(
        self,
        unused_label_averaging,
        sort_method,
        correlation_threshold,
        unused_label_FFT,
        phasing_binning,
        original_size,
        preprocessing_binning,
        output_size,
        keep_size,
        fix_voxel,
        unused_label_disp_strain,
        data_frame,
        save_frame,
        ref_axis_q,
        isosurface_strain,
        strain_method,
        phase_offset,
        phase_offset_origin,
        offset_method,
        centering_method,
        unused_label_refraction,
        correct_refraction,
        optical_path_method,
        dispersion,
        absorption,
        threshold_unwrap_refraction,
        unused_label_options,
        simulation,
        invert_phase,
        flip_reconstruction,
        phase_ramp_removal,
        threshold_gradient,
        save_raw,
        save_support,
        save,
        debug,
        roll_modes,
        unused_label_data_vis,
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
        unused_label_average,
        averaging_space,
        threshold_avg,
        unused_label_apodize,
        apodize,
        apodization_window,
        half_width_avg_phase,
        apodization_mu,
        apodization_sigma,
        apodization_alpha,
        unused_label_strain,
        strain_folder,
        reconstruction_file,
        run_strain,
    ):
        """
        Loading argument from strain tab widgets but also values of
        parameters used in preprocessing that are common Runs postprocessing
        script from bcdi package to extract the strain from the reconstructed
        phase. Also plots images depending on the given isosurface.

        Parameters used when averaging several reconstruction:

        :param sort_method: e.g. "variance/mean"
         'mean_amplitude' or 'variance' or 'variance/mean' or 'volume',
         metric for averaging
        :param correlation_threshold: e.g. 0.90
         minimum correlation between two arrays to average them

        Parameters related to centering:

        :param centering_method: e.g. "max_com"
        'com' (center of mass), 'max', 'max_com' (max then com), 'do_nothing'
        :param roll_modes: e.g. [0, 0, 0]
        correct a roll of few pixels after the decomposition into modes in PyNX
        axis=(0, 1, 2)

        Prameters relative to the FFT window and voxel sizes:

        :param original_size: e.g. [150, 256, 500]
         size of the FFT array before binning. It will be modified to take
         into account binning during phasing automatically. Leave it to None
         if the shape did not change.
        :param phasing_binning: e.g. [1, 1, 1]
         binning factor applied during phase retrieval
        :param preprocessing_binning: e.g. [1, 2, 2]
         binning factors in each dimension used in preprocessing (not phase
         retrieval)
        :param output_size: e.g. [100, 100, 100]
         (z, y, x) Fix the size of the output array, leave None to use the
         object size
        :param keep_size: e.g. False
         True to keep the initial array size for orthogonalization (slower),
         it will be cropped otherwise
        :param fix_voxel: e.g. 10
         voxel size in nm for the interpolation during the geometrical
         transformation. If a single value is provided, the voxel size will be
         identical in all 3 directions. Set it to None to use the default voxel
         size (calculated from q values, it will be different in each
         dimension).

        Parameters related to the strain calculation:

        :param data_frame: e.g. "detector"
         in which frame is defined the input data, available options:

         - 'crystal' if the data was interpolated into the crystal frame using
           xrayutilities or (transformation matrix + align_q=True)
         - 'laboratory' if the data was interpolated into the laboratory frame
           using the transformation matrix (align_q: False)
         - 'detector' if the data is still in the detector frame

        :param ref_axis_q: e.g. "y"
         axis along which q will be aligned (data_frame= 'detector' or
         'laboratory') or is already aligned (data_frame='crystal')
        :param save_frame: e.g. "laboratory"
         in which frame should be saved the data, available options:

         - 'crystal' to save the data with q aligned along ref_axis_q
         - 'laboratory' to save the data in the laboratory frame (experimental
           geometry)
         - 'lab_flat_sample' to save the data in the laboratory frame, with
           all sample angles rotated back to 0. The rotations for 'laboratory'
           and 'lab_flat_sample' are realized after the strain calculation
           (which is always done in the crystal frame along ref_axis_q)

        :param isosurface_strain: e.g. 0.2
         threshold use for removing the outer layer (the strain is undefined
         at the exact surface voxel)
        :param strain_method: e.g. "default"
         how to calculate the strain, available options:

         - 'default': use the single value calculated from the gradient of
           the phase
         - 'defect': it will offset the phase in a loop and keep the smallest
           magnitude value for the strain.
           See: F. Hofmann et al. PhysRevMaterials 4, 013801 (2020)

        Parameters related to the refraction correction:

        :param correct_refraction: e.g. True
         True for correcting the phase shift due to refraction
        :param optical_path_method: e.g. "threshold"
         'threshold' or 'defect', if 'threshold' it uses isosurface_strain to
         define the support  for the optical path calculation, if 'defect'
         (holes) it tries to remove only outer layers even if the amplitude is
         lower than isosurface_strain inside the crystal
        :param dispersion: e.g. 5.0328e-05
         delta value used for refraction correction, for Pt:
         3.0761E-05 @ 10300eV, 5.0328E-05 @ 8170eV, 3.2880E-05 @ 9994eV,
         4.1184E-05 @ 8994eV, 5.2647E-05 @ 7994eV, 4.6353E-05 @ 8500eV
         Ge 1.4718E-05 @ 8keV
        :param absorption: e.g. 4.1969e-06
         beta value, for Pt:  2.0982E-06 @ 10300eV, 4.8341E-06 @ 8170eV,
         2.3486E-06 @ 9994eV, 3.4298E-06 @ 8994eV, 5.2245E-06 @ 7994eV,
         4.1969E-06 @ 8500eV
        :param threshold_unwrap_refraction: e.g. 0.05
         threshold used to calculate the optical path. The threshold for
         refraction correction should be low, to correct for an object larger
         than the real one, otherwise it messes up the phase

        Parameters related to the phase:

        :param simulation: e.g. False
         True if it is a simulation, the parameter invert_phase will be set
         to 0 (see below)
        :param invert_phase: e.g. True
        True for the displacement to have the right sign (FFT convention),
        it is False only for simulations
        :param flip_reconstruction: e.g. True
         True if you want to get the conjugate object
        :param phase_ramp_removal: e.g. "gradient"
         'gradient' or 'upsampling', 'gradient' is much faster
        :param threshold_gradient: e.g. 1.0
         upper threshold of the gradient of the phase, use for ramp removal
        :param phase_offset: e.g. 0
         manual offset to add to the phase, should be 0 in most cases
        :param phase_offset_origin: e.g. [12, 32, 65]
         the phase at this voxel will be set to phase_offset, leave None to
         use the default position computed using offset_method (see below)
        :param offset_method: e.g. "mean"
         'com' (center of mass) or 'mean', method for determining the phase
         offset origin

        Parameters related to data visualization:

        :param debug: e.g. False
         True to show all plots for debugging
        :param align_axis: e.g. False
         True to rotate the crystal to align axis_to_align along ref_axis for
         visualization. This is done after the calculation of the strain and
         has no effect on it.
        :param ref_axis: e.g. "y"
         it will align axis_to_align to that axis if align_axis is True
        :param axis_to_align: e.g. [-0.01166, 0.9573, -0.2887]
         axis to align with ref_axis in the order x y z (axis 2, axis 1,
         axis 0)
        :param strain_range: e.g. 0.001
         range of the colorbar for strain plots
        :param phase_range: e.g. 0.4
         range of the colorbar for phase plots
        :param grey_background: e.g. True
         True to set the background to grey in phase and strain plots
        :param tick_spacing: e.g. 50
         spacing between axis ticks in plots, in nm
        :param tick_direction: e.g. "inout"
         direction of the ticks in plots: 'out', 'in', 'inout'
        :param tick_length: e.g. 3
         length of the ticks in plots
        :param tick_width: e.g. 1
         width of the ticks in plots

        Parameters for averaging several reconstructed objects:

        :param averaging_space: e.g. "reciprocal_space"
         in which space to average, 'direct_space' or 'reciprocal_space'
        :param threshold_avg: e.g. 0.90
         minimum correlation between arrays for averaging

        Parameters for phase averaging or apodization:

        :param half_width_avg_phase: e.g. 0
         (width-1)/2 of the averaging window for the phase, 0 means no phase
         averaging
        :param apodize: e.g. False
         True to multiply the diffraction pattern by a filtering window
        :param apodization_window: e.g. "blackman"
         filtering window, multivariate 'normal' or 'tukey' or 'blackman'
        :param apodization_mu: e.g. [0.0, 0.0, 0.0]
         mu of the gaussian window
        :param apodization_sigma: e.g. [0.30, 0.30, 0.30]
         sigma of the gaussian window
        :param apodization_alpha: e.g. [1.0, 1.0, 1.0]
         shape parameter of the tukey window

        Parameters related to saving:

        :param save_rawdata: e.g. False
         True to save the amp-phase.vti before orthogonalization
        :param save_support: e.g. False
         True to save the non-orthogonal support for later phase retrieval
        :param save: e.g. True
         True to save amp.npz, phase.npz, strain.npz and vtk files
        """

        # Save parameter values
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
        # parameters related to the refraction correction
        self.Dataset.correct_refraction = correct_refraction
        self.Dataset.optical_path_method = optical_path_method
        self.Dataset.dispersion = dispersion
        self.Dataset.absorption = absorption
        self.Dataset.threshold_unwrap_refraction\
            = threshold_unwrap_refraction
        # options #
        self.Dataset.simulation = simulation
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
        self.Dataset.averaging_space = averaging_space
        self.Dataset.threshold_avg = threshold_avg
        # setup for phase averaging or apodization
        self.Dataset.half_width_avg_phase = half_width_avg_phase
        self.Dataset.apodize = apodize
        self.Dataset.apodization_window = apodization_window
        self.Dataset.apodization_mu = apodization_mu
        self.Dataset.apodization_sigma = apodization_sigma
        self.Dataset.apodization_alpha = apodization_alpha
        self.Dataset.reconstruction_file = strain_folder + reconstruction_file

        if run_strain:
            # Save directory
            save_dir = f"{self.postprocessing_folder}/result_{self.Dataset.save_frame}/"

            # Disable all widgets until the end of the program
            for w in self._list_widgets_strain.children[:-1]:
                w.disabled = True

            for w in self._list_widgets_preprocessing.children[:-2]:
                w.disabled = True

            # Extract dict, list and tuple from strings
            list_parameters = [
                "original_size", "output_size", "axis_to_align",
                "apodization_mu", "apodization_sigma", "apodization_alpha"]

            tuple_parameters = [
                "phasing_binning", "preprocessing_binning",
                "phase_offset_origin", "roll_modes"]

            try:
                for p in list_parameters:
                    if getattr(self.Dataset, p) == "":
                        setattr(self.Dataset, p, [])
                    else:
                        setattr(self.Dataset, p, literal_eval(
                            getattr(self.Dataset, p)))
            except ValueError:
                gutil.hash_print(f"Wrong list syntax for {p}")

            try:
                for p in tuple_parameters:
                    if getattr(self.Dataset, p) == "":
                        setattr(self.Dataset, p, ())
                    else:
                        setattr(self.Dataset, p, literal_eval(
                            getattr(self.Dataset, p)))
            except ValueError:
                gutil.hash_print(f"Wrong tuple syntax for {p}")

            # Empty parameters are set to None (bcdi syntax)
            if self.Dataset.output_size == []:
                self.Dataset.output_size = None

            if self.Dataset.fix_voxel == 0:
                self.Dataset.fix_voxel = None

            if self.Dataset.phase_offset_origin == ():
                self.Dataset.phase_offset_origin = (None)

            # Check beamline for save folder
            try:
                # Change data_dir and root folder depending on beamline
                if self.Dataset.beamline == "SIXS_2019":
                    data_dir = self.Dataset.data_dir

                elif self.Dataset.beamline == "P10":
                    data_dir = f"{self.Dataset.data_dir}{self.Dataset.sample_name}_{self.Dataset.scan:05d}/e4m/"

                elif self.Dataset.beamline in ("ID01", "ID01BLISS"):
                    data_dir = self.Dataset.data_dir

            except AttributeError:
                for w in self._list_widgets_strain.children[:-1]:
                    w.disabled = False

                for w in self._list_widgets_preprocessing.children[:-2]:
                    w.disabled = False

                print("You need to initialize all the parameters with the \
                    preprocess tab first, some parameters are used here such \
                    as the energy, detector distance, ...""")
                return

            try:
                gutil.create_yaml_file(
                    fname=f"{self.postprocessing_folder}/config_postprocessing.yml",
                    scan=self.Dataset.scan,
                    root_folder=self.Dataset.root_folder,
                    save_dir=save_dir,
                    data_dir=data_dir,
                    sample_name=self.Dataset.sample_name,
                    comment=self.Dataset.comment,
                    reconstruction_file=self.Dataset.reconstruction_file,
                    backend=self.matplotlib_backend,
                    # parameters used when averaging several reconstruction #
                    sort_method=self.Dataset.sort_method,
                    averaging_space=self.Dataset.averaging_space,
                    correlation_threshold=self.Dataset.correlation_threshold,
                    # parameters related to centering #
                    centering_method=self.Dataset.centering_method,
                    roll_modes=self.Dataset.roll_modes,
                    # parameters relative to the FFT window and voxel sizes #
                    original_size=self.Dataset.original_size,
                    phasing_binning=self.Dataset.phasing_binning,
                    preprocessing_binning=self.Dataset.preprocessing_binning,
                    output_size=self.Dataset.output_size,
                    keep_size=self.Dataset.keep_size,
                    fix_voxel=self.Dataset.fix_voxel,
                    # parameters related to the strain calculation #
                    data_frame=self.Dataset.data_frame,
                    ref_axis_q=self.Dataset.ref_axis_q,
                    save_frame=self.Dataset.save_frame,
                    isosurface_strain=self.Dataset.isosurface_strain,
                    strain_method=self.Dataset.strain_method,
                    # define beamline related parameters #
                    beamline=self.Dataset.beamline,
                    is_series=self.Dataset.is_series,
                    actuators=self.Dataset.actuators,
                    # setup for custom scans #
                    custom_scan=self.Dataset.custom_scan,
                    custom_images=self.Dataset.custom_images,
                    custom_monitor=self.Dataset.custom_monitor,
                    rocking_angle=self.Dataset.rocking_angle,
                    sdd=self.Dataset.sdd,
                    energy=self.Dataset.energy,
                    beam_direction=self.Dataset.beam_direction,
                    sample_offsets=self.Dataset.sample_offsets,
                    tilt_angle=self.Dataset.tilt_angle,
                    direct_beam=self.Dataset.direct_beam,
                    dirbeam_detector_angles=self.Dataset.dirbeam_detector_angles,
                    bragg_peak=self.Dataset.bragg_peak,
                    outofplane_angle=self.Dataset.outofplane_angle,
                    inplane_angle=self.Dataset.inplane_angle,
                    specfile_name=self.Dataset.specfile_name,
                    # detector related parameters #
                    detector=self.Dataset.detector,
                    roi_detector=self.Dataset.roi_detector,
                    template_imagefile=self.Dataset.template_imagefile,
                    # parameters related to the refraction correction #
                    correct_refraction=self.Dataset.correct_refraction,
                    optical_path_method=self.Dataset.optical_path_method,
                    dispersion=self.Dataset.dispersion,
                    absorption=self.Dataset.absorption,
                    threshold_unwrap_refraction=self.Dataset.threshold_unwrap_refraction,
                    # parameters related to the phase #
                    simulation=self.Dataset.simulation,
                    invert_phase=self.Dataset.invert_phase,
                    flip_reconstruction=self.Dataset.flip_reconstruction,
                    phase_ramp_removal=self.Dataset.phase_ramp_removal,
                    threshold_gradient=self.Dataset.threshold_gradient,
                    phase_offset=self.Dataset.phase_offset,
                    phase_offset_origin=self.Dataset.phase_offset_origin,
                    offset_method=self.Dataset.offset_method,
                    # parameters related to data visualization #
                    debug=self.Dataset.debug,
                    align_axis=self.Dataset.align_axis,
                    ref_axis=self.Dataset.ref_axis,
                    axis_to_align=self.Dataset.axis_to_align,
                    strain_range=self.Dataset.strain_range,
                    phase_range=self.Dataset.phase_range,
                    grey_background=self.Dataset.grey_background,
                    tick_spacing=self.Dataset.tick_spacing,
                    tick_direction=self.Dataset.tick_direction,
                    tick_length=self.Dataset.tick_length,
                    tick_width=self.Dataset.tick_width,
                    # parameters for temperature estimation #
                    # get_temperature=self.Dataset.get_temperature,
                    # reflection=self.Dataset.reflection,
                    # reference_spacing=self.Dataset.reference_spacing,
                    # reference_temperature=self.Dataset.reference_temperature,
                    # parameters for phase averaging or apodization #
                    half_width_avg_phase=self.Dataset.half_width_avg_phase,
                    apodize=self.Dataset.apodize,
                    apodization_window=self.Dataset.apodization_window,
                    apodization_mu=self.Dataset.apodization_mu,
                    apodization_sigma=self.Dataset.apodization_sigma,
                    apodization_alpha=self.Dataset.apodization_alpha,
                    # parameters related to saving #
                    save_rawdata=self.Dataset.save_rawdata,
                    save_support=self.Dataset.save_support,
                    save=self.Dataset.save,
                )
                # Run bcdi_postprocessing
                print(
                    "\n#########################################################################################\n"
                )
                print(f"Running: $ {self.path_scripts}/bcdi_strain.py")
                print(
                    f"Config file: {self.postprocessing_folder}/config_postprocessing.yml")
                print(
                    "\n#########################################################################################\n"
                )

                # Construct the argument parser
                ap = argparse.ArgumentParser()

                # Load the config file
                config_file = self.postprocessing_folder + "/config_postprocessing.yml"
                parser = ConfigParser(config_file)
                args = parser.load_arguments()
                args["time"] = f"{datetime.now()}"

                # Run function
                run_postprocessing(prm=args)
                gutil.hash_print("End of script")

                # Get data from saved file
                phase_fieldname = "disp" if self.Dataset.invert_phase else "phase"

                files = sorted(
                    glob.glob(
                        f"{self.postprocessing_folder}/**/S{self.Dataset.scan}_amp{phase_fieldname}strain*{self.Dataset.comment}.h5",
                        recursive=True),
                    key=os.path.getmtime)
                self.Dataset.strain_output_file = files[0]

                print(
                    "\n#########################################################################################"
                )
                print(f"Result file used to extract results saved in the .cxi file:")
                print(f"{self.Dataset.strain_output_file}")
                print("\nMake sure it is the latest one.")
                print(
                    "##########################################################################################\n"
                )

            except KeyboardInterrupt:
                gutil.hash_print("Strain analysis stopped by user ...")

            finally:
                # At the end of the function
                self._list_widgets_strain.children[-2].disabled = False

                # Refresh folders
                self.sub_directories_handler(change=self.Dataset.scan_folder)

                # PyNX folder, refresh values
                self._list_widgets_phase_retrieval.children[1].value\
                    = self.preprocessing_folder
                self.pynx_folder_handler(change=self.preprocessing_folder)

                self.tab_data.children[1].value = self.preprocessing_folder
                self.plot_folder_handler(
                    change=self.preprocessing_folder)

        if not run_strain:
            plt.close()
            for w in self._list_widgets_strain.children[:-1]:
                w.disabled = False

            for w in self._list_widgets_preprocessing.children[:-2]:
                w.disabled = False

            # Refresh folders
            self.sub_directories_handler(change=self.Dataset.scan_folder)

            # PyNX folder, refresh values
            self._list_widgets_phase_retrieval.children[1].value\
                = self.preprocessing_folder
            self.pynx_folder_handler(change=self.preprocessing_folder)

            # Plot folder, refresh values
            self.tab_data.children[1].value = self.preprocessing_folder
            self.plot_folder_handler(change=self.preprocessing_folder)

            gutil.hash_print("Cleared window.")
            clear_output(True)

    def init_facet_analysis(
        self,
        unused_label_facet,
        parent_folder,
        vtk_file,
        load_data,
    ):
        """
        Allows one to:

            Load a vtk file (previously created in paraview via the FacetAnalyser
            plugin)
            Realign the particle by assigning a vector to 2 of its facets
            Extract information from each facet

        :param parent_folder: all .vtk files in the parent_folder subsirectories
         will be shown in the dropdown list.
        :param vtk_file: path to vtk file
        :param load_data: True to load vtk file dataframe
        """
        if load_data:
            # Disable text widget to avoid bugs
            self.tab_facet.children[1].disabled = True
            try:
                self.Dataset.facet_filename = vtk_file
            except AttributeError:
                pass

            try:
                self.Facets = facet_analysis.Facets(
                    filename=os.path.basename(vtk_file),
                    pathdir=vtk_file.replace(os.path.basename(vtk_file), ""))
                print(
                    "Facets object saved as self.Facets, call help(self.Facets) \
                    for more details.")

                # Button to rotate data
                button_rotate = Button(
                    description="Work on facet data",
                    continuous_update=False,
                    button_style='',
                    layout=Layout(width='40%'),
                    style={'description_width': 'initial'},
                    icon='fast-forward')

                # Button to view data
                button_view_particle = Button(
                    description="View particle",
                    continuous_update=False,
                    button_style='',
                    layout=Layout(width='40%'),
                    style={'description_width': 'initial'},
                    icon='fast-forward')

                # Common button as widget
                buttons_facets = widgets.HBox(
                    [button_rotate, button_view_particle])

                @ button_rotate.on_click
                def action_button_rotate(selfbutton):
                    clear_output(True)
                    display(buttons_facets)

                    # Run interactive function
                    @ interact(
                        facet_a_id=widgets.Dropdown(
                            options=[
                                i + 1 for i in range(self.Facets.nb_facets)],
                            value=1,
                            description='Facet a id:',
                            continuous_update=True,
                            layout=Layout(width='45%'),
                            style={'description_width': 'initial'}),
                        facet_b_id=widgets.Dropdown(
                            options=[
                                i + 1 for i in range(self.Facets.nb_facets)],
                            value=2,
                            description='Facet b id:',
                            continuous_update=True,
                            layout=Layout(width='45%'),
                            style={'description_width': 'initial'}),
                        u0=widgets.Text(
                            value="[1, 1, 1]",
                            placeholder="[1, 1, 1]",
                            description='Vector perpendicular to facet a:',
                            continuous_update=False,
                            # layout = Layout(width='20%'),
                            style={'description_width': 'initial'},),
                        v0=widgets.Text(
                            value="[1, -1, 0]",
                            placeholder="[1, -1, 0]",
                            description='Vector perpendicular to facet b:',
                            continuous_update=False,
                            # layout = Layout(width='20%'),
                            style={'description_width': 'initial'},),
                        w0=widgets.Text(
                            value="[1, 1, -2]",
                            placeholder="[1, 1, -2]",
                            description='Cross product of u0 and v0:',
                            continuous_update=False,
                            # layout = Layout(width='20%'),
                            style={'description_width': 'initial'},),
                        hkl_reference=widgets.Text(
                            value="[1, 1, 1]",
                            placeholder="[1, 1, 1]",
                            description='Reference for interplanar angles:',
                            continuous_update=False,
                            # layout = Layout(width='20%'),
                            style={'description_width': 'initial'},),
                        elev=widgets.BoundedIntText(
                            value=90,
                            placeholder=90,
                            min=0,
                            max=360,
                            description='Elevation of the axes in degrees:',
                            continuous_update=False,
                            layout=Layout(width='70%'),
                            style={'description_width': 'initial'},),
                        azim=widgets.BoundedIntText(
                            value=0,
                            placeholder=0,
                            min=0,
                            max=360,
                            description='Azimuth of the axes in degrees:',
                            continuous_update=False,
                            layout=Layout(width='70%'),
                            style={'description_width': 'initial'},),
                    )
                    def fix_facets(
                        facet_a_id,
                        facet_b_id,
                        u0,
                        v0,
                        w0,
                        hkl_reference,
                        elev,
                        azim,
                    ):
                        """
                        Function to interactively visualize the two facets that
                        will be chosen, to also help pick two vectors.
                        """
                        # Save parameters value
                        self.Facets.facet_a_id = facet_a_id
                        self.Facets.facet_b_id = facet_b_id
                        self.Facets.u0 = u0
                        self.Facets.v0 = v0
                        self.Facets.w0 = w0
                        self.Facets.hkl_reference = hkl_reference
                        self.Facets.elev = elev
                        self.Facets.azim = azim

                        # Extract list from strings
                        list_parameters = ["u0", "v0",
                                           "w0", "hkl_reference"]
                        try:
                            for p in list_parameters:
                                if getattr(self.Facets, p) == "":
                                    setattr(self.Facets, p, [])
                                else:
                                    setattr(self.Facets, p, literal_eval(
                                        getattr(self.Facets, p)))
                                # print(f"{p}:", getattr(self.Dataset, p))
                        except ValueError:
                            gutil.hash_print(f"Wrong list syntax for {p}")

                        # Plot the chosen facet to help the user to pick the facets
                        # he wants to use to orient the particule
                        self.Facets.extract_facet(
                            facet_id=self.Facets.facet_a_id, plot=True,
                            elev=self.Facets.elev, azim=self.Facets.azim,
                            output=False, save=False)
                        self.Facets.extract_facet(
                            facet_id=self.Facets.facet_b_id, plot=True,
                            elev=self.Facets.elev, azim=self.Facets.azim,
                            output=False, save=False)

                        display(Markdown("""# Field data"""))
                        display(self.Facets.field_data)

                        button_fix_facets = Button(
                            description="Fix parameters and extract data.",
                            layout=Layout(width='50%', height='35px'))
                        display(button_fix_facets)

                        @ button_fix_facets.on_click
                        def action_button_fix_facets(selfbutton):
                            """
                            Fix facets to compute the new rotation matrix and
                            launch the data extraction.
                            """
                            clear_output(True)

                            display(button_fix_facets)

                            display(
                                Markdown("""# Computing the rotation matrix"""))

                            # Take those facets' vectors
                            u = np.array([
                                self.Facets.field_data.n0[self.Facets.facet_a_id],
                                self.Facets.field_data.n1[self.Facets.facet_a_id],
                                self.Facets.field_data.n2[self.Facets.facet_a_id]])
                            v = np.array([
                                self.Facets.field_data.n0[self.Facets.facet_b_id],
                                self.Facets.field_data.n1[self.Facets.facet_b_id],
                                self.Facets.field_data.n2[self.Facets.facet_b_id]])

                            self.Facets.set_rotation_matrix(
                                u0=self.Facets.u0 /
                                np.linalg.norm(self.Facets.u0),
                                v0=self.Facets.v0 /
                                np.linalg.norm(self.Facets.v0),
                                w0=self.Facets.w0 /
                                np.linalg.norm(self.Facets.w0),
                                u=u,
                                v=v,
                            )

                            self.Facets.rotate_particle()

                            display(
                                Markdown("""# Computing interplanar angles from \
                                    reference"""))
                            print(
                                f"Used reference: {self.Facets.hkl_reference}")
                            self.Facets.fixed_reference(
                                hkl_reference=self.Facets.hkl_reference)

                            display(
                                Markdown("""# Strain values for each surface voxel \
                                and averaged per facet"""))
                            self.Facets.plot_strain(
                                elev=self.Facets.elev, azim=self.Facets.azim)

                            display(Markdown(
                                """# Displacement values for each surface voxel \
                                and averaged per facet"""))
                            self.Facets.plot_displacement(
                                elev=self.Facets.elev, azim=self.Facets.azim)

                            display(Markdown("""# Evolution curves"""))
                            self.Facets.evolution_curves()

                            # Also save edges and corners data
                            self.Facets.save_edges_corners_data()

                            display(Markdown("""# Field data"""))
                            display(self.Facets.field_data)

                            button_save_facet_data = Button(
                                description="Save data",
                                layout=Layout(width='50%', height='35px'))
                            display(button_save_facet_data)

                            @ button_save_facet_data.on_click
                            def action_button_save_facet_data(selfbutton):
                                """Save data ..."""
                                try:
                                    # Create subfolder
                                    try:
                                        os.mkdir(
                                            f"{self.Dataset.root_folder}{self.Dataset.scan_name}/postprocessing/facets_analysis/")
                                        print(
                                            f"Created {self.Dataset.root_folder}{self.Dataset.scan_name}/postprocessing/facets_analysis/")
                                    except (FileExistsError, PermissionError):
                                        print(
                                            f"{self.Dataset.root_folder}{self.Dataset.scan_name}/postprocessing/facets_analysis/ exists")

                                    # Save data
                                    self.Facets.save_data(
                                        f"{self.Dataset.scan_folder}/postprocessing/facets_analysis/field_data_{self.Dataset.scan}.csv")
                                    print(
                                        f"Saved field data as {self.Dataset.scan_folder}/postprocessing/facets_analysis/\
                                        field_data_{self.Dataset.scan}.csv")

                                    self.Facets.to_hdf5(
                                        f"{self.Dataset.scan_folder}{self.Dataset.scan_name}.cxi")
                                    print(
                                        f"Saved Facets class attributes in {self.Dataset.scan_folder}{self.Dataset.scan_name}.cxi")
                                except AttributeError:
                                    print(
                                        "Initialize the directories first to save the figures and data ...")

                @ button_view_particle.on_click
                def action_button_view_particle(selfbutton):
                    clear_output(True)
                    display(buttons_facets)

                    # Display visualisation window of facet class
                    display(self.Facets.window)

                # Display button
                display(buttons_facets)

            except TypeError:
                gutil.hash_print("Data type not supported.")

        if not load_data:
            self.tab_facet.children[1].disabled = False
            self.vtk_file_handler(parent_folder)
            gutil.hash_print("Cleared window.")
            clear_output(True)

    # Other tabs function

    def display_data_frame(
        self,
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
            self.tab_data_frame.children[1].disabled = True
            try:
                # csv data
                if show_logs == "load_csv":
                    try:
                        logs = pd.read_csv(csv_file)
                    except ValueError:
                        gutil.hash_print("Data type not supported.")
                    # else:
                    #     print(
                    #         f"""
                    #         ###############################################################################
                    #         For a more detailed analysis, please proceed as follows
                    #         import pandas as pd
                    #         df = pd.read_csv({self.csv_file})
                    #         You can then work on the `df` dataframe as you please.
                    #         ###############################################################################
                    #         """
                    #     )

                # field data from facet analysis
                elif show_logs == "load_field_data":
                    logs = self.Facets.field_data.copy()

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
                def pick_columns(
                        cols):
                    display(logs[list(cols)])

            except (FileNotFoundError, UnboundLocalError):
                gutil.hash_print("Wrong path")
            except AttributeError:
                print("You need to run the facet analysis in the dedicated tab first."
                      "Then this function will load the resulting DataFrame.")

        else:
            self.tab_data_frame.children[1].disabled = False
            self.csv_file_handler(parent_folder)
            clear_output(True)

    def load_data(
        self,
        unused_label_plot,
        folder,
        filename,
        cmap,
        data_use,
    ):
        """
        Allows the user to plot an array (1D, 2D or 3D) from npz, npy or .cxi
        files.

        :param folder: folder in which the files are located
        :param cmap: cmap used for plots
        :param filename: file name, can be multiple
        :param data_use: e.g. "2D"
         Can be "2D", "3D", "slices", "create_support", "extract_support",
         "smooth_support", "show_image", "hf_glance", "delete"
        """

        if data_use == "2D":
            # Disable widgets
            for w in self.tab_data.children[:-3]:
                w.disabled = True

            # Plot data
            for p in filename:
                gutil.hash_print(f"Showing {p}")
                plot.Plotter(
                    folder + "/" + p,
                    plot=data_use,
                    log="interact",
                    cmap=cmap
                )

        if data_use == "3D" and len(filename) == 1:
            # Disable widgets
            for w in self.tab_data.children[:-2]:
                w.disabled = True

            # Plot data
            plot.Plotter(
                folder + "/" + filename[0],
                plot=data_use,
                log="interact",
                cmap=cmap
            )

        if data_use == "slices":
            # Disable widgets
            for w in self.tab_data.children[:-3]:
                w.disabled = True

            # Plot data
            for p in filename:
                gutil.hash_print(f"Showing {p}")
                plot.Plotter(
                    folder + "/" + p,
                    plot=data_use,
                    log="interact",
                    cmap=cmap
                )

        elif data_use == "create_support" and len(filename) == 1:
            # Disable widgets
            for w in self.tab_data.children[:-2]:
                w.disabled = True

            # Initialize class
            sup = support.SupportTools(
                path_to_data=folder + "/" + filename[0])

            # Interactive function to loadt threshold value
            window_support = interactive(
                sup.compute_support,
                threshold=widgets.FloatText(
                    value=0.05,
                    step=0.001,
                    max=1,
                    min=0.001,
                    continuous_update=False,
                    description='Threshold:',
                    readout=True,
                    layout=Layout(width='20%'),
                    style={
                        'description_width': 'initial'},
                    disabled=False),
                compute=widgets.ToggleButton(
                    value=False,
                    description='Compute support ...',
                    button_style='',
                    icon='step-forward',
                    layout=Layout(width='45%'),
                    style={'description_width': 'initial'})
            )

            def support_handler(change):
                """Handles changes on the widget used for the initialization."""
                if not change.new:
                    window_support.children[0].disabled = False

                if change.new:
                    window_support.children[0].disabled = True

            window_support.children[1].observe(support_handler, names="value")

            display(window_support)

            # Update PyNX folder values
            self.pynx_folder_handler(change=self.preprocessing_folder)

        elif data_use == "extract_support" and len(filename) == 1:
            # Disable widgets
            for w in self.tab_data.children[:-2]:
                w.disabled = True

            # Initialize class
            sup = support.SupportTools(
                path_to_data=folder + "/" + filename[0])

            # Extract the support from the data file and save it as npz
            sup.extract_support()

            # Update PyNX folder values
            self.pynx_folder_handler(change=self.preprocessing_folder)

        elif data_use == "smooth_support" and len(filename) == 1:
            # Disable widgets
            for w in self.tab_data.children[:-2]:
                w.disabled = True

            # Initialize class
            sup = support.SupportTools(
                path_to_support=folder + "/" + filename[0])

            # Interactive function to loadt threshold value
            window_support = interactive(
                sup.gaussian_convolution,
                sigma=widgets.FloatText(
                    value=0.05,
                    step=0.001,
                    max=1,
                    min=0.001,
                    continuous_update=False,
                    description='Sigma:',
                    readout=True,
                    layout=Layout(width='20%'),
                    style={'description_width': 'initial'}),
                threshold=widgets.FloatText(
                    value=0.05,
                    step=0.001,
                    max=1,
                    min=0.001,
                    continuous_update=False,
                    description='Threshold:',
                    readout=True,
                    layout=Layout(width='20%'),
                    style={'description_width': 'initial'}),
                compute=widgets.ToggleButton(
                    value=False,
                    description='Compute support ...',
                    button_style='',
                    icon='step-forward',
                    layout=Layout(width='45%'),
                    style={'description_width': 'initial'})
            )

            def support_handler(change):
                """Handles changes on the widget used for the initialization."""
                if not change.new:
                    window_support.children[0].disabled = False

                if change.new:
                    window_support.children[0].disabled = True

            window_support.children[1].observe(support_handler, names="value")

            display(window_support)

            # Update PyNX folder values
            self.pynx_folder_handler(change=self.preprocessing_folder)

        elif data_use == "show_image":
            # Disable widgets
            for w in self.tab_data.children[:-2]:
                w.disabled = True

            try:
                for p in filename:
                    gutil.hash_print(f"Showing {p}")
                    display(Image(filename=folder + "/" + p))

            except (FileNotFoundError, ValueError):
                gutil.hash_print("Could not load image from file.")

        elif data_use == "hf_glance":
            # Disable widgets
            for w in self.tab_data.children[:-2]:
                w.disabled = True

            # Show tree
            for p in filename:
                try:
                    gutil.hash_print(f"Showing {p}")
                    display(H5Glance(folder + "/" + filename[0]))
                except TypeError:
                    gutil.hash_print(
                        "This tool supports .nxs, .cxi or .hdf5 files only.")

        elif data_use in [
            "3D", "create_support", "extract_support",
            "smooth_support",
        ] and len(filename) != 1:
            gutil.hash_print("Please select only one file.")

        elif data_use == "delete":
            # Disable widgets
            for w in self.tab_data.children[:-2]:
                w.disabled = True

            button_delete_data = Button(
                description=f"Delete files ?",
                button_style='',
                layout=Layout(width='70%'),
                style={'description_width': 'initial'},
                icon='step-forward')

            @ button_delete_data.on_click
            def action_button_delete_data(selfbutton):
                """Delete files."""
                for p in filename:
                    try:
                        os.remove(folder + "/" + p)
                        gutil.hash_print(f"Removed {p}")

                    except FileNotFoundError:
                        gutil.hash_print(f"Could not remove {p}")

            display(button_delete_data)

        elif data_use is False:
            plt.close()
            for w in self.tab_data.children[:-2]:
                w.disabled = False
            self.plot_folder_handler(change=folder)
            gutil.hash_print("Cleared window.")
            clear_output(True)

    @ staticmethod
    def display_readme(contents):
        """
        Help text about different steps in data analysis workflow.

        :param contents: e.g. "Preprocessing"
         Possible values are "Preprocessing", "Phase retrieval",
         "Postprocessing" or "Facet analysis"
        """
        if contents == "Preprocessing":
            clear_output(True)
            print(help(gwaihir.gui.gui.Interface.initialize_preprocessing))

        elif contents == "Phase retrieval":
            clear_output(True)
            print(help(gwaihir.gui.gui.Interface.initialize_phase_retrieval))

        elif contents == "Postprocessing":
            clear_output(True)
            print(help(gwaihir.gui.gui.Interface.initialize_postprocessing))

        elif contents == "Facet analysis":
            clear_output(True)
            print(help(facet_analysis.Facets))
            print("""
                The output DataFrame can be opened in the `Logs` tab.
                The "View particle" tool helps you visualizing the particle
                facets.
                """)

        elif contents is "GUI":
            display(Markdown("# Welcome to `Gwaihir`"))
            display(Markdown("Remember that a detailed tutorial on the installation of each package is "
                             " available on the [Github](https://github.com/DSimonne/gwaihir#welcome),"
                             " together with a video that presents the data analysis workflow."
                             ))
            display(Markdown("On the other tabs of this README are presented the main functions used for"
                             " data analysis and their parameters."))
            display(Markdown(""))

            display(Markdown("# Example of parameter values"))
            display(Markdown("## SixS data (SOLEIL)"))
            display(Markdown(
                "Most of the initial guesses are valid. Be careful about the energy, scan number, central pixel and detector."
                " If you are working with the vertical configuration, make sure that the mask is correct."))

            display(Markdown("## ID01 data (ESRF)"))
            display(Markdown("* Scan number: `11`"))
            display(Markdown(
                "* Data directory: `/data/id01/inhouse/david/UM2022/ID01/CXIDB-I182/CH4760/`"))
            display(Markdown("* Detector: `Maxipix`"))
            display(Markdown("* Template imagefile: `S11/data_mpx4_%05d.edf.gz`"))
            display(Markdown("* Sample offsets: `(90, 0, 0)`"))
            display(Markdown("* Sample detector distance (m): `0.50678`"))
            display(Markdown("* X-ray energy (eV): `9000`"))
            display(Markdown("* Beamline: `ID01`"))
            display(Markdown(
                "* specfile name: `/data/id01/inhouse/david/UM2022/ID01/CXIDB-I182/CH4760/l5.spec` (in my case, please use a direct path)"))
            display(Markdown("* Rocking angle: `outofplane`"))

            display(Markdown("## P10 data (PETRA)"))
            display(Markdown("* Sample name: `align_03`"))
            display(Markdown("* Scan number: `11`"))
            display(
                Markdown("* Data directory: `/data/id01/inhouse/david/UM2022/Petra/raw/`"))
            display(Markdown("* Detector: `Eiger4M`"))
            display(Markdown("* Template imagefile: `_master.h5`"))
            display(Markdown("* Sample offsets: `(0, 0, 0, 0)`"))
            display(Markdown("* Sample detector distance (m): `0.50678`"))
            display(Markdown("* X-ray energy (eV): `9000`"))
            display(Markdown("* Beamline: `P10`"))
            display(Markdown(
                "* specfile name: `/data/id01/inhouse/david/UM2022/ID01/CXIDB-I182/CH4760/l5.spec` (in my case, please use a direct path)"))
            display(Markdown("* Rocking angle: `inplane`"))
            display(Markdown("* Pixel size (in phase retrieval): `75`"))

            display(Markdown("# To go further ..."))
            display(Markdown("* All the plotting functions are accessible in `gwaihir.plot`, try to use the `Plotter` Class"
                             " that reads all kind of numpy arrays."
                             " e.g. `Plotter(filename=\"TestGui/S11/preprocessing/S11_maskpynx_align-q-y_norm_252_420_392_1_1_1.npz\", plot=\"2D\")`"
                             ))
            display(Markdown("* I highly recommend the use of [Paraview](https://www.paraview.org/) for 3D contouring."
                             " Many tutorials can be found online: <https://www.bu.edu/tech/support/research/training-consulting/online-tutorials/paraview/>"))
            display(Markdown(
                "* If you saved your data in the cxi format, you can visualize it with JupyterLab !"))
            display(Markdown("* `Qt5Agg` is a backend that does not work on remote servers, if you install `Gwaihir` on"
                             " your local computer, you can use this backend for masking. "
                             " We are currently working on implementing a solution in Jupyter Notebook with Bokeh."))
            display(Markdown("* If you saved your data in the `.cxi` format, you can visualize it with JupyterLab !"
                             " Otherwise you can use [`silx`](http://www.silx.org/doc/silx/0.7.0/applications/view.html) from the terminal"))

            display(Markdown(
                "## Type the following code to stop the scrolling in the output cell, then reload the cell."))
            display(Markdown("`%%javascript`"))
            display(
                Markdown("`IPython.OutputArea.prototype._should_scroll = function(lines) {`"))
            display(Markdown("`return false;`"))
            display(Markdown("`}`"))

            display(Markdown("To contact me <david.simonne@synchrotron-soleil.fr>"))

    # Below are handlers

    def init_handler(self, change):
        """Handles changes on the widget used for the initialization."""
        if not change.new:
            for w in self._list_widgets_init_dir.children[:8]:
                w.disabled = False

            for w in self._list_widgets_preprocessing.children[:-1]:
                w.disabled = True

        if change.new:
            for w in self._list_widgets_init_dir.children[:8]:
                w.disabled = True

            for w in self._list_widgets_preprocessing.children[:-1]:
                w.disabled = False

            self.beamline_handler(
                change=self._list_widgets_preprocessing.children[1].value)
            self.bragg_peak_centering_handler(
                change=self._list_widgets_preprocessing.children[13].value)
            self.reload_data_handler(
                change=self._list_widgets_preprocessing.children[25].value)
            # self.orthogonalisation_handler(
            #     change=self._list_widgets_preprocessing.children[44].value)

    def sub_directories_handler(self, change):
        """Handles changes linked to root_folder subdirectories"""
        try:
            sub_dirs = [x[0] + "/" for x in os.walk(change.new)]
        except AttributeError:
            sub_dirs = [x[0] + "/" for x in os.walk(change)]
        finally:
            if self._list_widgets_init_dir.children[-2].value:
                self._list_widgets_strain.children[-4].options = sub_dirs
                self.tab_data.children[1].options = sub_dirs
                self.tab_facet.children[1].options = sub_dirs
                self._list_widgets_phase_retrieval.children[1].options = sub_dirs

    def beamline_handler(self, change):
        """Handles changes on the widget used for the beamline."""
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

    def bragg_peak_centering_handler(self, change):
        """Handles changes related to the centering of the Bragg peak."""
        try:
            if change.new == "manual":
                self._list_widgets_preprocessing.children[14].disabled = False

            if change.new != "manual":
                self._list_widgets_preprocessing.children[14].disabled = True

        except AttributeError:
            if change == "manual":
                self._list_widgets_preprocessing.children[14].disabled = False

            if change != "manual":
                self._list_widgets_preprocessing.children[14].disabled = True

    def reload_data_handler(self, change):
        """Handles changes related to data reloading."""
        try:
            if change.new:
                for w in self._list_widgets_preprocessing.children[26:28]:
                    w.disabled = False

            if not change.new:
                for w in self._list_widgets_preprocessing.children[26:28]:
                    w.disabled = True

        except AttributeError:
            if change:
                for w in self._list_widgets_preprocessing.children[26:28]:
                    w.disabled = False

            if not change:
                for w in self._list_widgets_preprocessing.children[26:28]:
                    w.disabled = True

    def orthogonalisation_handler(self, change):
        """Handles changes related to data orthogonalisation."""
        try:
            if change.new:
                for w in self._list_widgets_preprocessing.children[45:68]:
                    w.disabled = False

            if not change.new:
                for w in self._list_widgets_preprocessing.children[45:68]:
                    w.disabled = True
        except AttributeError:
            if change:
                for w in self._list_widgets_preprocessing.children[45:68]:
                    w.disabled = False

            if not change:
                for w in self._list_widgets_preprocessing.children[45:68]:
                    w.disabled = True

    def preprocess_handler(self, change):
        """Handles changes on the widget used for the preprocessing."""
        try:
            if not change.new:
                self._list_widgets_init_dir.children[8].disabled = False

                for w in self._list_widgets_preprocessing.children[:-2]:
                    w.disabled = False

                self.beamline_handler(
                    change=self._list_widgets_preprocessing.children[1].value)
                self.bragg_peak_centering_handler(
                    change=self._list_widgets_preprocessing.children[13].value)
                self.reload_data_handler(
                    change=self._list_widgets_preprocessing.children[25].value)
                # self.orthogonalisation_handler(
                #     change=self._list_widgets_preprocessing.children[44].value)

            if change.new:
                self._list_widgets_init_dir.children[8].disabled = True

                for w in self._list_widgets_preprocessing.children[:-2]:
                    w.disabled = True

        except AttributeError:
            if not change:
                self._list_widgets_init_dir.children[8].disabled = False

                for w in self._list_widgets_preprocessing.children[:-2]:
                    w.disabled = False

                self.beamline_handler(
                    change=self._list_widgets_preprocessing.children[1].value)
                self.bragg_peak_centering_handler(
                    change=self._list_widgets_preprocessing.children[13].value)
                self.reload_data_handler(
                    change=self._list_widgets_preprocessing.children[25].value)
                # self.orthogonalisation_handler(
                #     change=self._list_widgets_preprocessing.children[44].value)

            if change:
                self._list_widgets_init_dir.children[8].disabled = True

                for w in self._list_widgets_preprocessing.children[:-2]:
                    w.disabled = True

    def csv_file_handler(self, change):
        """List all .csv files in change subdirectories"""
        csv_files = []

        try:
            for x in os.walk(change.new):
                csv_files += sorted(glob.glob(f"{x[0]}/*.csv"),
                                    key=os.path.getmtime)

        except AttributeError:
            for x in os.walk(change):
                csv_files += sorted(glob.glob(f"{x[0]}/*.csv"),
                                    key=os.path.getmtime)

        finally:
            self.tab_data_frame.children[2].options = csv_files

    def pynx_folder_handler(self, change):
        """Handles changes on the widget used to load a data file."""
        try:
            list_all_npz = [os.path.basename(f) for f in sorted(
                glob.glob(change.new + "/*.npz"),
                key=os.path.getmtime)]

            list_probable_iobs_files = [os.path.basename(f) for f in sorted(
                glob.glob(change.new + "/*_pynx_align*.npz"),
                key=os.path.getmtime)]

            list_probable_mask_files = [os.path.basename(f) for f in sorted(
                glob.glob(change.new + "/*maskpynx*.npz"),
                key=os.path.getmtime)]

            # support list
            self._list_widgets_phase_retrieval.children[4].options = [""]\
                + [os.path.basename(f) for f in sorted(
                    glob.glob(change.new + "/*.npz"), key=os.path.getmtime)]

            # obj list
            self._list_widgets_phase_retrieval.children[5].options = [""]\
                + [os.path.basename(f) for f in sorted(
                    glob.glob(change.new + "/*.npz"), key=os.path.getmtime)]

        except AttributeError:
            list_all_npz = [os.path.basename(f) for f in sorted(
                glob.glob(change + "/*.npz"), key=os.path.getmtime)]

            list_probable_iobs_files = [os.path.basename(f) for f in sorted(
                glob.glob(change + "/*_pynx_align*.npz"), key=os.path.getmtime)]

            list_probable_mask_files = [os.path.basename(f) for f in sorted(
                glob.glob(change + "/*maskpynx*.npz"), key=os.path.getmtime)]

            # support list
            self._list_widgets_phase_retrieval.children[4].options = [""]\
                + [os.path.basename(f) for f in sorted(
                    glob.glob(change + "/*.npz"), key=os.path.getmtime)]

            # obj list
            self._list_widgets_phase_retrieval.children[5].options = [""]\
                + [os.path.basename(f) for f in sorted(
                    glob.glob(change + "/*.npz"), key=os.path.getmtime)]

        finally:
            # Find probable iobs file
            temp_list = list_all_npz.copy()
            for f in list_probable_iobs_files:
                try:
                    temp_list.remove(f)
                except ValueError:
                    # Not in list
                    pass
            sorted_iobs_list = list_probable_iobs_files + temp_list + [""]

            # Find probable mask file
            temp_list = list_all_npz.copy()
            for f in list_probable_mask_files:
                try:
                    temp_list.remove(f)
                except ValueError:
                    # not in list
                    pass
            sorted_mask_list = list_probable_mask_files + temp_list + [""]

            # iobs list
            self._list_widgets_phase_retrieval.children[2].options = sorted_iobs_list

            # mask list
            self._list_widgets_phase_retrieval.children[3].options = sorted_mask_list

    def pynx_psf_handler(self, change):
        """Handles changes related to the psf."""
        try:
            if change.new:
                for w in self._list_widgets_phase_retrieval.children[16:20]:
                    w.disabled = False

            if not change.new:
                for w in self._list_widgets_phase_retrieval.children[16:20]:
                    w.disabled = True

        except AttributeError:
            if change:
                for w in self._list_widgets_phase_retrieval.children[16:20]:
                    w.disabled = False

            if not change:
                for w in self._list_widgets_phase_retrieval.children[16:20]:
                    w.disabled = True

        self.pynx_peak_shape_handler(
            change=self._list_widgets_phase_retrieval.children[16].value)

    def pynx_peak_shape_handler(self, change):
        """Handles changes related to psf the peak shape."""
        try:
            if change.new != "pseudo-voigt":
                self._list_widgets_phase_retrieval.children[18].disabled = True

            if change.new == "pseudo-voigt":
                self._list_widgets_phase_retrieval.children[18].disabled = False

        except AttributeError:
            if change != "pseudo-voigt":
                self._list_widgets_phase_retrieval.children[18].disabled = True

            if change == "pseudo-voigt":
                self._list_widgets_phase_retrieval.children[18].disabled = False

    def run_pynx_handler(self, change):
        """Handles changes related to the phase retrieval."""
        if change.new:
            for w in self._list_widgets_phase_retrieval.children[:-2]:
                w.disabled = True
            self._list_widgets_phase_retrieval.children[-4].disabled = False

        elif not change.new:
            for w in self._list_widgets_phase_retrieval.children[:-2]:
                w.disabled = False

            self.pynx_psf_handler(
                change=self._list_widgets_phase_retrieval.children[15].value)

    def strain_folder_handler(self, change):
        """Handles changes on the widget used to load a data file."""
        try:
            options = [""] + [os.path.basename(f) for f in sorted(
                glob.glob(change.new + "/*.h5")
                + glob.glob(change.new + "/*.cxi")
                + glob.glob(change.new + "/*.npy")
                + glob.glob(change.new + "/*.npz"),
                key=os.path.getmtime)
            ]

        except AttributeError:
            options = [""] + [os.path.basename(f) for f in sorted(
                glob.glob(change + "/*.h5")
                + glob.glob(change + "/*.cxi")
                + glob.glob(change + "/*.npy")
                + glob.glob(change + "/*.npz"),
                key=os.path.getmtime)
            ]

        finally:
            self._list_widgets_strain.children[-3].options = options

    def plot_folder_handler(self, change):
        """Handles changes on the widget used to load a data file."""
        try:
            options = [""] + [os.path.basename(f) for f in sorted(
                glob.glob(change.new + "/*.npy")
                + glob.glob(change.new + "/*.npz")
                + glob.glob(change.new + "/*.cxi")
                + glob.glob(change.new + "/*.h5")
                + glob.glob(change.new + "/*.png"),
                key=os.path.getmtime)
            ]

        except AttributeError:
            options = [""] + [os.path.basename(f) for f in sorted(
                glob.glob(change + "/*.npy")
                + glob.glob(change + "/*.npz")
                + glob.glob(change + "/*.cxi")
                + glob.glob(change + "/*.h5")
                + glob.glob(change + "/*.png"),
                key=os.path.getmtime)
            ]

        finally:
            self.tab_data.children[2].options = [os.path.basename(f)
                                                 for f in options]

            if self.plot_only:
                self.tab_data.children[1].options = [
                    x[0] + "/" for x in os.walk(os.getcwd())
                ]

    def vtk_file_handler(self, change):
        """List all .vtk files in change subdirectories"""
        vtk_files = []

        try:
            for x in os.walk(change.new):
                vtk_files += sorted(glob.glob(f"{x[0]}/*.vtk"),
                                    key=os.path.getmtime)

        except AttributeError:
            for x in os.walk(change):
                vtk_files += sorted(glob.glob(f"{x[0]}/*.vtk"),
                                    key=os.path.getmtime)

        finally:
            self.tab_facet.children[2].options = vtk_files
