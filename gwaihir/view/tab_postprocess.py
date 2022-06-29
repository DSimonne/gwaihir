import ipywidgets as widgets
from ipywidgets import interact, Button, Layout, interactive
from IPython.display import display, Markdown, clear_output, Image


class TabPostprocessing(widgets.Box):
    """

    """

    def __init__(self):
        """

        """
        super(TabPostProcessing, self).__init__()

        # Brief header describing the tab
        self.header = 'Postprocess'

        # Create tab widgets
        self._list_widgets = widgets.VBox(
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

            skip_unwrap=widgets.Checkbox(
                value=False,
                description='Skip phase unwrap',
                layout=Layout(width='15%'),
                style={
                    'description_width': 'initial'}
            ),

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

            reconstruction_files=widgets.Dropdown(
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

        # Create window
        self.window = widgets.VBox([
            self._list_widgets.children[0],
            widgets.HBox(self._list_widgets.children[1:3]),
            self._list_widgets.children[3],
            widgets.HBox(self._list_widgets.children[4:6]),
            widgets.HBox(self._list_widgets.children[6:8]),
            widgets.HBox(self._list_widgets.children[8:10]),
            self._list_widgets.children[10],
            widgets.HBox(self._list_widgets.children[11:13]),
            widgets.HBox(self._list_widgets.children[13:17]),
            widgets.HBox(self._list_widgets.children[17:21]),
            self._list_widgets.children[21],
            widgets.HBox(self._list_widgets.children[22:24]),
            widgets.HBox(self._list_widgets.children[24:27]),
            self._list_widgets.children[27],
            widgets.HBox(self._list_widgets.children[28:31]),
            widgets.HBox(self._list_widgets.children[31:33]),
            widgets.HBox(self._list_widgets.children[33:36]),
            widgets.HBox(self._list_widgets.children[36:38]),
            self._list_widgets.children[38],
            widgets.HBox(self._list_widgets.children[39:42]),
            widgets.HBox(self._list_widgets.children[42:44]),
            self._list_widgets.children[44],
            widgets.HBox(self._list_widgets.children[45:49]),
            self._list_widgets.children[49],
            widgets.HBox(self._list_widgets.children[50:52]),
            self._list_widgets.children[52],
            widgets.HBox(self._list_widgets.children[53:56]),
            widgets.HBox(self._list_widgets.children[56:59]),
            self._list_widgets.children[-4],
            self._list_widgets.children[-3],
            self._list_widgets.children[-2],
            self._list_widgets.children[-1],
        ])

        # Assign handler
        self._list_widgets.children[-4].observe(
            self.strain_folder_handler, names="value")

    # Define handlers
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
            self._list_widgets.children[-3].options = options
