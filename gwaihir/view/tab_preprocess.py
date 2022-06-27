import ipywidgets as widgets
from ipywidgets import interact, Button, Layout, interactive
from IPython.display import display, Markdown, clear_output, Image


class TabPreprocess(widgets.Box):
    """

    """

    def __init__(self):
        """

        """
        super(TabPreProcess, self).__init__()

        self._list_widgets = widgets.VBox(

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
                         "ID01BLISS", "ID27"],
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

        # Create window
        self.tab_beamline = widgets.VBox([
            self._list_widgets.children[0],
            self._list_widgets.children[1],
            widgets.HBox(self._list_widgets.children[2:4]),
            widgets.HBox(self._list_widgets.children[4:7]),
            self._list_widgets.children[7],
            self._list_widgets.children[8],
            self._list_widgets.children[9],
            widgets.HBox(self._list_widgets.children[10:12]),
        ])

        # Parameters related to data cropping/padding/centering
        self.tab_reduction = widgets.VBox([
            self._list_widgets.children[12],
            widgets.HBox(self._list_widgets.children[13:15]),
            self._list_widgets.children[15],
            widgets.HBox(self._list_widgets.children[16:19]),
            self._list_widgets.children[19],
            widgets.HBox(self._list_widgets.children[20:23]),
            self._list_widgets.children[23],
        ])

        # Parameters used when reloading processed data
        self.tab_save_load = widgets.VBox([
            self._list_widgets.children[24],
            widgets.HBox(self._list_widgets.children[25:28]),
            self._list_widgets.children[28],
            widgets.HBox(self._list_widgets.children[29:34]),
        ])

        # Group all preprocess tabs into a single one, besides detector and
        # setup parameter
        self.window = widgets.VBox([
            self.tab_beamline,
            self.tab_reduction,
            self.tab_save_load,
            self._list_widgets.children[-1]
        ])

        # Assign handlers
        self._list_widgets.children[1].observe(
            self.beamline_handler, names="value")
        self._list_widgets.children[13].observe(
            self.bragg_peak_centering_handler, names="value")
        self._list_widgets.children[25].observe(
            self.reload_data_handler, names="value")
        self._list_widgets.children[-2].observe(
            self.preprocess_handler, names="value")

    # Create handlers
    def beamline_handler(self, change):
        """Handles changes on the widget used for the beamline."""
        try:
            if change.new in ["SIXS_2019", "ID01"]:
                for w in self._list_widgets.children[2:7]:
                    w.disabled = True

            if change.new not in ["SIXS_2019", "ID01"]:
                for w in self._list_widgets.children[2:7]:
                    w.disabled = False
        except AttributeError:
            if change in ["SIXS_2019", "ID01"]:
                for w in self._list_widgets.children[2:7]:
                    w.disabled = True

            if change not in ["SIXS_2019", "ID01"]:
                for w in self._list_widgets.children[2:7]:
                    w.disabled = False

    def bragg_peak_centering_handler(self, change):
        """Handles changes related to the centering of the Bragg peak."""
        try:
            if change.new == "manual":
                self._list_widgets.children[14].disabled = False

            if change.new != "manual":
                self._list_widgets.children[14].disabled = True

        except AttributeError:
            if change == "manual":
                self._list_widgets.children[14].disabled = False

            if change != "manual":
                self._list_widgets.children[14].disabled = True

    def reload_data_handler(self, change):
        """Handles changes related to data reloading."""
        try:
            if change.new:
                for w in self._list_widgets.children[26:28]:
                    w.disabled = False

            if not change.new:
                for w in self._list_widgets.children[26:28]:
                    w.disabled = True

        except AttributeError:
            if change:
                for w in self._list_widgets.children[26:28]:
                    w.disabled = False

            if not change:
                for w in self._list_widgets.children[26:28]:
                    w.disabled = True

    def preprocess_handler(self, change):
        """Handles changes on the widget used for the preprocessing."""
        try:
            if not change.new:
                self._list_widgets_init_dir.children[8].disabled = False

                for w in self._list_widgets.children[:-2]:
                    w.disabled = False

                self.beamline_handler(
                    change=self._list_widgets.children[1].value)
                self.bragg_peak_centering_handler(
                    change=self._list_widgets.children[13].value)
                self.reload_data_handler(
                    change=self._list_widgets.children[25].value)

            if change.new:
                self._list_widgets_init_dir.children[8].disabled = True

                for w in self._list_widgets.children[:-2]:
                    w.disabled = True

        except AttributeError:
            if not change:
                self._list_widgets_init_dir.children[8].disabled = False

                for w in self._list_widgets.children[:-2]:
                    w.disabled = False

                self.beamline_handler(
                    change=self._list_widgets.children[1].value)
                self.bragg_peak_centering_handler(
                    change=self._list_widgets.children[13].value)
                self.reload_data_handler(
                    change=self._list_widgets.children[25].value)

            if change:
                self._list_widgets_init_dir.children[8].disabled = True

                for w in self._list_widgets.children[:-2]:
                    w.disabled = True
