import ipywidgets as widgets
from ipywidgets import interact, Button, Layout, interactive
from IPython.display import display, Markdown, clear_output, Image


class TabInstrument(widgets.Box):
    """

    """

    def __init__(self):
        """

        """
        super(TabInstrument, self).__init__()

        self._list_widgets = widgets.VBox(
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

            detector_distance=widgets.FloatText(
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
        )

        # Create window
        self.window = widgets.VBox([
            self._list_widgets.children[0],
            self._list_widgets.children[1],
            widgets.HBox(self._list_widgets.children[2:4]),
            self._list_widgets.children[4],
            self._list_widgets.children[5],
            self._list_widgets.children[6],
            widgets.HBox(self._list_widgets.children[7:9]),
            self._list_widgets.children[9],
            widgets.HBox(self._list_widgets.children[10:12]),
            widgets.HBox(self._list_widgets.children[12:14]),
            widgets.HBox(self._list_widgets.children[14:17]),
            widgets.HBox(self._list_widgets.children[17:20]),
            widgets.HBox(self._list_widgets.children[20:25]),
        ])

        # Assign handlers
        self._list_widgets.children[1].observe.observe(
            self.orthogonalisation_handler, names="value")

    # Define handlers
    def orthogonalisation_handler(self, change):
        """Handles changes related to data orthogonalisation."""
        try:
            if change.new:
                for w in self._list_widgets.children[2:]:
                    w.disabled = False

            if not change.new:
                for w in self._list_widgets.children[2:]:
                    w.disabled = True
        except AttributeError:
            if change:
                for w in self._list_widgets.children[2:]:
                    w.disabled = False

            if not change:
                for w in self._list_widgets.children[2:]:
                    w.disabled = True
