import ipywidgets as widgets
from ipywidgets import interact, Button, Layout, interactive
from IPython.display import display, Markdown, clear_output, Image


class TabPhaseRetrieval(widgets.Box):
    """

    """

    def __init__(self):
        """

        """
        super(TabPhaseRetrieval, self).__init__()

        # Brief header describing the tab
        self.header = 'Phase retrieval'

        # Create tab widgets
        self._list_widgets = widgets.VBox(
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
                    ('Run batch job (slurm)', "batch"),
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

        # Create window
        self.window = widgets.VBox([
            widgets.VBox(self._list_widgets.children[:6]),
            widgets.HBox(self._list_widgets.children[6:8]),
            self._list_widgets.children[8],
            widgets.HBox(self._list_widgets.children[9:11]),
            widgets.HBox(self._list_widgets.children[11:14]),
            self._list_widgets.children[14],
            widgets.HBox(self._list_widgets.children[15:19]),
            self._list_widgets.children[19],
            self._list_widgets.children[20],
            widgets.HBox(self._list_widgets.children[21:25]),
            self._list_widgets.children[25],
            self._list_widgets.children[29],
            widgets.HBox(self._list_widgets.children[30:34]),
            widgets.HBox(self._list_widgets.children[34:37]),
            self._list_widgets.children[26],
            widgets.HBox(self._list_widgets.children[27:29]),
            self._list_widgets.children[-5],
            self._list_widgets.children[-4],
            self._list_widgets.children[-3],
            self._list_widgets.children[-2],
            self._list_widgets.children[-1],
        ])

        # Assign handler
        self._list_widgets.children[1].observe(
            self.pynx_folder_handler, names="value")
        self._list_widgets.children[15].observe(
            self.pynx_psf_handler, names="value")
        self._list_widgets.children[16].observe(
            self.pynx_peak_shape_handler, names="value")
        self._list_widgets.children[-4].observe(
            self.run_pynx_handler, names="value")
        self._list_widgets.children[-2].observe(
            self.run_pynx_handler, names="value")

    # Define handlers
    def pynx_folder_handler(self, change):
        """Handles changes on the widget used to load a data file."""
        try:
            list_all_npz = [os.path.basename(f) for f in sorted(
                glob.glob(change.new + "/*.npz"),
                key=os.path.getmtime)]

            list_probable_iobs_files = [os.path.basename(f) for f in sorted(
                glob.glob(change.new + "/*_pynx_*.npz"),
                key=os.path.getmtime)]

            list_probable_mask_files = [os.path.basename(f) for f in sorted(
                glob.glob(change.new + "/*maskpynx*.npz"),
                key=os.path.getmtime)]

            # support list
            self._list_widgets.children[4].options = [""]\
                + [os.path.basename(f) for f in sorted(
                    glob.glob(change.new + "/*.npz"), key=os.path.getmtime)]

            # obj list
            self._list_widgets.children[5].options = [""]\
                + [os.path.basename(f) for f in sorted(
                    glob.glob(change.new + "/*.npz"), key=os.path.getmtime)]

        except AttributeError:
            list_all_npz = [os.path.basename(f) for f in sorted(
                glob.glob(change + "/*.npz"), key=os.path.getmtime)]

            list_probable_iobs_files = [os.path.basename(f) for f in sorted(
                glob.glob(change + "/*_pynx_*.npz"), key=os.path.getmtime)]

            list_probable_mask_files = [os.path.basename(f) for f in sorted(
                glob.glob(change + "/*maskpynx*.npz"), key=os.path.getmtime)]

            # support list
            self._list_widgets.children[4].options = [""]\
                + [os.path.basename(f) for f in sorted(
                    glob.glob(change + "/*.npz"), key=os.path.getmtime)]

            # obj list
            self._list_widgets.children[5].options = [""]\
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
            self._list_widgets.children[2].options = sorted_iobs_list

            # mask list
            self._list_widgets.children[3].options = sorted_mask_list

    def pynx_psf_handler(self, change):
        """Handles changes related to the psf."""
        try:
            if change.new:
                for w in self._list_widgets.children[16:20]:
                    w.disabled = False

            if not change.new:
                for w in self._list_widgets.children[16:20]:
                    w.disabled = True

        except AttributeError:
            if change:
                for w in self._list_widgets.children[16:20]:
                    w.disabled = False

            if not change:
                for w in self._list_widgets.children[16:20]:
                    w.disabled = True

        self.pynx_peak_shape_handler(
            change=self._list_widgets.children[16].value)

    def pynx_peak_shape_handler(self, change):
        """Handles changes related to psf the peak shape."""
        try:
            if change.new != "pseudo-voigt":
                self._list_widgets.children[18].disabled = True

            if change.new == "pseudo-voigt":
                self._list_widgets.children[18].disabled = False

        except AttributeError:
            if change != "pseudo-voigt":
                self._list_widgets.children[18].disabled = True

            if change == "pseudo-voigt":
                self._list_widgets.children[18].disabled = False

    def run_pynx_handler(self, change):
        """Handles changes related to the phase retrieval."""
        if change.new:
            for w in self._list_widgets.children[:-2]:
                w.disabled = True
            self._list_widgets.children[-4].disabled = False

        elif not change.new:
            for w in self._list_widgets.children[:-2]:
                w.disabled = False

            self.pynx_psf_handler(
                change=self._list_widgets.children[15].value)
