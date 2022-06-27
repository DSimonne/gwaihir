import ipywidgets as widgets
from ipywidgets import interact, Button, Layout, interactive
from IPython.display import display, Markdown, clear_output, Image


class TabReadme(widgets.Box):
    """

    """

    def __init__(self):
        """

        """
        super(TabReadme, self).__init__()

        self._list_widgets = widgets.VBox(
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

        # Create window
        self.window = widgets.VBox([
            self._list_widgets.children[0],
            widgets.HBox(self._list_widgets.children[1:3]),
            self._list_widgets.children[3],
            self._list_widgets.children[4],
            self._list_widgets.children[5],
            self._list_widgets.children[6],
            self._list_widgets.children[7],
            self._list_widgets.children[8],
            self._list_widgets.children[-1],
        ])

        # Assign handlers
        self._list_widgets.children[4].observe(
            self.sub_directories_handler, names="value")
        self._list_widgets.children[8].observe(
            self.init_handler, names="value")

    # Create handlers
    def init_handler(self, change):
        """Handles changes on the widget used for the initialization."""
        if not change.new:
            for w in self._list_widgets.children[:8]:
                w.disabled = False

            for w in self._list_widgets_preprocessing.children[:-1]:
                w.disabled = True

        if change.new:
            for w in self._list_widgets.children[:8]:
                w.disabled = True

            for w in self._list_widgets_preprocessing.children[:-1]:
                w.disabled = False

            self.beamline_handler(
                change=self._list_widgets_preprocessing.children[1].value)
            self.bragg_peak_centering_handler(
                change=self._list_widgets_preprocessing.children[13].value)
            self.reload_data_handler(
                change=self._list_widgets_preprocessing.children[25].value)

    def sub_directories_handler(self, change):
        """Handles changes linked to root_folder subdirectories"""
        try:
            sub_dirs = [x[0] + "/" for x in os.walk(change.new)]
        except AttributeError:
            sub_dirs = [x[0] + "/" for x in os.walk(change)]
        finally:
            if self._list_widgets.children[-2].value:
                self._list_widgets_strain.children[-4].options = sub_dirs
                self.tab_data.children[1].options = sub_dirs
                self.tab_facet.children[1].options = sub_dirs
                self._list_widgets_phase_retrieval.children[1].options = sub_dirs
