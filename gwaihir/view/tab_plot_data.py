import ipywidgets as widgets
from ipywidgets import interact, Button, Layout, interactive
from IPython.display import display, Markdown, clear_output, Image


class TabPlotData(widgets.Box):
    """

    """

    def __init__(self, plot_tab_only=False):
        """

        """
        super(TabPlotData, self).__init__()

        self._list_widgets = widgets.VBox(
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
                    + glob.glob(os.getcwd() + "/*.nxs")
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

        # Create window
        self.window = self._list_widgets

        # Assign handlers
        self._list_widgets.children[1].observe(
            self.plot_folder_handler, names="value")

    # Define handlers
    def plot_folder_handler(self, change):
        """Handles changes on the widget used to load a data file."""
        try:
            options = [""] + [os.path.basename(f) for f in sorted(
                glob.glob(change.new + "/*.npy")
                + glob.glob(change.new + "/*.npz")
                + glob.glob(change.new + "/*.cxi")
                + glob.glob(change.new + "/*.h5")
                + glob.glob(change.new + "/*.nxs")
                + glob.glob(change.new + "/*.png"),
                key=os.path.getmtime)
            ]

        except AttributeError:
            options = [""] + [os.path.basename(f) for f in sorted(
                glob.glob(change + "/*.npy")
                + glob.glob(change + "/*.npz")
                + glob.glob(change + "/*.cxi")
                + glob.glob(change + "/*.h5")
                + glob.glob(change + "/*.nxs")
                + glob.glob(change + "/*.png"),
                key=os.path.getmtime)
            ]

        finally:
            self._list_widgets.children[2].options = [os.path.basename(f)
                                                      for f in options]

            if plot_tab_only:
                self._list_widgets.children[1].options = [
                    x[0] + "/" for x in os.walk(os.getcwd())
                ]
