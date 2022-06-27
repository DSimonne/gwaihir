import ipywidgets as widgets
from ipywidgets import interact, Button, Layout, interactive
from IPython.display import display, Markdown, clear_output, Image


class TabDataFrame(widgets.Box):
    """

    """

    def __init__(self):
        """

        """
        super(TabDataFrame, self).__init__()

        self._list_widgets = widgets.VBox(
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

        # Create window
        self.window = self._list_widgets

        # Assign handlers
        self._list_widgets.children[1].observe(
            self.csv_file_handler, names="value")

    # Handler
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
            self.window.children[2].options = csv_files
