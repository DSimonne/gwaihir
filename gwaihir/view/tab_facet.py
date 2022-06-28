import ipywidgets as widgets
from ipywidgets import interact, Button, Layout, interactive
from IPython.display import display, Markdown, clear_output, Image


class TabFacet(widgets.Box):
    """

    """

    def __init__(self):
        """

        """
        super(TabFacet, self).__init__()

        self._list_widgets = widgets.VBox(
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

        # Create window
        self.window = self._list_widgets

        # Assign handlers
        self._list_widgets.children[1].observe(
            self.vtk_file_handler, names="value")

    # Define handlers
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
            self._list_widgets.children[2].options = vtk_files
