import ipywidgets as widgets
import os
import glob


class TabFacet(widgets.VBox):
    """
    A custom ipywidgets tab for loading and analyzing VTK files for facet analysis.

    Attributes:
    -----------
    header: str
        Brief header describing the tab.
    box_style: str
        CSS style to apply to the tab container.
    unused_label_facet: widgets.HTML
        HTML widget containing a description of the VTK loading process.
    parent_folder: widgets.Dropdown
        Dropdown widget for selecting the parent folder.
    vtk_file: widgets.Dropdown
        Dropdown widget for selecting a specific VTK file.
    load_data: widgets.ToggleButtons
        Toggle button for loading VTK data or clearing/reloading the folder list.
    children: Tuple[Widget]
        Tuple of child widgets to display in the tab.

    Methods:
    --------
    __init__(self, box_style: str = "") -> None:
        Initializes a new instance of the TabFacet class.

    vtk_file_handler(self, change: dict) -> None:
        Event handler for updating the list of VTK files when the parent folder changes.

    __str__(self) -> str:
        Returns a string representation of the TabFacet instance.
    """

    def __init__(self, box_style=""):
        """

        """
        super(TabFacet, self).__init__()

        # Brief header describing the tab
        self.header = 'Facet analysis'
        self.box_style = box_style

        # Define widgets
        self.unused_label_facet = widgets.HTML(
            description="<p style='font-weight: bold;font-size:1.2em'>\
             Extract facet specific data from vtk file",
            style={
                'description_width': 'initial'},
            layout=widgets.Layout(width='90%', height="35px"))

        self.parent_folder = widgets.Dropdown(
            options=[x[0] + "/" for x in os.walk(os.getcwd())],
            value=os.getcwd() + "/",
            placeholder=os.getcwd() + "/",
            description='Parent folder:',
            continuous_update=False,
            layout=widgets.Layout(width='90%'),
            style={'description_width': 'initial'})

        self.vtk_file = widgets.Dropdown(
            options=sorted(glob.glob(os.getcwd()+"*.vtk"),
                           key=os.path.getmtime),
            description='vtk file in subdirectories:',
            layout=widgets.Layout(width='90%'),
            style={'description_width': 'initial'})

        self.load_data = widgets.ToggleButtons(
            options=[
                ("Clear/ Reload folder", False),
                ('Load .vtk file', "load_csv"),
            ],
            value=False,
            description='Load vtk data',
            button_style='',
            layout=widgets.Layout(width='90%'),
            style={'description_width': 'initial'})

        # Define children
        self.children = (
            self.unused_label_facet,
            self.parent_folder,
            self.vtk_file,
            self.load_data,
        )

        # Assign handlers
        self.parent_folder.observe(
            self.vtk_file_handler, names="value")

    # Define handlers
    def vtk_file_handler(self, change):
        """List all .vtk files in change subdirectories"""
        vtk_files = []
        if hasattr(change, "new"):
            change = change.new

        for x in os.walk(change):
            vtk_files += sorted(glob.glob(f"{x[0]}/*.vtk"),
                                key=os.path.getmtime)

        self.vtk_file.options = vtk_files

    def __str__(self):
        return "Facet tab"
