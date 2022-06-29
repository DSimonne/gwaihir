import ipywidgets as widgets


class TabReadme(widgets.VBox):
    """

    """

    def __init__(self, box_style=""):
        """

        """
        super(TabReadme, self).__init__()

        # Brief header describing the tab
        self.header = 'README'
        self.box_style = box_style

        # Create tab widgets
        self._list_widgets = widgets.VBox(
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
                style={'description_width': 'initial'})
        )

        # Create window
        self.window = self._list_widgets
