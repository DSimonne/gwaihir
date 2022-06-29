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

        # Brief header describing the tab
        self.header = 'README'

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
