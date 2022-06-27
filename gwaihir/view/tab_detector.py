import ipywidgets as widgets
from ipywidgets import interact, Button, Layout, interactive
from IPython.display import display, Markdown, clear_output, Image


class TabDetector(widgets.Box):
    """

    """

    def __init__(self):
        """

        """
        super(TabDetector, self).__init__()

        self._list_widgets = widgets.VBox(
            unused_label_detector=widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>\
                Parameters related to the detector used</p>",
                style={
                    'description_width': 'initial'},
                layout=Layout(width='90%', height="35px")),

            detector=widgets.Dropdown(
                options=[
                    "Eiger2M", "Maxipix", "Eiger4M", "Merlin", "Timepix"],
                value="Merlin",
                description='Detector',
                continuous_update=False,
                disabled=True,
                style={'description_width': 'initial'}),

            roi_detector=widgets.Text(
                placeholder="""[low_y_bound, high_y_bound, low_x_bound, high_x_bound]""",
                description='Fix roi area, will overwrite cropping parameters',
                disabled=True,
                continuous_update=False,
                layout=Layout(
                    width='90%'),
                style={'description_width': 'initial'}),

            photon_threshold=widgets.IntText(
                value=0,
                description='Photon Threshold:',
                disabled=True,
                continuous_update=False,
                tooltip="data[data < photon_threshold] = 0",
                style={'description_width': 'initial'}),

            photon_filter=widgets.Dropdown(
                options=[
                    'loading', 'postprocessing'],
                value="loading",
                continuous_update=False,
                description='Photon filter',
                disabled=True,
                tooltip="When the photon threshold should be applied, if \
                'loading', it is applied before binning; if 'postprocessing',\
                 it is applied at the end of the script before saving",
                style={'description_width': 'initial'}),

            background_file=widgets.Text(
                value="",
                placeholder=f"{os.getcwd()}/background.npz'",
                description='Background file',
                disabled=True,
                continuous_update=False,
                layout=Layout(
                    width='90%'),
                style={'description_width': 'initial'}),

            flatfield_file=widgets.Text(
                value="",
                placeholder=f"{os.getcwd()}/flatfield_maxipix_8kev.npz",
                description='Flatfield file',
                disabled=True,
                continuous_update=False,
                layout=Layout(
                    width='90%'),
                style={'description_width': 'initial'}),

            hotpixels_file=widgets.Text(
                value="",
                placeholder=f"{os.getcwd()}/mask_merlin.npz",
                description='Hotpixels file',
                disabled=True,
                continuous_update=False,
                layout=Layout(
                    width='90%'),
                style={'description_width': 'initial'}),

            template_imagefile=widgets.Text(
                value='Pt_ascan_mu_%05d.nxs',
                description='Template imagefile',
                disabled=True,
                layout=Layout(
                    width='90%'),
                style={'description_width': 'initial'})
        )

        # Create window
        self.window = widgets.VBox([
            self._list_widgets.children[0],
            self._list_widgets.children[1],
            self._list_widgets.children[2],
            widgets.HBox(self._list_widgets.children[3:5]),
            self._list_widgets.children[5],
            self._list_widgets.children[6],
            self._list_widgets.children[7],
            self._list_widgets.children[8],
        ])
