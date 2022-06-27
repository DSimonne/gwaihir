import glob
import os

import ipywidgets as widgets
from ipywidgets import interact, Button, Layout, interactive
from IPython.display import display, Markdown, clear_output, Image

try:
    # This imports all necessary operators. GPU will be auto-selected
    from pynx.cdi import SupportUpdate, ScaleObj, AutoCorrelationSupport
    from pynx.cdi import InitPSF, ShowCDI, HIO, RAAR, ER, SupportTooLarge
    pynx_import_success = True
except ModuleNotFoundError:
    pynx_import_success = False
    print(
        "Could not load PyNX, the phase retrieval tab will be disabled."
        "\nMake sure you have the right version of PyNX installed."
    )


class Interface:
    """
    This class is a Graphical User Interface (GUI).

    It makes extensive use of the ipywidgets and is thus meant to be
    used with a jupyter notebook. Additional informations are provided
    in the "README" tab of the GUI.
    """

    def __init__(self, plot_tab_only=False):
        """
        All the widgets for the GUI are defined here. They are regrouped in
        a few tabs that design the GUI, the tabs are: tab_init tab_detector
        tab_setup tab_preprocess tab_data_frame tab_pynx tab_strain
        tab_data tab_facet tab_readme.

        Also defines:
            path_scripts: path to folder in which bcdi script are stored
            user_name: user_name used to login to slurm if working
                on the ESRF cluster

        :param plot_tab_only: True to only work with the plotting tab
        """
        super(Interface, self).__init__()

        # Initialize future attributes
        self.Dataset = None
        self.run_phase_retrieval = False
        self.run_pynx_tools = False
        self.text_file = None
        self.params = None
        self.Facets = None
        self.preprocessing_folder = None
        self.postprocessing_folder = None
        self.reconstruction_files = None
        self.strain_output_file = None

        if plot_tab_only:
            self.window = widgets.Tab(
                children=[
                    self.tab_data,
                ])
            self.window.set_title(0, 'Plot data')
        else:
            # Get path to scripts folder
            self.path_package = inspect.getfile(gwaihir).split("__")[0]
            self.path_scripts = self.path_package.split(
                "/lib/python")[0] + "/bin"
            print(
                f"Using `{self.path_scripts}`\n"
                "as absolute path to scripts containing folder.\n"
                # "This should be correct if gwaihir was installed in an environment.\n"
                # "Otherwise change self.path_scripts attribute to the correct folder.\n"
            )

            # Get user name
            try:
                self.user_name = getpass.getuser()

                print(
                    f"Login used for batch jobs: {self.user_name}\n"
                    # "If wrong login, please change self.user_name attribute"
                )
            except Exception as e:
                print(
                    "Could not get user name, please create self.user_name \
                    attribute for jobs"
                )
                raise e

            if pynx_import_success:
                self.window = widgets.Tab(
                    children=[
                        self.tab_init,
                        self.tab_detector,
                        self.tab_setup,
                        self.tab_preprocess,
                        self.tab_data_frame,
                        self.tab_pynx,
                        self.tab_strain,
                        self.tab_data,
                        self.tab_facet,
                        self.tab_readme,
                    ])
                self.window.set_title(0, 'Scan detail')
                self.window.set_title(1, 'Detector')
                self.window.set_title(2, 'Setup')
                self.window.set_title(3, "Preprocess")
                self.window.set_title(4, 'Logs')
                self.window.set_title(5, 'Phase retrieval')
                self.window.set_title(6, 'Postprocess')
                self.window.set_title(7, 'Handle data')
                self.window.set_title(8, 'Facets')
                self.window.set_title(9, 'Readme')

            elif not pynx_import_success:
                self.window = widgets.Tab(
                    children=[
                        self.tab_init,
                        self.tab_detector,
                        self.tab_setup,
                        self.tab_preprocess,
                        self.tab_data_frame,
                        self.tab_strain,
                        self.tab_data,
                        self.tab_facet,
                        self.tab_readme,
                    ])
                self.window.set_title(0, 'Scan detail')
                self.window.set_title(1, 'Detector')
                self.window.set_title(2, 'Setup')
                self.window.set_title(3, "Preprocess")
                self.window.set_title(4, 'Logs')
                self.window.set_title(5, 'Strain')
                self.window.set_title(6, 'Plot data')
                self.window.set_title(7, 'Facets')
                self.window.set_title(8, 'Readme')

        # Display the final window
        display(self.window)
