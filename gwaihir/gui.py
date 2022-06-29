import glob
import os
import inspect
import getpass

import ipywidgets as widgets
from ipywidgets import interact, Button, Layout, interactive
from IPython.display import display, Markdown, clear_output, Image

# Try to import all necessary operators for PyNX to see if it is available
# GPU will be auto-selected
try:
    from pynx.cdi import SupportUpdate, ScaleObj, AutoCorrelationSupport
    from pynx.cdi import InitPSF, ShowCDI, HIO, RAAR, ER, SupportTooLarge
    pynx_import_success = True
except ModuleNotFoundError:
    pynx_import_success = False
    print(
        "Could not load PyNX."
        "\nThe phase retrieval tab will be disabled."
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
        The different tabs of the GUI are loaded from the submodule view.
        They are then laid out side by side by using the ipywidgets.Tabs()
         method.
        The currently supported tabs are:
            - TabStartup
            - TabDetector
            - TabInstrument
            - TabPreprocess
            - TabDataFrame
            - TabPhaseRetrieval
            - TabPostprocess
            - TabPlotData
            - TabFacet
            - TabReadme

        Here is also defined:
            path_scripts: path to folder in which bcdi and pynx scripts are
                stored
            user_name: user_name used to login to slurm if working on the ESRF
                cluster

        :param plot_tab_only: True to only work with the plotting tab
        """
        super(Interface, self).__init__()

        # Initialize future attributes
        self.Dataset = None
        self.text_file = None
        self.params = None
        self.Facets = None
        self.preprocessing_folder = None
        self.postprocessing_folder = None
        self.reconstruction_files = None
        self.strain_output_file = None

        # Display only the plot tab
        if plot_tab_only:
            self.window = widgets.Tab(
                children=[
                    TabPlotData(),
                ])

        # Display all the tabs
        else:
            # Get path to scripts folder
            path_package = inspect.getfile(gwaihir).split("__")[0]
            self.path_scripts = path_package.split(
                "/lib/python")[0] + "/bin"
            print(
                f"Using scripts contained in '{self.path_scripts}'"
            )

            # Get user name
            try:
                self.user_name = getpass.getuser()

                print(
                    f"Login used for batch jobs: {self.user_name}"
                )
            except Exception as e:
                self.user_name = None

                print(
                    "Could not get user name."
                    "\nPlease create self.user_name attribute for jobs"
                )
                raise e

            if pynx_import_success:
                self.window = widgets.Tab(
                    children=[
                        TabStartup(),
                        TabDetector(),
                        TabInstrument(),
                        TabPreprocess(),
                        TabDataFrame(),
                        TabPhaseRetrieval(),
                        TabPostprocess(),
                        TabPlotData(),
                        TabFacet(),
                        TabReadme(),
                    ])

            elif not pynx_import_success:
                self.window = widgets.Tab(
                    children=[
                        TabStartup(),
                        TabDetector(),
                        TabInstrument(),
                        TabPreprocess(),
                        TabDataFrame(),
                        TabPostprocess(),
                        TabPlotData(),
                        TabFacet(),
                        TabReadme(),
                    ])

        # Set tab names
        for j, Tab in enumerate(self.window.children)
            self.window.set_title(j, Tab.header)

        # Display the final window
        display(self.window)
