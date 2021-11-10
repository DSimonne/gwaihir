from tornado.ioloop import PeriodicCallback
from skimage.measure import marching_cubes
from scipy.spatial.transform import Rotation
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import os
import h5py as h5
import tables as tb
import glob

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import ipywidgets as widgets
from ipywidgets import interact, Button, Layout, interactive, fixed
from IPython.display import Markdown, Latex, clear_output
from IPython.core.display import display, HTML
# import ipyfilechooser
import ipyvolume as ipv

import warnings
warnings.filterwarnings("ignore")


# Classes

class Plotter():
    """Class based on interactive functions for plotting.
    Gets data array from file and plot if specified
    """

    def __init__(
        self,
        filename=None,
        data_array=None,
        plot=False,
        log=False,
        cmap="YlGnBu_r",
        figsize=(10, 10),
        fontsize=15
    ):
        """Either directly loads a data array from 'data_array' or extracts a
        data array from 'filename'. Plots the data according to the value of
        'plot'.

        :param filename: path to data, supported files extensions are .cxi,
         .npy or .npz
        :param data_array: a np.ndarray
        :param plot: either '2D', 'slices', '3D' or False
        :param log: True to have a logarithmic scale
         False to have a linear scale
        :param cmap: matplotlib cmap used for plot, default 'YlGnBu_r'
         Other possible values are 'Cool', 'Gray', 'Gray_r', 'Hot', 'Hsv',
         'Inferno', 'Jet', 'Plasma', 'Rainbow', 'Viridis'
        :param figsize: default (10, 10)
        :param fontsize: default 15
        """
        self.filename = filename
        self.data_array = data_array
        self.plot = plot
        self.log = log
        self.cmap = cmap
        self.figsize = figsize
        self.fontsize = fontsize

        # Future attributes
        self.interact_scale = False

        # Get data array from any of the supported files
        if isinstance(filename, str) and not data_array:
            self.get_data_array()

        elif isinstance(filename, np.ndarray) and not filename:
            # Plot data
            if self.plot == "2D":
                self.plot_data(figsize=self.figsize,
                               log=self.log, cmap=self.cmap)

            elif self.plot == "slices" and self.data_array.ndim == 3:
                self.plot_3d_slices(figsize=None, log=self.log, cmap=self.cmap)

            elif self.plot == "3D" and self.data_array.ndim == 3:
                ThreeDViewer(self.data_array)

            else:
                print(
                    "#########################################################"
                    "########################################################\n"
                    f"Loaded data array\n"
                    f"\tNb of dimensions: {self.data_array.ndim}\n"
                    f"\tShape: {self.data_array.shape}\n"
                    "\t'plot' keyword accepted values: ('2D', 'slices', 3D')"
                    "#########################################################"
                    "########################################################"
                )

        else:
            print("Please provide either a filename or directly an np.ndarray.")

    def get_data_array(self):
        """Get numpy array from file.

        :param plot: either '2D', 'slices', '3D' or False
        """
        # No need to select data array interactively
        if self.filename.endswith((".npy", ".h5", ".cxi")):
            if self.filename.endswith(".npy"):
                try:
                    self.data_array = np.load(self.filename)

                except ValueError:
                    print("Could not load data ... ")

            elif self.filename.endswith(".cxi"):
                try:
                    self.data_array = h5.File(self.filename, mode='r')[
                        'entry_1/data_1/data'][()]

                except (KeyError, OSError):
                    print("""
                        The file could not be loaded, verify that you are
                        loading a file with an hdf5 architecture (.nxs, .cxi,
                        .h5, ...) and that the file exists. Otherwise, verify
                        that the data is saved in f.root.entry_1.data_1.data[:],
                        as it should be following csi conventions.
                        """)

            elif self.filename.endswith(".h5"):
                try:
                    self.data_array = h5.File(self.filename, mode='r')[
                        'entry_1/data_1/data'][()]
                    if self.data_array.ndim == 4:
                        self.data_array = self.data_array[0]
                    # Due to labelling of axes x,y,z and not z,y,x
                    self.data_array = np.swapaxes(self.data_array, 0, 2)

                except (KeyError, OSError):
                    raise KeyError("""
                        The file could not be loaded, verify that you are\
                        loading a file with an hdf5 architecture (.nxs, .cxi,\
                        .h5, ...) and that the file exists. Otherwise, verify\
                        that the data is saved in f.root.entry_1.data_1.data[:],\
                        as it should be following csi conventions.\
                        """)

            # Plot data
            if self.plot == "2D":
                self.plot_data(figsize=self.figsize,
                               log=self.log, cmap=self.cmap)

            elif self.plot == "slices" and self.data_array.ndim == 3:
                self.plot_3d_slices(figsize=None, log=self.log, cmap=self.cmap)

            elif self.plot == "3D" and self.data_array.ndim == 3:
                ThreeDViewer(self.data_array)

            else:
                print(
                    "#########################################################"
                    "########################################################\n"
                    f"Loaded data array from {self.filename}\n"
                    f"\tNb of dimensions: {self.data_array.ndim}\n"
                    f"\tShape: {self.data_array.shape}\n"
                    "\t'plot' keyword accepted values: ('2D', 'slices', 3D')"
                    "#########################################################"
                    "########################################################"
                )

        # Need to select data array interactively
        elif self.filename.endswith(".npz"):
            # Open npz file and allow the user to pick an array
            try:
                rawdata = np.load(self.filename)

                @interact(
                    file=widgets.Dropdown(
                        options=rawdata.files,
                        value=rawdata.files[0],
                        description='Pick an array to load:',
                        disabled=False,
                        style={'description_width': 'initial'}))
                def open_npz(file):
                    # Pick an array
                    self.data_array = rawdata[file]

                    # Plot data
                    if self.plot == "2D":
                        self.plot_data(
                            figsize=self.figsize, log=self.log, cmap=self.cmap)

                    elif self.plot == "slices" and self.data_array.ndim == 3:
                        self.plot_3d_slices(
                            figsize=None, log=self.log, cmap=self.cmap)

                    elif self.plot == "3D" and self.data_array.ndim == 3:
                        ThreeDViewer(self.data_array)

                    else:
                        print(
                            "###########################################\
                            # \
                            # \n"
                            f"Loaded data array from {self.filename}\n"
                            f"\tNb of dimensions: {self.data_array.ndim}\n"
                            f"\tShape: {self.data_array.shape}\n"
                            "############################################\
                            # \
                            # "
                        )

            except ValueError:
                print("Could not load data.")

        else:
            print("Data type not supported.")

    def plot_data(self, **kwargs):
        """Run plot_data function with class arguments."""
        for k, v in kwargs.items():
            setattr(self, k, v)

        plot_data(
            data_array=self.data_array,
            figsize=self.figsize,
            fontsize=self.fontsize,
            cmap=self.cmap
        )

    def plot_3d_slices(self, **kwargs):
        """Run plot_3d_slices function with class arguments."""
        for k, v in kwargs.items():
            setattr(self, k, v)

        plot_3d_slices(
            data_array=self.data_array,
            figsize=self.figsize,
            log=self.log,
            cmap=self.cmap
        )


class ThreeDViewer(widgets.Box):
    """Widget to display 3D objects from CDI optimisation, loaded from a result
    CXI file or a mode file.

    This is a quick & dirty implementation but should be useful.
    Quickly adapted from @Vincent Favre Nicolin (ESRF)
    """

    def __init__(self, input_file=None, html_width=None):
        """

        :param input_file: the data filename or directly the 3D data array.
        :param html_width: html width in %. If given, the width of the
         notebook will be changed to that value (e.g. full width with 100)
        """
        super(ThreeDViewer, self).__init__()

        if html_width is not None:
            display(
                HTML("<style>.container { width:%d%% !important; }\
                    </style>" % int(html_width)
                     )
            )

        # focus_label = widgets.Label(value='Focal distance (cm):')
        self.threshold = widgets.FloatSlider(
            value=5, min=0, max=20, step=0.02, description='Contour.',
            disabled=False, continuous_update=False, orientation='horizontal',
            readout=True, readout_format='.01f')
        self.toggle_phase = widgets.ToggleButtons(
            options=['Abs', 'Phase'], description='',
            disabled=False, value='Phase', button_style='')
        self.toggle_rotate = widgets.ToggleButton(
            value=False, description='Rotate', tooltips='Rotate')
        self.pcb_rotate = None
        hbox1 = widgets.HBox([self.toggle_phase, self.toggle_rotate])

        self.toggle_dark = widgets.ToggleButton(
            value=False, description='Dark', tooltips='Dark/Light theme')
        self.toggle_box = widgets.ToggleButton(
            value=True, description='Box', tooltips='Box ?')
        self.toggle_axes = widgets.ToggleButton(
            value=True, description='Axes', tooltips='Axes ?')
        hbox_toggle = widgets.HBox(
            [self.toggle_dark, self.toggle_box, self.toggle_axes])

        self.colormap = widgets.Dropdown(
            options=['Cool', 'Gray', 'Gray_r', 'Hot', 'Hsv',
                     'Inferno', 'Jet', 'Plasma', 'Rainbow', 'Viridis'],
            value='Jet', description='Colors:', disabled=True)
        self.colormap_range = widgets.FloatRangeSlider(value=[20, 80],
                                                       min=0,
                                                       max=100,
                                                       step=1,
                                                       description='Range:',
                                                       disabled=False,
                                                       continuous_update=False,
                                                       orientation='horizontal',
                                                       readout=True,
                                                       # readout_format='.1f'
                                                       )
        self.toggle_plane = widgets.ToggleButton(
            value=False, description='Cut planes', tooltips='Cut plane')
        self.plane_text = widgets.Text(
            value="", description="", tooltips='Plane equation')
        hbox_plane = widgets.HBox([self.toggle_plane, self.plane_text])

        self.clipx = widgets.FloatSlider(
            value=1, min=-1, max=1, step=0.1, description='Plane Ux',
            disabled=False, continuous_update=False, orientation='horizontal',
            eadout=True, readout_format='.01f')
        self.clipy = widgets.FloatSlider(
            value=1, min=-1, max=1, step=0.1, description='Plane Uy',
            disabled=False, continuous_update=False, orientation='horizontal',
            readout=True, readout_format='.01f')
        self.clipz = widgets.FloatSlider(
            value=1, min=-1, max=1, step=0.1, description='Plane Uz',
            disabled=False, continuous_update=False, orientation='horizontal',
            readout=True, readout_format='.01f')
        self.clipdist = widgets.FloatRangeSlider(
            value=[0, 100], min=0, max=100, step=0.5, description='Planes dist',
            disabled=False, continuous_update=False, orientation='horizontal',
            readout=True, readout_format='.1f')

        # self.toggle_mode = widgets.ToggleButtons(options=['Volume','X','Y','Z'])
        self.progress = widgets.IntProgress(
            value=10, min=0, max=10, description='Processing:',
            bar_style='', style={'bar_color': 'green'}, orientation='horizontal')

        # Set observers
        self.threshold.observe(self.on_update_plot)
        self.toggle_phase.observe(self.on_change_type)
        self.colormap.observe(self.on_update_plot)
        self.colormap_range.observe(self.on_update_plot)
        self.clipx.observe(self.on_update_plot)
        self.clipy.observe(self.on_update_plot)
        self.clipz.observe(self.on_update_plot)
        self.clipdist.observe(self.on_update_plot)
        self.toggle_plane.observe(self.on_update_plot)

        self.toggle_dark.observe(self.on_update_style)
        self.toggle_box.observe(self.on_update_style)
        self.toggle_axes.observe(self.on_update_style)

        self.toggle_rotate.observe(self.on_animate)

        # Future attributes
        # self.mesh = None
        # self.color = None
        # self.d0 = None
        # self.progress.value = None

        # Create final box
        self.vbox = widgets.VBox([self.threshold,
                                  hbox1, hbox_toggle,
                                  self.colormap, self.colormap_range,
                                  hbox_plane,
                                  self.clipx, self.clipy, self.clipz,
                                  self.clipdist,
                                  self.progress,
                                  # self.fc
                                  ])

        # Load data
        if isinstance(input_file, np.ndarray):
            data_array = input_file

            self.output_view = widgets.Output()
            with self.output_view:
                self.fig = ipv.figure(
                    width=900, height=600, controls_light=True)
                # if input_file is not None:
                #     if isinstance(input_file, str):
                #         if os.path.isfile(input_file):
                #             self.change_file(input_file)
                #     elif isinstance(input_file, np.ndarray):
                self.set_data(d=data_array)
                display(self.fig)

            self.window = widgets.HBox([self.output_view, self.vbox])

            display(self.window)

        else:
            print("Could not load data")

    def on_update_plot(self, v=None):
        """Update the plot according to parameters. The points are re-computed.

        :param k: ignored
        :return:
        """
        if v is not None and v['name'] != 'value':
            return
        self.progress.value = 7

        # See https://github.com/maartenbreddels/ipyvolume/issues/174
        # to support using normals

        # Unobserve as we disable/enable buttons and that triggers events
        try:
            self.clipx.unobserve(self.on_update_plot)
            self.clipy.unobserve(self.on_update_plot)
            self.clipz.unobserve(self.on_update_plot)
            self.clipdist.unobserve(self.on_update_plot)
        except:
            pass

        if self.toggle_plane.value:
            self.clipx.disabled = False
            self.clipy.disabled = False
            self.clipz.disabled = False
            self.clipdist.disabled = False
            # Cut volume with clipping plane
            uz, uy, ux = self.clipz.value, self.clipy.value, self.clipx.value
            u = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2)
            if np.isclose(u, 0):
                ux = 1
                u = 1

            nz, ny, nx = self.d.shape
            z, y, x = np.meshgrid(np.arange(nz), np.arange(
                ny), np.arange(nx), indexing='ij')

            # Compute maximum range of clip planes & fix dist range
            tmpz, tmpy, tmpx = np.where(abs(self.d) >= self.threshold.value)
            tmp = (tmpx * ux + tmpy * uy + tmpz * uz) / u
            tmpmin, tmpmax = tmp.min() - 1, tmp.max() + 1
            if tmpmax > self.clipdist.min:  # will throw an exception if min>max
                self.clipdist.max = tmpmax
                self.clipdist.min = tmpmin
            else:
                self.clipdist.min = tmpmin
                self.clipdist.max = tmpmax

            # Compute clipping mask
            c = ((x * ux + y * uy + z * uz) / u > self.clipdist.value[0]) * (
                ((x * ux + y * uy + z * uz) / u < self.clipdist.value[1]))
            self.plane_text.value = "%6.1f < (%4.2f*x %+4.2f*y %+4.2f*z) < %6.1f" % (
                self.clipdist.value[0], ux / u, uy / u,
                uz / u, self.clipdist.value[1])
        else:
            self.clipx.disabled = True
            self.clipy.disabled = True
            self.clipz.disabled = True
            self.clipdist.disabled = True
            self.plane_text.value = ""
            c = 1
        try:
            verts, faces, _, _ = marching_cubes(
                abs(self.d) * c, level=self.threshold.value, step_size=1)
            vals = self.rgi(verts)
            if self.toggle_phase.value == "Phase":
                self.colormap.disabled = True
                rgba = complex2rgbalin(vals)
                color = rgba[..., :3] / 256
            elif self.toggle_phase.value in ['Abs', 'log10(Abs)']:
                self.colormap.disabled = False
                cs = cm.ScalarMappable(
                    norm=Normalize(
                        vmin=self.colormap_range.value[0],
                        vmax=self.colormap_range.value[1]),
                    cmap=self.colormap.value.lower())
                color = cs.to_rgba(abs(vals))[..., :3]
            else:
                # TODO: Gradient
                gx, gy, gz = self.rgi_gx(verts), self.rgi_gy(
                    verts), self.rgi_gz(verts)
                color = np.empty((len(vals), 3), dtype=np.float32)
                color[:, 0] = abs(gx)
                color[:, 1] = abs(gy)
                color[:, 2] = abs(gz)
                color *= 100
                self.color = color
            x, y, z = verts.T
            self.mesh = ipv.plot_trisurf(x, y, z, triangles=faces, color=color)
            self.fig.meshes = [self.mesh]
        # Keep general exception for debugging purposes
        except Exception as ex:
            print(ex)

        try:
            self.clipx.observe(self.on_update_plot)
            self.clipy.observe(self.on_update_plot)
            self.clipz.observe(self.on_update_plot)
            self.clipdist.observe(self.on_update_plot)
        except:
            pass
        self.progress.value = 10

    def on_update_style(self, v):
        """
        Update the plot style - for all parameters which
        do not involved recomputing
        the displayed object.
        :param k: ignored
        :return:
        """
        if v['name'] == 'value':
            if self.toggle_dark.value:
                ipv.pylab.style.set_style_dark()
            else:
                ipv.pylab.style.set_style_light()
                # Fix label colours (see self.fig.style)
                ipv.pylab.style.use(
                    {'axes': {'label': {'color': 'black'},
                              'ticklabel': {'color': 'black'}}})
            if self.toggle_box.value:
                ipv.pylab.style.box_on()
            else:
                ipv.pylab.style.box_off()
            if self.toggle_axes.value:
                ipv.pylab.style.axes_on()
            else:
                ipv.pylab.style.axes_off()

    # def on_select_file(self, v):
    #     """
    #     Called when a file selection has been done
    #     :param v:
    #     :return:
    #     """
    #     self.change_file(self.fc.selected)

    # def change_file(self, file_name):
    #     """
    #     Function used to load data from a new file
    #     :param file_name: the file where the object data is loaded,
    #      either a CXI or modes h5 file
    #     :return:
    #     """
    #     self.progress.value = 3
    #     print('Loading:', file_name)

    #     try:
    #         self.toggle_plane.unobserve(self.on_update_plot)
    #         self.toggle_plane.value = False
    #         self.toggle_plane.observe(self.on_update_plot)
    #         d = h5.File(file_name, mode='r')['entry_1/data_1/data'][()]
    #         if d.ndim == 4:
    #             d = d[0]
    #         d = np.swapaxes(d, 0, 2)  # Due to labelling of axes x,y,z and not z,y,x
    #         if 'log' in self.toggle_phase.value:
    #             self.d0 = d
    #             d = np.log10(np.maximum(0.1, abs(d)))
    #         self.set_data(d)
    #     except:
    #         print("Failed to load file - is this a \
    #               result CXI result or a modes file from a 3D CDI analysis ?")

    def on_change_type(self, v):
        """Change scale"""
        if v['name'] == 'value':
            if isinstance(v['old'], str):
                newv = v['new']
                oldv = v['old']
                if 'log' in oldv and 'log' not in newv:
                    d = self.d0
                    self.set_data(d, threshold=10 ** self.threshold.value)
                elif 'log' in newv and 'log' not in oldv:
                    self.d0 = self.d
                    d = np.log10(np.maximum(0.1, abs(self.d0)))
                    self.set_data(d, threshold=np.log10(self.threshold.value))
                    return
            self.on_update_plot()

    def set_data(self, d, threshold=None):
        """Check if data is complex or not"""
        self.progress.value = 5
        self.d = d
        self.toggle_phase.unobserve(self.on_change_type)
        if np.iscomplexobj(d):
            if self.toggle_phase.value == 'log10(Abs)':
                self.toggle_phase.value = 'Abs'
            self.toggle_phase.options = ('Abs', 'Phase')
        else:
            if self.toggle_phase.value == 'Phase':
                self.toggle_phase.value = 'Abs'
            self.toggle_phase.options = ('Abs', 'log10(Abs)')
        self.toggle_phase.observe(self.on_change_type)

        self.threshold.unobserve(self.on_update_plot)
        self.colormap_range.unobserve(self.on_update_plot)
        self.threshold.max = abs(self.d).max()
        if threshold is None:
            self.threshold.value = self.threshold.max / 2
        else:
            self.threshold.value = threshold
        self.colormap_range.max = abs(self.d).max()
        self.colormap_range.value = [0, abs(self.d).max()]
        self.threshold.observe(self.on_update_plot)
        self.colormap_range.observe(self.on_update_plot)

        # print(abs(self.d).max(), self.threshold.value)
        nz, ny, nx = self.d.shape
        z, y, x = np.arange(nz), np.arange(ny), np.arange(nx)
        # Interpolate probe to object grid
        self.rgi = RegularGridInterpolator(
            (z, y, x), self.d, method='linear',
            bounds_error=False, fill_value=0)

        # Also prepare the phase gradient
        gz, gy, gx = np.gradient(self.d)
        a = np.maximum(abs(self.d), 1e-6)
        ph = self.d / a
        gaz, gay, gax = np.gradient(a)
        self.rgi_gx = RegularGridInterpolator(
            (z, y, x), ((gx - gax * ph) / (ph * a)).real,
            method='linear', bounds_error=False, fill_value=0)
        self.rgi_gy = RegularGridInterpolator(
            (z, y, x), ((gy - gay * ph) / (ph * a)).real,
            method='linear', bounds_error=False, fill_value=0)
        self.rgi_gz = RegularGridInterpolator(
            (z, y, x), ((gz - gaz * ph) / (ph * a)).real,
            method='linear', bounds_error=False, fill_value=0)

        # Fix extent
        ipv.pylab.xlim(0, max(self.d.shape))
        ipv.pylab.ylim(0, max(self.d.shape))
        ipv.pylab.zlim(0, max(self.d.shape))
        ipv.squarelim()
        self.on_update_plot()

    def on_animate(self):
        """Trigger the animation (rotation around vertical axis)

        :param v:
        :return:
        """
        if self.pcb_rotate is None:
            self.pcb_rotate = PeriodicCallback(self.callback_rotate, 50.)
        if self.toggle_rotate.value:
            self.pcb_rotate.start()
        else:
            self.pcb_rotate.stop()

    def callback_rotate(self):
        """Used for periodic rotation."""
        # ipv.view() only supports a rotation against
        # the starting azimuth and elevation
        # ipv.view(azimuth=ipv.view()[0]+1)

        # Use a quaternion and the camera's 'up' as rotation axis
        x, y, z = self.fig.camera.up
        n = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        a = np.deg2rad(2.5) / 2  # angular step
        sa, ca = np.sin(a / 2) / n, np.cos(a / 2)
        r = Rotation.from_quat((sa * x, sa * y, sa * z, ca))
        self.fig.camera.position = tuple(r.apply(self.fig.camera.position))


# Methods

def plot_data(
    data_array,
    figsize=(10, 10),
    fontsize=15,
    log="interact",
    cmap="YlGnBu_r"
):
    """Create figure based on the data dimensions.

    :param data_array: np.ndarray to plot
    :param figsize: default (10, 10)
    :param fontsize: default 15
    :param log: True, False or "interact"
    :param cmap: matplotlib cmap used for plot, default 'YlGnBu_r'
    """
    # Get dimensions
    data_dimensions = data_array.ndim

    if data_dimensions == 1:
        plt.close()
        fig, ax = plt.subplots(figsize=figsize)
        if log:
            ax.plot(np.log(data_array))
            plt.show()
        elif log is False:
            ax.plot(data_array)
            plt.show()
        elif log == "interact":
            @interact(
                scale=widgets.ToggleButtons(
                    options=["linear", "logarithmic"],
                    value="linear",
                    description='Scale',
                    disabled=False,
                    style={'description_width': 'initial'}),
                figsize=fixed(figsize)
            )
            def plot_with_interactive_scale(scale, figsize):
                # Create figure
                if not figsize:
                    figsize = (data_array.ndim*5, 7)
                    print("Figure size defaulted to", figsize)

                _, ax = plt.subplots(figsize=figsize)

                # Get scale
                log = scale == "logarithmic"

                if log:
                    ax.plot(np.log(data_array))
                else:
                    ax.plot(data_array)
                plt.show()

    elif data_dimensions == 2:
        if isinstance(log, bool):
            plot_2d_image(data_array, log=log, cmap=cmap)
        elif log == "interact":
            @interact(
                scale=widgets.ToggleButtons(
                    options=["linear", "logarithmic"],
                    value="linear",
                    description='Scale',
                    disabled=False,
                    style={'description_width': 'initial'}),
                figsize=fixed(figsize)
            )
            def plot_with_interactive_scale(scale, figsize):
                # Create figure
                if not figsize:
                    figsize = (10, 10)
                    print("Figure size defaulted to", figsize)

                fig, ax = plt.subplots(figsize=figsize)

                # Get scale
                log = scale == "logarithmic"

                # Plot
                plot_2d_image(data_array, log=log, fig=fig, ax=ax, cmap=cmap)

                # Show figure
                plt.show()

    elif data_dimensions == 3:
        @interact(
            axplot=widgets.Dropdown(
                options=["xy", "yz", "xz"],
                value="xy",
                description='First 2 axes:',
                disabled=False,
                style={'description_width': 'initial'}),
            ComplexNumber=widgets.ToggleButtons(
                options=["Real", "Imaginary", "Module", "Phase"],
                value="Module",
                description='Plotting options',
                disabled=False,
                tooltip=['Plot only contour or not', "", ""])
        )
        def plot_3d(
            axplot,
            ComplexNumber
        ):

            # Decide what we want to plot
            if ComplexNumber == "Real":
                data = np.real(data_array)
            elif ComplexNumber == "Imaginary":
                data = np.imag(data_array)
            elif ComplexNumber == "Module":
                data = np.abs(data_array)
            elif ComplexNumber == "Phase":
                data = np.angle(data_array)

            # Take the shape of that array along 2 axis
            if axplot == "xy":
                print(
                    f"The shape of this projection \
                    is {np.shape(data[:, :, 0])}")

                r = np.shape(data[0, 0, :])
                print(f"The range in the last axis is [0, {r[0]}]")

            elif axplot == "yz":
                print(
                    f"The shape of this projection \
                    is {np.shape(data[0, :, :])}")

                r = np.shape(data[:, 0, 0])
                print(f"The range in the last axis is [0, {r[0]}]")

            elif axplot == "xz":
                print(
                    f"The shape of this projection \
                    is {np.shape(data[:, 0, :])}")

                r = np.shape(data[0, :, 0])
                print(f"The range in the last axis is [0, {r[0]}]")

            @interact(
                i=widgets.IntSlider(
                    min=0,
                    max=r[0]-1,
                    step=1,
                    description='Index along last axis:',
                    disabled=False,
                    orientation='horizontal',
                    continuous_update=False,
                    readout=True,
                    readout_format='d',
                ),
                # PlottingOptions=widgets.ToggleButtons(
                #     options=[("2D image", "2D"),
                #              ("2D image with contour", "2DC"),
                #              # ("3D surface plot", "3D")
                #              ],
                #     value="2D",
                #     description='Plotting options',
                #     disabled=False,
                #     button_style='',
                #     tooltip=['Plot only contour or not', "", ""],
                #     # icon='check'
                # ),
                scale=widgets.ToggleButtons(
                    options=["linear", "logarithmic"],
                    value="linear",
                    description='Scale',
                    disabled=False,
                    style={'description_width': 'initial'}),
            )
            def PickLastAxis(i,
                             # PlottingOptions,
                             scale
                             ):
                if axplot == "xy":
                    dt = data[:, :, i]
                elif axplot == "yz":
                    dt = data[i, :, :]
                elif axplot == "xz":
                    dt = data[:, i, :]

                else:
                    raise TypeError("Choose xy, yz or xz as axplot.")

                # Create figure
                plt.close()
                print(figsize)
                fig, ax = plt.subplots(1, 1, figsize=figsize)

                # Get scale
                log = scale == "logarithmic"

                # Plot 2D image in interactive environment
                plot_2d_image(two_d_array=dt, log=log,
                              fig=fig, ax=ax, cmap=cmap)
                plt.show()

                # if PlottingOptions == "2D":
                # elif PlottingOptions == "2DC":
                #     # Show contour plot instead

                #     plt.close()

                #     log = True if scale == "logarithmic"  else False
                #     plot_2d_image_contour(two_d_array=dt, log=log)

                #     plt.show()


def plot_2d_image(two_d_array, fig=None, ax=None, log=False, cmap="YlGnBu_r"):
    """Plot 2d image from 2d array.

    :param two_d_array: np.ndarray to plot, must be 2D
    :param fig: plt.figure to plot in, default is None and
     will create a figure
    :param ax: axes of figure, default is None and will create axes
    :param log: True to have a logarithmic scale, False to have a linear scale
    :param cmap: matplotlib cmap used for plot, default 'YlGnBu_r'
    """
    # Find max and min
    # dmax = two_d_array.max()
    # dmin = two_d_array.min()

    if not fig and not ax:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    scale = "logarithmic" if log else "linear"

    try:
        ax.imshow(
            two_d_array,
            norm={"linear": None, "logarithmic": LogNorm()}[
                scale],
            cmap=cmap,
            # cmap="cividis",
            # extent=(0, 2, 0, 2),
            # vmin=dmin,
            # vmax=dmax,
        )

        # Create axis for colorbar
        # cbar_ax = make_axes_locatable(ax).append_axes(
        #     position='right', size='5%', pad=0.1)

        # Create colorbar
        # cbar = fig.colorbar(mappable=img, cax=cbar_ax)

    except TypeError:
        # plt.close()
        print("Using complex data, automatically switching to array module")

        ax.imshow(
            np.abs(two_d_array),
            norm={"linear": None, "logarithmic": LogNorm()}[
                scale],
            cmap=cmap,
            # cmap="cividis",
            # extent=(0, 2, 0, 2),
            # vmin=dmin,
            # vmax=dmax,
        )

        # # Create axis for colorbar
        # cbar_ax = make_axes_locatable(ax).append_axes(
        #     position='right', size='5%', pad=0.1)

        # Create colorbar
        # cbar = fig.colorbar(mappable=img, cax=cbar_ax)

    except ValueError:
        plt.close()
        if scale == "logarithmic":
            print("Log scale can not handle this kind of data ...")
        else:
            pass


def plot_3d_slices(data_array, figsize=None, log=False, cmap="YlGnBu_r"):
    """Create figure for 3d data.

    :param data_array: np.ndarray to plotn must be 3D
    :param figsize: default (10, 10)
    :param log: boolean (True, False) or anything else which
     raises an interactive window
    :param cmap: matplotlib cmap used for plot, default 'YlGnBu_r'
    """
    if isinstance(log, bool):
        # Create figure
        if not figsize:
            figsize = (data_array.ndim*5, 7)
            print("Figure size defaulted to", figsize)

        fig, axs = plt.subplots(
            1, data_array.ndim, figsize=figsize)

        # Each axis has a dimension
        shape = data_array.shape

        two_d_array = data_array[shape[0]//2, :, :]
        plot_2d_image(two_d_array, fig=fig, ax=axs[0], log=log, cmap=cmap)

        two_d_array = data_array[:, shape[1]//2, :]
        plot_2d_image(two_d_array, fig=fig, ax=axs[1], log=log, cmap=cmap)

        two_d_array = data_array[:, :, shape[2]//2]
        plot_2d_image(two_d_array, fig=fig, ax=axs[2], log=log, cmap=cmap)

        # Show figure
        plt.show()

    else:
        @interact(
            scale=widgets.ToggleButtons(
                options=["linear", "logarithmic"],
                value="linear",
                description='Scale',
                disabled=False,
                style={'description_width': 'initial'}),
            figsize=fixed(figsize)
        )
        def plot_with_interactive_scale(scale, figsize):
            # Create figure
            if not figsize:
                figsize = (data_array.ndim*5, 7)
                print("Figure size defaulted to", figsize)

            fig, axs = plt.subplots(
                1, data_array.ndim, figsize=figsize)

            # Each axis has a dimension
            shape = data_array.shape

            # Get scale
            log = scale == "logarithmic"

            try:
                two_d_array = data_array[shape[0]//2, :, :]
                plot_2d_image(two_d_array, fig=fig,
                              ax=axs[0], log=log, cmap=cmap)

                two_d_array = data_array[:, shape[1]//2, :]
                plot_2d_image(two_d_array, fig=fig,
                              ax=axs[1], log=log, cmap=cmap)

                two_d_array = data_array[:, :, shape[2]//2]
                plot_2d_image(two_d_array, fig=fig,
                              ax=axs[2], log=log, cmap=cmap)

                # Show figure
                plt.show()
            except IndexError:
                plt.close()
                print("Is this a 3D array?")


def complex2rgbalin(
    s,
    gamma=1.0,
    smax=None,
    smin=None,
    percentile=(None, None),
    alpha=(0, 1),
    final_type='uint8'
):
    """
    Returns RGB image with with colour-coded phase and linear amplitude
    in brightness. Optional exponent gamma is applied to the amplitude.
    Taken from PyNX

    :param s: the complex data array (likely 2D, but can have higher
     dimensions)
    :param gamma: gamma parameter to change the brightness curve
    :param smax: maximum value (brightness = 1). If not supplied and
     percentile is not set, the maximum amplitude of the array is used.
    :param smin: minimum value(brightness = 0). If not supplied and
     percentile is not set, the maximum amplitude of the array is used.
    :param percentile: a tuple of two values (percent_min, percent_max)
     setting the percentile (between 0 and 100): the smax and smin values
     will be  set as the percentile value in the array (see numpy.percentile).
     These two values (when not None) supersede smax and smin.
     Example: percentile=(0,99) to scale the brightness to 0-1 between the 1%
     and 99% percentile of the data amplitude.
    :param alpha: the minimum and maximum value for the alpha channel,
     normally (0,1). Useful to have different max/min alpha when going
     through slices of one object
    final_type: either 'float': values are in the [0..1] range,
     or 'uint8' (0..255) (new default)
    Returns:
     the RGBA array, with the same diemensions as the input array,
     plus one additional R/G/B/A dimension appended.
    """
    rgba = phase2rgb(s)
    a = np.abs(s)
    if percentile is not None:
        if percentile[0] is not None:
            smin = np.percentile(a, percentile[0])
        if percentile[1] is not None:
            smax = np.percentile(a, percentile[1])
        if smax is not None and smin is not None and smin > smax:
            smin, smax = smax, smin
    if smax is not None:
        a = (a - smax) * (a <= smax) + smax
    if smin is not None:
        a = (a - smin) * (a >= smin)
    a /= a.max()
    a = a ** gamma
    rgba[..., 3] = alpha[0] + alpha[1] * a
    if final_type == 'float':
        return rgba
    return (rgba * 255).astype(np.uint8)


def phase2rgb(s):
    """
    Crates RGB image with colour-coded phase from a complex array.

    :param s: a complex numpy array

    Returns: an RGBA numpy array, with one additional dimension added
    """
    ph = np.angle(s)
    t = np.pi / 3
    rgba = np.zeros(list(s.shape) + [4])
    rgba[..., 0] = (ph < t) * (ph > -t) + (ph > t) * (ph < 2 * t) * \
        (2 * t - ph) / t + (ph > -2 * t) * (ph < -t) * (
        ph + 2 * t) / t
    rgba[..., 1] = (ph > t) + (ph < -2 * t) * (-2 * t - ph) / \
        t + (ph > 0) * (ph < t) * ph / t
    rgba[..., 2] = (ph < -t) + (ph > -t) * (ph < 0) * (-ph) / \
        t + (ph > 2 * t) * (ph - 2 * t) / t
    return rgba
