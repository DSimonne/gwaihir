import numpy as np
import tables as tb
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import ipywidgets as widgets
from ipywidgets import interact, Button, Layout, interactive, fixed
from IPython.display import display, Markdown, Latex, clear_output

import glob
import cmath

import warnings
warnings.filterwarnings("ignore")

### Use widgets to interact with files (.cxi or .npz)


class Plotter(object):
	"""
	docstring for Plotter
	"""

	def __init__(self, filename):
	    # Path of file to be imported
	    self.filename = filename
	    
	    # Get data array
	    self.get_data()
	    
	def get_data(self):
		# Load data
		if self.filename.endswith(".npy"):
			try:
				self.data = np.load(self.filename)

				# Now we have the array, we must find its shape 
				self.dimensions = np.ndim(self.data)

				# Plot according to the number of dimensions
				self.plot_data()

			except Exception as E:
				raise NameError("Wrong path")


		elif self.filename.endswith(".cxi"):
			# Open the file following the .cxi conventions
			try:
				with tb.open_file(self.filename, "r") as f:
					# Since .cxi files follow a specific architectture, we know where our data is.
					self.data = f.root.entry_1.data_1.data[:]

					# Now we have the array, we must find its shape 
					self.dimensions = np.ndim(self.data)

					# Plot according to the number of dimensions
					self.plot_data()
			except Exception as E:
				return("""
				    The file could not be loaded, verify that you are loading a file with an hdf5 architecture (.nxs, .cxi, .h5, ...) and that the file exists.
				    Otherwise, verify that the data is saved in f.root.entry_1.data_1.data[:], as it should be following csi conventions.
				    """)


		elif self.filename.endswith(".npz"):
			# Open npz file and allow the user to pick an array
			try:
				rawdata = np.load(self.filename)
				self.npz_files = rawdata.files

				@interact(
					file = widgets.Dropdown(
					    options = rawdata.files,
					    value = rawdata.files[0],
					    description = 'Pick an array to load:',
					    disabled=False,
					    style = {'description_width': 'initial'}),
					# datapath =fixed(self.filename)
					)
				def open_npz(file):
					# Pick an array
					self.data = rawdata[file]

					# Now we have the array, we must find its shape 
					self.dimensions = np.ndim(self.data)

					# Plot according to the number of dimensions
					self.plot_data()

			except Exception as E:
				raise NameError("Wrong path")


	def plot_data(self):
	    """
	    """
	    
	    if self.dimensions == 1:
	        fig, ax = plt.subplots(figsize = (15,15))
	        ax.plot(self.data)
	        
	    elif self.dimensions == 2:
	        # also xy or yx, linear or log
	        fig, ax = plt.subplots(figsize = (15,15))
	        ax.plot(self.data[0], self.data[1])
	        
	    elif self.dimensions == 3:
	        @interact(
	            axplot = widgets.Dropdown(
	                options = ["xy", "yz", "xz"],
	                value = "xy",
	                description = 'First 2 axes:',
	                disabled=False,
	                style = {'description_width': 'initial'}),
	            ComplexNumber = widgets.ToggleButtons(
	                options = ["Real", "Imaginary", "Module", "Phase"],
	                value = "Module",
	                description='Plotting options',
	                disabled=False,
	                button_style='', # 'success', 'info', 'warning', 'danger' or ''
	                tooltip=['Plot only contour or not', "", ""])
	                )
	        def plot_3d(
	            axplot,
	            ComplexNumber
	            ):

	            # Decide what we want to plot
	            if ComplexNumber == "Real":
	                data = np.real(self.data)
	            elif ComplexNumber == "Imaginary":
	                data = np.imag(self.data)
	            elif ComplexNumber == "Module":
	                data = np.abs(self.data)
	            elif ComplexNumber == "Phase":
	                data = np.angle(self.data)

	            # Take the shape of that array along 2 axis
	            if axplot == "xy":
	                print(f"The shape of this projection is {np.shape(data[:, :, 0])}")

	                r = np.shape(data[0, 0, :])
	                print(f"The range in the last axis is {r[0]}")


	            elif axplot == "yz":
	                print(f"The shape of this projection is {np.shape(data[0, :, :])}")

	                r = np.shape(data[:, 0, 0])
	                print(f"The range in the last axis is {r[0]}")

	            elif axplot == "xz":
	                print(f"The shape of this projection is {np.shape(data[:, 0, :])}")

	                r = np.shape(data[0, :, 0])
	                print(f"The range in the last axis is {r[0]}")


	            @interact(
	                i = widgets.IntSlider(
	                    min=0,
	                    max=r[0]-1,
	                    step=1,
	                    description='Index along last axis:',
	                    disabled=False,
	                    orientation='horizontal',
	                    continuous_update=False,
	                    readout=True,
	                    readout_format='d',
	                    # style = {'description_width': 'initial'}
	                    ),
	                PlottingOptions = widgets.ToggleButtons(
	                    options = [("2D plot", "2D"),
	                            ("2D contour plot", "2DC"),
	                            # ("3D surface plot", "3D")
	                            ],
	                    value = "2D",
	                    description='Plotting options',
	                    disabled=False,
	                    button_style='', # 'success', 'info', 'warning', 'danger' or ''
	                    tooltip=['Plot only contour or not', "", ""],
	                    #icon='check'
	                    ),
	                scale = widgets.Dropdown(
	                    options = ["linear", "logarithmic"],
	                    value = "linear",
	                    description = 'Scale',
	                    disabled=False,
	                    style = {'description_width': 'initial'}),
	            )
	            def PickLastAxis(i, PlottingOptions, scale):
	                if axplot == "xy":
	                    dt = data[:, :, i]
	                elif axplot == "yz":
	                    dt = data[i, :, :]
	                elif axplot == "xz":
	                    dt = data[:, i, :]

	                else:
	                    raise TypeError("Choose xy, yz or xz as axplot.")

	                dmax = dt.max()
	                dmin = dt.min()

	                # Show image
	                if PlottingOptions == "2D":
	                    fig, ax = plt.subplots(figsize = (15, 15))
	                    img = ax.imshow(dt,
	                                norm= {"linear" : None, "logarithmic" : LogNorm()}[scale],
	                                cmap="cividis",
	                                )

	                    divider = make_axes_locatable(ax)
	                    cax = divider.append_axes('right', size='5%', pad=0.05)

	                    fig.colorbar(img, cax=cax, orientation='vertical')
	                    plt.show()
	                    plt.close()

	                elif PlottingOptions == "2DC" :
	                    # Show contour plot instead
	                    try:
	                        fig, ax = plt.subplots(figsize = (15,15))
	                        ticks = [dmin + n * (dmax-dmin)/10 for n in range(0, 11)] if scale == "linear" else [pow(10, x) for x in range (0, len(str(dmax)))]

	                        img = ax.contour(dt,
	                                        ticks,
	                                        norm= {"linear" : None, "logarithmic" : LogNorm()}[scale],
	                                        cmap='cividis')

	                        divider = make_axes_locatable(ax)
	                        cax = divider.append_axes('right', size='5%', pad=0.05)

	                        fig.colorbar(img, cax=cax, orientation='vertical')
	                        plt.show()
	                        plt.close()

	                    except IndexError:
	                        plt.close()
	                        print("No contour levels were found within the data range. Meaning there is very little variation in the dat, change index")

	                # elif PlottingOptions == "3D" :
	                #     plt.close()

	                #     # Create figure and add axis
	                #     fig = plt.figure(figsize=(15,15))
	                #     ax = plt.subplot(111, projection='3d')

	                #     # Create meshgrid

	                #     X, Y = np.meshgrid(np.arange(0, dt.shape[0], 1), np.arange(0, dt.shape[1], 1))

	                #     plot = ax.plot_surface(X=X, Y=Y, Z=dt, cmap='YlGnBu_r', vmin=dmin, vmax=dmax)

	                #     # Adjust plot view
	                #     ax.view_init(elev=50, azim=225)
	                #     ax.dist=11

	                #     # Add colorbar
	                #     cbar = fig.colorbar(plot, ax=ax, shrink=0.6)

	                #     # Edit colorbar ticks and labels
	                #     ticks = [dmin + n * (dmax-dmin)/10 for n in range(0, 11)]
	                #     tickslabel = [f"{t}" for t in ticks]

	                #     cbar.set_ticks(ticks)
	                #     cbar.set_ticklabels(tickslabel)
