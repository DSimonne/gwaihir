import glob
import numpy as np
import tables as tb

from scipy.ndimage import gaussian_filter
from IPython.display import display, Markdown, Latex, clear_output

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

class SupportTools(object):
	"""
    Class that regroups the methods used to create/extract/optimize support in BCDI
    """
    
	def __init__(self, path_to_data = None, path_to_support = None, saving_directory = None):

		self.path_to_data = path_to_data
		self.path_to_support = path_to_support

		if saving_directory == None:
			try:
				self.saving_directory = self.path_to_data.replace(self.path_to_data.split("/")[-1], "")
			except AttributeError:
				try:
					self.saving_directory = self.path_to_support.replace(self.path_to_support.split("/")[-1], "")
				except Exception as e:
					raise e
		else:
			self.saving_directory = saving_directory
		print("Saving directory:", self.saving_directory)
		

	def extract_support(self, compute = True):
		"""
		Extract support as a 3D array of 0 et 1 from a reconstruction
		"""

		# Work on cxi files
		if compute:
			if self.path_to_data.endswith(".cxi"):
				with tb.open_file(self.path_to_data, "r") as f:

				    # Since .cxi files follow a specific architectture, we know where the mask is.
				    support = f.root.entry_1.image_1.mask[:]

				    # Save support
				    np.savez(self.saving_directory + "extracted_support.npz", support = support)
				    print(f"Saved support in {self.saving_directory} as extracted_support.npz")
				    self.plot_3d_support(support)

			# elif self.self.path_to_data.endswith(".npz"):

			else:
				print("Data type not supported")

		else:
			clear_output(True)
			print("Set compute to true to continue")


	def gaussian_convolution(self, sigma, threshold, compute = True):
		"""
		Apply a gaussian convolution to the support, to avoid having holes inside.
		"""
		if compute:
			try:
				old_support = np.load(self.path_to_support)["support"]
			except:
				old_support = np.load(self.path_to_support)["data"]

			bigdata = 100 * old_support
			conv_support = np.where(gaussian_filter(bigdata, sigma) > threshold, 1, 0)

			np.savez(self.saving_directory + f"filter_sig{sigma}_t{threshold}", oldsupport = old_support, support = conv_support)

			print(f"Support saved in {self.saving_directory} as \nfilter_sig{sigma}_t{threshold}")
			self.plot_3d_support(conv_support)
	            
		else:
			clear_output(True)
			print("Set compute to true to continue")


	def compute_support(self, threshold, compute = True):
		"""
		Create support from data, based on maximum value of electronic density module, a threshold is applied.
		"""

		if compute:
			with tb.open_file(self.path_to_data, "r") as f:
			    # Since .cxi files follow a specific architecture, we know where our data is
			    if self.path_to_data.endswith(".cxi"):
			    	electronic_density = f.root.entry_1.data_1.data[:]

			    elif self.path_to_data.endswith(".h5"):
			    	# Take first mode
			    	electronic_density = f.root.entry_1.data_1.data[:][0]

			    print(f"Shape of real space complex electronic density array {np.shape(electronic_density)}")

			    # Find max value in image, we work with the module
			    amp = np.abs(electronic_density)
			    print(f"Maximum value in amplitude array: {amp.max()}")
			    
			    # Define support based on max value and threshold
			    support = np.where(amp < threshold * amp.max(), 0, 1)

			    # Check % occupied by the support
			    rocc = np.where(support == 1)
			    rnocc = np.where(support == 0)
			    print(f"Percentage of 3D array occupied by support:\n{np.shape(rocc)[1] / np.shape(rnocc)[1]}")

			    # Save support
			    np.savez(self.saving_directory + "computed_support.npz", support = support)
			    print(f"Saved support in {self.saving_directory} as computed_support.npz")
			    self.plot_3d_support(support)

		else:
			clear_output(True)
			print("Set compute to true to continue")


	def plot_3d_support(self, array):
	    """
	    """
	    shape = array.shape

	    if array.ndim == 3:
	        two_d_array = array[shape[0]//2,:,:]
	        self.plot_2d_support(two_d_array, dim = 0)

	        two_d_array = array[:,shape[1]//2,:]
	        self.plot_2d_support(two_d_array, dim = 1)

	        two_d_array = array[:,:,shape[2]//2]
	        self.plot_2d_support(two_d_array, dim = 2)


	def plot_2d_support(self, two_d_array, dim):
	    """
	    """
	    # Find max and min
	    dmax = two_d_array.max()
	    dmin = two_d_array.min()

	    fig, ax = plt.subplots(figsize = (5, 5))
	    img = ax.imshow(two_d_array,
	                origin='lower',
	                cmap='YlGnBu_r',
	                extent=(0, 2, 0, 2),
	                vmin = dmin,
	                vmax = dmax)

	    # Create axis for colorbar
	    cbar_ax = make_axes_locatable(ax).append_axes(position='right', size='5%', pad=0.1)

	    # Create colorbar
	    cbar = fig.colorbar(mappable=img, cax=cbar_ax)

	    # Edit colorbar ticks and labels
	    ticks = [dmin + n * (dmax-dmin)/10 for n in range(0, 11)]
	    tickslabel = [f"{t}" for t in ticks]

	    cbar.set_ticks(ticks)
	    cbar.set_ticklabels(tickslabel)
	    ax.set_title(f"Middle slice on dimension {dim}")
	    plt.show()