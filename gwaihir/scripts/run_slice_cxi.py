#!/usr/bin/python3
import sys
import numpy as np
import tables as tb
import ast
import shutil
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

import glob
import cmath

# Annoying warnings bc of .cxi files
import warnings
warnings.filterwarnings("ignore")

# Print help
try:
	print ('Data dir:',  sys.argv[1])
	print ('Plotted Value:',  sys.argv[2])
	print ('Plot type:',  sys.argv[3])
	print ("Index for slice:", sys.argv[4])

except IndexError:
    print("""
        Arg 1: Path to directory with cxi files
        Arg 2: Real/Imaginary/Module/Phase
        Arg 3: 2D / 2DC (contour or normal plot)
        Arg 4: index of slice ("mid" takes the middle of each axis for the projection)
        """)
    exit()

scan_dir = sys.argv[1]
PlottedValue = sys.argv[2]
PlottedDimensions = sys.argv[3]
PlottedIndex = sys.argv[4]

def Plotting(scan_dir, datapath, PlottedValue, PlottedAxes, PlottedDimensions, PlottedIndex):
	"""Interactive function to plot the cxi files, only open in read mode"""

	# Open the file
	try:
		with tb.open_file(datapath, "r") as f:
			# Since .cxi files follow a specific architectture, we know where our data is.
			data = f.root.entry_1.data_1.data[:]

	except Exception as E:
		raise NameError("Wrong path")

	# Save run name
	run = "Run" + datapath.split("Run")[1].split("_LLK")[0]
	print(run)

	# Decide what we want to plot
	if PlottedValue == "Real":
		PlottedArrayType = np.real(data)
	elif PlottedValue == "Imaginary":
		PlottedArrayType = np.imag(data)
	elif PlottedValue == "Module":
		PlottedArrayType = np.abs(data)
	else:
		PlottedArrayType = np.angle(data)

	# Print the shape of that array along 2 axis, use the last dimension for plotting and Project along two axes
	if PlottedAxes == "xy":
		print(f"The shape of this projection is {np.shape(data[:, :, 0])}")

		r = np.shape(data[0, 0, :])
		print(f"Length of last axis: {r[0]}")

	elif PlottedAxes == "yz":
		print(f"The shape of this projection is {np.shape(data[0, :, :])}")

		r = np.shape(data[:, 0, 0])
		print(f"Length of last axis: {r[0]}")

	elif PlottedAxes == "zx":
		print(f"The shape of this projection is {np.shape(data[:, 0, :])}")

		r = np.shape(data[0, :, 0])
		print(f"Length of last axis: {r[0]}")

	# Create a new figure
	plt.close()
	if PlottedIndex == "mid":
		i = r[0]//2 # min=0, max=r[0]-1, step=1,
	else:
		i = int(sys.argv[5])

	# Print the shape of that array along 2 axis, use the last dimension for plotting and Project along two axes
	if PlottedAxes == "xy":
	    TwoDPlottedArray = PlottedArrayType[:, :, i]

	elif PlottedAxes == "yz":
	    TwoDPlottedArray = PlottedArrayType[i, :, :]

	elif PlottedAxes == "zx":
	    TwoDPlottedArray = PlottedArrayType[:, i, :]

	# Find max and min
	dmax = TwoDPlottedArray.max()
	dmin = TwoDPlottedArray.min()

	# print(f"Current index: {i}")
	# print(np.mean(TwoDPlottedArray))

	# Create figure and add axis
	# fig = plt.figure(figsize=(20,10))
	# ax = fig.add_subplot(111)

	# Remove x and y ticks
	# ax.xaxis.set_tick_params(size=0)
	# ax.yaxis.set_tick_params(size=0)
	# ax.set_xticks([])
	# ax.set_yticks([])

	# Show image
	if PlottedDimensions == "2D":
	    plt.close()
	    fig, ax = plt.subplots(figsize = (15,15),
	    dpi = 150
	    )
	    img = ax.imshow(TwoDPlottedArray,
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
	    if PlottedAxes == "xy":
	        plt.savefig(f"{scan_dir}slices/xy/{run}_{PlottedAxes}_{PlottedValue}_{i}_2D.png")
	        print(f"Saved as {scan_dir}slices/xy/{run}_{PlottedAxes}_{PlottedValue}_{i}_2D.png")
	    elif PlottedAxes == "yz":
	        plt.savefig(f"{scan_dir}slices/yz/{run}_{PlottedAxes}_{PlottedValue}_{i}_2D.png")
	        print(f"Saved as {scan_dir}slices/yz/{run}_{PlottedAxes}_{PlottedValue}_{i}_2D.png")
	    elif PlottedAxes == "zx":
	        plt.savefig(f"{scan_dir}slices/zx/{run}_{PlottedAxes}_{PlottedValue}_{i}_2D.png")
	        print(f"Saved as {scan_dir}slices/zx/{run}_{PlottedAxes}_{PlottedValue}_{i}_2D.png") 
	    plt.tight_layout()

	elif PlottedDimensions == "2DC" :
	    plt.close()
	    # Show contour plot instead
	    try:
	        fig, ax = plt.subplots(figsize = (15,15),
	        dpi = 150
	        )

	        img = ax.contour(TwoDPlottedArray,
	                    origin = "lower",
	                    extent=(0, 2, 0, 2),
	                    cmap='YlGnBu_r',
	                    vmin=dmin,
	                    vmax=dmax)

	        # Create axis for colorbar
	        cbar_ax = make_axes_locatable(ax).append_axes(position='right', size='5%', pad=0.1)

	        # Create colorbar
	        cbar = fig.colorbar(mappable=img, cax=cbar_ax)

	        # Edit colorbar ticks and labels
	        ticks = [dmin + n * (dmax-dmin)/10 for n in range(0, 11)]
	        tickslabel = [f"{t}" for t in ticks]

	        cbar.set_ticks(ticks)
	        cbar.set_ticklabels(tickslabel)
	        if PlottedAxes == "xy":
	            plt.savefig(f"{scan_dir}slices/xy/{run}_{PlottedAxes}_{PlottedValue}_{i}_2DC.png")
	            print(f"Saved as {scan_dir}slices/xy/{run}_{PlottedAxes}_{PlottedValue}_{i}_2DC.png")
	        elif PlottedAxes == "yz":
	            plt.savefig(f"{scan_dir}slices/yz/{run}_{PlottedAxes}_{PlottedValue}_{i}_2DC.png")
	            print(f"Saved as {scan_dir}slices/yz/{run}_{PlottedAxes}_{PlottedValue}_{i}_2DC.png")
	        elif PlottedAxes == "zx":
	            plt.savefig(f"{scan_dir}slices/zx/{run}_{PlottedAxes}_{PlottedValue}_{i}_2DC.png")
	            print(f"Saved as {scan_dir}slices/zx/{run}_{PlottedAxes}_{PlottedValue}_{i}_2DC.png")      
	            #plt.show()
	        plt.tight_layout()

	    except IndexError:
	        plt.close()
	        print("No contour levels were found within the data range. Meaning there is very little variation in the data, change index")# 


try:
	os.mkdir(f"{scan_dir}slices")
	os.mkdir(f"{scan_dir}slices/xy")
	os.mkdir(f"{scan_dir}slices/yz")
	os.mkdir(f"{scan_dir}slices/zx")

except FileExistsError:
    pass

# Load data
filenames = glob.glob(f"{scan_dir}*LLK*.cxi")

for f in filenames:	    
    # Plot
    Plotting(scan_dir, f, PlottedValue, "xy", PlottedDimensions, PlottedIndex)
    Plotting(scan_dir, f, PlottedValue, "yz", PlottedDimensions, PlottedIndex)
    Plotting(scan_dir, f, PlottedValue, "zx", PlottedDimensions, PlottedIndex)