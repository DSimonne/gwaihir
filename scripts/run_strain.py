#!/usr/bin/python3
# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

from collections.abc import Sequence
from datetime import datetime
from functools import reduce
import gc

try:
    import hdf5plugin  # for P10, should be imported before h5py or PyTables
except ModuleNotFoundError:
    pass
import h5py
from matplotlib import pyplot as plt
from numbers import Real
import numpy as np
import os
import pprint
import tkinter as tk
from tkinter import filedialog
import bcdi.graph.graph_utils as gu
from bcdi.experiment.detector import create_detector
from bcdi.experiment.setup import Setup
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.preprocessing.preprocessing_utils as pru
import bcdi.simulation.simulation_utils as simu
import bcdi.utils.utilities as util
import bcdi.utils.validation as valid


helptext = """
Interpolate the output of the phase retrieval into an orthonormal frame,
and calculate the strain component along the direction of the experimental diffusion
vector q.

Input: complex amplitude array, output from a phase retrieval program.
Output: data in an orthonormal frame (laboratory or crystal frame), amp_disp_strain
array.The disp array should be divided by q to get the displacement (disp = -1*phase
here).

Laboratory frame: z downstream, y vertical, x outboard (CXI convention)
Crystal reciprocal frame: qx downstream, qz vertical, qy outboard
Detector convention: when out_of_plane angle=0   Y=-y , when in_plane angle=0   X=x

In arrays, when plotting the first parameter is the row (vertical axis), and the
second the column (horizontal axis). Therefore the data structure is data[qx, qz,
qy] for reciprocal space, or data[z, y, x] for real space
"""

"""Part of script to allow systematic use
defining scan, root_folder, save_dir, comment, sample_name and template_imagefile
Remember to get the correct angles from the correct_anbfgles_detector.py script

Remember that you may have to change the mask, the central pixel, the rocking angle, the angles...

"""

import ast
import glob
import sys

# Print help
try:
    print ('Data dir:',  sys.argv[1])
    print ('Scan:',  sys.argv[2])
    print ('Out-of-plane_angle:',  sys.argv[3])
    print ('In-plane_angle:',  sys.argv[4])
    print ('Tilt angle:',  sys.argv[5])
except IndexError:
    print("""
        Arg 1: Path of target directory (before /S{scan} ... )
        Arg 2: Scan(s) number, list or single value
        Arg 3: Out-of-plane_angle
        Arg 4: In-plane_angle
        Arg 5: Tilt angle
        """)
    exit()

scan = int(sys.argv[2])
outofplane_angle = float(sys.argv[3]) # detector angle in deg (rotation around x outboard): delta ID01, delta SIXS, gamma 34ID
inplane_angle = float(sys.argv[4])  # detector angle in deg(rotation around y vertical up): nu ID01, gamma SIXS, tth 34ID
tilt_angle = float(sys.argv[5])  # angular step size for rocking angle, eta ID01, mu SIXS, does not matter for energy scan

for i, element in enumerate(sys.argv):
    if "flip" in element:
        if "true" in element.lower():
            flip_reconstruction = True  # True if you want to get the conjugate object
            print("Flipping the reconstruction")
    else:
    	flip_reconstruction = False  # True if you want to get the conjugate object

# folder of the experiment, where all scans are stored
root_folder = os.getcwd() + "/" + sys.argv[1] 
print("Root folder:", root_folder)

#Scan folder
scan_folder = root_folder + f"S{scan}/"
print("Scan folder:", scan_folder)

# Data folder
data_folder = scan_folder + "data/" # folder of the experiment, where all scans are stored
print("Data folder:", data_folder)

save_dir = scan_folder + "result_crystal_frame/"  # images will be saved here, leave it to None otherwise (default to data directory's parent)
# save_dir = None

comment = ''  # comment in filenames, should start with _
sample_name = "S"  # str or list of str of sample names (string in front of the scan number in the folder name).

rocking_angle_sixs = "mu" # choose between mu or omega
# template_imagefile = 'NoMirror_ascan_mu_%05d_R.nxs'
filename = glob.glob(f"{data_folder}*{rocking_angle_sixs}*{scan}*")[0]
template_imagefile = filename.split("/data/")[-1].split("%05d"%scan)[0] +"%05d_R.nxs"
print("Template: ", template_imagefile)

# Save all the prints from the script
stdoutOrigin=sys.stdout

if not isinstance(save_dir, str):
	save_dir = "result/"
README_file = f"{save_dir}README_strain.md"
print("Save folder:", save_dir)
try:
	os.mkdir(save_dir)
except:
	pass

with open(README_file, 'w') as outfile:
    outfile.write("```bash\n")
sys.stdout = open(README_file, "a")

"""end of personal script"""

#########################################################
# parameters used when averaging several reconstruction #
#########################################################
sort_method = 'variance/mean'  # 'mean_amplitude' or 'variance' or 'variance/mean' or 'volume', metric for averaging
correlation_threshold = 0.90

data_dirname = None
#########################################################
# parameters relative to the FFT window and voxel sizes #
#########################################################
#original_size = [168, 1024, 800]  # size of the FFT array before binning. It will be modify to take into account binning
# during phasing automatically. Leave it to () if the shape did not change.
original_size = []
phasing_binning = (1, 1, 1)  # binning factor applied during phase retrieval
preprocessing_binning = (1, 1, 1)  # binning factors in each dimension used in preprocessing (not phase retrieval)
#output_size = (50, 50, 50)  # (z, y, x) Fix the size of the output array, leave it as () otherwise
output_size = None
keep_size = False  # True to keep the initial array size for orthogonalization (slower), it will be cropped otherwise
fix_voxel = None  # voxel size in nm for the interpolation during the geometrical transformation. If a single value is
# provided, the voxel size will be identical is all 3 directions. Set it to None to use the default voxel size
# (calculated from q values, it will be different in each dimension).

#############################################################
# parameters related to displacement and strain calculation #
#############################################################
data_frame = 'detector'  # 'crystal' if the data was interpolated into the crystal frame using (xrayutilities) or
# (transformation matrix + align_q=True)
# 'laboratory' if the data was interpolated into the laboratory frame using the transformation matrix (align_q = False)
# 'detector' if the data is still in the detector frame
ref_axis_q = ("y")  # axis along which q will be aligned (data_frame= 'detector' or 'laboratory')
# or is already aligned (data_frame='crystal')
save_frame = 'crystal'  # 'crystal', 'laboratory' or 'lab_flat_sample'
# 'crystal' to save the data with q aligned along ref_axis_q
# 'laboratory' to save the data in the laboratory frame (experimental geometry)
# 'lab_flat_sample' to save the data in the laboratory frame, with all sample angles rotated back to 0
# rotations for 'laboratory' and 'lab_flat_sample' are realized after the strain calculation
# (which is done in the crystal frame along ref_axis_q)
isosurface_strain = 0.5  # threshold use for removing the outer layer (strain is undefined at the exact surface voxel)
strain_method = 'default'  # 'default' or 'defect'. If 'defect', will offset the phase in a loop and keep the smallest
# magnitude value for the strain. See: F. Hofmann et al. PhysRevMaterials 4, 013801 (2020)
phase_offset = 0  # manual offset to add to the phase, should be 0 in most cases
phase_offset_origin = (None)  # the phase at this voxel will be set to phase_offset, None otherwise
offset_method = 'mean'  # 'COM' or 'mean', method for removing the offset in the phase
centering_method = 'max_com'  # 'com' (center of mass), 'max', 'max_com' (max then com), 'do_nothing'
# TODO: where is q for energy scans? Should we just rotate the reconstruction to have q along one axis,
#  instead of using sample offsets?

######################################
# define beamline related parameters #
######################################
beamline = "SIXS_2019"  # name of the beamline, used for data loading and normalization by monitor and orthogonalisation
# supported beamlines: 'ID01', 'SIXS_2018', 'SIXS_2019', 'CRISTAL', 'P10', '34ID'
actuators = None
rocking_angle = "inplane"  # # "outofplane" for a sample rotation around x outboard, "inplane" for a sample rotation
# around y vertical up, does not matter for energy scan
#  "inplane" e.g. phi @ ID01, mu @ SIXS "outofplane" e.g. eta @ ID01
sdd = 1.18  # sample to detector distance in m
energy = 8500  # x-ray energy in eV, 6eV offset at ID01
beam_direction = np.array(
    [1, 0, 0]
)  # incident beam along z, in the frame (z downstream, y vertical up, x outboard)
# outofplane_angle = -0.01815149389135301 # detector angle in deg (rotation around x outboard): delta ID01, delta SIXS, gamma 34ID
# inplane_angle = 37.51426377175866  # detector angle in deg(rotation around y vertical up): nu ID01, gamma SIXS, tth 34ID
# tilt_angle = 0.007737016574585642  # angular step size for rocking angle, eta ID01, mu SIXS, does not matter for energy scan
sample_offsets = None  # tuple of offsets in degrees of the sample around (downstream, vertical up, outboard)
# the sample offsets will be subtracted to the motor values
specfile_name = None #'analysis/alias_dict_2021.txt'
# template for ID01: name of the spec file without '.spec'
# template for SIXS_2018: full path of the alias dictionnary, typically root_folder + 'alias_dict_2019.txt'
# template for all other beamlines: ''

##########################
# setup for custom scans #
##########################
custom_scan = False  # set it to True for a stack of images acquired without scan, e.g. with ct in a macro, or when
# there is no spec/log file available, or for 34ID
custom_motors = {
    "delta": inplane_angle,
    "gamma": outofplane_angle,
    "theta": 1.0540277,
    "phi": -4.86,
}
###############################
# detector related parameters #
###############################
detector = "Merlin"    # "Eiger2M", "Maxipix", "Eiger4M", "Merlin" or "Timepix"
nb_pixel_x = None  # fix to declare a known detector but with less pixels (e.g. one tile HS), leave None otherwise
nb_pixel_y = None  # fix to declare a known detector but with less pixels (e.g. one tile HS), leave None otherwise
pixel_size = None  # use this to declare the pixel size of the "Dummy" detector if different from 55e-6
# template_imagefile = 'mirror_ascan_mu_%05d_R.nxs'
# template for ID01: 'data_mpx4_%05d.edf.gz' or 'align_eiger2M_%05d.edf.gz'
# template for SIXS_2018: 'align.spec_ascan_mu_%05d.nxs'
# template for SIXS_2019: 'spare_ascan_mu_%05d.nxs'
# template for Cristal: 'S%d.nxs'
# template for P10: '_master.h5'
# template for NANOMAX: '%06d.h5'
# template for 34ID: 'Sample%dC_ES_data_51_256_256.npz'

###################################################
# parameters related to the refraction correction #
###################################################
correct_refraction = False  # True for correcting the phase shift due to refraction
optical_path_method = 'threshold'  # 'threshold' or 'defect', if 'threshold' it uses isosurface_strain to define the
# support  for the optical path calculation, if 'defect' (holes) it tries to remove only outer layers even if
# the amplitude is lower than isosurface_strain inside the crystal
dispersion = 5.0328E-05  # delta
# Pt:  3.0761E-05 @ 10300eV, 5.0328E-05 @ 8170eV
# 3.2880E-05 @ 9994eV, 4.1184E-05 @ 8994eV, 5.2647E-05 @ 7994eV, 4.6353E-05 @ 8500eV / Ge 1.4718E-05 @ 8keV
absorption = 4.8341E-06  # beta
# Pt:  2.0982E-06 @ 10300eV, 4.8341E-06 @ 8170eV
# 2.3486E-06 @ 9994eV, 3.4298E-06 @ 8994eV, 5.2245E-06 @ 7994eV, 4.1969E-06 @ 8500eV
threshold_unwrap_refraction = 0.05  # threshold used to calculate the optical path
# the threshold for refraction/absorption corrections should be low, to correct for an object larger than the real one,
# otherwise it messes up the phase

###########
# options #
###########
simu_flag = False  # set to True if it is simulation, the parameter invert_phase will be set to 0
invert_phase = True  # True for the displacement to have the right sign (FFT convention), False only for simulations
# flip_reconstruction = False  # True if you want to get the conjugate object
phase_ramp_removal = (
    "gradient"  # 'gradient'  # 'gradient' or 'upsampling', 'gradient' is much faster
)
threshold_gradient = (
    1.0  # upper threshold of the gradient of the phase, use for ramp removal
)
save_raw = False  # True to save the amp-phase.vti before orthogonalization
save_support = (
    True  # True to save the non-orthogonal support for later phase retrieval
)
save = True  # True to save amp.npz, phase.npz, strain.npz and vtk files
debug = False  # set to True to show all plots for debugging
roll_modes = (
    0,
    0,
    0,
)  # axis=(0, 1, 2), correct a roll of few pixels after the decomposition into modes in PyNX
############################################
# parameters related to data visualization #
############################################
align_axis = False  # for visualization, if True rotates the crystal to align axis_to_align along ref_axis after the
# calculation of the strain
ref_axis = "y"  # will align axis_to_align to that axis
axis_to_align = np.array([-0.011662456997498807, 0.957321364700986, -0.28879022106682123])
# axis to align with ref_axis in the order x y z (axis 2, axis 1, axis 0)
strain_range = 0.002  # for plots 0.001?
phase_range = np.pi  # for plotsn np.pi/2 ?
grey_background = True  # True to set the background to grey in phase and strain plots
tick_spacing = 100  # for plots, in nm
tick_direction = 'inout'  # 'out', 'in', 'inout'
tick_length = 3  # 10  # in plots
tick_width = 1  # 2  # in plots

##########################################
# parameters for temperature estimation #
##########################################
get_temperature = False  # only available for platinum at the moment
reflection = np.array(
    [1, 1, 1]
)  # measured reflection, use for estimating the temperature
reference_spacing = 2.254761  # for calibrating the thermal expansion, if None it is fixed to 3.9236/norm(reflection) Pt
reference_temperature = (
    None  # used to calibrate the thermal expansion, if None it is fixed to 293.15K (RT)
)

##########################################################
# parameters for averaging several reconstructed objects #
##########################################################
avg_method = 'reciprocal_space'  # 'real_space' or 'reciprocal_space'
avg_threshold = 0.90  # minimum correlation within reconstructed object for averaging

############################################
# setup for phase averaging or apodization #
############################################
hwidth = (
    1 # (width-1)/2 of the averaging window for the phase, 0 means no phase averaging
)
apodize_flag = True  # True to multiply the diffraction pattern by a filtering window
apodize_window = (
    "blackman"  # filtering window, multivariate 'normal' or 'tukey' or 'blackman'
)
mu = np.array([0.0, 0.0, 0.0])  # mu of the gaussian window
sigma = np.array([0.30, 0.30, 0.30])  # sigma of the gaussian window
alpha = np.array([1.0, 1.0, 1.0])  # shape parameter of the tukey window
##################################
# end of user-defined parameters #
##################################

from gwaihir.runner import strain

strain.strain_bcdi(
    scan, 
    root_folder,
    save_dir,
    data_dirname,
    sample_name, 
    comment, 
    sort_method, 
    correlation_threshold,
    original_size, 
    phasing_binning, 
    preprocessing_binning, 
    output_size, 
    keep_size, 
    fix_voxel,
    data_frame,
    ref_axis_q,
    save_frame,
    isosurface_strain,
    strain_method,
    phase_offset,
    phase_offset_origin,
    offset_method,
    centering_method,
    beamline,
    actuators,
    rocking_angle,
    sdd,
    energy,
    beam_direction,
    outofplane_angle,
    inplane_angle,
    tilt_angle,
    sample_offsets,
    specfile_name,
    custom_scan,
    custom_motors,
    detector,
    nb_pixel_x,
    nb_pixel_y,
    pixel_size,
    template_imagefile,
    correct_refraction,
    optical_path_method,
    dispersion,
    absorption,
    threshold_unwrap_refraction,
    simu_flag,
    invert_phase,
    flip_reconstruction,
    phase_ramp_removal,
    threshold_gradient,
    save_raw,
    save_support,
    save,
    debug,
    roll_modes,
    align_axis,
    ref_axis,
    axis_to_align,
    strain_range,
    phase_range,
    grey_background,
    tick_spacing,
    tick_direction,
    tick_length,
    tick_width,
    get_temperature,
    reflection,
    reference_spacing,
    reference_temperature,
    avg_method,
    avg_threshold,
    hwidth,
    apodize_flag,
    apodize_window,
    mu,
    sigma,
    alpha,
    h5_data = None,
    GUI = False,
)