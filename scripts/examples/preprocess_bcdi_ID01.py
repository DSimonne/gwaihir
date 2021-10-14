#!/users/simonne/anaconda3/envs/rnice.BCDI/bin/python3
# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

try:
    import hdf5plugin  # for P10, should be imported before h5py or PyTables
except ModuleNotFoundError:
    pass
import xrayutilities as xu
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend(
    "Qt5Agg"
)  # "Qt5Agg" or "Qt4Agg" depending on the version of Qt installer, bug with Tk
import os
import scipy.signal  # for medfilt2d
from scipy.ndimage.measurements import center_of_mass
import sys
from scipy.io import savemat
import tkinter as tk
from tkinter import filedialog
import gc
import bcdi.graph.graph_utils as gu
from bcdi.experiment.detector import Detector
from bcdi.experiment.setup import Setup
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.preprocessing.preprocessing_utils as pru
import bcdi.utils.utilities as util
import bcdi.utils.validation as valid


helptext = """
Prepare experimental data for Bragg CDI phasing: crop/pad, center, mask, normalize and filter the data.

Beamlines currently supported: ESRF ID01, SOLEIL CRISTAL, SOLEIL SIXS, PETRAIII P10 and APS 34ID-C.

Output: data and mask as numpy .npz or Matlab .mat 3D arrays for phasing

File structure should be (e.g. scan 1):
specfile, hotpixels file and flatfield file in:    /rootdir/
data in:                                           /rootdir/S1/data/

output files saved in:   /rootdir/S1/pynxraw/ or /rootdir/S1/pynx/ depending on 'use_rawdata' option
"""

"""Part of script to allow systematic use
defining scans, root_folder, save_dir and template_imagefile

Remenber that you may have to change the mask, the central pixel, the rocking angle, the angles...

"""

import glob

import bcdi
print("Checking environment for bcdi...")
print(bcdi.__file__)

# Print help
try:
    print ('Data dir:',  sys.argv[1])
    print ('Scan (s):',  sys.argv[2])
except IndexError:
    print("""
        Arg 1: Path of target directory (before /S{scan} ... )
        Arg 2: Scan(s) number, list or single value
        """)
    exit()

# reload mask
for i, element in enumerate(sys.argv):
    if "reload" in element:
        if "true" in element.lower():
            reload_previous = True  # True if you want to get the conjugate object
    else:
        reload_previous = False  # True if you want to get the conjugate object

# folder of the experiment, where all scans are stored
root_folder = os.getcwd() + "/" + sys.argv[1] 
print("Root folder:", root_folder)

# scans, transforming string of list into python list object
if sys.argv[2].startswith("["):
    scans = ast.literal_eval(sys.argv[2])
    scan = scans[0]

elif sys.argv[2]=="all":
    subdirnames = [x[1] for x in os.walk(root_folder)][0]
    scans = [int(s.replace("S", "")) for s in sorted(subdirnames) if s.startswith("S")]
    print(scans)
    scan = scans[0]

else:
    try:
        scans = int(sys.argv[2])
        scan = scans
    except Exception as e:
        print("Wrong scan input")
        raise e

# Save all the prints from the script
stdoutOrigin=sys.stdout

save_dir = root_folder + f"S{scan}/pynxraw/"
README_file = f"{save_dir}README_preprocess.md"
print("Save folder:", save_dir)

try:
    os.mkdir(save_dir)
except:
    pass

with open(README_file, 'w') as outfile:
    outfile.write("```bash\n")
sys.stdout = open(README_file, "a")

"""end of personal script"""


# scans = 76 # np.arange(1401, 1419+1, 3)  # scan number or list of scan numbers
# scans = np.concatenate((scans, np.arange(1147, 1195+1, 3)))
# bad_indices = np.argwhere(scans == 738)
# scans = np.delete(scans, bad_indices)

# root_folder = "/data/id01/inhouse/data/IHR/hc4050/id01/"  # folder of the experiment, where all scans are stored
# root_folder = "/data/id01/inhouse/data/IHR/hc4050_a/id01/"  # folder of the experiment, where all scans are stored
# root_folder = "/data/id01/inhouse/data/IHR/hc4050_a/id01/test/BCDI_2021_07_26_165851/"  # folder of the experiment, 
# root_folder = "/data/visitor/hc4534/id01/B8_S1_P2/BCDI_2021_09_02_145714/"  # folder of the experiment, up to spec file
root_folder = "/data/visitor/hc4534/id01/B8_S1_P2/BCDI_2021_09_02_203654/"  # folder of the experiment, up to spec filesave_dir = None  # images will be saved here, leave it to None otherwise
data_dirname = None  # leave None to use the beamline default, '' empty string when there is no subfolder
# (data directly in the scan folder), or a non-empty string for the subfolder name
# (default to scan_folder/pynx/ or scan_folder/pynxraw/ depending on the setting of use_rawdata)
sample_name = "S"  # str or list of str of sample names (string in front of the scan number in the folder name).
# If only one name is indicated, it will be repeated to match the number of scans.
user_comment = ''  # string, should start with "_"
debug = False  # set to True to see plots
binning = (1, 1, 1)  # binning to apply to the data
# (stacking dimension, detector vertical axis, detector horizontal axis)

##############################
# parameters used in masking #
##############################
flag_interact = False  # True to interact with plots, False to close it automatically
background_plot = (
    "0.5"  # in level of grey in [0,1], 0 being dark. For visual comfort during masking
)

#########################################################
# parameters related to data cropping/padding/centering #
#########################################################
centering = "max"  # Bragg peak determination: 'max' or 'com', 'max' is better usually.
#  It will be overridden by 'fix_bragg' if not empty
fix_bragg = (
    []
)  # fix the Bragg peak position [z_bragg, y_bragg, x_bragg] considering the full detector
# It is useful if hotpixels or intense aliens. Leave it [] otherwise.
fix_size = []  # crop the array to predefined size considering the full detector,
# leave it to [] otherwise [zstart, zstop, ystart, ystop, xstart, xstop]. ROI will be defaulted to []
center_fft = "crop_sym_ZYX"
# 'crop_sym_ZYX','crop_asym_ZYX','pad_asym_Z_crop_sym_YX', 'pad_sym_Z_crop_asym_YX',
# 'pad_sym_Z', 'pad_asym_Z', 'pad_sym_ZYX','pad_asym_ZYX' or 'skip'
pad_size = []  # size after padding, e.g. [256, 512, 512]. Use this to pad the array.
# used in 'pad_sym_Z_crop_sym_YX', 'pad_sym_Z', 'pad_sym_ZYX'

##############################################
# parameters used in intensity normalization #
##############################################
normalize_flux = 'skip'  # 'monitor' to normalize the intensity by the default monitor values, 'skip' to do nothing

#################################
# parameters for data filtering #
#################################
mask_zero_event = False  # mask pixels where the sum along the rocking curve is zero - may be dead pixels
flag_medianfilter = 'skip'
# set to 'median' for applying med2filter [3,3]
# set to 'interp_isolated' to interpolate isolated empty pixels based on 'medfilt_order' parameter
# set to 'mask_isolated' it will mask isolated empty pixels
# set to 'skip' will skip filtering
medfilt_order = 7   # for custom median filter, number of pixels with intensity surrounding the empty pixel

#################################################
# parameters used when reloading processed data #
#################################################
#reload_previous = True  # True to resume a previous masking (load data and mask)
reload_orthogonal = False  # True if the reloaded data is already intepolated in an orthonormal frame
preprocessing_binning = (1, 1, 1)  # binning factors in each dimension of the binned data to be reloaded

##################
# saving options #
##################
save_rawdata = True  # save also the raw data when use_rawdata is False
save_to_npz = True  # True to save the processed data in npz format
save_to_mat = False  # True to save also in .mat format
save_to_vti = False  # save the orthogonalized diffraction pattern to VTK file
save_asint = False  # if True, the result will be saved as an array of integers (save space)

######################################
# define beamline related parameters #
######################################
beamline = 'ID01'  # name of the beamline, used for data loading and normalization by monitor
# supported beamlines: 'ID01', 'SIXS_2018', 'SIXS_2019', 'CRISTAL', 'P10', 'NANOMAX', '34ID'
actuators = None  # {'rocking_angle': 'actuator_1_1'}
# Optional dictionary that can be used to define the entries corresponding to actuators in data files
# (useful at CRISTAL where the location of data keeps changing)
# e.g.  {'rocking_angle': 'actuator_1_3', 'detector': 'data_04', 'monitor': 'data_05'}
is_series = True  # specific to series measurement at P10

custom_scan = False  # set it to True for a stack of images acquired without scan, e.g. with ct in a macro, or when
# there is no spec/log file available
custom_images = None  # np.arange(11353, 11453, 1)  # list of image numbers for the custom_scan, None otherwise
custom_monitor = None  # monitor values for normalization for the custom_scan, None otherwise

rocking_angle = "outofplane"  # "outofplane" for a sample rotation around x outboard, "inplane" for a sample rotation
# around y vertical up, "energy"

follow_bragg = False  # only for energy scans, set to True if the detector was also scanned to follow the Bragg peak
# specfile_name = "spec/2021_07_20_085405_ni" #'analysis/alias_dict_2021.txt'
# specfile_name = "spec/2021_07_24_083204_test" #'analysis/alias_dict_2021.txt'
# specfile_name = "spec/BCDI_2021_07_26_165851" #'analysis/alias_dict_2021.txt'# July 
# specfile_name = "spec/BCDI_2021_07_26_165851" #'analysis/alias_dict_2021.txt'# July 
# specfile_name = "spec/BCDI_2021_09_02_145714" #'analysis/alias_dict_2021.txt'# september
specfile_name = "spec/BCDI_2021_09_02_203654" #'analysis/alias_dict_2021.txt'# september
# template for SIXS: full path of the alias dictionnary, typically root_folder + 'alias_dict_2020.txt'
# template for all other beamlines: ''

###############################
# detector related parameters #
###############################
detector = "Maxipix"    # "Eiger2M", "Maxipix", "Eiger4M", "Merlin" or "Timepix"
linearity_func = (None)  # lambda array_1d: (array_1d*(7.484e-22*array_1d**4 - 3.447e-16*array_1d**3 +
# 5.067e-11*array_1d**2 - 6.022e-07*array_1d + 0.889)) # MIR
# np.divide(array_1d, (1-array_1d*1.3e-6))  # Sarah_1
# (array_1d*(7.484e-22*array_1d**4 - 3.447e-16*array_1d**3 + 5.067e-11*array_1d**2 - 6.022e-07*array_1d + 0.889)) # MIR
# linearity correction for the detector, leave None otherwise.
# You can use def instead of a lambda expression but the input array should be 1d (flattened 2D detector array).
x_bragg = None  # horizontal pixel number of the Bragg peak, can be used for the definition of the ROI
y_bragg = None  # vertical pixel number of the Bragg peak, can be used for the definition of the ROI
roi_detector = None
# roi_detector = [y_bragg - 200, y_bragg + 200, x_bragg - 150, x_bragg + 150]
# [Vstart, Vstop, Hstart, Hstop]
# leave None to use the full detector. Use with center_fft='skip' if you want this exact size.
photon_threshold = 0  # data[data < photon_threshold] = 0
photon_filter = 'loading'  # 'loading' or 'postprocessing', when the photon threshold should be applied
# if 'loading', it is applied before binning; if 'postprocessing', it is applied at the end of the script before saving
background_file = None  # root_folder + 'background.npz'  # non empty file path or None
# hotpixels_file = "/home/david/Documents/PhD_local/PhDScripts/SIXS_January_2021/analysis/mask_merlin.npy"
hotpixels_file = None
flatfield_file = None  # root_folder + "flatfield_maxipix_8kev.npz"  # non empty file path or None
# template_imagefile = root_folder + 'detector/2021_07_20_085405_ni/data_mpx4_%05d.edf.gz'
# template_imagefile = root_folder + 'detector/2021_07_24_072032_b8_s1_p2/data_mpx4_%05d.edf.gz'
# template_imagefile = root_folder + 'mpx/data_mpx4_%05d.edf.gz'# july and september 2021
template_imagefile = root_folder + 'mpx/data_mpx4_%05d.edf'# july and september 2021# template for ID01: 'data_mpx4_%05d.edf.gz' or 'align_eiger2M_%05d.edf.gz'
# template for SIXS_2018: 'align.spec_ascan_mu_%05d.nxs'
# template for SIXS_2019: 'spare_ascan_mu_%05d.nxs'
# template for Cristal: 'S%d.nxs'
# template for P10: '_master.h5'
# template for NANOMAX: '%06d.h5'
# template for 34ID: 'Sample%dC_ES_data_51_256_256.npz'
nb_pixel_x = None  # fix to declare a known detector but with less pixels (e.g. one tile HS), leave None otherwise
nb_pixel_y = None  # fix to declare a known detector but with less pixels (e.g. one tile HS), leave None otherwise

################################################################################
# define parameters below if you want to orthogonalize the data before phasing #
################################################################################
use_rawdata = True  # False for using data gridded in laboratory frame/ True for using data in detector frame
interp_method = 'linearization'  # 'xrayutilities' or 'linearization'
fill_value_mask = 0  # 0 (not masked) or 1 (masked). It will define how the pixels outside of the data range are
# processed during the interpolation. Because of the large number of masked pixels, phase retrieval converges better if
# the pixels are not masked (0 intensity imposed). The data is by default set to 0 outside of the defined range.
beam_direction = (1, 0, 0)  # beam direction in the laboratory frame (downstream, vertical up, outboard)
sample_offsets = (-0.0011553664, 0, 0) # tuple of offsets in degrees of the sample around (downstream, vertical up, outboard)
# convention: the sample offsets will be subtracted to the motor values
sdd = 0.83 # in m, sample to detector distance in m
energy = 12994  # np.linspace(11100, 10900, num=51)  # x-ray energy in eV
custom_motors = None # {"mu": 18, "delta": 0, "gamma": 36}
# custom_motors = None  # {"mu": 0, "phi": -15.98, "chi": 90, "theta": 0, "delta": -0.5685, "gamma": 33.3147}
# use this to declare motor positions if there is not log file, None otherwise
# example: {"eta": np.linspace(16.989, 18.989, num=100, endpoint=False), "phi": 0, "nu": -0.75, "delta": 36.65}
# ID01: eta, chi, phi, nu, delta
# CRISTAL: mgomega, gamma, delta
# SIXS: beta, mu, gamma, delta
# P10: om, phi, chi, mu, gamma, delta
# NANOMAX: theta, phi, gamma, delta, energy, radius
# 34ID: mu, phi (incident angle), chi, theta (inplane), delta (inplane), gamma (outofplane)

#######################################################################################################
# parameters when orthogonalizing the data before phasing using the linearized transformation matrix #
#######################################################################################################
align_q = True  # used only when interp_method is 'linearization', if True it rotates the crystal to align q
# along one axis of the array
ref_axis_q = "y"  # q will be aligned along that axis
use_central_pixel = False # to use the angles from the nexus file
# calculate the correct angles beforehand !!
print("I hope you have used the right angles ...")
outofplane_angle = (
    0 # detector angle in deg (rotation around x outboard, typically delta),
)
# corrected for the direct beam position. Leave None to use the uncorrected position.
inplane_angle = (
    0
)  # detector angle in deg(rotation around y vertical up, typically gamma),
# corrected for the direct beam position. Leave None to use the uncorrected position.
#########################################################################
# parameters for xrayutilities to orthogonalize the data before phasing #
#########################################################################
# Make sure the central pixel is right !!
# xrayutilities uses the xyz crystal frame: for incident angle = 0, x is downstream, y outboard, and z vertical up
sample_inplane = (1, 0, 0)  # sample inplane reference direction along the beam at 0 angles in xrayutilities frame
sample_outofplane = (0, 0, 1)  # surface normal of the sample at 0 angles in xrayutilities frame
offset_inplane = 0  # outer detector angle offset as determined by xrayutilities area detector initialization
cch1 = 159.555  # direct beam vertical position in the full unbinned detector for xrayutilities 2D detector calibration
cch2 = 729.561  # direct beam horizontal position in the full unbinned detector for xrayutilities 2D detector calibration
detrot = 0  # detrot parameter from xrayutilities 2D detector calibration
tiltazimuth = 360  # tiltazimuth parameter from xrayutilities 2D detector calibration
tilt = 0  # tilt parameter from xrayutilities 2D detector calibration

##################################
# end of user-defined parameters #
##################################

# Run file