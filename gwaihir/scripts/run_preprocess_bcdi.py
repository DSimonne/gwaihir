#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ast
import sys
import glob
import os
from gwaihir.runner import preprocess

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


######################################
# parameters loaded via the terminal #
######################################

# Reload mask or not
for i, element in enumerate(sys.argv):
    if "reload" in element:
        if "true" in element.lower():
            reload_previous = True  # True if you want to get the conjugate object
    else:
        reload_previous = False  # True if you want to get the conjugate object


# Folder of the experiment, where all scans are stored
root_folder = os.getcwd() + "/" + sys.argv[1] 
print("Root folder:", root_folder)


# Scans, transforming string of list into python list object
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


# Scan folder
scan_folder = root_folder + f"S{scan}/"
print("Scan folder:", scan_folder)


# Data folder
data_folder = scan_folder + "data/" # folder of the experiment, where all scans are stored
print("Data folder:", data_folder)


# Filename
try:
    filename = glob.glob(f"{data_folder}*mu*{scan}*")[0]
    print("Mu scan")
except IndexError:
        filename = glob.glob(f"{data_folder}*omega*{scan}*")[0]
        print("Omega scan") 
template_imagefile = filename.split("/data/")[-1].split("%05d"%scan)[0] +"%05d_R.nxs"
print("Template: ", template_imagefile)


save_dir = None  # images will be saved here, leave it to None otherwise
data_dirname = None  # leave None to use the beamline default, '' empty string when there is no subfolder

# Save all the prints from the script
stdoutOrigin=sys.stdout

if not isinstance(save_dir, str):
    save_dir = scan_folder +"pynxraw/"
README_file = f"{save_dir}README_preprocess.md"
print("Save folder:", save_dir)
try:
    os.mkdir(save_dir)
except:
    pass

with open(README_file, 'w') as outfile:
    outfile.write("```bash\n")
sys.stdout = open(README_file, "a")


# Create parameters
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
reload_orthogonal = (False)  # True if the reloaded data is already intepolated in an orthonormal frame
preprocessing_binning = (1, 1, 1)  # binning factors in each dimension of the binned data to be reloaded

##################
# saving options #
##################
save_rawdata = True  # save also the raw data when use_rawdata is False
save_to_npz = True  # True to save the processed data in npz format
save_to_mat = False  # True to save also in .mat format
save_to_vti = False  # save the orthogonalized diffraction pattern to VTK file
save_asint = (False)  # if True, the result will be saved as an array of integers (save space)

######################################
# define beamline related parameters #
######################################
beamline = 'SIXS_2019'  # name of the beamline, used for data loading and normalization by monitor
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

rocking_angle = "inplane"  # "outofplane" for a sample rotation around x outboard, "inplane" for a sample rotation
# around y vertical up, "energy"

follow_bragg = False  # only for energy scans, set to True if the detector was also scanned to follow the Bragg peak
specfile_name = None
# template for ID01: name of the spec file without '.spec'
# template for SIXS: full path of the alias dictionnary, typically root_folder + 'alias_dict_2020.txt'
# template for all other beamlines: ''

###############################
# detector related parameters #
###############################
detector = "Merlin"    # "Eiger2M", "Maxipix", "Eiger4M", "Merlin" or "Timepix"
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
hotpixels_file = "/home/david/Documents/PhDScripts/SIXS_June_2021/reconstructions/analysis/mask_merlin_better_flipped.npy"
flatfield_file = None  # root_folder + "flatfield_maxipix_8kev.npz"  # non empty file path or None
# template_imagefile = 'Pt_Al2O3_ascan_mu_%05d_R.nxs'
# template for ID01: 'data_mpx4_%05d.edf.gz' or 'align_eiger2M_%05d.edf.gz'
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
use_rawdata = False  # False for using data gridded in laboratory frame/ True for using data in detector frame
interp_method = 'linearization'  # 'xrayutilities' or 'linearization'
fill_value_mask = 0  # 0 (not masked) or 1 (masked). It will define how the pixels outside of the data range are
# processed during the interpolation. Because of the large number of masked pixels, phase retrieval converges better if
# the pixels are not masked (0 intensity imposed). The data is by default set to 0 outside of the defined range.
beam_direction = (1, 0, 0)  # beam direction in the laboratory frame (downstream, vertical up, outboard)
sample_offsets = (0, 0)  # tuple of offsets in degrees of the sample around (downstream, vertical up, outboard)
# convention: the sample offsets will be subtracted to the motor values
sdd = 1.18 # in m, sample to detector distance in m
energy = 8500  # np.linspace(11100, 10900, num=51)  # x-ray energy in eV
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
outofplane_angle = (
    20  # detector angle in deg (rotation around x outboard, typically delta),
)
# corrected for the direct beam position. Leave None to use the uncorrected position.
inplane_angle = (
    30
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
cch1 = 271  # direct beam vertical position in the full unbinned detector for xrayutilities 2D detector calibration
cch2 = 213  # direct beam horizontal position in the full unbinned detector for xrayutilities 2D detector calibration
detrot = 0  # detrot parameter from xrayutilities 2D detector calibration
tiltazimuth = 360  # tiltazimuth parameter from xrayutilities 2D detector calibration
tilt = 0  # tilt parameter from xrayutilities 2D detector calibration

##################################
# end of user-defined parameters #
##################################

from gwaihir.runner import preprocess

preprocess.preprocess_bcdi(
    scans,
    root_folder,
    save_dir,
    data_dirname, 
    sample_name,
    user_comment,
    debug,
    binning,
    flag_interact,
    background_plot,
    centering, 
    fix_bragg, 
    fix_size, 
    center_fft, 
    pad_size,
    normalize_flux, 
    mask_zero_event, 
    flag_medianfilter, 
    medfilt_order,
    reload_previous, 
    reload_orthogonal, 
    preprocessing_binning,
    save_rawdata, 
    save_to_npz, 
    save_to_mat, 
    save_to_vti, 
    save_asint,
    beamline, 
    actuators, 
    is_series, 
    custom_scan, 
    custom_images, 
    custom_monitor, 
    rocking_angle, 
    follow_bragg, 
    specfile_name,
    detector, 
    linearity_func, 
    roi_detector, 
    photon_threshold, 
    photon_filter, 
    background_file, 
    hotpixels_file, 
    flatfield_file, 
    template_imagefile, 
    nb_pixel_x, 
    nb_pixel_y,
    use_rawdata, 
    interp_method, 
    fill_value_mask, 
    beam_direction, 
    sample_offsets, 
    sdd, 
    energy, 
    custom_motors,
    align_q, 
    ref_axis_q, 
    outofplane_angle, 
    inplane_angle, 
    sample_inplane, 
    sample_outofplane, 
    offset_inplane, 
    cch1, 
    cch2, 
    detrot, 
    tiltazimuth, 
    tilt,
    GUI = False
    )


with open(README_file, 'a') as outfile:
    outfile.write("```")
