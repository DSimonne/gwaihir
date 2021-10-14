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
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import tkinter as tk
from tkinter import filedialog
import sys
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.preprocessing.preprocessing_utils as pru
from bcdi.experiment.detector import Detector
from bcdi.experiment.setup import Setup
import bcdi.utils.utilities as util

helptext = """
Calculate exact inplane and out-of-plane detector angles from the direct beam and Bragg peak positions,
based on the beamline geometry.

Supported beamlines: ESRF ID01, PETRAIII P10, SOLEIL SIXS, SOLEIL CRISTAL.

For Pt samples it gives also an estimation of the temperature based on the thermal expansion.

Input: direct beam and Bragg peak position, sample to detector distance, energy
Output: corrected inplane, out-of-plane detector angles for the Bragg peak.
"""

"""Part of script to allow systematic use
defining scan, root_folder, save_dir, comment, sample_name and template_imagefile

Remenber that you may have to change the mask, the central pixel, the rocking angle, the angles...

"""

import pandas as pd
import ast
import os
import glob
import pickle

# Print help
try:
    print ('Data dir:',  sys.argv[1])
    print ('Scan:',  sys.argv[2])
except IndexError:
    print("""
        Arg 1: Path of target directory (before /S{scan} ... )
        Arg 2: Scan(s) number, list or single value
        """)
    exit()

scan = int(sys.argv[2])

particle = sys.argv[1].split("/")[-2]
print("Particle (from file name):", particle)

condition = sys.argv[1].split("/")[-3]
print("Condition (from file name):", condition)

given_temperature = sys.argv[1].split("/")[-4]
print("Given temp (from file name):", given_temperature)

# folder of the experiment, where all scans are stored
root_folder = os.getcwd() + "/" + sys.argv[1] 
print("Root folder:", root_folder)

#Scan folder
scan_folder = root_folder + f"S{scan}/"
print("Scan folder:", scan_folder)

# Data folder
data_folder = scan_folder + "data/" # folder of the experiment, where all scans are stored
print("Data folder:", data_folder)

sample_name = "S"  # str or list of str of sample names (string in front of the scan number in the folder name).

# Saving directory
save_dir = scan_folder + "postprocessing/corrections/"  # images will be saved here, leave it to None otherwise (default to data directory's parent)

# CSV file if iterating on scans
csv_file = os.getcwd() + "/scan_data.csv"

README_file = f"{save_dir}README_correct_angles.md"
print("Save folder:", save_dir)

try:
    os.mkdir(scan_folder + "postprocessing")
except:
    pass

try:
    os.mkdir(save_dir)
except:
    pass

# Save all the prints from the script
stdoutOrigin=sys.stdout

with open(README_file, 'w') as outfile:
    outfile.write("```bash\n")
sys.stdout = open(README_file, "a")

"""end of personal script"""

# scan = 1353
# root_folder = "/data/id01/inhouse/data/IHR/hc4050/id01/"  # folder of the experiment, where all scans are stored
# root_folder = "/data/id01/inhouse/data/IHR/hc4050_a/id01/"  # folder of the experiment, where all scans are stored
# root_folder = "/data/id01/inhouse/data/IHR/hc4050_a/id01/test/BCDI_2021_07_26_165851/"  # folder of the experiment, 
# root_folder = "/data/visitor/hc4534/id01/B8_S1_P2/BCDI_2021_09_02_145714/"  # folder of the experiment, up to spec file
root_folder = "/data/visitor/hc4534/id01/B8_S1_P2/BCDI_2021_09_02_203654/"  # folder of the experiment, up to spec filesave_dir = None  # images will be saved here, leave it to None otherwise
# sample_name = "S"
filtered_data = False  # set to True if the data is already a 3D array, False otherwise
# Should be the same shape as in specfile
peak_method = 'maxcom'  # Bragg peak determination: 'max', 'com' or 'maxcom'.
normalize_flux = 'skip'  # 'monitor' to normalize the intensity by the default monitor values, 'skip' to do nothing
debug = False  # True to see more plots

######################################
# define beamline related parameters #
######################################
beamline = 'ID01'  # name of the beamline, used for data loading and normalization by monitor
# supported beamlines: 'ID01', 'SIXS_2018', 'SIXS_2019', 'CRISTAL', 'P10'
actuators = None  # {'rocking_angle': 'actuator_1_3'}
# Optional dictionary that can be used to define the entries corresponding to actuators in data files
# (useful at CRISTAL where the location of data keeps changing)
# e.g.  {'rocking_angle': 'actuator_1_3', 'detector': 'data_04', 'monitor': 'data_05'}
is_series = True  # specific to series measurement at P10
custom_scan = False  # True for a stack of images acquired without scan, e.g. with ct in a macro (no info in spec file)
custom_images = None  # list of image numbers for the custom_scan
custom_monitor = None  # monitor values for normalization for the custom_scan
custom_motors = None
# {"eta": np.linspace(16.989, 18.989, num=100, endpoint=False), "phi": 0, "nu": -0.75, "delta": 36.65}
# ID01: eta, phi, nu, delta
# CRISTAL: mgomega, gamma, delta
# P10: om, phi, chi, mu, gamma, delta
# SIXS: beta, mu, gamma, delta
rocking_angle = "outofplane"  # "outofplane" or "inplane"
# specfile_name = "spec/2021_07_20_085405_ni" #'analysis/alias_dict_2021.txt'# July 2021
# specfile_name = "spec/2021_07_24_083204_test" #'analysis/alias_dict_2021.txt'# July 
# specfile_name = "spec/BCDI_2021_07_26_165851" #'analysis/alias_dict_2021.txt'# July 
# specfile_name = "spec/BCDI_2021_09_02_145714" #'analysis/alias_dict_2021.txt'# september
specfile_name = "spec/BCDI_2021_09_02_203654" #'analysis/alias_dict_2021.txt'# september
# template for SIXS_2018: full path of the alias dictionnary 'alias_dict.txt', typically: root_folder + 'alias_dict.txt'
# template for all other beamlines: ''

#############################################################
# define detector related parameters and region of interest #
#############################################################
detector = "Maxipix"    # "Eiger2M" or "Maxipix" or "Eiger4M"
x_bragg = None  # horizontal pixel number of the Bragg peak, can be used for the definition of the ROI
y_bragg = None   # vertical pixel number of the Bragg peak, can be used for the definition of the ROI
roi_detector = None
# [y_bragg - 400, y_bragg + 400, x_bragg - 400, x_bragg + 400]  #
# [y_bragg - 290, y_bragg + 350, x_bragg - 350, x_bragg + 350]  # Ar  # HC3207  x_bragg = 430
# leave it as None to use the full detector. Use with center_fft='do_nothing' if you want this exact size.
high_threshold = 1000000  # everything above will be considered as hotpixel
hotpixels_file = None
# hotpixels_file = "/home/experiences/sixs/simonne/Documents/SIXS_June_2021/ruche_dir/reconstructions/analysis/mask_merlin_better_flipped.npy"
# hotpixels_file = "/home/experiences/sixs/simonne/Documents/SIXS_June_2021/masks/mask_merlin_better.npy"
# hotpixels_file = "/home/experiences/sixs/simonne/Documents/SIXS_Jan_2021/masks/mask_merlin.npy"  # root_folder + 'hotpixels_HS4670.npz'  # non empty file path or None
flatfield_file = None  # root_folder + "flatfield_maxipix_8kev.npz"  # non empty file path or None
# template_imagefile = root_folder + 'detector/2021_07_20_085405_ni/data_mpx4_%05d.edf.gz'
# template_imagefile = root_folder + 'detector/2021_07_24_072032_b8_s1_p2/data_mpx4_%05d.edf.gz'
# template_imagefile = root_folder + 'mpx/data_mpx4_%05d.edf.gz'# july and september 2021
template_imagefile = root_folder + 'mpx/data_mpx4_%05d.edf'# july and september 2021
# template_imagefile ="Pt_Al2O3_ascan_mu_%05d_R.nxs"
# template for ID01: 'data_mpx4_%05d.edf.gz' or 'align_eiger2M_%05d.edf.gz'
# template for SIXS_2018: 'align.spec_ascan_mu_%05d.nxs'
# template for SIXS_2019: 'spare_ascan_mu_%05d.nxs'
# template for Cristal: 'S%d.nxs'
# template for P10: '_master.h5'
# template for NANOMAX: '%06d.h5'
# template for 34ID: 'Sample%dC_ES_data_51_256_256.npz'

###################################
# define setup related parameters #
###################################
beam_direction = (1, 0, 0)  # beam along z
sample_offsets = (-0.0011553664, 0, 0) # tuple of offsets in degrees of the sample around (downstream, vertical up, outboard)
# convention: the sample offsets will be subtracted to the motor values
directbeam_x = 159.555  # direct beam vertical position in the full unbinned detector for xrayutilities 2D detector calibration
directbeam_y = 729.561  # direct beam horizontal position in the full unbinned detector for xrayutilities 2D detector calibration
direct_inplane = 0.0  # outer angle in xrayutilities
direct_outofplane = 0.0
sdd = 0.83  # sample to detector distance in m
energy = 12994  # in eV, offset of 6eV at ID01

################################################
# parameters related to temperature estimation #
################################################
get_temperature = False  # True to estimate the temperature using the reference spacing of the material. Only for Pt.
reflection = np.array([1, 1, 1])  # measured reflection, use for estimating the temperature
# reference_spacing = None  # for calibrating the thermal expansion, if None it is fixed to Pt 3.9236/norm(reflection)
# reference_spacing = 2.254761  # d_111 at room temperature, from scan 1353, with corrected angles, SIXS jan
reference_spacing = 2.269545  # d_111 at room temperature, from scan 670, with corrected angles, SIXS june
reference_temperature = None  # used to calibrate the thermal expansion, if None it is fixed to 293.15K (RT)

##########################################################
# end of user parameters
##########################################################


# Run file