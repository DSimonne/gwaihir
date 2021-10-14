#!/usr/bin/python3

"""Part of script to allow systematic use
defining scan, root_folder, save_dir, comment, sample_name and template_imagefile

Remenber that you may have to change the mask, the central pixel, the rocking angle, the angles...

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gwaihir.sixs import ReadNxs4 as rd
import ast
import sys
import os
import glob

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

# Scan number
scan = int(sys.argv[2])

# Root folder
root_folder = os.getcwd() + "/" + sys.argv[1] 
print("Root folder:", root_folder)

# Scan folder
scan_folder = root_folder + f"S{scan}/"
print("Scan folder:", scan_folder)

# Data folder
data_folder = scan_folder + "data/" # folder of the experiment, where all scans are stored
print("Data folder:", data_folder)

# Sample name
sample_name = "S"  # str or list of str of sample names (string in front of the scan number in the folder name).

# Template imagefile
try:
    filename = glob.glob(f"{data_folder}*mu*{scan}*")[0]
    template_imagefile = filename.split("/data/")[-1].split("%05d"%scan)[0] +"%05d_R.nxs"
    print("Template: ", template_imagefile)

except IndexError:
    try:
        filename = glob.glob(f"{data_folder}*omega*{scan}*")[0]
        template_imagefile = filename.split("/data/")[-1].split("%05d"%scan)[0] +"%05d_R.nxs"
        print("Template: ", template_imagefile)

    except IndexError:
        # Not SixS data
        template_imagefile = root_folder + 'mpx/data_mpx4_%05d.edf'# july and september 2021

# Saving directory
save_dir = scan_folder + "postprocessing/corrections/"  # images will be saved here, leave it to None otherwise (default to data directory's parent)

# CSV file if iterating on scans
csv_file = root_folder + "/metadata.csv"
print("Csv file:", csv_file)

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
# root_folder = "/home/david/Documents/PhD_local/PhDScripts/SIXS_January_2021/Pt_Al2O3/Ar/"
# sample_name = "S"
filtered_data = False  # set to True if the data is already a 3D array, False otherwise
# Should be the same shape as in specfile
peak_method = 'maxcom'  # Bragg peak determination: 'max', 'com' or 'maxcom'.
normalize_flux = 'skip'  # 'monitor' to normalize the intensity by the default monitor values, 'skip' to do nothing
debug = False  # True to see more plots

######################################
# define beamline related parameters #
######################################
beamline = ('SIXS_2019')  # name of the beamline, used for data loading and normalization by monitor
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
rocking_angle = "inplane"  # "outofplane" or "inplane"
specfile_name = None
# template for ID01: name of the spec file without '.spec'
# template for SIXS_2018: full path of the alias dictionnary 'alias_dict.txt', typically: root_folder + 'alias_dict.txt'
# template for all other beamlines: ''

#############################################################
# define detector related parameters and region of interest #
#############################################################
detector = "Merlin"    # "Eiger2M" or "Maxipix" or "Eiger4M"
x_bragg = None  # horizontal pixel number of the Bragg peak, can be used for the definition of the ROI
y_bragg = None   # vertical pixel number of the Bragg peak, can be used for the definition of the ROI
roi_detector = None
# [y_bragg - 400, y_bragg + 400, x_bragg - 400, x_bragg + 400]  #
# [y_bragg - 290, y_bragg + 350, x_bragg - 350, x_bragg + 350]  # Ar  # HC3207  x_bragg = 430
# leave it as None to use the full detector. Use with center_fft='do_nothing' if you want this exact size.
high_threshold = 1000000  # everything above will be considered as hotpixel
hotpixels_file = "/home/david/Documents/PhDScripts/SIXS_June_2021/reconstructions/analysis/mask_merlin_better_flipped.npy"
# hotpixels_file = "/home/experiences/sixs/simonne/Documents/SIXS_June_2021/ruche_dir/reconstructions/analysis/mask_merlin_better_flipped.npy"
# hotpixels_file = "/home/experiences/sixs/simonne/Documents/SIXS_June_2021/masks/mask_merlin_better.npy"
# hotpixels_file = "/home/experiences/sixs/simonne/Documents/SIXS_Jan_2021/masks/mask_merlin.npy"  # root_folder + 'hotpixels_HS4670.npz'  # non empty file path or None
flatfield_file = None  # root_folder + "flatfield_maxipix_8kev.npz"  # non empty file path or None
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
sample_offsets = None  # tuple of offsets in degrees of the sample around (downstream, vertical up, outboard)
# convention: the sample offsets will be subtracted to the motor values
directbeam_x = 271  # x horizontal,  cch2 in xrayutilities
directbeam_y = 213 # SIXS jan 2021   # y vertical,  cch1 in xrayutilities
# directbeam_y = 236 # SIXS june 2021   # y vertical,  cch1 in xrayutilities
direct_inplane = 0.0  # outer angle in xrayutilities
direct_outofplane = 0.0
sdd = 1.18  # sample to detector distance in m
energy = 8500  # in eV, offset of 6eV at ID01

################################################
# parameters related to temperature estimation #
################################################
get_temperature = True  # True to estimate the temperature using the reference spacing of the material. Only for Pt.
reflection = np.array([1, 1, 1])  # measured reflection, use for estimating the temperature
# reference_spacing = None  # for calibrating the thermal expansion, if None it is fixed to Pt 3.9236/norm(reflection)
# reference_spacing = 2.254761  # d_111 at room temperature, from scan 1353, with corrected angles, SIXS jan
reference_spacing = 2.269545  # d_111 at room temperature, from scan 670, with corrected angles, SIXS june
reference_temperature = None  # used to calibrate the thermal expansion, if None it is fixed to 293.15K (RT)

##########################################################
# end of user parameters
##########################################################

# Run file
from gwaihir.runner import correct_angles

metadata = correct_angles.correct_angles_detector(
    filename,
    direct_inplane,
    direct_outofplane,
    get_temperature,
    reflection,
    reference_spacing,
    reference_temperature,
    high_threshold,
    save_dir,
    scan,
    root_folder,
    sample_name,
    filtered_data,
    peak_method,
    normalize_flux,
    debug,
    beamline,
    actuators,
    is_series,
    custom_scan,
    custom_images,
    custom_monitor,
    custom_motors,
    rocking_angle,
    specfile_name,
    detector,
    x_bragg,
    y_bragg,
    roi_detector,
    hotpixels_file,
    flatfield_file,
    template_imagefile,
    beam_direction,
    sample_offsets,
    directbeam_x,
    directbeam_y,
    sdd,
    energy,
    GUI = False
    )

################################################ END OF BCDI SCRIPT ################################################################

# Added script
sys.stdout.close()
sys.stdout = stdoutOrigin

# Save rocking curve data
np.savez(f"{scan_folder}postprocessing/interpolated_rocking_curve.npz",
    tilt_values = metadata["tilt_values"],
    rocking_curve = metadata["rocking_curve"],
    interp_tilt = metadata["interp_tilt"],
    interp_curve = metadata["interp_curve"],
    )

# Save in a csv file
try:
    if beamline == "SIXS_2019":
        # Load dataset, quite slow 
        data = rd.DataSet(filename)

        ## Add new data
        temp_df = pd.DataFrame([[
            scan,
            metadata["q"][0], metadata["q"][1], metadata["q"][2], metadata["qnorm"], metadata["dist_plane"],
            metadata["bragg_inplane"], metadata["bragg_outofplane"],
            metadata["bragg_x"], metadata["bragg_y"],
            data.x[0], data.y[0], data.z[0], data.mu[0], data.delta[0], data.omega[0],
            data.gamma[0], data.gamma[0] - data.mu[0], 
            (data.mu[-1] - data.mu[-0]) / len(data.mu), data.integration_time[0], len(data.integration_time), 
            metadata["interp_fwhm"], metadata["COM_rocking_curve"],
            # data.ssl3hg[0], data.ssl3vg[0], 
            # data.ssl1hg[0], data.ssl1vg[0]
            ]],
            columns = [
                "scan",
                "qx", "qy", "qz", "q_norm", "d_hkl", 
                "inplane_angle", "out_of_plane_angle",
                "bragg_x", "bragg_y",
                "x", "y", "z", "mu", "delta", "omega",
                "gamma", 'gamma-mu',
                "step size", "integration time", "steps", 
                "FWHM", "COM_rocking_curve",
            #     "ssl3hg", "ssl3vg", 
            #     "ssl1hg", "ssl1vg", 
            ])
    else:
        ## Add new data
        temp_df = pd.DataFrame([[
            scan,
            metadata["q"][0], metadata["q"][1], metadata["q"][2], metadata["qnorm"], metadata["dist_plane"],
            metadata["bragg_inplane"], metadata["bragg_outofplane"],
            metadata["bragg_x"], metadata["bragg_y"],
            metadata["interp_fwhm"], metadata["COM_rocking_curve"],
            ]],
            columns = [
                "scan",
                "qx", "qy", "qz", "q_norm", "d_hkl", 
                "inplane_angle", "out_of_plane_angle",
                "bragg_x", "bragg_y",
                "FWHM", "COM_rocking_curve",
            ])

    # Load all the logs
    try:
        df = pd.read_csv(csv_file)

        # Replace old data linked to this scan, no problem if this row does not exist yet
        indices = df[df['scan'] == scan].index
        df.drop(indices , inplace=True)

        result = pd.concat([df, temp_df])

    except FileNotFoundError:
        result = temp_df

    # Save
    result.to_csv(csv_file, index = False)
    print(f"Saved logs in {csv_file}")

# except AttributeError:
#     print("Could not extract metadata from dataset ...")

except Exception as e:
    raise e

with open(README_file, 'a') as outfile:
    outfile.write("```")

# End of added script

print("End of script \n")
plt.close()
plt.ioff()