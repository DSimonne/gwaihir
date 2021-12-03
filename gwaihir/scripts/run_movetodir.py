#!/usr/bin/python3
import os
import shutil
import sys
import ast
import glob
import gwaihir
import inspect

"""Python script to move file from datadir to folder where it will be used by preprocess.bcdi
    Arg 1: directory
    Arg 2: Scan(s) number, list or single value
"""

# Print help
try:
    print('Data dir:',  sys.argv[1])
    print('Root folder:',  sys.argv[2])
    print('Scan:',  sys.argv[3])
except IndexError:
    print("""
        Arg 1: Original data directory 
        Arg 2: Path of EXISTING target directory (e.g. Pt_Al2O3/) (subdirectories {scan}/data & {scan}/pynx_raw will be updated/created)
        Arg 3: Scan number

        Looks recursively for one mu or omega scan including the scan number (glob.glob).
        """)
    exit()

OG_folder = sys.argv[1]
root_folder = sys.argv[2]
scan_name = sys.argv[3]

# Assign scan folder
scan_folder = root_folder + scan_name + "/"
print("Scan folder:", scan_folder)

# Assign preprocessing folder
preprocessing_folder = scan_folder + "preprocessing/"

# Assign postprocessing folder
postprocessing_folder = scan_folder + "postprocessing/"

# Assign data folder
data_folder = scan_folder + "data/"

# Create final directory, if not yet existing
if not os.path.isdir(root_folder):
    print(root_folder)
    full_path = ""
    for d in root_folder.split("/"):
        full_path += d + "/"
        try:
            os.mkdir(full_path)
        except FileExistsError:
            pass

print("Updating directories ...")

# Scan directory
try:
    os.mkdir(f"{scan_folder}")
    print(f"\tCreated {scan_folder}")
except FileExistsError:
    print(f"\t{scan_folder} exists")

# /data directory
try:
    os.mkdir(f"{data_folder}")
    print(f"\tCreated {data_folder}")
except FileExistsError:
    print(f"\t{data_folder} exists")

# /preprocessing directory
try:
    os.mkdir(f"{preprocessing_folder}")
    print(f"\tCreated {preprocessing_folder}")
except FileExistsError:
    print(f"\t{preprocessing_folder} exists")

# /postprocessing directory
try:
    os.mkdir(f"{postprocessing_folder}")
    print(f"\tCreated {postprocessing_folder}", end="\n\n")
except FileExistsError:
    print(f"\t{postprocessing_folder} exists", end="\n\n")
