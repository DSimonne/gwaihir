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
    print ('OG data dir:',  sys.argv[1])
    print ('Target data dir:',  sys.argv[2])
    print ('Scan (s):',  sys.argv[3])
except IndexError:
    print("""
        Arg 1: Original data directory 
        Arg 2: Path of EXISTING target directory (e.g. Pt_Al2O3/) (subdirectories S{scan}/data & S{scan}/pynx_raw will be updated/created)
        Arg 3: Scan(s) number, list or single value

        Looks recursively for one mu or omega scan including the scan number (glob.glob).
        """)
    exit()

OG_folder = sys.argv[1]
TG_folder = sys.argv[2]
scan_list = sys.argv[3]

# transform string of list into python list object
if scan_list.startswith("["):
    scans = ast.literal_eval(scan_list)
    
else:
    scans = [scan_list]

# Load data
for scan in scans:
    print(f"Moving scan {scan}...")
    try:
        os.mkdir(f"{TG_folder}S{scan}")
        print(f"Created {TG_folder}S{scan}")
    except FileExistsError:
        print(f"{TG_folder}S{scan} exists")
        pass

    try:
        os.mkdir(f"{TG_folder}S{scan}/data")
        print(f"Created {TG_folder}S{scan}/data")
    except FileExistsError:
        print(f"{TG_folder}S{scan}/data exists")
        pass

    try:
        os.mkdir(f"{TG_folder}S{scan}/pynxraw")
        print(f"Created {TG_folder}S{scan}/pynxraw")
    except FileExistsError:
        print(f"{TG_folder}S{scan}/pynxraw exists")
        pass

    try:
        root_dir = inspect.getfile(gwaihir).split("__")[0]
        shutil.copy(f"{root_dir}gwaihir/data_files/pynx_run.txt", f"{TG_folder}S{scan}/pynxraw")
        print(f"Copied pynx-run-no-support.txt to {TG_folder}S{scan}/pynxraw")
    except FileExistsError:
        print(f"{TG_folder}S{scan}/pynxraw/pynx-run-no-support.txt exists")
        pass

    try:
        filename = glob.glob(f"{OG_folder}*mu*{scan}*.nxs", recursive=True)[0]
        print("Mu scan.")
        shutil.copy2(filename, f"{TG_folder}S{scan}/data")
        print(f"Copied {filename} to {TG_folder}S{scan}/data")
    except FileExistsError:
        print(f"{TG_folder}S{scan}/data/{filename} exists")
        pass
    except IndexError:
        try:
            filename = glob.glob(f"{OG_folder}*omega*{scan}*.nxs", recursive=True)[0]
            print("Omega scan.")
            shutil.copy2(filename, f"{TG_folder}S{scan}/data")
            print(f"Copied {filename} to {TG_folder}S{scan}/data")
        except FileExistsError:
            print(f"{TG_folder}S{scan}/data/{filename} exists")
            pass
        except IndexError:
            print("Not a mu or an omega scan.")
            pass
            
    print("\n")