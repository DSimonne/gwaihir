#!/usr/bin/python3

# use on p9
import numpy as np
import tables as tb
import glob
import os
import operator
import sys

try:
    n_keep = int(sys.argv[1])
except:
    print("n_keep must be an integer")

# CXI files alreqdy filtered by LLK
cxi_files = glob.glob("*LLK*.cxi")

if cxi_files == []:
    print(f"No *LLK*.cxi files cwd.")

else:
    std = {}

    for filename in cxi_files:
        print("Computing standard deviation of object modulus for ", filename)
        with tb.open_file(filename, "r") as f:
            data = f.root.entry_1.image_1.data[:]
            std[filename] = np.std(np.abs(data))
            
    print(f"Keeping {n_keep} reconstructions ...")
    nb_files = len(cxi_files)
    sorted_dict = sorted(std.items(), key=operator.itemgetter(1))

    if n_keep < nb_files:
        for f, std in sorted_dict[n_keep:]:
            print(f"Removed scan {f}")
            os.remove(f)

        print("Filtered the reconstructionns.")
    else:
        print("n_keep is superior or equal to the number of reconstructions...")
