#!/usr/bin/python3

"""Compute distance from gamma and delta"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import sys

# Print help
try:
    print ('Pos_x1:',  sys.argv[1])
    print ('Pos_y1:',  sys.argv[2])
    print ('Pos_x2:',  sys.argv[3])
    print ('Pos_y2:',  sys.argv[4])
    print ('Intervals:',  sys.argv[5])
    print ('Delta:',  sys.argv[6])
    print ('Gamma:',  sys.argv[7], end = "\n\n")

except IndexError:
    print("""
        Arg 1: Pos_x1
        Arg 2: Pos_y1
        Arg 3: Pos_x2
        Arg 4: Pos_y2
        Arg 5: Intervals
        Arg 6: Delta
        Arg 7: Gamma

        Remember to change the central pixel and wavelength for different experiments !!
        """)
    exit()

x1 = int(sys.argv[1])
y1 = int(sys.argv[2])

x2 = int(sys.argv[3])
y2 = int(sys.argv[4])

# give bragg x and bragg y in entry (pixel x and y where we want to have q)
# also need angles at central pixel

inplane_angle = float(sys.argv[5]) # Delta
outofplane_angle = float(sys.argv[6]) # Gamma

###################################
# define setup related parameters #
###################################
beam_direction = (1, 0, 0)  # beam along z
sample_offsets = None  # tuple of offsets in degrees of the sample around (downstream, vertical up, outboard)
# convention: the sample offsets will be subtracted to the motor values
directbeam_x = 271  # x horizontal,  cch2 in xrayutilities
directbeam_y = 236 #213   # y vertical,  cch1 in xrayutilities
direct_inplane = 0.0  # outer angle in xrayutilities
direct_outofplane = 0.0
sdd = 1.18  # sample to detector distance in m
energy = 8500  # in eV, offset of 6eV at ID01

# Compute
inplane_coeff = 1
outofplane_coeff = 1
pixelsize_x = 5.5e-05
pixelsize_y = 5.5e-05

wavelength = 12.398 * 1e-7 / energy
print("Lambda: ", wavelength, end = "\n\n")


vecteurs_q = {}
for j, (x, y) in enumerate([(x1, y1), (x2, y2)]):
    print("Computing q for position:", j)
    x_direct_0 = directbeam_x + inplane_coeff *\
                 (direct_inplane*np.pi/180*sdd/pixelsize_x)
    y_direct_0 = directbeam_y - outofplane_coeff *\
                 direct_outofplane*np.pi/180*sdd/pixelsize_y

    bragg_inplane = inplane_angle + inplane_coeff *\
                    (pixelsize_x*(x-x_direct_0)/sdd*180/np.pi)


    bragg_outofplane = outofplane_angle - outofplane_coeff *\
                       pixelsize_y*(y-y_direct_0)/sdd*180/np.pi

    print("Inplane angle:", bragg_inplane)
    print("Outofplane angle:", bragg_outofplane)

    # gamma is anti-clockwise
    kout = 2 * np.pi / wavelength * np.array(
        [np.cos(np.pi * bragg_inplane / 180) * np.cos(np.pi * bragg_outofplane / 180),  # z
         np.sin(np.pi * bragg_outofplane / 180),  # y
         np.sin(np.pi * bragg_inplane / 180) * np.cos(np.pi * bragg_outofplane / 180)])  # x

    kin = 2*np.pi/wavelength * np.asarray(beam_direction)

    q = (kout - kin) / 1e10  # convert from 1/m to 1/angstrom
    qnorm = np.linalg.norm(q)
    print("q vector in (z, y, x)", q)
    print("q norm:", qnorm, end = "\n\n")
    vecteurs_q[f"q_{j}"] = qnorm

t = abs((2*np.pi)/((vecteurs_q["q_0"]- vecteurs_q["q_1"])/int(sys.argv[5])))

print(f"Particle size: {t} (A)")