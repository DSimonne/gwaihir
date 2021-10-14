# -*- coding: utf-8 -*-
import hdf5plugin
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import scipy, numpy, pylab
from numpy import *
from pylab import *
from scipy import *
import fabio
import glob
import scipy.ndimage
from scipy.ndimage.measurements import center_of_mass
import matplotlib.colors as col


"""

Attached is a python's file that allows to compare reconstructions:

- by making statistics (comparison of LKK, LKKf, nb of pixels of support, etc) between the reconstructions

- plotting the modulus/phase of the reconstruction as *.png files, beacuse the use of silx view is something painful.


"""

###########Define a colormap
cdict = {'red':  ((0.0, 1.0, 1.0),
                  (0.11, 0.0, 0.0),
                  (0.36, 0.0, 0.0),
                  (0.62, 1.0, 1.0),
                  (0.87, 1.0, 1.0),
                  (1.0, 0.0, 0.0)),
          'green': ((0.0, 1.0, 1.0),
                  (0.11, 0.0, 0.0),
                  (0.36, 1.0, 1.0),
                  (0.62, 1.0, 1.0),
                  (0.87, 0.0, 0.0),
                  (1.0, 0.0, 0.0)),
          'blue': ((0.0, 1.0, 1.0),
                  (0.11, 1.0, 1.0),
                  (0.36, 1.0, 1.0),
                  (0.62, 0.0, 0.0),
                  (0.87, 0.0, 0.0),
                  (1.0, 0.0, 0.0))}
my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)


def make_segmented_cmap(): 
    white = '#ffffff'
    black = '#000000'
    red = '#ff0000'
    blue = '#0000ff'
    anglemap = col.LinearSegmentedColormap.from_list(
        'anglemap', [black, red, white, blue, black], N=256, gamma=1)
    return anglemap
segmented_cmap = make_segmented_cmap()

###########open files / case of *.cxi files
# path_files & savedir to be changed
path_files = '/data/id01/inhouse/richard/LTP-HC4050/BCDI/RESULTS/VFN/test_inversion_111'
path_files = '/data/id01/inhouse/data/IHR/ihhc3567/id01/analysis/Results/S183/pynxraw'
savedir = '/data/id01/inhouse/data/IHR/ihhc3567/id01/analysis/Res/'
files = glob.glob(path_files+'/*.cxi' and path_files+'/*LLK*')

###########read files / case of *.cxi files
ii, total = 0, 0
for name in files:
    data = h5py.File(name, "r")
    reconst = data['entry_1/data_1/data'][:]
    ii += 1
    total += np.max(np.abs(reconst))
    data.close()
average_max_value = np.int(total/ii)
print('average max. value:',average_max_value)


###########create *.png image with statistics
fig, axes = plt.subplots(3,2,figsize=(10,10))
print('run','max_reconst','obj_std','obj_average','LLKf','LLK','nb_point_support')
for name in files:
    data = h5py.File(name, "r")
    nb_point_support = data['entry_1/image_1/process_1/results/nb_point_support'][...]
    run = int(name.split('/')[-1].split('Run')[1].split('_')[0])
    LLKf = data['entry_1/image_1/process_1/results/free_llk_poisson'][...] # Poisson free log-likelihood
    LLK = data['entry_1/image_1/process_1/results/llk_poisson'][...] # Poisson log-likelihood
    reconst = data['entry_1/data_1/data'][:]
    modulus = np.abs(reconst)
    support = data['entry_1/image_1/support'][:]
    modulus = modulus[support>0]
    histo = scipy.ndimage.histogram(np.abs(modulus),0,average_max_value,200)
    val = np.array(range(200))*average_max_value/200.
    obj_average = sum(histo*val)/sum(histo)
    obj_std = np.sqrt(sum(histo*(val - obj_average)**2)/sum(histo))
    print(run,' ',np.max(np.abs(reconst)),' ',obj_std,' ',obj_average,' ',LLKf,' ', LLK, ' ',nb_point_support,' ')
    axes[0,0].scatter(run,LLKf, c = 'red', marker = '+', linewidth = 3)
    axes[0,0].grid()
    axes[0,0].set_ylabel('LLKfree')
    axes[0,1].scatter(run,LLK, c = 'red', marker = '+', linewidth = 3)
    axes[0,1].set_ylabel('LLK')
    axes[0,1].grid()
    axes[1,0].scatter(run,obj_average, c = 'red', marker = '+', linewidth = 3)
    axes[1,0].grid()
    axes[1,0].set_ylabel('Modulus obj. avg.')
    axes[1,1].scatter(run,obj_std, c = 'red', marker = '+', linewidth = 3)
    axes[1,1].set_ylabel('Modulus obj. std deviation')
    axes[1,1].grid()
    axes[2,0].scatter(run,np.max(np.abs(reconst)), c = 'red', marker = '+', linewidth = 3)
    axes[2,0].grid()
    axes[2,0].set_xlabel('Run')
    axes[2,0].set_ylabel('Modulus obj. max.')
    axes[2,1].scatter(run,nb_point_support, c = 'red', marker = '+', linewidth = 3)
    axes[2,1].set_ylabel('Number points of support')
    axes[2,1].set_xlabel('Run')
    axes[2,1].grid()
plt.tight_layout()
plt.savefig(savedir+'/Stat_reconst.png')


###########create folder with modulus & phase results of the object
for name in files:
    data = h5py.File(name, "r")
    run = int(name.split('/')[-1].split('Run')[1].split('_')[0])
    reconst = data['entry_1/data_1/data'][:]
    support = data['entry_1/image_1/support'][:]
    modulus = np.abs(reconst)
    phase = np.angle(reconst)
    crop_output = data['entry_1/image_1/process_1/configuration/crop_output'][...]
    cx,cy,cz = center_of_mass(support)
    if(crop_output):
        xmin,ymin,zmin = 0,0,0
        xmax,ymax,zmax = support.shape
    else:
        x,y,z = support.sum(axis=1).sum(axis=1)[0:int(cx)],support.sum(axis=2).sum(axis=0)[0:int(cy)],support.sum(axis=0).sum(axis=0)[0:int(cz)]
        xmin,ymin,zmin = (x!=0).argmax(axis=0) - 5,(y!=0).argmax(axis=0) - 5,(z!=0).argmax(axis=0) - 5
        x,y,z = support.sum(axis=1).sum(axis=1)[int(cx):-1],support.sum(axis=2).sum(axis=0)[int(cy):-1],support.sum(axis=0).sum(axis=0)[int(cz):-1]
        xmax,ymax,zmax = (x==0).argmax(axis=0) + int(cx) + 5,(y==0).argmax(axis=0) + int(cy) + 5,(z==0).argmax(axis=0) + int(cz) + 5
    plt.figure(figsize=(12,10))
    plt.subplot(3,3,1)
    plt.imshow(modulus[xmin:xmax,ymin:ymax,zmin:zmax].sum(axis=0))
    plt.xlabel("z")
    plt.ylabel("y")
    plt.colorbar()
    plt.axis('tight')
    plt.grid()
    plt.subplot(3,3,2)
    plt.imshow(modulus[xmin:xmax,ymin:ymax,zmin:zmax].sum(axis=1))
    plt.xlabel("z")
    plt.ylabel("x")
    plt.colorbar()
    plt.axis('tight')
    plt.title("Integrated modulus")
    plt.grid()
    plt.subplot(3,3,3)
    plt.imshow(modulus[xmin:xmax,ymin:ymax,zmin:zmax].sum(axis=2))
    plt.xlabel("y")
    plt.ylabel("x")
    plt.colorbar()
    plt.axis('tight')
    plt.grid()
    plt.subplot(3,3,4)
    plt.imshow(modulus[int(cx),ymin:ymax,zmin:zmax])
    plt.xlabel("z")
    plt.ylabel("y")
    plt.colorbar()
    plt.axis('tight')
    plt.grid()
    plt.subplot(3,3,5)
    plt.imshow(modulus[xmin:xmax,int(cy),zmin:zmax])
    plt.title("Modulus @ middle slice")
    plt.xlabel("z")
    plt.ylabel("x")
    plt.colorbar()
    plt.axis('tight')
    plt.grid()
    plt.subplot(3,3,6)
    plt.imshow(modulus[xmin:xmax,ymin:ymax,int(cz)])
    plt.xlabel("y")
    plt.ylabel("x")
    plt.colorbar()
    plt.axis('tight')
    plt.grid()
    plt.subplot(3,3,7)
    plt.imshow((phase*support)[int(cx),ymin:ymax,zmin:zmax],cmap = segmented_cmap)
    plt.xlabel("z")
    plt.ylabel("y")
    plt.colorbar()
    plt.axis('tight')
    plt.grid()
    plt.subplot(3,3,8)
    plt.imshow((phase*support)[xmin:xmax,int(cy),zmin:zmax],cmap = segmented_cmap)
    plt.title("Phase @ middle slice")
    plt.xlabel("z")
    plt.ylabel("x")
    plt.colorbar()
    plt.axis('tight')
    plt.grid()
    plt.subplot(3,3,9)
    plt.imshow((phase*support)[xmin:xmax,ymin:ymax,int(cz)],cmap = segmented_cmap)
    plt.xlabel("y")
    plt.ylabel("x")
    plt.colorbar()
    plt.axis('tight')
    plt.grid()
    plt.tight_layout()
    if not os.path.exists(savedir+'/Images/'):
        os.makedirs(savedir+'/Images/')
    plt.savefig(savedir+'/Images/Reconst_run'+str(run)+'.png')


