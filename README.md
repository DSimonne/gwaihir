# Welcome to the BCDI part of this package

The goal of the module is to provide a way to process and analyse BCDI data either via jupyter notebook or the command line. The next step is to provide a GUI in jupyter notebook that could save much time during analysis while also being more user friendly.

It requires the python packages `bcdi` and `pynx`

The idea is to allow command line use of the package for fast analysis but also to use a GUI (for now jupyter notebooks) to allow new user to understand what is happening during the process but also to allow experienced users to finally tune their parameters and to offer a non "black-box" approach to the data analysis

## GUI
The use of python objects for the GUI and for the datasets could allow one to easily share the data analysis between one another.

This means that:
* A `Dataset` object could be reopened via the GUI, 
* The GUI should present several tabs, describing the data analysis (e.g. the parameters used during the analysis) so that one group could directly have a better understanding of the data history, these parameters would all be saved as attributes of the data object
* Guidelines for the analysis will be provided in the GUI, e.g. in a README tab
* These ideas apply the concept of thorondor, a similar package but for XANES data analysis (still in development)
* All the parameters related to the different beamlines will be in the code. It will only require a single widget to change the beamline, while keeping the same output architecture and output format, making it much easier to share and compare the results. Most of the code will be common to all the beamlines. Python power!
* Current hardcoded parameters (e.g. spec file location, ...) will not require the user to open scripts but just to edit a widget in notebook,
* One GUI for all the data, one GUI to rule them all


### Limitations
* Lots of coding to do, but we are a phd legion
* GUI can be RAM extensive, should be OK on clusters and recent computers
* Paraview, currently working on a debian package at SIXS (fred picca)


## Current workflow for BCDI data analysis 

We can define separate steps in the processing and analysis of BCDI data:
* preprocessing: masking, cropping, ... of the diffraction intensity  (BCDI package)
* phase retrieval: semi-automated, computes the amplitude and phase from the diffractogram intensity, depends on the input parameters, via jupyter notebook or terminal (PyNX package)
* postprocessing:
	* solution selection: semi-automated, using different criteria such as the free Log-likelihood (PyNX package) or the standard deviation of the reconstructed object's amplitude, via jupyter notebook or terminal (@Clatlan)
	* decomposition into modes of the best solutions to create one final object, via jupyter notebook or terminal (PyNX package)
	* orthogonalisation of the data into the crystal frame for object comparison (BCDI package)
	* data correction: compute the q vector at the centre of mass from the detector angles (BCDI package)
	* computation of the displacement and strain of the object from its phase and q vector (BCDI package)
* data analysis: 
	* visualisation via paraview, depends on isosurface, usually selected as the lower foot of the curve in the amplitude histogram 
	* facet retrieval, determine the size and normals to the facets on the isosurface from the normals to isosurface's points, depends on input parameters (FacetAnalyzer package in Paraview)
	* facet analysis, rotate the orientation matrix of the object so that the facet normals correspond to the normals to the crystallographic planes (e.g. 100 in crystal frame with y perpendicular to the interface becomes 111 ...) (@DupraxM and RichardMI)
	* plotting of mean strain and displacement values for each facet via a jupyter notebook


# How to use the terminal scripts for bcdi analysis

There are a few scripts that are meant to directly retrieve the paraview file (.vti) from the 3D intensity collected with the detector. The steps described above are automatic until facet retrieval.
If you want to use the scripts for BCDI at ID01 or SIXS, you need to:
* Have cloned and installed the repository
* Make sure that the `phdutils/bcdi/terminal_scripts` folder is in your PATH and PYTHONPATH
* Make sure that the first line of the scripts using the bcdi environment (`#!/users/simonne/anaconda3/envs/rnice.BCDI/bin/python3` for me) is replaced by an up-to-date BCDI environment for you. Sadly you do not have the authorization to use mine =D
* Respect the following architecture for the folders, there folder you need to create by hand:
	* ./Temperature/Condition/ParticleName/ScanNumber/...
	* e.g.: `./350/Helium/III_B2/`
* Make sure that the spec files, template image file, data folders, rocking angle, etc are exact in the scripts.
* You must launch all the scripts from the `./` folder, the following directories will be created.
* In the `./` folder, launch `quick_process_ID01.sh` with the first argument the path up to the Sxxx folder and as second argument the scan number, e.g. `quick_process_ID01.sh 350/Helium/III_B2/ 603`

In the scan folder, different subfodlers with be automatically generated, keping the same example:
* `/data` : contains the .nxs data (for SIXS, otherwise empty)
* `/pynxraw`:
    * `README_preprocess.md`: Preprocessing output logs    
    * `S603_data_before_masking_stack.npz`: 3D detector data before masking
    * `S603_maskpynx_align-q-y_norm_200_360_324_1_1_1.npz`: 3D detector mask
  	* `S603_pynx_align-q-y_norm_200_360_324_1_1_1.npz`: 3D detector intensity after masking
  	* `S603_pynx_align-q-y_norm_200_360_324_1_1_1.cxi`: hdf5 file containing metadata and detector data
  	* `finalmask_S603_align-q-y_norm_200_360_324_1_1_1.png`: image, sum of mask in each direction
  	* `finalsum_S603_align-q-y_norm_200_360_324_1_1_1.png`: image, sum of detector data in each direction
  	* `PhasingNotebook.ipynb`: notebook for phase retrieval, output in /reconstruction folder
  	* `pynx_run.txt` contains the input parameter for phase retrieval if performed via terminal
  	* `reconstructions/`:
  	    *  `modes.h5`: Output of decomposition into modes via jupyter notebook
  	    *  `*.cxi`: single solutions from phase retrieval
  	* `all/`: 
  	    *  `modes_job.h5`: Output of decomposition into modes via terminal
  	    *  `README_modes.md`: Logs of modes decomposition via terminal
  	    *  `README_pynx.md`: Logs of phase retrieval via terminal
  	    *  `*.cxi`: single solutions from phase retrieval
* `/postprocessing`:
    * `CompareFacetsEvolution.ipynb`: notebook for facet analysis
    * `/corrections`: contains the rocking curve, and the detector image at the maximum of the rocking curve
        * `central_slice.png`: Image, detector image at peak_pos
        * `rocking_curve.png`: Image, rocking curve and its interpolation
        * `README_correct_angles.md`:  Logs of data correction 
        * `correct_detector_data.npz`: Contains interpolated rocking curve, FWHH, peak_position, detector bragg angles, ... 
* `/result_crystal_frame`: 
    * `README_strain.md`: Logs of strain computation
    * `S603_ampdispstrain_mode_avg3_apodize_blackman_crystalframe.npz`: contains the orthogonalised data with amplitude, phase, displacement and strain
    * `S603_amp-disp-strain_mode_avg3_apodize_blackman_crystalframe.vti`: contains the orthogonalised data with amplitude, phase, displacement and strain, to be viewed via paraview