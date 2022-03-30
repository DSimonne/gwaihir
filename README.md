# Welcome 

Contact : david.simonne@universite-paris-saclay.fr

You can install gwaihir via the `setup.py` script (`pip install .`)

Gwaihir is also avaible on pypi.org, each new stable version from the master branch is uploaded: `https://pypi.org/project/gwaihir/`
On the contrary, if you follow the github changes on the you will have the latest updates.

# GUI Preview:
## Pre-processing data
https://user-images.githubusercontent.com/51970962/154160601-f3e7878a-d2c6-4560-95e5-adf7087f59ab.mp4

## Phase retrieval
https://user-images.githubusercontent.com/51970962/154160830-f3c6460b-14e5-4bcc-99f5-e8691278a4e9.mp4

## Data plotting
https://user-images.githubusercontent.com/51970962/154160549-c5caea1b-afa0-4a29-a5a8-aff8a1a5158b.mp4

## Post-processing
https://user-images.githubusercontent.com/51970962/154236802-24643473-1ee9-4d01-823c-beca07ea1c58.mp4

## Facet analysis
No video yet.

## CXI file
An example file can be downloaded at: https://www.dsimonne.eu/Attachments/align_031968.cxi

# Using `Gwaihir` only as a plotting tool in Jupyter Notebook
![image](https://user-images.githubusercontent.com/51970962/157677934-d6983756-48d3-4a1d-8394-a86f0d2b721e.png)

# Installing different packages

* First, I advise you to create a `/Packages` directory to keep these.
* Secondly, I advise you to create a virtual environment to help with debogging, and so that once everything works, you don't update a package by mistake. To do so please follow the following steps:

## Create a virtual environment

* `mkdir py38-env`
* `cd py38-env/`
* `python3.8 -m venv .`
* `source bin/activate` # To activate the environment

Then you should create an alias such as: `alias source_p9="source /home/user/py38-env/bin/activate"`

## Install gwaihir
* `cd /Packages`
* `git clone https://github.com/DSimonne/gwaihir.git`
* `cd gwaihir`
* `source_p9`
* `pip install .`

## Install PyNX
* Send a thank you email to Vincent Favre-Nicolin =D
* `cd /Packages`
* `mkdir PyNX_install`
* `cd PyNX_install/`
* `curl -O http://ftp.esrf.fr/pub/scisoft/PyNX/pynx-devel-nightly.tar.bz2`      # Installation details within install-pynx-venv.sh
* `source_p9`
* `pip install pynx-devel-nightly.tar.bz2[cuda,gui,mpi]`                  # Install with extras cuda, mpi, cdi
* cite `PyNX: high-performance computing toolkit for coherent X-ray imaging based on operators is out: J. Appl. Cryst. 53 (2020), 1404`, also available as `arXiv:2008.11511`


## Install bcdi
* Send a thank you email to Jerome Carnis =D
* `cd /Packages`
* `git clone https://github.com/carnisj/bcdi.git`
* `cd bcdi`
* `source_p9`
* `pip install .`
* cite `DOI: 10.5281/zenodo.3257616`

## Install facet-analyser (Debian 11 only)
* Send a thank you email to Fred Picca =D
* `cd /Packages`
* `git clone https://salsa.debian.org/science-team/facet-analyser.git`
* `cd facet-analyser`
* `git checkout`
* `sudo mk-build-deps -i`
* Make sure that you have qt installed, for me I had to install `libqt5opengl5-dev` (debian-testing)
* `debuild -b`
* if the package creation fail, try to ignore the test in /debian/rules (line 19)
* `sudo debi`
* The package is now installed. You can check the locations of its files with the command `dpkg -L facet-analyser`
* You should see a file named `/usr/lib/x86_64-linux-gnu/paraview-5.9/plugins/FacetAnalyser/FacetAnalyser.so`
* Now launch `/usr/bin/paraview` (if not installed yet, good luck, refer to `https://www.paraview.org/Wiki/ParaView:Build_And_Install#Installing`)
* In paraview, go to Tools > Manage Plugins > Load New
* Here type the path to the plugin that was printed with the `dpkg -L facet-analyser` command.
* Feel free to add it to `/usr/bin/plugin` so that it is loaded automatically.
* cite `Grothausmann, R. (2015). Facet Analyser : ParaView plugin for automated facet detection and measurement of interplanar angles of tomographic objects. March.`

# Clusters at ESRF

## slurm
`ssh -X <login>@slurm-nice-devel`

Demande GPU

`srun -N 1 --partition=p9gpu --gres=gpu:1 --time=01:00:00 --pty bash`

### Environments on slurm
* python3: your personal environemnt
* p9_3.8_dev : optimised for BCDI, gwaihir and PyNX, development version, `source /data/id01/inhouse/david/py38-dev/bin/activate`
* p9_3.8_stable : optimised for BCDI, gwaihir and PyNX, stable version, `source /data/id01/inhouse/david/py38-stable/bin/activate`
* p9.pynx-devel : fonctionne pour pynx, frequently updated : `source /sware/exp/pynx/devel.p9/bin/activate`

You are not allowed to modify these environments but you should link a kernel if you wish to use them in jupyter.

To do so:
* `source_p9`
* `ipython kernel install --name "p9_gwaihir --user`

## Connect with ssh without using password (mandatory for batch jobs)
* Login into slurm (make sure that you asked for a GPU)
* Open a terminal (new -> terminal)

Enter the following commands (replace `<username>` with your username, for me it is simonne)
* `cd`
* `ssh-keygen -t rsa` (press enter when prompted, ~ 3 times)
* `ssh <username>@slurm-nice-devel mkdir -p .ssh`
* `cat .ssh/id_rsa.pub | ssh <username>@slurm-nice-devel 'cat >> .ssh/authorized_keys'`

You should not need a password anymore when login into slurm, make sure it is the case by typing
* `ssh <username>@slurm-nice-devel`

# Clusters at SOLEIL

## GRADES

## SixS3