# Welcome 

Contact : david.simonne@universite-paris-saclay.fr

You can install gwaihir via the setup.py script, so that you can use via a package after in python, see below

Gwaihir is also avaible on pypi.org, each new version from the master branch is uploaded: `https://pypi.org/project/gwaihir/`
On the contrary, if you follow the github changes on the you will have the latest updates.

# Installing different packages

First, I advise you to create a Packages directory to keep these.
Secondly, I advise you to create a virtual environment to help with debogging, and so that once everything works, you don't update a package by mistake. To do so please follow the following steps:

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
* `pip install pynx-devel-nightly.tar.bz2[cuda,gui,mpi]`                        # Install avec les extras cuda, mpi, cdi
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
* `debuild -b`
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
* p9_3.8_dev : optimised for BCDI, gwaihir and PyNX, development version
* p9_3.8_stable : optimised for BCDI, gwaihir and PyNX, stable version
* p9.pynx-devel : fonctionne pour pynx, frequently updated : `source /sware/exp/pynx/devel.p9/bin/activate`

You are not allowed to modify these environments but you should link a kernel if you wish to use them in jupyter.

To do so:
* `source_p9`
* `ipython kernel install --name "p9_gwaihir --user`

# Clusters at SOLEIL

## GRADES

## SixS3