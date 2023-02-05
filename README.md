# Welcome 

Contact : david.simonne@universite-paris-saclay.fr

You can install the latest version of the package by cloning the repository and via the `setup.py` script (`pip install .`)

* You can install gwaihir by typing `pip install -U gwaihir` in your command line. A stable version from the master branch uploaded to pypi.org will be used (`https://pypi.org/project/gwaihir/`)
* Otherwise, you can install the latest commit of the package by cloning this repository and typing `pip install .` in the terminal (you must be in the same directory as the `setup.py` script), this will allow you to have the latest updates.

Here is a link to a poster that tries to present Gwaihir:
[Poster_Gwaihir.pdf](https://www.dsimonne.eu/PhDAttachments/Poster_Gwaihir.pdf)

And to the paper: [Simonne, D., Carnis, J., Atlan, C., Chatelier, C., Favre-Nicolin, V., Dupraz, M., Leake, S.  J., Zatterin, E., Resta, A., Coati, A. & Richard, M.  I. (2022). J. Appl. Cryst. 55, 1045-1054](https://scripts.iucr.org/cgi-bin/paper?S1600576722005854)


![Gwahir](https://user-images.githubusercontent.com/51970962/168030371-7212abe3-f8be-4fef-9231-8b1be87abc2e.png)

## Important code snippets

To increase the width of the cells in Jupyter Notebook:

```python
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:75% !important; }</style>"))
```

To avoid automatic cell scrolling:

```javascript
%%javascript
IPython.OutputArea.prototype._should_scroll = function(lines) {
    return false;
}
```

To open the GUI:

```python
from gwaihir.gui import Interface
GUI = Interface()
```


# GUI Preview:
## Pre-processing data
https://user-images.githubusercontent.com/51970962/154160601-f3e7878a-d2c6-4560-95e5-adf7087f59ab.mp4

## Phase retrieval
https://user-images.githubusercontent.com/51970962/154160830-f3c6460b-14e5-4bcc-99f5-e8691278a4e9.mp4

## Data plotting
https://user-images.githubusercontent.com/51970962/154160549-c5caea1b-afa0-4a29-a5a8-aff8a1a5158b.mp4

## Post-processing
https://user-images.githubusercontent.com/51970962/154236802-24643473-1ee9-4d01-823c-beca07ea1c58.mp4

An example file can be downloaded at: https://www.dsimonne.eu/Attachments/align_031968.cxi

# Known bog

The first time you install `gwaihir`, it is possible that when you open the Interface, you only see some text printed, or the content of the readme tab, but you cannot interact with anything. This is solved by restarting your computer.

Bog with printed text:
![index](https://user-images.githubusercontent.com/51970962/197808456-26cbb8b3-266f-419e-88c1-dd020dc6f432.jpeg)

Bog with README tab:
![image](https://user-images.githubusercontent.com/51970962/197807789-dc597beb-fd6f-4d4f-a9a4-0464319ef542.png)

# Clusters at ESRF

Gwaihir **only** works with the p9 partition at the ESRF, optimized for phase retrieval.

If you want to use it for data analysis, you can install `gwaihir` and `bcdi` on rnice.

## Jupyter-slurm

How to access:
* Web browser: https://jupyter-slurm.esrf.fr/hub/spawn
* Terminal (for advanced users) :
    * `ssh -X <login>@slurm-nice-devel`
    * Ask for a GPU: `srun -N 1 --partition=p9gpu --gres=gpu:1 --time=06:00:00 --pty bash`

#### Available environements
* `/usr/bin/python3`: your personal environemnt
* p9.dev : optimised for BCDI, gwaihir and PyNX, development version, `source /data/id01/inhouse/david/p9.dev/bin/activate`
* p9.stable : optimised for BCDI, gwaihir and PyNX, stable version, `source /data/id01/inhouse/david/p9.stable/bin/activate`
* p9.pynx-devel : pynx only, frequently updated : `source /sware/exp/pynx/devel.p9/bin/activate`

You are not allowed to **modify** these environments but you can **link** a kernel if you wish to **use** them in jupyter.

To do so:
* Source the environment; e.g. `source /data/id01/inhouse/david/p9.dev/bin/activate`
* Make sure that:
    * you are on slurm
    * you requested a GPU
* Create the kernel:
    * `python3 -m ipykernel install --user --name p9.stable`
* [Documentation](https://ipython.readthedocs.io/en/stable/install/kernel_install.html)

Once you feel confident, you should create your own environment, to avoid sudden updates that may impact your work!

To list the kernels you have installed: `jupyter kernelspec list`

And to remove them: `jupyter kernelspec uninstall <kernelname>`

Make sure that you are using the right kernel on your Jupyter Notebook !

### Set up `ssh` connection without using password (mandatory for batch jobs)
* Login into slurm (make sure that you asked for a GPU)
* Open a terminal (new -> terminal)

Enter the following commands (replace `<username>` with your username, for me it is simonne)
* `cd`
* `ssh-keygen -t rsa` (press enter when prompted, ~ 3 times)
* `ssh <username>@slurm-nice-devel mkdir -p .ssh`
* `cat .ssh/id_rsa.pub | ssh <username>@slurm-nice-devel 'cat >> .ssh/authorized_keys'`

You should not need a password anymore when login into slurm, make sure it is the case by typing
* `ssh <username>@slurm-nice-devel`

# CLuster at SOLEIL

## GRADES

To analyse data recorded at SOLEIL from your personal computer, you can use Jupyter Notebook via GRADES. The documentation is here (accessible on- site via the SOLEIL wifi or with the SOLEIL VPN) : http://confluence.synchrotron-soleil.fr/display/EG/Service%3A+Jupyter+Notebook

* Use this link to open Jupyter Notebook : http://grades-01.synchrotron-soleil.fr/notebook/
* Launch a Terminal (command line) from the upper-right 'New' menu. The system is a Debian 10, so you can enter linux commands there.

`PyNX` is already installed on GRADES, a version that is out of my control. So you 'just' have to download the `bcdi` and `gwaihir` packages by typing `pip3 install --proxy=http://195.221.0.35:8080 -U gwaihir bcdi`

If you encounter an error with gwaihir or bcdi, it is possible that the pip packages are not up to date. Then you should follow the procedure described [below](https://github.com/DSimonne/gwaihir#installing-different-packages-yourself) and manually download the packages. Just replace `pip install` by `pip3 install --proxy=http://195.221.0.35:8080`,  the proxy IP can be 195.221.0.34:8080 or 195.221.0.35:8080 on the ReS (offices and VPN), and 195.221.10.6:8080 or 195.221.10.7:8080 on the REL (beam-lines, RAS).

You can also directly use a virtual machine provided by GRADES
* Go to https://re-grades-01.exp.synchrotron-soleil.fr/qemu-web-desktop/
* Use the SUNset/LDAP id.
* Click on `Create` then on `Connect` (after 10 sec).
* Once the desktop is available, search "gwaihir" in the bottom left menu and execute the program.
* You are now connected to the ruche with an environment that has all the necessary packages, you just need to open a notebook.

## SixS

A GPU is installed on sixs3, a computer available on the beamline, for phase retrieval.

Please respect the following steps:
* Make sure that you are logged in as `com-sixs`
* Activate the environment `source_py3.9` or `source /home/experiences/sixs/simonne/Documents/py39-env/bin/activate`, this environment is protected and you cannot modify it.
* Launch `jupyter notebook`
* Go to the test_data folder and then choose the beamline you want to test
* Follow the instructions in the notebook

## Cristal

A GPU is installed on cristal4, a computer available on the beamline, for phase retrieval.

Please respect the following steps:
* Make sure that you are logged in as `com-cristal`
* Activate the environment `source_gwaihir` or `source /home/experiences/crystal/com-cristal/PackagesGwaihir/py-gwaihir/bin/activate`, this environment is protected and you cannot modify it.
* Launch `jupyter notebook`
* Go to the test_data folder and then choose the beamline you want to test
* Follow the instructions in the notebook

# Installing different packages yourself

* First, I advise you to create a `/Packages` directory to keep these.
* Secondly, I advise you to create a virtual environment to help with debogging, and so that once everything works, you don't update a package by mistake. To do so please follow the following steps:

## Create a virtual environment

* `mkdir py38-env`
* `cd py38-env/`
* `python3.8 -m venv .`
* `source bin/activate` # To activate the environment
* Make sure `wheel` and `setuptools` are installed: `pip install wheel setuptools pip --upgrade`

Then you should create an alias such as: `alias source_p9="source /home/user/py38-env/bin/activate"`

## Specific instructions for the p.9 cluster
* If `vtk` does not install (on the p9 cluster at the ESRF for example), you can type : `pip install --trusted-host www.silx.org --find-links http://www.silx.org/pub/wheelhouse vtk`, you may also need to remove the version requirements in `bcdi/setup.py`
* If `PyQt5` does not install (also on the p9 cluster at the ESRF), you can install it by activating your environment from the rnice cluster.

## 1) Install PyNX
* Use the latest version
* `cd /Packages`
* `mkdir PyNX_install`
* `cd PyNX_install/`
* `curl -O http://ftp.esrf.fr/pub/scisoft/PyNX/pynx-devel-nightly.tar.bz2` # Installation details within install-pynx-venv.sh
* `source_p9`
* `pip install pynx-devel-nightly.tar.bz2[cuda,gui,mpi]` # Install with extras cuda, mpi, cdi
* cite `PyNX: high-performance computing toolkit for coherent X-ray imaging based on operators is out: J. Appl. Cryst. 53 (2020), 1404`, also available as `arXiv:2008.11511`

## 2) Install gwaihir
* `cd /Packages`
* `git clone https://github.com/DSimonne/gwaihir.git`
* `cd gwaihir`
* `source_p9`
* `pip install .`
* cite `Simonne, D., Carnis, J., Atlan, C., Chatelier, C., Favre-Nicolin, V., Dupraz, M., Leake, S.  J., Zatterin, E., Resta, A., Coati, A. & Richard, M.  I. (2022). J. Appl. Cryst. 55, 1045-1054.`

## 3) Install bcdi
* Latest version tested : v0.2.8
* `cd /Packages`
* `git clone https://github.com/carnisj/bcdi.git`
* `cd bcdi`
* `source_p9`
* `pip install .`
* cite `DOI: 10.5281/zenodo.3257616`

## 4) Install facet-analyser (Debian 11 only)
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


# To go further ...

## Using `Gwaihir` only as a plotting tool in Jupyter Notebook
![image](https://user-images.githubusercontent.com/51970962/157677934-d6983756-48d3-4a1d-8394-a86f0d2b721e.png)


## Quick navigation between `vtk` files in the GUI (outdated but it can give you some ideas)

It is possible to automate the navigation in the GUI !

Here I have a pandas DataFrame that contains data about my scans, I use to automate the navigation:

```python
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob

df = pd.read_csv("reconstructions/scans_data.csv")

GUI.tab_facet.children[3].value = False
GUI.window.selected_index = 0
time.sleep(1)

GUI._list_widgets_init_dir.children[7].value = False
time.sleep(1)

scan = 3600
row = df[df.scan == scan]

particle = row.particle.values[0]
temp = row.temp_given.values[0]
condition = row.condition.values[0]

GUI._list_widgets_init_dir.children[2].value = scan
GUI._list_widgets_init_dir.children[
    3].value = f"/data/id01/inhouse/david/SIXS_June_2021/reconstructions/{temp}/{condition}/{particle}/S{scan}/data/"
GUI._list_widgets_init_dir.children[
    4].value = f"/data/id01/inhouse/david/SIXS_June_2021/reconstructions/{temp}/{condition}/{particle}/"
time.sleep(1)
GUI._list_widgets_init_dir.children[7].value = True

time.sleep(1)

GUI.window.selected_index = 9

GUI.tab_facet.children[3].value = "load_csv"
```