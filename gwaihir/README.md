![MVC3](https://user-images.githubusercontent.com/51970962/221527352-fd5981b8-46b2-4add-bf77-065db09b078f.png)
(https://www.freecodecamp.org/news/content/images/size/w1600/2021/04/MVC3.png)


The package is build on the Model - View - Controller architecture described above, the modules are described below:
* `controller`: stores methods that are acessed via the GUI interface, such as running the phase retrieval
* `view`: stores `ipywidgets.widgets` Classes that each correspond to a tab of the GUI
* `dataset.py`: the "Model" is here the `Dataset` Class, that stores all the data and metadata, and allows to save it all in a single `cxi` file for reproducibility.
* `gui.py`: stores the main `ipywidgets.widgets` Class : `Interface`, that links the controller methods and the interactive views.

Additionally, we have:
* `scripts`: folder with different scripts used to
* * submit batch scripts to SLURM
* * automatic facet analysis
* * automatic 3D plotting of `.vti` files
* `plot.py` : stores the `Plotter` Class that allows you to plot any kind of data, used in the GUI but also be used on its own
