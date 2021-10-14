# BCDI and ipywidgets

## How to use
* Connect to slurm, make sure that you do not need passwords when connecting for the batch scripts (see `http://www.linuxproblem.org/art_9.html`)
* Add this environemnet to your jupyter kernels: `source /data/id01/inhouse/david/py38-env/bin/activate`
* Open a jupyter notebook with this kernel and then run:

`from gwaihir.gui.gui import Interface`
`Test = Interface()`

## Guidelines for widgets
* Always press enter after editing a widget
* If somehow widgets are disabled when they should not be, please rerun the Interface cell and raise an issue with the details of each step that was undertaken, so that I can correct the bog

The widgets number might change with the evolution code, you can try to find the good one by playing with the different *_list_widgets* (use tqb to complete)

Some screenshots of the work so far:
![Tab1](https://user-images.githubusercontent.com/51970962/130641516-ffe670b1-7b72-4b86-bef4-3b8bf4b7a797.png)
![Tab2](https://user-images.githubusercontent.com/51970962/130641522-9801d342-a1cc-4e87-8cb6-76cd78c909d3.png)
![Tab3](https://user-images.githubusercontent.com/51970962/130641578-f2515a53-09ba-47ac-a08e-cf093647d517.png)
![Tab4](https://user-images.githubusercontent.com/51970962/130641621-f6fafbaf-ac05-49e2-b9b5-e3ee2373b9e0.png)
![Tab5](https://user-images.githubusercontent.com/51970962/130641630-80fca919-ebb6-4ece-8638-95bbfd8a3dd3.png)
![Tab6](https://user-images.githubusercontent.com/51970962/130641638-9d59df04-2e60-495a-9de4-fcc0c3dfb9fe.png)
![Tab7](https://user-images.githubusercontent.com/51970962/130641648-48aaf34e-e70f-42f7-8a14-e283c519759e.png)
![Tab8](https://user-images.githubusercontent.com/51970962/130641650-62abc8d6-c45e-46ab-902e-d8a1211774ba.png)
![Output1](https://user-images.githubusercontent.com/51970962/130641658-20c82525-6a87-4414-baba-30defcba4328.png)
![Output2](https://user-images.githubusercontent.com/51970962/130641661-31ab2181-c1d4-4b24-89ed-8e4e2f15c5ca.png)