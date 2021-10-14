import setuptools

with open("gwaihir/README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gwaihir",
    version="0.0.1",
    description="Python package for BCDI data analysis",
    author="David Simonne",
    author_email="david.simonne@universite-paris-saclay.fr",
    data_files=[('', ["gwaihir/data_files/pynx_run.txt",
                      "gwaihir/data_files/CompareFacetsEvolution.ipynb", 
                      "gwaihir/data_files/PhasingNotebook.ipynb", 
                      "licence.txt",
                      "gwaihir/sixs/alias_dict_2021.txt"
                     ])],
    url="https://github.com/DSimonne/gwaihir/tree/master",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    scripts=[
        "gwaihir/scripts/compute_q.py",
        "gwaihir/scripts/run_correct_angles_detector.py",
        "gwaihir/scripts/run_movetodir.py",
        "gwaihir/scripts/run_preprocess_bcdi.py",
        "gwaihir/scripts/run_rotate.py",
        "gwaihir/scripts/run_slice_cxi.py",
        "gwaihir/scripts/run_std_filter.py",
        "gwaihir/scripts/run_strain.py",
        "gwaihir/scripts/job.slurm",
        "gwaihir/scripts/run_slurm_job.sh",
    ],
    keywords = "BCDI SXRD",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
	include_package_data=True,
    python_requires='>=3.6',
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "ipywidgets",
        "ipython",
        "scipy",
        "xrayutilities",
        "tables",
        "PyQt5"
        ]
)