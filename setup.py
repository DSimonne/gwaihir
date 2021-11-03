import setuptools

with open("gwaihir/README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gwaihir",
    version="0.0.1",
    description="Python package for BCDI data analysis",
    author="David Simonne",
    author_email="david.simonne@universite-paris-saclay.fr",
    download_url="https://pypi.org/project/gwaihir/",
    data_files=[('', ["gwaihir/data_files/pynx_run.txt",
                      "gwaihir/data_files/CompareFacetsEvolution.ipynb",
                      "gwaihir/data_files/PhasingNotebook.ipynb",
                      "licence.txt",
                      "gwaihir/scripts/examples/config_postprocessing_sixs.yml",
                      "gwaihir/scripts/examples/config_preprocessing_sixs.yml",
                      "gwaihir/scripts/examples/workflow_sixs.sh",
                      ])],
    url="https://github.com/DSimonne/gwaihir/tree/master",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    scripts=[
        "gwaihir/scripts/job_esrf.slurm",
        "gwaihir/scripts/run_correct_angles_detector.py",
        "gwaihir/scripts/run_movetodir.py",
        "gwaihir/scripts/run_rotate.py",
        "gwaihir/scripts/run_slurm_job.sh",
        "gwaihir/scripts/run_std_filter.py",
    ],
    keywords=["BCDI", "ipywidgets", "PyNX"],
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
        "ipyvolume",
        "scipy",
        "xrayutilities",
        "tables",
        "PyQt5",
        "h5py",
        #         "bcdi",
        #         "PyNX",
    ]
)
