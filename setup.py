import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gwaihir",
    version="0.0.7",
    description="Python package for BCDI data analysis",
    author="David Simonne",
    author_email="david.simonne@universite-paris-saclay.fr",
    download_url="https://pypi.org/project/gwaihir/",
    data_files=[('', ["licence.txt",
                      ])],
    url="https://github.com/DSimonne/gwaihir/tree/master",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    scripts=[
        "gwaihir/scripts/handle_sixs_data.py",
        "gwaihir/scripts/job_esrf.slurm",
        "gwaihir/scripts/run_slurm_job.sh",
        "gwaihir/scripts/plot_paraview_slices.py",
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
        "datetime",
        "argparse",
        "matplotlib",
        "ipywidgets",
        "ipython",
        "ipyvolume",
        "scipy",
        "xrayutilities",
        "tables",
        "h5glance",
        "h5py",
        "jupyter",
        "bokeh",
        "panel",
        "scikit-image",
        "ipython_genutils",
    ]
)
