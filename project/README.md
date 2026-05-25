# CTA200 Final Project

This repository contains my final project submission for CTA200.

## Contents

* `Optimization.py`
* `Working_With_Visibilities.py`
* Jupyter notebooks used for visibility simulation and analysis
* `project.tex`
* `CTA200_project.pdf`
* `project_image/` containing generated figures

## Notes on `hera_sim`

The standard `hera_sim` installation in the environment caused import conflicts with other packages. To run the visibility simulation notebook successfully, I installed `hera_sim` directly from the GitHub repository using:

pip install git+https://github.com/hera-team/hera_sim.git

After installing from the GitHub source, the visibility simulation notebook executed correctly and the generated simulation outputs were saved within the notebooks.

## Included simulation data

The repository also includes the generated simulation data file:

`single_source_example.uvh5`

This allows the visibility analysis notebook to be run directly without regenerating the simulation from scratch.

The file can be loaded in the notebook using:

from pyuvdata import UVData

uvdata = UVData.from_file("single_source_example.uvh5")
