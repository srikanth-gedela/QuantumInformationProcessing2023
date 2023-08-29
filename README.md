# SpringSchool2023

Tutorial materials for the QIP Spring School 2023

* [T1 - `garden_DWAVE`](garden_DWAVE): Solution of the Garden Optimization problem on the D-Wave quantum annealer.

* [T2 - `tsp_DWAVE`](tsp_DWAVE): Solution of the Traveling Salesman Problem through large cities in Europe using the D-Wave quantum annealer.

* [T3 - `garden_QAOA`](garden_QAOA): QAOA (Qiskit) solution of the Garden Optimization problem.

* [T4 - `knapsack_QAOA`](knapsack_QAOA): Combinatorial binary problem formulation of the Knapsack problem and its solution using the QAOA (Qiskit) algorithm.

## Installation on JUNIQ

No installation is required on `JUNIQ` - all the packages are readily available
in the default Python 3 kernel.

One only needs to clone the repository using the JupyterLab `Git/Clone a Repository` tool.

## Installation without JUNIQ

### Clone the repository

```bash
git clone https://jugit.fz-juelich.de/qip/springschool2023.git
```

### D-Wave tutorials

For the D-Wave tutorials, one needs to install the D-Wave Ocean SDK as described in the [documentation of the package `dwave-ocean-sdk`](https://docs.ocean.dwavesys.com/en/stable/overview/install.html). For the TSP, it is additionally helpful to install [`cartopy`](https://scitools.org.uk/cartopy/docs/latest/) for the visualizations to work.

### QAOA tutorials

#### Setup a new local python environment and switch to it

With `conda`:

```bash
conda create -n ss23-qaoa python=3
conda activate ss23-qaoa
```

Or with `venv`:

```bash
python3 -m venv ss23-qaoa
source ss23-qaoa/bin/activate
```

#### Install packages

```bash
pip install qiskit 'qiskit-optimization[cplex]' matplotlib pandas networkx openpyxl jupyterlab ipywidgets
```

#### Add the new tutorial environment to the list of the Jupyter kernels

```bash
python -m ipykernel install --name=ss23-qaoa --user
```

### Run JupyterLab

```bash
jupyter lab
```
