# Minimal version of HANS based on µGrid

Playground for testing migration to µGrid.

## Roadmap
- [X] Active learning in base class, test with pressure only
- [X] Implement GP surrogates for shear stress
- [X] Problem class with YAML input
- [ ] File I/O (fields, scalars, gp_params, train_data)
- [X] Plotting scripts (file pipeline)
- [ ] Adaptive time stepping
- [ ] dtool Datasets
- [ ] Constitutive laws from HANS
- [ ] Docstrings
- [ ] Unit tests and integration tests
- [ ] Documentation with examples
- [ ] MD setup
- [ ] MPI parallel
- [ ] JOSS submission (w or w/o elasticity)

## Installation

Install [muGrid](https://muspectre.github.io/muGrid/GettingStarted.html)'s Python bindings
```
pip install -v --force-reinstall --no-cache --no-binary muGrid muGrid
```
and make sure MPI and PnetCDF get detected.

After that run
```
pip install -e .
```
for an editable installation.

## Minimal example
Simulation inputs are commonly stored in YAML files. A typical input file might look like this:

```yaml
# my_input_file.yaml
options:
    output: journal_bearing_run-01
    write_freq: 10
grid:
    dx: 1.e-5
    dy: 1.
    Nx: 100
    Ny: 1
    xE: ['D', 'N', 'N']
    xW: ['D', 'N', 'N']
    yS: ['P', 'P', 'P']
    yN: ['P', 'P', 'P']
    xE_D: 877.7007
    xW_D: 877.7007
geometry:
    type: journal
    CR: 1.e-2
    eps: 0.7
    U: 0.1
    V: 0.
numerics:
    tol: 1e-9
    dt: 1e-10
    max_it: 200
properties:
    shear: 0.0794
    bulk: 0.
    EOS: DH
    P0: 101325
    rho0: 877.7007
    T0: 323.15
    C1: 3.5e10
    C2: 1.23
gp: ...
```

The input files canbe used to start a simulation from the command line
```bash
python -m hans_mugrid -i input.yaml
```
or from a Python script
```python
from hans_mugrid.problem import Problem

myProblem = Problem.from_yaml('my_input_file.yaml')
myProblem.run()
```

Simulation output is stored under the location specified in the input file. After successful completion, you should find the following files.
- `config.yml`: A sanitized version of your simulation input.
- `field.nc`: NetCDF file containing the solution and stress fields.
- `history.csv`: Contains the time series of scalar quantities (step, Ekin, residual, ...)
- `gp_[shear,press].csv` (Optional): Contains the time series of GP hyperparameters, database size, etc.
- `Xtrain.npy` (Optional): Training data inputs
- `Ytrain.npy` (Optional): Training data observations

The code comes with a few handy command line tools for visualization...

## Tests
...

## Documentation
...

## Funding
...