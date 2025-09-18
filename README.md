# GaPFlow
*Gap-averaged flow simulations with Gaussian Process regression.*

This code implements the solution of time-dependent lubrication problems as described in:
- [Holey, H. et al., Tribology Letters 70 (2022)](https://doi.org/10.1007/s11249-022-01576-5)

The extension to atomistic-continuum multiscale simulations with Gaussian process (GP) surrogate models has been described in:
- [Holey, H. et al., Science Advances 11, eadx4546 (2025)](https://doi.org/10.1126/sciadv.adx4546)

The code uses [µGrid](https://muspectre.github.io/muGrid/) for handling macroscale fields and [tinygp](https://tinygp.readthedocs.io/en/stable/index.html) as GP library. Molecular dynamics (MD) simulations run with [LAMMPS](https://docs.lammps.org) through its [Python interface](https://docs.lammps.org/Python_head.html).

## Roadmap
- [X] Active learning in base class, test with pressure only
- [X] Implement GP surrogates for shear stress
- [X] Problem class with YAML input
- [X] Output fields, history, gap
- [X] Start plotting scripts
- [X] Output GP hyperparams, training data
- [X] Adaptive time stepping
- [X] Adaptive time stepping GP
- [X] Equation of state w/ plots
- [X] Plotting scripts
- [ ] Catch sigterm
- [ ] Viscosity models with plots
- [ ] dtool Datasets
- [ ] MD setup
- [ ] Unit tests and integration tests
- [ ] Docstrings & typing
- [ ] MPI parallel
- [ ] Documentation with examples
- [ ] JOSS submission (w or w/o elasticity)

## Installation

Install [µGrid](https://muspectre.github.io/muGrid/GettingStarted.html)'s Python bindings
```
pip install -v --force-reinstall --no-cache --no-binary muGrid muGrid
```
and make sure MPI and PnetCDF get detected. For manual installations of PnetCDF (recommended), you may need to tell `pkg_config` where to find it, e.g.
```
export PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig:$HOME/.local/lib64/pkgconfig:$PKG_CONFIG_PATH
```
for instalations under `$HOME/.local/`.

After that run
```
pip install -e .[test]
```
for an editable installation with optional dependecies for testing (using `pytest`).

## Minimal example
Simulation inputs are commonly provided in YAML files. A typical input file might look like this:

```yaml
# examples/journal.yaml
options:
    output: data/journal
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
```

Note that this example uses fixed-form constitutive laws without GP surrogate models or MD data. More example input files can be found in the [examples](examples/) directory.

The input files can be used to start a simulation from the command line
```bash
python -m GaPFlow -i my_input_file.yaml
```
or from a Python script
```python
from GaPFlow.problem import Problem

myProblem = Problem.from_yaml('my_input_file.yaml')
myProblem.run()
```
Simulation output is stored under the location specified in the input file. After successful completion, you should find the following files.
- `config.yml`: A sanitized version of your simulation input.
- `gap.nc`: NetCDF file containing the gap height and gradients.
- `sol.nc`: NetCDF file containing the solution and stress fields.
- `history.csv`: Contains the time series of scalar quantities (step, Ekin, residual, ...)
- `gp_[shear,press].csv` (Optional): Contains the time series of GP hyperparameters, database size, etc.
- `Xtrain.npy` (Optional): Training data inputs
- `Ytrain.npy` (Optional): Training data observations

The code comes with a few handy command line tools for visualization...

## Documentation
...

## Funding
This work received funding from the German Research Foundation (DFG) through GRK 2450 and from the Alexander von Humboldt Foundation through a Feodor Lynen Fellowship.