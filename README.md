# Minimal version of HANS based on µGrid

Playground for testing migration to µGrid.

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


## Roadmap
- [X] Active learning in base class, test with pressure only
- [X] Implement GP surrogates for shear stress
- [ ] Problem class with YAML input (HydrodynamicProblem, ElastoHydrodynamicProblem, ThermoElastoHydrodynamicProblem)
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