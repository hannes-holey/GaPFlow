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

## Minimal example
Simulation inputs are commonly stored in YAML files.

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