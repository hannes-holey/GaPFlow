---
title: 'GaPFlow: Gap-averaged flow simulations with Gaussian process regression'
tags:
  - Multiscale simulations
  - Gaussian process regression
  - Lubrication
authors:
  - name: Christoph Huber
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Hannes Holey
    orcid: 0000-0002-4547-8791
    affiliation: 2
affiliations:
  - name: Institute for Applied Materials, Karlsruhe Institute for Technology, Strasse am Forum 7, 76131 Karlsruhe, Germany
    index: 1
  - name: Center for Complexity and Biosystems, Department of Physics, University of Milan, Via Celoria 16, 20133 Milan, Italy
    index: 2
date: 17 October 2025
bibliography: paper.bib
---

# Summary

Fluid flow in confined geometries is common in both natural systems and many engineering applications.
When the characteristic length of the confining dimension approaches the nanometer scale, the molecular nature of the fluid can no longer be neglected.
This is particularly relevant for lubricated frictional contacts, where surface roughness can lead to local gap heights of only a few nanometers [@archard1962_lubrication;@glovnea2003_measurement].
The constitutive laws that describe the fluid's response to extreme loading conditions (e.g. high shear rates) need to account for molecular effects, such as fluid layering [@gao1997_layering] and wall slip [@pit2000_direct;@zhu2001_ratedependent].

Molecular dynamics (MD) simulations have become a standard tool for describing lubricant flow in nanometer-scale constrictions [@ewen2018_advances], and have been used to parameterize common constitutive laws for viscosity and wall slip [@martini2006_molecular;@savio2015_multiscale;@codrignani2023_continuum].
While such models can be readily incorporated into existing lubrication solvers, they lack the feedback mechanism from the macroscopic to the molecular scale.
The rigidity of purely sequential coupling schemes suggests that they are not ideal for capturing the extreme and diverse environments typical for frictional contacts. 
`GaPFlow` provides a simulation framework that enables concurrent multiscale simulations of nanofluidic flows relying on fixed-form, parametric constitutive fluid models.
By employing nonparametric surrogate models from probabilistic machine learning, `GaPFlow` can adapt to previously unseen flow regimes through active learning and provides an uncertainty measure for the prediction of shear and normal stresses in lubricated frictional contacts.

# Statement of need

`GaPFlow` is a numerical solver for fluid flows in confined geometries, such as the narrow gaps found in lubricated contacts.
Traditional lubrication models solve the Reynolds equation, a simplified form of the Navier-Stokes equation expressed as a single partial differential equation for the fluid pressure.
In contrast, `GaPFlow` solves the lubrication problem in the formulation proposed by @holey2022_heightaveraged, which evolves gap-averaged conserved quantities, such as mass or momentum, in time.
This formulation is agnostic to the constitutive behavior of the confined fluid, making it suitable for multiscale simulations in which the fluid response is provided by molecular dynamics (MD) simulations.
`GaPFlow` uses a surrogate model based on Gaussian process (GP) regression to interpolate between data obtained from MD, and to select new configurations based on the GP uncertainty to augment an existing MD database (a.k.a. active learning) [@holey2025_active].

The following papers have used `GaPFlow` so far:

- @holey2022_heightaveraged
- @holey2024_sound
- @holey2025_active

# Components and external dependencies

`GaPFlow` serves as a *glue code* that integrates the various components required for multiscale simulations of confined fluid flows. It relies on a small set of external dependencies which are summarized below.

## Solver for gap-averaged balance laws

Averaging the general form of a conservation law over the gap coordinate ($z$) with spatially and temporally varying integral bounds, i.e. the topographies of the lower ($h_0$) and upper wall ($h_1$), leads to a balance law of the form

$$
\frac{\partial \bar{\mathbf{q}}}{\partial t} = - \frac{\partial \bar{\mathbf{f}}_x}{\partial x} - \frac{\partial \bar{\mathbf{f}}_y}{\partial y} - \mathbf{s},
$$

where $\bar{\mathbf{q}}\equiv\bar{\mathbf{q}}(x,y,t)=h^{-1}\int_{h_0}^{h_1}\mathbf{q}(x,y,z,t)dz$ collects the densities of conserved variables (e.g. $\mathbf{q}=(\rho, j_x, j_y)^\top$ for mass and in-plane momentum) and $\bar{\mathbf{f}}_i\equiv\bar{\mathbf{f}}_i(x,y,t)=h^{-1}\int_{h_0}^{h_1}\mathbf{f}_i(x,y,z,t)dz$ are the corresponding fluxes in direction $i\in\{x,y\}$ with $h=h_1 - h_0$. 
The source term $\mathbf{s}$ accounts for fluxes across the bottom and top walls ($\mathbf{f}_z$). 
The current implementation uses a finite volume discretization on a regular grid and the MacCormack explicit time-integration scheme [@maccormack2003_effect] to solve the transient lubrication problem. 
The [µGrid](https://muspectre.github.io/muGrid/) library is used to assemble the discretized density and flux fields into a unified container and to export the simulation results in the [NetCDF](https://www.unidata.ucar.edu/software/netcdf) file format.

## GP regression

The fluxes required to close the macroscopic equations can either be obtained from deterministic constitutive laws or modeled using Gaussian process (GP) regression.
The GP models are trained on data generated by MD, or, for testing purposes, on sparsified datasets sampled from predefined constitutive laws. 
`GaPFlow` employs the [tinygp](https://tinygp.readthedocs.io/en/stable/index.html) library for constructing and training GP models, taking advantage of its flexibility
For example, it allows the implementation of custom kernels for the joint prediction of wall shear stresses at the top and bottom walls using a multi-output GP that shares a common noise process.
Since [tinygp](https://tinygp.readthedocs.io/en/stable/index.html) is built on [JAX](https://github.com/jax-ml/jax), `GaPFlow` also benefits from automatic differentiation of the GP models, e.g. to compute the speed of sound from the pressure model.

## Automatic setup of MD runs

In active learning simulations the GP uncertainty determines when and where new MD data are required.
When this occurs, the main simulation loop pauses and waits for the MD simulation to complete.
`GaPFlow` uses the Python interface of [LAMMPS](https://docs.lammps.org) [@thompson2022_lammps] to execute these simulations in parallel.
The correct and fully automated setup of MD runs is likely the most critical step in the multiscale framework.
To facilitate this process, users can subclass the abstract base class `GaPFlow.md.MolecularDynamics` implementing only two methods: one for generating the input files and one for reading the output files.
This design gives users complete control over the MD setup while maintaining a consistent interface with the main solver.
`GaPFlow` provides two examples how this can be done: 1. A simple example that relies entirely on LAMMPS to set up a Lennard-Jones system (fluid and walls), and 2. A more advanced example that uses [ASE](https://ase-lib.org/) [@larsen2017_atomic] and [moltemplate](https://www.moltemplate.org/) [@jewett2021_moltemplate] to construct an alkane fluid confined between gold walls.
Both MD setups use the Gaussian dynamics algorithm by @strong2017_dynamics as implemented in LAMMPS, to control the mass flux according to the continuum solution.

## Data management

## Elastic deformations

# Acknowledgments

The authors gratefully acknowledge support by the German Research Foundation (DFG) through GRK 2450.
H.H. thanks the Alexander von Humboldt Foundation for support through the Feodor Lynen fellowship.

# References