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

# Features

## Solver for gap-averaged balance laws

## GP regression and active learning

## Automatic setup of MD runs

## Elastic deformations

# Acknowledgments

The authors gratefully acknowledge support by the German Research Foundation (DFG) through GRK 2450.
H.H. thanks the Alexander von Humboldt Foundation for support through the Feodor Lynen fellowship.

# References