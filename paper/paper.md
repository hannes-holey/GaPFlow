---
title: 'GaPFlow: Gap-averaged flow simulations with Gaussian Process regression'
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

`GaPFlow` is a numerical solver for fluid flows in confined geometries, such as the narrow gaps found in lubricated contacts.
Most lubrication problems solve the Reynolds equation, a simplified form of the Navier-Stokes equation formulated as a single partial differential equation for the fluid pressure.
`GaPFlow` solves the lubrication problem in the form proposed by [@holey2022_heightaveraged], which propagates gap-averaged conserved quantities, such as mass or momentum, in time.
This formulation is agnostic to the constitutive behavior of the confined fluid, which makes it suitable for multiscale simulations, where the constitutive behavior is determined from molecular dynamics (MD) simulations.
`GaPFlow` uses a surrogate model based on Gaussian process (GP) regression to interpolate between data obtained from MD, and to select new configurations to augment an exsiting MD database (active learning).

# Statement of need

# Acknowledgments

# References