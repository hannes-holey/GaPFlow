GaPFlow
=======

*Gap-averaged flow simulations with Gaussian Process regression.*

This code implements the solution of time-dependent lubrication problems
as described in:

- `Holey, H. et al., Tribology Letters 70 (2022) <https://doi.org/10.1007/s11249-022-01576-5>`__

The extension to atomistic-continuum multiscale simulations with
Gaussian process (GP) surrogate models has been described in: 

- `Holey, H. et al., Science Advances 11, eadx4546 (2025) <https://doi.org/10.1126/sciadv.adx4546>`__

The code uses `µGrid <https://muspectre.github.io/muGrid/>`__ for
handling macroscale fields and
`tinygp <https://tinygp.readthedocs.io/en/stable/index.html>`__ as GP
library. Molecular dynamics (MD) simulations run with
`LAMMPS <https://docs.lammps.org>`__ through its `Python
interface <https://docs.lammps.org/Python_head.html>`__. Elastic
deformation is computed using
`ContactMechanics <https://contactengineering.github.io/ContactMechanics/>`__.

Funding
-------

This work received funding from the German Research Foundation (DFG)
through GRK 2450 and from the Alexander von Humboldt Foundation through
a Feodor Lynen Fellowship.
