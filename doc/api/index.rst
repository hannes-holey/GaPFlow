API Reference
=============

Main drivers
------------

.. currentmodule:: GaPFlow

.. autosummary::
   :toctree: summary

   Problem
   Database

Solvers
-------

Implicit solver (FEM)
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: GaPFlow.solver_implicit

.. autosummary::
   :toctree: summary

   solver_implicit.ImplicitSolver

Explicit solver (FVM)
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: GaPFlow.solver_explicit

.. autosummary::
   :toctree: summary

   solver_explicit.ExplicitSolver
   integrate

Models
------

.. currentmodule:: GaPFlow.models

.. autosummary::
   :toctree: summary

   gp.GaussianProcessSurrogate
   Pressure
   BulkStress
   WallStress
   Viscosity
   pressure
   sound
   viscosity
   viscous
   profiles

Molecular dynamics
------------------

.. currentmodule:: GaPFlow.md

.. autosummary::
   :toctree: summary

   MolecularDynamics
   Mock
   LennardJones
   GoldAlkane
   moltemplate

Topography
----------

.. currentmodule:: GaPFlow.topography

.. autosummary::
   :toctree: summary

   Topography
   ElasticDeformation
   ForceBalance

Parallelization
---------------

.. currentmodule:: GaPFlow.parallel

.. autosummary::
   :toctree: summary

   DomainDecomposition

Visualization
-------------

.. currentmodule:: GaPFlow.viz

.. autosummary::
   :toctree: summary

   animations
   plotting

Analysis
--------

.. currentmodule:: GaPFlow.analysis

.. autosummary::
   :toctree: summary

   compute_metrics
   print_metrics
   create_overview_plot
