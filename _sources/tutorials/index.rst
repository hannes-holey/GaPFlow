Tutorials
=========

This section contains a series of tutorials that guide users through features and applications of the GaPFlow framework. 
Each tutorial is designed to provide hands-on experience with different aspects, going from basic concepts to 
simulations that integrate all previously covered parts.

The first tutorial introduces the mathematical foundation for the multiscale approach, splitting the problem into a micro- and 
macro-problem. Subsequent tutorials build upon this foundation, covering the constitutive equations for stress, equations of state,
and viscosity. The pieces are brought together in tutorials 4, 5 and 6, which focus on confined fluids and lubrication problems.
Tutorial 7 introduces Gaussian Processes for surrogate modeling of the micro-problem, using actual MD simulation data in tutorial 8.
The final tutorial illustrates the coupling between fluid pressure and the elastic deformation of the solid surfaces.

.. toctree::
    :maxdepth: 1

    01_macro_equations.ipynb
    02_stress_sympy.ipynb
    03_constitutive_laws.ipynb
    04_confined_fluids.ipynb
    05_lubrication_1d.ipynb
    06_lubrication_2d.ipynb
    07_gp_mock.ipynb
    08_gp_md.ipynb
    09_elastic_deformation.ipynb