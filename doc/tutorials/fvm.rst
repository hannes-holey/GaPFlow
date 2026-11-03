Explicit Solver (FVM)
=====================

The first tutorial introduces the mathematical foundation for the multiscale approach, splitting the problem into a micro- and
macro-problem. Subsequent tutorials build upon this foundation, covering the constitutive equations for stress, equations of state,
and viscosity. The pieces are brought together in tutorials 4, 5 and 6, which focus on confined fluids and lubrication problems.
Tutorial 7 introduces Gaussian Processes for surrogate modeling of the micro-problem, using actual MD simulation data in tutorial 8.
Tutorial 9 illustrates the coupling between fluid pressure and the elastic deformation of the solid surfaces.
The final tutorial shows how custom input features, such as a spatially varying wall slip, can be added to the GP models.

.. toctree::
    :maxdepth: 1

    fvm_01_macro_equations.ipynb
    fvm_02_stress_sympy.ipynb
    fvm_03_constitutive_laws.ipynb
    fvm_04_confined_fluids.ipynb
    fvm_05_lubrication_1d.ipynb
    fvm_06_lubrication_2d.ipynb
    fvm_07_gp_mock.ipynb
    fvm_08_gp_md.ipynb
    fvm_09_elastic_deformation.ipynb
    fvm_10_gp_custom_features.ipynb
