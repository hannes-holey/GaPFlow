Quick start
===========

Installation
------------

``GaPFlow`` can be installed via

::

    pip install GaPFlow


A serial build of LAMMPS is provided for most platforms which allows testing all of ``GaPFlow``'s functionality.
For production simulations it is however recommended to build GaPFlow with parallel LAMMPS on your system.
You need to have MPI installed (e.g. via ``apt install openmpi-bin libopenmpi-dev`` on Debian-based systems).

To compile for your specific platform, run

::

    pip install --no-binary GaPFlow GaPFlow.[parallel]

You can check your installation by running ``gpf_info`` from the command line.

::

    ==========
    GaPFlow
    ==========
    Version: ...
    
    ==========
    LAMMPS
    ==========
    Version: ...
    Shared lib: <path-to-your-python-env>/lib/python3.X/site-packages/GaPFlow/_vendor/lammps/liblammps_mpi.[so,dylib]
    MPI: True
    mpi4py: True
    packages: ['EXTRA-FIX', 'MANYBODY', 'MOLECULE']
    
    ==========
    muGrid
    ==========
    Version:  ...
    NetCDF4: True
    MPI: False

We currently do not use parallel functionalities of µGrid, so MPI support is not required.

Minimal example
---------------

Simulation inputs are commonly provided in YAML files. A typical input
file might look like this:

.. code:: yaml

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

Note that this example uses fixed-form constitutive laws without GP
surrogate models or MD data. More example input files can be found in
the examples directory.

The input files can be used to start a simulation from the command line

.. code:: bash

   python -m GaPFlow -i my_input_file.yaml

or from a Python script

.. code:: python

   from GaPFlow.problem import Problem

   myProblem = Problem.from_yaml('my_input_file.yaml')
   myProblem.pre_run()
   myProblem.run()

Simulation output is stored under the location specified in the input
file. After successful completion, you should find the following files.
- ``config.yml``: A sanitized version of your simulation input. 
- ``topo.nc``: NetCDF file containing the gap height and gradients. 
- ``sol.nc``: NetCDF file containing the solution and stress fields. 
- ``history.csv``: Contains the time series of scalar quantities (step, Ekin, residual, …) 
- ``gp_[xz,yz,zz].csv`` (Optional): Contains the time series of GP hyperparameters, database size, etc. 
- ``Xtrain.npy`` (Optional): Training data inputs
- ``Ytrain.npy`` (Optional): Training data observations
- ``Ytrain_err.npy`` (Optional): Training data observation error

The code comes with a few useful command line tools for visualizations like this one

.. figure:: assets/journal.gif
   :alt: Journal bearing simulation

which shows the transient solution of a 1D journal bearing with active
learning of the underlying constitutive behavior.