Visualization
=============

GaPFlow provides several command line tools for visualizing simulation output.
These tools operate on the output files produced by a simulation run:

- ``sol.nc``: Solution fields (density, momentum, pressure, shear stress)
- ``topo.nc``: Gap height and gradients
- ``history.csv``: Time series of domain integrals (mass, kinetic energy, load, friction) and solver diagnostics (residual, time step, etc.)
- ``gp_zz.csv``, ``gp_xz.csv`` (optional): GP hyperparameter history

.. figure:: assets/journal.gif
   :alt: Journal bearing simulation

   Transient solution of a 1D journal bearing with active learning of
   the underlying constitutive behavior, created with ``gpf_animate1d``.

File discovery
--------------

All visualization commands automatically search for the required output files
by recursively walking the directory tree starting from the current working
directory. Found simulations are listed with an index and their last-modified
date, and you are prompted to select one or more interactively.

For example, running ``gpf_plot_frame`` from a directory that contains several
simulation output folders might show::

    0: data/journal_dh                          20/02/2026 14:30
    1: data/journal_dh_gp                       20/02/2026 15:12
    Enter keys (space separated or range [start]-[end] or combination of both):

Plotting
--------

gpf_plot_height
^^^^^^^^^^^^^^^

Plot the gap height profile from simulation results.

.. code:: bash

   gpf_plot_height [options]

.. list-table::
   :header-rows: 1
   :widths: 20 10 10 60

   * - Flag
     - Type
     - Default
     - Description
   * - ``-d``, ``--dim``
     - int
     - 1
     - Dimension. ``1``: plots centerline profile. ``2``: plots 2D heatmaps of height and gradients.
   * - ``--show-defo``
     - flag
     - off
     - Show displacement in a separate subfigure with the initial gap height as reference.
   * - ``--show-pressure``
     - flag
     - off
     - Show the pressure profile in a separate subfigure.

gpf_plot_history
^^^^^^^^^^^^^^^^

Plot the time evolution of scalar quantities from ``history.csv`` and of the GP model parameters.
The figure contains one column per kind of data; panels and columns without data are skipped.

- Physical: mass, kinetic energy, load, friction force at the bottom and top wall, maximum sound
  velocity, recorded by both solvers.
- Numerical: solver diagnostics (residuals, time step, Newton iterations), which differ between solvers.
- GP models (if ``gp_zz.csv``, ``gp_xz.csv``, or ``gp_yz.csv`` are present): database size, ratio of
  maximum variance to tolerance, normalised observation noise, and objective, with line styles
  distinguishing the models. Active learning is triggered when the variance ratio exceeds one.

By default, all columns use the simulation time as x-axis, such that runs of different
solvers can be compared. When plotting a single run, the step number is shown on a secondary
x-axis at the top (or the time with ``--step``).

.. code:: bash

   gpf_plot_history [options]

.. list-table::
   :header-rows: 1
   :widths: 20 10 10 60

   * - Flag
     - Type
     - Default
     - Description
   * - ``-s``, ``--step``
     - flag
     - off
     - Use the step number instead of time as x-axis (for a single run, time is then shown on
       the secondary x-axis).

gpf_plot_frame
^^^^^^^^^^^^^^

Plot a single time step of the solution fields (density, momentum, pressure, shear stress).
In 1D, multiple selected runs are overlaid in different colors, with a legend showing the run directories.

.. code:: bash

   gpf_plot_frame [options]

.. list-table::
   :header-rows: 1
   :widths: 20 10 10 60

   * - Flag
     - Type
     - Default
     - Description
   * - ``-d``, ``--dim``
     - int
     - 1
     - Dimension. ``1``: line plots. ``2``: 2D heatmaps.
   * - ``-f``, ``--frame``
     - int
     - -1
     - Frame index to plot. ``-1`` selects the last frame.

gpf_plot_frames
^^^^^^^^^^^^^^^

Overlay the centerline solution from all time steps on shared axes. A color gradient indicates time progression (darker = later).

.. code:: bash

   gpf_plot_frames [options]

.. list-table::
   :header-rows: 1
   :widths: 20 10 10 60

   * - Flag
     - Type
     - Default
     - Description
   * - ``-e``, ``--every``
     - int
     - 1
     - Plot every Nth frame. Useful for reducing clutter when many time steps are available.

Animation
---------

gpf_animate1d
^^^^^^^^^^^^^

Create an animation of a 1D simulation. Can be displayed interactively or saved to MP4.
If GP history files (``gp_zz.csv``, ``gp_xz.csv``) are present, uncertainty bands are shown automatically.

.. code:: bash

   gpf_animate1d [options]

.. list-table::
   :header-rows: 1
   :widths: 20 10 10 60

   * - Flag
     - Type
     - Default
     - Description
   * - ``-s``, ``--save``
     - flag
     - off
     - Save animation to an MP4 file.
   * - ``-p``, ``--path``
     - str
     - ``.``
     - Path to search for simulation output files.
   * - ``-m``, ``--mode``
     - str
     - ``single``
     - File selection mode (``single``: interactive selection of one simulation).

gpf_animate2d
^^^^^^^^^^^^^

Create an animation of a 2D simulation as a grid of heatmaps evolving over time.

.. code:: bash

   gpf_animate2d

This command has no additional arguments.
