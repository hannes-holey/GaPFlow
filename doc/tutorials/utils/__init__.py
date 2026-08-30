#
# Copyright 2025 Christoph Huber
#
# ### MIT License
#
"""Utility functions for tutorials."""

from .topography import (
    regen_conv_slider_pocket_1d,
    regen_conv_slider_pocket_1d_giacopini,
    regen_conv_slider_pocket_1d_hansen,
    regen_conv_slider_pocket_2d,
    regen_conv_slider_pocket_2d_wide,
    regen_twin_parabolic_slider,
    regen_twin_parabolic_slider_id,
)
from .plotting import (
    animate_comparison,
    plot_solver_comparison_rho_jx,
    plot_reynolds_comparison_rho,
    plot_reynolds_comparison_rho_2d,
    animate_advection,
    animate_3d_surface,
    animate_3d_advection,
    plot_3d_snapshot,
    plot_lid_driven_cavity,
    plot_overview_1d,
    plot_overview_2d,
    plot_midsection_2d,
)

from .solver_comparison import (  # noqa: F401
    load_template,
    build_config,
    run_solver,
    run_all_solvers,
    print_timing,
    build_reynolds_config,
    run_reynolds_solver,
    run_reynolds_comparison,
    print_reynolds_timing,
    run_reynolds_solver_2d,
    run_reynolds_comparison_2d,
)

from .analytics import heat_equation_1d

__all__ = [
    'regen_conv_slider_pocket_1d',
    'regen_conv_slider_pocket_1d_giacopini',
    'regen_conv_slider_pocket_1d_hansen',
    'regen_conv_slider_pocket_2d',
    'regen_conv_slider_pocket_2d_wide',
    'regen_twin_parabolic_slider',
    'regen_twin_parabolic_slider_id',
    'animate_comparison',
    'plot_solver_comparison_rho_jx',
    'plot_reynolds_comparison_rho',
    'plot_reynolds_comparison_rho_2d',
    'animate_advection',
    'animate_3d_surface',
    'animate_3d_advection',
    'plot_3d_snapshot',
    'plot_lid_driven_cavity',
    'plot_overview_1d',
    'plot_overview_2d',
    'plot_midsection_2d',
    'heat_equation_1d',
]
