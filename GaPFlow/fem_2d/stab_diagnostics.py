"""Stabilization diagnostics for parameter analysis across use cases.

This module captures and saves stabilization parameters (raw and scaled)
to enable comparative analysis across different simulation configurations.
"""

import json
import os
from typing import Any, Dict

import numpy as np


class StabDiagnostics:
    """Captures and saves stabilization parameter data for analysis.

    Collects tau values (raw and block-scaled) at regular intervals to
    support analysis of stabilization parameter sensitivity across use cases.

    Parameters
    ----------
    solver : FEMSolver2D
        Reference to the solver instance for accessing fields and scaling.
    output_dir : str
        Directory path for saving JSON output files.
    case_name : str
        Identifier for this simulation case (typically config filename).
    save_interval : int, optional
        Capture data every N timesteps (default: 10).
    """

    def __init__(self, solver, output_dir: str, case_name: str,
                 save_interval: int = 10):
        self.solver = solver
        self.output_dir = output_dir
        self.case_name = case_name
        self.save_interval = save_interval

        # Initialize data structure
        self.data: Dict[str, Any] = {
            "case_name": case_name,
            "config_path": None,
            "input_variables": {},
            "alpha_values": {},
            "block_scales": {},
            "timesteps": []
        }

        # Capture static data at initialization
        self._capture_config_path()
        self._capture_input_variables()
        self._capture_alpha_values()
        self._capture_block_scales()

    def _capture_config_path(self) -> None:
        """Capture the config file path if available."""
        p = self.solver.problem
        if hasattr(p, 'outdir') and p.outdir:
            config_path = os.path.join(p.outdir, 'config.yml')
            if os.path.exists(config_path):
                self.data["config_path"] = config_path

    def _capture_input_variables(self) -> None:
        """Capture input variables used in tau calculations."""
        p = self.solver.problem

        # Grid parameters
        Nx = p.grid['Nx']
        Ny = p.grid['Ny']
        dx = p.grid['Lx'] / Nx
        dy = p.grid['Ly'] / Ny
        h = np.sqrt(dx * dy)
        grid_scale = (Nx * Ny) / 2500.0

        self.data["input_variables"] = {
            # From prop dict
            "rho0": float(p.prop['rho0']),
            "P0": float(p.prop.get('P0', 1.0)),
            "eta": float(p.prop.get('shear', 0.0)),  # shear viscosity
            # From geo dict
            "U": float(p.geo['U']),
            "V": float(p.geo['V']),
            # From numerics dict
            "dt": float(p.numerics['dt']),
            # From grid dict
            "Nx": int(Nx),
            "Ny": int(Ny),
            "Lx": float(p.grid['Lx']),
            "Ly": float(p.grid['Ly']),
            # Derived
            "h": float(h),
            "grid_scale": float(grid_scale),
        }

        # Energy-specific variables (if enabled)
        if self.solver.energy and p.energy_spec:
            self.data["input_variables"]["cv"] = float(p.energy_spec.get('cv', 0))
            self.data["input_variables"]["T_wall"] = float(
                p.energy_spec.get('T_wall', 0))
        else:
            self.data["input_variables"]["cv"] = None
            self.data["input_variables"]["T_wall"] = None

    def _capture_alpha_values(self) -> None:
        """Capture stabilization alpha parameters from config."""
        fem = self.solver.problem.fem_solver

        self.data["alpha_values"] = {
            "pressure": float(fem['pressure_stab_alpha']),
            "momentum": float(fem['momentum_stab_alpha']),
            "energy": float(fem['energy_stab_alpha']),
        }

    def _capture_block_scales(self) -> None:
        """Capture block scaling factors from solver."""
        if hasattr(self.solver, 'scaling') and self.solver.scaling is not None:
            char_scales = self.solver.scaling.char_scales
            self.data["block_scales"] = {
                "rho": float(char_scales.get('rho', 1.0)),
                "j": float(char_scales.get('jx', 1.0)),  # jx == jy
            }
            if 'E' in char_scales:
                self.data["block_scales"]["E"] = float(char_scales['E'])
            else:
                self.data["block_scales"]["E"] = None
        else:
            # Scaling not yet initialized
            self.data["block_scales"] = {"rho": None, "j": None, "E": None}

    def capture(self, step: int) -> None:
        """Capture tau values at current timestep.

        Parameters
        ----------
        step : int
            Current simulation timestep number.
        """
        # Get tau values from quadrature manager
        quad_mgr = self.solver.quad_mgr

        # Interior slice (same as used in computation)
        s = np.s_[..., :-1, :-1]

        # tau_mass is constant, take first value
        tau_mass_arr = quad_mgr.get('tau_mass')[s]
        tau_mass = float(tau_mass_arr.flat[0]) if tau_mass_arr.size > 0 else 0.0

        # tau_mom varies spatially, compute statistics
        tau_mom_arr = quad_mgr.get('tau_mom')[s]
        if tau_mom_arr.size > 0:
            tau_mom_min = float(np.min(tau_mom_arr))
            tau_mom_max = float(np.max(tau_mom_arr))
            tau_mom_mean = float(np.mean(tau_mom_arr))
        else:
            tau_mom_min = tau_mom_max = tau_mom_mean = 0.0

        # tau_energy if energy enabled
        if self.solver.energy:
            tau_energy_arr = quad_mgr.get('tau_energy')[s]
            tau_energy = float(tau_energy_arr.flat[0]) if tau_energy_arr.size > 0 else 0.0
        else:
            tau_energy = None

        # Get block scales for effective tau computation
        scales = self.data["block_scales"]
        rho_scale = scales["rho"] if scales["rho"] else 1.0
        j_scale = scales["j"] if scales["j"] else 1.0
        E_scale = scales["E"] if scales["E"] else 1.0

        # Compute effective tau (after block scaling)
        # The RHS is divided by the block scale, so effective tau = tau / scale
        tau_mass_eff = tau_mass / rho_scale
        tau_mom_min_eff = tau_mom_min / j_scale
        tau_mom_max_eff = tau_mom_max / j_scale
        tau_mom_mean_eff = tau_mom_mean / j_scale

        if tau_energy is not None:
            tau_energy_eff = tau_energy / E_scale
        else:
            tau_energy_eff = None

        # Store timestep data
        timestep_data = {
            "step": int(step),
            "tau_raw": {
                "mass": tau_mass,
                "mom_min": tau_mom_min,
                "mom_max": tau_mom_max,
                "mom_mean": tau_mom_mean,
                "energy": tau_energy,
            },
            "tau_effective": {
                "mass": tau_mass_eff,
                "mom_min": tau_mom_min_eff,
                "mom_max": tau_mom_max_eff,
                "mom_mean": tau_mom_mean_eff,
                "energy": tau_energy_eff,
            }
        }

        self.data["timesteps"].append(timestep_data)

    def save(self) -> None:
        """Save JSON files to output directory.

        Creates:
        - stab_data.json: Full time-series data
        - summary.json: Aggregated statistics for quick comparison
        """
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Re-capture block scales in case they weren't available at init
        if self.data["block_scales"]["rho"] is None:
            self._capture_block_scales()

        # Save full data
        data_path = os.path.join(self.output_dir, 'stab_data.json')
        with open(data_path, 'w') as f:
            json.dump(self.data, f, indent=2)

        # Generate and save summary
        summary = self._generate_summary()
        summary_path = os.path.join(self.output_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics from captured data."""
        summary = {
            "case_name": self.case_name,
            "input_variables": self.data["input_variables"],
            "alpha_values": self.data["alpha_values"],
            "block_scales": self.data["block_scales"],
            "num_timesteps_captured": len(self.data["timesteps"]),
        }

        if not self.data["timesteps"]:
            summary["tau_summary"] = None
            summary["tau_effective_summary"] = None
            return summary

        timesteps = self.data["timesteps"]

        # Extract raw tau values
        mass_values = [t["tau_raw"]["mass"] for t in timesteps]
        mom_mean_values = [t["tau_raw"]["mom_mean"] for t in timesteps]
        energy_values = [t["tau_raw"]["energy"] for t in timesteps
                         if t["tau_raw"]["energy"] is not None]

        summary["tau_summary"] = {
            "mass": {
                "initial": mass_values[0] if mass_values else None,
                "final": mass_values[-1] if mass_values else None,
                "mean": float(np.mean(mass_values)) if mass_values else None,
            },
            "mom": {
                "initial_mean": mom_mean_values[0] if mom_mean_values else None,
                "final_mean": mom_mean_values[-1] if mom_mean_values else None,
                "overall_mean": float(np.mean(mom_mean_values)) if mom_mean_values else None,
            },
            "energy": {
                "initial": energy_values[0] if energy_values else None,
                "final": energy_values[-1] if energy_values else None,
                "mean": float(np.mean(energy_values)) if energy_values else None,
            } if energy_values else None,
        }

        # Extract effective tau values
        mass_eff_values = [t["tau_effective"]["mass"] for t in timesteps]
        mom_eff_values = [t["tau_effective"]["mom_mean"] for t in timesteps]
        energy_eff_values = [t["tau_effective"]["energy"] for t in timesteps
                             if t["tau_effective"]["energy"] is not None]

        summary["tau_effective_summary"] = {
            "mass": {
                "mean": float(np.mean(mass_eff_values)) if mass_eff_values else None,
            },
            "mom": {
                "mean": float(np.mean(mom_eff_values)) if mom_eff_values else None,
            },
            "energy": {
                "mean": float(np.mean(energy_eff_values)) if energy_eff_values else None,
            } if energy_eff_values else None,
        }

        return summary
