# Diagonal Scaling for 2D FEM Energy Equation

**Date**: 2026-01-13
**Status**: Implemented

## Problem Summary

The 2D FEM energy advection diverges at step 1 due to matrix ill-conditioning:
- Condition number: ~7.3×10¹⁰
- Root cause: Energy variable E ranges 0→2×10⁶ (E = ρ·cv·T), creating large off-diagonal Jacobian entries while the diagonal (from time derivative) remains small

### Comparison with 1D

The 1D case works with dt=0.1 because:
- 1D diagonal contribution: ~0.133 per node
- 2D diagonal contribution: ~0.01 per node (13x smaller due to element area dx·dy vs length dx)

## Implemented Approach: Diagonal Scaling in FEM Assembly Layout

**Why this approach:**
- Scaling integrated into assembly layout (consistent with codebase architecture)
- Block indices stored during COO construction (no redundant computation)
- Characteristic scales derived from problem specification (not hardcoded)
- O(nnz) overhead (negligible)
- Extensible for adaptive scaling (Options B/C)

## Implementation Details

### 1. Scaling Transformation

Transform the linear system `J·dq = -R` to:
```
J* · dq* = -R*

where:
  D_q = diag(q_scale)     # variable scaling (characteristic magnitudes)
  D_R = diag(R_scale)     # residual scaling (same as D_q by convention)
  J*  = D_R⁻¹ · J · D_q
  R*  = D_R⁻¹ · R
  dq  = D_q · dq*
```

### 2. Scale Factor Derivation

Scales are derived from problem specification in `FEMSolver2D._get_characteristic_scales()`:

| Variable | Derivation | Example Value |
|----------|------------|---------------|
| ρ | `prop['rho0']` | ~1000 kg/m³ |
| jx, jy | `rho_ref * max(U, V)` | ~1000 kg/(m²·s) |
| E | `rho_ref * cv * T_ref` | ~2×10⁸ J/m³ |

Residual scales match the corresponding variable scales (mass↔ρ, momentum↔j, energy↔E).

### 3. Code Structure

```
assembly_layout.py
├── MatrixCOOPattern     # Extended with res_block_idx, var_block_idx (int8)
├── RHSPattern           # Extended with res_block_idx (int8)
├── ScalingInfo          # New: stores coo_scale, rhs_scale, sol_scale
│   ├── scale_system()   # M_scaled, R_scaled = scale_system(M, R)
│   └── unscale_solution() # dq = unscale_solution(dq_scaled)
└── FEMAssemblyLayout
    └── build_scaling()  # Creates ScalingInfo from char_scales + block indices

solver_fem_2d.py
├── _build_matrix_coo_pattern()  # Now stores res_block_idx, var_block_idx
├── _build_rhs_pattern()         # Now stores res_block_idx
├── _get_characteristic_scales() # New: derives scales from problem spec
├── _init_linear_solver()        # Now calls build_scaling()
└── update_dynamic()             # Uses self.scaling.scale_system/unscale_solution
```

### 4. Newton Loop Integration

```python
for it in range(max_iter):
    M, R = self.solver_step_fun(q)

    if R_norm < tol:
        break

    # Scale system for better conditioning
    M_scaled, R_scaled = self.scaling.scale_system(M, R)

    # Assemble and solve
    self.linear_solver.assemble(M_scaled, R_scaled)
    dq_scaled = self.linear_solver.solve(nb_inner_pts, nb_vars)

    # Unscale solution
    dq = self.scaling.unscale_solution(dq_scaled)

    q = q + alpha * dq
```

### 5. Performance

| Operation | Cost | When |
|-----------|------|------|
| Store block indices | O(nnz) | Once during COO construction |
| Build ScalingInfo | O(nnz) | Once at solver init |
| Scale M | O(nnz) multiply | Each Newton iteration |
| Scale R | O(n) divide | Each Newton iteration |
| Unscale dq | O(n) multiply | Each Newton iteration |

Memory overhead: ~15 MB for 100×100 grid with 4 variables.

### 6. Future Extensions

**Option B - Adaptive diagonal scaling:**
```python
def update_scaling_adaptive(self, J_diagonal):
    self.coo_scale = 1.0 / np.sqrt(np.abs(J_diagonal))
```

**Option C - Solution-based scaling:**
```python
def update_scaling_solution(self, q_current):
    self.coo_scale = 1.0 / np.maximum(np.abs(q_current), eps)
```

## Key Clarifications

### Periodic BCs Work Correctly

There is no periodic BC bug. The implementation correctly handles periodicity:

1. **muGrid ghost cell communication** handles periodic wrap-around before FEM assembly. Ghost cells at periodic boundaries receive values from the opposite domain side before field interpolation/derivatives are computed.

2. **FROM indices reflect periodic wrap-around** - The `index_mask_padded_local` in `grid_index.py` correctly maps ghost cells at periodic boundaries to their corresponding inner points, enabling periodic coupling in the Jacobian columns.

3. **TO indices correctly exclude ghost points** - Only inner points have residual equations (`sq_TO_inner` returns -1 for boundary corners). Ghost points provide neighbor information for derivative computations, not residual equations.

## Expected Outcome

After implementing diagonal scaling:
- Condition number reduced from ~10¹⁰ to ~10³-10⁴
- Stable convergence with reasonable timesteps (dt ~ 0.01-0.1)
- No changes to physics or assembly code

## Alternative Approaches (Not Recommended)

1. **Full non-dimensionalization** - More invasive, requires changes throughout physical models
2. **SUPG stabilization** - For convection-dominated problems, but doesn't address conditioning
3. **Reducing timestep** - Works (dt=1e-5 converges) but computationally expensive
