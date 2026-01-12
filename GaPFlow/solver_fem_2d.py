#
# Copyright 2025 Christoph Huber
#
# ### MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
from . import HAS_PETSC
from .fem_1d.num_solver import Solver
from .fem_2d.elements import TriangleQuadrature
from .fem_2d.terms import NonLinearTerm, get_active_terms
from .fem_2d.assembly_layout import (
    FEMAssemblyLayout,
    MatrixCOOPattern,
    RHSPattern,
    JacobianTermMap,
    RHSTermMap,
)
from .fem_2d.grid_index import GridIndexManager
from .fem_2d.quad_fields import QuadFieldManager

from functools import cached_property
import numpy as np
import time
from muGrid import Timer

import numpy.typing as npt
from typing import TYPE_CHECKING, Tuple, Dict
if TYPE_CHECKING:
    from .problem import Problem

NDArray = npt.NDArray[np.floating]


class FEMSolver2D:
    """FEM Solver for 2D problems using triangular elements."""

    def __init__(self, fem_spec: dict, problem: "Problem") -> None:
        self.fem_spec = fem_spec
        self.problem = problem
        self.num_solver = Solver(problem.fem_solver)
        self.quad_mgr = None  # QuadFieldManager, initialized via init()
        self.linear_solver = None  # Linear solver (PETSc or SciPy)
        self.timer = Timer()

    def _init_convenience_accessors(self) -> None:
        """Initialize convenience accessors for problem and grid properties."""
        p = self.problem
        self.decomp = p.decomp

        self.per_x = p.decomp.periodic_x
        self.per_y = p.decomp.periodic_y
        self.energy = p.fem_solver['equations']['energy']
        self.dynamic = True  # 2D always uses time-stepping

        self.global_coords = p.decomp.icoordsg

        nb_subdomain_pts = p.decomp.nb_subdomain_grid_pts
        self.Nx_inner = nb_subdomain_pts[0]
        self.Ny_inner = nb_subdomain_pts[1]

        self.dx = p.grid['dx']
        self.dy = p.grid['dy']
        self.A = self.dx * self.dy

        self.variables = ['rho', 'jx', 'jy']
        self.residuals = ['mass', 'momentum_x', 'momentum_y']
        self.field_map = {'rho': p.q[0], 'jx': p.q[1], 'jy': p.q[2]}
        if self.energy:
            self.variables.append('E')
            self.residuals.append('energy')
            self.field_map['E'] = p.energy.energy

        # Grid index manager (handles masks, connectivity, stencils)
        bc_neumann = {
            'xW': p.grid['bc_xW_N'], 'xE': p.grid['bc_xE_N'],
            'yS': p.grid['bc_yS_N'], 'yN': p.grid['bc_yN_N'],
        }
        self.grid_idx = GridIndexManager(
            Nx_inner=self.Nx_inner, Ny_inner=self.Ny_inner,
            bc_at_W=p.decomp.is_at_xW and not self.per_x,
            bc_at_E=p.decomp.is_at_xE and not self.per_x,
            bc_at_S=p.decomp.is_at_yS and not self.per_y,
            bc_at_N=p.decomp.is_at_yN and not self.per_y,
            decomp=p.decomp, variables=self.variables, bc_neumann=bc_neumann,
        )

        # Convenience aliases from grid index manager
        self.nb_inner_pts = self.grid_idx.nb_inner_pts
        self.sq_per_row = self.grid_idx.sq_per_row
        self.sq_per_col = self.grid_idx.sq_per_col
        self.nb_sq = self.grid_idx.nb_sq
        self.sq_x_arr = self.grid_idx.sq_x_arr
        self.sq_y_arr = self.grid_idx.sq_y_arr

        self.res_size = len(self.residuals) * self.nb_inner_pts
        self.var_size = len(self.variables) * self.grid_idx.nb_contributors

        # Quadrature data for triangular elements
        quad = TriangleQuadrature(self.dx, self.dy)
        w = quad.weights
        Nl, Nr = quad.N_left, quad.N_right

        # Precomputed element tensors: elem[i,j,q] = N_i[q] * N_j[q] * w[q] * A
        self.elem_tensor_left = Nl[:, None, :] * Nl[None, :, :] * w * self.A
        self.elem_tensor_right = Nr[:, None, :] * Nr[None, :, :] * w * self.A

        # Precomputed test function integrals: integral[i,q] = N_i[q] * w[q] * A
        self.test_wA_left = Nl * w * self.A
        self.test_wA_right = Nr * w * self.A
        self.test_wA_left_dx = quad.dN_left_dx * w * self.A
        self.test_wA_right_dx = quad.dN_right_dx * w * self.A
        self.test_wA_left_dy = quad.dN_left_dy * w * self.A
        self.test_wA_right_dy = quad.dN_right_dy * w * self.A

    def _get_active_terms(self) -> None:
        """Initialize list of active terms from problem fem_solver config."""
        self.terms = get_active_terms(self.fem_spec)

    def _init_quad_fields(self) -> None:
        """Initialize quadrature field manager with operators and fields."""
        self.quad_mgr = QuadFieldManager(
            problem=self.problem,
            energy=self.energy,
            variables=self.variables,
        )
        # Convenience aliases for backward compatibility
        self.nodal_fields = self.quad_mgr.nodal_fields
        self.quad_fields = self.quad_mgr.quad_fields

    def _build_jit_functions(self) -> None:
        """Build JIT-compiled gradient functions for all physical models."""
        p = self.problem

        p.pressure.build_grad()
        p.wall_stress_xz.build_grad()
        p.wall_stress_yz.build_grad()
        if self.energy:
            p.energy.build_grad()

    def _build_terms(self) -> None:
        """Build term contexts with callable references to quad_fields."""
        p = self.problem

        def make_getter(name):
            return lambda: self.quad_mgr.get(name)

        quad_field_names = self.quad_mgr.get_needed_fields()

        for term in self.terms:
            term_ctx = {
                name: make_getter(name)
                for name in quad_field_names
            }

            # Non-quad values
            term_ctx['dt'] = p.numerics['dt']

            if self.energy:
                term_ctx['k'] = lambda: p.energy.k

            term.build(term_ctx)

    # =========================================================================
    # Quadrature Field Update (delegates to quad_mgr)
    # =========================================================================

    def update_quad(self) -> None:
        """Full quadrature update (nodal fields, interpolation, derived quantities)."""
        self.quad_mgr.update_nodal_fields()
        self.quad_mgr.update_quad_nodal()
        self.quad_mgr.update_quad_computed()

    def update_prev_quad(self) -> None:
        """Store current quad values for time derivatives."""
        self.quad_mgr.store_prev_values()

    # =========================================================================
    # Assembly Layout (unified COO pattern and term mappings)
    # =========================================================================

    @cached_property
    def assembly_layout(self) -> FEMAssemblyLayout:
        """Precomputed assembly layout for O(nnz) matrix/vector assembly."""
        return self._build_assembly_layout()

    def _build_assembly_layout(self) -> FEMAssemblyLayout:
        """Build complete assembly layout with COO pattern and term mappings."""
        nb_vars = len(self.variables)
        nb_res = len(self.residuals)
        Nx, Ny = self.decomp.nb_domain_grid_pts

        # 1. Build local-to-global point mapping
        mask_local = self.grid_idx.index_mask_padded_local('')
        mask_global = self.decomp.index_mask_padded_global
        l2g_pts = np.zeros(self.grid_idx.nb_contributors, dtype=np.int32)
        valid = mask_local >= 0
        l2g_pts[mask_local[valid]] = mask_global[valid]

        # 2. Build matrix COO pattern
        matrix_coo, local_to_coo = self._build_matrix_coo_pattern(
            l2g_pts, nb_vars, nb_res)

        # 3. Build RHS pattern
        rhs = self._build_rhs_pattern(l2g_pts, nb_res)

        # 4. Build Jacobian term maps
        jacobian_terms = {}
        for term in self.terms:
            for dep_var in term.dep_vars:
                key = (term.name, term.res, dep_var)
                if term.d_dx_resfun:
                    jacobian_terms[key] = self._build_jacobian_term_deriv(
                        term, dep_var, local_to_coo, 'x')
                elif term.d_dy_resfun:
                    jacobian_terms[key] = self._build_jacobian_term_deriv(
                        term, dep_var, local_to_coo, 'y')
                else:
                    jacobian_terms[key] = self._build_jacobian_term_zero_der(
                        term, dep_var, local_to_coo)

        # 5. Build RHS term maps
        rhs_terms = {}
        for term in self.terms:
            key = (term.name, term.res)
            rhs_terms[key] = self._build_rhs_term(term)

        return FEMAssemblyLayout(
            matrix_coo=matrix_coo,
            rhs=rhs,
            jacobian_terms=jacobian_terms,
            rhs_terms=rhs_terms,
            dx=self.dx,
            dy=self.dy,
            nb_vars=nb_vars,
            nb_inner_pts=self.nb_inner_pts,
            nb_global_pts=Nx * Ny,
        )

    def _build_matrix_coo_pattern(self, l2g_pts: NDArray, nb_vars: int,
                                  nb_res: int) -> Tuple[MatrixCOOPattern, dict]:
        """Build sparse matrix COO pattern and local-to-COO map."""
        inner_pts, contrib_pts = self.grid_idx.get_stencil_connectivity()
        nnz_per_block = len(inner_pts)
        total_nnz = nb_res * nb_vars * nnz_per_block

        local_rows = np.empty(total_nnz, dtype=np.int32)
        local_cols = np.empty(total_nnz, dtype=np.int32)
        global_rows = np.empty(total_nnz, dtype=np.int32)
        global_cols = np.empty(total_nnz, dtype=np.int32)

        global_inner = l2g_pts[inner_pts]
        global_contrib = l2g_pts[contrib_pts]

        idx = 0
        for res_idx in range(nb_res):
            row_offset = res_idx * self.nb_inner_pts
            for var_idx in range(nb_vars):
                col_offset = var_idx * self.grid_idx.nb_contributors
                n = nnz_per_block

                local_rows[idx:idx + n] = row_offset + inner_pts
                local_cols[idx:idx + n] = col_offset + contrib_pts
                global_rows[idx:idx + n] = global_inner * nb_res + res_idx
                global_cols[idx:idx + n] = global_contrib * nb_vars + var_idx

                idx += n

        # Build local-to-COO lookup dictionary
        local_to_coo = {}
        for coo_idx in range(total_nnz):
            key = (int(local_rows[coo_idx]), int(local_cols[coo_idx]))
            local_to_coo[key] = coo_idx

        matrix_coo = MatrixCOOPattern(
            nnz=total_nnz,
            local_rows=local_rows,
            local_cols=local_cols,
            global_rows=global_rows,
            global_cols=global_cols,
        )

        return matrix_coo, local_to_coo

    def _build_rhs_pattern(self, l2g_pts: NDArray, nb_res: int) -> RHSPattern:
        """Build RHS vector assembly pattern."""
        local_size = nb_res * self.nb_inner_pts
        global_rows = np.empty(local_size, dtype=np.int32)
        global_pts = l2g_pts[:self.nb_inner_pts]

        for res_idx in range(nb_res):
            start = res_idx * self.nb_inner_pts
            end = start + self.nb_inner_pts
            global_rows[start:end] = global_pts * nb_res + res_idx

        return RHSPattern(size=local_size, global_rows=global_rows)

    def _build_rhs_term(self, term: NonLinearTerm) -> RHSTermMap:
        """Build RHS term map for a single term."""
        TO = self.grid_idx.sq_TO_inner
        res_idx = self.residuals.index(term.res)
        sq_x, sq_y = self.sq_x_arr, self.sq_y_arr

        # Select test weights based on term type
        if term.d_dx_resfun:
            test_wA_left = self.test_wA_left_dx if term.der_testfun else self.test_wA_left
            test_wA_right = self.test_wA_right_dx if term.der_testfun else self.test_wA_right
        elif term.d_dy_resfun:
            test_wA_left = self.test_wA_left_dy if term.der_testfun else self.test_wA_left
            test_wA_right = self.test_wA_right_dy if term.der_testfun else self.test_wA_right
        else:
            test_wA_left = self.test_wA_left
            test_wA_right = self.test_wA_right

        # Triangle configs: (TO_indices, test_weights, quad_offset)
        triangle_configs = [
            (TO[:, [0, 2, 1]], test_wA_left, 0),   # left: [bl, tl, br], quad 0-2
            (TO[:, [3, 1, 2]], test_wA_right, 3),  # right: [tr, br, tl], quad 3-5
        ]

        output_idx_list = []
        weights_list = []
        quad_idx_list = []
        sq_x_list = []
        sq_y_list = []

        for TO_tri, test_wA, quad_offset in triangle_configs:
            for i in range(3):
                valid = TO_tri[:, i] >= 0
                sq_valid = np.where(valid)[0]
                if len(sq_valid) > 0:
                    for q in range(3):
                        output_idx_list.append(
                            res_idx * self.nb_inner_pts + TO_tri[sq_valid, i])
                        weights_list.append(
                            np.full(len(sq_valid), test_wA[i, q], dtype=np.float64))
                        quad_idx_list.append(
                            np.full(len(sq_valid), q + quad_offset, dtype=np.int32))
                        sq_x_list.append(sq_x[sq_valid])
                        sq_y_list.append(sq_y[sq_valid])

        if not output_idx_list:
            return RHSTermMap(
                output_idx=np.array([], dtype=np.int32),
                weights=np.array([], dtype=np.float64),
                quad_idx=np.array([], dtype=np.int32),
                sq_x=np.array([], dtype=np.int32),
                sq_y=np.array([], dtype=np.int32),
            )

        return RHSTermMap(
            output_idx=np.concatenate(output_idx_list),
            weights=np.concatenate(weights_list),
            quad_idx=np.concatenate(quad_idx_list),
            sq_x=np.concatenate(sq_x_list),
            sq_y=np.concatenate(sq_y_list),
        )

    def _lookup_coo_indices(self, local_to_coo: Dict[Tuple[int, int], int],
                            row_block: NDArray, col_block: NDArray) -> NDArray:
        """
        Vectorized COO index lookup from dictionary.

        Returns array of COO indices, with -1 for positions not in pattern.
        """
        coo_idx = np.full(len(row_block), -1, dtype=np.int32)
        for k in range(len(row_block)):
            key = (int(row_block[k]), int(col_block[k]))
            if key in local_to_coo:
                coo_idx[k] = local_to_coo[key]
        return coo_idx

    def _build_jacobian_term_zero_der(self, term: NonLinearTerm, dep_var: str,
                                      local_to_coo: dict) -> JacobianTermMap:
        """Build Jacobian term map for zero-derivative term (3×3 per triangle)."""
        res_idx = self.residuals.index(term.res)
        var_idx = self.variables.index(dep_var)
        sq_x, sq_y = self.sq_x_arr, self.sq_y_arr

        TO = self.grid_idx.sq_TO_inner
        FROM = self.grid_idx.sq_FROM_padded(dep_var)

        # Triangle configs: (TO_indices, FROM_indices, elem_tensor, quad_offset)
        triangle_configs = [
            (TO[:, [0, 2, 1]], FROM[:, [0, 2, 1]], self.elem_tensor_left, 0),
            (TO[:, [3, 1, 2]], FROM[:, [3, 1, 2]], self.elem_tensor_right, 3),
        ]

        coo_idx_list = []
        weights_list = []
        quad_idx_list = []
        sq_x_list = []
        sq_y_list = []

        for TO_tri, FROM_tri, elem_tensor, quad_offset in triangle_configs:
            for i in range(3):
                for j in range(3):
                    valid = (TO_tri[:, i] >= 0) & (FROM_tri[:, j] >= 0)
                    sq_valid = np.where(valid)[0]

                    if len(sq_valid) > 0:
                        row_block = res_idx * self.nb_inner_pts + TO_tri[sq_valid, i]
                        col_block = var_idx * self.grid_idx.nb_contributors + FROM_tri[sq_valid, j]
                        coo_idx = self._lookup_coo_indices(local_to_coo, row_block, col_block)
                        valid_coo = coo_idx >= 0

                        if np.any(valid_coo):
                            sq_final = sq_valid[valid_coo]
                            for q in range(3):
                                coo_idx_list.append(coo_idx[valid_coo])
                                weights_list.append(np.full(
                                    len(sq_final), elem_tensor[i, j, q], dtype=np.float64))
                                quad_idx_list.append(np.full(
                                    len(sq_final), q + quad_offset, dtype=np.int32))
                                sq_x_list.append(sq_x[sq_final])
                                sq_y_list.append(sq_y[sq_final])

        if not coo_idx_list:
            return JacobianTermMap(
                coo_idx=np.array([], dtype=np.int32),
                weights=np.array([], dtype=np.float64),
                quad_idx=np.array([], dtype=np.int32),
                sq_x=np.array([], dtype=np.int32),
                sq_y=np.array([], dtype=np.int32),
            )

        return JacobianTermMap(
            coo_idx=np.concatenate(coo_idx_list),
            weights=np.concatenate(weights_list),
            quad_idx=np.concatenate(quad_idx_list),
            sq_x=np.concatenate(sq_x_list),
            sq_y=np.concatenate(sq_y_list),
        )

    def _build_jacobian_term_deriv(self, term: NonLinearTerm, dep_var: str,
                                   local_to_coo: dict, direction: str) -> JacobianTermMap:
        """Build Jacobian term map for derivative term (±1/d* stencil)."""
        res_idx = self.residuals.index(term.res)
        var_idx = self.variables.index(dep_var)

        TO = self.grid_idx.sq_TO_inner
        FROM = self.grid_idx.sq_FROM_padded(dep_var)

        TO_left = TO[:, [0, 2, 1]]
        TO_right = TO[:, [3, 1, 2]]

        if direction == 'x':
            FROM_neg_left, FROM_pos_left = FROM[:, 0], FROM[:, 1]
            FROM_neg_right, FROM_pos_right = FROM[:, 2], FROM[:, 3]
            test_wA_left = self.test_wA_left_dx if term.der_testfun else self.test_wA_left
            test_wA_right = self.test_wA_right_dx if term.der_testfun else self.test_wA_right
        else:
            FROM_neg_left, FROM_pos_left = FROM[:, 0], FROM[:, 2]
            FROM_neg_right, FROM_pos_right = FROM[:, 1], FROM[:, 3]
            test_wA_left = self.test_wA_left_dy if term.der_testfun else self.test_wA_left
            test_wA_right = self.test_wA_right_dy if term.der_testfun else self.test_wA_right

        coo_idx_list = []
        weights_list = []
        quad_idx_list = []
        sq_x_list = []
        sq_y_list = []
        signs_list = []

        sq_x, sq_y = self.sq_x_arr, self.sq_y_arr

        triangle_configs = [
            (TO_left, FROM_neg_left, FROM_pos_left, test_wA_left, 0),
            (TO_right, FROM_neg_right, FROM_pos_right, test_wA_right, 3),
        ]

        for TO_tri, FROM_neg, FROM_pos, test_wA, quad_offset in triangle_configs:
            for i in range(3):
                for FROM_pts, sign in [(FROM_neg, -1.0), (FROM_pos, 1.0)]:
                    valid = (TO_tri[:, i] >= 0) & (FROM_pts >= 0)
                    sq_valid = np.where(valid)[0]

                    if len(sq_valid) == 0:
                        continue

                    row_block = res_idx * self.nb_inner_pts + TO_tri[sq_valid, i]
                    col_block = var_idx * self.grid_idx.nb_contributors + FROM_pts[sq_valid]
                    coo_idx = self._lookup_coo_indices(local_to_coo, row_block, col_block)
                    valid_coo = coo_idx >= 0

                    if not np.any(valid_coo):
                        continue

                    sq_final = sq_valid[valid_coo]
                    for q in range(3):
                        coo_idx_list.append(coo_idx[valid_coo])
                        weights_list.append(np.full(
                            len(sq_final), test_wA[i, q], dtype=np.float64))
                        quad_idx_list.append(np.full(
                            len(sq_final), q + quad_offset, dtype=np.int32))
                        sq_x_list.append(sq_x[sq_final])
                        sq_y_list.append(sq_y[sq_final])
                        signs_list.append(np.full(len(sq_final), sign, dtype=np.float64))

        if not coo_idx_list:
            return JacobianTermMap(
                coo_idx=np.array([], dtype=np.int32),
                weights=np.array([], dtype=np.float64),
                quad_idx=np.array([], dtype=np.int32),
                sq_x=np.array([], dtype=np.int32),
                sq_y=np.array([], dtype=np.int32),
                signs=np.array([], dtype=np.float64),
            )

        return JacobianTermMap(
            coo_idx=np.concatenate(coo_idx_list),
            weights=np.concatenate(weights_list),
            quad_idx=np.concatenate(quad_idx_list),
            sq_x=np.concatenate(sq_x_list),
            sq_y=np.concatenate(sq_y_list),
            signs=np.concatenate(signs_list),
        )

    # =========================================================================
    # Sparse Assembly Methods (O(nnz) direct COO value computation)
    # =========================================================================

    def get_tang_matrix_sparse(self) -> NDArray:
        """Assemble tangent matrix directly to COO values - O(nnz) memory."""
        layout = self.assembly_layout
        coo_values = np.zeros(layout.matrix_coo.nnz, dtype=np.float64)

        for term in self.terms:
            for dep_var in term.dep_vars:
                self._accumulate_term_sparse(term, dep_var, coo_values)

        return coo_values

    def _accumulate_term_sparse(self, term: NonLinearTerm, dep_var: str,
                                coo_values: NDArray):
        """Dispatch to appropriate sparse assembly method based on term type."""
        if term.d_dx_resfun:
            self._accumulate_term_deriv_sparse(term, dep_var, coo_values, 'x')
        elif term.d_dy_resfun:
            self._accumulate_term_deriv_sparse(term, dep_var, coo_values, 'y')
        else:
            self._accumulate_term_zero_der_sparse(term, dep_var, coo_values)

    def _accumulate_term_zero_der_sparse(self, term: NonLinearTerm, dep_var: str,
                                         coo_values: NDArray):
        """Sparse assembly for zero-derivative term."""
        key = (term.name, term.res, dep_var)
        idx = self.assembly_layout.jacobian_terms[key]

        if len(idx.coo_idx) == 0:
            return

        res_deriv = self.get_res_deriv_vals(term, dep_var)
        res_vals = res_deriv[idx.quad_idx, idx.sq_x, idx.sq_y]
        contrib = idx.weights * res_vals
        np.add.at(coo_values, idx.coo_idx, contrib)

    def _accumulate_term_deriv_sparse(self, term: NonLinearTerm, dep_var: str,
                                      coo_values: NDArray, direction: str):
        """Sparse assembly for derivative term with ±1/d* stencil."""
        key = (term.name, term.res, dep_var)
        idx = self.assembly_layout.jacobian_terms[key]

        if len(idx.coo_idx) == 0:
            return

        res_deriv = self.get_res_deriv_vals(term, dep_var)
        res_vals = res_deriv[idx.quad_idx, idx.sq_x, idx.sq_y]
        inv_d = 1.0 / (self.dx if direction == 'x' else self.dy)
        contrib = idx.weights * res_vals * idx.signs * inv_d
        np.add.at(coo_values, idx.coo_idx, contrib)

    # =========================================================================
    # Residual Vector Assembly
    # =========================================================================

    def get_res_deriv_vals(self, term: NonLinearTerm, dep_var: str) -> NDArray:
        """Evaluate derivative of residual function w.r.t. dep_var at quadrature points."""
        dep_var_vals = {v: self.quad_mgr.get(v) for v in term.dep_vars}
        return term.evaluate_deriv(dep_var, *[dep_var_vals[v] for v in term.dep_vars])

    def get_res_vals(self, term: NonLinearTerm) -> NDArray:
        """Evaluate residual function at quadrature points."""
        dep_var_vals = {v: self.quad_mgr.get(v) for v in term.dep_vars}
        return term.evaluate(*[dep_var_vals[v] for v in term.dep_vars])

    def residual_vector_term(self, term: NonLinearTerm) -> NDArray:
        """Wrapper for different spatial derivatives (nb_inner_pts,)."""
        if term.d_dx_resfun:
            return self._residual_vector_term_deriv(term, 'x')
        elif term.d_dy_resfun:
            return self._residual_vector_term_deriv(term, 'y')
        else:
            return self._residual_vector_term_zero_der(term)

    def _assemble_residual_contributions(self, contrib_left: NDArray,
                                         contrib_right: NDArray) -> NDArray:
        """Assemble triangle contributions into residual vector."""
        R = np.zeros((self.nb_inner_pts,), dtype=float)
        TO = self.grid_idx.sq_TO_inner

        # Triangle configs: (TO_indices, contributions)
        for TO_tri, contrib in [(TO[:, [0, 2, 1]], contrib_left),
                                (TO[:, [3, 1, 2]], contrib_right)]:
            for i in range(3):
                valid = TO_tri[:, i] != -1
                np.add.at(R, TO_tri[valid, i], contrib[valid, i])
        return R

    def _residual_vector_term_zero_der(self, term: NonLinearTerm) -> NDArray:
        """Get residual vector for term without derivative in residual function."""
        res_fun_vals = self.get_res_vals(term)  # (6, sq_per_row, sq_per_col)

        sx, sy = self.sq_x_arr, self.sq_y_arr
        res_left = res_fun_vals[0:3, sx, sy].T   # (nb_sq, 3)
        res_right = res_fun_vals[3:6, sx, sy].T

        # contrib[sq, i] = sum_q(N_i[q] * res[sq, q] * w[q]) * A
        contrib_left = np.einsum('iq,sq->si', self.test_wA_left, res_left)
        contrib_right = np.einsum('iq,sq->si', self.test_wA_right, res_right)

        return self._assemble_residual_contributions(contrib_left, contrib_right)

    def _residual_vector_term_deriv(self, term: NonLinearTerm, direction: str) -> NDArray:
        """Get residual vector for term with spatial derivative in residual function."""
        # Select derivative operator and test weights based on direction
        if direction == 'x':
            get_field_deriv = self.quad_mgr.get_deriv_dx
            test_wA_left = self.test_wA_left_dx if term.der_testfun else self.test_wA_left
            test_wA_right = self.test_wA_right_dx if term.der_testfun else self.test_wA_right
        else:  # 'y'
            get_field_deriv = self.quad_mgr.get_deriv_dy
            test_wA_left = self.test_wA_left_dy if term.der_testfun else self.test_wA_left
            test_wA_right = self.test_wA_right_dy if term.der_testfun else self.test_wA_right

        # Compute dF/d* using chain rule: (6, sq_per_row, sq_per_col)
        dF = np.zeros((6, self.sq_per_row, self.sq_per_col))
        for dep_var in term.dep_vars:
            dF_dvar = self.get_res_deriv_vals(term, dep_var)  # (6, X, Y)
            dvar_2 = get_field_deriv(dep_var)                 # (2, X, Y)
            dvar_6 = np.repeat(dvar_2, 3, axis=0)             # expand to (6, X, Y)
            dF += dF_dvar * dvar_6

        sx, sy = self.sq_x_arr, self.sq_y_arr
        res_left = dF[0:3, sx, sy].T   # (nb_sq, 3)
        res_right = dF[3:6, sx, sy].T

        contrib_left = np.einsum('iq,sq->si', test_wA_left, res_left)
        contrib_right = np.einsum('iq,sq->si', test_wA_right, res_right)

        return self._assemble_residual_contributions(contrib_left, contrib_right)

    def get_residual_vec(self) -> NDArray:
        """Assemble full residual vector from all terms (res_size,)."""
        res_vec = np.zeros(self.res_size)
        for term in self.terms:
            sl = self._res_slice(term.res)
            res_vec[sl] += self.residual_vector_term(term)
        return res_vec

    # =========================================================================
    # System Assembly Entry Points
    # =========================================================================

    def _res_slice(self, res_name: str) -> slice:
        """Slice for residual block in local arrays."""
        i = self.residuals.index(res_name)
        return slice(i * self.nb_inner_pts, (i + 1) * self.nb_inner_pts)

    def get_M(self) -> NDArray:
        """Get tangent matrix (COO values for sparse assembly)."""
        return self.get_tang_matrix_sparse()

    def get_M_dense(self) -> NDArray:
        """Get tangent matrix as dense array (for testing/debugging)."""
        coo_values = self.get_tang_matrix_sparse()
        layout = self.assembly_layout
        rows = layout.matrix_coo.local_rows
        cols = layout.matrix_coo.local_cols

        M = np.zeros((self.res_size, self.var_size), dtype=np.float64)
        np.add.at(M, (rows, cols), coo_values)
        return M

    def get_R(self) -> NDArray:
        """Get residual vector."""
        return self.get_residual_vec()

    # =========================================================================
    # Solution Vector Management
    # =========================================================================

    def _sol_slice(self, var_name: str) -> slice:
        """Slice for solution vector (uses nb_inner_pts, not nb_contributors)."""
        i = self.variables.index(var_name)
        return slice(i * self.nb_inner_pts, (i + 1) * self.nb_inner_pts)

    def get_nodal_val(self, field_name: str) -> NDArray:
        """Returns the inner nodal values of a field in shape (nb_inner_pts,)."""
        return self.field_map[field_name][1:-1, 1:-1].flatten(order='F')

    def get_q_nodal(self) -> NDArray:
        """Returns the full solution vector q in nodal values shape (nb_vars*nb_inner_pts,)."""
        q_nodal = np.zeros(self.res_size)
        for var in self.variables:
            q_nodal[self._sol_slice(var)] = self.get_nodal_val(var)
        return q_nodal

    def set_q_nodal(self, q_nodal: NDArray) -> None:
        """Sets the full solution vector q from nodal values shape (nb_vars*nb_inner_pts,).
        """
        for var in self.variables:
            var_nodal = q_nodal[self._sol_slice(var)]
            self.field_map[var][1:-1, 1:-1] = var_nodal.reshape(
                (self.Nx_inner, self.Ny_inner), order='F')

    # =========================================================================
    # Solver Interface
    # =========================================================================

    def solver_step_fun(self, q_guess: NDArray) -> Tuple[NDArray, NDArray]:
        """Newton solver step: set guess, update fields, return (M, R)."""
        self.set_q_nodal(q_guess)
        with self.timer("update_quad"):
            self.update_quad()
        with self.timer("jacobian"):
            M = self.get_M()
        with self.timer("residual"):
            R = self.get_R()
        return M, R

    def update_output_fields(self) -> None:
        """Update nodal output fields (wall stress, bulk stress) for plotting/output."""
        p = self.problem
        p.wall_stress_xz.update()
        p.wall_stress_yz.update()
        if hasattr(p, 'bulk_stress'):
            p.bulk_stress.update()

    def update_dynamic(self) -> None:
        """Do a single dynamic time step update using PETSc."""
        p = self.problem
        fem_solver = p.fem_solver

        with self.timer("timestep"):
            with self.timer("update_prev_quad"):
                self.update_prev_quad()

            tic = time.time()

            q = self.get_q_nodal().copy()
            max_iter = fem_solver.get('max_iter', 100)
            tol = fem_solver.get('R_norm_tol', 1e-6)
            alpha = fem_solver.get('alpha', 1.0)

            for it in range(max_iter):
                with self.timer("newton_iteration"):
                    M, R = self.solver_step_fun(q)
                    R_norm = np.linalg.norm(R)

                    if R_norm < tol:
                        break

                    # Assemble and solve with PETSc
                    with self.timer("petsc_assemble"):
                        self.linear_solver.assemble(M, R)
                    with self.timer("petsc_solve"):
                        dq = self.linear_solver.solve(self.nb_inner_pts, len(self.variables))

                    q = q + alpha * dq

                    # Update solver state
                    self.set_q_nodal(q)
                    p.decomp.communicate_ghost_buffers(p)
                    with self.timer("update_quad_post"):
                        self.update_quad()

            toc = time.time()
            self.time_inner = toc - tic
            self.inner_iterations = it + 1  # Store number of iterations

            with self.timer("update_output_fields"):
                self.update_output_fields()

        p._post_update()

    def update(self) -> None:
        """Top-level solver update function."""
        self.update_dynamic()

    def print_status_header(self) -> None:
        """Print header for simulation status output."""
        p = self.problem
        if not p.options['silent'] and p.decomp.rank == 0:
            print(75 * '-')
            print(f"{'Step':<6s} {'Timestep':<12s} {'Time':<12s} {'Iter':<6s} {'Conv. Time':<12s} {'Residual':<12s}")
            print(75 * '-')
            p.write(params=False)

    def print_status(self, scalars=None) -> None:
        """Print status line for simulation."""
        p = self.problem
        if not p.options['silent'] and p.decomp.rank == 0:
            print(f"{p.step:<6d} {p.dt:<12.4e} {p.simtime:<12.4e} "
                  f"{self.inner_iterations:<6d} {self.time_inner:<12.4e} {p.residual:<12.4e}")

    def print_timer_summary(self, save_json: str = "") -> None:
        """Print timer summary and optionally save to JSON.

        Parameters
        ----------
        save_json : str, optional
            Path to save JSON output. If empty, no file is written.
        """
        if self.problem.decomp.rank == 0:
            self.timer.print_summary()
            if save_json:
                with open(save_json, 'w') as f:
                    f.write(self.timer.to_json())

    def pre_run(self, **kwargs) -> None:
        """Initialize solver before running."""
        with self.timer("preparation"):
            self._init_convenience_accessors()
            with self.timer("init_quad_fields"):
                self._init_quad_fields()
            self._get_active_terms()
            with self.timer("build_jit_functions"):
                self._build_jit_functions()
            self._build_terms()
            with self.timer("init_petsc"):
                self._init_linear_solver()

            # Initial quad update
            with self.timer("initial_update_quad"):
                self.update_quad()

            self.update_prev_quad()

            # Update output fields for initial frame
            self.update_output_fields()

        self.time_inner = 0.0
        self.inner_iterations = 0

    def _init_linear_solver(self):
        """Initialize linear solver for sparse system solves."""
        solver_type = self.problem.fem_solver.get('linear_solver', 'direct')
        petsc_info = self.assembly_layout.get_petsc_info()

        if HAS_PETSC:
            from .fem_2d.petsc_system import PETScSystem
            self.linear_solver = PETScSystem(petsc_info, solver_type=solver_type)
        else:
            from mpi4py import MPI
            if MPI.COMM_WORLD.size > 1:
                raise RuntimeError(
                    "PETSc is required for parallel execution. "
                    "Install petsc4py or run in serial mode."
                )
            from .fem_2d.scipy_system import ScipySystem
            self.linear_solver = ScipySystem(petsc_info, solver_type=solver_type)
            print("Note: Using SciPy sparse solver (PETSc not available). "
                  "Install petsc4py for better performance and parallel support.")
