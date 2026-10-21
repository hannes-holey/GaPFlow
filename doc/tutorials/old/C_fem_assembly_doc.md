# Grid and Domain Decomposition

In the following, we distinguish between `density` and `mass flux` fields. Note that density nodes correspond to `mass balance` equations and mass flux nodes to `momentum equations`.

In contrast to the P1P1 approach, the mass flux field $\underline{j}$ is now approximated by second-order triangular elements, effectively leading to a finer mesh. That is, for $n_\rho$ discretization points along one axis, we have $n_j = 2*n_\rho - 1$ nodes.

For local (process-specific) indexing, we agree that both density and mass flux have the indices $[0,0]$ at the first **inner** node. The ghost nodes left/below get negative indices - more on that later.

That gives us a nice property for easily finding nodes where both density and mass flux live. Namely, on even-even points. Indices $[2n, 2m]$ of the mass flux field correspond to indices $[n, m]$ of the density field. These nodes are also always corners of the squares that span two triangles each. Note that this is not a construction rule for the mass flux field. For that, see below.

Also, we know that:
- odd-odd indices refer to mass flux nodes that live in the middle / on the diagonal of the squares.
- even-odd indices refer to mass flux nodes on the vertical line between two square nodes
- odd-even indices refer to mass flux nodes on the horizontal line between two square nodes

By having these rules, we can systematically determine all possible contributions / non-zeros in the tangential matrix. For each of these four combinations, we can formulate a stencil (see `elements.py` `stencil_*` attributes) and collect all contributions that the nodes selected by the stencil can have on the origin point. This is the same methodology as before, just with four different stencils for four different types of nodes. And as these types are exclusive, we do not obtain redundant entries.

For the finer mass flux mesh, we need an additional `DomainDecomposition` object with higher mesh resolution. Here, it gets a little bit awkward, because we cannot distribute evenly. More precisely, let's assume a global domain with 10 x 10 nodes and four processes. Regarding the density field, each process holds a 5 x 5 **nodes** inner subdomain (note that this corresponds to a 4 x 4 **squares** inner subdomain) with ghost cell depth of 1, resulting in a 7 x 7 nodes padded subdomain. Here, ghost cell exchange of depth 1 is taken in all directions.

For the mass flux mesh, we have, as stated in the beginning, $n_j = 2*n_\rho - 1$ nodes, meaning we have $2*10 - 1 = 19$ nodes along each axis. However, if we look at one subdomain and fill in mass flux nodes on and between the density nodes, we obtain $n_j = 9$. This is because the missing mass flux node lives **between** the density node subdomains. Therefore we set the following rule: At the interface between two adjacent subdomains, the interface mass flux nodes will live on the west and south subdomains.

For our example, this means that our `SW` subdomain holds 10 x 10 mass flux nodes (14 x 14 on padded subdomain), `SE` subdomain holds 9 x 10 mass flux nodes (13 x 14 on padded subdomain), `NW` subdomain holds 10 x 9 mass flux nodes, and `NE` subdomain holds 9 x 9 mass flux nodes. Very important: this is non-periodic. In the fully periodic case, each subdomain would hold 10 x 10 mass flux nodes (all extended on the `E` and `N` side).

As already hinted at, the mass flux domain must possess a ghost cell depth of two in order to capture all contributions from nodes living on other processes. This has implications on the boundary condition (BC) treatment. But as far as I can see, we can apply the same methodology as in the earlier approach. Note: The ghost nodes implicitly enforce the boundary conditions. Residuals are **not** evaluated on the ghost nodes, they just **contribute** to the `inner` node residuals. This has the advantage, that we do not need to delete and overwrite rows in our equations system. And this has a special implication for the Neumann boundary condition, because the ghost node values `orient` themselves (value-wise) at the last inner node. Therefore, we need to catch in the tangential matrix the fact that with variation of the inner node, also the Neumann ghost nodes vary (Neumann index forwarding).

- Dirichlet: set all nodes with the distance $h_j$ and $2*h_j$ to the Dirichlet value. Both ghost layers are set to the Dirichlet value in `_apply_mass_flux_bcs`. This does not impose an additional Neumann condition — confirmed in Phase 1 testing.
- Neumann: Nothing changes, just that we need to calculate the Neumann values for both $h_j, 2*h_j$ ghost nodes. Also, both nodes need to be reference by the adjacent inner node in the tangential matrix (Neumann index forwarding).

The height field will continue to live on the density grid. It will also stay the 'basis' for writing the solutions fields.

No decision has yet been made about the energy discretization. I will probably try to use the density grid for it. All interactions / implications should be deducable from the density - mass flux interaction that is described here.

Additional note on the subdomain interface node: The interface mass flux node lives as  an inner node on the owning subdomain; the adjacent subdomain receives it through the standard ghost exchange. The outermost ghost layer on the owning side contains mass flux nodes that fall outside any valid square origin and are simply never accessed. This is in contrast to the non-owning side: This side receives two valid ghost layers. But this is an implicit consequence of how the quadrature / convolution is applied to the mass flux field and must not be implemented explicitly.

## Field Indexing and Neumann Index Forwarding

The following precomputed index arrays are the heart of the indexing:

- `index_mask_inner_local`
- `index_mask_padded_local`
- `index_mask_padded_global`

The first two use local indices starting from 0 on each process; global indices are unique across all processes. Both masks use -1 as sentinel (no DOF at this node).

`index_mask_inner_local` assigns a local index to every inner node. These are the nodes that own a residual equation. Ghost nodes are -1 — they do not own a residual, but their values are used when evaluating the residuals of adjacent inner nodes.

`index_mask_padded_local` extends the inner mask to ghost nodes that represent a **free DOF** — i.e. nodes that can appear as a column in the tangential matrix. There are exactly four ghost node cases:

- **Inter-subdomain ghost nodes** (MPI): assigned new sequential local indices on this process. Their global index is obtained from `index_mask_padded_global` (populated by muGrid via coordinate ghost exchange), which holds the correct global index of the corresponding inner node on the adjacent process.
- **Neumann ghost nodes**: receive the same local index as the inner node they depend on. Tangential matrix contributions attributed to a Neumann ghost node automatically accumulate onto the correct DOF without special handling.
- **Dirichlet ghost nodes**: stay -1. Their values are prescribed, so they are not free DOFs and do not appear as columns in the tangential matrix.
- **Serial periodic ghost nodes**: receive the local index of the inner node on the opposite side of the domain. This is handled by direct mask assignment before the general ghost index loop, not by the inter-subdomain path.

The additional local indices assigned to inter-subdomain ghosts are contiguous in arbitrary order; only their mapping to global indices matters. The information for the padded global indices comes naturally from the `DomainDecomposition`.

# Quadrature

Quadrature is straightforward and quadrature points are the same for both density and mass flux fields. For the mass flux, this has the implication that convolution is a little bit more tricky. Namely, the stencil is larger and the evaluation of the convolution is only needed at even-even points (because these points represent squares and the associated triangles). That is why - for now - the convolution is done with a numpy-based approach instead of the muGrid convolution. But the original structure is kept so that we could later go back to the muGrid approach.

Note that I changed the output ordering of the quadrature value arrays computed from the nodal values. I will explain the change later in `Assembly`.

# Data Flow Framework

**Important:** The stencils are used exclusively to precompute the nnz sparsity pattern (the coordinate lists of non-zero entries in the global matrix). The assembly loop itself operates per element (square) and never references the stencils directly.

Note: `block` refers to a block in the tangential matrix defined by the dependency of one type of residual on one type of field variable.

In the original approach, we created an `nnz` list for one `block` where all possible contributions were recorded. Because density and mass flux had the exact same discretization, we could use the same `nnz-coordinate` list for all blocks $\rho \rightarrow R_\rho$, $\rho \rightarrow R_{j_x}$, $\rho \rightarrow R_{j_y}$, $j_x \rightarrow R_\rho$ and so on ($i \rightarrow j$ meaning $\frac{\partial R_j}{\partial \phi^h_i}$). A simple index shift of the number of `nnz` per block sufficed.

Now however, we have different blocks. I want to resort to compiling a contiguous `nnz-coordinate` list that comprises all blocks. So the core of the data flow will be the 1D `nnz` value array. Theoretically, we could decompose that into individual `nnz` value arrays for each block and maybe reuse it for similar blocks such as $\rho \rightarrow R_{j_x}$, $\rho \rightarrow R_{j_y}$, but that may be an optimization for later.

So that `nnz` value list will be the heart of the solver framework. We build it using the given stencils. As a pre-step, we need to analyze the stencils regarding the density-velocity, velocity-density, and density-density connectivity. So for each given stencil, we derived four sub-stencils. That means, for each point in the stencil, check if the point is a density node (even-even). Thereby, we can derive the different connectivities for the blocks. Similar to the original approach we want to collect coordinate lists for the translation of:

- `nnz index` from/to (field indices) - we need to compile this in both directions because we later need to pre-compile `nnz_indices` lists for efficient, vectorized assembly
- `nnz index` to global matrix coordinates

Note that each `nnz` value represents the contribution of one node of field $\phi^h_i$ to the residual on the node of another field (can be the same field) $R_j$. When we know which fields we are talking about, those two indices fully describe the contribution.

# Global Matrix

We need a new rule for the global point-interleaved ordering so that all sub-processes can determine the correct coordinates given the global field indices provided by `DomainDecomposition`.

For that, we should use the mass flux field indices. On each mass flux node we have two directions, and dependent on energy activation, we have one or two entries on the density nodes.

Challenge is for given mass flux field index, to know how many density field indices that correlates to.

residual indices: $j_x=0, j_y=1, \rho=2, e=3$

For given index:
- derive i,j (dependent on density/velocity)
- if $\rho, e$ then convert i,j to velocity i,j

For given mass flux field index
- get count of nodes until the previous index
    - get number of covered density nodes (multiply by 2 if energy enabled)
    - add 2 * (mass flux index) (note: not -1 because e.g. on index 1, we have nodes from index 0)
- add residual index -> done

The count formula: for mass flux index k with r = k // cols_v, c = k % cols_v, the number of density nodes emitted before block k is `((r+1)//2) * M + ((c+1)//2 if r%2==0 else 0)` — i.e. ceil(r/2) complete even rows each contributing M nodes, plus the count of even columns strictly before c in the current row (if r is even). Note: the earlier formula `(k // cols_v) * M + min(k % cols_v // 2 + 1, M)` was incorrect (wrong row factor, off-by-one in column count). Corrected and verified in Phase 3 testing.

# Assembly

The assembly is restructured to be more memory-efficient, while keeping performance on a similar level.

First of all, I want to keep the term-based definition of the residuals. In the original approach, we looped through all terms and for each term, looped through its dependent field variables. So we did the assembly for individual term-dependent variable combination.

As a first optimization, I would like to pre-collect (residual-dependent variable) `res-dep_var` combination, meaning we may have several `term-dep_var` combinations within one residual-dep_var slot. This could be collected in a dict so that for given res-dep_var combination, we have - by reference - quick access to all associated terms that have this dependent variable. Thereby, we can reduce loops.

Now, for each `res-dep_var` (similar to what we called `block` in the previous sections) I want to precompute the following:

- `shape-weighting`
- `nnz`-index

## Shape-Weighting

To understand the idea better, let's picture the actually vectorized assembly as an iterative process. We wanna iterate through the squares, since they are the smallest repeating and regular units. For each square, we iterate through the quad points. And for each quad point, we look at all node-node interactions. Let's denote the (block-specific) number of interactions $n_{contr}$. For density-density $n_{contr} = 9$, for density-velocity and velocity-density $n_{contr} = 18$, and for velocity-velocity $n_{contr} = 36$. So for each square we obtain $n_{tri} \cdot n_{quad} \cdot n_{contr}$ contribution values (where $n_{quad}$ specifies quadrature points per triangle). The main point is now that for each square, the required `shape-weighting` of these $n_{tri} \cdot n_{quad} \cdot n_{contr}$ contributions is the same!

The weighting is a mixture of the shape function evaluations on the quad points from both the residual / test function and the field variable / trial function.

For vectorization we do the following. We have our quadrature values in the shape $(n_{sq,x}, n_{sq,y}, 2 * n_{quad})$. We flatten this array and repeat each entry $n_{contr}$ times. The precomputed `shape-weighting` array has the length $2 * n_{quad} \cdot n_{contr}$ (corresponding to the length of the repeated quadrature values for **one square**) and is repeated in full for $n_{sq,x} \cdot n_{sq,y}$ times.

What results from this element-wise multiplication is a 1D array `quad-specific` contribution values. Since we repeat for each quadrature point in one element the same node-node-contribution pattern, we can fold these three contributions together:

```python
folded = contribs.reshape(-1, n_quad_per_tri, n_contr).sum(axis=1).reshape(-1)
```

## NNZ List

These shortened `element-specific` contribution values now need to be injected in the `nnz` array. Here comes the second pre-computed array into play, the `nnz`-index list. This one cannot be applied square-wise or modular, since it contains all the geometric indexing quirks. Its length is $n_{sq,x} \cdot n_{sq,y} \cdot n_{tri} \cdot n_{contr}$ and it is specific to each `res-dep_var` / `block`.

## Limitations

But now come some limitations. First, the `shape-weighting` is dependent on the derivative combination of the term. It may either act on the trial function derivative, the test function derivative, or both. But since the precomputed `shape-weighting` arrays have minimal memory cost, this is just a task of cleanly implementing the pre-compute.

The more challenging limitation is the boundary. We actually need to distinguish eight different boundary sections: N,E,S,W,NE,SE,SW,NW, where the last four represent the corners. The only reason is that

- for Dirichlet BC, some node-node contributions need to be set to zero

Note that for Neumann BC, every point still contributes. The Neumann index forwarding is applied by the function that creates the local, padded subdomain grid points and the association to the `nnz` array is done by the `nnz`-index list. And for inter-subdomain boundaries or periodic boundaries, all nodes contribute.

So if the dependent variable has a Dirichlet BC, we need to compile separate `shape-weighting` arrays for nine different cases. From the whole subdomain, we need to extract the respective parts in the following way:

```python
#shape: (sq_x, sq_y, 2 * n_quad)

inner = data[1:-1, 1:-1, :]   

north = data[0, 1:-1, :]
south = data[-1, 1:-1, :]
west  = data[1:-1, 0, :]
east  = data[1:-1, -1, :]

nw = data[0, 0, :]
ne = data[0, -1, :]
sw = data[-1, 0, :]
se = data[-1, -1, :]
```

For that, we need to define a rule on the ordering of these parts so that the (Dirichlet-specific) `nnz`-index list is in accordance to that

And in the other cases we should be fine with applying the single `shape-weighting` on everything.

## Residual

Since residual assembly is simpler (no trial function, no node-node interaction), a targeted einsum over the output field directly might be sufficient and more readable. Details to be figured.
