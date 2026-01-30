# qec_analysis.py
import stim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import itertools
from collections import defaultdict
import os
from scipy.sparse import coo_matrix
import osqp
from scipy import sparse
# -----------------------------
# helper functions for A-matrix
# -----------------------------
def make_one_body_row_from_stim(i1, g_index, p_i, p_ij, p_ijk, p_ijkl):
    cols, vals = [], []

    # singles
    if (i1,) in g_index:
        cols.append(g_index[(i1,)]) 
        vals.append(1.0)

    # pairs
    for (a, b) in p_ij:
        if i1 in (a, b):
            cols.append(g_index[(a, b)])
            vals.append(1.0)

    # triples
    for (a, b, c) in p_ijk:
        if i1 in (a, b, c):
            cols.append(g_index[(a, b, c)])
            vals.append(1.0)

    # quads
    for (a, b, c, d) in p_ijkl:
        if i1 in (a, b, c, d):
            cols.append(g_index[(a, b, c, d)])
            vals.append(1.0)

    return cols, vals

def make_two_body_row_from_stim(i1, i2, g_index, p_i, p_ij, p_ijk, p_ijkl):
    cols, vals = [], []

    # singles
    if (i1,) in g_index:
        cols.append(g_index[(i1,)])
        vals.append(1.0)
    if (i2,) in g_index:
        cols.append(g_index[(i2,)])
        vals.append(1.0)

    # pairs
    for (a, b) in p_ij:
        if i1 in (a, b) and i2 not in (a, b):
            cols.append(g_index[(a, b)])
            vals.append(1.0)
        if i2 in (a, b) and i1 not in (a, b):
            cols.append(g_index[(a, b)])
            vals.append(1.0)

    # triples
    for (a, b, c) in p_ijk:
        if i1 in (a, b, c) and i2 not in (a, b, c):
            cols.append(g_index[(a, b, c)])
            vals.append(1.0)
        if i2 in (a, b, c) and i1 not in (a, b, c):
            cols.append(g_index[(a, b, c)])
            vals.append(1.0)

    # quads
    for (a, b, c, d) in p_ijkl:
        if i1 in (a, b, c, d) and i2 not in (a, b, c, d):
            cols.append(g_index[(a, b, c, d)])
            vals.append(1.0)
        if i2 in (a, b, c, d) and i1 not in (a, b, c, d):
            cols.append(g_index[(a, b, c, d)])
            vals.append(1.0)

    return cols, vals


def make_three_body_row_from_stim(i1, i2, i3, g_index, p_i, p_ij, p_ijk, p_ijkl):
    cols, vals = [], []

    # singles
    for idx in (i1, i2, i3):
        if (idx,) in g_index:
            cols.append(g_index[(idx,)])
            vals.append(1.0)

    # triple term itself
    key = tuple(sorted((i1, i2, i3)))
    if key in g_index:
        cols.append(g_index[key])
        vals.append(1.0)

    # pairs
    for (a, b) in p_ij:
        for idx in (i1, i2, i3):
            if idx in (a, b) and all(x not in (a, b) for x in (i1, i2, i3) if x != idx):
                cols.append(g_index[(a, b)])
                vals.append(1.0)

    # triples of form (i?, j, k) with j,k outside
    for (a, b, c) in p_ijk:
        for idx in (i1, i2, i3):
            if idx in (a, b, c) and all(x not in (a, b, c) for x in (i1, i2, i3) if x != idx):
                cols.append(g_index[(a, b, c)])
                vals.append(1.0)

    # quads of form (i1,i2,i3,l)
    for (a, b, c, d) in p_ijkl:
        if set((i1, i2, i3)).issubset((a, b, c, d)):
            cols.append(g_index[(a, b, c, d)])
            vals.append(1.0)

    # quads of form (i?, j,k,l) with j,k,l outside
    for (a, b, c, d) in p_ijkl:
        for idx in (i1, i2, i3):
            if idx in (a, b, c, d) and all(x not in (a, b, c, d) for x in (i1, i2, i3) if x != idx):
                cols.append(g_index[(a, b, c, d)])
                vals.append(1.0)

    return cols, vals
def make_four_body_row_from_stim(i1, i2, i3, i4, g_index, p_i, p_ij, p_ijk, p_ijkl):
    """
    Build four-body equation row for detectors (i1,i2,i3,i4).
    Equation:
        s_{i1 i2 i3 i4} =
            g_{i1}+g_{i2}+g_{i3}+g_{i4}
          + sum_j (g_{i1 j}+g_{i2 j}+g_{i3 j}+g_{i4 j})
          + (g_{i1 i2 i3}+g_{i1 i2 i4}+g_{i1 i3 i4}+g_{i2 i3 i4})
          + sum_{j<k} (g_{i1 j k}+g_{i2 j k}+g_{i3 j k}+g_{i4 j k})
          + sum_l (g_{i1 i2 i3 l}+g_{i1 i2 i4 l}+g_{i1 i3 i4 l}+g_{i2 i3 i4 l})
          + sum_{j<k<l} (g_{i1 j k l}+g_{i2 j k l}+g_{i3 j k l}+g_{i4 j k l})
    """
    cols, vals = [], []
    I = (i1, i2, i3, i4)

    # singles
    for idx in I:
        if (idx,) in g_index:
            cols.append(g_index[(idx,)])
            vals.append(1.0)

    # pairs with one inside, one outside
    for (a, b) in p_ij:
        for idx in I:
            if idx in (a, b) and all(x not in (a, b) for x in I if x != idx):
                cols.append(g_index[(a, b)])
                vals.append(1.0)

    # triples: all-inside special ones
    triples_inside = [
        tuple(sorted((i1, i2, i3))),
        tuple(sorted((i1, i2, i4))),
        tuple(sorted((i1, i3, i4))),
        tuple(sorted((i2, i3, i4))),
    ]
    for key in triples_inside:
        if key in g_index:
            cols.append(g_index[key])
            vals.append(1.0)

    # triples anchored: (i?, j, k) with j,k outside
    for (a, b, c) in p_ijk:
        for idx in I:
            if idx in (a, b, c) and all(x not in (a, b, c) for x in I if x != idx):
                cols.append(g_index[(a, b, c)])
                vals.append(1.0)

    # quads: special ones with 3 inside + 1 outside
    for (a, b, c, d) in p_ijkl:
        for triple in triples_inside:
            if set(triple).issubset((a, b, c, d)):
                cols.append(g_index[(a, b, c, d)])
                vals.append(1.0)

    # quads anchored: (i?, j,k,l) with j,k,l outside
    for (a, b, c, d) in p_ijkl:
        for idx in I:
            if idx in (a, b, c, d) and all(x not in (a, b, c, d) for x in I if x != idx):
                cols.append(g_index[(a, b, c, d)])
                vals.append(1.0)

    return cols, vals

def build_equation_system(spins, g_index, p_i, p_ij, p_ijk, p_ijkl):
    """
    spins : array of shape (n_shots, n_det)
        ±1 values for each shot and detector
    g_index : dict
        mapping g-term tuple -> column index
    p_i, p_ij, p_ijk, p_ijkl : dicts
        keys define which correlators exist (from DEM)
    """
    n_shots, n_det = spins.shape

    row_idx, col_idx, data, rhs = [], [], [], []
    eq = 0

    # --- one-body equations
    for (i,) in p_i:
        cols, vals = make_one_body_row_from_stim(i, g_index, p_i, p_ij, p_ijk, p_ijkl)
        row_idx.extend([eq] * len(cols))
        col_idx.extend(cols)
        data.extend(vals)

        rhs_val = np.mean(spins[:, i])       # ⟨σ_i⟩
        rhs_val = np.log(rhs_val)
        rhs.append(rhs_val)
        eq += 1

    # --- two-body equations
    for (i1, i2) in p_ij:
        cols, vals = make_two_body_row_from_stim(i1, i2, g_index, p_i, p_ij, p_ijk, p_ijkl)
        row_idx.extend([eq] * len(cols))
        col_idx.extend(cols)
        data.extend(vals)

        rhs_val = np.mean(spins[:, i1] * spins[:, i2])   # ⟨σ_i σ_j⟩
        rhs_val = np.log(rhs_val)
        rhs.append(rhs_val)
        eq += 1

    # --- three-body equations
    for (i1, i2, i3) in p_ijk:
        cols, vals = make_three_body_row_from_stim(i1, i2, i3, g_index, p_i, p_ij, p_ijk, p_ijkl)
        row_idx.extend([eq] * len(cols))
        col_idx.extend(cols)
        data.extend(vals)

        rhs_val = np.mean(spins[:, i1] * spins[:, i2] * spins[:, i3])  # ⟨σ_i σ_j σ_k⟩
        rhs_val = np.log(rhs_val)
        rhs.append(rhs_val)
        eq += 1

    # --- four-body equations
    for (i1, i2, i3, i4) in p_ijkl:
        cols, vals = make_four_body_row_from_stim(i1, i2, i3, i4, g_index, p_i, p_ij, p_ijk, p_ijkl)
        row_idx.extend([eq] * len(cols))
        col_idx.extend(cols)
        data.extend(vals)

        rhs_val = np.mean(spins[:, i1] * spins[:, i2] * spins[:, i3] * spins[:, i4])  # ⟨σ_i σ_j σ_k σ_l⟩
        rhs_val = np.log(rhs_val)
        rhs.append(rhs_val)
        eq += 1

    # --- assemble sparse system
    A = coo_matrix((data, (row_idx, col_idx)), shape=(eq, len(g_index)))
    S = np.array(rhs)

    return A, S

# -----------------------------
# main workflow
# -----------------------------
def process_file(file_folder, out_file="p_ij.png", distance=3):
    # load circuit + data
    c = stim.Circuit.from_file(file_folder + 'circuit_noisy.stim')
    dem = c.detector_error_model()
    N = c.num_detectors
    M = c.num_observables    # or infer from the obs file below
    # load files -> unpacked bool arrays [shots, bits]
    dets = stim.read_shot_data_file(
        path=file_folder + 'detection_events.b8', format="b8", bit_packed=False,
        num_detectors=N, num_measurements=0, num_observables=0)

    # obs  = stim.read_shot_data_file(
    # path=file_folder + 'obs_flips_actual.b8', format="b8", bit_packed=False,
    # num_observables=M)

    # preprocess detectors (ordering)
    coordinates = c.get_detector_coordinates()
    detector_coordinates = np.array([v[-3:] for v in coordinates.values()])
    detector_coordinates -= detector_coordinates.min(axis=0)
    num_shift = (distance**2 - 1) // 2
    detector_coordinates[num_shift:, 2] += 1
    detector_coordinates_dict = {i: row for i, row in enumerate(detector_coordinates)}

    groups = defaultdict(list)
    for idx, (x, y, t) in detector_coordinates_dict.items():
        groups[(x, y)].append((t, idx))
    for xy in groups:
        groups[xy].sort()
    spatial_keys = sorted(groups.keys(), key=lambda xy: (groups[xy][0][0], groups[xy][0][1]))
    order_space_first = [idx for xy in spatial_keys for (t, idx) in groups[xy]]

    # keep odd shots
    x = dets[::2, :].astype(np.float64)
    x = x[:, order_space_first]
    sigmas = 1.0 - 2.0 * x
    shots = sigmas.shape[0]

    p_i, p_ij, p_ijk, p_ijkl = {}, {}, {}, {}

    for error in dem:
        if error.type == 'error':
            p = error.args_copy()[0]
            targets = [t.val for t in error.targets_copy() if t.is_relative_detector_id()]
            key = tuple(sorted(targets))

            if len(key) == 1:
                p_i[key] = p
            elif len(key) == 2:
                p_ij[key] = p
            elif len(key) == 3:
                p_ijk[key] = p
            elif len(key) == 4:
                p_ijkl[key] = p
    g_index = {}
    # singles
    for key in p_i:
        g_index[key] = len(g_index)
    # pairs
    for key in p_ij:
        g_index[key] = len(g_index)
    # triples
    for key in p_ijk:
        g_index[key] = len(g_index)
    # quads
    for key in p_ijkl:
        g_index[key] = len(g_index)
    inv_g_index = {col: key for key, col in g_index.items()}

    A, S = build_equation_system(sigmas, g_index, p_i, p_ij, p_ijk, p_ijkl)

    P = (A.T @ A).tocsc()   # convert to CSC explicitly
    q = -(A.T @ S)

    # Constraint matrix: here just bounds on g, so use identity in CSC
    G = sparse.eye(len(q), format="csc")
    l = -10 * np.ones(len(q))
    u = 0   * np.ones(len(q))

    prob = osqp.OSQP()
    prob.setup(P=P, q=q, A=G, l=l, u=u, verbose=True)
    res = prob.solve()
    g_sol = res.x
    p = (1-np.exp(g_sol)) / 2
    p_ij_matrix = np.zeros((N, N))
    for i1, i2 in itertools.combinations(range(N), 2):
        p_ij_matrix[i1, i2] = p[g_index[(i1, i2)]]
        p_ij_matrix[i2, i1] = p[g_index[(i1, i2)]]

    # plot
    vmax = np.max(p_ij_matrix)
    cmap = mcolors.LinearSegmentedColormap.from_list("", ["white", "red"])
    fig, ax = plt.subplots(figsize=(3, 3))
    im = ax.imshow(p_ij_matrix, cmap=cmap, origin="lower", vmin=0, vmax=vmax)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved heatmap to {out_file}")

# -----------------------------
# example usage
# -----------------------------
if __name__ == "__main__":
    folder = "/Users/xlmori/Desktop/QEC_GNN-RNN/google_105Q_surface_code_d3_d5_d7/google_qec3v5_experiment_data/surface_code_bZ_d5_r03_center_5_5/"
    # normalize and split path
    parts = os.path.normpath(folder).split(os.sep)
    # take the last 3 pieces and join with underscores
    name = "_".join(parts[-3:])
    out_file = f"figures/{name}_pij_plot.png"
    process_file(folder, out_file=out_file)
