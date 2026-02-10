"""
Test that make_mpp_circuit produces equivalent detectors and observable 0
as the standard circuit, by:
  1. Explicit DEM probability comparison (raw + XOR-aggregated)
  2. Random error vector through matched DEM matrices
  3. Old method (error-chain from data.py) vs MPP intermediate logicals
  4. Statistical comparison of decoded logical error rates
"""

import stim
import numpy as np
import pymatching


def make_standard_circuit(distance, rounds, error_rate):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=error_rate,
        after_reset_flip_probability=error_rate,
        before_measure_flip_probability=error_rate,
        before_round_data_depolarization=error_rate,
    )


def make_mpp_circuit(distance, rounds, error_rate):
    std = make_standard_circuit(distance, rounds, error_rate)

    qubit_coords = {}
    for ins in std.flattened():
        if ins.name == "QUBIT_COORDS":
            args = ins.gate_args_copy()
            for t in ins.targets_copy():
                qubit_coords[t.value] = tuple(args)

    data_qubits = []
    for ins in std.flattened():
        if ins.name == "M":
            data_qubits = [t.value for t in ins.targets_copy()]

    if qubit_coords and data_qubits:
        data_coords = {q: qubit_coords[q] for q in data_qubits if q in qubit_coords}
        min_y = min(y for _, y in data_coords.values())
        logical_z_qubits = sorted(
            [q for q, (_, y) in data_coords.items() if y == min_y]
        )
    else:
        logical_z_qubits = [1, 3, 5]

    n_anc = distance**2 - 1

    def shift_rec(x, std_count, mpp_count):
        std_pos = std_count + x
        num_mpps_before = min(std_pos // n_anc, mpp_count)
        return x + num_mpps_before - mpp_count

    modified = stim.Circuit()
    std_count = 0
    mpp_count = 0
    obs_idx = 1

    for ins in std.flattened():
        name = ins.name

        if name == "DETECTOR":
            targets = ins.targets_copy()
            args = ins.gate_args_copy()
            shifted = [
                stim.target_rec(shift_rec(t.value, std_count, mpp_count))
                for t in targets
            ]
            modified.append("DETECTOR", shifted, args)
        elif name == "OBSERVABLE_INCLUDE":
            targets = ins.targets_copy()
            args = ins.gate_args_copy()
            obs_id = int(args[0]) if args else 0
            shifted = [
                stim.target_rec(shift_rec(t.value, std_count, mpp_count))
                for t in targets
            ]
            modified.append("OBSERVABLE_INCLUDE", shifted, obs_id)
        else:
            modified.append(ins)

        if name in ["M", "MR"]:
            std_count += len(ins.targets_copy())

        if name == "MR":
            mpp_targets = []
            for i, q in enumerate(logical_z_qubits):
                mpp_targets.append(stim.target_z(q))
                if i < len(logical_z_qubits) - 1:
                    mpp_targets.append(stim.target_combiner())
            modified.append("MPP", mpp_targets)
            modified.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], obs_idx)
            obs_idx += 1
            mpp_count += 1

    return modified


# ─── helpers ──────────────────────────────────────────────────────────


def dem_to_mechanisms(dem):
    """Return list of (probability, frozenset_of_detectors, obs0_flip_bool)."""
    result = []
    for ins in dem.flattened():
        if ins.type == "error":
            p = ins.args_copy()[0]
            det_set = set()
            has_obs0 = False
            for t in ins.targets_copy():
                if t.is_relative_detector_id():
                    det_set.add(t.val)
                elif t.is_logical_observable_id() and t.val == 0:
                    has_obs0 = True
            result.append((p, frozenset(det_set), has_obs0))
    return result


def xor_prob(p1, p2):
    """Combine two independent error probabilities via XOR: P(exactly one fires)."""
    return p1 * (1 - p2) + p2 * (1 - p1)


def aggregate_by_signature(mechanisms):
    """Group mechanisms by (det_pattern, obs0_flip). Combine probs via XOR."""
    agg = {}
    for p, det_set, has_obs0 in mechanisms:
        key = (det_set, has_obs0)
        if key not in agg:
            agg[key] = p
        else:
            agg[key] = xor_prob(agg[key], p)
    return agg


# ─── Test 1: Explicit DEM probability comparison ─────────────────────


def test_dem_probabilities():
    print("=" * 64)
    print("TEST 1  –  Explicit DEM probability comparison")
    print("  Raw mechanism counts + XOR-aggregated probability match")
    print("=" * 64)

    all_ok = True
    for d in [3, 5]:
        for r in [1, 3, 5, 10]:
            p = 0.005

            std_circuit = make_standard_circuit(d, r, p)
            mpp_circuit = make_mpp_circuit(d, r, p)

            std_dem = std_circuit.detector_error_model()
            mpp_dem = mpp_circuit.detector_error_model(allow_gauge_detectors=True)

            std_mechs = dem_to_mechanisms(std_dem)
            mpp_mechs = dem_to_mechanisms(mpp_dem)

            std_agg = aggregate_by_signature(std_mechs)
            mpp_agg = aggregate_by_signature(mpp_mechs)

            std_keys = set(std_agg)
            mpp_keys = set(mpp_agg)
            shared = std_keys & mpp_keys
            only_std = std_keys - mpp_keys
            only_mpp = mpp_keys - std_keys

            max_diff = 0.0
            diffs = []
            for k in shared:
                diff = abs(std_agg[k] - mpp_agg[k])
                diffs.append((diff, std_agg[k], mpp_agg[k], k))
                max_diff = max(max_diff, diff)

            keys_ok = len(only_std) == 0 and len(only_mpp) == 0
            prob_ok = max_diff < 1e-6
            ok = keys_ok and prob_ok
            tag = "OK" if ok else "FAIL"

            print(
                f"\n  d={d} r={r:>2}: raw_mechs std={len(std_mechs)} mpp={len(mpp_mechs)}  "
                f"unique_sigs std={len(std_keys)} mpp={len(mpp_keys)}  [{tag}]"
            )
            print(
                f"         shared={len(shared)}  "
                f"only_std={len(only_std)}  only_mpp={len(only_mpp)}  "
                f"max_Δp={max_diff:.1e}"
            )

            # Show 5 largest-probability mechanisms to eyeball
            diffs.sort(key=lambda x: -x[1])  # sort by std probability descending
            print(f"         Top 5 mechanisms by probability:")
            for i, (diff, p_std, p_mpp, k) in enumerate(diffs[:5]):
                n_det = len(k[0])
                print(
                    f"           [{i}] p_std={p_std:.6e}  p_mpp={p_mpp:.6e}  "
                    f"Δ={diff:.1e}  n_det={n_det}  obs0={k[1]}"
                )

            # Show worst-matching mechanisms if any differ
            diffs.sort(key=lambda x: -x[0])  # sort by difference descending
            if diffs[0][0] > 1e-15:
                print(f"         Worst-matching mechanisms:")
                for i, (diff, p_std, p_mpp, k) in enumerate(diffs[:3]):
                    print(
                        f"           p_std={p_std:.6e}  p_mpp={p_mpp:.6e}  "
                        f"Δ={diff:.1e}  det={sorted(k[0])[:5]}...  obs0={k[1]}"
                    )

            all_ok &= ok

    print(f"\n  {'ALL PASSED' if all_ok else 'SOME FAILED'}\n")
    return all_ok


# ─── Test 2: Random error vector through matched DEM matrices ─────────


def test_random_error_vector():
    """
    Build a unified parity-check matrix from the aggregated mechanisms.
    Draw random error vectors and check that syndromes + obs0 match.
    """
    print("=" * 64)
    print("TEST 2  –  Random error vector through aggregated DEM matrices")
    print("  Build H, L from aggregated (signature, prob) pairs; apply same e")
    print("=" * 64)

    def signature_to_matrices(agg, n_det):
        """Convert aggregated {(det_set, obs0): prob} to H, L matrices."""
        keys = sorted(agg.keys(), key=lambda k: (sorted(k[0]), k[1]))
        n_mech = len(keys)
        H = np.zeros((n_det, n_mech), dtype=np.uint8)
        L = np.zeros(n_mech, dtype=np.uint8)
        probs = np.zeros(n_mech)
        for j, (det_set, has_obs0) in enumerate(keys):
            for di in det_set:
                H[di, j] = 1
            L[j] = int(has_obs0)
            probs[j] = agg[(det_set, has_obs0)]
        return H, L, probs, keys

    all_ok = True
    rng = np.random.default_rng(42)

    for d in [3, 5]:
        for r in [1, 3, 5, 100]:
            p = 0.005

            std_circuit = make_standard_circuit(d, r, p)
            mpp_circuit = make_mpp_circuit(d, r, p)

            std_dem = std_circuit.detector_error_model()
            mpp_dem = mpp_circuit.detector_error_model(allow_gauge_detectors=True)

            std_agg = aggregate_by_signature(dem_to_mechanisms(std_dem))
            mpp_agg = aggregate_by_signature(dem_to_mechanisms(mpp_dem))

            n_det = std_dem.num_detectors

            # Both should have the same keys after test 1
            std_keys = set(std_agg)
            mpp_keys = set(mpp_agg)
            if std_keys != mpp_keys:
                print(f"  d={d} r={r:>2}: signatures differ — SKIP")
                all_ok = False
                continue

            # Build matrices from the same ordered key list
            H_std, L_std, p_std, keys = signature_to_matrices(std_agg, n_det)
            H_mpp, L_mpp, p_mpp, _ = signature_to_matrices(mpp_agg, n_det)

            # Since keys are the same, H and L should be identical by construction
            assert np.array_equal(H_std, H_mpp), "H matrices should match"
            assert np.array_equal(L_std, L_mpp), "L matrices should match"

            # Draw random error vectors and propagate
            n_trials = 200
            n_mech = len(keys)
            det_mismatch = 0
            obs0_mismatch = 0

            for _ in range(n_trials):
                e = (rng.random(n_mech) < 0.1).astype(np.uint8)

                syn_std = (H_std @ e) % 2
                syn_mpp = (H_mpp @ e) % 2
                obs0_std = (L_std @ e) % 2
                obs0_mpp = (L_mpp @ e) % 2

                if not np.array_equal(syn_std, syn_mpp):
                    det_mismatch += 1
                if obs0_std != obs0_mpp:
                    obs0_mismatch += 1

            ok = det_mismatch == 0 and obs0_mismatch == 0
            tag = "OK" if ok else "FAIL"
            print(
                f"  d={d} r={r:>2}: {n_mech} unique mechanisms, {n_trials} trials  "
                f"det_mismatch={det_mismatch}  obs0_mismatch={obs0_mismatch}  [{tag}]"
            )
            all_ok &= ok

    print(f"\n  {'ALL PASSED' if all_ok else 'SOME FAILED'}\n")
    return all_ok


# ─── Test 3: Old method (error-chain) vs MPP intermediate logicals ────


def old_method_logicals(dem, errors):
    """
    Reproduce data.py's error-chain method:
    For each DEM mechanism that flips L0, assign time = max detector time.
    From sampled error vectors, accumulate parity per time step, running XOR.
    """
    coords = dem.get_detector_coordinates()
    det_times = np.array([v[-1] for v in coords.values()]).astype(int)

    term_dets = []
    term_logs = []
    for ins in dem.flattened():
        if ins.type != "error":
            continue
        ts = ins.targets_copy()
        term_dets.append(
            np.fromiter((t.val for t in ts if t.is_relative_detector_id()), dtype=int)
        )
        term_logs.append(
            np.fromiter((t.val for t in ts if t.is_logical_observable_id()), dtype=int)
        )

    n_terms = len(term_dets)
    L0_mask = np.array([0 in logs for logs in term_logs], dtype=bool)
    has_dets = np.array([d.size > 0 for d in term_dets], dtype=bool)

    time_idx = np.full(n_terms, -1, dtype=int)
    for i, d in enumerate(term_dets):
        if L0_mask[i] and has_dets[i]:
            time_idx[i] = det_times[d].max()

    valid_cols = time_idx >= 0
    T = det_times.max() + 1

    E = errors[:, valid_cols].astype(np.uint8)
    idx = time_idx[valid_cols].astype(np.int32)

    order = np.argsort(idx, kind="stable")
    E_sorted = E[:, order]
    idx_sorted = idx[order]

    seg_starts = np.r_[0, np.flatnonzero(np.diff(idx_sorted)) + 1]
    uniq_idx = idx_sorted[seg_starts]

    counts_grouped = np.add.reduceat(E_sorted, seg_starts, axis=1)
    counts = np.zeros((errors.shape[0], T), dtype=np.uint16)
    counts[:, uniq_idx] = counts_grouped

    parity = counts & 1
    logicals = np.bitwise_xor.accumulate(parity, axis=1).astype(np.int32)
    return logicals, det_times, time_idx, valid_cols


def test_old_vs_mpp():
    """
    Sample from the MPP circuit's DEM with return_errors=True.
    Compute intermediate logicals two ways on the SAME sample:
      - MPP: obs_flips[:, 1:R+1]
      - Old: error-chain method from data.py
    Compare shot-by-shot.
    """
    print("=" * 64)
    print("TEST 3  –  Old method (error-chain) vs MPP intermediate logicals")
    print("  Same DEM sample, shot-by-shot comparison")
    print("=" * 64)

    all_ok = True
    shots = 50_000

    for d in [3, 5]:
        for r in [5, 10, 25]:
            for p in [0.003, 0.007]:
                mpp_circuit = make_mpp_circuit(d, r, p)
                mpp_dem = mpp_circuit.detector_error_model(
                    allow_gauge_detectors=True
                )

                sampler = mpp_dem.compile_sampler(seed=42)
                det_events, obs_flips, errors = sampler.sample(
                    shots, return_errors=True
                )

                # MPP logicals: obs 1..R
                mpp_logicals = obs_flips[:, 1 : r + 1].astype(np.int32)

                # Old method
                old_logicals, det_times, time_idx, valid_cols = old_method_logicals(
                    mpp_dem, errors
                )
                T = old_logicals.shape[1]

                # Sanity: old method final == obs 0
                final_ok = np.array_equal(
                    old_logicals[:, -1], obs_flips[:, 0].astype(np.int32)
                )

                # Find time alignment: MPP obs k -> which old time t?
                # MPP after round k captures errors up through round k.
                # Detector times: figure out the mapping.
                unique_det_times = sorted(set(det_times))

                # For each MPP round k, find best-matching old time column
                alignment = []
                for k in range(r):
                    mpp_col = mpp_logicals[:, k]
                    best_t, best_agree = -1, 0
                    for t in range(T):
                        agree = np.mean(old_logicals[:, t] == mpp_col)
                        if agree > best_agree:
                            best_agree = agree
                            best_t = t
                    alignment.append((k + 1, best_t, best_agree))

                # Compute per-round disagreement at best alignment
                n_disagree_total = 0
                per_round = []
                for k, (obs_k, best_t, best_agree) in enumerate(alignment):
                    n_dis = (mpp_logicals[:, k] != old_logicals[:, best_t]).sum()
                    n_disagree_total += n_dis
                    per_round.append(n_dis / shots * 100)

                overall = n_disagree_total / (shots * r) * 100
                ok = final_ok and overall < 2.0  # tolerate small disagreement
                tag = "OK" if ok else "FAIL"

                print(
                    f"\n  d={d} r={r:>2} p={p}: "
                    f"final_match={final_ok}  overall_disagree={overall:.3f}%  [{tag}]"
                )
                # Print alignment for first, middle, last rounds
                show = [0, r // 2, r - 1]
                for idx in show:
                    obs_k, best_t, best_agree = alignment[idx]
                    print(
                        f"    obs {obs_k:>3} -> old_t={best_t:>3}  "
                        f"agree={best_agree*100:.2f}%  "
                        f"disagree={per_round[idx]:.2f}%"
                    )

                # Detailed look at disagreements for round 1
                k0 = 0
                t0 = alignment[0][1]
                disagree_mask = mpp_logicals[:, k0] != old_logicals[:, t0]
                n_dis_r1 = disagree_mask.sum()
                if n_dis_r1 > 0:
                    disagree_shots = np.where(disagree_mask)[0][:3]
                    print(f"    Round 1 disagreements ({n_dis_r1} shots):")
                    for s in disagree_shots:
                        fired = np.where(errors[s] & valid_cols)[0]
                        fired_t = time_idx[fired]
                        print(
                            f"      shot {s}: mpp={mpp_logicals[s,k0]} "
                            f"old={old_logicals[s,t0]}  "
                            f"fired_times={sorted(fired_t)}"
                        )

                all_ok &= ok

    print(f"\n  {'ALL PASSED' if all_ok else 'SOME FAILED'}\n")
    return all_ok


# ─── Test 4: Statistical LER comparison ──────────────────────────────


def test_statistical_ler():
    """
    Sample independently from both circuits, decode with their respective
    DEMs, compare logical error rates statistically.
    """
    print("=" * 64)
    print("TEST 3  –  Statistical logical error rate comparison")
    print("  Sample & decode independently, check LER is consistent")
    print("=" * 64)

    shots = 500_000
    all_ok = True

    for d in [3, 5]:
        for r in [5, 10]:
            p = 0.005

            std_circuit = make_standard_circuit(d, r, p)
            mpp_circuit = make_mpp_circuit(d, r, p)

            std_dem = std_circuit.detector_error_model()
            # Use the STANDARD DEM for decoding the MPP circuit (obs0 only)
            mpp_dem = mpp_circuit.detector_error_model(allow_gauge_detectors=True)

            # Decode standard circuit
            std_det, std_obs = std_circuit.compile_detector_sampler().sample(
                shots, separate_observables=True
            )
            std_matcher = pymatching.Matching.from_detector_error_model(std_dem)
            std_pred = std_matcher.decode_batch(std_det)
            std_err = (std_pred[:, 0] != std_obs[:, 0]).mean()

            # Decode MPP circuit — use standard DEM for matching
            mpp_det, mpp_obs = mpp_circuit.compile_detector_sampler().sample(
                shots, separate_observables=True
            )
            # Use standard DEM for decoding (matches memory: "use standard DEM for decoding")
            mpp_pred = std_matcher.decode_batch(mpp_det)
            mpp_err = (mpp_pred[:, 0] != mpp_obs[:, 0]).mean()

            # Binomial 95% CI: p ± 1.96*sqrt(p(1-p)/n)
            se_std = np.sqrt(std_err * (1 - std_err) / shots)
            se_mpp = np.sqrt(mpp_err * (1 - mpp_err) / shots)
            # Are they within 3 sigma of each other?
            diff = abs(std_err - mpp_err)
            combined_se = np.sqrt(se_std**2 + se_mpp**2)
            z = diff / combined_se if combined_se > 0 else 0
            ok = z < 3.0

            tag = "OK" if ok else "FAIL"
            print(
                f"  d={d} r={r:>2}: "
                f"std_LER={std_err:.4f}±{1.96*se_std:.4f}  "
                f"mpp_LER={mpp_err:.4f}±{1.96*se_mpp:.4f}  "
                f"z={z:.1f}  [{tag}]"
            )
            all_ok &= ok

    print(f"\n  {'ALL PASSED' if all_ok else 'SOME FAILED'}\n")
    return all_ok


# ─── main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ok1 = test_dem_probabilities()
    ok2 = test_random_error_vector()
    ok3 = test_old_vs_mpp()
    ok4 = test_statistical_ler()

    results = [
        ("DEM probabilities", ok1),
        ("Random error vector", ok2),
        ("Old vs MPP logicals", ok3),
        ("Statistical LER", ok4),
    ]

    print("=" * 64)
    if all(ok for _, ok in results):
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        for name, ok in results:
            if not ok:
                print(f"  - {name}: FAILED")
    print("=" * 64)
