import stim
import numpy as np
import torch
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from enum import Enum
from args import Args


def add_mpp_to_circuit(circuit: stim.Circuit, distance: int) -> stim.Circuit:
    """Insert noiseless MPP Z_L after each MR block for intermediate logical labels.

    obs 0 = final logical (unchanged), obs 1..N = intermediate logicals per round.
    """
    qubit_coords = {}
    for ins in circuit.flattened():
        if ins.name == "QUBIT_COORDS":
            args = ins.gate_args_copy()
            for t in ins.targets_copy():
                qubit_coords[t.value] = tuple(args)

    data_qubits = []
    for ins in circuit.flattened():
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

    for ins in circuit.flattened():
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


class FlipType(Enum):
    BIT = 1
    PHASE = 2

class Dataset:
    """
    Class that is used to generate graphs of errors that occur
    in quantum computers.

    Call generate_batch() to generate a batch of graphs.

    References:
    https://github.com/quantumlib/Stim/blob/main/doc/getting_started.ipynb
    https://github.com/LangeMoritz/GNN_decoder
    """
    def __init__(self, args: Args, flip: FlipType = FlipType.BIT):
        self.device = args.device
        self.error_rate = args.error_rate
        self.batch_size = args.batch_size
        self.t = args.t
        self.dt = args.dt
        self.distance = args.distance
        self.n_stabilizers = self.distance ** 2 - 1
        self.k = args.k
        self.seed = args.seed
        self.norm = args.norm
        self.label_mode = args.label_mode

        if flip is FlipType.BIT:
            self.code_task = "surface_code:rotated_memory_z"
        elif flip is FlipType.PHASE:
            self.code_task = "surface_code:rotated_memory_x"
        else:
            raise AttributeError("Unknown flip type.")
        self.__init_circuit()

    def __init_circuit(self):
        """
        Initializes circuits and samplers based on self.label_mode:
        - "last": DEM sampler, no intermediate label precomputation
        - "error_chain": DEM sampler + error mechanism -> time mapping
        - "mpp": MPP circuit detector sampler for intermediate labels
        """
        circuit = stim.Circuit.generated(
            self.code_task,
            distance=self.distance,
            rounds=self.t,
            after_clifford_depolarization=self.error_rate,
            after_reset_flip_probability=self.error_rate,
            before_measure_flip_probability=self.error_rate,
            before_round_data_depolarization=self.error_rate,
        )
        self.circuits = [circuit]
        self.dem = [circuit.detector_error_model()]

        # DEM sampler (used by "last" and "error_chain" modes)
        self.samplers = [dem.compile_sampler(seed=self.seed) for dem in self.dem]

        # Detector coordinates (always needed for graph construction)
        self.detector_coordinates = []
        self._sampler_t = []
        for dem in self.dem:
            coordinates = dem.get_detector_coordinates()
            detector_coordinates = np.array([v[-3:] for v in coordinates.values()])
            detector_coordinates -= detector_coordinates.min(axis=0)
            self.detector_coordinates.append(detector_coordinates.astype(np.int64))
            self._sampler_t.append(int(detector_coordinates[:, -1].max()))

        # Estimate acceptance rate (fraction of shots with ≥1 detection event)
        pilot = self.samplers[0].sample(shots=min(10000, self.batch_size * 3))
        n_accept = np.count_nonzero(np.any(pilot[0], axis=1))
        self._accept_rate = max(n_accept / pilot[0].shape[0], 0.05)

        # Precompute fully-connected edge weight matrix
        self._precompute_edge_weights()

        # Mode-specific initialization
        if self.label_mode == "error_chain":
            self._init_error_chain_data()
        elif self.label_mode == "mpp":
            self._init_mpp_samplers()

    def _init_error_chain_data(self):
        """Precompute error mechanism -> time mapping for error-chain labels."""
        for idx, dem in enumerate(self.dem):
            det_times = self.detector_coordinates[idx][:, -1].astype(int)

            term_dets = []
            term_logs = []
            for ins in dem.flattened():
                if ins.type != "error":
                    continue
                ts = ins.targets_copy()
                term_dets.append(np.fromiter((t.val for t in ts if t.is_relative_detector_id()), dtype=int))
                term_logs.append(np.fromiter((t.val for t in ts if t.is_logical_observable_id()), dtype=int))

            n_terms = len(term_dets)
            L0_mask = np.array([0 in logs for logs in term_logs], dtype=bool)
            has_dets = np.array([dets.size > 0 for dets in term_dets], dtype=bool)

            self.time_idx = np.full(n_terms, -1, dtype=int)
            for i, dets in enumerate(term_dets):
                if L0_mask[i] and has_dets[i]:
                    self.time_idx[i] = det_times[dets].max()

            self.valid_cols = self.time_idx >= 0

    def _init_mpp_samplers(self):
        """Build MPP circuits and compile their detector samplers."""
        self.mpp_circuits = [add_mpp_to_circuit(c, self.distance) for c in self.circuits]
        self.mpp_samplers = [c.compile_detector_sampler(seed=self.seed) for c in self.mpp_circuits]

    def _precompute_edge_weights(self):
        """Precompute fully-connected edge weight matrix for chunk-local positions."""
        unique_xy = np.unique(self.detector_coordinates[0][:, :2], axis=0)
        chunk_pos = np.array(
            [(x, y, tl) for x, y in unique_xy for tl in range(self.dt)],
            dtype=np.int64,
        )
        # Pairwise L-inf distance → inverse-square weights
        diff = chunk_pos[:, None, :] - chunk_pos[None, :, :]
        dist = np.abs(diff).max(axis=2).astype(np.float32)
        with np.errstate(divide='ignore'):
            self._edge_weights = np.where(dist > 0, 1.0 / dist**2, 0.0).astype(np.float32)

        # Coordinate → position index lookup (3D array for O(1) vectorized access)
        max_x = int(unique_xy[:, 0].max())
        max_y = int(unique_xy[:, 1].max())
        self._pos_idx = np.full((max_x + 1, max_y + 1, self.dt), -1, dtype=np.int64)
        for i, (x, y, tl) in enumerate(chunk_pos):
            self._pos_idx[x, y, tl] = i

        # Cache for local pair indices by group size
        self._local_pairs = {}

    def sample_syndromes(self, sampler_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Samples detection events and logical labels. Return shape depends on label_mode:
        - "last":        (detection_array [B, s], last_flip [B, 1])
        - "error_chain": (detection_array [B, s], logicals_each_round [B, T])
        - "mpp":         (detection_array [B, s], logicals_each_round [B, T])

        Only shots with at least one detection event are retained.
        """
        if self.label_mode == "last":
            return self._sample_last(sampler_idx)
        elif self.label_mode == "mpp":
            return self._sample_mpp(sampler_idx)
        else:
            return self._sample_error_chain(sampler_idx)

    def _sample_last(self, sampler_idx: int):
        """Sample from DEM, return only final logical label."""
        sampler = self.samplers[sampler_idx]
        detection_events_list, observable_flips_list = [], []
        n_draw = int(self.batch_size / self._accept_rate * 1.1) + 1
        while len(detection_events_list) < self.batch_size:
            detection_events, observable_flips, _ = sampler.sample(shots=n_draw)
            mask = np.any(detection_events, axis=1)
            detection_events_list.extend(detection_events[mask])
            observable_flips_list.extend(observable_flips[mask])
            if len(detection_events_list) < self.batch_size:
                remaining = self.batch_size - len(detection_events_list)
                rate = max(mask.sum() / n_draw, 0.05)
                n_draw = int(remaining / rate * 1.2) + 1
        detection_array = np.array(detection_events_list[:self.batch_size])
        flips_array = np.array(observable_flips_list[:self.batch_size], dtype=np.int32)
        return detection_array.astype(bool), flips_array[:, :1]

    def _sample_mpp(self, sampler_idx: int):
        """Sample from MPP circuit, extract intermediate logical labels from obs_flips."""
        sampler = self.mpp_samplers[sampler_idx]
        num_det = self.mpp_circuits[sampler_idx].num_detectors
        detection_events_list, observable_flips_list = [], []
        n_draw = int(self.batch_size / self._accept_rate * 1.1) + 1
        while len(detection_events_list) < self.batch_size:
            result = sampler.sample(shots=n_draw, append_observables=True)
            detection_events = result[:, :num_det]
            observable_flips = result[:, num_det:]
            mask = np.any(detection_events, axis=1)
            detection_events_list.extend(detection_events[mask])
            observable_flips_list.extend(observable_flips[mask])
            if len(detection_events_list) < self.batch_size:
                remaining = self.batch_size - len(detection_events_list)
                rate = max(mask.sum() / n_draw, 0.05)
                n_draw = int(remaining / rate * 1.2) + 1
        detection_array = np.array(detection_events_list[:self.batch_size])
        obs_flips = np.array(observable_flips_list[:self.batch_size], dtype=np.int32)
        # obs_flips[:, 1:] = intermediate logicals (noiseless MPP after each MR round, times 0..t-1)
        # obs_flips[:, 0]  = final logical (from noisy data qubit M, time t)
        logicals_each_round = np.hstack([obs_flips[:, 1:], obs_flips[:, 0:1]])
        return detection_array.astype(bool), logicals_each_round

    def _sample_error_chain(self, sampler_idx: int):
        """Sample from DEM with return_errors, compute intermediate labels from error mechanisms."""
        sampler = self.samplers[sampler_idx]
        detection_events_list, observable_flips_list, err_data_list = [], [], []
        while len(detection_events_list) < self.batch_size:
            detection_events, observable_flips, err_data = sampler.sample(
                shots=self.batch_size, return_errors=True)
            shots_w_flips = np.sum(detection_events, axis=1) != 0
            detection_events_list.extend(detection_events[shots_w_flips, :])
            observable_flips_list.extend(observable_flips[shots_w_flips, :])
            err_data_list.extend(err_data[shots_w_flips, :])
        detection_array = np.array(detection_events_list[:self.batch_size])
        flips_array = np.array(observable_flips_list[:self.batch_size], dtype=np.int32)
        err_data_array = np.array(err_data_list[:self.batch_size], dtype=bool)

        # Accumulate counts per time step for all shots
        E = err_data_array[:, self.valid_cols].astype(np.uint8)
        idx = self.time_idx[self.valid_cols].astype(np.int32)
        T = self._sampler_t[0] + 1

        order = np.argsort(idx, kind="stable")
        E_sorted = E[:, order]
        idx_sorted = idx[order]

        seg_starts = np.r_[0, np.flatnonzero(np.diff(idx_sorted)) + 1]
        uniq_idx = idx_sorted[seg_starts]

        counts_grouped = np.add.reduceat(E_sorted, seg_starts, axis=1)

        counts = np.zeros((self.batch_size, T), dtype=np.uint16)
        counts[:, uniq_idx] = counts_grouped

        parity = counts & 1
        logicals_each_round = np.bitwise_xor.accumulate(parity, axis=1).astype(np.int32)
        assert np.array_equal(logicals_each_round[:, -1], flips_array[:, 0]), \
            "Error-chain final logical value mismatch"
        return detection_array.astype(bool), logicals_each_round

    def get_sliding_window(self, node_features: list[np.ndarray], sampler_t: int
                           ) -> tuple[list[np.ndarray], np.ndarray]:
        """
        Applies a sliding window to the input node features in time,
        segmenting each shot's data into overlapping time chunks.

        There are g = t - dt + 2 chunks for each shot.
        """
        chunk_labels = [0] * len(node_features)  # Placeholder for per-batch chunk label arrays

        for batch, coordinates in enumerate(node_features):
            # Extract the time column from the coordinates
            times, counts = np.unique(coordinates[:, -1], return_counts=True)

            # Initialize chunk mapping: early, middle, late
            start = times < self.dt
            end = times > sampler_t - self.dt
            middle = ~(start | end)

            # Scale counts to estimate total number of node events after splitting
            counts[start] *= (times + 1)[start]                     # Early window
            counts[middle] *= self.dt                                # Full windows
            counts[end] *= -((times - 1) - sampler_t)[end]          # Late window
            new_size = np.sum(counts)

            # Allocate space for new coordinates and chunk indices
            new_coordinates = np.zeros((new_size, 3), dtype=np.uint64)
            chunk_label = np.zeros(new_size, dtype=np.uint64)

            # Sliding window index vector: [0, 1, ..., sampler_t - dt + 1]
            j_values = np.arange(sampler_t - self.dt + 2)[:, None]  # Shape: [num_chunks, 1]
            # Time values reshaped for broadcasting
            time_column = coordinates[:, -1][None, :]                # Shape: [1, num_points]
            # Create boolean mask indicating which time steps belong to which chunk
            mask = (time_column < j_values + self.dt) & (time_column >= j_values)  # [num_chunks, num_points]

            # Extract all matching (chunk, index) pairs
            indices = np.where(mask)
            # Sort by chunk index to maintain temporal order
            sorted_idx = np.argsort(indices[0])
            selected_points = coordinates[indices[1][sorted_idx]].copy()
            # Convert time coordinates to local (chunk-relative) time
            selected_points[:, -1] -= indices[0][sorted_idx]

            # Store results
            new_coordinates[:len(selected_points)] = selected_points
            chunk_label[:len(selected_points)] = indices[0][sorted_idx]

            node_features[batch] = new_coordinates
            chunk_labels[batch] = chunk_label

        # Concatenate all chunk labels into one array (for the whole batch)
        chunk_labels = np.concatenate(chunk_labels)
        return node_features, chunk_labels

    def get_node_features(self, syndromes: np.ndarray, sampler_idx: int):
        """
        Converts detection event indices into physical node features (x, y, t)
        and assigns them to batch and chunk labels via a sliding window over time.

        Returns float32 features, batch labels, chunk labels, and integer coords
        (for edge weight lookup).
        """
        node_features = [self.detector_coordinates[sampler_idx][s] for s in syndromes]

        node_features, chunk_labels = self.get_sliding_window(node_features, self._sampler_t[sampler_idx])

        batch_labels = np.repeat(np.arange(self.batch_size), [len(i) for i in node_features])
        coords_int = np.vstack(node_features)  # uint64
        node_features_float = coords_int.astype(np.float32)

        return node_features_float, batch_labels, chunk_labels, coords_int

    def _compute_fc_edges(self, coords_int, group_starts, group_sizes):
        """Compute fully connected edges with precomputed weights.

        For each group of nodes (same batch+chunk), creates all directed pairs
        and looks up edge weights from the precomputed weight matrix.
        """
        edge_src_list = []
        edge_tgt_list = []
        edge_weight_list = []

        unique_sizes, size_inverse = np.unique(group_sizes, return_inverse=True)

        for ui, s in enumerate(unique_sizes):
            if s <= 1:
                continue

            # All groups with this size
            starts = group_starts[size_inverse == ui]

            # Local pair indices for this size (cached)
            if s not in self._local_pairs:
                src_local, tgt_local = np.where(~np.eye(s, dtype=bool))
                self._local_pairs[s] = (src_local, tgt_local)
            src_local, tgt_local = self._local_pairs[s]

            # Global node indices: [n_groups, n_pairs_per_group] → flat
            global_src = (starts[:, None] + src_local[None, :]).ravel()
            global_tgt = (starts[:, None] + tgt_local[None, :]).ravel()

            # Look up position indices from integer coordinates
            src_coords = coords_int[global_src]
            tgt_coords = coords_int[global_tgt]
            pos_src = self._pos_idx[src_coords[:, 0], src_coords[:, 1], src_coords[:, 2]]
            pos_tgt = self._pos_idx[tgt_coords[:, 0], tgt_coords[:, 1], tgt_coords[:, 2]]

            weights = self._edge_weights[pos_src, pos_tgt]

            edge_src_list.append(global_src)
            edge_tgt_list.append(global_tgt)
            edge_weight_list.append(weights)

        if edge_src_list:
            edge_src = np.concatenate(edge_src_list)
            edge_tgt = np.concatenate(edge_tgt_list)
            edge_weights = np.concatenate(edge_weight_list)
            edge_index = torch.from_numpy(np.stack([edge_src, edge_tgt]).astype(np.int64))
            edge_attr = torch.from_numpy(edge_weights)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros(0, dtype=torch.float32)

        return edge_index, edge_attr

    def generate_batch(self):
        """
        Generates a batch of graphs.

        Returns:
            node_features, edge_index, labels, label_map, edge_attr,
            last_label, flips_full
        """
        sampler_idx = np.random.choice(len(self.samplers))
        syndromes, label_data = self.sample_syndromes(sampler_idx)

        if self.label_mode == "last":
            last_label = torch.from_numpy(label_data).to(dtype=torch.float32, device=self.device)
            flips_full = last_label
        else:
            flips_full = label_data[:, self.dt - 1:]  # shape: [B, g_max]
            flips_full = torch.from_numpy(flips_full).to(dtype=torch.float32, device=self.device)
            last_label = flips_full[:, -1:]

        node_features, batch_labels, chunk_labels, coords_int = self.get_node_features(syndromes, sampler_idx)
        node_features = torch.from_numpy(node_features)

        # Fast label_map: exploit sorted (batch, chunk) ordering
        g_max = self._sampler_t[sampler_idx] - self.dt + 2
        combined = batch_labels.astype(np.int64) * g_max + chunk_labels.astype(np.int64)
        change = np.empty(len(combined), dtype=bool)
        change[0] = True
        change[1:] = combined[1:] != combined[:-1]
        group_starts = np.flatnonzero(change)
        group_sizes = np.diff(np.append(group_starts, len(combined)))

        label_map = np.column_stack([batch_labels[group_starts], chunk_labels[group_starts]])
        labels = np.repeat(np.arange(len(group_starts), dtype=np.int64), group_sizes)
        label_map = torch.from_numpy(label_map)
        labels = torch.from_numpy(labels)

        edge_index, edge_attr = self._compute_fc_edges(coords_int, group_starts, group_sizes)

        node_features = node_features.to(self.device)
        labels = labels.to(self.device)
        label_map = label_map.to(dtype=torch.long, device=self.device)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)

        return node_features, edge_index, labels, label_map, edge_attr, last_label, flips_full

    def plot_graph(self, node_features, edge_index, labels, graph_idx):
        node_features = node_features.cpu().numpy()
        features = node_features[labels == graph_idx]
        min_t, max_t = 0, self.dt - 1
        edge_mask = (edge_index[0] == np.nonzero(labels == graph_idx)).cpu().numpy()
        edges = edge_index[:, np.any(edge_mask, axis=0)]

        ax = plt.axes(projection="3d")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("t")
        ax.set_xlim(0, self.distance)
        ax.set_ylim(0, self.distance)
        ax.set_zlim(min_t, max_t)
        ax.set_zticks(range(min_t, max_t + 1))
        ax.view_init(elev=60, azim=-90, roll=0)
        ax.set_aspect('equal', adjustable='box')
        plt.gca().invert_yaxis()

        c = ["red" if np.round(feature[3]) == 0 else "green" for feature in features]
        ax.scatter(*features.T[:3], c=c)

        edge_coordinates = node_features[edges].T[:3]
        plt.plot(*edge_coordinates, c="blue", alpha=0.3)

        x_stabs = np.nonzero(self.syndrome_mask == 1)
        z_stabs = np.nonzero(self.syndrome_mask == 3)
        ax.scatter(x_stabs[1], x_stabs[0], min_t, c="red",  alpha=0.3, s=50, label="X stabilizers")
        ax.scatter(z_stabs[1], z_stabs[0], min_t, c="green", alpha=0.3, s=50, label="Z stabilizers")
        plt.legend()
        plt.show()
