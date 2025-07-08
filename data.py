import stim 
import numpy as np
import numpy.typing as npt
import torch
from tqdm import tqdm
import time, itertools
import matplotlib.pyplot as plt
from enum import Enum
from args import Args
from torch_geometric.nn.pool import knn_graph
from utils import make_surface_code_with_logical_z_tracking
import collections

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
        self.error_rates = args.error_rates 
        self.batch_size = args.batch_size
        self.t = args.t
        self.dt = args.dt 
        self.distance = args.distance
        self.n_stabilizers = self.distance ** 2 - 1
        self.sliding = args.sliding
        self.k = args.k
        self.seed = args.seed
        self.norm = args.norm
        
        if not self.sliding:
            for t in args.t:
                assert t % args.dt == args.dt - 1 
        if flip is FlipType.BIT:
            self.code_task = "surface_code:rotated_memory_z"
        elif flip is FlipType.PHASE:
            self.code_task = "surface_code:rotated_memory_x"
        else: 
            raise AttributeError("Unknown flip type.")
        self.__init_circuit()
    
    def __init_circuit(self):
        """
        Initializes the Stim circuits used for sampling detection events and logical observables.

        This involves:
        - Creating a list of circuits for each (error_rate, t) combination using
          `make_surface_code_with_intermediate_observables`, which includes intermediate
          OBSERVABLE_INCLUDE instructions to enable time-resolved logical labels.
        - Compiling those circuits into Stim detector samplers.
        - Precomputing and storing the physical detector coordinates for each circuit.
        - Building a syndrome mask (stabilizer layout) to identify stabilizer types (X vs Z).

        Note:
            - Currently only supports "rotated_memory_x" (phase-flip decoding).
            - Assumes all `t` values in self.t are consistent for indexing detector coordinates.
        """

        # Generate all (error_rate, t) pairs to create one circuit per setting
        combinations = itertools.product(self.error_rates, self.t)

        # Build one circuit per combination, with intermediate logical observables inserted
        self.circuits = [
            make_surface_code_with_logical_z_tracking(
                distance=self.distance,
                rounds=t,
                error_rate=error_rate
            )
            for error_rate, t in combinations
        ]

        # Compile each circuit into a Stim detector sampler for fast sampling
        self.samplers = [
            circuit.compile_detector_sampler(seed=self.seed)
            for circuit in self.circuits
        ]

        # Precompute and store the physical detector coordinates for each circuit
        # These are used to map detection event indices into real space-time coordinates
        self.detector_coordinates = []
        for circuit in self.circuits:
            # Hardcoded to use "rotated_memory_x" regardless of self.code_task
            detector_coordinate = stim.Circuit.generated(
                code_task="surface_code:rotated_memory_x",  # NOTE: hardcoded
                distance=self.distance,
                rounds=self.t[0]
            ).get_detector_coordinates()

            # Convert from dict to array and rescale x, y by 1/2 (Stim convention)
            detector_coordinate = np.array(list(detector_coordinate.values()))
            detector_coordinate[:, :2] /= 2
            self.detector_coordinates.append(detector_coordinate.astype(np.int64))

        # Build a mask that identifies where stabilizers are placed in the lattice
        # syndrome_x marks positions of X stabilizers (value 1)
        # syndrome_z is a rotated version and scaled by 3 to encode Z stabilizers
        sz = self.distance + 1
        syndrome_x = np.zeros((sz, sz), dtype=np.uint8)
        syndrome_x[::2, 1:sz - 1:2] = 1
        syndrome_x[1::2, 2::2] = 1
        syndrome_z = np.rot90(syndrome_x) * 3

        # The final syndrome mask distinguishes X (1) and Z (3) stabilizers
        self.syndrome_mask = syndrome_x + syndrome_z


    def sample_syndromes(self, sampler_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Samples a batch of detection events and corresponding logical observables
        from a Stim circuit compiled with intermediate observable tracking.
    
        Returns:
            detection_array: a boolean array of shape [batch_size, s] where
                - s = total number of detectors = t * num_stabilizers.
                Each entry indicates whether a detection event occurred on a 
                given stabilizer at a given round.
            
            flips_array: an integer array of shape [batch_size, self.t], where
                - self.t = number of logical observable measurement points
                Each entry flips_array[b, i] is 1 if a logical bit- or phase-flip
                has occurred in batch element `b` up to and including chunk `i`,
                as reported by the corresponding OBSERVABLE_INCLUDE instruction.
                Otherwise, it is 0.
    
        Notes:
            - The circuit used must include intermediate OBSERVABLE_INCLUDE statements
              at chunk boundaries (e.g., after each dt rounds).
            - Only shots with at least one detection event are retained.
            - This method returns one logical label per time step, allowing
              for training of a sequential model with per-step targets.
            - When setting the number of QEC rounds to t, stim will return t + 1 X
            stabilizers, and t - 1 Z stabilizers. In total, there are t + 1 'time 
            steps'. This means, that there are g = t - dt + 2 chunks for each shot. 
        """

        sampler = self.samplers[sampler_idx]
        detection_events_list, observable_flips_list = [], []
        # Sample until we get a batch where each element has at least
        # one detection event. 
        while len(detection_events_list) < self.batch_size: 
            detection_events, observable_flips = sampler.sample(
                shots=self.batch_size, separate_observables=True)
            shots_w_flips = np.sum(detection_events, axis=1) != 0 # only include cases where there is at least one detection event.
            detection_events_list.extend(detection_events[shots_w_flips, :])
            observable_flips_list.extend(observable_flips[shots_w_flips, :])
        detection_array = np.array(detection_events_list[:self.batch_size])
        flips_array = np.array(observable_flips_list[:self.batch_size], dtype=np.int32)
        return detection_array.astype(bool), flips_array
    
    def get_sliding_window(self, node_features: list[np.ndarray], sampler_t: int
                           ) -> tuple[list[np.ndarray], np.ndarray]:
        """
        Applies a sliding window to the input node features in time,
        segmenting each shot's data into overlapping time chunks.

        This is used to divide each graph (shot) into smaller graph segments
        that span dt rounds of the circuit. The result is a per-chunk 
        representation suitable for sequential processing (e.g., in an RNN).

        Args:
            node_features: List of length batch_size. Each element is an array of 
                shape [n_i, 3] containing the node features (x, y, t) for a single
                shot (i.e., detection events).
            sampler_t: The number of rounds used in the circuit (i.e., full time duration, = t).

        Returns:
            A tuple (node_features, chunk_labels):
                node_features: Modified list where each entry's coordinates are mapped 
                    into chunk-local time and reordered to align with chunk boundaries.
                chunk_labels: A 1D array indicating to which chunk (window) each 
                    node in the batch belongs. This is later used for pooling and batching.
        Note:
            - When setting the number of QEC rounds to t, stim will return t + 1 X
            stabilizers, and t - 1 Z stabilizers. In total, there are t + 1 'time 
            steps'. This means, that there are g = t - dt + 2 chunks for each shot. 
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

            counts[middle] *= self.dt                              # Full windows
            counts[end] *= -((times - 1) - sampler_t)[end]         # Late window
            new_size = np.sum(counts)
            # Allocate space for new coordinates and chunk indices

            new_coordinates = np.zeros((new_size, 3), dtype=np.uint64)
            chunk_label = np.zeros(new_size, dtype=np.uint64)

            # Sliding window index vector: [0, 1, ..., sampler_t - dt + 1]
            j_values = np.arange(sampler_t - self.dt + 2)[:, None]  # Shape: [num_chunks, 1]

            # Time values reshaped for broadcasting
            time_column = coordinates[:, -1][None, :]  # Shape: [1, num_points]

            # Create boolean mask indicating which time steps belong to which chunk
            mask = (time_column < j_values + self.dt) & (time_column >= j_values)  # Shape: [num_chunks, num_points]

            # Extract all matching (chunk, index) pairs
            indices = np.where(mask)
            
            # Sort by chunk index to maintain temporal order
            sorted_idx = np.argsort(indices[0])
            selected_points = coordinates[indices[1][sorted_idx]].copy()

            # Convert time coordinates to local (chunk-relative) time
            selected_points[:, -1] -= indices[0][sorted_idx]
            
            # Store results
            new_coordinates[:len(selected_points)] = selected_points
            # Store for this batch element
            chunk_label[:len(selected_points)] = indices[0][sorted_idx]

            node_features[batch] = new_coordinates
            chunk_labels[batch] = chunk_label
            
        # Concatenate all chunk labels into one array (for the whole batch)
        chunk_labels = np.concatenate(chunk_labels)
        return node_features, chunk_labels

    def get_node_features(self, syndromes: np.ndarray, sampler_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Converts detection event indices into physical node features and assigns 
        them to batch and chunk labels, optionally applying a sliding window.

        Args:
            syndromes: Boolean array of shape [batch_size, s] where s is the total
                number of detectors (t * number of stabilizers). Each entry indicates
                whether a detection event occurred at a given space-time location.
            sampler_idx: Index into the list of circuits/samplers. This determines 
                which surface code configuration (e.g., error rate, number of rounds) 
                was used for this batch.

        Returns:
            node_features: ndarray of shape [n, 5] where each row is (x, y, t, type_x, type_z).
                - x, y, t: spatial and temporal position of a detection event
                - type_x, type_z: one-hot encoding of stabilizer type
            batch_labels: ndarray of shape [n], mapping each node to a batch element
            chunk_labels: ndarray of shape [n], mapping each node to a time chunk (graph)
        Note:
            - When setting the number of QEC rounds to t, stim will return t + 1 X
            stabilizers, and t - 1 Z stabilizers. In total, there are t + 1 'time 
            steps'. This means, that there are g = t - dt + 2 chunks for each shot. 
        """

        # Decode syndrome indices into (x, y, t) coordinates using precomputed Stim detector layout
        # Result: list of arrays, one per shot, each with shape [num_events_in_shot, 3]
        node_features = [self.detector_coordinates[sampler_idx][s] for s in syndromes]

        if self.sliding:
            # Total number of rounds used in this circuit
            sampler_t = self.circuits[sampler_idx].num_detectors // self.n_stabilizers
            # Apply a sliding window over time to divide events into overlapping chunks
            # Returns updated node_features with local time coordinates and chunk_labels
            node_features, chunk_labels = self.get_sliding_window(node_features, sampler_t)
        
        # Construct a batch_labels array that repeats batch indices according to number of events
        # Example: if shot 0 has 3 events and shot 1 has 5, this will be [0, 0, 0, 1, 1, 1, 1, 1]
        batch_labels = np.repeat(np.arange(self.batch_size), [len(i) for i in node_features])
        counter = collections.Counter(list(zip(batch_labels, chunk_labels)))
        g = self.t[0] - self.dt + 2

        # Combine all node features into a single array [total_nodes, 3]
        node_features = np.vstack(node_features)

        if not self.sliding:
            # If sliding window is not used, manually compute chunk index and local time:
            #   - chunk = t // dt
            #   - local_t = t % dt
            chunk_labels = node_features[:, -1] // self.dt
            node_features[:, -1] = node_features[:, -1] % self.dt

        # Determine stabilizer type at each (x, y) coordinate using the precomputed mask
        # syndrome_mask == 3 indicates Z stabilizer; else it's X stabilizer
        stabilizer_type = self.syndrome_mask[node_features[:, 1], node_features[:, 0]] == 3
        stabilizer_type = stabilizer_type[:, np.newaxis]  # Shape: [n, 1]

        # Add one-hot stabilizer type to feature vector: [x, y, t, is_Z, is_X]
        node_features = np.hstack((node_features, stabilizer_type, ~stabilizer_type)).astype(np.float32)

        return node_features, batch_labels, chunk_labels

    
    def get_edges(self, node_features: np.ndarray, labels) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Returns edges between nodes. The edges are of shape [n_edges, 2].

        Use ord=torch.inf for the supremum norm, ord=2 for euclidean norm.
        """
        # Compute edges.
        edge_index = knn_graph(node_features, self.k, batch=labels)

        # Compute the distances between the nodes:
        delta = node_features[edge_index[1]] - node_features[edge_index[0]]
        edge_attr = torch.linalg.norm(delta, ord=self.norm, dim=1)

        # Inverse square of the norm between two nodes.
        edge_attr = 1 / edge_attr ** 2

        return edge_index, edge_attr

    def generate_batch(self):
        """
        Generates a batch of graphs. 

        Returns: 
            node_features: tensor of shape [n, 5] ([x, y, t, (stabilizer type)]).
            edge_index: tensor of shape [n_edges, 2]. Represents the edges, 
                i.e. the adjacency matrix. 
            labels: tensor of shape [n]. Represents which node features belong
                to which combination of batch element and chunk. 
                This is used when computing global_mean_pool following
                graph convolutions. The reason being there is no 
                explicit batch dimension. Therefore, a list of 
                labels is needed to keep track of which node features
                belong to which batch element. Further, each batch element
                consists of multiple graphs, or chunks. Therefore, an integer
                is assigned to each combination of batch element and chunk.
            label_map: tensor of shape [n_graphs]. Maps labels to
                [batch element, chunk].  
            edge_attr: tensor of shape [n_edges]. Represents the edge weights.
            flips: tensor of shape [batch_size, g]. Indicates if a logical 
                bit- or phase-flip has occurred at the end of each chunk.
            last_label: tensor of shape [batch_size, 1]. Indicates if a logical 
                bit- or phase-flip has occurred at the end of the whole circuit.

        Note:
            - When setting the number of QEC rounds to t, stim will return t + 1 X
              stabilizers, and t - 1 Z stabilizers. In total, there are t + 1 'time 
              steps'. This means, that there are g = t - dt + 2 chunks for each shot. 
            - Only logical observables measured at the end of each chunk are kept,
              i.e., we discard the first `dt` entries and keep the final g entries.
            - Because stim gives t logical observables, we copy the last label to
              the last chunk ending at the last perfect stabilizer measurement
        """
        # Sample syndromes and full logical flips per time step
        sampler_idx = np.random.choice(len(self.samplers))
        syndromes, flips = self.sample_syndromes(sampler_idx)

        # Keep only labels at chunk boundaries (i.e., end of each chunk)
        flips = flips[:, self.dt - 1:]  # shape: [batch_size, g - 1], where g = t - dt + 2
        flips = torch.from_numpy(flips).to(dtype=torch.float32, device=self.device)
        # Append the last label one more time to get [B, g]
        last_label = flips[:, -1:]  # shape [B, 1]
        flips = torch.cat([flips, last_label], dim=1)  # shape [B, g]

        # Extract graph structure and labels for non-empty chunks
        node_features, batch_labels, chunk_labels = self.get_node_features(syndromes, sampler_idx)
        node_features = torch.from_numpy(node_features)

        # Map each unique (batch, chunk) pair to a unique graph index
        label_map = np.array(list(zip(batch_labels, chunk_labels)))
        label_map, counts = np.unique(label_map, axis=0, return_counts=True)
        labels = np.repeat(np.arange(counts.shape[0]), counts).astype(np.int64)
        label_map = torch.from_numpy(label_map)
        labels = torch.from_numpy(labels)

        # Extract graph edges and attributes
        edge_index, edge_attr = self.get_edges(node_features, labels)

        # align labels with chunk indices: 
        aligned_flips, lengths = self.align_labels_to_outputs(label_map, flips)

        # Move everything to the appropriate device
        node_features = node_features.to(self.device)
        labels = labels.to(self.device)
        label_map = label_map.to(dtype=torch.float32, device=self.device)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)
        lengths = lengths.to(self.device)

        return node_features, edge_index, labels, label_map, edge_attr, aligned_flips, lengths, last_label
    
    def align_labels_to_outputs(self, label_map: torch.Tensor, flips_full: torch.Tensor) -> torch.Tensor:
        """
        Given label_map and full logical flips, return a label tensor aligned
        with the packed GRU output (i.e., labels only for non-empty chunks, in GRU order).

        Args:
            label_map: Tensor of shape [n_graphs, 2], with (batch_idx, chunk_idx)
            flips_full: Tensor of shape [B, g], with one label per possible chunk

        Returns:
            aligned_labels: Tensor of shape [B, L], aligned with GRU output
        """
        B = int(label_map[:, 0].max().item()) + 1
        lengths = torch.bincount(label_map[:, 0].long(), minlength=B)  # number of real chunks per batch
        max_len = lengths.max().item()

        aligned_flips = torch.zeros(B, max_len, device=self.device)
        offsets = torch.zeros(B, dtype=torch.long, device=self.device)

        for i in range(label_map.size(0)):
            b = int(label_map[i, 0])
            t = int(label_map[i, 1])
            pos = offsets[b].item()
            aligned_flips[b, pos] = flips_full[b, t]
            offsets[b] += 1

        return aligned_flips, lengths  # counts = lengths for masking



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

        # Plotting nodes.
        c = ["red" if np.round(feature[3]) == 0 else "green" for feature in features]
        ax.scatter(*features.T[:3], c=c)
        
        # Plotting edges.
        edge_coordinates = node_features[edges].T[:3]
        plt.plot(*edge_coordinates, c="blue", alpha=0.3)
       
        # Plotting stabilizers.
        x_stabs = np.nonzero(self.syndrome_mask == 1)
        z_stabs = np.nonzero(self.syndrome_mask == 3)
        ax.scatter(x_stabs[1], x_stabs[0], min_t, c="red",  alpha=0.3, s=50, label="X stabilizers")
        ax.scatter(z_stabs[1], z_stabs[0], min_t, c="green", alpha=0.3, s=50, label="Z stabilizers")
        plt.legend()
        plt.show()
        
if __name__ == "__main__":
    args = Args(error_rates=[0.002], t=[99], sliding=True, dt=8)
    dataset = Dataset(args)
    t0 = time.perf_counter()
    node_features, edge_index, labels, label_map, edge_attr, flips = dataset.generate_batch()
    for i in tqdm(range(10)):
        dataset.plot_graph(node_features, edge_index, labels, i)
    print(f"{time.perf_counter() - t0:.3f} seconds")

