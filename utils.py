import os
import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch_geometric.nn import GraphConv
import torch
import logging
import time
from typing import Dict
StateDict = Dict[str, torch.Tensor]
import stim

def make_surface_code_with_logical_z_tracking(distance: int, rounds: int, error_rate: float) -> stim.Circuit:
    base = stim.Circuit.generated(
        code_task="surface_code:rotated_memory_x",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=error_rate,
        after_reset_flip_probability=error_rate,
        before_measure_flip_probability=error_rate,
        before_round_data_depolarization=error_rate,
    )

    # Split into prefix, repeat block, and suffix
    for i, instr in enumerate(base):
        if isinstance(instr, stim.CircuitRepeatBlock):
            prefix = base[:i]
            repeat_block = instr
            suffix = base[i+1:]
            break
    else:
        raise ValueError("No REPEAT block found in generated circuit.")
    def get_logical_z_qubits_west_edge(circuit: stim.Circuit) -> list[int]:
        coords = circuit.get_final_qubit_coordinates()
        return sorted(
            [q for q, (x, y) in coords.items() if x == 1],
            key=lambda q: coords[q][1]  # sort by y
        )
    # Get logical Z path: western row of data qubits in rotated layout
    logical_z_qubits = get_logical_z_qubits_west_edge(base)

    def logical_z_measurement_block(index: int) -> stim.Circuit:
        c = stim.Circuit()
        anc = 0
        c.append("R", [anc])
        for q in logical_z_qubits:
            c.append("H", [q])
        for q in logical_z_qubits:
            c.append("CX", [q, anc])
        for q in logical_z_qubits:
            c.append("H", [q])
        c.append("MR", [anc])
        c += stim.Circuit(f"OBSERVABLE_INCLUDE({index}) rec[-1]")
        return c

    # Patch detector offsets in repeat block
    def patch_detector_offsets(circuit_block, extra_offset):
        patched = stim.Circuit()
        for instr in circuit_block:
            if instr.name == "DETECTOR":
                targets = instr.targets_copy()
                args = []
                rec_count = sum(t.is_measurement_record_target for t in targets)
                rec_seen = 0
                for t in targets:
                    if t.is_measurement_record_target:
                        rec_seen += 1
                        shift = extra_offset if rec_seen == rec_count else 0
                        args.append(f"rec[{t.value - shift}]")
                    else:
                        args.append(str(t))
                loc = "(" + ", ".join(str(x) for x in instr.gate_args_copy()) + ")" if instr.gate_args_copy() else ""
                patched += stim.Circuit(f"DETECTOR{loc} " + " ".join(args))
            else:
                patched.append(instr)
        return patched

    patched_repeat_block = patch_detector_offsets(repeat_block.body_copy(), extra_offset=1)

    new_circuit = stim.Circuit()
    new_circuit.append("QUBIT_COORDS", [0], [0, 0])  # Ancilla at index 0, coord (0, 0)
    new_circuit += prefix
    new_circuit += logical_z_measurement_block(0)

    for r in range(1, rounds - 1):
        new_circuit += patched_repeat_block
        new_circuit += logical_z_measurement_block(r)
    # no ancilla based measurement of Z_L in last round, take direct measurement of
    # original circuit instead
    new_circuit += patched_repeat_block

    patched_suffix = stim.Circuit()

    def patch_detector_offsets_suffix(suffix):
        patched = stim.Circuit()
        for instr in suffix:
            if instr.name == "OBSERVABLE_INCLUDE":
                # Only relabel observable index; keep original rec[] indices
                targets = instr.targets_copy()
                rec_targets = " ".join(f"rec[{t.value}]" for t in targets)
                patched += stim.Circuit(f"OBSERVABLE_INCLUDE({rounds - 1}) {rec_targets}")

            else:
                patched.append(instr)
        return patched

    patched_suffix = patch_detector_offsets_suffix(suffix)
    new_circuit += patched_suffix

    return new_circuit


def group(x, label_map):
        """
        Groups graphs according to which batch element they belong to. 

        Args:
        x: tensor of shape [n_graphs, embedding size]. 
        label_map: tensor of shape [n_graphs]. 
    
        Returns: 
        A tensor of shape [batch size, g, embedding size] where
            g represents the number of graphs belonging to a batch element. 
            If t = 24 and dt = 5, then g = 5, i.e. g = t - dt + 2.
            Batch elements may contain less than t - dt + 2 graphs. 
            This happens when there are no detection events in a chunk. 
            For instance, if t = 24 and dt = 5, and no detection
            events occur between timesteps 0 and 4, there would
            be no graph for this chunk. Therefore, any "missing" graphs are 
            replaced with zeros, such that the dimensions work out properly. 
            The zero padding happens at the end of the sequence, e.g. if 
            g = 5 and some batch element consists only of graphs 2 and 3,
            the result would look like [2, 3, 0, 0, 0], where 2 and 3 
            represent the graph embeddings for graphs 2 and 3, and the zeros
            represent zero-padding.  
        """     
        counts = torch.unique(label_map[:, 0], return_counts=True)[-1]
        grouped = torch.split(x, list(counts))
        padded = pad_sequence(grouped, batch_first=True)
        # padded has shape [batch, t, embedding_features[-1]]
        return pack_padded_sequence(padded, counts.cpu(), batch_first=True, enforce_sorted=False)

class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, act=nn.ReLU()):
        super().__init__()
        self.layer = GraphConv(in_features, out_features)
        self.act = act
    
    def forward(self, x, edge_index, edge_attr):
        x = self.layer(x, edge_index, edge_attr)
        return self.act(x)

class TrainingLogger:
    def __init__(self, logfile=None, statsfile=None):
        if logfile:
            os.makedirs("./logs", exist_ok=True)
        logging.basicConfig(filename=f"./logs/{logfile}.out", level=logging.INFO, format="%(message)s")
        self.logs = []
        self.statsfile = statsfile
        self.best_accuracy = 0 
    
    def on_epoch_begin(self, epoch):
        self.t0 = time.perf_counter()
        self.epoch = epoch
        logging.info(f"EPOCH {epoch} starting")
    
    def on_epoch_end(self, logs=None):
        epoch_time = time.perf_counter() - self.t0
        if logs["accuracy"] > self.best_accuracy:
            self.best_accuracy = logs["accuracy"]
        logging.info(
            f"EPOCH {self.epoch} finished in {epoch_time:.3f} seconds with lr = {logs['lr']:.2e}:\n"
            f"\tloss = {logs['loss']:.5f}, accuracy = {logs['accuracy']:.4f} ({self.best_accuracy:.4f})\n"
            f"\tmodel time = {logs['model_time']:.2f} seconds, "
            f"data time = {logs['data_time']:.2f} seconds"
        )
        self.logs.append(logs)

    def on_training_begin(self, args):
        logging.info(f"Training with t = {args.t}, dt = {args.dt}, distance = {args.distance}")
    
    def on_training_end(self):
        stats = np.vstack((
            [logs["model_time"] for logs in self.logs],
            [logs["data_time"] for logs in self.logs],
            [logs["lr"] for logs in self.logs],
            [logs["loss"] for logs in self.logs],
            [logs["accuracy"] for logs in self.logs],
        ))
        if self.statsfile:
            os.makedirs("./stats", exist_ok=True)
            np.save(f"./stats/{self.statsfile}", stats)

def standard_deviation(p, n):
    """
    Standard deviation of the Binomial distribution.
    https://en.wikipedia.org/wiki/Binomial_distribution
    """
    return np.sqrt(p * (1 - p) / n)
