"""Use the released DIRAC annotation model for RNA and Visium-graph transfer."""
import argparse
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch_geometric.index import index2ptr
from torch_geometric.loader import ClusterData
from torch_geometric.utils import sort_edge_index

from prepare_folds import SEED, WORK


import pymetis
import sodirac
from sodirac import main as dirac


PATIENTS = ("AT6", "AT10", "AT13", "AT15", "AT3", "AT4")
CLASSES = ("CT", "LE_IT", "necrosis_related")
EPOCHS = 100
SUBGRAPH_NODES = 4096


class MetisClusterData(ClusterData):
    """Use the METIS wheel when PyG's optional compiled partitioner is unavailable."""

    def _metis(self, edge_index, num_nodes):
        if self.num_parts == 1:
            return torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
        row, column = sort_edge_index(edge_index, num_nodes=num_nodes)
        pointer = index2ptr(row, size=num_nodes).cpu().numpy()
        adjacency = pymetis.CSRAdjacency(pointer, column.cpu().numpy())
        parts = pymetis.part_graph(self.num_parts, adjacency=adjacency,
                                   recursive=self.recursive, options=pymetis.Options(seed=SEED))
        assignments = torch.as_tensor(np.asarray(parts.vertex_part), dtype=torch.long)
        assert assignments.numel() == num_nodes and assignments.unique().numel() == self.num_parts
        return assignments.to(edge_index.device)


def read_csr(group):
    return sparse.csr_matrix((group["data"][:], group["indices"][:], group["indptr"][:]),
                             shape=tuple(group.attrs["shape"]))


def read_graph(group):
    data = read_csr(group["expression"]).toarray()
    std = data.std(axis=0, ddof=1, dtype=np.float64).astype(data.dtype)
    std[std == 0] = 1
    data -= data.mean(axis=0, dtype=np.float64).astype(data.dtype)
    data /= std
    adjacency = read_csr(group["adjacency"]).tocoo()
    edges = torch.from_numpy(np.stack((adjacency.row, adjacency.col)).astype(np.int64))
    return data, edges


def run_fold(patient, destination, epochs):
    data_path = WORK / "fold_inputs" / f"{patient}.h5"
    with h5py.File(data_path, "r") as h5:
        assert h5.attrs["held_out_patient"] == patient
        source_patients = json.loads(h5.attrs["source_patients"])
        assert len(source_patients) == 5 and patient not in source_patients
        reference, query = h5["reference"], h5["query"]
        assert "label" not in query
        source, source_edges = read_graph(reference)
        target, target_edges = read_graph(query)
        labels = reference["label"][:].astype(np.int64)
        samples = query["sample"].asstr()[:]
        ids = query["unit_id"].asstr()[:]
    assert set(labels) == {0, 1, 2}
    sodirac.utils.seed_torch(seed=SEED)
    model_folder = destination / patient
    model_folder.mkdir(parents=True, exist_ok=True)
    app = dirac.annotate_app(save_path=str(model_folder), use_gpu=True)
    assert isinstance(app.device, torch.device) and app.device.type == "cuda"
    # DIRAC pairs source and target batches with zip. Equal part counts use the full two graphs.
    parts = max(1, int(np.ceil(max(len(source), len(target))/SUBGRAPH_NODES)))
    inputs = app._get_data(
        source_data=source, source_label=labels, source_edge_index=source_edges,
        target_data=target, target_edge_index=target_edges, num_workers=0,
        num_parts_source=parts, num_parts_target=parts,
    )
    assert len(inputs["source_graph_dl"]) == len(inputs["target_graph_dl"]) == parts
    del source, target
    model = app._get_model(samples=inputs)
    output = app._train_dirac_annotate(
        samples=inputs, models=model, n_epochs=epochs, filter_low_confidence=False,
    )
    scores = output["target_prob"]
    assert scores.shape == (len(ids), len(CLASSES))
    assert np.isfinite(scores).all()
    np.testing.assert_allclose(scores.sum(axis=1), 1, rtol=0, atol=2e-6)
    np.testing.assert_array_equal(scores.argmax(axis=1), output["target_pred"])
    table = pd.DataFrame(scores, columns=CLASSES)
    table.insert(0, "held_out_patient", patient)
    table.insert(0, "unit_id", ids)
    table.insert(0, "sample", samples)
    temporary = destination / f"{patient}.build.csv"
    table.to_csv(temporary, index=False)
    temporary.replace(destination / f"{patient}.csv")
    (model_folder / "run.json").write_text(json.dumps({
        "held_out_patient": patient, "reference_patients": source_patients,
        "sodirac_version": sodirac.__version__, "seed": SEED, "epochs": epochs,
        "input": "fold-specific coding RNA with separate domain scaling and native Visium adjacency",
        "parts_per_domain": parts, "max_nominal_part_size": SUBGRAPH_NODES,
        "target_labels_available_in_training": False,
        "unlabelled_target_adaptation": True,
        "metis_backend": "PyMetis for ClusterData",
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("patients", nargs="*", choices=PATIENTS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    default = WORK / ("smoke/DIRAC" if args.smoke else "fold_predictions/DIRAC")
    destination = args.output_dir or default
    dirac.ClusterData = MetisClusterData
    torch.set_num_threads(3)
    torch.set_float32_matmul_precision("high")
    for patient in args.patients or PATIENTS:
        if (destination / f"{patient}.csv").exists():
            continue
        start = time.monotonic()
        print("BEGIN", patient, flush=True)
        run_fold(patient, destination, epochs=2 if args.smoke else EPOCHS)
        torch.cuda.empty_cache()
        print("DONE", patient, f"{time.monotonic()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
