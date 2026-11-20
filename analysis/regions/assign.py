"""Convert six fold predictions to the analytical CT and LE/IT bin100 regions."""
import argparse

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

CLASSES = ["CT", "LE_IT", "necrosis_related"]


def assign(folds, graph, component):
    if folds.shape != (6, len(component), 3) or not np.isfinite(folds).all():
        raise ValueError("Expected six finite fold-by-node probability arrays with three classes")
    if (folds < 0).any() or not np.allclose(folds.sum(axis=2), 1, atol=1e-5):
        raise ValueError("Invalid class probabilities")
    if pd.isna(component).any() or graph.shape != (len(component), len(component)):
        raise ValueError("Tissue component or graph node order is invalid")
    if (graph != graph.T).nnz or not np.isfinite(graph.data).all() or (graph.data < 0).any():
        raise ValueError("Provide a finite, nonnegative undirected within-tissue adjacency")
    graph = graph.tocsr(copy=True)
    graph.setdiag(0)
    graph.eliminate_zeros()
    u, v = graph.nonzero()
    if np.any(component[u] != component[v]):
        raise ValueError("Found an edge across native tissue components")
    avg = folds.mean(axis=0)
    winner = avg.argmax(axis=1)
    votes = (folds.argmax(axis=2) == winner).sum(axis=0)
    raw = np.where((avg.max(axis=1) >= .60) & (votes >= 4), winner, -1)
    retained = np.full(len(component), -1, dtype=int)
    for code in (0, 1):
        ix = np.flatnonzero(raw == code)
        if len(ix):
            _, island = connected_components(graph[ix][:, ix], directed=False)
            retained[ix[np.bincount(island)[island] >= 20]] = code
    for comp in np.unique(component):
        ix = component == comp
        if any(np.count_nonzero(retained[ix] == code) < 100 for code in (0, 1)):
            retained[ix] = -1
    return pd.DataFrame({"prob_CT": avg[:, 0], "prob_LE_IT": avg[:, 1], "prob_necrosis": avg[:, 2],
                         "votes": votes, "raw_label": raw, "analysis_label": retained})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs=6, required=True, help="Six outputs from algorithm/predict.py in fold order")
    parser.add_argument("--nodes", required=True, help="CSV: unit_id,x_um,y_um,native_component, in adjacency order")
    parser.add_argument("--graph", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    nodes = pd.read_csv(args.nodes, dtype={"unit_id": str})
    if not nodes.unit_id.is_unique or len(set(args.folds)) != 6:
        raise ValueError("Provide six distinct folds and unique node IDs")
    preds = []
    for filename in args.folds:
        with np.load(filename, allow_pickle=False) as f:
            np.testing.assert_array_equal(f["classes"].astype(str), CLASSES)
            np.testing.assert_array_equal(f["unit_id"].astype(str), nodes.unit_id)
            preds.append(f["probabilities"])
    out = assign(np.stack(preds), sparse.load_npz(args.graph), nodes.native_component.to_numpy())
    nodes.join(out).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
