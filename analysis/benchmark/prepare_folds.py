"""Build patient-held-out inputs from the common coding-RNA and Visium caches."""
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
from scipy import sparse


from run_baselines import CACHE, WORK, SEED, section_table


GENES = WORK / "common_genes.csv"
N_FEATURES = 2000


def choose_fold_features(folds, rows, genes):
    moments = {}
    for row in rows.itertuples(index=False):
        with np.load(CACHE / row.sample / "moments.npz") as panel:
            moments[row.sample] = tuple(panel[name].astype(float) for name in ("mean", "second"))
    choices = {}
    for fold in folds.itertuples(index=False):
        source = rows.loc[rows.patient.ne(fold.target_patient)]
        expected = set(fold.source_patients.split(", "))
        assert set(source.patient) == expected and len(expected) == 5
        weights = source.nodes.to_numpy() / source.nodes.sum()
        means, seconds = zip(*(moments[name] for name in source["sample"]), strict=True)
        variation = weights @ np.stack(seconds) - (weights @ np.stack(means))**2
        assert len(variation) == len(genes)
        keep = np.flatnonzero(genes.eligible_feature.to_numpy(dtype=bool) & (variation > 0))
        ranked = keep[np.argsort(-variation[keep], kind="stable")[:N_FEATURES]]
        assert len(ranked) == N_FEATURES
        choices[fold.target_patient] = np.sort(ranked)
    return choices


def read_section(row, union):
    folder = CACHE / row.sample
    with np.load(folder / "nodes.npz") as nodes:
        ids = nodes["unit_id"]
        is_reference_label = nodes["hard"].astype(bool)
        classes = nodes["y"].argmax(axis=1).astype(np.int8)
    expr = sparse.load_npz(folder / "log_cpm.npz").tocsr()[:, union]
    adjacency = sparse.load_npz(folder / "adjacency.npz").tocsr()
    assert len(ids) == row.nodes and is_reference_label.sum() == row.hard
    assert len(set(ids)) == len(ids) == expr.shape[0] == adjacency.shape[0]
    assert (adjacency - adjacency.T).nnz == 0 and not adjacency.diagonal().any()
    return expr, adjacency, ids, is_reference_label, classes


def write_csr(group, matrix):
    matrix.sort_indices()
    group.attrs["shape"] = matrix.shape
    for key in ("data", "indices", "indptr"):
        options = {"compression": "gzip", "compression_opts": 1} if key != "data" else {}
        group.create_dataset(key, data=getattr(matrix, key), **options)


def save_partition(group, rows, sections, columns, reference):
    expr, graphs, ids, samples, labels = [], [], [], [], []
    for row in rows.itertuples(index=False):
        X, A, node_ids, hard, y = sections[row.sample]
        keep = np.flatnonzero(hard) if reference else np.arange(len(node_ids))
        expr.append(X[keep, :][:, columns])
        graphs.append(A[keep, :][:, keep])
        ids.extend(node_ids[keep].tolist())
        samples.extend([row.sample]*len(keep))
        if reference:
            labels.extend(y[keep].tolist())
    write_csr(group.create_group("expression"), sparse.vstack(expr, format="csr"))
    write_csr(group.create_group("adjacency"), sparse.block_diag(graphs, format="csr"))
    group.create_dataset("unit_id", data=np.asarray(ids, dtype="S"))
    group.create_dataset("sample", data=np.asarray(samples, dtype="S"))
    if reference:
        assert set(labels) == {0, 1, 2}
        group.create_dataset("label", data=np.asarray(labels, dtype=np.int8))


def main():
    rows = section_table().sort_values("sample")
    patients = sorted(rows.patient.unique())
    folds = pd.DataFrame({"target_patient": patients, "source_patients": [
        ", ".join(p for p in patients if p != target) for target in patients]})
    genes = pd.read_csv(GENES)
    assert set(patients) == set(rows.patient) and len(patients) == 6
    assert genes.stable_id.is_unique
    features = choose_fold_features(folds, rows, genes)
    union = np.unique(np.concatenate(list(features.values())))
    folder = WORK / "fold_inputs"
    folder.mkdir(parents=True, exist_ok=True)
    print(f"Materializing {len(rows)} slices with {len(union)} shared staging columns", flush=True)
    sections = {row.sample: read_section(row, union) for row in rows.itertuples(index=False)}
    for patient in patients:
        path = folder / f"{patient}.h5"
        if path.exists():
            continue
        source = rows.loc[rows.patient.ne(patient)]
        query = rows.loc[rows.patient.eq(patient)]
        assert set(source["sample"]).isdisjoint(query["sample"])
        staged = path.with_suffix(".build.h5")
        chosen = features[patient]
        columns = np.searchsorted(union, chosen)
        with h5py.File(staged, "w") as out:
            out.attrs["held_out_patient"] = patient
            out.attrs["source_patients"] = json.dumps(sorted(source.patient.unique()))
            out.attrs["seed"] = SEED
            out.create_dataset("genes", data=genes.iloc[chosen].stable_id.to_numpy(dtype="S"))
            save_partition(out.create_group("reference"), source, sections, columns, reference=True)
            save_partition(out.create_group("query"), query, sections, columns, reference=False)
        staged.replace(path)
        print(f"READY {patient}: {int(source.hard.sum()):,} reference labels; {query.nodes.sum():,} query spots",
              flush=True)


if __name__ == "__main__":
    main()
