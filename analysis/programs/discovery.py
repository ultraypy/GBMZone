"""cNMF discovery on the frozen P85-supported, patient-balanced cell set."""
import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from cnmf import cNMF
from scipy import sparse


def run(counts, out):
    a = ad.read_h5ad(counts)
    x = sparse.csr_matrix(a.X, dtype=np.float64)
    if (x.data < 0).any() or not np.array_equal(x.data, np.rint(x.data)):
        raise ValueError("Provide protein-coding integer counts")
    if not a.var_names.is_unique or not a.obs_names.is_unique or a.obs.patient_id.isna().any():
        raise ValueError("Provide unique gene symbols, unique units and known patient IDs")
    detected = np.asarray((x > 0).sum(axis=0)).ravel()
    coverage = np.zeros(a.n_vars, dtype=int)
    for _, ix in a.obs.groupby("patient_id", observed=True).indices.items():
        coverage += np.asarray((x[ix] > 0).sum(axis=0)).ravel() >= 5
    eligible = ((detected >= max(30, np.ceil(.01 * a.n_obs))) & (coverage >= 3)
                & ~a.var_names.str.upper().str.startswith(("MT-", "RPL", "RPS")))
    hv = a[:, eligible].copy()
    sc.pp.highly_variable_genes(hv, flavor="seurat_v3", n_top_genes=2000, batch_key="patient_id")
    out.mkdir(parents=True, exist_ok=True)
    genes_file = out / "discovery_hvgs.txt"
    pd.Series(hv.var_names[hv.var.highly_variable]).to_csv(genes_file, header=False, index=False)
    ks = list(range(3, 13))
    model = cNMF(output_dir=str(out), name="global")
    model.prepare(counts_fn=str(counts), components=ks, n_iter=30, seed=1130,
                  genes_file=str(genes_file), max_NMF_iter=600)
    model.factorize(worker_i=0, total_workers=1)
    model.combine(components=ks, skip_missing_files=False)
    rows = []
    for k in ks:
        s = model.consensus(k=k, density_threshold=.5, show_clustering=False, build_ref=False,
                            skip_density_and_return_after_stats=True)
        rows.append({"K": k, **s.iloc[:, 0].to_dict()})
    stats = pd.DataFrame(rows)
    def scaled(y):
        return (y - y.min()) / max(y.max() - y.min(), 1e-12)
    stats["selection_score"] = (scaled(stats.silhouette) + scaled(-stats.prediction_error)) / 2
    k = int(stats.loc[stats.selection_score >= stats.selection_score.max() - .02, "K"].min())
    stats["selected"] = stats.K.eq(k)
    stats.to_csv(out / "rank_selection.csv", index=False)
    model.consensus(k=k, density_threshold=.5, show_clustering=False, build_ref=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    run(args.counts, args.out)
