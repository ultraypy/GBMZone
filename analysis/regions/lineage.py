"""Donor-held-out lineage transfer on the frozen 1,771-feature panel."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import SGDClassifier

SEED = 20261119

def row_sum(x):
    return np.asarray(x.sum(axis=1)).ravel()

def normalize(x):
    total = row_sum(x)
    z = (sparse.diags(1e4 / np.maximum(1, total), dtype="float32") @ x).tocsr()
    z.data = np.log1p(z.data)
    return z

def thinning(x, full_total, coding_depth, seed):
    rng = np.random.default_rng(seed)
    frac = np.minimum(1, coding_depth / np.maximum(1, full_total))
    z = x.copy()
    z.data = rng.binomial(z.data.astype(np.int64), np.repeat(frac, np.diff(z.indptr))).astype(np.float32)
    z.eliminate_zeros()
    return z

def train(lopo, x, labels, total):
    features = [normalize(x[lopo])]
    for count in [100, 300, 600, 1000]:
        features.append(normalize(thinning(x[lopo], total[lopo], count, SEED+count)))
    model = SGDClassifier(loss="log_loss", alpha=.001, max_iter=1500, tol=1e-4,
                          class_weight="balanced", random_state=SEED)
    model.fit(sparse.vstack(features, format="csr"), np.tile(labels[lopo], len(features)))
    return model

def prediction(model, x):
    p = model.predict_proba(normalize(x))
    j = p.argmax(1)
    confidence = p.max(1)
    confidence[row_sum(x) == 0] = 0
    return model.classes_[j], confidence



def aligned_columns(adata, ids):
    field = "ensembl_gene_ids" if "ensembl_gene_ids" in adata.var else "stable_ids"
    match = pd.Index(adata.var[field].astype(str))
    unique = ~match.duplicated(keep=False)
    positions = pd.Series(np.arange(len(match))[unique], index=match[unique]).reindex(ids)
    if positions.isna().any():
        raise ValueError("Missing or ambiguous frozen classifier/outcome genes")
    return positions.to_numpy(int)


def main():
    import anndata as ad
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="Frozen reference h5ad, coding integer counts")
    parser.add_argument("--reference-meta", required=True, help="CSV: unit_id,patient,lineage,total_coding_counts")
    parser.add_argument("--cellbin", required=True, help="One donor's OMIX cellbin counts")
    parser.add_argument("--assignment", required=True, help="CSV: unit_id,region. Include only units in eligible paired regions")
    parser.add_argument("--patient", required=True)
    parser.add_argument("--confidence", type=float, default=.60, choices=[.60, .75])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    panel = pd.read_csv(Path(__file__).with_name("classifier_features.csv")).stable_id
    outcomes = ["CA9", "PDPN", "PYGL", "GRM3", "OPALIN", "CNTNAP2", "DPP10"]
    ref = ad.read_h5ad(args.reference)
    meta = pd.read_csv(args.reference_meta).set_index("unit_id", verify_integrity=True)
    if set(meta.index) != set(ref.obs_names):
        raise ValueError("Reference count rows and metadata differ")
    meta = meta.loc[ref.obs_names]
    if meta[["patient", "lineage", "total_coding_counts"]].isna().any().any():
        raise ValueError("Reference donor, lineage and full coding depth are required")
    use = meta.patient.ne(args.patient).to_numpy()
    if use.all() or not use.any():
        raise ValueError("Reference must include the matched donor to hold out and other donors for training")
    ref_x = sparse.csr_matrix(ref.X[:, aligned_columns(ref, panel)], dtype=np.float32)
    if (ref_x.data < 0).any() or not np.array_equal(ref_x.data, np.rint(ref_x.data)):
        raise ValueError("Reference must use integer counts")
    if (row_sum(ref_x) > meta.total_coding_counts.to_numpy()).any():
        raise ValueError("Full reference coding depth is smaller than the feature subtotal")
    model = train(use, ref_x, meta.lineage.to_numpy(), meta.total_coding_counts.to_numpy())
    del ref, ref_x
    counts = ad.read_h5ad(args.cellbin, backed="r")
    try:
        if set(counts.obs.donor_code.astype(str)) != {args.patient}:
            raise ValueError("Wrong query patient")
        regions = pd.read_csv(args.assignment).set_index("unit_id", verify_integrity=True)
        if set(regions.region) != {"core", "edge"} or not regions.index.isin(counts.obs_names).all():
            raise ValueError("Regions must contain core and edge cellbins from this donor")
        region = regions.region.reindex(counts.obs_names)
        feature_idx = aligned_columns(counts, panel)
        targets = counts.var.set_index("gene_symbol")
        selected = targets[targets.index.isin(outcomes)]
        if not selected.index.is_unique or set(selected.index) != set(outcomes):
            raise ValueError("Ambiguous or missing target gene")
        targets = selected.loc[outcomes, "ensembl_gene_ids"].astype(str)
        if set(targets) & (set(panel) | set(pd.read_csv(Path(__file__).with_name("model_genes.csv")).stable_id)):
            raise ValueError("Outcome overlaps a classifier input")
        target_idx = aligned_columns(counts, targets)
        batches = []
        for start in range(0, counts.n_obs, 16000):
            stop = min(start + 16000, counts.n_obs)
            use = region.iloc[start:stop].notna().to_numpy()
            if not use.any():
                continue
            x = sparse.csr_matrix(counts.X[start:stop])[use]
            if (x.data < 0).any() or not np.array_equal(x.data, np.rint(x.data)):
                raise ValueError("Query must use integer counts")
            label, conf = prediction(model, x[:, feature_idx].astype(np.float32))
            depth = row_sum(x)
            if (depth <= 0).any():
                raise ValueError("Empty library in eligible cellbin")
            vals = x[:, target_idx].toarray()
            block = pd.DataFrame(vals, columns=outcomes)
            for j, gene in enumerate(outcomes):
                block[gene + "_cp10k_sum"] = vals[:, j] / depth * 1e4
            block["library"], block["n_cells"] = depth, 1
            block["region"] = region.iloc[start:stop].to_numpy()[use]
            block["lineage"] = np.where(conf >= args.confidence, label, "Unresolved")
            block["depth_stratum"] = np.searchsorted([200, 400, 800, 1600], depth, side="right")
            batches.append(block.groupby(["region", "lineage", "depth_stratum"]).sum())
        summed = pd.concat(batches).groupby(level=[0, 1, 2]).sum().reset_index()
        summed["patient"] = args.patient
        summed.to_csv(args.out, index=False)
    finally:
        counts.file.close()


if __name__ == "__main__":
    main()
