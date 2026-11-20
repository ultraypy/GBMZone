"""Use registered coordinates to aggregate coding counts within each patient's two regions."""
import argparse

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def calculate(matrix, bins, resolution, cellbin_qc=None):
    xy = np.asarray(matrix.obsm["spatial"]) * float(matrix.uns["um_per_coordinate"])
    if xy.shape != (matrix.n_obs, 2) or not np.isfinite(xy).all():
        raise ValueError("Expected registered spatial coordinates in the count matrix")
    # Deposited coordinates are DNB bin origins or cell centroids, with a scale of 0.5 µm.
    grid_um = 50.
    bin_index = (bins[["x_um", "y_um"]].to_numpy() - grid_um / 2) / grid_um
    if not np.allclose(bin_index, np.rint(bin_index)):
        raise ValueError("Bin centers do not match the registered 50 µm grid")
    index = pd.MultiIndex.from_arrays(np.rint(bin_index).astype(int).T)
    if not index.is_unique:
        raise ValueError("Duplicate bin centers")
    ix = index.get_indexer(pd.MultiIndex.from_arrays(np.floor(xy / grid_um).astype(int).T))
    inside = ix >= 0
    if resolution == "cellbin":
        if cellbin_qc is None or len(cellbin_qc) != matrix.n_obs:
            raise ValueError("Cellbins require the registered cell-level eligibility mask")
        inside &= cellbin_qc
    assignments = np.full(matrix.n_obs, -1, dtype=int)
    assignments[inside] = bins.analysis_label.to_numpy(int)[ix[inside]]
    plotted_xy = xy + grid_um / 2 if resolution == "bin100" else xy
    # Retain each full coding library as the depth denominator, including unmapped features.
    counts = np.zeros((2, matrix.n_vars), dtype=np.int64)
    for begin in range(0, matrix.n_obs, 8192):
        end = min(begin + 8192, matrix.n_obs)
        x = sparse.csr_matrix(matrix.X[begin:end])
        if np.any(x.data < 0) or not np.array_equal(x.data, np.rint(x.data)):
            raise ValueError("X must contain nonnegative integer counts")
        x = x.astype(np.int64)
        for k in (0, 1):
            counts[k] += np.asarray(x[assignments[begin:end] == k].sum(axis=0)).ravel()
    rows = []
    for k, label in enumerate(["CT", "LE_IT"]):
        use = assignments == k
        block = pd.DataFrame(np.floor(plotted_xy[use] / 500).astype(int), columns=["x", "y"])
        block["component"] = bins.native_component.to_numpy()[ix[use]]
        n_blocks = int(block.value_counts().ge(5 if resolution == "bin100" else 20).sum())
        umi = int(counts[k].sum())
        rows.append(dict(region=label, units=int(use.sum()), qualified_blocks=n_blocks,
                         library=umi, eligible=umi >= 10000 and n_blocks >= 5))
    return counts, pd.DataFrame(rows), assignments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, help="One OMIX coding-count h5ad")
    parser.add_argument("--bins", required=True, help="Output from assign.py")
    parser.add_argument("--resolution", choices=["bin100", "cellbin"], required=True)
    parser.add_argument("--cellbin-qc", help="CSV of eligible and inside-mask cellbin unit_id values")
    parser.add_argument("--patient", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from pathlib import Path
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    matrix = ad.read_h5ad(args.matrix, backed="r")
    try:
        if set(matrix.obs.donor_code.astype(str)) != {args.patient} or str(matrix.uns["resolution"]) != args.resolution:
            raise ValueError("Patient or resolution does not match the deposited count file")
        cellbin_qc = None
        if args.resolution == "cellbin":
            if not args.cellbin_qc:
                parser.error("--cellbin-qc is required for cellbin")
            eligible = pd.read_csv(args.cellbin_qc, dtype={"unit_id": str}).unit_id
            if not eligible.is_unique or not eligible.isin(matrix.obs_names).all():
                raise ValueError("Cellbin eligibility list does not match this matrix")
            cellbin_qc = matrix.obs_names.isin(eligible)
        counts, meta, labels = calculate(matrix, pd.read_csv(args.bins), args.resolution, cellbin_qc)
        if args.resolution == "cellbin":
            take = (labels >= 0) & meta.eligible.all()
            pd.DataFrame({"unit_id": matrix.obs_names[take], "region": np.where(labels[take] == 0, "core", "edge")}).to_csv(
                out / "cellbin_regions.csv.gz", index=False)
        ids = matrix.var["ensembl_gene_ids"].fillna("").astype(str).str.strip()
        good = ids.str.fullmatch(r"ENSG\d+(?:\.\d+)?")
        ids = ids.str.replace(r"\.\d+$", "", regex=True)
        good &= ~ids.duplicated(keep=False)
        frame = pd.DataFrame(counts[:, good], index=[args.patient + "_CT", args.patient + "_LE_IT"], columns=ids[good])
        frame.to_csv(out / "counts.csv.gz", index_label="sample_id")
        meta.insert(0, "sample_id", frame.index)
        meta["patient"] = args.patient
        meta.to_csv(out / "samples.csv", index=False)
    finally:
        matrix.file.close()


if __name__ == "__main__":
    main()
