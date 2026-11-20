"""Fit the manuscript patient + region count model with PyDESeq2 0.4.12."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cpus", type=int, default=4)
    args = parser.parse_args()
    matrices, samples = [], []
    for folder in args.patient_dirs:
        meta = pd.read_csv(folder / "samples.csv").set_index("sample_id", verify_integrity=True)
        if len(meta) != 2 or set(meta.region) != {"CT", "LE_IT"} or meta.patient.nunique() != 1:
            raise ValueError(f"Incomplete patient pair in {folder.name}")
        if not meta.eligible.all():
            continue
        x = pd.read_csv(folder / "counts.csv.gz", index_col="sample_id")
        if not x.index.is_unique or set(meta.index) != set(x.index):
            raise ValueError("Count rows and samples differ")
        matrices.append(x.loc[meta.index])
        samples.append(meta)
    if len(samples) < 6:
        raise ValueError("At least six eligible patient pairs are required")
    data = pd.concat(matrices, join="inner", verify_integrity=True).sort_index(axis=1)
    meta = pd.concat(samples, verify_integrity=True).loc[data.index].copy()
    if meta.patient.nunique() != len(samples):
        raise ValueError("Duplicate patient across inputs")
    x = data.to_numpy()
    if not np.isfinite(x).all() or (x < 0).any() or not np.array_equal(x, np.rint(x)):
        raise ValueError("Counts must be nonnegative integers")
    model_inputs = pd.read_csv(Path(__file__).with_name("model_genes.csv")).stable_id
    keep = ~data.columns.isin(model_inputs) & data.ge(10).sum(axis=0).ge(3)
    meta["role"] = meta.region.map({"CT": "CT", "LE_IT": "LE"})
    inference = DefaultInference(n_cpus=args.cpus)
    dds = DeseqDataSet(counts=data.loc[:, keep].astype(np.int64), metadata=meta[["patient", "role"]],
                       design_factors=["patient", "role"], ref_level=["role", "LE"],
                       refit_cooks=True, inference=inference, quiet=True)
    dds.deseq2()
    design = dds.obsm["design_matrix"]
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("Rank-deficient paired design")
    stats = DeseqStats(dds, contrast=["role", "CT", "LE"], alpha=.05,
                       cooks_filter=True, independent_filter=True, inference=inference, quiet=True)
    stats.summary()
    result = stats.results_df.copy()
    result["lfc_low95"] = result.log2FoldChange - 1.96 * result.lfcSE
    result["lfc_high95"] = result.log2FoldChange + 1.96 * result.lfcSE
    result.to_csv(args.out, index_label="stable_id")


if __name__ == "__main__":
    main()
