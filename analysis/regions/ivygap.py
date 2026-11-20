"""Repeat the donor-paired IvyGAP gene analysis on author-annotated CT and LE samples."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fpkm", required=True, help="CSV: sample_id, then uniquely mapped coding Ensembl IDs")
    parser.add_argument("--samples", required=True, help="CSV: sample_id,donor_id,region. CT or LE")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    x = pd.read_csv(args.fpkm).set_index("sample_id", verify_integrity=True)
    sample = pd.read_csv(args.samples).set_index("sample_id", verify_integrity=True)
    x = x.loc[sample.index]
    exclude = pd.read_csv(Path(__file__).with_name("model_genes.csv")).stable_id
    x = x.loc[:, ~x.columns.isin(exclude)]
    if not np.isfinite(x.to_numpy()).all() or (x < 0).any().any():
        raise ValueError("Provide nonnegative FPKM for all samples")
    means = np.log1p(x).groupby([sample.donor_id, sample.region]).mean()
    core, edge = (means.xs(v, level=1) for v in ("CT", "LE"))
    donors = core.index.intersection(edge.index)
    if not 1 <= len(donors) <= 18:
        raise ValueError("Exact enumeration requires 1–18 paired donors")
    d = (core.loc[donors] - edge.loc[donors]).to_numpy()
    n = len(donors)
    signs = 2 * ((np.arange(2**n)[:, None] >> np.arange(n)) & 1) - 1
    effect = d.mean(axis=0)
    p = np.concatenate([np.mean(abs(signs @ block / n) >= abs(block.mean(axis=0)) - 1e-12, axis=0)
                        for block in np.array_split(d, max(1, int(np.ceil(d.shape[1] / 256))), axis=1)])
    result = pd.DataFrame({"stable_id": x.columns, "mean_log1p_FPKM_difference": effect, "p": p,
                           "n_patients": n, "n_positive": (d > 0).sum(axis=0), "n_negative": (d < 0).sum(axis=0),
                           "q_all_measured_noninput": multipletests(p, method="fdr_bh")[1]})
    result.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
