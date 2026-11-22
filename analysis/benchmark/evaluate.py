"""Evaluate native domains and probability-based class transfer on matched hard spots."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, f1_score, roc_auc_score, adjusted_rand_score,
                             adjusted_mutual_info_score, normalized_mutual_info_score)

CLASSES = ["CT", "LE_IT", "necrosis_related"]


def prediction_metrics(patient, method, truth, prob, weights):
    calls = prob.argmax(axis=1)
    cm = confusion_matrix(truth, calls, sample_weight=weights, labels=[0, 1, 2])
    classes = []
    for code, name in enumerate(CLASSES):
        positive = int((truth == code).sum())
        negative = len(truth) - positive
        observed, called = cm[code].sum(), cm[:, code].sum()
        matched = cm[code, code]
        auc = roc_auc_score(truth == code, prob[:, code], sample_weight=weights) if min(positive, negative) >= 20 else np.nan
        classes.append(dict(method=method, patient=patient, category=name, positive=positive, negative=negative,
                            auroc=auc, recall=matched/observed if observed else np.nan,
                            f1=2*matched/(observed+called) if observed else np.nan))
    summary = dict(method=method, patient=patient,
                   macro_f1=f1_score(truth, calls, average="macro", labels=np.unique(truth),
                                     sample_weight=weights, zero_division=0),
                   macro_auroc=pd.Series([row["auroc"] for row in classes]).mean(),
                   hard_nodes=len(truth))
    return summary, classes





def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spots", required=True, help="All spots for one method. See the input contract.")
    parser.add_argument("--method", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    tab = pd.read_csv(args.spots)
    if tab.duplicated(["sample", "unit_id"]).any():
        raise ValueError("Repeated spot within section")
    if tab.groupby("sample")[["patient", "site"]].nunique().ne(1).any().any():
        raise ValueError("A section must have one patient and site")
    if not set(tab.hard.unique()) <= {False, True}:
        raise ValueError("hard must be Boolean")
    if not (tab.patient == tab.held_out_patient).all():
        raise ValueError("Predicted section does not match the held-out donor")
    if args.method in {"BANKSY", "SpaGCN", "GraphST", "stClinic"}:
        seurat = ["seurat_" + k for k in CLASSES]
        tab[CLASSES] = tab.groupby(["sample", "domain"])[seurat].transform("mean").to_numpy()
    else:
        tab["domain"] = tab[CLASSES].to_numpy().argmax(axis=1)
    probs = tab[CLASSES].to_numpy()
    if not np.isfinite(probs).all() or np.any(probs < 0) or not np.allclose(probs.sum(1), 1, atol=2e-6):
        raise ValueError("Invalid class probabilities")
    hard = tab.loc[tab.hard].copy()
    if hard.empty or not hard.truth.isin([0, 1, 2]).all():
        raise ValueError("Missing or invalid hard pathology calls")
    if set(hard["sample"]) != set(tab["sample"]):
        raise ValueError("Each benchmark section must contain a hard pathology call")
    partitions, patients, classes = [], [], []
    for (patient, site, sample), g in hard.groupby(["patient", "site", "sample"]):
        truth, domains = g.truth, g.domain
        partitions.append(dict(patient=patient, site=site, sample=sample, method=args.method,
                               eligible=truth.nunique() >= 2, hard_spots=len(g),
                               ari=adjusted_rand_score(truth, domains),
                               ami=adjusted_mutual_info_score(truth, domains, average_method="arithmetic"),
                               nmi=normalized_mutual_info_score(truth, domains, average_method="arithmetic")))
    for patient, g in hard.groupby("patient"):
        sites = g.site.nunique()
        sections_per_site = g.groupby("site")["sample"].transform("nunique")
        n_spots = g.groupby("sample")["unit_id"].transform("size")
        w = (1 / (sites * sections_per_site * n_spots)).to_numpy()
        np.testing.assert_allclose(w.sum(), 1)
        summary, detail = prediction_metrics(patient, args.method, g.truth.to_numpy(), g[CLASSES].to_numpy(), w)
        patients.append(summary)
        classes.extend(detail)
    args.out.mkdir(exist_ok=True, parents=True)
    for name, rows in [("section", partitions), ("patient", patients), ("class", classes)]:
        pd.DataFrame(rows).to_csv(args.out / (name + ".csv"), index=False)


if __name__ == "__main__":
    main()
