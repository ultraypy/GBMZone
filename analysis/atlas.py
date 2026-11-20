"""Summarize the frozen cluster identities and reproduce the paired CLR inputs in Figure 1."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", required=True, help="One row per primary-cohort cell: unit_id,patient_id,tissue_role,cluster_id")
    parser.add_argument("--clusters", required=True, help="Tables/S2/cluster_annotation_evidence.csv from Data S1")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cells = pd.read_csv(args.cells, dtype=str)
    if not cells.unit_id.is_unique or cells[["patient_id", "cluster_id"]].isna().any().any():
        raise ValueError("Missing identity or duplicate cell")
    labels = pd.read_csv(args.clusters)[["cluster_id", "fixed_cell_type", "fixed_state"]]
    cells = cells.merge(labels, how="left", on="cluster_id", validate="many_to_one")
    if cells.fixed_cell_type.isna().any():
        raise ValueError("An input cluster is absent from the frozen annotation")
    malignant = ["Malignant_AC", "Malignant_MES", "Malignant_NPC", "Malignant_OPC"]
    lineages = malignant + ["Astrocyte", "Excitatory_neuron", "Inhibitory_neuron", "Myeloid",
                           "Oligodendrocyte", "T_NK", "Vascular_related"]
    cells["identity"] = cells.fixed_cell_type
    use = cells.fixed_cell_type.eq("Malignant_glial_candidate")
    cells.loc[use, "identity"] = "Malignant_" + cells.loc[use, "fixed_state"].str.split("-").str[0]
    if not cells.identity.isin(lineages).all():
        raise ValueError("Unrecognized broad identity in the frozen annotation")
    args.out.mkdir(exist_ok=True, parents=True)
    cells.groupby(["patient_id", "cluster_id"]).size().rename("cells").to_csv(args.out / "cluster_counts.csv")
    subset = cells[cells.tissue_role.isin(["T", "NT"])]
    tab = pd.crosstab([subset.patient_id, subset.tissue_role], subset.identity).reindex(columns=lineages, fill_value=0)
    paired = tab.index.get_level_values(0).value_counts().loc[lambda counts: counts == 2].index
    tab = tab[tab.index.get_level_values(0).isin(paired)]
    effects = []
    for name, columns in [("whole_capture", lineages), ("within_malignant", malignant)]:
        counts = tab[columns]
        log = np.log(counts + .5)
        clr = log.sub(log.mean(axis=1), axis=0)
        diff = clr.xs("T", level=1) - clr.xs("NT", level=1)
        effects.append(diff.rename_axis(columns="feature").stack().rename("CLR_TminusNT").reset_index().assign(composition=name))
    pd.concat(effects, ignore_index=True).to_csv(args.out / "composition_effects.csv", index=False)


if __name__ == "__main__":
    main()
