"""Recompute the patient-level results in Figures 1–6 and S26 from Data S1."""
import argparse
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

from regions.paired_tests import FOCAL, estimate_group


@lru_cache(maxsize=20)
def signs(n):
    if not 1 <= n <= 18:
        raise ValueError("Exact enumeration supports 1–18 independent patient pairs")
    bits = (np.arange(2**n)[:, None] >> np.arange(n)) & 1
    return (2 * bits - 1).astype(float)


def flip(x):
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite patient difference")
    return np.mean(np.abs(signs(len(x)) @ x / len(x)) >= abs(x.mean()) - 1e-12)


def adjust(p, method="fdr_bh"):
    return np.where(p.notna(), multipletests(p.fillna(1), method=method)[1], np.nan)


def checked(observed, reference, keys, columns):
    """Match every row by its scientific key, including untested comparisons."""
    merged = observed.merge(reference, on=keys, validate="one_to_one", how="outer",
                            indicator=True, suffixes=("", "_reference"))
    if not merged["_merge"].eq("both").all():
        raise ValueError(f"Unmatched rows in {keys}")
    for col in columns:
        np.testing.assert_allclose(merged[col], merged[col + "_reference"],
                                   rtol=1e-8, atol=1e-10, equal_nan=True, err_msg=col)


def composition(read):
    effects = read("Figure 1/composition_effects")
    keys = ["composition", "feature"]
    if effects.duplicated(keys + ["patient_id"]).any():
        raise ValueError("Duplicate patient within a composition contrast")
    rows = []
    for group, g in effects.groupby(keys):
        diff = (g.CLR_T - g.CLR_NT).to_numpy()
        rows.append((*group, len(diff), diff.mean(), flip(diff)))
    out = pd.DataFrame(rows, columns=keys + ["n_patients", "mean_CLR_TminusNT", "p_exact_two_sided"])
    if len(out) != 15:
        raise ValueError("Expected 11 whole-capture and four within-malignant comparisons")
    out["q_BH_15"] = adjust(out.p_exact_two_sided)
    checked(out, read("Figure 1/composition_tests"), keys, list(out.columns[2:]))
    return {"fig1_composition": out}


def metabolism(read):
    out = {}
    keys = ["scope", "module", "first_state", "second_state"]
    for cohort in ("internal", "external"):
        source = read(f"Figure 2/metabolic_{cohort}_patients")
        if source.duplicated(keys + ["patient_id"]).any():
            raise ValueError("Repeated patient within metabolic contrast")
        rows = []
        for group, g in source.groupby(keys):
            pairs = g[g.cell_eligible]
            d = (pairs.second_score - pairs.first_score).to_numpy()
            passed = g.panel_eligible.all() and len(d) >= 6
            rows.append((*group, len(d), d.mean() if len(d) else np.nan, flip(d) if passed else np.nan))
        frame = pd.DataFrame(rows, columns=keys + ["n_patients", "mean_difference", "p_value"])
        scopes = frame.groupby("scope")
        if not scopes.size().eq(462).all():
            raise ValueError("Each metabolic family must include 77 modules × six state contrasts")
        frame["q_BH462"] = scopes.p_value.transform(adjust)
        frame["q_BY462"] = scopes.p_value.transform(lambda p: adjust(p, "fdr_by"))
        ref = read(f"Figure 2/metabolic_{cohort}").rename(columns={"q_BY462_dependence_audit": "q_BY462"})
        checked(frame, ref, keys, ["n_patients", "mean_difference", "p_value", "q_BH462", "q_BY462"])
        out[f"fig2_metabolism_{cohort}"] = frame
    ivy = read("Figure 2/Figure2F_IvyGAP_donor_differences")
    if ivy.duplicated(["program", "donor_id"]).any():
        raise ValueError("Duplicate anatomical donor within program")
    rows = []
    for program, g in ivy.groupby("program"):
        d = (g.CT_score - g.LE_score).to_numpy()
        rows.append((program, len(d), d.mean(), flip(d)))
    prog = pd.DataFrame(rows, columns=["program", "patients", "effect", "p"])
    prog["q"] = adjust(prog.p)
    checked(prog, read("Figure 2/program_ivy"), ["program"], ["patients", "effect", "p", "q"])
    out["fig2_ivy_programs"] = prog
    return out


def spatial(read):
    out = {}
    for cohort in ("internal", "external"):
        source = read(f"Figure 3/spatial_{cohort}_patients")
        keys = ["model", "program"] if cohort == "internal" else ["program"]
        ref = read(f"Figure 3/spatial_{cohort}_tests")
        if cohort == "internal":
            source["effect"] = source.observed_I - source.null_mean_I
            np.testing.assert_allclose(source.effect, source.delta_I, atol=1e-12)
            pcol, qcol, method = "exact_patient_sign_p", "q_patient_all16", "fdr_bh"
            id_col = "patient"
        else:
            source["effect"] = source.patient_mean_delta_I
            pcol, qcol, method = "exact_two_sided_patient_sign_p", "q_BY8_primary", "fdr_by"
            id_col = "patient_id"
        if source.duplicated(keys + [id_col]).any():
            raise ValueError("Duplicate patient within spatial comparison")
        rows = []
        for key, g in source.groupby(keys[0] if len(keys) == 1 else keys):
            if len(keys) == 1:
                key = (key,)
            d = g.effect.dropna()
            nonzero = d[d.abs() > 1e-12]
            p = binomtest((nonzero > 0).sum(), len(nonzero)).pvalue if len(d) >= 6 and len(nonzero) else np.nan
            rows.append((*key, d.mean(), p))
        frame = pd.DataFrame(rows, columns=keys + ["mean_delta_I", pcol])
        frame[qcol] = adjust(frame[pcol], method)
        checked(frame, ref, keys, ["mean_delta_I", pcol, qcol])
        out[f"fig3_spatial_{cohort}"] = frame
    return out


def benchmark(read):
    scores = read("Figure 4/per_section_partition_agreement")
    metrics = ["ari", "ami", "nmi"]
    by_site = scores[scores.eligible].groupby(["method", "patient", "site"])[metrics].mean()
    by_patient = by_site.groupby(["method", "patient"]).mean()
    partition = by_patient.groupby("method").agg(["mean", "std"])
    partition.columns = ["_".join(k) for k in partition]
    partition["patients"] = by_patient.groupby("method").size()
    partition = partition.reset_index()
    checked(partition, read("Figure 4/Fig4_b_partition_summary"), ["method"], list(partition.columns[1:]))
    per_patient = read("Figure 4/Fig4_cd_patient_metrics")
    if per_patient.duplicated(["method", "patient"]).any():
        raise ValueError("Duplicate benchmark patient")
    performance = per_patient.groupby("method")[["macro_f1", "macro_auroc"]].agg(["mean", "std", "count"])
    performance.columns = ["_".join(k) for k in performance]
    performance = performance.reset_index()
    checked(performance, read("Figure 4/Fig4_cd_score_summary"), ["method"], list(performance.columns[1:]))
    return {"fig4_partition": partition, "fig4_classification": performance}


def ablation(read):
    patients = read("Tables/S6/configuration_patient_metrics")
    keys = ["group", "modality"]
    metrics = ["macro_f1", "macro_auroc", "soft_brier", "ari", "ami", "nmi"]
    if patients.duplicated(keys + ["patient"]).any():
        raise ValueError("Duplicate patient within a GBMZone configuration")
    means = patients.groupby(keys, as_index=False)[metrics].mean()
    checked(means, read("Tables/S6/configuration_cohort_metrics"), keys, metrics)
    return {"figS26_configuration_means": means}


def regional_genes(read):
    directory = "Tables/S7/"
    methods = ["rna_pathway_bin100", "rna_pathway_cellbin", "rna_bin100", "rna_cellbin"]
    primary = read(directory + methods[0] + "_all_genes").set_index("stable_id", verify_integrity=True)
    table = primary[["gene", "log2FoldChange", "padj"]].copy()
    table["four_internal"] = table.padj.lt(.05)
    for name in methods[1:]:
        sensitivity = read(directory + name + "_all_genes").set_index("stable_id", verify_integrity=True)
        table = table.join(sensitivity[["log2FoldChange", "padj"]].add_prefix(name + "_"))
        same = table.log2FoldChange * table[name + "_log2FoldChange"] > 0
        table["four_internal"] &= same & table[name + "_padj"].lt(.05)
    ivy = read(directory + "IvyGAP_all_genes").set_index("stable_id", verify_integrity=True)
    table = table.join(ivy[["mean_log1p_FPKM_difference", "q_all_measured_noninput"]])
    same = table.log2FoldChange * table.mean_log1p_FPKM_difference > 0
    table["primary_and_Ivy"] = same & table.padj.lt(.05) & table.q_all_measured_noninput.lt(.05)
    table["four_and_Ivy"] = table.four_internal & table.primary_and_Ivy
    selected = read(directory + "selected_gene_evidence")
    subset = table.loc[selected.stable_id].reset_index()
    checked(subset, selected, ["stable_id"], ["log2FoldChange", "padj"])
    ref = read("Figures 5 and 6/internal_Ivy_score_comparison")
    section = read("Figures 5 and 6/bin100_patient_score_effects")
    section = section[section.method.eq("pathway_tran") & section.adjustment.eq("technical")]
    ivy_pairs = read("Figures 5 and 6/Ivy_patient_effects")
    panels = []
    for cohort, source in [("own", section), ("Ivy", ivy_pairs)]:
        rows = []
        if source.duplicated(["family", "panel", "patient"]).any():
            raise ValueError("Repeated patient for a regional panel")
        for (family, panel), g in source.groupby(["family", "panel"]):
            d = g.effect.dropna().to_numpy()
            rows.append((family, panel, len(d), d.mean() if len(d) else np.nan,
                         flip(d) if len(d) >= 6 else np.nan))
        scores = pd.DataFrame(rows, columns=["family", "panel", "n_patients", "mean_effect", "p"])
        scores["q_family"] = scores.groupby("family").p.transform(adjust)
        numeric = ["n_patients", "mean_effect", "p", "q_family"]
        checked(scores, ref.rename(columns={c + "_" + cohort: c for c in numeric}),
                ["family", "panel"], numeric)
        panels.append(scores.assign(cohort=cohort))
    return {"fig5_regional_gene_concordance": table.reset_index(),
            "fig5_regional_panels": pd.concat(panels, ignore_index=True)}


def lineage(read):
    rows = []
    keys = ["analysis", "gene", "lineage", "method"]
    for cohort in ["internal", "external"]:
        source = read(f"Tables/S9/{cohort}_patient_gene_coverage")
        if source.duplicated(["analysis", "patient", "gene", "lineage"]).any():
            raise ValueError("Duplicate gene/lineage within patient")
        for analysis, membership in source.groupby("analysis"):
            for gene, cell_type in FOCAL.items():
                g = membership[(membership.gene == gene) & (membership.lineage == cell_type)].copy()
                g["eligible"] = (g[["n_core", "n_edge"]].min(axis=1) >= 50) & (g[["umi_core", "umi_edge"]].min(axis=1) >= 5000)
                used = g.eligible
                g.loc[used, "log2fc"] = (np.log2((g.loc[used, "target_edge"] + .5) / g.loc[used, "umi_edge"])
                                          - np.log2((g.loc[used, "target_core"] + .5) / g.loc[used, "umi_core"]))
                estimates, pairs, umi = estimate_group(g)
                for test in ["paired_t", "exact_sign_flip_mean", "exact_sign", "exact_signed_rank", "count_weighted_REML_mKH"]:
                    effect, lo, hi, p, _ = estimates.get(test, (np.nan,) * 5)
                    rows.append((analysis, gene, cell_type, test, len(pairs), umi, effect, lo, hi, p))
    cols = ["n_patients", "total_target_umi", "estimate_log2", "ci95_low", "ci95_high", "p"]
    frame = pd.DataFrame(rows, columns=keys + cols)
    if not frame.groupby(["analysis", "method"]).size().eq(7).all():
        raise ValueError("Each within-lineage family must contain all seven gene-lineage pairs")
    frame["q"] = frame.groupby(["analysis", "method"]).p.transform(adjust)
    primary = frame[frame.analysis.isin(["pathway_tran_0.60", "snu_main"]) & frame.method.eq("paired_t")]
    checked(primary, read("Tables/S9/primary_within_lineage_tests"), keys, cols + ["q"])
    terms = read("Tables/S9/decomposition_by_lineage")
    ix = ["analysis", "patient", "gene"]
    parts = terms.groupby(ix).composition.sum().to_frame()
    for unresolved, key in [(False, "within_classified"), (True, "within_unresolved")]:
        parts[key] = terms[terms.lineage.eq("Unresolved") == unresolved].groupby(ix).within.sum()
    parts = parts.fillna(0)
    parts["total"] = parts.sum(axis=1)
    parts = parts.reset_index()
    checked(parts, read("Tables/S9/decomposition_by_patient"), ix,
            ["total", "composition", "within_classified", "within_unresolved"])
    return {"fig6_within_lineage": frame, "fig6_decomposition": parts}


def main():
    runners = dict(zip("123456", [composition, metabolism, spatial, benchmark, regional_genes, lineage],
                       strict=True))
    runners["S26"] = ablation
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        help="The manuscript supplementary archive Data S1.zip; optional for Figures 4 and S26")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figures", nargs="+", choices=list(runners), default=list("123456"))
    parser.add_argument("--plot", action="store_true", help="Also render a summary of each result as a PDF")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.data is None and not set(args.figures) <= {"4", "S26"}:
        parser.error("--data is required for Figures 1–3 and 5–6")
    source = ZipFile(args.data) if args.data else nullcontext(None)
    with source as archive:
        def read(member):
            if archive is None:
                figure = "figureS26" if member.startswith("Tables/S6/") else "figure4"
                root = Path(__file__).resolve().parents[1] / "results" / figure / "source-data"
                return pd.read_csv(root / (member.split("/")[-1] + ".csv"))
            with archive.open(member + ".csv") as stream:
                return pd.read_csv(stream)
        for number in dict.fromkeys(args.figures):
            results = runners[number](read)
            if number == "1" and "Figure 1/Figure1C_patient_marker_means.csv" in archive.namelist():
                from figure1_markers import summarize
                results.update(summarize(read))
            for name, frame in results.items():
                frame.to_csv(args.out / (name + ".csv"), index=False)
            if args.plot:
                from summary_plots import plot
                plot(number, results, read, args.out)
                if "fig1c_markers" in results:
                    from figure1_markers import plot_panel
                    plot_panel(results, args.out)
            print(f"Figure {number} complete", flush=True)


if __name__ == "__main__":
    main()
