"""Small, reproducible views of the numerical results. PDF output uses embedded text."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(number, results, read, out):
    plt.rcParams.update({"pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7, 4.2), layout="constrained")
    if number == "1":
        effects = results["fig1_composition"].query("composition == 'whole_capture'").sort_values("mean_CLR_TminusNT")
        ax.barh(effects.feature.str.replace("_", " "), effects.mean_CLR_TminusNT, color="#507e9d")
        ax.set_xlabel("Paired mean CLR difference (T minus NT)")
    elif number == "2":
        modules = read("Figure 2/metabolic_internal")[["module", "subsystem"]].drop_duplicates()
        for cohort, scope, symbol in [("internal", "all_cluster_full", "o"), ("external", "adult20_full", "s")]:
            table = results["fig2_metabolism_" + cohort].merge(modules)
            chosen = table[(table.scope == scope) & (table.first_state == "AC-like") & (table.second_state == "MES-like")]
            chosen = chosen[chosen.subsystem.str.contains("oxidative phosphorylation|citric acid|tricarboxylic", case=False)]
            if chosen.empty:
                raise ValueError("Missing oxidative metabolism panels")
            ax.scatter(chosen.mean_difference, chosen.subsystem, marker=symbol, label=cohort)
        ax.set_xlabel("Mean RNA score difference (MES-like minus AC-like)")
        ax.legend(frameon=False)
    elif number == "3":
        for cohort, marker in [("internal", "o"), ("external", "s")]:
            table = results["fig3_spatial_" + cohort]
            if cohort == "internal":
                table = table[table.model == "composition"]
            ax.scatter(table.mean_delta_I, table.program, marker=marker, label=cohort)
        ax.set_xlabel("Mean patient ΔI (platform-specific neighborhood)")
        ax.legend(frameon=False)
    elif number in {"4", "S26"}:
        if number == "4":
            table = results["fig4_classification"].sort_values("macro_f1_mean")
            labels, f1, auc = table.method, table.macro_f1_mean, table.macro_auroc_mean
        else:
            table = results["figS26_configuration_means"].sort_values("macro_f1")
            labels, f1, auc = table.modality, table.macro_f1, table.macro_auroc
        ax.scatter(f1, labels, label="Macro F1")
        ax.scatter(auc, labels, marker="s", label="Macro AUROC")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Mean across evaluable patients")
        ax.legend(frameon=False)
    elif number == "5":
        genes = ["CA9", "PDPN", "PYGL", "GRM3", "OPALIN", "CNTNAP2", "DPP10"]
        table = results["fig5_regional_gene_concordance"].set_index("gene").loc[genes]
        ax.barh(genes, table.log2FoldChange, color=np.where(table.log2FoldChange > 0, "#c1654f", "#469880"))
        ax.set_xlabel("Regional log2 fold change (CT-like relative to LE/IT-like)")
    else:
        frame = results["fig6_within_lineage"]
        for analysis, offset, color in [("pathway_tran_0.60", -.13, "#3f7296"), ("snu_main", .13, "#c8813c")]:
            d = frame[(frame.analysis == analysis) & (frame.method == "paired_t")].sort_values("gene")
            y = np.arange(len(d)) + offset
            ax.errorbar(d.estimate_log2, y, xerr=[d.estimate_log2 - d.ci95_low, d.ci95_high - d.estimate_log2],
                        fmt="o", label=analysis, color=color)
            ax.set_yticks(np.arange(len(d)), d.gene)
        ax.set_xlabel("Within-lineage difference (edge minus core, log2 scale, 95% CI)")
        ax.legend(frameon=False)
    if number not in {"4", "S26"}:
        ax.axvline(0, linewidth=.8, color="#777", zorder=0)
    fig.savefig(out / f"figure{number}_summary.pdf")
    plt.close(fig)
