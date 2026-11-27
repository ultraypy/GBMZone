"""Reconstruct the Figure 1C marker panel from patient-by-identity summaries in Data S1."""
import numpy as np
import pandas as pd


def summarize(read):
    sampling = read("Figure 1/Figure1C_identity_sampling").sort_values("column")
    markers = read("Figure 1/Figure1C_marker_panel").sort_values("row")
    patients = read("Figure 1/Figure1C_patient_marker_means")
    if sampling.identity_id.duplicated().any() or markers.gene.duplicated().any():
        raise ValueError("Expected a unique entry per cell identity and marker")
    if patients.duplicated(["patient_id", "identity_id", "gene"]).any():
        raise ValueError("Duplicate patient within a marker and cell identity")
    counts = patients[["patient_id", "identity_id", "cells"]].drop_duplicates()
    if counts.duplicated(["patient_id", "identity_id"]).any():
        raise ValueError("Cell counts differ between markers within a patient and identity")
    totals = counts.groupby("identity_id").cells.sum()
    np.testing.assert_array_equal(totals.reindex(sampling.identity_id), sampling.cells)
    for fields in (["T", "NT", "Mixed"], ["frontal", "temporal", "occipital", "Unspecified"]):
        np.testing.assert_array_equal(sampling[fields].sum(axis=1), sampling.cells)

    # The input pools cells from all constituent clusters before applying the minimum.
    eligible = patients.loc[patients.cells >= 20]
    panel = eligible.groupby(["identity_id", "gene"], as_index=False).agg(
        mean_log1p=("mean_log1p", "mean"),
        detection_fraction=("detection_fraction", "mean"),
        n_patients=("patient_id", "nunique"),
    )
    panel = panel.merge(sampling[["identity_id", "identity", "column", "eligible_patients"]],
                        on="identity_id", validate="many_to_one")
    panel = panel.merge(markers[["gene", "row"]], on="gene", validate="many_to_one")
    panel = panel.sort_values(["row", "column"]).reset_index(drop=True)
    if len(panel) != len(sampling) * len(markers) or not panel.n_patients.eq(panel.eligible_patients).all():
        raise ValueError("Incomplete patient coverage for a displayed marker or identity")
    by_gene = panel.groupby("gene").mean_log1p
    sd = by_gene.transform("std", ddof=0)
    if sd.isna().any() or sd.eq(0).any():
        raise ValueError("A marker cannot be standardized across the displayed identities")
    panel["z_score"] = (panel.mean_log1p - by_gene.transform("mean")) / sd
    for field, member in (("z_score", "marker_dotplot_zscore"),
                          ("detection_fraction", "marker_dotplot_detection_fraction")):
        observed = panel.pivot(index="gene", columns="identity", values=field).loc[markers.gene, sampling.identity]
        expected = read("Figure 1/" + member).set_index("gene")
        if expected.index.tolist() != markers.gene.tolist() or expected.columns.tolist() != sampling.identity.tolist():
            raise ValueError(f"Reference row or column order differs in {member}")
        np.testing.assert_allclose(observed, expected, rtol=1e-8, atol=1e-10, err_msg=field)
    return {"fig1c_markers": panel.drop(columns="eligible_patients"), "fig1c_sampling": sampling}


def plot_panel(results, out):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    panel = results["fig1c_markers"]
    sampling = results["fig1c_sampling"].set_index("identity")
    genes = panel.loc[panel.column.eq(1), "gene"].tolist()
    identities = sampling.index.tolist()
    x = np.arange(len(identities))
    palette = {"T": "#C35E72", "NT": "#579BA7", "Mixed": "#C7CCD0",
               "frontal": "#78AEC9", "temporal": "#ED8279", "occipital": "#AD89BA",
               "Unspecified": "#AFAEB0"}
    fig = plt.figure(figsize=(8.4, 7.4), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=(1, 1, 12), width_ratios=(5, 1.45))
    tracks = [("Specimen", ["T", "NT", "Mixed"]),
              ("Lobe", ["frontal", "temporal", "occipital", "Unspecified"])]
    for i, (title, categories) in enumerate(tracks):
        ax = fig.add_subplot(grid[i, 0])
        baseline = np.zeros(len(identities))
        for name in categories:
            fraction = sampling[name] / sampling.cells
            ax.bar(x, fraction, bottom=baseline, color=palette[name], width=.94)
            baseline += fraction.to_numpy()
        ax.set(xlim=(-.5, len(x)-.5), ylim=(0, 1), xticks=[], yticks=[], ylabel=title)
        for spine in ax.spines.values():
            spine.set_visible(False)
    ax = fig.add_subplot(grid[2, 0])
    z = panel.pivot(index="gene", columns="identity", values="z_score").loc[genes, identities]
    tiles = ax.imshow(z, cmap="RdBu_r", vmin=-4, vmax=4, interpolation="nearest", aspect="auto")
    bubble = 110
    ax.scatter(panel.column-1, panel.row-1, s=bubble*panel.detection_fraction,
               color="#263D49", edgecolors="white", linewidths=.25)
    ax.axvline(3.5, color="white", linewidth=2)
    ax.set_xticks(x, identities, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(genes)), genes, fontstyle="italic", fontsize=8)
    ax.tick_params(length=0)
    side = fig.add_subplot(grid[:, 1])
    side.set_axis_off()
    handles = [Patch(facecolor=color, label=key.capitalize() if key.islower() else key)
               for key, color in palette.items()]
    first = side.legend(handles=handles, loc="upper left", frameon=False,
                        title="Specimen and lobe", fontsize=8)
    side.add_artist(first)
    cax = side.inset_axes([.10, .38, .13, .24])
    fig.colorbar(tiles, cax=cax, ticks=[-4, 0, 4], label="Gene Z-score")
    dots = [Line2D([], [], marker="o", linestyle="", color="#263D49",
                   markersize=np.sqrt(bubble*f), label=f"{100*f:.0f}") for f in (.25, .5, 1)]
    side.legend(handles=dots, loc="lower left", title="Detected (%)", frameon=False, fontsize=8)
    fig.savefig(out / "fig1c_marker_panel.pdf")
    plt.close(fig)
