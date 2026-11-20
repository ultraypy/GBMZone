# Manuscript analysis workflows

This directory accompanies *Single-cell profiling and GBMZone-guided spatial mapping of human gliomas from three cerebral lobes*. It includes the patient-level analyses of all six main results and the code for the spatial and reference-based models.

## Start with the published figure data

From a checkout of the repository, use Python 3.10 or 3.11 and install the summary-analysis dependencies. Figure 4 can be run immediately using the six CSV files distributed with the repository.

```bash
python -m pip install -r analysis/requirements.txt
python analysis/reproduce.py --figures 4 --plot --out outputs/figure4
```

For all six figures, use the manuscript supplement named **Data S1.zip**. The script opens CSV files directly from the archive. The output contains recalculated results and six summary PDFs. The journal artwork, including the atlas layout and registered tissue views, is in the manuscript figure files.

```bash
python analysis/reproduce.py --data "Data S1.zip" --plot --out outputs/paper
```

| Figure | Starting observations | Recalculation |
| --- | --- | --- |
| 1 | Paired T and NT single-cell composition by donor | The 11 captured cell identities and four within-malignant proportions are tested together using paired CLR differences, exact sign flips and BH adjustment over 15 tests. |
| 2 | Donor-by-state metabolic RNA scores and IvyGAP donor-by-region program scores | The script recalculates all 77 × 6 state contrasts, retaining untested slots in each correction family, and tests eight programs across paired anatomical specimens. Internal and external scoring scales are retained. |
| 3 | Observed and conditional-null spatial autocorrelation per internal section and external patient effects | Internal ΔI is observed Moran's I minus its conditional null mean. Exact patient sign tests use BH across 16 internal comparisons and BY across eight external programs. |
| 4 | Per-section domain agreement and per-patient classification scores from GBMspace | ARI, AMI and NMI are averaged first within a site and then across sites per patient. Macro F1 and macro AUROC are averaged across evaluable patients for each method. |
| 5 | Four complete regional gene contrasts, IvyGAP gene results and patient-specific program effects | Stable Ensembl IDs link the RNA-plus-pathway and RNA region calls at bin100 and cellbin. The code counts genes with directionally consistent FDR support and re-tests regional programs in each cohort. |
| 6 | Counts and library sizes for each patient, region and lineage, together with lineage-level decomposition terms | Seven gene–lineage comparisons are repeated using paired log2 differences. The code applies the original coverage rules, calculates paired t and sensitivity tests, and sums the lineage contributions to each tissue-level difference. |

Every numerical figure workflow checks its results against the corresponding reference columns in Data S1. Statistics use the patient as the independent observation. A class with no evaluable AUROC and a gene–lineage pair below coverage retain a missing result. The four internal region configurations use the same underlying tissue.

## Working from processed matrices

The in-house single-cell and Stereo-seq coding-count matrices are deposited under [OMIX023673](https://ngdc.cncb.ac.cn/omix/release/OMIX023673). OMIX supplies expression counts and spatial coordinates. The tissue graphs, registered masks, per-cell atlas assignments, discovery-cell membership, GBMZone fold predictions and the annotated single-cell training reference are separate intermediate inputs for the optional workflows below. The statistical reproduction from Data S1 can run independently of those intermediates. The main internal spatial workflow uses RNA and pathway features.

**Atlas and expression programs.** Supply primary-cohort cell IDs with their fixed cluster, donor and specimen role. The frozen cluster annotation is in `Tables/S2/cluster_annotation_evidence.csv` in Data S1. T and NT for this step refer to the single-cell specimen role. For program discovery, supply the archived P85-supported 8,792 discovery cells with coding counts, unique gene symbols and `obs.patient_id`. The published run used cNMF 1.7.1, scanpy 1.10.2, anndata 0.11.4 and scikit-learn 1.4.2. Seurat-v3 HVG selection also requires scikit-misc. Program loadings, selected genes and metabolic target/control memberships are indexed under Table S3 in Data S1.

```bash
python analysis/atlas.py --cells primary_cells.csv --clusters cluster_annotation_evidence.csv --out outputs/atlas
python analysis/programs/discovery.py --counts p85_discovery_cells.h5ad --out outputs/cnmf
```

**Spatial organization.** Supply one section at a time. The node CSV contains `unit_id,x_um,y_um,component` for eligible bin100 centers, program scores `P01` through `P08`, seven disjoint lineage score columns with the prefix `marker_`, `total_counts` and `pct_mito_protein_coding`. The first command checks edges against the registered binary tissue mask and removes isolated bins and very small components. The second fits the technical and composition-adjusted models, then shuffles whole residual rows within tissue-component, depth and mitochondrial strata. Use the section-specific seed from the Figure 3 patient input table.

```bash
python analysis/programs/tissue_graph.py --nodes bin100_scores.csv --mask registered_mask.npz --out outputs/graph
python analysis/programs/spatial.py --nodes outputs/graph/nodes.csv.gz --graph outputs/graph/adjacency.npz --seed 2026111401 --out outputs/spatial.csv
```

**Pathology benchmark.** The six held-out patients and the matched pathology maps accompany [Figure 4](../results/figure4/). Commands for the five published comparators and their input contracts are in the [benchmark guide](benchmark/README.md). Trained model configurations are identified by their [checkpoint hashes](../docs/checkpoints.md).

**Regional gene expression.** Use a separate environment with `analysis/regions/requirements-de.txt` for a count-model refit. Start with six fold predictions for the same nodes from [the GBMZone predictor](../algorithm/predict.py). Bins are retained at mean class probability ≥0.60 with ≥4 matching folds. Regional islands require 20 bins, with ≥100 bins of each class in a retained native component. Region eligibility additionally requires 10,000 coding UMIs and five qualified 500 µm blocks. Each command below operates on a single donor until the joint model fit.

```bash
python analysis/regions/assign.py --folds fold{0..5}.npz --nodes outputs/graph/nodes.csv.gz --graph outputs/graph/adjacency.npz --out bins.csv
python analysis/regions/pseudobulk.py --patient GL01 --matrix GL01-ST01_bin100_protein_coding_counts.h5ad --bins bins.csv --resolution bin100 --out outputs/bin100/GL01
python analysis/regions/pseudobulk.py --patient GL01 --matrix GL01-ST01_cellbin_protein_coding_counts.h5ad --bins bins.csv --resolution cellbin --cellbin-qc registered_eligible_cellbins.csv --out outputs/cellbin/GL01
# Repeat the upstream commands for each donor before fitting each resolution separately.
python analysis/regions/paired_de.py --patient-dirs outputs/bin100/GL* --out outputs/CT_vs_LE.csv
python analysis/regions/ivygap.py --fpkm IvyGAP_coding_FPKM.csv.gz --samples IvyGAP_histology_samples.csv --out outputs/IvyGAP_genes.csv
```

The cellbin eligibility CSV lists deposited unit IDs that passed cellbin QC and fall inside the registered tissue mask. Both region calls and counts must come from the same section and registration. For the IvyGAP command, prepare one row per microdissection with author-annotated CT or LE and one column per uniquely mapped protein-coding Ensembl gene. The script averages log1p FPKM within donor and region and tests all measured non-model genes. The external counts and annotations are described in [GSE107559](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE107559).

**Cellular sources.** The manuscript reference has 16,623 annotated cells. The reference metadata contain the full coding library per cell. The fitted classifier excludes the target donor and uses the shared 1,771-feature panel. The primary lineage confidence cutoff is 0.60. Oligodendrocyte-like cellbins assigned to paired regions contribute to the OPALIN analysis. `region` takes the values `core` and `edge`. The reported expression contrast here is edge minus core. The independent surgical periphery comparison uses the UCSC Multiomic GBM data indexed in the manuscript.

```bash
python analysis/regions/lineage.py --reference reference.h5ad --reference-meta reference_cells.csv --patient GL01 --cellbin GL01-ST01_cellbin_protein_coding_counts.h5ad --assignment outputs/cellbin/GL01/cellbin_regions.csv.gz --out outputs/GL01_lineage.csv
# Combine the output rows for eligible donors into one CSV with a single header.
python analysis/regions/decompose.py pooled_lineage_counts.csv --prefix primary --out outputs/decomposition
python analysis/reproduce.py --data "Data S1.zip" --figures 5 6 --plot --out outputs/regional_results
```

The classifier step was run with scikit-learn 1.7.2, anndata 0.11.4, NumPy 2.0.2, pandas 2.3.3 and SciPy 1.15.2. The decomposition table uses `patient,region,lineage,depth_stratum,n_cells,library` and, for each outcome gene, its summed UMI count and summed per-cell CP10K. Composition and within-lineage contributions add back to the measured tissue contrast. A manuscript example is the mean difference for OPALIN within the oligodendrocyte lineage in seven internal patients and five independently sampled patients.
