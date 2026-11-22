# GBMspace pathology benchmark

Figure 4 compares eight methods on 38 sections from six patients in the GBMspace Visium dataset. Figure S26 compares six GBMZone input configurations on those same 38 sections and pathology scoring positions. From the repository root, run `python analysis/reproduce.py --figures 4 S26 --plot --out outputs/benchmark` to regenerate the numerical summaries. [GBMspace Visium dataset](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-17164)

## Comparator settings

For a refit, set `GBMZONE_BENCH` to an input/output directory. The section manifest `sections.csv` has the columns `sample,patient,site,cohort,nodes,hard`, with `cohort` set to `external`. Each section in `cache/<sample>/` contains a processed log-CPM CSR `log_cpm.npz`, an undirected sparse `adjacency.npz`, `nodes.npz` with `unit_id,hard,y`, and `moments.npz` with the gene-wise first and second moments `mean,second`. The columns of the expression matrices match `common_genes.csv` (`stable_id,eligible_feature`). Hard reference spots have dominant normalized class support ≥0.8. The geometry in `images/<sample>_geometry.npz` contains `unit_id,xy,pixels_per_um`. SpaGCN also uses the registered RGB image `images/<sample>.png`.

The bundled domain workflows use pybanksy 1.3.5, GraphST 1.1.1 or SpaGCN 1.2.7 in their respective environments. Each Seurat fold uses 30 principal components and anchor-based transfer from the other five patients. Its scores support both standalone classification and domain decoding. The bundled DIRAC annotation runner uses sodirac 0.1.8, source pathology labels and unlabeled query expression and adjacency for 100 training epochs.

```bash
export GBMZONE_BENCH=/path/to/benchmark-inputs
python analysis/benchmark/run_baselines.py --method BANKSY
python analysis/benchmark/run_baselines.py --method GraphST
python analysis/benchmark/run_baselines.py --method SpaGCN
python analysis/benchmark/prepare_folds.py
python analysis/benchmark/run_dirac.py AT10 AT13 AT15 AT3 AT4 AT6
# Start the Seurat command in the benchmark working directory containing fold_inputs/.
Rscript /path/to/GBMZone/analysis/benchmark/run_seurat.R
```

The stClinic benchmark jointly fit all 38 sections with the RNA-only unsupervised workflow from stClinic v0.0.10. It used log-normalized RNA for 1,850 genes, native Visium edges and reciprocal five-nearest-neighbor cross-section edges in ten joint principal components. Training used seed 666, 300 pretraining epochs and 400 regularized epochs. Mclust 6.1.4 assigned three domains to the embeddings using the EEE covariance model. Pathology labels enter only the separate domain-to-class mapping and evaluation steps. [stClinic release](https://github.com/cmzuo11/stClinic/tree/v0.0.10)

STELLAR uses closed-set transfer. Each fold uses five reference patients with hard pathology labels, 2,000 genes selected from reference patients, and the unlabeled target patient's native Visium graph. The three reference classes are CT, LE/IT and necrosis-associated tissue. The benchmark used 20 epochs, seed 1 and Adam with learning rate 0.001 and weight decay 0.05. The 38-section benchmark uses predictions from a complete replay of the six folds. On the 19 sections shared with the earlier replay, 93.9% of the hard-labeled spots received the same STELLAR classification. [STELLAR implementation](https://github.com/snap-stanford/stellar/tree/13c81ad41d4db7cc44c3a7d2d6cd5b15c5ad2b2c)

## Score predictions

Supply one table per method with `sample,unit_id,patient,held_out_patient,site,hard,truth`. Use integer truth codes 0, 1 and 2. Include every spot in the section, retaining the `hard` flag for evaluation. For GBMZone, DIRAC, Seurat and STELLAR supply probability columns `CT,LE_IT,necrosis_related`. For BANKSY, GraphST, SpaGCN and stClinic supply the native `domain` and the matching spot-level fold-specific Seurat scores `seurat_CT,seurat_LE_IT,seurat_necrosis_related`.

```bash
python analysis/benchmark/evaluate.py --method BANKSY --spots banksy_all_spots.csv --out outputs/banksy_metrics
```

ARI, AMI and NMI compare native labels on sections with at least two reference classes. For classification, each spatial domain gets the mean Seurat score vector across all its spots, followed by argmax. The patient scores equally weight sites, sections within each site, and labeled spots within each section. A one-versus-rest AUROC requires 20 positive and 20 negative hard-labeled spots. The number of evaluable patients can differ between metrics.
