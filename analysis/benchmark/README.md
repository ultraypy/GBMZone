# GBMspace pathology benchmark

Figure 4 uses the six-patient, 38-section [GBMspace Visium dataset](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-17164). To regenerate the published summaries from the public patient and section measurements, run `python analysis/reproduce.py --figures 4 --plot --out outputs/figure4` from the repository root.

## Refit published comparators

For a refit, set `GBMZONE_BENCH` to an input/output directory. The section manifest `sections.csv` has the columns `sample,patient,site,cohort,nodes,hard`, with `cohort` set to `external`. Each section in `cache/<sample>/` contains a processed log-CPM CSR `log_cpm.npz`, an undirected sparse `adjacency.npz`, `nodes.npz` with `unit_id,hard,y`, and `moments.npz` with the gene-wise first and second moments `mean,second`. The columns of the expression matrices match `common_genes.csv` (`stable_id,eligible_feature`). Hard reference spots have dominant normalized class support ≥0.8. The geometry in `images/<sample>_geometry.npz` contains `unit_id,xy,pixels_per_um`. SpaGCN also uses the registered RGB image `images/<sample>.png`.

The domain methods use pybanksy 1.3.5, GraphST 1.1.1 or SpaGCN 1.2.7 in their respective environments. DIRAC (sodirac 0.1.8) uses PyTorch Geometric and pymetis. GraphST and DIRAC use a GPU. Spatial domains and the two supervised reference-transfer methods run on the same processed sections.

```bash
export GBMZONE_BENCH=/path/to/benchmark-inputs
python analysis/benchmark/run_baselines.py --method BANKSY
python analysis/benchmark/run_baselines.py --method GraphST
python analysis/benchmark/run_baselines.py --method SpaGCN
python analysis/benchmark/prepare_folds.py
python analysis/benchmark/run_dirac.py
# Start the Seurat command in the benchmark working directory containing fold_inputs/.
Rscript /path/to/GBMZone/analysis/benchmark/run_seurat.R
```

Each supervised fold uses five reference patients and holds out the sixth. The per-fold HDF5 includes target expression and graph structure. Pathology labels are present only on the reference side. The source labels and model class probabilities follow CT, LE/IT and necrosis-associated tissue in that order.

## Score predictions

Supply one table per method with `sample,unit_id,patient,held_out_patient,site,hard,truth`. Use integer truth codes 0, 1 and 2. Include every spot in the section, retaining the `hard` flag for evaluation. For GBMZone, DIRAC and Seurat supply probability columns `CT,LE_IT,necrosis_related`. For BANKSY, GraphST and SpaGCN supply the native `domain` and the matching spot-level fold-specific Seurat scores `seurat_CT,seurat_LE_IT,seurat_necrosis_related`.

```bash
python analysis/benchmark/evaluate.py --method BANKSY --spots banksy_all_spots.csv --out outputs/banksy_metrics
```

ARI, AMI and NMI compare native labels on sections with at least two reference classes. For classification, each spatial domain gets the mean Seurat score vector across all its spots, followed by argmax. The patient scores equally weight sites, sections within each site, and labeled spots within each section. A one-versus-rest AUROC requires 20 positive and 20 negative hard-labeled spots. The number of evaluable patients can differ between metrics.
