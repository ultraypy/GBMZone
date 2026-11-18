# GBMZone

GBMZone predicts glioblastoma tissue regions from spatial transcriptomics and registered hematoxylin and eosin (H&E) images. A two-layer GraphSAGE model uses a within-tissue graph and combines RNA expression with optional GO/KEGG pathway features and frozen ResNet18 image features. It returns probabilities for cellular tumor, leading edge or infiltrating tumor, and necrosis-associated tissue.

The repository contains the model, input loaders, training and inference code. Supply processed spatial inputs and the model reference or checkpoint separately.

## Installation

Use Python 3.10 and run commands from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Select `--device cpu` for CPU execution or `--device cuda` with a CUDA-enabled PyTorch installation.

## Processed inputs

| File | Contents |
| --- | --- |
| `genes.csv` | `stable_id` column in the exact order of the RNA matrix columns and model reference. |
| `cache/<sample>/log_cpm.npz` | SciPy sparse matrix with spatial units as rows and model genes as columns. Use log1p counts per 10,000, with the library denominator computed over all coding genes before subsetting columns. |
| `cache/<sample>/adjacency.npz` | SciPy sparse, nonnegative graph matrix in the same row order. Build edges within the tissue mask and keep separate tissue components disconnected. |
| `cache/<sample>/nodes.npz` | NumPy archive with an aligned `unit_id` array. Training and evaluation also use `y` (three soft class proportions per unit), `mask` (eligible labels), and `hard` (eligible hard labels for held-out evaluation). |
| `features/<sample>.npz` | NumPy archive with the same `unit_id` array and `image_features` of shape `(n_units, 512)`. Required for H&E models. |
| `samples.csv` | Training metadata with `sample`, `patient`, and `site` columns. Use one row per section. |

Save NumPy identifiers as strings or integers so they load without pickle. Class proportions in eligible rows of `y` sum to one and follow the reference's class order. Use the same node order across each section's RNA, graph, labels and image features.

For H&E, `load_encoder` and `extract_he_features` in `algorithm/model.py` embed registered RGB images using the frozen ResNet18 state in the reference bundle. Pass each spatial unit's image coordinates in `(x, y)` order and the image scale in pixels per micrometer. The function samples a 220 µm square on a 224 × 224 grid and returns one vector per supplied center. Registration and tissue graph construction are upstream preparation steps.

## Predict tissue regions

```bash
python algorithm/predict.py \
  --weights 'checkpoints/pathway+he+tran.pt' --fold 0 \
  --genes-csv inputs/genes.csv --cache-dir inputs/cache \
  --features-dir inputs/features --sample section_01 \
  --device cpu --output outputs/section_01.npz
```

Set `--fold` to the fold to use from a trusted checkpoint. Inference uses its fitted preprocessing and writes `probabilities`, `unit_id`, and `classes` to a new `.npz`. For RNA-only and RNA-plus-pathway checkpoints, omit `--features-dir`. Load PyTorch checkpoint bundles only from a trusted source.

## Train on new data

```bash
python algorithm/train.py \
  --reference 'checkpoints/pathway+he+tran.pt' \
  --samples-csv inputs/samples.csv --genes-csv inputs/genes.csv \
  --cache-dir inputs/cache --features-dir inputs/features \
  --train-patients donor_01 donor_02 --validation-patients donor_03 \
  --methods pathway_he_tran --epochs 100 --seed 2026111682 \
  --device cpu --output-dir outputs/training
```

Choose `tran` (RNA), `pathway_tran` (RNA and pathways), `he_tran` (RNA and H&E), or `pathway_he_tran` (all inputs). Training initializes the graph model from scratch and fits scaling on the training patients. It reports held-out patient metrics after fitting. Each output bundle contains one fold at index `0`.

The `--reference` input supplies class order, gene order, pathways, and the optional frozen image encoder. An existing full-model bundle provides these definitions. To construct a reference for new inputs, save a dictionary with `torch.save` containing `classes`, `gene_order` (with an ordered `stable_id` list), `pathways` (pathway metadata), and `folds` (a list with a dictionary containing `mask` and `groups` at index `0`). Use a gene-by-pathway Boolean `mask` with at least one gene per pathway. Give `groups` one entry per pathway with `GO_BP` or `KEGG`. For H&E models, also provide `image_encoder_state` (ImageNet ResNet18 state dictionary) and `image_encoder` (encoder metadata).

The study evaluated six patient-held-out folds across 38 public GBMspace sections. To evaluate a particular patient's published-cohort sections, use that patient's held-out checkpoint fold for all of their sections.
