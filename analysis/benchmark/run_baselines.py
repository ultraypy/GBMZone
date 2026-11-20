"""Generate label-blind published spatial partitions on registered GBMspace sections."""
import argparse
import contextlib
import gc
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD


import os

WORK = Path(os.environ.get("GBMZONE_BENCH", ".")).resolve()
OUT = WORK
CACHE = WORK / "cache"
HISTOLOGY = WORK / "images"
SECTIONS = WORK / "sections.csv"
SEED = 1130
N_FEATURES = 2000
K = 3
METHODS = ("BANKSY", "GraphST", "SpaGCN")



def section_table():
    rows = pd.read_csv(SECTIONS).query('cohort == "external"').copy()
    assert len(rows) == 38 and rows.patient.nunique() == 6 and rows["sample"].is_unique
    return rows


def load_unsupervised(sample):
    from anndata import AnnData
    path = CACHE / sample
    with np.load(path / "nodes.npz") as node, np.load(HISTOLOGY / f"{sample}_geometry.npz") as he:
        ids = node["unit_id"]
        image_xy = he["xy"]
        np.testing.assert_array_equal(ids, he["unit_id"])
        pixels_per_um = float(he["pixels_per_um"])
    expr = sparse.load_npz(path / "log_cpm.npz").tocsr()
    assert expr.shape[0] == len(ids) and np.isfinite(expr.data).all()
    means = np.asarray(expr.mean(axis=0)).ravel()
    variation = np.asarray(expr.power(2).mean(axis=0)).ravel() - means**2
    eligible = np.flatnonzero((expr.getnnz(axis=0) >= 5) & (variation > 0))
    selected = eligible[np.argsort(-variation[eligible], kind="stable")[:N_FEATURES]]
    # Give the algorithms expression and registration only. Labels are loaded in a separate script.
    obs = pd.DataFrame(image_xy / pixels_per_um, index=ids.astype(str), columns=["x", "y"])
    data = AnnData(expr[:, selected].astype(np.float32), obs=obs)
    data.obsm["spatial"] = obs.to_numpy()
    return data, selected, image_xy, pixels_per_um


def cluster(embedding):
    assert np.isfinite(embedding).all()
    return KMeans(n_clusters=K, n_init=20, random_state=SEED).fit_predict(embedding)


def banksy(data):
    from banksy.initialize_banksy import initialize_banksy
    from banksy.embed_banksy import generate_banksy_matrix
    from matplotlib import pyplot as plt

    weights = initialize_banksy(data, coord_keys=("x", "y", "spatial"), num_neighbours=15,
                                nbr_weight_decay="scaled_gaussian", max_m=1, plt_edge_hist=False,
                                plt_nbr_weights=False, plt_agf_angles=False, plt_theta=False)
    _, augmented = generate_banksy_matrix(data, weights, lambda_list=[.8], max_m=1,
                                         plot_std=False, save_matrix=False, verbose=False)
    plt.close("all")
    embedding = TruncatedSVD(n_components=30, n_iter=5, random_state=SEED).fit_transform(augmented.X)
    return cluster(embedding), {"lambda": .8, "k_geom": 15, "max_m": 1, "embedding_dimensions": 30}


def graphst(data):
    import torch
    from GraphST import GraphST

    assert torch.cuda.is_available()
    data.var["highly_variable"] = True
    model = GraphST.GraphST(data, device=torch.device("cuda:0"), datatype="10X",
                           epochs=600, dim_output=64, random_seed=SEED)
    fitted = model.train()
    embedding = PCA(n_components=20, random_state=SEED).fit_transform(fitted.obsm["emb"])
    labels = cluster(embedding)
    del model, fitted
    torch.cuda.empty_cache()
    return labels, {"epochs": 600, "embedding_dimensions": 20, "author_backend": "10X"}


def spagcn(data, sample, image_xy, pixels_per_um):
    from PIL import Image
    import SpaGCN as spg

    with Image.open(HISTOLOGY / f"{sample}.png") as original:
        image = np.asarray(original.convert("RGB"))
    col, row = np.rint(image_xy).astype(int).T
    assert (col >= 0).all() and (row >= 0).all()
    assert (col < image.shape[1]).all() and (row < image.shape[0]).all()
    x, y = data.obsm["spatial"].T
    beta = max(3, int(round(55 * pixels_per_um)))
    distance = spg.calculate_adj_matrix(x=list(x), y=list(y), x_pixel=list(row), y_pixel=list(col),
                                       image=image, beta=beta, alpha=1, histology=True)
    length = spg.search_l(.5, distance, start=.01, end=max(1000, float(distance.max())), tol=.01, max_run=100)
    assert length is not None and np.isfinite(length) and length > 0
    # The author's PCA path expects a dense array or the legacy sparse .A attribute.
    data.X = data.X.toarray()
    model = spg.SpaGCN()
    model.set_l(length)
    model.train(data, distance, num_pcs=50, init_spa=True, init="kmeans", n_clusters=K,
                tol=5e-3, lr=.05, max_epochs=200)
    labels, probs = model.predict()
    assert np.isfinite(probs).all()
    return labels, {"epochs_max": 200, "p": .5, "l": float(length), "alpha": 1,
                    "beta_image_px": beta, "pc_count": 50, "histology": True}


def run(method, row):
    target = OUT / "predictions" / method / f"{row.sample}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    start = time.monotonic()
    log = OUT / "logs" / f"{method}_{row.sample}.log"
    random.seed(SEED)
    np.random.seed(SEED)
    if method != "BANKSY":
        import torch
        torch.set_num_threads(4)
        torch.manual_seed(SEED)
    program = {"BANKSY": banksy, "GraphST": graphst, "SpaGCN": spagcn}[method]
    print(f"BEGIN {method} {row.sample} {row.nodes} spots", flush=True)
    with log.open("w") as handle, contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
        try:
            data, genes, xy, scale = load_unsupervised(row.sample)
            labels, params = program(data, row.sample, xy, scale) if method == "SpaGCN" else program(data)
            assert len(labels) == len(data) == row.nodes
            assert np.isfinite(labels).all()
            used = len(np.unique(labels))
            assert 1 <= used <= K
            params.update(seconds=time.monotonic()-start, target_clusters=K, actual_clusters=used, seed=SEED)
            building = target.with_suffix(".build.npz")
            np.savez_compressed(building, unit_id=data.obs_names.to_numpy(str),
                                domain_label=labels.astype(np.int8), features=genes,
                                param_json=json.dumps(params))
            building.replace(target)
        except Exception:
            traceback.print_exc()
            raise
    print(f"DONE  {method} {row.sample} {params['seconds']:.1f}s k={used}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--sample")
    args = parser.parse_args()
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    table = section_table()
    if args.sample is not None:
        table = table.loc[table["sample"].eq(args.sample)]
        assert len(table) == 1
    versions = {name: importlib.metadata.version(name) for name in
                ({"BANKSY": "pybanksy", "GraphST": "GraphST", "SpaGCN": "SpaGCN"}[args.method], "anndata", "numpy", "scikit-learn")}
    versions.update(python=sys.executable)
    (OUT / "logs" / f"{args.method}_environment.json").write_text(json.dumps(versions, indent=2))
    failures = []
    for row in table.sort_values(["nodes", "sample"]).itertuples(index=False):
        try:
            run(args.method, row)
        except Exception as exc:
            failures.append(row.sample)
            print(f"FAILED {args.method} {row.sample} {type(exc).__name__}: {exc}", flush=True)
        finally:
            gc.collect()
    if failures:
        raise RuntimeError(f"{len(failures)} sections failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
