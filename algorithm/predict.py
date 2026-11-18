"""Predict with one explicitly chosen fold; no fitting or label use."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '2')
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from model import load_fold, normgraph, tgraph
from data import load_section, transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--fold', type=int, required=True, help='Explicit fold index: existing bundles0..5; newly trained bundles0')
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--features-dir')
    parser.add_argument('--genes-csv', required=True)
    parser.add_argument('--sample', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(); torch.set_num_threads(2)
    output = Path(args.output)
    if output.suffix != '.npz' or output.exists():
        parser.error('--output must be a new .npz file')
    if args.fold < 0:
        parser.error('--fold must be nonnegative')
    device = torch.device(args.device)
    model, pre, bundle = load_fold(args.weights, args.fold, device)
    if pd.read_csv(args.genes_csv)['stable_id'].astype(str).tolist() != list(bundle['gene_order']['stable_id']):
        parser.error('RNA gene order differs from the model')
    if model.use_image and not args.features_dir:
        parser.error('This model requires --features-dir')
    d = load_section(args.cache_dir, args.sample, model.s1.in_features, args.features_dir if model.use_image else None)
    x, image = transform(d, pre, model.use_image)
    a = tgraph(normgraph(d['a']), device)
    with torch.inference_mode():
        p = torch.softmax(model(torch.as_tensor(x, device=device), a,
            torch.as_tensor(image, device=device) if image is not None else None), 1).cpu().numpy()
    if not np.isfinite(p).all() or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError('Invalid predicted probabilities')
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, probabilities=p, unit_id=d['nodes']['unit_id'], classes=np.array(bundle['classes']))
    print(output)


if __name__ == '__main__':
    main()
