"""Normal supervised training. Validation patients never enter training/scalers."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '2')
import argparse
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from model import FourModel, METHODS, FILENAMES, normgraph, tgraph
from data import section_weights, load_section, transform


def fit_preprocessing(tab, data, mask, need_image):
    """This function receives training sections only."""
    sw = section_weights(tab)
    mean = sum(sw[s] * np.asarray(data[s]['x'].mean(0)).ravel() for s in tab['sample'])
    second = sum(sw[s] * np.asarray(data[s]['x'].power(2).mean(0)).ravel() for s in tab['sample'])
    sd = np.sqrt(np.maximum(second - mean ** 2, .0025)).astype(np.float32)
    pre = dict(gene_mean=mean.astype(np.float32), gene_sd=sd,
               source_patients=sorted(tab.patient.unique().tolist()),
               normalization='log1p coding CP10K; training-patient scaling; clip +/-5')
    if need_image:
        im = sum(sw[s] * data[s]['image'].mean(0) for s in tab['sample'])
        sec = sum(sw[s] * (data[s]['image'] ** 2).mean(0) for s in tab['sample'])
        pre.update(image_mean=im, image_sd=np.sqrt(np.maximum(sec - im ** 2, .0025)).astype(np.float32))
    else:
        pre.update(image_mean=None, image_sd=None)
    avg = sp.csr_matrix(mask.astype(np.float32) / mask.sum(0))
    pm = np.zeros(mask.shape[1], np.float64); ps = pm.copy()
    for s in tab['sample']:
        x, _ = transform(data[s], pre, False)
        score = (avg.T @ x.T).T
        pm += sw[s] * score.mean(0); ps += sw[s] * (score ** 2).mean(0)
    pre['pathway_sd'] = np.sqrt(np.maximum(ps - pm ** 2, .0025)).astype(np.float32)
    return pre


def evaluate(model, table, cache_dir, features_dir, preprocessing, device):
    """Called after fitting, only for reporting; no loss or parameter updates."""
    from sklearn.metrics import f1_score, roc_auc_score
    rows = []
    for patient, tab in table.groupby('patient'):
        sw = section_weights(tab); all_y = []; all_p = []; all_hard = []; all_soft = []; hw = []; mw = []
        for s in tab['sample']:
            d = load_section(cache_dir, s, model.s1.in_features, features_dir if model.use_image else None)
            x, image = transform(d, preprocessing, model.use_image)
            a = tgraph(normgraph(d['a']), device)
            with torch.inference_mode():
                p = torch.softmax(model(torch.as_tensor(x, device=device), a,
                    torch.as_tensor(image, device=device) if image is not None else None), 1).cpu().numpy()
            n = d['nodes']; hard = n['hard'].astype(bool); soft = n['mask'].astype(bool)
            all_y.append(n['y']); all_p.append(p); all_hard.append(hard); all_soft.append(soft)
            hw.append(np.full(len(p), sw[s] / max(1, hard.sum())))
            mw.append(np.full(len(p), sw[s] / max(1, soft.sum())))
        y = np.concatenate(all_y); p = np.concatenate(all_p); hard = np.concatenate(all_hard); soft = np.concatenate(all_soft)
        hw = np.concatenate(hw); mw = np.concatenate(mw); truth = y.argmax(1)
        auc = [roc_auc_score(truth[hard] == j, p[hard, j], sample_weight=hw[hard])
               for j in range(3) if (truth[hard] == j).sum() >= 20 and (truth[hard] != j).sum() >= 20]
        f1 = f1_score(truth[hard], p[hard].argmax(1), average='macro', labels=np.unique(truth[hard]),
                      sample_weight=hw[hard], zero_division=0) if hard.any() else None
        brier = np.average(((p[soft] - y[soft]) ** 2).sum(1), weights=mw[soft]) if soft.any() else None
        rows.append(dict(patient=patient, macro_auroc=float(np.mean(auc)) if auc else None,
                         macro_f1=float(f1) if f1 is not None else None,
                         soft_brier=float(brier) if brier is not None else None))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, help='Trusted full-model bundle: definitions and frozen ImageNet encoder only; learned GNN weights are NEVER used')
    parser.add_argument('--samples-csv', required=True)
    parser.add_argument('--genes-csv', required=True, help='Gene order of the supplied RNA matrices')
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--features-dir')
    parser.add_argument('--train-patients', nargs='+', required=True)
    parser.add_argument('--validation-patients', nargs='+', default=[])
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=2026111682)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    # Split checks happen before reading labels, creating outputs or initializing a model.
    train_ids = set(args.train_patients); val_ids = set(args.validation_patients)
    if train_ids & val_ids:
        parser.error('Training and validation patients must be disjoint')
    if args.epochs <= 0:
        parser.error('--epochs must be positive')
    table = pd.read_csv(args.samples_csv, dtype={'sample': str, 'patient': str, 'site': str})
    if not {'sample', 'patient', 'site'} <= set(table) or table[['sample', 'patient', 'site']].isna().any().any():
        parser.error('samples-csv requires nonmissing sample, patient and site columns')
    if table['sample'].duplicated().any() or not ((train_ids | val_ids) <= set(table.patient)):
        parser.error('Duplicate samples or unknown patients')
    train = table.loc[table.patient.isin(train_ids)].copy()
    val = table.loc[table.patient.isin(val_ids)].copy()
    if set(train['sample']) & set(val['sample']):
        parser.error('Training and validation sections overlap')
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        parser.error('output-dir must be new or empty; existing weights are never overwritten')
    reference = torch.load(args.reference, map_location='cpu', weights_only=False)
    f = reference['folds'][0]
    mask = np.asarray(f['mask'], bool); groups = f['groups']
    supplied = pd.read_csv(args.genes_csv)['stable_id'].astype(str).tolist()
    if supplied != list(reference['gene_order']['stable_id']) or len(supplied) != mask.shape[0]:
        parser.error('RNA gene order does not match reference definitions')
    need_image = any(m in ('pathway_he_tran', 'he_tran') for m in args.methods)
    if need_image and (not args.features_dir or 'image_encoder_state' not in reference):
        parser.error('Image models require --features-dir and a reference containing the frozen image encoder')
    torch.set_num_threads(2); device = torch.device(args.device)
    data = {s: load_section(args.cache_dir, s, len(supplied), args.features_dir if need_image else None) for s in train['sample']}
    # Validation sections are not loaded at this stage.
    for s, d in data.items():
        n = d['nodes']; use = n['mask'].astype(bool); y = n['y']
        if y.shape != (d['x'].shape[0], 3) or not use.any() or not np.isfinite(y[use]).all() or (y[use] < 0).any() or not np.allclose(y[use].sum(1), 1, atol=1e-4):
            raise ValueError(f'{s}: invalid supervised training labels')
    pre = fit_preprocessing(train, data, mask, need_image)
    assert set(pre['source_patients']) == train_ids and set(pre['source_patients']).isdisjoint(val_ids)
    sw = section_weights(train); names = list(train['sample']); transformed = {s: transform(data[s], pre, need_image) for s in names}
    x = torch.as_tensor(np.concatenate([transformed[s][0] for s in names]), device=device)
    image = torch.as_tensor(np.concatenate([transformed[s][1] for s in names]), device=device) if need_image else None
    y = torch.as_tensor(np.concatenate([np.where(data[s]['nodes']['mask'][:, None], data[s]['nodes']['y'], 0) for s in names]), device=device)
    weight = torch.as_tensor(np.concatenate([data[s]['nodes']['mask'].astype(np.float32) * sw[s] / data[s]['nodes']['mask'].sum() for s in names]), device=device)
    if not np.isclose(float(weight.sum().cpu()), 1):
        raise ValueError('Training loss weights must sum to one')
    a = tgraph(sp.block_diag([normgraph(data[s]['a']) for s in names], format='csr'), device)
    with torch.no_grad():nx = torch.sparse.mm(a, x); ni = torch.sparse.mm(a, image) if need_image else None
    out.mkdir(parents=True, exist_ok=True)
    for method in args.methods:
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.benchmark = False
        model = FourModel(method, mask, pre['pathway_sd'], groups, hidden=24).to(device)
        # Fresh initialization: no GNN state is loaded from the reference bundle.
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.001)
        for epoch in range(args.epochs):
            model.train(); optimizer.zero_grad(set_to_none=True)
            logits = model(x, a, image if model.use_image else None, nx, ni if model.use_image else None)
            loss = (weight * (-y * torch.log_softmax(logits, 1)).sum(1)).sum()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward(); optimizer.step()
            if epoch == 0 or (epoch + 1) % 20 == 0:print(method, epoch + 1, float(loss.detach().cpu()), flush=True)
        model.eval()
        ck = dict(method=method, state={k: v.detach().cpu() for k, v in model.state_dict().items()},
                  mask=mask, path_sd=pre['pathway_sd'], groups=groups, hidden=24, preprocessing=pre,
                  source_patients=sorted(train_ids), training_sections=names,
                  training_design='normal supervised training; no validation labels or sections in fitting',
                  epochs=args.epochs, seed=args.seed)
        bundle = dict(method=method, folds=[ck], classes=reference['classes'], gene_order=reference['gene_order'],
                      pathways=reference['pathways'], architecture='algorithm/model.py:FourModel',
                      description='Fresh normal training, training patients only')
        if model.use_image:
            image_keys = ('image', 'image_coordinates', 'patch_width_um', 'spot_diameter_um',
                          'crop_input_pixels', 'patch_method', 'encoder', 'encoder_url',
                          'encoder_reference', 'encoder_normalization')
            image_policy = {k: reference.get('image_crop_policy', {})[k]
                            for k in image_keys if k in reference.get('image_crop_policy', {})}
            bundle.update(image_encoder_state=reference['image_encoder_state'],
                          image_encoder=reference['image_encoder'], image_crop_policy=image_policy)
        torch.save(bundle, out / FILENAMES[method])
        if len(val):
            print(json.dumps(dict(method=method, validation=evaluate(model, val, args.cache_dir,
                args.features_dir, pre, device)), ensure_ascii=False), flush=True)
        del model, optimizer
        if device.type == 'cuda':torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
