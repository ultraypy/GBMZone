"""Processed-input helpers. No pathway screening or target-label selection."""
from pathlib import Path
import numpy as np
import scipy.sparse as sp


def section_weights(tab):
    """Equal patients -> equal sites -> equal sections."""
    result = {}
    for _, patient in tab.groupby('patient'):
        for _, site in patient.groupby('site'):
            for sample in site['sample']:
                result[sample] = 1 / (tab.patient.nunique() * patient.site.nunique() * len(site))
    if not np.isclose(sum(result.values()), 1):
        raise ValueError('Empty or invalid training sample table')
    return result


def load_section(cache_dir, sample, n_genes, features_dir=None):
    folder = Path(cache_dir) / sample
    x = sp.load_npz(folder / 'log_cpm.npz').astype(np.float32).tocsr()
    a = sp.load_npz(folder / 'adjacency.npz').astype(np.float32).tocsr()
    with np.load(folder / 'nodes.npz') as f:
        nodes = {k: f[k] for k in f.files}
    if x.shape[1] != n_genes or a.shape != (x.shape[0], x.shape[0]) or len(nodes['unit_id']) != x.shape[0]:
        raise ValueError(f'{sample}: inconsistent gene, node or graph dimensions')
    if not np.isfinite(x.data).all():
        raise ValueError(f'{sample}: nonfinite RNA values')
    result = dict(x=x, a=a, nodes=nodes)
    if features_dir is not None:
        with np.load(Path(features_dir) / f'{sample}.npz') as f:
            if not np.array_equal(f['unit_id'], nodes['unit_id']):
                raise ValueError(f'{sample}: H&E unit_id order mismatch')
            image = f['image_features'].astype(np.float32)
        if image.shape != (x.shape[0], 512) or not np.isfinite(image).all():
            raise ValueError(f'{sample}: invalid H&E features')
        result['image'] = image
    return result


def transform(section, preprocessing, use_image):
    x = np.clip((section['x'].toarray() - np.asarray(preprocessing['gene_mean'], np.float32)) /
                np.asarray(preprocessing['gene_sd'], np.float32), -5, 5).astype(np.float32)
    image = None
    if use_image:
        image = np.clip((section['image'] - np.asarray(preprocessing['image_mean'], np.float32)) /
                        np.asarray(preprocessing['image_sd'], np.float32), -5, 5).astype(np.float32)
    return x, image
