"""Build the 75 µm adjacency on registered bin centers inside a tissue mask."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage, sparse
from scipy.spatial import cKDTree


def keep_segments(mask,step,xy,rows,cols):
    h,w=mask.shape;good=np.ones(len(rows),bool)
    for start in range(0,len(rows),25000):
        end=min(start+25000,len(rows));aa=xy[rows[start:end]];bb=xy[cols[start:end]];n=max(2,int(np.ceil(np.linalg.norm(aa-bb,axis=1).max()/step))+1);take=np.ones(len(aa),bool)
        for t in np.linspace(0,1,n):
            ii=np.flatnonzero(take)
            if not len(ii):break
            pos=np.rint((aa[ii]*(1-t)+bb[ii]*t)/step).astype(int);valid=(pos[:,0]>=0)&(pos[:,0]<w)&(pos[:,1]>=0)&(pos[:,1]<h)
            jj=np.flatnonzero(valid);valid[jj]&=mask[pos[jj,1],pos[jj,0]];take[ii]&=valid
        good[start:end]=take
    return good



def build(nodes, mask, step):
    # Build the published 250 µm candidate graph, then restrict it to 75 µm.
    xy = nodes[["x_um", "y_um"]].to_numpy(dtype=float)
    if not np.isfinite(xy).all() or len(np.unique(xy, axis=0)) != len(xy) or step <= 0:
        raise ValueError("Expected distinct finite bin centers and positive mask pixel size")
    pix = np.rint(xy / step).astype(int)
    labels, _ = ndimage.label(mask, structure=ndimage.generate_binary_structure(2, 1))
    component = np.zeros(len(xy), dtype=int)
    valid = (pix >= 0).all(axis=1) & (pix[:, 0] < mask.shape[1]) & (pix[:, 1] < mask.shape[0])
    component[valid] = labels[pix[valid, 1], pix[valid, 0]]
    u, v = cKDTree(xy).query_pairs(250, output_type="ndarray").T
    same = nodes.component.to_numpy()[u] == nodes.component.to_numpy()[v]
    u, v = u[same], v[same]
    keep = keep_segments(mask, step, xy, u, v)
    dist = np.linalg.norm(xy[u] - xy[v], axis=1)
    keep &= (dist > 0) & (dist <= 75) & (component[u] == component[v])
    u, v = u[keep], v[keep]
    graph = sparse.csr_matrix((np.ones(2 * len(u)), (np.r_[u, v], np.r_[v, u])), shape=(len(nodes), len(nodes)))
    ix = np.flatnonzero((component > 0) & (np.bincount(component)[component] >= 30))
    while len(ix):
        active = graph[ix][:, ix].tocsr()
        keep = np.diff(active.indptr) > 0
        size = np.bincount(component[ix][keep], minlength=component.max() + 1)
        keep &= size[component[ix]] >= 20
        if keep.all():
            break
        ix = ix[keep]
    if not len(ix):
        raise ValueError("No eligible connected tissue component")
    return nodes.iloc[ix].assign(native_component=component[ix]), active


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", required=True, help="QC-passed bin centers: unit_id,x_um,y_um,component")
    parser.add_argument("--mask", required=True, help="Registered NPZ: mask and step_dnb (0.5 µm/DNB)")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    nodes = pd.read_csv(args.nodes, dtype={"unit_id": str})
    if not nodes.unit_id.is_unique:
        raise ValueError("Repeated unit_id")
    with np.load(args.mask, allow_pickle=False) as data:
        if "identity_registration_accepted" in data and not bool(data["identity_registration_accepted"]):
            raise ValueError("Registration did not pass")
        nodes, graph = build(nodes, data["mask"].astype(bool), float(data["step_dnb"]) * .5)
    args.out.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(args.out / "nodes.csv.gz", index=False)
    sparse.save_npz(args.out / "adjacency.npz", graph)


if __name__ == "__main__":
    main()
