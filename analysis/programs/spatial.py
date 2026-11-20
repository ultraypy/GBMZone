"""Conditional spatial organization on a registered within-tissue graph."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

def standard(x):
    x=np.asarray(x,float);return (x-x.mean(0))/np.maximum(x.std(0),1e-12)

def qbins(x,q):return np.asarray(pd.qcut(np.asarray(x),q,labels=False,duplicates='drop'),float)

def safe_qbins(x,q):
    for k in range(max(1,q),0,-1):
        v=qbins(x,k)
        if not np.isfinite(v).all():v=np.zeros(len(x))
        _,n=np.unique(v,return_counts=True)
        if n.min()>=20:return v.astype(int),k
    raise ValueError('stratum smaller20')

def strata_for(native,depth,mito):
    lab=np.full(len(native),-1,int);rows=[];counter=0
    for c in np.unique(native):
        ci=np.flatnonzero(native==c);db,nq=safe_qbins(depth[ci],min(10,max(1,len(ci)//60)))
        for d in np.unique(db):
            di=ci[db==d];mb,mq=safe_qbins(mito[di],3)
            for m in np.unique(mb):
                ix=di[mb==m];lab[ix]=counter
                rows.append(dict(stratum=counter,native_component=int(c),depth_bin=int(d),mito_bin=int(m),units=len(ix),depth_bins_requested=nq,mito_bins_requested=mq,depth_min=float(depth[ix].min()),depth_max=float(depth[ix].max()),mito_min=float(mito[ix].min()),mito_max=float(mito[ix].max())))
                counter+=1
    assert (lab>=0).all() and min(r['units'] for r in rows)>=20
    return lab,pd.DataFrame(rows)



def score(nodes, graph, seed, repeats):
    programs = [f"P{i:02d}" for i in range(1, 9)]
    markers = [c for c in nodes if c.startswith("marker_")]
    if len(markers) != 7:
        raise ValueError("Supply the seven non-overlapping lineage covariates")
    component = nodes.native_component.to_numpy()
    graph = graph.tocsr(copy=True).astype(float)
    if (graph.shape != (len(nodes), len(nodes)) or (graph != graph.T).nnz or
            np.any(graph.data < 0) or not np.isfinite(graph.data).all()):
        raise ValueError("The graph must be a finite, nonnegative undirected within-tissue adjacency")
    graph.setdiag(0)
    graph.eliminate_zeros()
    graph.data[:] = 1
    u, v = graph.nonzero()
    if np.any(component[u] != component[v]):
        raise ValueError("Found a cross-component edge")
    degree = np.diff(graph.indptr)
    if np.any(degree == 0):
        raise ValueError("Remove ineligible isolated bins before scoring")
    graph = sparse.diags(1 / degree) @ graph
    depth = np.log1p(nodes.total_counts.to_numpy())
    mito = nodes.pct_mito_protein_coding.to_numpy()
    zd, zm = standard(depth), standard(mito)
    tech = np.column_stack([pd.get_dummies(component), zd, zd**2, zd**3, zm, zm**2])
    response = nodes[programs].to_numpy()
    residuals = []
    for design in [tech, np.column_stack([tech, standard(nodes[markers].to_numpy())])]:
        if not np.isfinite(design).all() or not np.isfinite(response).all():
            raise ValueError("Nonfinite covariates or program scores")
        residual = response - design @ np.linalg.lstsq(design, response, rcond=None)[0]
        for c in np.unique(component):
            residual[component == c] -= residual[component == c].mean(axis=0)
        residuals.append(residual)
    e = np.column_stack(residuals)
    denom = (e * e).sum(axis=0)
    denom[denom < 1e-24] = np.nan
    def moran(z):
        return (z * (graph @ z)).sum(axis=0) / denom
    observed = moran(e)
    labels, _ = strata_for(component, depth, mito)
    groups = [np.flatnonzero(labels == l) for l in np.unique(labels)]
    rng = np.random.default_rng(seed)
    perm = np.arange(len(e))
    null = np.empty((repeats, e.shape[1]))
    for i in range(repeats):
        for ix in groups:
            perm[ix] = rng.permutation(ix)
        null[i] = moran(e[perm])
    baseline = null.mean(axis=0)
    delta = observed - baseline
    p = (1 + np.sum(abs(null - baseline) >= abs(delta) - 1e-12, axis=0)) / (repeats + 1)
    p[~np.isfinite(delta)] = np.nan
    return pd.DataFrame({"model": np.repeat(["technical", "composition"], 8), "program": programs * 2,
                         "observed_I": observed, "null_mean_I": baseline, "delta_I": delta,
                         "conditional_spatial_p": p})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", required=True, help="Active bins in graph order; see README")
    parser.add_argument("--graph", required=True, help="Symmetric 75 µm within-mask adjacency (.npz)")
    parser.add_argument("--seed", type=int, required=True, help="Slice-specific seed in Figure 3/spatial_internal_patients.csv")
    parser.add_argument("--repeats", type=int, default=1999)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    score(pd.read_csv(args.nodes), sparse.load_npz(args.graph), args.seed, args.repeats).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
