"""Paired regional statistics and symmetric cell-composition decomposition."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from argparse import ArgumentParser
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

GENES = ["CA9","PDPN","PYGL","GRM3","OPALIN","CNTNAP2","DPP10"]
LINEAGES = ["Malignant_glial_candidate","Astrocyte","Myeloid","Oligodendrocyte",
            "Inhibitory_neuron","Excitatory_neuron","T_NK","Vascular_related","Unresolved"]
FOCAL = dict(zip(GENES, [LINEAGES[i] for i in [0,0,2,3,3,4,5]]))
SIDES = ["core","edge"]
SEED = 20261119

def tests(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    n = len(a)
    est = a.mean() if n else np.nan
    result = dict(n_patients=n, mean=est, n_edge=int((a>0).sum()), n_core=int((a<0).sum()),
                  ci_low=np.nan, ci_high=np.nan, p=np.nan, lopo_min=np.nan, lopo_max=np.nan)
    if n >= 2:
        rng = np.random.default_rng(SEED)
        low, high = np.quantile(rng.choice(a, (10000,n)).mean(1), [.025,.975])
        lopo = (a.sum()-a)/(n-1)
        result.update(ci_low=low, ci_high=high, lopo_min=lopo.min(), lopo_max=lopo.max())
    if 3 <= n <= 18:
        signs = np.array(list(product((-1,1), repeat=n)))
        result["p"] = np.mean(np.abs(signs@a/n) >= abs(est)-1e-12)
    return result

def summarize(data, keys, column):
    return pd.DataFrame([dict(zip(keys, values), **tests(g[column]))
                         for values,g in data.groupby(keys,dropna=False,sort=False)])

def fdr(t, mask):
    t = t.copy()
    t["q"] = np.nan
    for _, g in t[mask].groupby("analysis",sort=False):
        p = g.p.to_numpy()
        t.loc[g.index,"q"] = np.where(np.isfinite(p),multipletests(np.nan_to_num(p,nan=1),method="fdr_bh")[1],np.nan)
    return t

def exact_parts(g, gene):
    n = g.n_cells.to_numpy(float).reshape(2,-1)
    p = n/n.sum(axis=1,keepdims=True)
    s = g[gene+"_cp10k_sum"].to_numpy(float).reshape(2,-1)
    m = np.divide(s,n,out=np.zeros_like(s),where=n>0)
    m[0,n[0]==0] = m[1,n[0]==0]
    m[1,n[1]==0] = m[0,n[1]==0]
    comp = (p[1]-p[0])*m.mean(axis=0)
    within = p.mean(axis=0)*(m[1]-m[0])
    raw = s[1].sum()/n[1].sum()-s[0].sum()/n[0].sum()
    assert np.isclose(raw, (comp+within).sum(), atol=1e-10)
    return raw, comp, within, n

def analyse(frame, prefix, out):
    depth = "depth_stratum" in frame
    outcomes = [g for g in GENES if g in frame and frame[g].notna().any()]
    has_cp10k = all(g+"_cp10k_sum" in frame for g in outcomes) and bool(outcomes)
    panels = ["analysis","patient","region","lineage"]
    numeric = [x for x in frame.select_dtypes(include="number") if x not in ("depth_stratum","threshold")]
    pb = frame.groupby(panels,as_index=False)[numeric].sum(min_count=1)
    pb.to_csv(out/f"{prefix}_pseudobulk.csv",index=False)
    comp, expr, d, dl, matched_depth = [], [], [], [], []
    lineage_coverage = pb.groupby("analysis").lineage.agg(set).to_dict()
    for (analysis,patient), sub in pb.groupby(["analysis","patient"]):
        if set(sub.region) != set(SIDES):
            continue
        cats = [l for l in LINEAGES if l in sub.lineage.values]
        cats += sorted(set(sub.lineage)-set(cats))
        g = sub.set_index(["region","lineage"]).reindex(pd.MultiIndex.from_product([SIDES,cats])).fillna(0)
        core, edge = (g.xs(view) for view in SIDES)
        if min(core.n_cells.sum(),edge.n_cells.sum()) <= 0:
            continue
        p0, p1 = core.n_cells/core.n_cells.sum(), edge.n_cells/edge.n_cells.sum()
        for lineage in LINEAGES:
            if lineage not in lineage_coverage[analysis]:
                comp.append(dict(analysis=analysis,patient=patient,lineage=lineage,core_share=np.nan,edge_share=np.nan,
                                 delta_pp=np.nan,lower_unresolved_pp=np.nan,upper_unresolved_pp=np.nan))
                continue
            pr0, pr1 = p0.get(lineage,0),p1.get(lineage,0)
            un0, un1 = p0.get("Unresolved",0), p1.get("Unresolved",0)
            comp.append(dict(analysis=analysis,patient=patient,lineage=lineage,core_share=pr0,edge_share=pr1,
                             delta_pp=(pr1-pr0)*100, lower_unresolved_pp=(pr1-pr0-un0)*100,
                             upper_unresolved_pp=(pr1-pr0+un1)*100))
        for gene in outcomes:
            covered = sub[gene].notna().all()
            if not covered:
                continue
            if has_cp10k:
                raw,c,w,n = exact_parts(g,gene)
                unresolved = cats.index("Unresolved") if "Unresolved" in cats else None
                w_unresolved = w[unresolved] if unresolved is not None else 0.
                w_eligible = w[np.min(n,axis=0)>=50].sum()
                d.append(dict(analysis=analysis,patient=patient,gene=gene,total=raw,composition=c.sum(),
                              within_classified=w.sum()-w_unresolved,within_unresolved=w_unresolved,
                              within_adequately_covered=w_eligible))
                for j, lineage in enumerate(cats):
                    dl.append(dict(analysis=analysis,patient=patient,gene=gene,lineage=lineage,composition=c[j],
                                   within=w[j],n_core=n[0,j],n_edge=n[1,j]))
            for lineage in LINEAGES[:-1]:
                a,b = core.reindex([lineage],fill_value=0).iloc[0],edge.reindex([lineage],fill_value=0).iloc[0]
                ok = min(a.n_cells,b.n_cells)>=50 and min(a.library,b.library)>=5000
                fc = np.log2((b[gene]+.5)/(b.library)) - np.log2((a[gene]+.5)/(a.library)) if ok else np.nan
                expr.append(dict(analysis=analysis,patient=patient,lineage=lineage,gene=gene,n_core=a.n_cells,n_edge=b.n_cells,
                                 umi_core=a.library,umi_edge=b.library,target_core=a[gene],target_edge=b[gene],
                                 eligible=ok,log2fc=fc))
                if depth and has_cp10k and ok:
                    strata = frame[(frame.analysis==analysis)&(frame.patient==patient)&(frame.lineage==lineage)]
                    c0 = strata[strata.region=="core"].set_index("depth_stratum")
                    c1 = strata[strata.region=="edge"].set_index("depth_stratum")
                    k = c0.index.intersection(c1.index)
                    weights = np.minimum(c0.loc[k,"n_cells"],c1.loc[k,"n_cells"])
                    weights = weights[weights>=10]
                    if weights.sum()>=50:
                        x,y = [np.average(z.loc[weights.index,gene+"_cp10k_sum"]/z.loc[weights.index,"n_cells"],weights=weights)
                               for z in (c0,c1)]
                        matched_depth.append(dict(analysis=analysis,patient=patient,lineage=lineage,gene=gene,
                                                  n_matched_each=int(weights.sum()),core=x,edge=y,
                                                  log2fc=np.log2((y+.05)/(x+.05))))
    comp = pd.DataFrame(comp)
    if comp.empty:
        raise ValueError("No paired patients")
    comp.to_csv(out/f"{prefix}_cell_proportion_pairs.csv",index=False)
    c = summarize(comp,["analysis","lineage"],"delta_pp")
    c = fdr(c,c.lineage.ne("Unresolved"))
    c.to_csv(out/f"{prefix}_cell_proportion_statistics.csv",index=False)
    ex = pd.DataFrame(expr)
    if not ex.empty:
        ex.to_csv(out/f"{prefix}_within_lineage_pairs.csv",index=False)
        entries = []
        for analysis in comp.analysis.unique():
            for gene in GENES:
                for lineage in LINEAGES[:-1]:
                    group = ex[(ex.analysis==analysis)&(ex.lineage==lineage)&(ex.gene==gene)&ex.eligible]
                    a = group.log2fc if (group.target_core+group.target_edge).sum()>=20 else []
                    entries.append(dict(analysis=analysis,gene=gene,lineage=lineage,primary=FOCAL[gene]==lineage,
                                        target_umi=int((group.target_core+group.target_edge).sum()), **tests(a)))
        t = pd.DataFrame(entries)
        t["q_exploratory"] = fdr(t,pd.Series(True,index=t.index))["q"]
        t = fdr(t,t.primary)
        t.to_csv(out/f"{prefix}_within_lineage_statistics.csv",index=False)
    if d:
        dp = pd.DataFrame(d)
        lp = pd.DataFrame(dl)
        dp.to_csv(out/f"{prefix}_decomposition_pairs.csv",index=False)
        lp.to_csv(out/f"{prefix}_decomposition_by_lineage.csv",index=False)
        ds = []
        for (analysis,gene),a in dp.groupby(["analysis","gene"],sort=False):
            arr = a[["total","composition","within_classified","within_unresolved"]].to_numpy()
            total,composition,resolved,unresolved = arr.mean(0)
            rng = np.random.default_rng(SEED)
            bootstrap = arr[rng.integers(0,len(arr),size=(10000,len(arr)))].mean(1)
            rng95 = np.quantile(bootstrap,[.025,.975],axis=0)
            same_sign = np.sign(bootstrap[:,0]) == np.sign(total)
            frac = composition/total if total != 0 else np.nan
            fi = np.quantile(bootstrap[same_sign,1]/bootstrap[same_sign,0],[.025,.975]) if same_sign.any() else [np.nan]*2
            ds.append(dict(analysis=analysis,gene=gene,n_patients=len(arr),total=total,composition=composition,
                           within_classified=resolved,within_unresolved=unresolved,composition_share=frac,
                           composition_share_ci_low=fi[0],composition_share_ci_high=fi[1],
                           bootstrap_total_direction_agreement=same_sign.mean(),
                           total_ci_low=rng95[0,0],total_ci_high=rng95[1,0],
                           composition_ci_low=rng95[0,1],composition_ci_high=rng95[1,1],
                           within_ci_low=rng95[0,2],within_ci_high=rng95[1,2]))
        pd.DataFrame(ds).to_csv(out/f"{prefix}_decomposition_summary.csv",index=False)
    if matched_depth:
        m = pd.DataFrame(matched_depth)
        m.to_csv(out/f"{prefix}_depth_matched_pairs.csv",index=False)
        mt = summarize(m,["analysis","gene","lineage"],"log2fc")
        mt.to_csv(out/f"{prefix}_depth_matched_statistics.csv",index=False)

if __name__=="__main__":
    parser=ArgumentParser()
    parser.add_argument("table",type=Path)
    parser.add_argument("--prefix",required=True)
    parser.add_argument("--out",type=Path,default=Path(__file__).parent/"tables")
    options=parser.parse_args()
    data=pd.read_csv(options.table)
    if {"method","threshold"}<=set(data):
        data["analysis"]=data.method+"_"+data.threshold.map("{:.2f}".format)
    elif "analysis" not in data:
        data["analysis"]=options.prefix
    options.out.mkdir(parents=True,exist_ok=True)
    analyse(data,options.prefix,options.out)
