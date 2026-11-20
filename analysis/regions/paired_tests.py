"""Patient-paired sensitivity checks with unchanged eligibility and BH family."""
from itertools import product

import numpy as np
import pandas as pd
from scipy import optimize, stats

FOCAL = {
    'CA9':'Malignant_glial_candidate', 'PDPN':'Malignant_glial_candidate',
    'PYGL':'Myeloid', 'GRM3':'Oligodendrocyte', 'OPALIN':'Oligodendrocyte',
    'CNTNAP2':'Inhibitory_neuron', 'DPP10':'Excitatory_neuron',
}
MISSING = [np.nan]*4

def infer_mean(d):
    n = len(d)
    loc = np.mean(d)
    se = stats.sem(d)
    half = stats.t.ppf(.975,n-1)*se
    if not se:
        return loc, loc, loc, float(loc == 0)
    p = 2*stats.t.sf(abs(loc/se),n-1)
    return loc,loc-half,loc+half,p

def permute(x):
    if not len(x):
        return 1.
    signs = np.array(list(product((-1,1),repeat=len(x))))
    observed = abs(x.sum())
    return np.mean(abs(signs@x)>=observed-1e-12)

def count_weighted_random_effects(row):
    # Within-pair log2 ratio and delta-method sampling variance from both counts.
    # Between-patient variance is estimated by REML. Modified Knapp-Hartung
    # uses a t reference with n_patients - 1 df and cannot shrink the model SE.
    x = row.log2fc.to_numpy(float)
    n = len(x)
    v = ((row.target_core+.5)**-1 + (row.target_edge+.5)**-1).to_numpy()/np.log(2)**2
    def estimate(tau2):
        w = 1/(v+tau2)
        loc = np.average(x,weights=w)
        rss = np.dot(w,(x-loc)**2)
        return loc,w.sum(),rss
    def objective(tau2):
        _,w,rss = estimate(tau2)
        return np.log(v+tau2).sum()+np.log(w)+rss
    opt = optimize.minimize_scalar(objective,bounds=(0.,max(1.,16*np.var(x,ddof=1))),
                                    method='bounded',options={'xatol':1e-10})
    assert opt.success
    tau2 = opt.x if opt.fun < objective(0) else 0.
    loc,w,rss = estimate(tau2)
    se = np.sqrt(max(1.,rss/(n-1))/w)
    half = stats.t.ppf(.975,n-1)*se
    p = 2*stats.t.sf(abs(loc/se),n-1)
    return loc,loc-half,loc+half,p,tau2

def estimate_group(sub):
    observed = sub[sub.eligible & sub.log2fc.notna()].copy()
    target_umi = (observed.target_core+observed.target_edge).sum()
    if len(observed)<3 or target_umi<20:
        return {},observed,target_umi
    x = observed.log2fc.to_numpy(float)
    nz = x[x!=0]
    rank = stats.rankdata(abs(nz),method='average')
    return {
        'exact_sign_flip_mean': (x.mean(),np.nan,np.nan,permute(x),np.nan),
        'exact_sign': (x.mean(),np.nan,np.nan,stats.binomtest((nz>0).sum(),len(nz)).pvalue if len(nz) else 1.,np.nan),
        'exact_signed_rank': (x.mean(),np.nan,np.nan,permute(rank*np.sign(nz)),np.nan),
        'paired_t': (*infer_mean(x),np.nan),
        'count_weighted_REML_mKH': count_weighted_random_effects(observed),
    },observed,target_umi
