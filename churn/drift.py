"""Population Stability Index (PSI): compares incoming data with the training distribution.
Rule of thumb: <0.10 stable, 0.10-0.25 watch, >0.25 significant drift."""
import numpy as np
from .data import NUM

def make_ref(df):
    ref = {}
    for c in NUM:
        e = np.unique(np.quantile(df[c], np.linspace(0, 1, 11)))
        e = np.r_[-np.inf, e[1:-1], np.inf]
        ref[c] = {"edges": e.tolist(), "p": (np.histogram(df[c], e)[0] / len(df)).tolist()}
    return ref

def psi(ref, values):
    a = np.histogram(values, np.array(ref["edges"]))[0] / max(len(values), 1)
    a, e = np.clip(a, 1e-4, None), np.clip(ref["p"], 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))

def report(ref, df):
    out = {}
    for c in NUM:
        v = psi(ref[c], df[c].values)
        out[c] = {"psi": round(v, 3), "status": "stable" if v < .1 else "watch" if v < .25 else "drift"}
    return out
