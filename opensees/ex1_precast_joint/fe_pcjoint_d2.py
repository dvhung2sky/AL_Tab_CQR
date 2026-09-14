"""Ex14-PC-D2: Ex14-PC-D with the residual resistance scatter filed where the taxonomy puts it.

Identical structural RVs and calibrated model. The one change: theta_R CoV 0.10 -> 0.18, the source
paper's own model/test capacity ratio (Nguyen & Tan 2023: mean 1.01, CoV 0.18), which IS the
unexplained specimen-to-specimen resistance scatter. In D that scatter was injected as observation
noise scored against a clean reference -- an inconsistency. Here it is an aleatory variable,
integrated exactly on the demand side, and the reference Pf includes it."""
import os, sys
import numpy as np
import fe_pcjoint_h as H
from fe_pcjoint_d import (NRV, N_STRUCT, N_POP, JOBS, BASE, GRID, _ln, _WD, _WL, _UNOM, _THA_MED,
                          NAMES, response)
import fe_pcjoint_d as D
TARGET_PF = float(os.environ.get("PF", "3e-3"))
CACHE = {}
_COV = dict(D._COV, thR=0.18)


def margins(U, Pdg, Pdmax, ufr, ulast, lam):
    thR = _ln(U[:, 11], _COV["thR"])
    dem = lam * (_WD * _ln(U[:, 12], _COV["D"]) + _WL * _ln(U[:, 13], _COV["L"]))
    thA = _ln(U[:, 14], _COV["thA"], _THA_MED)
    g1 = (thR * Pdmax - dem) / lam
    hit = (thR[:, None] * Pdg) >= dem[:, None]
    dstar = np.where(hit.any(1), GRID[hit.argmax(1)], ulast)
    G = np.vstack([g1, (ufr - dstar) / _UNOM, (thA * BASE["Lc"] - dstar) / _UNOM])
    return G.min(0), (G.argmin(0) + 1).astype(np.int8)


def make_world(seed, noise):
    return H._make_world(CACHE, seed, noise)


if __name__ == "__main__":
    H.build_population(sys.modules[__name__], "pcjointD2")
