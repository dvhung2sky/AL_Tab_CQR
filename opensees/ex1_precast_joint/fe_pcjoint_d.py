"""Ex14-PC-D: the DETERIORATED precast joint -- the concentration regime.

Same calibrated OpenSees model and three-mode energy-balance limit state as fe_pcjoint_h.py. The
change is the SCENARIO: an ageing precast structure whose bottom bars have corroded at the joint
face (exactly where grout ingress attacks precast connections), under a dead-load-dominated
accidental combination. Two new structural RVs and wider scatter on three existing ones; the
calibrated deterministic parameters are untouched.

Why this scenario and not another: on Ex14-PC-H the pre-fracture capacity is M_n ~ A_s f_y d with
f_y at CoV 0.05, so capacity CoV is 0.046 and failure probability is spread over 76 % of the
structural samples (n_eff 152k of 200k) -- there is no region for acquisition to find. A synthetic
check on the cached curves showed that capacity CoV 0.16-0.21 with live-load CoV 0.15 gives
n_eff 3-5k with the top 1 % of samples carrying 50-64 % of Pf. Corrosion is the physical mechanism
that produces capacity scatter of that size.

STRUCTURAL RVs (11, enter OpenSees)
    U0  f'c          LN 0.25   older concrete of uncertain quality (was 0.15)
    U1  f_y bottom   LN 0.08   JCSS upper range, no mill certificates (was 0.05)
    U2-4 eps_u bar k LN 0.35   pitting corrosion destroys ductility scatter-wise (was 0.15)
    U5  f_y top      LN 0.05   unchanged (inactive pre-fracture: top bars in compression)
    U6  eps_u top    LN 0.15   unchanged
    U7  k_H          LN 0.20   unchanged
    U8  E_c          LN 0.10   unchanged
    U9  A_s factor   LN mean 0.85, CoV 0.15   NEW: residual bottom-bar area, 15 % mean loss
    U10 d_eff        LN 0.03                  NEW: bar placement tolerance, sd 7.5 mm
DEMAND RVs (4, integrated exactly by the A12 formulation)
    U11 theta_R LN 0.10   U12 D LN 0.10   U13 L LN 0.15 (dead-dominated)   U14 theta_allow LN 0.30
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pc_joint as M
import fe_pcjoint_h as H

NRV, N_STRUCT = 15, 11
TARGET_PF = float(os.environ.get("PF", "3e-3"))
N_POP = int(float(os.environ.get("NPOP", "200000")))
JOBS = int(os.environ.get("JOBS", "9"))
CACHE = {}
BASE, GRID, _ln = H.BASE, H.GRID, H._ln
_WD, _WL, _UNOM, _THA_MED = H._WD, H._WL, H._UNOM, H._THA_MED
_COV = dict(fc=0.25, fy_b=0.08, esu_bar=0.35, fy_t=0.05, esu_t=0.15, kH=0.20, Ec=0.10,
            As=0.15, deff=0.03, thR=0.10, D=0.10, L=0.15, thA=0.30)
_AS_MEAN = 0.85
NAMES = ["f'c", "fy_bot", "esu_b1", "esu_b2", "esu_b3", "fy_top", "esu_top", "k_H", "E_c",
         "A_s", "d_eff", "theta_R", "D", "L", "theta_allow"]


def response(u_row):
    p = dict(BASE)
    p["fc"] = BASE["fc"] * _ln(u_row[0], _COV["fc"])
    p["fy_s"] = BASE["fy_s"] * _ln(u_row[1], _COV["fy_b"])
    p["fu_s"] = BASE["fu_s"] * _ln(u_row[1], _COV["fy_b"])
    p["esu_bars"] = [float(BASE["esu_s"] * _ln(u_row[2 + k], _COV["esu_bar"])) for k in range(3)]
    p["esu_s"] = float(np.mean(p["esu_bars"]))
    p["fy_t"] = BASE["fy_t"] * _ln(u_row[5], _COV["fy_t"])
    p["fu_t"] = BASE["fu_t"] * _ln(u_row[5], _COV["fy_t"])
    p["esu_t"] = BASE["esu_t"] * _ln(u_row[6], _COV["esu_t"])
    k = _ln(u_row[7], _COV["kH"])
    p["kH_out"] = BASE["kH_out"] * k
    p["kH_in"] = BASE["kH_in"] * k
    p["Ec"] = BASE["Ec"] * _ln(u_row[8], _COV["Ec"])
    p["d_bot"] = BASE["d_bot"] * float(np.sqrt(_ln(u_row[9], _COV["As"], _AS_MEAN)))  # area loss
    p["d_eff"] = BASE["d_eff"] * _ln(u_row[10], _COV["deff"])
    return H._run(p)


def margins(U, Pdg, Pdmax, ufr, ulast, lam):
    thR = _ln(U[:, 11], _COV["thR"])
    dem = lam * (_WD * _ln(U[:, 12], _COV["D"]) + _WL * _ln(U[:, 13], _COV["L"]))
    thA = _ln(U[:, 14], _COV["thA"], _THA_MED)
    g1 = (thR * Pdmax - dem) / lam
    hit = (thR[:, None] * Pdg) >= dem[:, None]
    dstar = np.where(hit.any(1), GRID[hit.argmax(1)], ulast)
    g2 = (ufr - dstar) / _UNOM
    g3 = (thA * BASE["Lc"] - dstar) / _UNOM
    G = np.vstack([g1, g2, g3])
    return G.min(0), (G.argmin(0) + 1).astype(np.int8)


def make_world(seed, noise):
    return H._make_world(CACHE, seed, noise)


if __name__ == "__main__":
    H.build_population(sys.modules[__name__], "pcjointD")
