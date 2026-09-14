"""Ex14-FE-Yi2008 reliability benchmark: RC frame under middle-column removal.

Deterministic core = `yi2008_frame.py`, calibrated against Yi, He, Xiao & Kunnath (2008),
ACI Struct. J. 105(4):433-439 -- one-third scale model of the lower 3 storeys of a 4-bay 8-storey
RC frame, middle column of the first storey replaced by jacks and lowered under displacement
control. Status: CALIBRATED at frame level, 11 of 13 scored quantities inside the band
(`CALIBRATION_FRAME.md`).

Same limit state as Ex14-PC-D3, so the two examples are directly comparable:

    fail  <=>  r > C(x_s, theta_allow),   r = lam (0.6 D + 0.4 L) / theta_R
    C = min(C_fr, C_acc),  C_fr = max_{d < u_fr} P_ps(d),  C_acc = max_{d < theta_allow*l_n} P_ps(d)
    P_ps(d) = (1/d) integral_0^d R(u) du            energy balance, Izzuddin et al. (2008)

l_n = 2467 mm, the clear span of the paper's own Eq. (1). R(u) is the frame resistance
109 kN - N(u) that the test reports.

RANDOM VARIABLES (13: 9 structural -> OpenSees, 4 demand -> integrated exactly)
    U0 f_cu       LN 0.15   measured cube strength 25 MPa
    U1 f_y        LN 0.06   HRB400 D12, measured 416 MPa
    U2 f_u        LN 0.05   measured 526 MPa
    U3 eps_su     LN 0.15   elongation d10 = 23 %
    U4 A_gt       LN 0.15   uniform elongation at f_u, 14 %
    U5 E_s        LN 0.03   200 GPa (not stated in the paper)
    U6 f_y,stir   LN 0.06   HPB235, 370 MPa
    U7 cover      LN 0.15   10 mm clear cover at 1/3 scale
    U8 bond s_y   LN 0.25   multiplier on the Zhao-Sritharan slip
    U9  theta_R   LN 0.18   model uncertainty
    U10 D         LN 0.10   dead load
    U11 L         LN 0.15   live load
    U12 theta_all LN 0.30   acceptance rotation, median 0.040 rad

Env: NPOP, PF, JOBS, UMAX, DU.
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import multiprocessing as mp
import yi2008_frame as Y

NRV, N_STRUCT = 13, 9
TARGET_PF = float(os.environ.get("PF", "1e-2"))
N_POP = int(float(os.environ.get("NPOP", "10000")))
JOBS = int(os.environ.get("JOBS", "9"))
# Analysis window. MEASURED: of 120 random samples at du = 1.0, every divergence occurred between
# 150 and 199 mm and none below; a 150 mm window gives 100 % completion, so no run is ever excluded
# and no numerical stop can masquerade as capacity loss. 150 mm = 0.061 rad on the clear span, which
# still contains the whole compressive-arch phase and the restraint reversal the test reports at 140 mm.
U_MAX = float(os.environ.get("UMAX", "150.0"))
DU = float(os.environ.get("DU", "1.0"))
L_CLEAR = 2467.0                                   # mm, the paper's Eq. (1) clear span
BASE = {"Lc": L_CLEAR}                             # driver compatibility: reads Mod.BASE["Lc"]
# A run must REACH the full window. Accepting a short run would let NUMERICAL divergence masquerade
# as physical capacity loss -- u_last varied 203/276/290 mm across three probe samples at du = 1.0.
MIN_FRAC = 0.99
U_FR_MIN = 30.0          # a fracture call below this is a numerical wobble, not a rupture
GRID = np.arange(0.0, U_MAX + 1e-9, 2.0)
CACHE = {}
NAMES = ["f_cu","f_y","f_u","eps_su","A_gt","E_s","fy_stir","cover","bond_sy",
         "theta_R","D","L","theta_allow"]
_COV = dict(fcu=0.15, fy=0.06, fu=0.05, esu=0.15, agt=0.15, Es=0.03, fys=0.06,
            cov=0.15, bond=0.25, thR=0.18, D=0.10, L=0.15, thA=0.30)
# TAGSFX keeps a different analysis window in its own files; THA_MED is the acceptance-criterion
# median, tunable because it decides which of the three modes governs (rupture is pre-empted by
# acceptance whenever theta_A*Lc < u_fr, which at 0.040 rad is almost always).
TAGSFX = os.environ.get("TAGSFX", "")
_WD, _WL = 0.60, 0.40
_THA_MED = float(os.environ.get("THA_MED", "0.040"))   # 0.040 rad ~ 99 mm: straddles the window so the
                                         # acceptance mode discriminates; ASCE 41 CP for RC beams 0.02-0.05
_UNOM = 300.0
_BASE = dict(FCU=25.0, FY=416.0, FU=526.0, ESU=0.23, AGT=0.14, ES=200000.0,
             FY_STIR=370.0, COVER=10.0, BOND_SY_MULT=0.4)


def _ln(u, cov, mean=1.0):
    s = np.sqrt(np.log(1.0 + cov ** 2))
    return mean * np.exp(s * u - 0.5 * s ** 2)


def response(u_row):
    """One frame pushdown. Returns (P_ps on GRID, max P_ps, u_fr, u_last) or None."""
    Y.FCU = _BASE["FCU"] * _ln(u_row[0], _COV["fcu"]); Y.FC = Y.FC_FACTOR * Y.FCU
    Y.FY = _BASE["FY"] * _ln(u_row[1], _COV["fy"])
    Y.FU = max(_BASE["FU"] * _ln(u_row[2], _COV["fu"]), 1.02 * Y.FY)
    Y.ESU = _BASE["ESU"] * _ln(u_row[3], _COV["esu"])
    Y.AGT = min(_BASE["AGT"] * _ln(u_row[4], _COV["agt"]), 0.9 * Y.ESU)
    Y.ES = _BASE["ES"] * _ln(u_row[5], _COV["Es"])
    Y.FY_STIR = _BASE["FY_STIR"] * _ln(u_row[6], _COV["fys"])
    Y.COVER = _BASE["COVER"] * _ln(u_row[7], _COV["cov"])
    Y.BOND_SY_MULT = _BASE["BOND_SY_MULT"] * _ln(u_row[8], _COV["bond"])
    try:
        u, N, R, dxL, dxR = Y.run(u_max=U_MAX, du=DU)
    except Exception:
        return None
    if len(u) < 20 or u[-1] < MIN_FRAC * U_MAX:
        return None
    Pd = np.array([np.trapezoid(R[:i + 1], u[:i + 1]) / x if x > 1e-9 else 0.0
                   for i, x in enumerate(u)])
    # bar rupture / resistance collapse: first loss of >20 % of the running max in a 15 mm window.
    # If none occurs the frame has NOT ruptured inside the window, so u_fr = inf and mode 2 simply
    # never governs -- never u_last, which is a numerical quantity.
    # MEASURED DEFECT: without the U_FR_MIN guard the "20 % drop of the running maximum" test fires
    # trivially early, when that maximum is still tiny -- 96 of 19,803 specimens were assigned a
    # fracture at a MEDIAN of 15 mm, truncating their capacity to a near-zero value. Those 96
    # outliers dragged R^2 of ln C_fr from 0.817 down to 0.515 and with it the whole benchmark's
    # conditioning. Requiring the response to have developed first fixes it; the result is flat for
    # the guard anywhere in 30-75 mm, so it is not a tuned threshold.
    u_fr = float("inf")
    for i in range(5, len(R)):
        if u[i] < U_FR_MIN:
            continue
        j = int(np.searchsorted(u, u[i] + 15.0))
        if j < len(R) and R[:i].max() > 0 and (R[:i].max() - R[i:j + 1].min()) > 0.20 * R[:i].max():
            u_fr = float(u[i]); break
    return (np.interp(GRID, u, Pd, right=Pd[-1]).astype(np.float32),
            float(Pd.max()), u_fr, float(u[-1]))


def margins(U, Pdg, Pdmax, ufr, ulast, lam):
    thR = _ln(U[:, 9], _COV["thR"])
    dem = lam * (_WD * _ln(U[:, 10], _COV["D"]) + _WL * _ln(U[:, 11], _COV["L"]))
    thA = _ln(U[:, 12], _COV["thA"], _THA_MED)
    g1 = (thR * Pdmax - dem) / lam
    hit = (thR[:, None] * Pdg) >= dem[:, None]
    dstar = np.where(hit.any(1), GRID[hit.argmax(1)], ulast)
    G = np.vstack([g1, (ufr - dstar) / _UNOM, (thA * L_CLEAR - dstar) / _UNOM])
    return G.min(0), (G.argmin(0) + 1).astype(np.int8)


def _chunk(args):
    lo, Us = args
    return lo, [response(u) for u in Us]


def make_world(seed, noise):
    d = CACHE[seed]
    g = d["m"] / d["S0"]
    return d["U"], g.copy(), g, float(np.mean(g <= 0))


if __name__ == "__main__":
    if sys.argv[1:] and sys.argv[1] == "probe":
        Us = np.random.default_rng(0).standard_normal((3, NRV))
        for k in range(3):
            t0 = time.time(); r = response(Us[k]); dt = time.time() - t0
            print(f"  sample {k}: {dt:5.1f} s  " + ("FAILED" if r is None else
                  f"P_ps,max={r[1]/1e3:6.1f} kN  u_fr={r[2]:5.0f}  u_last={r[3]:5.0f} mm"), flush=True)
        sys.exit()
    t0 = time.time()
    U = np.random.default_rng(0).standard_normal((N_POP, NRV))
    rfn = f"resp_yi2008{TAGSFX}_N{N_POP}.npy"
    if os.path.exists(rfn):
        r = np.load(rfn, allow_pickle=True).item()
        Pdg, Pdmax, ufr, ulast, bad = r["Pd"], r["Pdmax"], r["ufr"], r["ulast"], r["bad"]
    else:
        print(f"evaluating {N_POP:,} frame pushdowns on {JOBS} workers "
              f"(u_max={U_MAX}, du={DU}) ...", flush=True)
        step = max(1, N_POP // 200)
        tasks = [(lo, U[lo:lo + step]) for lo in range(0, N_POP, step)]
        Pdg = np.zeros((N_POP, len(GRID)), np.float32)
        Pdmax = np.zeros(N_POP); ufr = np.zeros(N_POP); ulast = np.zeros(N_POP)
        bad = np.zeros(N_POP, bool); done = 0
        seen = np.zeros(N_POP, bool)
        pfn = rfn.replace(".npy", "_partial.npy")
        if os.path.exists(pfn):                       # resume: never lose a stopped run again
            q = np.load(pfn, allow_pickle=True).item()
            Pdg, Pdmax, ufr, ulast, bad, seen = q["Pd"], q["Pdmax"], q["ufr"], q["ulast"], q["bad"], q["seen"]
            tasks = [t for t in tasks if not seen[t[0]]]
            done = int(seen.sum()); print(f"  resuming: {done:,} already done", flush=True)
        with mp.Pool(JOBS) as pool:
            for lo, out in pool.imap_unordered(_chunk, tasks):
                for k, rr in enumerate(out):
                    i = lo + k
                    if rr is None: bad[i] = True; continue
                    Pdg[i], Pdmax[i], ufr[i], ulast[i] = rr
                for k in range(len(out)): seen[lo + k] = True
                done += len(out)
                if done % 1000 < len(out):
                    np.save(rfn.replace(".npy", "_partial.npy"),
                            dict(U=U, grid=GRID, Pd=Pdg, Pdmax=Pdmax, ufr=ufr, ulast=ulast,
                                 bad=bad, seen=seen), allow_pickle=True)
                el = time.time() - t0
                print(f"  {done:,}/{N_POP:,} ({int(bad.sum())} failed) [{el/60:.1f} min, "
                      f"eta {el/max(done,1)*(N_POP-done)/60:.0f} min]", flush=True)
        np.save(rfn, dict(U=U, grid=GRID, Pd=Pdg, Pdmax=Pdmax, ufr=ufr, ulast=ulast, bad=bad),
                allow_pickle=True)
    ok = ~bad
    print(f"  excluding {int(bad.sum()):,} non-convergent samples ({bad.mean()*100:.2f} %); "
          f"population renormalised to {int(ok.sum()):,}", flush=True)
    U, Pdg, Pdmax, ufr, ulast = U[ok], Pdg[ok], Pdmax[ok], ufr[ok], ulast[ok]
    NP2 = len(U); bad = np.zeros(NP2, bool); ok = ~bad
    lo_, hi_ = 1.0e2, 1.0e6
    for _ in range(50):
        mid = 0.5 * (lo_ + hi_)
        gg, _ = margins(U[ok], Pdg[ok], Pdmax[ok], ufr[ok], ulast[ok], mid)
        lo_, hi_ = (mid, hi_) if (gg <= 0).mean() < TARGET_PF else (lo_, mid)
    lam = 0.5 * (lo_ + hi_)
    g, mode = margins(U, Pdg, Pdmax, ufr, ulast, lam)
    S0 = float(g.std())
    np.save(f"resp_yi2008{TAGSFX}_N{NP2}_clean.npy", dict(U=U, grid=GRID, Pd=Pdg, Pdmax=Pdmax,
            ufr=ufr, ulast=ulast, bad=bad), allow_pickle=True)
    fn = f"cache_yi2008{TAGSFX}_pf{TARGET_PF:.0e}_N{NP2}_seed0.npy"
    np.save(fn, dict(U=U, m=g, S0=S0, lam=lam, nbad=int(bad.sum()), mode=mode), allow_pickle=True)
    f = g <= 0
    print(f"\n{fn}  [{(time.time()-t0)/60:.1f} min]")
    print(f"  lambda = {lam/1e3:.2f} kN   N = {NP2:,}   NRV = {NRV}")
    print(f"  Pf = {f.mean():.5f} ({int(f.sum())} failures)  counting CoV = "
          f"{1/np.sqrt(max(f.sum(),1)):.3f}")
    print(f"  modes: strength {int((mode[f]==1).sum())}, rupture {int((mode[f]==2).sum())}, "
          f"acceptance {int((mode[f]==3).sum())}")
