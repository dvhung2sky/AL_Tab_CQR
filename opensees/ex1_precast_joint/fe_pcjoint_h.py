"""Ex14-PC-H: the HARD variant of the precast-joint benchmark.

Same calibrated OpenSees model as fe_pcjoint.py -- Lp = 300, L_bear = 100, top bars continuous,
(EI)eff = 0.3 EcIg, l_s = 1550 -- and the same energy-balance limit state. Only the PROBABILISTIC
layer changes. Ex14-PC was easy for a reason that the target Pf cannot fix: six of its ten RVs had
|r| <= 0.12 with the margin, one model-uncertainty variable (theta_R, CoV 0.18) carried most of the
scatter, and one mode governed 570 of 600 failures. Effective dimension ~4, one smooth branch.

THREE CHANGES, each with its own justification.

1. BAR-LEVEL FRACTURE STRAIN (3 RVs where there was 1).  eps_u is a local necking/flaw property
   whose scatter is dominated by WITHIN-batch variability, so the three bottom bars do not share
   one value. Each bar now carries its own MinMax limit, the section loses its tension chord bar
   by bar, and u_fr is set by the WEAKEST of three. That is a minimum-type extreme value: it makes
   the boundary non-smooth in a way a single lumped eps_u cannot be. Yield strength stays a single
   batch variable -- f_y IS well correlated within a member, so splitting it would not be honest.

2. theta_R CoV 0.18 -> 0.10.  At 0.18 the model-uncertainty term is four times the structural
   response spread (P_d,max CoV 0.046) and simply masks the structure. Halving it does not remove
   the epistemic term, it stops it from being the whole problem.

3. A THIRD, COMPETING MODE.  Ex14-PC had g2's rest point d* derived from the same demand as g1, so
   the two modes were nearly redundant (both negative in 186 of 196 failures). g3 adds an
   acceptance rotation theta_allow that is INDEPENDENT of the demand: the frame must reach its rest
   point within a deformation the code allows, whether or not the bars would have survived further.
   Median 0.085 rad with CoV 0.30 is a BENCHMARK DESIGN CHOICE, not a code number -- it is set to
   straddle the measured fracture rotation (163/1550 = 0.105 rad) so that neither deformation
   branch dominates the other. Stated here rather than presented as a derived value.

        g1 = (theta_R * max_d P_ps - P_dem) / lam          strength
        g2 = (u_fr - d*) / u_nom                           bar fracture (weakest of three)
        g3 = (theta_allow * l_s - d*) / u_nom              acceptance rotation
        g  = min(g1, g2, g3)

RANDOM VARIABLES (13)
    U0  f'c           LN 0.15     U7   k_H          LN 0.20
    U1  f_y bottom    LN 0.05     U8   E_c          LN 0.10
    U2  eps_u bar 1   LN 0.15     U9   theta_R      LN 0.10   <- was 0.18
    U3  eps_u bar 2   LN 0.15     U10  D            LN 0.10
    U4  eps_u bar 3   LN 0.15     U11  L            LN 0.40
    U5  f_y top       LN 0.05     U12  theta_allow  LN 0.30, median 0.085 rad   <- new
    U6  eps_u top     LN 0.15
Only U0..U8 enter OpenSees; U9..U12 are applied to the cached response, so any target Pf is a
bisection on lam with no re-analysis.

Build:  NPOP=200000 PF=3e-3 JOBS=9 python3 fe_pcjoint_h.py
"""
import os, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import multiprocessing as mp
import pc_joint as M

NRV = 13
TARGET_PF = float(os.environ.get("PF", "3e-3"))
N_POP = int(float(os.environ.get("NPOP", "200000")))
JOBS = int(os.environ.get("JOBS", "9"))
CACHE = {}

BASE = dict(M.PIH)
BASE.update(Lp=300.0, L_bear=100.0, top_continuous=True, EI_fac=0.3, Lc=1550.0,
            du=2.0, u_max=200.0)

_COV = dict(fc=0.15, fy_b=0.05, esu_bar=0.15, fy_t=0.05, esu_t=0.15,
            kH=0.20, Ec=0.10, thR=0.10, D=0.10, L=0.40, thA=0.30)
_WD, _WL = 0.60, 0.40
_UNOM = 140.0
_THA_MED = 0.085                      # rad; see docstring -- a declared benchmark choice
GRID = np.arange(0.0, BASE["u_max"] + 1e-9, BASE["du"])


def _ln(u, cov, mean=1.0):
    s = np.sqrt(np.log(1.0 + cov ** 2))
    return mean * np.exp(s * u - 0.5 * s ** 2)


def response(u_row):
    """One OpenSees pushdown. Returns (Pd on GRID, max Pd, u_fr, u_last) or None."""
    p = dict(BASE)
    p["fc"] = BASE["fc"] * _ln(u_row[0], _COV["fc"])
    p["fy_s"] = BASE["fy_s"] * _ln(u_row[1], _COV["fy_b"])
    p["fu_s"] = BASE["fu_s"] * _ln(u_row[1], _COV["fy_b"])
    p["esu_bars"] = [float(BASE["esu_s"] * _ln(u_row[2 + k], _COV["esu_bar"])) for k in range(3)]
    p["esu_s"] = float(np.mean(p["esu_bars"]))          # interface/backup path
    p["fy_t"] = BASE["fy_t"] * _ln(u_row[5], _COV["fy_t"])
    p["fu_t"] = BASE["fu_t"] * _ln(u_row[5], _COV["fy_t"])
    p["esu_t"] = BASE["esu_t"] * _ln(u_row[6], _COV["esu_t"])
    k = _ln(u_row[7], _COV["kH"])
    p["kH_out"] = BASE["kH_out"] * k
    p["kH_in"] = BASE["kH_in"] * k
    p["Ec"] = BASE["Ec"] * _ln(u_row[8], _COV["Ec"])
    return _run(p)


def _run(p):
    """Pushdown with a step-refinement ladder. A run that dies before 16 mm is a stiff first
    increment (seen on high-f'c draws: 'Domain::update failed' at step 0), not a collapse; at
    du/8 it converges. Triggered on ~0.1 % of samples, so the mean cost is unchanged."""
    for du in (p["du"], p["du"] / 4, p["du"] / 8, p["du"] / 16):
        out = _post(*M.pushdown(dict(p, du=du)))
        if out is not None:
            return out
    return None


MIN_U_LAST = 50.0     # mm. A run must REACH this, not merely return this many points: at the
                      # retry ladder's du/16 = 0.125 mm, "8 points" is 1 mm of displacement, and a
                      # dead analysis then enters the population as an ultra-weak specimen.


def _post(u, P):
    """Pushdown curve -> (P_ps on GRID, max P_ps, u_fr, u_last), or None if it died early."""
    if len(u) < 8 or u[-1] < MIN_U_LAST:
        return None
    Pd = np.array([np.trapezoid(P[:i + 1], u[:i + 1]) / ui if ui > 1e-9 else 0.0
                   for i, ui in enumerate(u)])
    u_fr = float(u[-1])
    for i in range(5, len(P)):
        j = np.searchsorted(u, u[i] + 5.0)
        if j < len(P) and P[:i].max() > 0 and (P[:i].max() - P[i:j + 1].min()) > 0.20 * P[:i].max():
            u_fr = float(u[i]); break
    return (np.interp(GRID, u, Pd, right=Pd[-1]).astype(np.float32),
            float(Pd.max()), u_fr, float(u[-1]))


def margins(U, Pdg, Pdmax, ufr, ulast, lam):
    """Vectorised three-mode limit state. Returns (g, governing mode 1/2/3)."""
    thR = _ln(U[:, 9], _COV["thR"])
    dem = lam * (_WD * _ln(U[:, 10], _COV["D"]) + _WL * _ln(U[:, 11], _COV["L"]))
    thA = _ln(U[:, 12], _COV["thA"], _THA_MED)
    g1 = (thR * Pdmax - dem) / lam
    hit = (thR[:, None] * Pdg) >= dem[:, None]
    dstar = np.where(hit.any(1), GRID[hit.argmax(1)], ulast)
    g2 = (ufr - dstar) / _UNOM
    g3 = (thA * BASE["Lc"] - dstar) / _UNOM
    G = np.vstack([g1, g2, g3])
    return G.min(0), (G.argmin(0) + 1).astype(np.int8)


def _chunk(args, resp=None):
    lo, Us = args
    resp = resp or response
    return lo, [resp(u) for u in Us]


def _make_world(cache, seed, noise):
    """Observations carry heavy-tailed, heteroscedastic noise -- the suite protocol.

    The pushdown is deterministic, but a real specimen's resistance is not: bond, workmanship,
    casting and measurement scatter are not captured by the 13 RVs. t-4 gives the occasional
    large excursion those mechanisms produce; the heteroscedastic scale makes the scatter grow
    away from the design point, where the model is least supported by test data. The REFERENCE Pf
    is still computed from the clean g -- only what the methods observe is noisy."""
    d = cache[seed]
    U = d["U"]
    g = d["m"] / d["S0"]
    if noise > 0:
        rng = np.random.default_rng(seed)
        z = rng.standard_t(4, size=len(U)) / np.sqrt(4 / 2)          # unit variance
        het = 1.0 + 0.5 * np.linalg.norm(U, axis=1) / np.sqrt(U.shape[1])
        y = g + noise * het * z
    else:
        y = g.copy()
    return U, y, g, float(np.mean(g <= 0))


def make_world(seed, noise):
    return _make_world(CACHE, seed, noise)


NAMES = ["f'c", "fy_bot", "esu_b1", "esu_b2", "esu_b3", "fy_top", "esu_top",
         "k_H", "E_c", "theta_R", "D", "L", "theta_allow"]


def build_population(mod, tag):
    """Evaluate mod.response over N_POP samples in parallel, calibrate lam to TARGET_PF with
    mod.margins, write resp_<tag>_N*.npy and cache_<tag>_pf*_N*_seed0.npy."""
    import functools
    N_POP, NRV, JOBS, TARGET_PF = mod.N_POP, mod.NRV, mod.JOBS, mod.TARGET_PF
    t0 = time.time()
    U = np.random.default_rng(0).standard_normal((N_POP, NRV))
    rfn = f"resp_{tag}_N{N_POP}.npy"
    if os.path.exists(rfn):
        r = np.load(rfn, allow_pickle=True).item()
        Pdg, Pdmax, ufr, ulast, bad = r["Pd"], r["Pdmax"], r["ufr"], r["ulast"], r["bad"]
        print(f"reusing {rfn}", flush=True)
    else:
        print(f"evaluating {N_POP:,} pushdowns on {JOBS} workers ...", flush=True)
        step = 2000
        tasks = [(lo, U[lo:lo + step]) for lo in range(0, N_POP, step)]
        Pdg = np.zeros((N_POP, len(GRID)), np.float32)
        Pdmax = np.zeros(N_POP); ufr = np.zeros(N_POP); ulast = np.zeros(N_POP)
        bad = np.zeros(N_POP, bool); done = 0
        with mp.Pool(JOBS) as pool:
            for lo, out in pool.imap_unordered(functools.partial(_chunk, resp=mod.response), tasks):
                for k, rr in enumerate(out):
                    i = lo + k
                    if rr is None:
                        bad[i] = True; continue
                    Pdg[i], Pdmax[i], ufr[i], ulast[i] = rr
                done += len(out)
                print(f"  {done:,}/{N_POP:,} ({int(bad.sum())} failed) [{time.time()-t0:.0f}s]",
                      flush=True)
        np.save(rfn, dict(U=U, grid=GRID, Pd=Pdg, Pdmax=Pdmax, ufr=ufr, ulast=ulast, bad=bad),
                allow_pickle=True)

    ok = ~bad
    lo, hi = 1.0e3, 6.0e5
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        gg, _ = mod.margins(U[ok], Pdg[ok], Pdmax[ok], ufr[ok], ulast[ok], mid)
        if ((gg <= 0).sum() + bad.sum()) / N_POP < TARGET_PF:
            lo = mid
        else:
            hi = mid
    lam = 0.5 * (lo + hi)
    g = np.full(N_POP, -1.0); mode = np.ones(N_POP, np.int8)
    g[ok], mode[ok] = mod.margins(U[ok], Pdg[ok], Pdmax[ok], ufr[ok], ulast[ok], lam)
    S0 = float(g.std())
    fn = f"cache_{tag}_pf{TARGET_PF:.0e}_N{N_POP}_seed0.npy"
    np.save(fn, dict(U=U, m=g, S0=S0, lam=lam, nbad=int(bad.sum()), mode=mode), allow_pickle=True)

    gn = g / S0; f = gn <= 0; nf = int(f.sum())
    print(f"\n{fn}   [{time.time()-t0:.0f}s]")
    print(f"  lambda = {lam/1e3:.2f} kN   nbad = {int(bad.sum())}   NRV = {NRV}")
    print(f"  Pf = {f.mean():.6f}  ({nf} failures)  ref CoV = {1/np.sqrt(max(nf,1)):.3f}  "
          f"1 point = {100/max(nf,1):.3f}%")
    print(f"  g/S0: mean {gn.mean():+.3f}  sd {gn.std():.3f}  "
          f"skew {float(((gn-gn.mean())**3).mean()/gn.std()**3):+.2f}")
    print(f"  modes among failures: strength {int((mode[f]==1).sum())}, "
          f"fracture {int((mode[f]==2).sum())}, acceptance {int((mode[f]==3).sum())}")
    print(f"  u_fr: mean {ufr[ok].mean():.1f} mm  CoV {ufr[ok].std()/ufr[ok].mean():.3f}")
    cs = [abs(np.corrcoef(U[ok, j], gn[ok])[0, 1]) for j in range(NRV)]
    print("  |corr| with g: " + "  ".join(f"{n}={c:.2f}" for n, c in zip(mod.NAMES, cs)))
    print(f"  variables with |r| > 0.15: {sum(c > 0.15 for c in cs)} of {NRV}")


if __name__ == "__main__":
    import sys
    build_population(sys.modules[__name__], "pcjointH")
