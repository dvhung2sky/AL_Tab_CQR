"""AL-Tab-CQR: active-learning reliability analysis with TabICLv2 and conformalized quantile regression.

    python3 al_tab_cqr.py data/ex2_rc_frame_30k.npz --seed 0 --nfe 300

The population file holds X (N x NRV standard-normal inputs) and g (limit-state value, fail <=> g <= 0),
evaluated once by the OpenSees models in opensees/. The loop only reads g at the points it acquires.

Loop (Algorithms 1-3 of the paper):
  1. random initial design of N0 points
  2. every OOF_EVERY refits: out-of-fold CQR offset Qc = (1-ALPHA) quantile of max(q_lo - y, y - q_hi)
  3. fit TabICLv2, score CAND random candidates by  a = w exp(-0.5 (mu/w)^2),  w = 0.5 (q_hi - q_lo) + Qc
  4. pick BATCH points greedily, each pick multiplying the rest by 1 - exp(-0.5 (d / (LP rho))^2)
  5. repeat to NFE evaluations, then fit T(g) = g / (1 + |g|) and read out  Pf = mean(median <= 0)

The acquisition fits (steps 2-4) use raw g, as in the runs reported in the paper; T(g) is used for the
final readout. T is strictly increasing with T(0) = 0, so the failure set is unchanged by it.
"""
import argparse, os, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
torch.set_num_threads(1)                    # torch 2.11 CPU threading can segfault TabICL
from scipy.spatial import cKDTree
from sklearn.model_selection import KFold
from tabicl import TabICLRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA, A_LO, A_HI = 0.10, 0.05, 0.95
CHUNK = 50_000                              # TabICL changes predictions at >= 65,536 query rows
T = lambda t: t / (1.0 + np.abs(t))


def make_reg(device):
    return TabICLRegressor(n_estimators=1, device=device, kv_cache=False,
                           model_path=os.path.join(HERE, "tabicl-regressor-v2-20260212.ckpt"))


def oof_cqr_offset(reg, X, y, k):
    lo, hi = np.empty(len(X)), np.empty(len(X))
    for tr, va in KFold(min(k, len(X)), shuffle=True, random_state=0).split(X):
        reg.fit(X[tr], y[tr])
        q = reg.predict(X[va], output_type=["mean", "quantiles"], alphas=[A_LO, A_HI])["quantiles"]
        lo[va], hi[va] = q[:, 0], q[:, 1]
    return float(np.quantile(np.maximum(lo - y, y - hi), 1 - ALPHA))


def acquire(reg, X, g, seed, nfe, n0=12, batch=8, cand=6000, lp=1.0, koof=3, oof_every=5):
    """Active-learning design: population indices of the nfe evaluated points."""
    n = len(X)
    idx = list(np.random.default_rng(seed + 777).choice(n, size=n0, replace=False))
    crng = np.random.default_rng(seed + 999)
    qc, refit = 0.0, 0
    while len(idx) < nfe:
        Xd, yd = X[idx], g[idx]
        if refit % oof_every == 0:
            qc = oof_cqr_offset(reg, Xd, yd, koof)
        refit += 1
        reg.fit(Xd, yd)
        c = crng.choice(n, size=min(cand, n), replace=False); Pc = X[c]
        o = reg.predict(Pc, output_type=["mean", "quantiles"], alphas=[A_LO, A_HI])
        mu, ql, qh = o["mean"], o["quantiles"][:, 0], o["quantiles"][:, 1]
        w = np.maximum(0.5 * (qh - ql) + qc, 1e-12)
        rho = cKDTree(Xd).query(Pc, k=1)[0] + 1e-9
        a = w * np.exp(-0.5 * (mu / w) ** 2); a[np.isin(c, idx)] = -1.0
        for _ in range(min(batch, nfe - len(idx))):
            j = int(np.argmax(a))
            if a[j] <= 0:
                break
            idx.append(int(c[j]))
            if lp > 0:
                a *= 1.0 - np.exp(-0.5 * (np.linalg.norm(Pc - Pc[j], axis=1) / (lp * rho[j])) ** 2)
            a[j] = -1.0
    return np.asarray(idx)


def readout(reg, X, g, idx):
    """Failure indicators on the whole population from the T(g) median prediction."""
    reg.fit(X[idx], T(g[idx]))
    med = np.concatenate([reg.predict(X[i:i + CHUNK], output_type=["mean", "quantiles"],
                                      alphas=[0.5])["quantiles"][:, 0] for i in range(0, len(X), CHUNK)])
    return med <= 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nfe", type=int, default=300); ap.add_argument("--n0", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--lp", type=float, default=1.0)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--save", help="write design indices (.npy)")
    a = ap.parse_args()

    z = np.load(a.data); X, g = z["X"], z["g"]
    truth = g <= 0; nf = int(truth.sum()); pf_ref = float(truth.mean())
    reg = make_reg(a.device); t0 = time.time()
    idx = acquire(reg, X, g, a.seed, a.nfe, n0=a.n0, batch=a.batch, lp=a.lp)
    pred = readout(reg, X, g, idx)
    fp, fn = int((pred & ~truth).sum()), int((~pred & truth).sum())
    pf = float(pred.mean())
    print(f"{os.path.basename(a.data)} N={len(X):,} NRV={X.shape[1]} seed={a.seed} NFE={a.nfe}\n"
          f"  Pf_ref={pf_ref:.5f}  Pf={pf:.5f}  rel-err={abs(pf - pf_ref) / pf_ref * 100:.2f}%  "
          f"FP={fp} FN={fn}  set error D={(fp + fn) / nf * 100:.2f}%  [{time.time() - t0:.0f}s]")
    if a.save:
        np.save(a.save, idx)


if __name__ == "__main__":
    main()
