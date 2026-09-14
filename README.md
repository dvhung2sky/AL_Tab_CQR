# AL-Tab-CQR

Active-learning structural reliability analysis of progressive collapse with an in-context tabular
foundation model (TabICLv2) and conformalized quantile regression (CQR).

Code and data for: *Active-Learning Structural Reliability Assessment of Progressive Collapse based on
In-Context Tabular Learning and Conformalized Quantile Regression* (Nguyen, Dang, Tran, Nguyen, Pham;
Hanoi University of Civil Engineering).

## Contents

| path | what |
|---|---|
| `al_tab_cqr.py` | the AL-Tab-CQR method (Algorithms 1–3): TabICLv2 surrogate, out-of-fold CQR offset, band acquisition, locally penalized batches, T(g) readout |
| `opensees/ex1_precast_joint/` | Example 1: OpenSeesPy model of precast interior joint P-I-H (Nguyen & Tan 2023) with corroded bottom bars; entry `fe_pcjoint_d2.py` |
| `opensees/ex2_rc_frame/` | Example 2: OpenSeesPy model of the Yi et al. (2008) RC frame under middle-column removal; entry `fe_yi2008.py` |
| `data/ex1_precast_joint_200k.npz` | Example 1 population: 200,000 samples, 199,857 converged; 15 random variables; Pf = 3.00e-3 (600 failures) |
| `data/ex2_rc_frame_30k.npz` | Example 2 population: 30,000 samples, 27,671 converged; 13 random variables; Pf = 9.97e-3 (276 failures) |

Each `.npz` holds `X` (standard-normal inputs, one row per realization), `g` (normalized limit-state
value, failure when `g <= 0`), `mode` (1 insufficient resistance, 2 bar rupture, 3 excessive chord
rotation), `lam` (load scale, N) and `S0` (normalization). `g` is the OpenSees result for every row, so
the method can be run without OpenSees: it only reads `g` at the points it acquires.

## Install

    pip install numpy scipy scikit-learn torch tabicl==2.1.1 openseespy

Tested with Python 3.14, numpy 2.4.3, scipy 1.17.1, scikit-learn 1.6.1, torch 2.11.0, tabicl 2.1.1,
openseespy 3.8.0. OpenSeesPy is needed only to regenerate the populations. The TabICLv2 checkpoint
(`tabicl-regressor-v2-20260212.ckpt`, 114 MB) downloads from Hugging Face on first run.

## Run

    python3 al_tab_cqr.py data/ex2_rc_frame_30k.npz --seed 0 --nfe 300
    python3 al_tab_cqr.py data/ex1_precast_joint_200k.npz --seed 0 --nfe 300 --batch 8 --lp 1.0

To regenerate a population with OpenSees (hours to days; `JOBS` sets the worker count). Each script
samples `X`, runs every pushdown, drops non-convergent samples, calibrates `lam` to the target Pf
and writes a `cache_*.npy`. The shipped `.npz` files come from these models, but a full
regeneration has not been re-run to confirm bit-identical output:

    cd opensees/ex2_rc_frame && NPOP=30000 PF=1e-2 THA_MED=0.085 UMAX=200 TAGSFX=w200 python3 fe_yi2008.py
    cd opensees/ex1_precast_joint && NPOP=200000 PF=3e-3 python3 fe_pcjoint_d2.py
