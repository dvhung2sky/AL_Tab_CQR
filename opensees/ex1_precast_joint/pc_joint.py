"""Ex14-PC: OpenSeesPy model of a PRECAST RC double-span sub-assemblage under middle-column loss.

Calibration target: specimen P-I-H of Nguyen & Tan (2023), Engineering Structures 293:116603,
half-scale, middle-column-removal scenario (MCRS).

WHAT THE MODEL MUST REPRODUCE (published values; never used as model input)
    K_0      = 10.46 kN/mm  initial stiffness
    yield    = 127.4 kN at MJD 22.5 mm
    P_CAA    = 150.1 kN at MJD 135.6 mm
    transition: resistance falls to ~0 between MJD 163 and 235 mm when the BOTTOM bars fracture
               at the joint interface, and recovers only once the 3.45 mm tie-rod tension gap
               closes as the beam pulls the supports inward
    P_CA     = 169.0 kN at MJD 332.8 mm, ending with fracture of the TOP bars
    theta_cr = 11.75 deg = 0.205 rad;  P_n = 117.7 kN;  P_d,max = 132.5 kN (energy balance)
Acceptance = the source paper's OWN component-model accuracy, so the model is not asked to beat it:
    loads          predicted/test  mean 1.01, CoV 0.18  ->  [0.82, 1.20]
    displacements  predicted/test  mean 0.95, CoV 0.17  ->  [0.78, 1.12]

MECHANISM THE MODEL HAS TO CONTAIN
    The middle joint is pushed down, so the beam SAGS about it: the bottom bars at the joint
    interface are the tension chord and fracture first -- exactly what the test reports.
    Compressive arch action needs restraint against OUTWARD support movement; catenary action
    needs restraint against INWARD movement, and that is the branch carrying the 3.45 mm gap.

OPENSEES ANALOGUE OF THE PAPER'S COMPONENT-BASED MODEL
    * fibre `forceBeamColumn` + `Corotational` transformation (catenary action is geometric);
    * `zeroLengthSection` with `Bond_SP01` steel at each middle-joint face = the paper's bar-slip
      springs k_bt / k_bb; without it the model is far too stiff and fractures far too early;
    * `MinMax` fracture on the steel at the measured ultimate strain;
    * the rig restraint as one `zeroLength` spring per support carrying the measured stiffness in
      each direction AND the measured assembly gap;
    * rigid joint panel (the paper uses linear-elastic joint shear springs of very large stiffness
      and reports no joint shear failure).

UNITS: N, mm, MPa.  1 kN/m = 1 N/mm exactly, so the published tie-rod stiffnesses drop straight in.

INPUT PROVENANCE
    [READ]   read from the papers (Nguyen & Tan 2023, Figs 4(a),(e), Tables 2, 3, 5, 7, Sec. 3.3.5)
    [ASSUME] engineering assumption, listed in the calibration report
"""
import math
import numpy as np
import openseespy.opensees as ops

# ----------------------------------------------------------------------------------------------
# Specimen P-I-H (half-scale)
# ----------------------------------------------------------------------------------------------
PIH = dict(
    name="P-I-H",
    # ---- geometry (mm) ----
    # [READ] SHEAR SPAN, middle-column face to the outer roller. The precast beam is 1750 mm long
    # but Fig. 4(a) insets the roller 200 mm from its end, so the span that carries shear is 1550.
    # Cross-check: the paper's own P_n = 2 M_n / l_s with nominal f_y = 500 MPa gives 121.5 kN at
    # l_s = 1550 against the reported 117.7 kN (ratio 1.03); at l_s = 1750 it gives 107.7 (0.92).
    Lc=1750.0,          # default; l_s = 1550 is the competing reading -- see CALIBRATION.md
    col_w=350.0,        # [READ] column 350 x 350
    b=200.0,            # [READ] beam width
    h=300.0,            # [READ] beam total depth (225 precast + 75 cast-in-situ topping)
    d_eff=250.0,        # [READ] effective depth; bars 50 mm from the top and bottom fibres
    cover=20.0,         # [READ] clear cover to the stirrup
    # ---- longitudinal reinforcement ----
    n_top=3, d_top=20.0,        # [READ] 3H20 continuous through the joint
    n_bot=3, d_bot=20.0,        # [READ] 3H20 headed, anchored 300 mm inside the joint
    d_stir=8.0, s_stir=50.0,    # [READ] R8 two-legged hoops at 50 mm over 500 mm from the joint
    fy_h=345.0, esu_h=0.143,    # [READ] R8 smooth bar: fy 345, fu 423, eps_u 14.3 %
    # ---- materials ----
    fc=41.9, Ec=32400.0,        # [READ] cylinder strength and modulus
    eps_c0=0.0023,              # [ASSUME] strain at peak stress
    # [READ] Table 2. The BOTTOM bars are headed (welding lowers fu and esu); the TOP bars run
    # straight through the joint and are ordinary deformed bars -- far more ductile, which matters
    # because they are the catenary tie after the bottom bars fracture.
    fy_s=565.0, fu_s=597.0, esu_s=0.056,      # bottom: headed H20
    fy_t=527.0, fu_t=636.0, esu_t=0.103,      # top: straight H20
    Es=201100.0,                # [READ]
    # ---- rig restraint at the beam-end supports ----
    # [READ] Table 7: compression 10 477 kN/m, 0 mm gap; tension 68 288 kN/m, 3.45 mm gap.
    # [READ] Sec. 3.3.5 resolves which is which: after the bottom bars fracture the resistance
    # stays at zero "until the tension gap (3.45 mm) in the tie rods was filled by the beam's
    # INWARD movement" -- so the 3.45 mm gap sits on the branch resisting INWARD movement (the
    # catenary branch), and the gap-free 10 477 kN/m branch resists OUTWARD movement (CAA).
    kH_out=10477.0, gap_out=0.0,
    kH_in=68288.0,  gap_in=3.45,
    # ---- analysis ----
    half=True,          # model one span with a symmetry boundary at the middle joint. The full
                        # model bifurcates: one span's hinge fractures first and the other keeps
                        # carrying load, which a symmetric specimen cannot do.
    n_ele=6,            # (unused once plastic-hinge integration replaced the smeared mesh)
    bond_su_mult=35.0,  # [ASSUME] Bond_SP01 su/sy; Zhao & Sritharan suggest 30-40
    top_continuous=False,   # top 3H20 run through the joint (no headed anchorage)
    EI_fac=0.3,         # [READ] (EI)eff = 0.3 EI, ASCE 41-13 Table 10-5 (used by the source group)
    L_bear=50.0,        # [ASSUME] effective bearing length of the joint interface (mm)
    Lp=150.0,           # [ASSUME] plastic hinge length ~0.5h. A fibre mesh localises the hinge
                        # into one integration weight and fractures the bars far too early, so the
                        # hinge length has to be set explicitly (HingeRadau).
    k_rot=0.0,          # rotational restraint at the OUTER support (N.mm/rad); 0 = pinned end
    u_max=400.0,
    du=0.5,
)

PIH_TEST = dict(K0=10.456e3, P_y=127.4e3, u_y=22.5, P_CAA=150.1e3, u_CAA=135.6,
                P_CA=169.0e3, u_CA=332.8, theta_cr=0.205, P_n=117.7e3, Pd_max=132.5e3)


# ----------------------------------------------------------------------------------------------
# materials and sections
# ----------------------------------------------------------------------------------------------
def _materials(p):
    fc, Ec, ec0 = p["fc"], p["Ec"], p["eps_c0"]
    ft, Ets = 0.09 * fc, Ec / 20.0        # softening over ~0.002 strain: a snap-back kills Newton
    ops.uniaxialMaterial("Concrete02", 11, -fc, -ec0, -0.15 * fc, -0.005, 0.1, ft, Ets)
    # Confined core by Mander et al. (1988) from the ACTUAL hoop data (R8 two-legged at 50 mm),
    # replacing an earlier guess of 1.15 fc. The guess made the section LOSE moment after yield,
    # which is the opposite of the measured compressive-arch branch.
    bc = p["b"] - 2 * p["cover"] - p["d_stir"]
    hc = p["h"] - 2 * p["cover"] - p["d_stir"]
    Ash = 2.0 * math.pi * p["d_stir"] ** 2 / 4.0          # two legs
    rho_s = Ash * (bc + hc) / (bc * hc * p["s_stir"])     # volumetric hoop ratio
    fl = 0.5 * rho_s * p["fy_h"]                          # effective lateral pressure
    k = fl / fc
    fcc = fc * (-1.254 + 2.254 * math.sqrt(1.0 + 7.94 * k) - 2.0 * k)
    ecc = ec0 * (1.0 + 5.0 * (fcc / fc - 1.0))
    ecu = 0.004 + 1.4 * rho_s * p["fy_h"] * p["esu_h"] / fcc
    ops.uniaxialMaterial("Concrete02", 12, -fcc, -ecc, -0.2 * fcc, -ecu, 0.1, ft, Ets)
    # Interface concrete. In a zeroLengthSection the section "strain" is a SLIP in mm, so a normal
    # strain-based concrete law would crush at 0.0023 mm of slip. The concrete does not slip -- only
    # the bars do -- so the interface concrete is elastic-no-tension and effectively rigid in
    # bearing, and all interface deformation is carried by Bond_SP01 in the bars.
    # Bearing law for the interface: rescale the strain axis by an effective bearing length so
    # the concrete can crush. A rigid ENT lets the beam jam in compression once the bars fracture.
    Lbear = p.get("L_bear", 50.0)
    ops.uniaxialMaterial("Concrete02", 17, -1.15 * fc, -ec0 * Lbear, -0.2 * fc, -0.02 * Lbear,
                         0.1, 0.02 * fc, Ec / Lbear)
    def _steel(tag_raw, tag_frac, fy, fu, esu):
        b = (fu / fy - 1.0) / (esu * p["Es"] / fy - 1.0)
        ops.uniaxialMaterial("Steel02", tag_raw, fy, p["Es"], b, 18.0, 0.925, 0.15)
        ops.uniaxialMaterial("MinMax", tag_frac, tag_raw, "-min", -esu, "-max", esu)
    _steel(13, 14, p["fy_s"], p["fu_s"], p["esu_s"])                    # bottom, headed
    # Bar-level fracture-strain variability: each bottom bar gets its own MinMax limit, so the
    # section loses its tension chord bar by bar and u_fr is driven by the WEAKEST bar, not by a
    # single lumped esu. Absent from p => the single-material path above, i.e. the calibrated model.
    for _i, _e in enumerate(p.get("esu_bars") or []):
        _steel(130 + _i, 140 + _i, p["fy_s"], p["fu_s"], _e)
    _steel(23, 24, p["fy_t"], p["fu_t"], p["esu_t"])                    # top, straight
    # Zhao & Sritharan (2007) strain penetration; sy is a SLIP in mm inside a zeroLengthSection
    alpha, db = 0.4, p["d_top"]
    sy = 2.54 * (db / 8300.0 * p["fy_s"] / math.sqrt(fc) * (2 * alpha + 1)) ** (1 / alpha) + 0.34
    su = p.get("bond_su_mult", 35.0) * sy
    ops.uniaxialMaterial("Bond_SP01", 15, p["fy_s"], sy, p["fu_s"], su, 0.5, alpha)
    ops.uniaxialMaterial("MinMax", 16, 15, "-min", -su, "-max", su)
    return sy


def _fibre_section(p, tag, steel_tag, c_cover=11, c_core=12, steel_top=None, bot_tags=None):
    b, h, c, ds = p["b"], p["h"], p["cover"], p["d_stir"]
    yb, yt = -h / 2.0, h / 2.0
    zb, zt = -b / 2.0, b / 2.0
    ycb, yct = yb + c + ds, yt - c - ds
    zcb, zct = zb + c + ds, zt - c - ds
    y_top = yt - (h - p["d_eff"])              # bars 50 mm from the top fibre
    y_bot = yb + (h - p["d_eff"])
    A1t = math.pi * p["d_top"] ** 2 / 4.0
    A1b = math.pi * p["d_bot"] ** 2 / 4.0
    ops.section("Fiber", tag)
    ops.patch("rect", c_core, 8, 12, ycb, zcb, yct, zct)
    ops.patch("rect", c_cover, 2, 12, yb, zb, ycb, zt)
    ops.patch("rect", c_cover, 2, 12, yct, zb, yt, zt)
    ops.patch("rect", c_cover, 10, 2, ycb, zb, yct, zcb)
    ops.patch("rect", c_cover, 10, 2, ycb, zct, yct, zt)
    ops.layer("straight", steel_top or steel_tag, p["n_top"], A1t, y_top, zcb, y_top, zct)
    if bot_tags:                       # one fibre per bar, at the same positions ops.layer uses
        nb = len(bot_tags)
        zs = [zcb + (zct - zcb) * k / (nb - 1) for k in range(nb)] if nb > 1 else [0.5 * (zcb + zct)]
        for _m, _z in zip(bot_tags, zs):
            ops.fiber(y_bot, _z, A1b, _m)
    else:
        ops.layer("straight", steel_tag, p["n_bot"], A1b, y_bot, zcb, y_bot, zct)
    return tag


# ----------------------------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------------------------
NSUP_L, NSUP_R = 1, 2
NFACE_L, NFACE_R = 11, 12          # joint side of the middle-joint faces
NBEAM_L, NBEAM_R = 21, 22          # beam side of the same faces (coincident; slip between)
NMID = 30


def build(p):
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    Lc, cw, n = p["Lc"], p["col_w"], p["n_ele"]
    xL, xfL, xM, xfR, xR = 0.0, Lc, Lc + cw / 2.0, Lc + cw, 2 * Lc + cw

    _materials(p)
    top_beam = 23 if p.get("top_continuous") else 24
    _bt = [140 + i for i in range(len(p["esu_bars"]))] if p.get("esu_bars") else None
    _fibre_section(p, 1, 14, steel_top=top_beam, bot_tags=_bt)      # beam plastic hinge
    top_int = 18 if p.get("top_continuous") else 16
    if p.get("top_continuous"):
        ops.uniaxialMaterial("Elastic", 18, p["Es"] / 10.0)   # continuous: no pull-out
    _fibre_section(p, 2, 16, c_cover=17, c_core=17, steel_top=top_int)
    A = p["b"] * p["h"]
    # Elastic interior of the plastic-hinge element. It must use a CRACKED stiffness: a gross-EI
    # interior makes the whole span far too stiff and hides that error in the span calibration.
    # (EI)eff = 0.3 EI per ASCE 41-13 Table 10-5, which is what Nguyen & Tan themselves adopt for
    # RC/PC beams and columns.
    ops.section("Elastic", 3, p["Ec"] * p.get("EI_fac", 0.3), A,
                p["b"] * p["h"] ** 3 / 12.0)
    ops.geomTransf("Corotational", 1)
    # one force-based element per span with a plastic hinge of length Lp at each end
    ops.beamIntegration("HingeRadau", 1, 1, p["Lp"], 1, p["Lp"], 3)

    half = p.get("half", True)
    ops.node(NSUP_L, xL, 0.0)
    ops.node(NFACE_L, xfL, 0.0); ops.node(NBEAM_L, xfL, 0.0)
    ops.node(NMID, xM, 0.0)
    if not half:
        ops.node(NSUP_R, xR, 0.0)
        ops.node(NFACE_R, xfR, 0.0); ops.node(NBEAM_R, xfR, 0.0)

    rig = (1e7, p["Ec"] * 200.0, 1e11)                    # rigid joint panel
    ops.element("elasticBeamColumn", 3001, NFACE_L, NMID, *rig, 1)
    if not half:
        ops.element("elasticBeamColumn", 3002, NMID, NFACE_R, *rig, 1)

    ops.element("zeroLengthSection", 2001, NFACE_L, NBEAM_L, 2, "-orient", 1, 0, 0, 0, 1, 0)
    if not half:
        ops.element("zeroLengthSection", 2002, NFACE_R, NBEAM_R, 2, "-orient", 1, 0, 0, 0, 1, 0)
    # a fibre zeroLengthSection carries P and Mz only -- tie the shear DOF, or the interface is a
    # mechanism and the tangent is singular at step 0
    ops.equalDOF(NFACE_L, NBEAM_L, 2)
    if not half:
        ops.equalDOF(NFACE_R, NBEAM_R, 2)

    ops.element("forceBeamColumn", 1100, NSUP_L, NBEAM_L, 1, 1)
    if not half:
        ops.element("forceBeamColumn", 1200, NSUP_R, NBEAM_R, 1, 1)

    ops.fix(NSUP_L, 0, 1, 0)          # vertical roller, free rotation
    if not half:
        ops.fix(NSUP_R, 0, 1, 0)
    ops.fix(NMID, 1, 0, 1)            # symmetry of a symmetric specimen

    big = 1e4
    springs = [(4001, NSUP_L, -1.0, xL)] + ([] if half else [(4002, NSUP_R, +1.0, xR)])
    for etag, nsup, out, xs in springs:
        gnd = nsup + 50
        ops.node(gnd, xs, 0.0); ops.fix(gnd, 1, 1, 1)
        pts = sorted([
            (out * big,          -out * (big - p["gap_out"]) * p["kH_out"]),
            (out * p["gap_out"],  0.0),
            (-out * p["gap_in"],  0.0),
            (-out * big,          out * (big - p["gap_in"]) * p["kH_in"]),
        ])
        mtag = 20 + etag % 10
        ops.uniaxialMaterial("ElasticMultiLinear", mtag,
                             "-strain", *[q[0] for q in pts],
                             "-stress", *[q[1] for q in pts])
        if p.get("k_rot", 0.0) > 0.0:
            ops.uniaxialMaterial("Elastic", 30 + etag % 10, p["k_rot"])
            ops.element("zeroLength", etag, gnd, nsup,
                        "-mat", mtag, 30 + etag % 10, "-dir", 1, 3)
        else:
            ops.element("zeroLength", etag, gnd, nsup, "-mat", mtag, "-dir", 1)
    return NMID


def pushdown(p, verbose=False):
    """Displacement-controlled pushdown at the middle joint. Returns (u [mm], P [N])."""
    mid = build(p)
    ops.timeSeries("Linear", 1); ops.pattern("Plain", 1, 1)
    ops.load(mid, 0.0, -1.0, 0.0)
    ops.system("UmfPack"); ops.numberer("RCM"); ops.constraints("Transformation")
    ops.test("NormDispIncr", 1e-6, 30, 0); ops.algorithm("Newton")
    ops.integrator("DisplacementControl", mid, 2, -p["du"])
    ops.analysis("Static")

    u, P, du = [0.0], [0.0], p["du"]
    guard = 0
    while u[-1] < p["u_max"] and guard < 40000:
        guard += 1
        ok = ops.analyze(1)
        if ok != 0:
            for alg, tol, it, f in (("Newton", 1e-4, 60, 1.0), ("ModifiedNewton", 1e-4, 200, 0.5),
                                    ("KrylovNewton", 1e-3, 300, 0.25),
                                    ("NewtonLineSearch", 1e-3, 300, 0.1)):
                ops.test("NormDispIncr", tol, it, 0); ops.algorithm(alg)
                ops.integrator("DisplacementControl", mid, 2, -du * f)
                ok = ops.analyze(1)
                if ok == 0:
                    break
            ops.test("NormDispIncr", 1e-6, 30, 0); ops.algorithm("Newton")
            ops.integrator("DisplacementControl", mid, 2, -du)
            if ok != 0:
                if verbose:
                    print(f"    no convergence at u = {u[-1]:.1f} mm")
                break
        # Resistance from EQUILIBRIUM (sum of support reactions), not from the load factor.
        # A loose-tolerance fallback step can "converge" onto a state whose load factor is stale:
        # that produced a spurious flat plateau after bar fracture, with the beam carrying V = 0.1 kN
        # while P was reported as 76 kN. Cross-checking the two catches it.
        ops.reactions()
        R = ops.nodeReaction(NSUP_L, 2) * (2.0 if p.get("half", True) else 1.0)
        if not p.get("half", True):
            R = ops.nodeReaction(NSUP_L, 2) + ops.nodeReaction(NSUP_R, 2)
        lam = ops.getLoadFactor(1) * (2.0 if p.get("half", True) else 1.0)
        if abs(R - lam) > 0.02 * max(abs(lam), 1.0e3):
            if verbose:
                print(f"    equilibrium check failed at u = {-ops.nodeDisp(mid, 2):.1f} mm "
                      f"(reactions {R/1e3:.1f} kN vs load factor {lam/1e3:.1f} kN)")
            break
        u.append(-ops.nodeDisp(mid, 2)); P.append(R)
    return np.array(u), np.array(P)


# ----------------------------------------------------------------------------------------------
# response metrics
# ----------------------------------------------------------------------------------------------
def metrics(u, P):
    out = {}
    if len(u) < 10:
        return out
    Pd = np.array([np.trapezoid(P[:i + 1], u[:i + 1]) / ui if ui > 1e-9 else 0.0
                   for i, ui in enumerate(u)])
    out["Pd_max"] = float(Pd.max()); out["u_Pd_max"] = float(u[int(Pd.argmax())])
    m = u <= 5.0
    out["K0"] = float(np.polyfit(u[m], P[m], 1)[0]) if m.sum() > 3 else float("nan")
    i_pk = None
    for i in range(3, len(P) - 3):
        if P[i] >= P[i - 1] and P[i] > P[i + 1] and P[i] > 0.3 * P.max() and u[i] > 10.0:
            i_pk = i; break
    if i_pk is not None:
        j = i_pk + int(np.argmin(P[i_pk:]))
        k = j + int(np.argmax(P[j:]))
        out.update(P_CAA=float(P[i_pk]), u_CAA=float(u[i_pk]),
                   P_valley=float(P[j]), u_valley=float(u[j]),
                   P_CA=float(P[k]), u_CA=float(u[k]))
    else:
        k = int(np.argmax(P))
        out.update(P_CA=float(P[k]), u_CA=float(u[k]))
    out["u_end"] = float(u[-1]); out["P_max"] = float(P.max())
    return out


def demo():
    u, P = pushdown(PIH, verbose=True)
    assert len(u) > 20, f"analysis died after {len(u)} steps"
    assert P.max() > 0, "no resistance developed"
    m = metrics(u, P)
    print(f"  u_end={m['u_end']:.0f} mm  P_max={m['P_max']/1e3:.1f} kN  "
          f"K0={m['K0']/1e3:.2f} kN/mm  Pd_max={m['Pd_max']/1e3:.1f} kN")
    if "P_CAA" in m:
        print(f"  CAA {m['P_CAA']/1e3:.1f} kN @ {m['u_CAA']:.0f} mm | "
              f"valley {m['P_valley']/1e3:.1f} kN @ {m['u_valley']:.0f} mm | "
              f"CA {m['P_CA']/1e3:.1f} kN @ {m['u_CA']:.0f} mm")
    return u, P, m


if __name__ == "__main__":
    demo()
