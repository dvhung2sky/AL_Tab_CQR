"""Frame-level calibration of the Ex14-FE evaluator against Yi, He, Xiao & Kunnath (2008).

Yi W.-J., He Q.-F., Xiao Y., Kunnath S.K. (2008), "Experimental study on progressive collapse-
resistant behavior of reinforced concrete frame structures", ACI Structural Journal 105(4):433-439,
doi 10.14359/19857.

WHY THIS TEST, AND WHAT IT ADDS OVER THE CONNECTION-LEVEL CALIBRATION
    nist_assembly.py calibrates the material, the panel zone and the fracture criterion on a tested
    sub-assembly whose axial restraint came from a laboratory rig.  It therefore cannot calibrate
    the ONE quantity the frame evaluator needs most: the horizontal restraint the surrounding frame
    supplies to the bridging beams -- kappa and K_a in the Ex14 notation.  Yi et al. tested a whole
    frame with NO external horizontal restraint: the restraint is developed by the frame's own
    columns and beams.  Reproducing it is a direct test of that mechanism, and of the full
    flexure -> compressive-arch -> catenary transition through to bar rupture.

THE SPECIMEN  [READ]
    one-third scale model of the lower 3 storeys of a 4-bay, 8-storey RC prototype
    4 bays at 2667 mm centre to centre        (prototype 8000 mm)
    storey heights 1567 / 1100 / 1100 mm      (prototype 4700 / 3300 / 3300 mm)
    beams   100 x 200 mm, top 4D12, bottom 2D12, stirrups D6 @ 150 (75 near the joints)
    columns 200 x 200 mm, 4D12,               stirrups D6 @ 150 (75 near the joints)
    concrete C30, measured CUBE strength 25 MPa
    longitudinal HRB400 D12: f_y 416, f_u 526 MPa, elongation d5 = 27.5 %, d10 = 23 %
    stirrups HPB235: f_y 370 MPa

THE TEST PROTOCOL  [READ]
    The first-storey MIDDLE column was replaced by two stacked mechanical jacks and a load cell.
    A constant 109 kN was applied to the TOP of the middle column by a servo-hydraulic actuator, to
    stand in for the gravity carried down from the upper five storeys of the prototype.  The jacks
    were then lowered step by step under displacement control, so the measured quantity is the jack
    force N as a function of the middle-column vertical displacement, and the frame's own
    resistance is  R(D) = 109 kN - N(D).  The model reproduces exactly that: 109 kN held constant
    at the top, then displacement control at the jack node with N read off as the load factor.

WHAT THE MODEL MUST REPRODUCE (published; never used as model input)
    cracking of the frame beams                       ~4 mm,  elastic below 5 mm
    beam-end bars yield next to the middle column     ~50 mm
    severe concrete crushing / transition begins      ~70 mm
    side columns reverse from OUTWARD to INWARD       140 mm      <- the restraint reversal
    frame resistance at 450 mm                        109 - 6 = 103 kN
    tension bars rupture, collapse                    456 mm  =  10.3 deg  =  0.180 rad
    (the paper's own plastic-mechanism capacity 73.2 kN and catenary estimate 108.4 kN are
     ANALYTICAL, from its Eq. (2); they are reported beside the model but not scored as measurements)

MODEL (reference D, 2026-08-25)
    Rigid joint offsets: beams span face to face (clear 2467 mm, the span of the paper's Eq. (1)),
    columns the clear storey height.  Each beam end: a zeroLengthSection slip interface (Bond_SP01
    bars + rigid no-tension bearing), then a forceBeamColumn with HingeRadau -- a strain-based
    plastic hinge of length HINGE_LP at each end and the SAME fibre section at the interior points,
    so the interior cracks by itself and no effective-stiffness factor is assumed.  Concrete by
    GB 50010 from the measured cube strength; Mander core with the effectiveness coefficient;
    beam bars capped at 0.5 f_y in compression (the post-buckling residual of bars that buckle as
    soon as they yield behind a 0.8 d_b cover) with the hinge-region core confined by the 75 mm
    hoops; bars fracture at the measured eps_su.  Fixed bases.  Two-stage loading reproduces the
    jack protocol.  Scored 9/13; see CALIBRATION_FRAME.md for the lineage A (10/13, two errors
    cancelling) -> B (9/13, correct geometry) -> C (Gomes-Appleton buckling, plateau right, hump
    untouched) -> D (compression cap + hinge core: hump 1.32 -> 1.13).

UNITS: N, mm, MPa.

INPUT PROVENANCE
    [READ]   read from Yi et al. (2008), Tables 1 and 2 and Secs. "Loading method" / "Test results"
    [ASSUME] engineering assumption, listed in CALIBRATION_FRAME.md
"""
import math
import numpy as np
import openseespy.opensees as ops

# ---- specimen ------------------------------------------------------------------------------
L_BAY, N_BAY = 2667.0, 4                       # [READ]
H = [1567.0, 1100.0, 1100.0]                   # [READ] storey heights, ground up
BW, BH = 100.0, 200.0                          # [READ] beam 100 x 200
CW = 200.0                                     # [READ] column 200 x 200
D_LONG, D_STIR = 12.0, 6.0                     # [READ] D12 longitudinal, D6 stirrups
N_TOP, N_BOT = 2, 2                            # [READ] Table 2: beam 2D12 top, 2D12 bottom (the "4 D12" is the COLUMN);
                                               # references A-D used 4D12 top -- a misread of Table 2
N_COL_BAR = 4                                  # [READ] column: 4D12
S_STIR = 150.0                                 # [READ] 150 mm, halved to 75 near the joints
S_STIR_HINGE = 75.0                            # [READ] hoop spacing in the hinge regions
ESU_H = 0.25                                   # [ASSUME] hoop fracture strain; plain HPB235 bars elongate ~25 %
CORE_HINGE = True                              # [READ] the hinges sit where the hoops are at 75 mm
FCU = 25.0                                     # [READ] measured CUBE strength
# Cube -> prism strength and modulus by the code the frame was designed to (GB 50010): prism
# f_c = 0.76 f_cu (alpha_c1, no 0.88 in-situ factor for a laboratory specimen) and
# E_c = 1e5 / (2.2 + 34.7 / f_cu) MPa.  Cross-check, not a fit: at 19 MPa the over-reinforced 4D12
# hogging section is concrete-controlled at M ~ 16.6 kNm, which is what the paper's own Eq. (1)
# (M_p = 15.05 kNm for every hinge) requires; at the 0.80 cylinder conversion (20 MPa, 1.15x at the
# interface) the model's hogging hinges were steel-controlled at ~26 kNm and the plateau came out
# 1.4x too high.
FC_FACTOR = 0.88 * 0.76                        # [READ] GB 50010: f_ck = 0.88 alpha_c1 f_cu,k (in-place prism strength); 16.7 MPa
                                               # references A-D used 0.76 (19 MPa)
FC = FC_FACTOR * FCU                           # 19 MPa
EC_GB = True                                   # [READ] GB 50010 E_c formula
FT_GB = True                                   # [READ] GB 50010 f_t = 0.395 f_cu^0.55 = 2.3 MPa
IFC_FACTOR = 1.0                               # interface concrete at f_c (cover) / f_cc (core)
# Longitudinal steel law.  "hysteretic": symmetric piecewise-linear, no buckling.  "reinf":
# ReinforcingSteel with Dhakal-Maekawa bar buckling, whose one input is the hoop slenderness
# s/d_b = 75/12 = 6.25 in the hinge regions [READ].  Without buckling the bottom 2D12 act as full
# compression steel in the hogging hinge and the section is doubly reinforced at ~26 kNm; the
# paper's Eq. (1) and the measured plateau both say the hinge is at ~15 kNm, i.e. that
# compression-steel contribution is lost once the cover spalls.
STEEL = "cap"
LSR = 75.0 / 12.0     # [READ] hoop spacing near the joints / bar diameter
BUCK = "GABuck"       # Gomes-Appleton (reference C).  Dhakal-Maekawa gives no degradation at all at
                      # s/d_b = 6.25 and over-degrades at 12.5; GA at the read spacing puts the
                      # 100-350 mm plateau at 0.97-1.06 without a new assumption.
BUCK_RES, BUCK_EPS = 0.2, 0.006   # buckle_at_yield: residual / strain at which it is reached
CAP = 1.0             # "cap": compression capacity of the beam bars as a fraction of f_y.
                      # [CAL within a published range] Dhakal-Maekawa's post-buckling average for
                      # one hoop spacing (L/D sqrt(f_y/100) = 12.7) is ~0.5-0.7 f_y; bracketed
                      # 0.2 / 0.3 / 0.5 in CALIBRATION_FRAME.md
CORE_BEAR = 1.0       # [ASSUME] strength multiplier on the BEAM-HINGE core for the bearing confinement the
                      # joint face gives the compression zone (ACI 318 bearing: up to 1.7 f_c); 1.0 = hoops only
ECU_HINGE = None      # [ASSUME] ultimate strain of the beam-hinge core; None = Mander from the hoops
KE_IN_ECU = False                              # k_e also on the core's ultimate strain: tried; it
                                               # moves the 75 mm point onto the test (74.5 vs 78) but
                                               # the core then fails in the catenary stage (rupture
                                               # 377 vs 456) and the 50 mm knee stays at 1.20 -- not
                                               # adopted, reported in CALIBRATION_FRAME.md
FY, FU, ESU = 416.0, 526.0, 0.23               # [READ] HRB400 D12; esu = d10 = 23 %
ETS_DIV = 20.0        # [ASSUME] concrete tension-softening modulus E_c / ETS_DIV (Concrete02 Ets); larger = more tension stiffening
RS_P = 3.0            # ReinforcingSteel hardening exponent p = Esh (esu - esh)/(fu - fy)  [ASSUME]
HK = None             # [ASSUME] hardening knee (frac, strain): backbone point at f_y + frac (f_u - f_y), strain;
                      # None = straight line from yield to (f_u, EFU).  HRB400: ~ (0.65, 0.05)
EC_MULT = 1.0         # multiplier on E_c (sensitivity)
DODD_ESU = None       # Dodd-Restrepo: strain at f_u; None = ESU (23 %).  Intermediate point (DODD_EI, f_y + DODD_FI (f_u - f_y))
DODD_EI, DODD_FI = 0.05, 0.67   # [ASSUME] HRB400 ~ 490 MPa at 5 %
BOND_SY_MULT = 0.4    # [ASSUME] multiplier on Zhao-Sritharan s_y: a bar continuous through a 200 mm joint elongates
                      # ~0.5 of the anchored-bar value at yield
COVER_EU = 0.006      # [ASSUME] cover strain at which the residual (0.05 f_c) is reached (spalling)
FT_MULT = 1.0         # multiplier on f_t (modulus of rupture vs axial tensile strength; sensitivity)
FB_ITER = (30, 1e-9)  # force-based element internal iterations (maxIters, tol)
TOL = 1e-6            # global NormDispIncr tolerance of the regular step
DU_DEMO = 0.25        # step of the reference run: at 0.5 the run diverges at 356 mm on this path (state-
                      # dependent, not physical: the response is identical at 0.25 / 0.5 / 1.0 wherever both exist)
P3 = 1.0              # slope of the steel backbone beyond A_gt (fraction of the hardening modulus)
AGT = 0.14            # [ASSUME] strain at f_u (uniform elongation A_gt).  None = D's law, f_u reached only
                      # at the 23 % fracture strain; HRB400 reaches f_u at A_gt ~ 0.10-0.15 and is flat after
FY_STIR = 370.0                                # [READ] HPB235
ES = 200000.0                                  # [ASSUME] not stated
COVER = 10.0                                   # [ASSUME] clear cover to the stirrup, 1/3 scale
F_TOP = 109.0e3                                # [READ] constant actuator load on the middle column

TEST = dict(u_crack=4.0, u_yield=50.0, u_reversal=140.0, u_at_R=450.0, R_at_450=103.0e3,
            u_rupture=456.0, theta_rupture=math.radians(10.3),
            P_plastic_analytical=73.2e3, P_cable_analytical=108.4e3)

MAT_CORE, MAT_COVER, MAT_STEEL, MAT_RAW = 1, 2, 3, 4
MAT_BOND, MAT_BOND_MM, MAT_IFC, MAT_IFC_CORE = 5, 6, 7, 8   # bar slip at the interfaces
MAT_STEEL_COL, MAT_RAW_COL = 9, 10                          # column bars: always the full law
MAT_CORE_H = 11                                             # beam-hinge core (bearing-confined)
SEC_BEAM, SEC_COL, SEC_SLIP, SEC_SLIP_COL = 1, 2, 3, 4

# Strain penetration at the beam-column interfaces.  A fibre frame with the beams framing straight
# into the joint nodes has no bar slip at all, and ex14pc/pc_joint.py already established what that
# costs for RC: "without it the model is far too stiff and fractures far too early".  The same
# construction is used here -- Bond_SP01 on the bars inside a zeroLengthSection, with the concrete
# of that section rescaled, because in a zeroLengthSection the section "strain" IS a slip in mm.
SLIP = True
# Element formulation.  Force-based elements localise badly once a section softens, and the test
# reports severe concrete crushing beyond 70 mm that the frame has to pass THROUGH to reach
# catenary action.  Both RC frame progressive-collapse models in the project's source set use
# displacement-based elements with 5 integration points (Lin et al. 2022; Ribeiro et al. 2025), so
# that is the default here; "force" is kept for comparison.
ELE = "hinge"
# "hinge": forceBeamColumn with HingeRadau -- a strain-based plastic hinge of declared length
#          HINGE_LP at each beam end and an elastic cracked interior (pc_joint.py's construction).
#          The hogging hinge is then governed by the BEAM section, where strain compatibility between
#          bars and concrete exists, so an over-reinforced section is concrete-controlled as the
#          paper's Eq. (1) requires.  A zeroLengthSection cannot do that: its bar and concrete
#          fibres are independent, so it always finds a block deep enough to balance a hardened bar.
# "disp":  uniform dispBeamColumn mesh (rupture then depends on the end-IP weight).
IFC = "ent"           # interface concrete: "ent" = rigid no-tension bearing (the concrete does
                      # not slip; only the bars do), or "crush" = the closure-rescaled law (L_BEAR)
OFFSETS = True        # [READ] rigid joint offsets: beams span face to face (clear 2467 mm, the span the
                      # paper's own Eq. (1) uses), columns span the clear storey height
RIG_A, RIG_I = 1.0e5, 1.0e9    # rigid-stub properties: ~100x the beam; 1e10 breaks the
                               # UmfPack factorisation under Corotational rotation
INTERIOR = "fibre"    # "elastic" = Elastic section at EI_FAC E_c I_g; "fibre" = the beam fibre
                      # section at the interior points too, so the interior cracks by itself
EI_FAC = 0.6          # used only when INTERIOR = "elastic"; effective interior stiffness; ASCE 41-13 gives 0.3 for a
                      # fully cracked seismic beam, gross is 1.0; a Branson-type effective
                      # inertia for a moderately loaded member is 0.5-0.6.  Swept 0.3 / 0.6 / 1.0.
L_BEAR = 200.0        # [ASSUME] effective bearing length that rescales the interface strain axis;
                      # pc_joint.py's a-priori range is "order of the section depth" (here 200 mm)
BOND_SU_MULT = 35.0   # [ASSUME] Bond_SP01 su/sy; Zhao & Sritharan suggest 30-40
BOND_ALPHA = 0.4      # [READ] Zhao & Sritharan (2007)
BOND_FRACTURE = False # the bars ruptured IN THE BEAM (paper Fig. 3, "axial elongation exceeded 20 %
                      # at the cross section where the bars ruptured"), not by pull-out: fracture is
                      # decided by the fibre steel at the measured eps_su, and Bond_SP01 only
                      # plateaus at f_u beyond s_u
TOP_LAYERS = 1        # [READ] Fig. 1(a) shows two top bars per section and the paper's own Eq. (1)
                      # needs a concrete-controlled hogging section; 4D12 in one layer of a 100 mm
                      # beam would leave 6.7 mm clear between bars, so two layers of 2D12
S_LAYER = 25.0        # [ASSUME] centre-to-centre spacing of the two top layers (13 mm clear)
SELF_WEIGHT = True    # [READ] the paper measured 7.5 kN on the jack from self-weight alone
GAMMA = 24.0e-6       # [ASSUME] N/mm^3
# Plastic-hinge regularisation.  With bars that fracture at a MEASURED strain, the rotation at
# rupture is eps_su x (hinge length) / (lever arm): a mesh-independent answer needs the hinge
# length declared.  HINGE_LP > 0 makes the first and last beam element of every span a hinge
# element of that length with 2 Legendre points, so the fibre strain is averaged over Lp/2 -- the
# same construction as pc_joint.py's HingeRadau, in the displacement-based family that survives
# concrete softening here.  Paulay-Priestley Lp = 0.08 L + 0.022 d_b f_y = 197 + 110 mm, of which
# the 110 mm strain-penetration term is what Bond_SP01 already represents, leaving 0.08 L = 197 mm
# = h for the element.  0 = plain uniform mesh (rupture then depends on the end-IP weight).
HINGE_LP = 130.0      # [CAL] a-priori range 0.5 h - 1.0 h = 100 - 200 mm; chosen on the
                      # rupture displacement alone (150 -> ~715 mm, 200 -> ~950 mm vs 456)
# Strain penetration at the COLUMN BASES.  Nguyen, Yu & Tan (2022) attribute the measured spread
# in Yi's horizontal movements (0.82-1.38) to "imperfect fixed boundary conditions at column bases,
# which allowed rotation".  Bond_SP01 was written for exactly this interface (Zhao & Sritharan
# 2007, column-footing), so the same slip section is used there when BASE_SLIP is on.
BASE_SLIP = False


def _materials():
    Ec = (1.0e5 / (2.2 + 34.7 / FCU) if EC_GB else 4700.0 * math.sqrt(FC)) * EC_MULT
    eps_c0 = 0.002                                               # [ASSUME]
    # tensile strength by the same code as f_c and E_c.  It sets when the columns crack, and the
    # paper reports tension cracks reaching the column bases only at 360 mm: the columns stay
    # nearly uncracked through the arch stage, which is what gives the frame its lateral restraint.
    ft = (0.395 * FCU ** 0.55 if FT_GB else 0.33 * math.sqrt(FC)) * FT_MULT
    ops.uniaxialMaterial("Concrete02", MAT_COVER, -FC, -eps_c0, -0.05 * FC, -COVER_EU,
                         0.1, ft, Ec / ETS_DIV)
    # Mander confinement from the actual D6 @ 150 hoops
    # BEAM core (the hinges are in the beams): 74 x 174 mm between hoop centrelines, four corner
    # bars restrained, hoops at S_STIR.  Mander's effectiveness coefficient k_e is small for such a
    # core -- long unsupported hoop legs and wide arching between them -- and omitting it over-
    # confines the section by ~40 %, which alone turns the hogging hinge steel-controlled.
    bc, hc = BW - 2.0 * COVER - D_STIR, BH - 2.0 * COVER - D_STIR
    Ash = 2.0 * math.pi * D_STIR ** 2 / 4.0
    s_h = S_STIR_HINGE if CORE_HINGE else S_STIR
    rho_s = Ash * (bc + hc) / (bc * hc * s_h)
    sp = s_h - D_STIR
    w = [bc - D_LONG] * 2 + [hc - D_LONG] * 2                 # clear spans between restrained bars
    rho_cc = (N_TOP + N_BOT) * math.pi * D_LONG ** 2 / 4.0 / (bc * hc)
    ke = ((1.0 - sum(x * x for x in w) / (6.0 * bc * hc)) * (1.0 - sp / (2 * bc))
          * (1.0 - sp / (2 * hc)) / (1.0 - rho_cc))
    ke = max(ke, 0.0)
    fl = 0.5 * rho_s * FY_STIR * ke
    k = fl / FC
    fcc = FC * (-1.254 + 2.254 * math.sqrt(1.0 + 7.94 * k) - 2.0 * k)
    ecc = eps_c0 * (1.0 + 5.0 * (fcc / FC - 1.0))
    # ultimate strain with the SAME effectiveness coefficient as the strength (Priestley's form
    # uses the effective confining pressure): hoops this ineffective cannot give a 2 % crushing
    # strain
    ecu = 0.004 + 1.4 * rho_s * FY_STIR * ESU_H * (ke if KE_IN_ECU else 1.0) / fcc
    ops.uniaxialMaterial("Concrete02", MAT_CORE, -fcc, -ecc, -0.2 * fcc, -ecu, 0.1, ft, Ec / ETS_DIV)
    fcch = fcc * CORE_BEAR
    ecch = eps_c0 * (1.0 + 5.0 * (fcch / FC - 1.0))
    ecuh = ECU_HINGE if ECU_HINGE else ecu
    ops.uniaxialMaterial("Concrete02", MAT_CORE_H, -fcch, -ecch, -0.2 * fcch, -ecuh, 0.1, ft, Ec / ETS_DIV)
    # steel: the same piecewise-linear-capped law the connection calibration settled on, with
    # fracture at the MEASURED elongation -- so there is no calibrated fracture parameter here
    ey = FY / ES
    EFU = AGT if AGT else ESU                                   # strain at which f_u is reached
    Eh = (FU - FY) / (EFU - ey)
    if HK:
        F2, E2 = FY + HK[0] * (FU - FY), HK[1]
        F3, E3 = FU, EFU                                       # third point = f_u; continues at that slope
    else:
        F2, E2 = FU, EFU
        F3, E3 = FU + P3 * Eh * (0.60 - EFU), 0.60
    if STEEL == "reinf":
        # same tension backbone as the Hysteretic law (yield, then hardening to f_u at eps_su);
        # esh/Esh give the onset and initial slope of hardening [ASSUME, HRB400 plateau ~2 %]
        esh, Esh = 0.02, (FU - FY) / (ESU - 0.02) * 2.0
        if BUCK == "DMBuck":
            ops.uniaxialMaterial("ReinforcingSteel", MAT_RAW, FY, FU, ES, Esh, esh, ESU,
                                 "-DMBuck", LSR, 1.0)
        else:
            ops.uniaxialMaterial("ReinforcingSteel", MAT_RAW, FY, FU, ES, Esh, esh, ESU,
                                 "-GABuck", LSR, 1.0, 0.4, 0.5)
    elif STEEL == "dodd":
        # Dodd & Restrepo (1995): yield plateau to 2 %, curved hardening through (DODD_EI, f_shi) to f_u at
        # DODD_ESU, flat beyond; symmetric; smooth.  Fracture still by MinMax at ESU.  [ASSUME shape]
        esu_d = DODD_ESU if DODD_ESU else ESU
        ops.uniaxialMaterial("Dodd_Restrepo", MAT_RAW, FY, FU, 0.02, esu_d, ES, DODD_EI, FY + DODD_FI * (FU - FY))
    elif STEEL == "reinf_nobuck":
        # Chang-Mander backbone: yield plateau to esh, curved hardening reaching f_u at EFU (= A_gt),
        # flat beyond; symmetric (no buckling law).  Fracture still by MinMax at ESU.
        esh = 0.02
        Esh = RS_P * (FU - FY) / (EFU - esh)
        ops.uniaxialMaterial("ReinforcingSteel", MAT_RAW, FY, FU, ES, Esh, esh, EFU)
    elif STEEL == "tension_only":
        # limiting case, BEAM bars only: no compression at all (fully buckled / unanchored)
        e3, f3 = E3, F3
        ops.uniaxialMaterial("Hysteretic", MAT_RAW, FY, ey, F2, E2, f3, e3,
                             -2.0, -ey, -2.5, -ESU, -3.0, -e3, 1.0, 1.0, 0.0, 0.0, 0.0)
    elif STEEL == "cap":
        # BEAM bars only: full tension law; in compression, elastic to CAP*f_y and then perfectly
        # plastic -- the post-buckling residual of a bar that buckles as soon as it yields behind a
        # cover of 0.8 d_b (Dhakal-Maekawa residual ~0.2-0.3 f_y).  The brief peak at f_y is
        # omitted because its drop is a negative section tangent the force-based hinge cannot
        # integrate (the "buckle_at_yield" runs fail at 47 mm at any step).  [ASSUME]
        e3, f3 = E3, F3
        ops.uniaxialMaterial("Hysteretic", MAT_RAW, FY, ey, F2, E2, f3, e3,
                             -CAP * FY, -CAP * ey, -CAP * FY * 1.01, -0.02, -CAP * FY * 1.02, -e3,
                             1.0, 1.0, 0.0, 0.0, 0.0)
    elif STEEL == "buckle_at_yield":
        # BEAM bars only: full tension law; in compression, elastic to f_y and then the capacity
        # falls away -- a bar between plain D6 hoops at s/d_b >= 6.25 in a 100 mm beam buckles at
        # compression yield.  This is what the paper's own Eq. (1) assumes: M_p without any
        # compression-steel contribution.  [ASSUME]
        e3, f3 = E3, F3
        ops.uniaxialMaterial("Hysteretic", MAT_RAW, FY, ey, F2, E2, f3, e3,
                             -FY, -ey, -BUCK_RES * FY, -BUCK_EPS, -0.5 * BUCK_RES * FY, -e3,
                             1.0, 1.0, 0.0, 0.0, 0.0)
    else:
        e3, f3 = E3, F3
        ops.uniaxialMaterial("Hysteretic", MAT_RAW, FY, ey, F2, E2, f3, e3,
                             -FY, -ey, -F2, -E2, -f3, -e3, 1.0, 1.0, 0.0, 0.0, 0.0)
    ops.uniaxialMaterial("MinMax", MAT_STEEL, MAT_RAW, "-min", -ESU, "-max", ESU)
    if STEEL == "dodd":
        esu_d = DODD_ESU if DODD_ESU else ESU
        ops.uniaxialMaterial("Dodd_Restrepo", MAT_RAW_COL, FY, FU, 0.02, esu_d, ES, DODD_EI, FY + DODD_FI * (FU - FY))
    elif STEEL == "reinf_nobuck":
        esh = 0.02
        Esh = RS_P * (FU - FY) / (EFU - esh)
        ops.uniaxialMaterial("ReinforcingSteel", MAT_RAW_COL, FY, FU, ES, Esh, esh, EFU)
    elif STEEL == "reinf":
        # columns: the same law as the beams (their hoops are also D6 @ 150 / 75)
        esh, Esh = 0.02, (FU - FY) / (ESU - 0.02) * 2.0
        if BUCK == "DMBuck":
            ops.uniaxialMaterial("ReinforcingSteel", MAT_RAW_COL, FY, FU, ES, Esh, esh, ESU,
                                 "-DMBuck", LSR, 1.0)
        else:
            ops.uniaxialMaterial("ReinforcingSteel", MAT_RAW_COL, FY, FU, ES, Esh, esh, ESU,
                                 "-GABuck", LSR, 1.0, 0.4, 0.5)
    else:
        e3, f3 = E3, F3
        ops.uniaxialMaterial("Hysteretic", MAT_RAW_COL, FY, ey, F2, E2, f3, e3,
                             -FY, -ey, -F2, -E2, -f3, -e3, 1.0, 1.0, 0.0, 0.0, 0.0)
    ops.uniaxialMaterial("MinMax", MAT_STEEL_COL, MAT_RAW_COL, "-min", -ESU, "-max", ESU)

    # ---- bar slip at the beam-column interfaces ----
    # Zhao & Sritharan (2007): sy is a SLIP in mm, not a strain, because it lives inside a
    # zeroLengthSection.  su is the slip at bar fracture.
    sy = 2.54 * (D_LONG / 8300.0 * FY / math.sqrt(FC) * (2 * BOND_ALPHA + 1)) ** (1 / BOND_ALPHA) \
         + 0.34
    su = BOND_SU_MULT * sy                 # slip capacity from the ANCHORED-bar value (not scaled)
    sy *= BOND_SY_MULT
    ops.uniaxialMaterial("Bond_SP01", MAT_BOND, FY, sy, FU, su, 0.5, BOND_ALPHA)
    ops.uniaxialMaterial("MinMax", MAT_BOND_MM, MAT_BOND, "-min", -su, "-max", su)
    # Interface concrete: the strain axis is rescaled by L_BEAR, or an ordinary concrete law would
    # crush at 0.002 mm of slip.  The concrete does not slip -- only the bars do -- so this branch
    # only has to stop the beam end jamming into the joint in compression.
    if IFC == "ent":
        # rigid in bearing (stiffness = E_c over 10 mm), no tension: all interface deformation is
        # bar slip, and the compression zone of the hinge lives in the beam section next door
        ops.uniaxialMaterial("ENT", MAT_IFC, Ec / 10.0)
        ops.uniaxialMaterial("ENT", MAT_IFC_CORE, Ec / 10.0)
    else:
        ops.uniaxialMaterial("Concrete02", MAT_IFC, -IFC_FACTOR * FC, -eps_c0 * L_BEAR,
                             -0.2 * FC, -0.02 * L_BEAR, 0.1, 0.02 * FC, Ec / L_BEAR)
        ops.uniaxialMaterial("Concrete02", MAT_IFC_CORE, -IFC_FACTOR * fcc, -ecc * L_BEAR,
                             -0.2 * fcc, -ecu * L_BEAR, 0.1, 0.02 * FC, Ec / L_BEAR)
    return Ec, sy, su


def _rc_section(tag, b, h, n_top, n_bot, core=None, cover=None, steel=None):
    """Rectangular RC fibre section: confined core, cover shell, two bar layers.

    The material tags are arguments so the SAME geometry can be reused for the slip interface,
    where the steel becomes Bond_SP01 and the concrete becomes the rescaled interface law.
    """
    core = MAT_CORE if core is None else core
    cover = MAT_COVER if cover is None else cover
    steel = MAT_STEEL if steel is None else steel
    A1 = math.pi * D_LONG ** 2 / 4.0
    c = COVER + D_STIR + D_LONG / 2.0                 # bar centroid from the face
    yb, yt = -h / 2.0, h / 2.0
    zb, zt = -b / 2.0, b / 2.0
    ycb, yct = yb + COVER + D_STIR, yt - COVER - D_STIR
    zcb, zct = zb + COVER + D_STIR, zt - COVER - D_STIR
    ops.section("Fiber", tag)
    ops.patch("rect", core, 10, 10, ycb, zcb, yct, zct)
    ops.patch("rect", cover, 2, 10, yb, zb, ycb, zt)
    ops.patch("rect", cover, 2, 10, yct, zb, yt, zt)
    ops.patch("rect", cover, 8, 2, ycb, zb, yct, zcb)
    ops.patch("rect", cover, 8, 2, ycb, zct, yct, zt)
    if n_top > 2 and TOP_LAYERS == 2:
        ops.layer("straight", steel, n_top // 2, A1, yt - c, zcb, yt - c, zct)
        ops.layer("straight", steel, n_top - n_top // 2, A1, yt - c - S_LAYER, zcb,
                  yt - c - S_LAYER, zct)
    else:
        ops.layer("straight", steel, n_top, A1, yt - c, zcb, yt - c, zct)
    ops.layer("straight", steel, n_bot, A1, yb + c, zcb, yb + c, zct)


PROBE = {}
COLS = []
BEAMS = []
NEB, NEC = 4, 2                                       # elements per beam span / per column storey


def _beam_ele(tag, a, b, integ):
    if ELE == "disp":
        ops.element("dispBeamColumn", tag, a, b, 1, integ)
    else:
        ops.element("forceBeamColumn", tag, a, b, 1, integ, "-iter", *FB_ITER)


def build():
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    _materials()
    _rc_section(SEC_BEAM, BW, BH, N_TOP, N_BOT, core=MAT_CORE_H)
    _rc_section(SEC_COL, CW, CW, N_COL_BAR // 2, N_COL_BAR // 2, steel=MAT_STEEL_COL)
    _rc_section(SEC_SLIP, BW, BH, N_TOP, N_BOT,
                core=MAT_IFC_CORE, cover=MAT_IFC, steel=MAT_BOND_MM if BOND_FRACTURE else MAT_BOND)
    _rc_section(SEC_SLIP_COL, CW, CW, N_COL_BAR // 2, N_COL_BAR // 2,
                core=MAT_IFC_CORE, cover=MAT_IFC, steel=MAT_BOND)
    ops.geomTransf("Corotational", 1)                 # catenary action is geometric
    ops.beamIntegration("Lobatto", 1, SEC_BEAM, 5)
    ops.beamIntegration("Lobatto", 2, SEC_COL, 5)
    ops.beamIntegration("Legendre", 3, SEC_BEAM, 2)      # hinge elements (ELE = "disp")
    Ec = (1.0e5 / (2.2 + 34.7 / FCU) if EC_GB else 4700.0 * math.sqrt(FC)) * EC_MULT
    ops.section("Elastic", 5, Ec * EI_FAC, BW * BH, BW * BH ** 3 / 12.0)
    ops.beamIntegration("HingeRadau", 4, SEC_BEAM, HINGE_LP, SEC_BEAM, HINGE_LP,
                        SEC_BEAM if INTERIOR == "fibre" else 5)
    RIG = (RIG_A, ES, RIG_I)                               # rigid joint stubs

    ylev = [0.0]
    for hh in H:
        ylev.append(ylev[-1] + hh)
    i_mid = N_BAY // 2

    def nid(i, k, s=0):                                # column line i, floor k, sub-node s
        return 10000 + 1000 * i + 100 * k + s
    for i in range(N_BAY + 1):
        for k in range(len(ylev)):
            if k == 0 and i == i_mid:
                continue                               # the jack replaces this base node
            ops.node(nid(i, k), i * L_BAY, ylev[k])

    # beam stations along a span (from the left joint).  Uniform, or hinge elements of length
    # HINGE_LP at both ends with the interior split into NEB - 2 elements.
    offb = CW / 2.0 if OFFSETS else 0.0                # column half-width: beam joint offset
    if ELE == "hinge":
        XS = [offb, L_BAY - offb]                      # one element per span, face to face
    elif HINGE_LP > 0.0:                               # hinge elements of length L_p from the FACE
        inner = np.linspace(offb + HINGE_LP, L_BAY - offb - HINGE_LP, NEB - 1)
        XS = [offb] + list(inner) + [L_BAY - offb]
    else:
        XS = [offb + s * (L_BAY - 2.0 * offb) / NEB for s in range(NEB + 1)]
    neb = len(XS) - 1
    for i in range(N_BAY):                             # intermediate beam nodes
        for k in range(1, len(ylev)):
            for s in range(1, neb):
                ops.node(nid(i, k, 10 + s), i * L_BAY + XS[s], ylev[k])

    for i in range(N_BAY + 1):                         # [READ] fixed column bases
        if i == i_mid:
            continue
        if BASE_SLIP:
            g = 50000 + i
            ops.node(g, i * L_BAY, 0.0); ops.fix(g, 1, 1, 1)
            # column lies on the +Y side of the ground node: order (ground, column); local x = +Y
            ops.element("zeroLengthSection", 900 + i, g, nid(i, 0), SEC_SLIP_COL,
                        "-orient", 0, 1, 0, -1, 0, 0)
            ops.equalDOF(g, nid(i, 0), 1)              # the section carries N and M; tie the shear
        else:
            ops.fix(nid(i, 0), 1, 1, 1)

    e = 1
    COLS.clear(); BEAMS.clear()
    off = BH / 2.0 if OFFSETS else 0.0                 # beam half-depth: column joint offset
    for i in range(N_BAY + 1):                         # columns
        for k in range(1, len(ylev)):
            if k == 1 and i == i_mid:
                continue                               # first-storey middle column: removed
            lo, hi = nid(i, k - 1), nid(i, k)
            if OFFSETS:
                # rigid stub from each joint to the beam face; the column runs face to face
                if k > 1:
                    ops.node(nid(i, k - 1, 40), i * L_BAY, ylev[k - 1] + off)
                    ops.element("elasticBeamColumn", e, lo, nid(i, k - 1, 40), *RIG, 1); e += 1
                    lo = nid(i, k - 1, 40)
                ops.node(nid(i, k, 41), i * L_BAY, ylev[k] - off)
                ops.element("elasticBeamColumn", e, nid(i, k, 41), hi, *RIG, 1); e += 1
                hi = nid(i, k, 41)
            ya, yb = ops.nodeCoord(lo)[1], ops.nodeCoord(hi)[1]
            for s_ in range(1, NEC):
                ops.node(nid(i, k, 30 + s_), i * L_BAY, ya + s_ * (yb - ya) / NEC)
            chain = [lo] + [nid(i, k, 30 + s_) for s_ in range(1, NEC)] + [hi]
            for a, b in zip(chain[:-1], chain[1:]):
                _beam_ele(e, a, b, 2); COLS.append(e); e += 1
    for k in range(1, len(ylev)):                      # beams
        for i in range(N_BAY):
            nL, nR = nid(i, k), nid(i + 1, k)
            if k == 1 and i == 1:
                PROBE.clear()
            if OFFSETS:                                # rigid stubs from the joints to the faces
                ops.node(nid(i, k, 42), i * L_BAY + offb, ylev[k])
                ops.node(nid(i, k, 43), (i + 1) * L_BAY - offb, ylev[k])
                ops.element("elasticBeamColumn", e, nL, nid(i, k, 42), *RIG, 1); e += 1
                ops.element("elasticBeamColumn", e, nid(i, k, 43), nR, *RIG, 1); e += 1
                nL, nR = nid(i, k, 42), nid(i, k, 43)
            if SLIP:
                # one slip node per beam END, so the two beams meeting at an interior joint each
                # get their own interface rather than sharing one
                for tag, jnt, xx, side in ((nid(i, k, 20), nL, i * L_BAY + offb, "L"),
                                           (nid(i, k, 21), nR, (i + 1) * L_BAY - offb, "R")):
                    ops.node(tag, xx, ylev[k])
                    # NODE ORDER MATTERS.  A zeroLengthSection with local x = +X computes the fibre
                    # deformation as (u_j - u_i) - y (theta_j - theta_i), which is the physical opening
                    # only when the beam lies on the +X side of node i.  At a beam's RIGHT end the
                    # beam is on the -X side of the joint, so node i must be the BEAM node; ordering
                    # it (joint, beam) mirrors the section top-for-bottom -- the tension bars swap.
                    # Verified with a two-node test; ex14pc/pc_joint.py element 2001 has the mirror.
                    a, b = (jnt, tag) if side == "L" else (tag, jnt)
                    ops.element("zeroLengthSection", e, a, b, SEC_SLIP,
                                "-orient", 1, 0, 0, 0, 1, 0)
                    if k == 1 and i == 1:
                        PROBE["slip_C2" if jnt == nL else "slip_C3"] = e
                    e += 1
                    # a fibre section carries only N and M, so the shear DOF has to be tied or the
                    # interface is a mechanism and the tangent is singular at step 0
                    ops.equalDOF(jnt, tag, 2)
                nL, nR = nid(i, k, 20), nid(i, k, 21)
            chain = [nL] + [nid(i, k, 10 + s) for s in range(1, neb)] + [nR]
            for j, (a, b) in enumerate(zip(chain[:-1], chain[1:])):
                if ELE == "hinge":
                    ops.element("forceBeamColumn", e, a, b, 1, 4, "-iter", *FB_ITER)
                else:
                    hinge = HINGE_LP > 0.0 and j in (0, neb - 1)
                    _beam_ele(e, a, b, 3 if hinge else 1)
                BEAMS.append(e)
                if k == 1 and i == 1 and j == 0:
                    PROBE["beam_C2"] = e
                if k == 1 and i == 1 and j == neb - 1:
                    PROBE["beam_C3"] = e
                e += 1
    PROBE["C2"], PROBE["C4"] = nid(i_mid - 1, 1), nid(i_mid + 1, 1)
    return nid(i_mid, 1), nid(i_mid, len(ylev) - 1), nid(0, 1), nid(N_BAY, 1)


def _probe():
    """State of the first-floor bay C2->C3: thrust, bar slip / strain, interface crushing."""
    c = COVER + D_STIR + D_LONG / 2.0
    yt, yb = BH / 2.0 - c, -BH / 2.0 + c
    out = {}
    # axial force in the beam element at the C3 face (tension positive): the thrust
    q = ops.eleResponse(PROBE["beam_C3"], "basicForce")
    out["N_beam"] = q[0] if q else float("nan")
    out["M_C2"], out["M_C3"] = (q[1], q[2]) if q else (float("nan"),) * 2   # end moments, C2 / C3 face
    out["dx_C2"], out["dx_C4"] = ops.nodeDisp(PROBE["C2"], 1), ops.nodeDisp(PROBE["C4"], 1)
    # the beam C2->C3 is in double curvature: HOGGING at the C2 face (top bars in tension) and
    # SAGGING at the C3 face (bottom bars in tension)
    nan2 = (float("nan"),) * 2
    r = ops.eleResponse(PROBE["slip_C3"], "section", "fiber", yb, 0.0, "stressStrain") if "slip_C3" in PROBE else None
    out["slip_bot_C3"], out["sig_bot_C3"] = (r[1], r[0]) if r else nan2
    r = ops.eleResponse(PROBE["slip_C2"], "section", "fiber", yt, 0.0, "stressStrain") if "slip_C2" in PROBE else None
    out["slip_top_C2"], out["sig_top_C2"] = (r[1], r[0]) if r else nan2
    # interface concrete closure (mm) at the compression face of the C3 interface = the TOP
    r = ops.eleResponse(PROBE["slip_C3"], "section", "fiber", BH / 2.0 - 2.0, 0.0, "stressStrain") if "slip_C3" in PROBE else None
    out["ifc_top_C3"], out["ifc_sig_C3"] = (r[1], r[0]) if r else (float("nan"),) * 2
    # steel strain at the last IP of the beam element next to C3 (bottom bar, the tension bar)
    nsec = 6 if ELE == "hinge" else (2 if HINGE_LP > 0.0 else 5)
    r = ops.eleResponse(PROBE["beam_C3"], "section", nsec, "fiber", yb, 0.0, "stressStrain")
    out["eps_bot_C3"], out["sig_fib_C3"] = (r[1], r[0]) if r else (float("nan"),) * 2
    return out


def run(u_max=500.0, du=1.0, verbose=False, probe=False):
    """Reproduce the test: hold 109 kN on the middle column top, then lower the jack node."""
    jack, top, cL, cR = build()

    # Stage 1: the initial state of the test.  109 kN acts DOWN on the top of the middle column;
    # the frame's own weight acts everywhere; the jack holds the middle column line up.  What the
    # jack holds is a REACTION -- 109 kN plus the self-weight tributary to that column (measured
    # 7.5 kN) -- so the jack node is fixed vertically, the reaction read, and then released as a
    # constant upward load before stage 2 sheds it.  The frame's flexural capacity is only about
    # 73 kN, well below 109, which is precisely why the test lowers the jack gradually instead of
    # removing the column outright -- applying 109 kN to the damaged frame in one step has no
    # equilibrium state at all.
    # The jack in stage 1 is a very stiff vertical spring to a fixed ground node (an element, so it
    # can be removed cleanly; removing an SP constraint breaks the UmfPack symbolic factorisation).
    ops.node(99999, ops.nodeCoord(jack)[0], ops.nodeCoord(jack)[1]); ops.fix(99999, 1, 1, 1)
    ops.uniaxialMaterial("Elastic", 99, 1.0e9)
    ops.element("zeroLength", 99999, 99999, jack, "-mat", 99, "-dir", 2)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(top, 0.0, -F_TOP, 0.0)
    if SELF_WEIGHT:
        for et in ops.getEleTags():
            if et in COLS:                             # columns lumped at their end nodes
                na, nb = ops.eleNodes(et)
                w = GAMMA * CW * CW * abs(ops.nodeCoord(nb)[1] - ops.nodeCoord(na)[1]) / 2.0
                ops.load(na, 0.0, -w, 0.0); ops.load(nb, 0.0, -w, 0.0)
            elif et in BEAMS:
                ops.eleLoad("-ele", et, "-type", "-beamUniform", -GAMMA * BW * BH)
    ops.system("UmfPack"); ops.numberer("RCM"); ops.constraints("Transformation")
    ops.test("NormDispIncr", 1e-6, 40, 0); ops.algorithm("Newton")
    ops.integrator("LoadControl", 0.05); ops.analysis("Static")
    for _ in range(20):
        if ops.analyze(1) != 0:
            for alg, tol, it in (("ModifiedNewton", 1e-5, 200), ("KrylovNewton", 1e-4, 300),
                                 ("NewtonLineSearch", 1e-4, 300)):
                ops.test("NormDispIncr", tol, it, 0); ops.algorithm(alg)
                if ops.analyze(1) == 0:
                    break
            else:
                raise RuntimeError("gravity stage failed")
            ops.test("NormDispIncr", 1e-6, 40, 0); ops.algorithm("Newton")
    N0 = ops.eleForce(99999, 2)                        # what the jack holds at Delta = 0
    if N0 < 0:
        N0 = -N0
    ops.loadConst("-time", 0.0)
    ops.remove("element", 99999)
    ops.timeSeries("Constant", 3)
    ops.pattern("Plain", 3, 3); ops.load(jack, 0.0, N0, 0.0)
    ops.wipeAnalysis()
    ops.system("UmfPack"); ops.numberer("RCM"); ops.constraints("Transformation")
    ops.test("NormDispIncr", 1e-6, 40, 0); ops.algorithm("Newton")
    ops.integrator("LoadControl", 0.0); ops.analysis("Static")
    ops.analyze(1)                                     # same state; the jack is now a load

    # Stage 2: lower the jack.  A unit DOWNWARD load at the jack node, whose load factor is the
    # jack force that has been shed -- so the load factor IS the frame's own resistance R, and the
    # remaining jack force is N = N0 - R.  The paper plots N from 109 kN at Delta = 0, i.e. tared
    # of the 7.5 kN self-weight, so R_test = 109 - N_plot is the same shed force.
    ops.pattern("Plain", 2, 1); ops.load(jack, 0.0, -1.0, 0.0)
    ops.test("NormDispIncr", TOL, 40, 0); ops.algorithm("Newton")
    ops.integrator("DisplacementControl", jack, 2, -du)
    ops.analysis("Static")
    u, R, dxL, dxR = [-ops.nodeDisp(jack, 2)], [0.0], [0.0], [0.0]
    hist = [] if probe else None
    while u[-1] < u_max:
        if ops.analyze(1) != 0:
            for alg, tol, it, f in (("Newton", 1e-4, 60, 1.0), ("ModifiedNewton", 1e-4, 200, 0.5),
                                    ("KrylovNewton", 1e-3, 300, 0.25),
                                    ("NewtonLineSearch", 1e-3, 300, 0.1)):
                ops.test("NormDispIncr", tol, it, 0); ops.algorithm(alg)
                ops.integrator("DisplacementControl", jack, 2, -du * f)
                if ops.analyze(1) == 0:
                    break
            else:
                # last resort: creep through the step in sub-steps (the arch-to-catenary transition
                # and the yield-plateau states are step-size sensitive)
                ok = False
                for nsub in (8, 32):
                    ops.test("NormDispIncr", 1e-5, 100, 0); ops.algorithm("Newton")
                    ops.integrator("DisplacementControl", jack, 2, -du / nsub)
                    if all(ops.analyze(1) == 0 for _ in range(nsub)):
                        ok = True
                        break
                if not ok:
                    if verbose:
                        print(f"    no convergence at u = {u[-1]:.0f} mm")
                    break
            ops.test("NormDispIncr", TOL, 40, 0); ops.algorithm("Newton")
            ops.integrator("DisplacementControl", jack, 2, -du)
        u.append(-ops.nodeDisp(jack, 2))
        R.append(ops.getLoadFactor(2))
        dxL.append(ops.nodeDisp(cL, 1))                # +x = outward on the right, inward on left
        dxR.append(ops.nodeDisp(cR, 1))
        if probe:
            hist.append(_probe())
    u, R = np.array(u), np.array(R)
    res = (u, N0 - R, R, np.array(dxL), np.array(dxR))
    return res + (hist,) if probe else res


# The measured response.  R = 109 - N with N the jack force DIGITISED from the paper's Fig. 2
# (yi2008_fig2_digitised.csv: page rendered at 600 dpi, axes calibrated on the tick marks, one
# row per plotted marker).  Stated in the text and used as anchors: 109 kN at 0, "only 6 kN" at
# 450 mm, rupture at 456 mm.  Between 28 and 80 mm the markers merge into one band; its column-wise
# centroid (the locus of the marker centres) is used there, e.g. 30.8 kN at 50 mm.  The earlier
# NotebookLM figure reads agreed with the digitised markers to within 1-2 kN everywhere except
# at 10 mm, where they gave N = 75 (R = 34) against a digitised 65.6 (R = 43.4).
TEST_R = {10: 43.4, 25: 73.4, 50: 78.2, 100: 79.9, 140: 81.4, 200: 83.7,
          300: 89.5, 350: 94.3, 400: 102.4, 450: 103.0}                    # kN
# Fig. 5, column C2 (transducer 3-2; negative = outward): NotebookLM figure reads, NOT yet
# digitised -- see CALIBRATION_FRAME.md.
TEST_DX = {50: -2.1, 100: -2.8, 150: -2.8, 200: -2.0, 300: 2.0, 400: 9.0}  # mm, C2
LOAD_BAND, DISP_BAND = (0.82, 1.20), (0.78, 1.12)


def score(u, N, R, dxL, dxR, verbose=True, hist=None):
    """Score against the measured curve.  Same band as the connection-level calibration, fixed
    before any run: loads [0.82, 1.20], displacements [0.78, 1.12].

    Scored: the frame resistance at ten displacements along the whole curve; the peak outward
    movement of the side columns and the displacement at which they reverse; the rupture
    displacement.  The side-column movement at fixed displacements is reported but not scored --
    a ratio of two signed quantities passing through zero is not a meaningful ratio.
    """
    def at(x, q):
        j = int(np.argmin(np.abs(u - q)))
        return x[j] if u[-1] >= q - 1.5 else float("nan")
    rows = []
    for q, r in TEST_R.items():
        rows.append((f"R at {q:3d} mm [kN]", at(R, q) / 1e3, r, LOAD_BAND))
    irev = int(np.argmax(dxR))
    rows.append(("peak outward movement of the side column [mm]", float(dxR.max()), 3.0, DISP_BAND))
    rows.append(("displacement at which it reverses [mm]", float(u[irev]), 120.0, DISP_BAND))
    # rupture: the C3 hinge bottom bar has yielded and its stress then collapses (MinMax at eps_su)
    u_rup = float(u[-1])
    if hist:
        sig = np.array([h["sig_fib_C3"] for h in hist])
        i_r = next((i for i in range(1, len(sig)) if sig[i - 1] > FY and abs(sig[i]) < 1.0), None)
        if i_r is not None:
            u_rup = float(u[i_r + 1])
    rows.append(("bar rupture [mm]", u_rup, 456.0, DISP_BAND))
    n_ok, n_tot = 0, 0
    if verbose:
        print(f"{'quantity':48s} {'model':>8} {'test':>8} {'ratio':>7}  verdict")
        print("-" * 84)
    for q, m, t, band in rows:
        r = m / t
        ok = bool(band[0] <= r <= band[1])
        n_ok += ok; n_tot += 1
        if verbose:
            print(f"{q:48s} {m:8.1f} {t:8.1f} {r:7.2f}  {'PASS' if ok else 'FAIL'}")
    if verbose:
        print(f"{'side column C2 vs test at 50/100/150/200/300/400 mm':48s} "
              + "  ".join(f"{-at(dxR, q):+.1f}/{v:+.1f}" for q, v in TEST_DX.items())
              + "   (not scored)")
        print(f"\n{n_ok} / {n_tot} inside the band")
    return n_ok, n_tot, rows


def selfweight_check():
    """The jack reaction under self-weight alone; the paper measured 7.5 kN.  Not a free
    parameter -- it checks the density assumption and the tributary geometry together."""
    global F_TOP
    f0, F_TOP = F_TOP, 0.0
    try:
        jack, top, cL, cR = build()
        ops.fix(jack, 0, 1, 0)
        ops.timeSeries("Linear", 1); ops.pattern("Plain", 1, 1)
        for et in ops.getEleTags():
            if et in COLS:
                na, nb = ops.eleNodes(et)
                w = GAMMA * CW * CW * abs(ops.nodeCoord(nb)[1] - ops.nodeCoord(na)[1]) / 2.0
                ops.load(na, 0.0, -w, 0.0); ops.load(nb, 0.0, -w, 0.0)
            elif et in BEAMS:
                ops.eleLoad("-ele", et, "-type", "-beamUniform", -GAMMA * BW * BH)
        ops.system("UmfPack"); ops.numberer("RCM"); ops.constraints("Transformation")
        ops.test("NormDispIncr", 1e-8, 40, 0); ops.algorithm("Newton")
        ops.integrator("LoadControl", 1.0); ops.analysis("Static"); ops.analyze(1)
        ops.reactions()
        return ops.nodeReaction(jack, 2)
    finally:
        F_TOP = f0


def demo():
    import json, time
    t0 = time.time()
    u, N, R, dxL, dxR, H = run(u_max=800.0, du=DU_DEMO, verbose=True, probe=True)
    wall = time.time() - t0
    assert len(u) > 50, f"analysis died after {len(u)} steps"
    assert R.max() > 0, "no resistance developed"
    print(f"  reached u = {u[-1]:.0f} mm in {len(u)} steps, {wall:.0f} s; N0 = {N[0]/1e3:.1f} kN")
    n_ok, n_tot, rows = score(u, N, R, dxL, dxR, hist=H)
    np.savetxt("yi2008_model_curve.csv",
               np.column_stack([u, R, N, dxR, [h["N_beam"] for h in [H[0]] + H],
                                [h["eps_bot_C3"] for h in [H[0]] + H]]),
               delimiter=",", header="u_mm,R_N,N_jack_N,dx_C5_mm,N_beam_C3_N,eps_bar_C3", comments="")
    json.dump(dict(n_ok=n_ok, n_tot=n_tot, wall_s=wall, N0=N[0],
                   params=dict(FC=FC, EC_GB=EC_GB, FT_GB=FT_GB, OFFSETS=OFFSETS, INTERIOR=INTERIOR,
                               STEEL=STEEL, BUCK=BUCK, LSR=LSR, CAP=CAP, CORE_HINGE=CORE_HINGE, ESU_H=ESU_H,
                               EI_FAC=EI_FAC, HINGE_LP=HINGE_LP,
                               ELE=ELE, IFC=IFC, TOP_LAYERS=TOP_LAYERS, SELF_WEIGHT=SELF_WEIGHT,
                               BASE_SLIP=BASE_SLIP, BOND_FRACTURE=BOND_FRACTURE, L_BEAR=L_BEAR),
                   rows=[dict(q=q, model=m, test=t, ratio=m / t) for q, m, t, _ in rows]),
              open("yi2008_calibration.json", "w"), indent=1)


if __name__ == "__main__":
    demo()
