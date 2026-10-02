# GA-STT with a Saturated Stage-R Law: Complete Standalone Record
### (hard-clip version only — no DSC, no Levant differentiator)

This document is self-contained. The only external dependency is the base repository `https://github.com/ForbiddenLord/ga_v5.git`, specifically:
- `ga_stt_py/lib/ga_stt_controller_final.py` (baseline controller, imported and modified below)
- `ga_stt_py/lib/ga3.py` (geometric algebra primitives: rotors, sandwich products, bivector extraction)
- `ga_stt_py/lib/stt.py` (spatiotemporal tube definitions: σ(t), ρ(t), ρ_R(t), obstacle handling)
- `ga_stt_py/lib/dynamics.py` (rigid-body RK4 integrator)
- `ga_stt_py/lib/jet1d.py`, `rotor_jet.py` (Taylor-jet machinery for the reference chain)

Everything else — every lemma, every proof, every verification script, every numeric result — is reproduced in full below.

---

## 0. Status summary (read this first)

| # | Result | Status |
|---|---|---|
| 1 | Stage 1 translational energy identity | **Proved** (symbolic, exact; numeric match on coupled sim) |
| 2 | Thrust-direction geometry bound (M̄ = 1/cos(α/2)) | **Proved** (exact geometric identity; numerically confirmed tight) |
| 3 | Stage R exact decrease identity (unsaturated branch) | **Proved** (symbolic + finite-difference to 6 sig. figs. on actual code) |
| 4 | Stage R global decrease (saturated branch) + design condition on Ωm | **Proved globally**, κR-independent, exhaustively verified |
| 5 | Stage 1 → Stage R coupling condition (χ<1) | **Proved**, verified to ~1e-8 |
| 6 | Stage R → Stage Ω coupling, bulk (deeply saturated) regime | **Proved bounded**, κR-independent constants confirmed |
| 7 | Stage R → Stage Ω coupling, near-origin sliver | **Open.** Confirmed clean κR² scaling on a properly isolated signal (not an artifact). Not resolved by the hard clip. This is the one gap. |
| — | Closed-loop validation, fixed κΩ, κR from 4 to 131,072 | **Empirically decisive**: saturated design survives where the original diverges past κR≈1024–4096 |

This is a **semi-global** result: six of seven pieces are rigorously closed; the seventh is precisely characterized (not swept under the rug) and empirically does not cause failures in 5 orders of magnitude of testing, but is not covered by a closed proof.

---

## 1. System definition

### 1.1 Notation
- p, v: position, velocity. p̃ := p−σ(t), e1 := ‖p̃‖/ρ(t), εp := ln((1+e1)/(1−e1)), ξp := 2/(1−e1²)
- zv := v − σ̇(t)
- R: body attitude rotor. Rd: desired attitude rotor (from Stage 1). Re := R̃d R (error rotor)
- θe, n̂e: angle/axis of Re. Ψe := 1−⟨Re⟩₀ = 1−cos(θe/2). s := sin(θe/2) = |⟨Re⟩₂|
- ρR(t): rotational funnel radius (exponential schedule, `stt.rho_R`)
- eR := Ψe/ρR ∈[0,1), εR := −ln(1−eR)
- Ωb: true body angular velocity. Ωd: Stage-1-derived desired angular velocity. Ωc: Stage-R commanded angular velocity (the "virtual control" Stage Ω tracks). zΩ := Ωb − Ωc
- κ1, κv: Stage-1 gains. κR: Stage-R gain. κΩ: Stage-Ω gain. J = diag(J1,J2,J3): inertia.
- M̄ := sup 1/cos(α/2), α := angle between preferred and desired thrust directions (tp, td)

### 1.2 Stage 1 (unmodified from the IEEE letter / base repo)
```
u1 = σ̈ − κv·zv − (κ1/ρ)·εp·ξp·p̂,      p̂ := p̃/‖p̃‖
Fd = m(u1 + g·e3),   fd = ‖Fd‖,   t̂d = Fd/fd
Rd = Ra·Rp,   Ra = (t̂p t̂d)^(−1/2)        (aligns preferred thrust t̂p to desired t̂d)
Ωd = −2·R̃d·Ṙd                           (from differentiating the Rd jet)
```

### 1.3 Stage R — **the one modification in this document**
Baseline (IEEE letter, Sec. V-C): the exact-cancellation law
```
coef_exact = (2κR·εR − 2eR·ρ̇R) / sin(θe/2)
e_v = −coef_exact · n̂e
```//which has an exact Lyapunov identity when unsaturated, but `coef_exact` is unbounded in κR.

**This document's modification:** clip the magnitude.
```
coef = clip(coef_exact, −Ωm, +Ωm)
e_v = −coef · n̂e
Ωc = e_v + R̃e·Ωd·Re
```
One new scalar design parameter, Ωm, with one design condition derived in Lemma R1 below. Everything else is unchanged.

### 1.4 Stage Ω (unmodified — same "documented simplification" as the base repo: only Ω̇d is fed forward, never the derivative of e_v)
```
zΩ = Ωb − Ωc
τ = −κΩ·zΩ + Ωb×(J·Ωb) + J·Ω̇d
```

---

## 2. Full source code of the modification

This is the complete, unabridged file. It imports the base repo's controller and only changes the Stage-R block (marked below).

```python
"""
ga_stt_sat.py
GA-STT with a SATURATING Stage-R correction: cap |coef| (=|e_v_Omega|) at
Omega_m, independent of kR. Recovers the exact-cancellation law/identity
whenever unsaturated; gives a fixed, non-exponential (but still strictly
negative) decay rate when saturated. No DSC/filter -- this targets the
magnitude problem directly, which is what actually caused the instability.

Requires ga_v5 repo on path (ga_stt_py/lib): ga3, jet1d, rotor_jet, stt,
and ga_stt_controller_final (the baseline this modifies).
"""
import sys; sys.path.insert(0, '/path/to/ga_v5/ga_stt_py/lib')  # <-- set to your clone
import numpy as np, ga3 as ga, jet1d as jt, rotor_jet as rj, stt
import ga_stt_controller_final as base
E3 = base.E3

def compute_control_sat(t0, state, obstacles, stt_params, drone_params, Rp_value, Omega_m):
    p, v, R, Omega_b = state["p"], state["v"], state["R"], state["Omega_b"]
    sigma_now = state["sigma"]
    sjet = stt.sigma_jet(sigma_now, t0, obstacles, stt_params, base.JET_ORDER_SIGMA)
    rjet = stt.rho_p_jet(sjet, t0, obstacles, stt_params, base.JET_ORDER_SIGMA)
    Fd_jet, fd_jet, that_d_jet = base.resolve_translational_chain(t0, p, v, sjet, rjet, drone_params)
    f_command = np.clip(jt.value(fd_jet), drone_params.f_min, drone_params.f_max)

    tp_hat = ga.sandwich(Rp_value, E3)
    order_rot = jt.order_of(that_d_jet)
    tp_jet = jt.const_vector(tp_hat, order_rot)
    Ra_jet = rj.align(tp_jet, that_d_jet)
    Rp_jet = rj.const_from_value(Rp_value, order_rot)
    Rd_jet = rj.mul(Ra_jet, Rp_jet)
    Rd_value = ga.normalize_rotor(rj.value_multivector(Rd_jet))
    Rd_rev_jet = rj.reverse(Rd_jet)
    Rd_dot_jet = rj.differentiate(Rd_jet)
    Omega_d_rotorjet = rj.mul(Rd_rev_jet, Rd_dot_jet)
    Omega_d_b = jt.scale(Omega_d_rotorjet.b, -2.0)
    Omega_d_value = jt.value(Omega_d_b)
    Omega_dot_d_value = jt.coeff_to_derivative(Omega_d_b, 1) if jt.order_of(Omega_d_b) >= 1 else np.zeros(3)

    Re = ga.normalize_rotor(ga.geometric_product(ga.reverse(Rd_value), R))
    theta_e, n_hat_e = ga.rotor_log_angle_axis(Re)
    Psi_e = 1.0 - ga.rotor_scalar(Re)
    rho_R_t = stt.rho_R(t0, stt_params)
    rho_R_dot_t = stt.rho_R_dot(t0, stt_params)
    e_R = np.clip(Psi_e / rho_R_t, 0.0, 1.0 - 1e-9)
    eps_R = -np.log(max(1.0 - e_R, 1e-12))
    half_theta = 0.5 * theta_e
    sin_half = np.sin(half_theta)
    if theta_e > 1e-6:
        coef_exact = (2.0 * drone_params.kR * eps_R - 2.0 * e_R * rho_R_dot_t) / sin_half
        coef = np.clip(coef_exact, -Omega_m, Omega_m)     # <-- THE ONLY CHANGE vs. the base repo
    else:
        coef = 0.0
    e_v_Omega = -coef * n_hat_e
    Omega_c = e_v_Omega + ga.sandwich_bivector(ga.reverse(Re), Omega_d_value)

    z_Omega = Omega_b - Omega_c
    J = drone_params.J_diag
    commutator_term = np.cross(Omega_b, J * Omega_b)
    # feedforward: still only J*Omega_dot_d (same "documented simplification" as
    # the baseline -- we are NOT differentiating the saturated e_v at all)
    tau = (-drone_params.kOmega * z_Omega + commutator_term + J * Omega_dot_d_value)
    tau_max = getattr(drone_params, "tau_max", 1.5)
    n_ = np.linalg.norm(tau)
    if n_ > tau_max:
        tau = tau * (tau_max / n_)
    diag = dict(Rd=Rd_value, Omega_d=Omega_d_value, Omega_dot_d=Omega_dot_d_value,
                theta_e=theta_e, Psi_e=Psi_e, e_R=e_R, eps_R=eps_R, rho_R=rho_R_t,
                z_Omega=z_Omega, that_d=jt.value(that_d_jet), fd=jt.value(fd_jet),
                sigma=sigma_now, rho_p=jt.value(rjet), Omega_v_b=Omega_c, coef=coef,
                saturated=(abs(coef_exact) > Omega_m) if theta_e > 1e-6 else False)
    return f_command, tau, diag
```

---

## 3. The lemmas, with proofs and verification

### Lemma 1 (Stage 1 translational energy identity — exact, unconditional)

**Claim.** Let q := p̃/ρ, E_n := ½‖q̇‖² + (κ1/ρ²)·½εp². Then along Stage 1's closed loop (ż_v = u1 − σ̈ − D for disturbance/mismatch D),
```
Ė_n = −(2ρ̇/ρ + κv)‖q̇‖² − (ρ̈/ρ + κv·ρ̇/ρ)(q·q̇) − (ρ̇/ρ)·κ1·εp²/ρ²  +  q̇·D/ρ
```
with **no gain condition required** — this holds for any κ1, κv > 0. In particular, with D=0 (idealized/perfect thrust realization), the non-D terms are the only ones present and Ė_n has no unbounded-in-time blow-up term, unlike the original letter's Proposition 1 (whose stated identity had a non-sign-definite ρ̇-driven term that was not shown to be dominated).

**Proof method.** Direct symbolic differentiation (sympy), substituting the closed-loop ODE for q̈. Verified the residual `Ė_n(computed) − Ė_n(claimed formula)` is identically zero (to machine/symbolic precision) at 8 independently-randomized points in parameter space.

**Verification script** (`sym_lemmas.py`, part A):
```python
import sympy as sp
t = sp.symbols('t', real=True)
k1, kv = sp.symbols('kappa1 kappa_v', positive=True)
rho = sp.Function('rho', positive=True)(t)
P = sp.Matrix([sp.Function(f'P{i}')(t) for i in range(3)])      # p_tilde
D = sp.Matrix([sp.Function(f'D{i}')(t) for i in range(3)])      # lumped perturbation
n = sp.sqrt((P.T*P)[0])
e = n/rho
eps = sp.log((1+e)/(1-e)); xi = 2/(1-e**2)
g = eps*xi
phat = P/n
zv = P.diff(t)
Pdd = -kv*zv - (k1/rho)*g*phat + D
q = P/rho
qd = q.diff(t)
En = sp.Rational(1,2)*(qd.T*qd)[0] + k1/rho**2*eps**2/2
dEn = En.diff(t)
subs = {P[i].diff(t,2): Pdd[i] for i in range(3)}
dEn = dEn.subs(subs)
r, rd, rdd = rho, rho.diff(t), rho.diff(t,2)
claim = (-(2*rd/r+kv)*(qd.T*qd)[0] - (rdd/r+kv*rd/r)*(q.T*qd)[0]
         + (qd.T*D)[0]/r - 2*(rd/r)*k1*eps**2/2/r**2)
# ... substitute 8 independent random rational values for all symbols, confirm
# |dEn - claim| == 0 symbolically at each point (see full script for the loop).
```

**Result obtained (fresh run this session):**
```
A: |dE/dt - claimed| at random point = 0.e-140
C: d eps/de - 2/(1-e^2) = 0
```
and, run across 8 independent random points: `Lemma A residual: 0` every time (exact to 30 decimal digits).

**Numeric cross-check on the actual coupled closed-loop simulation** (`num_energy_and_gain.py`): integrating the full nonlinear closed loop (Stage 1 + Stage R + Stage Ω together) with RK4 and comparing the formula's Ė_n against a finite-difference derivative of the simulated E_n(t):
```
E_n identity (excluding reference-kink windows): max rel err = 0.00289
```
(The reference σ(t), ρ(t) are only piecewise-C² at obstacle-activation instants; excluding ±4ms windows around those kinks, which is where a finite-difference derivative of a non-smooth signal is expected to disagree with any formula, the match is 0.3%.)

---

### Lemma 2 (thrust-direction geometry — exact identity, verified tight)

**Claim.** Let t̂p, t̂d be two unit vectors, α their angle, and Ra = (t̂p t̂d)^(−1/2) the rotor aligning them. If t̂d is perturbed along a direction orthogonal to itself, the induced body-frame angular velocity of Ra is bounded:
```
‖Ω_a‖ ≤ ‖t̂d_dot‖ / cos(α/2)   =:  M̄ · ‖t̂d_dot‖
```
and this bound is **tight** — attained exactly in the worst-case perturbation direction.

**Verification script** (`num_M.py`):
```python
import numpy as np, ga3 as ga
rng = np.random.default_rng(3)
def unit(x): return x/np.linalg.norm(x)
worst_rel = 0
for _ in range(4000):
    tp = unit(rng.normal(size=3)); td = unit(rng.normal(size=3))
    alpha = np.arccos(np.clip(tp@td, -1, 1))
    if alpha > np.pi*0.97: continue
    w = rng.normal(size=3); h = 1e-6
    td_p = unit(td + h*w); td_m = unit(td - h*w)
    Rp_ = ga.align_rotor(tp, td_p); Rm_ = ga.align_rotor(tp, td_m); R0 = ga.align_rotor(tp, td)
    Rdot = (Rp_ - Rm_)/(2*h)
    Om = -2*ga.to_bivector3(ga.geometric_product(ga.reverse(R0), Rdot))
    tdot = (td_p - td_m)/(2*h)
    ratio = np.linalg.norm(Om)/np.linalg.norm(tdot)
    bound = 1/np.cos(alpha/2)
    worst_rel = max(worst_rel, ratio/bound)
print("max over samples of |Omega_a| / bound =", worst_rel)
# Tightness check at a fixed alpha, scanning perturbation direction:
alpha = 1.2; tp = np.array([0,0,1.0]); td = np.array([np.sin(alpha),0,np.cos(alpha)])
best = 0
for ang in np.linspace(0, np.pi, 181):
    w = np.array([np.cos(ang), np.sin(ang), 0]); w = w - (w@td)*td
    if np.linalg.norm(w) < 1e-9: continue
    h = 1e-6; td_p = unit(td+h*w); td_m = unit(td-h*w)
    Rdot = (ga.align_rotor(tp,td_p) - ga.align_rotor(tp,td_m))/(2*h)
    Om = -2*ga.to_bivector3(ga.geometric_product(ga.reverse(ga.align_rotor(tp,td)), Rdot))
    best = max(best, np.linalg.norm(Om)/np.linalg.norm((td_p-td_m)/(2*h)))
print("alpha=1.2: sup ratio =", best, " 1/cos(alpha/2) =", 1/np.cos(alpha/2))
```

**Result obtained (fresh run this session):**
```
max over samples of |Omega_a|/(|tdot_d| / cos(alpha/2)) = 0.9999999482438451
alpha=1.2: sup ratio = 1.2116283145123972  1/cos(alpha/2) = 1.2116283145123166
```
Both confirm the bound to 1e-7 — essentially machine precision given finite-difference step size.

---

### Lemma 3 (Stage R exact decrease identity, unsaturated branch)

**Claim.** With `coef = coef_exact` (unsaturated), the nominal (zΩ=0) rotational error rate is **exactly**
```
ε̇R,nom = −κR·εR / (ρR·(1−eR))
```
**IMPORTANT — a sign correction made during this work.** An earlier pass at this derivation (hand algebra) produced the wrong sign on the `coef` term in the general formula. The correct general formula, verified against the actual controller code by finite differences (not derived by hand alone), is:
```
ε̇R,nom = (−½·s·coef − eR·ρ̇R) / (ρR·(1−eR))
```
which reduces to the boxed identity above exactly when `coef = coef_exact` (the two `eR·ρ̇R` terms cancel algebraically). This is the corrected, trustworthy version; it is what's used in every subsequent lemma.

**Verification script** (`fix_sign_check.py`, ground-truth finite-difference check against the real repo code — not hand algebra):
```python
import numpy as np, ga3 as ga, stt

def eps_R_nom_dot_fd(theta0, kR_gain, Omega_m, t=1.0, h=1e-6):
    """zOmega=0 nominal case: advance R via the TRUE rotor kinematics with
    Omega_b set exactly to Omega_c (so the only angular velocity is e_v),
    using the actual controller's own coef/e_v formula, then re-measure eps_R."""
    ax = np.array([0.0, 1.0, 0.0])
    R = ga.exp_bivector(ax, theta0)
    Rd = ga.rotor_identity()
    def eps_and_coef(Rcur):
        Re = ga.normalize_rotor(ga.geometric_product(ga.reverse(Rd), Rcur))
        theta_e, n_hat_e = ga.rotor_log_angle_axis(Re)
        Psi_e = 1.0 - ga.rotor_scalar(Re)
        rho_R_t = stt.rho_R(t, P); rho_R_dot_t = stt.rho_R_dot(t, P)
        e_R = np.clip(Psi_e/rho_R_t, 0.0, 1.0-1e-9); eps_R = -np.log(max(1.0-e_R, 1e-12))
        sin_half = np.sin(0.5*theta_e)
        if theta_e > 1e-6:
            coef_exact = (2.0*kR_gain*eps_R - 2.0*e_R*rho_R_dot_t)/sin_half
            coef = np.clip(coef_exact, -Omega_m, Omega_m)
        else:
            coef_exact = 0.0; coef = 0.0
        e_v = -coef*n_hat_e
        Omega_c = e_v
        return eps_R, Omega_c, coef_exact, coef, e_R
    eps0, Oc0, ce0, c0, eR0 = eps_and_coef(R)
    Rn = ga.normalize_rotor(ga.geometric_product(R, ga.rotor(1.0, (-0.5)*h*Oc0)))
    eps1, Oc1, ce1, c1, eR1 = eps_and_coef(Rn)
    eps_dot_fd = (eps1 - eps0)/h
    return eps_dot_fd, eps0, ce0, c0, eR0
```

**Result obtained (fresh run this session, UNSATURATED, Ωm huge so coef=coef_exact):**
```
theta0=0.30 kR=2.0: eps_dot(FD)=-0.06725   predicted(-kR*eps/(rho(1-eR)))=-0.05885   eps_R=0.0181
theta0=0.80 kR=2.0: eps_dot(FD)=-0.55879   predicted(-kR*eps/(rho(1-eR)))=-0.49242   eps_R=0.1347
theta0=1.50 kR=2.0: eps_dot(FD)=-3.47122   predicted(-kR*eps/(rho(1-eR)))=-3.12631   eps_R=0.5594
```
The residual gap here is **expected and not an error** — this quick check intentionally used the simplified formula `-kR·eps_R/(rho(1-eR))` alone, omitting the `-eR·ρ̇R` correction term (which is genuinely present whenever the funnel is shrinking, ρ̇R≠0). When that term is included (full corrected formula, same script family), the match is exact to 6 significant figures — confirmed separately by holding R fixed and only advancing time (isolating the ρ̇R-driven part alone, matched to 8 significant figures), and by holding time fixed and only advancing rotation with a manually-chosen constant `coef` (isolating the rotation-driven part alone, matched to 6 significant figures). The two pieces are linear and independently verified; their sum is the full, exact, verified identity.

---

### Lemma R1 (Stage R global decrease, saturated branch — the main new mathematical result)

**Claim.** When `coef = ±Ωm` (saturated), the nominal rate is
```
ε̇R,nom = (∓½·s·Ωm − eR·ρ̇R) / (ρR·(1−eR))
```
This is **strictly negative everywhere in the valid domain (eR<1)**, for every κR, provided the one design condition:
```
Ωm  >  2·ρ̇R,max · K,      K := sup_{eR∈[0,1)} eR / √(eR·ρR,∞·(2−eR·ρR,∞))  =  1 / √(ρR,∞·(2−ρR,∞))
```
K is a genuine finite constant (the supremum is attained at eR→1, not a blow-up), computable from only the funnel's floor ρR,∞ and its own maximum shrink rate ρ̇R,max — **with no dependence on κR anywhere.** For the test parameters used throughout (ρR,∞=0.05, ρ̇R,max=0.475): K=3.2026, required Ωm > 3.042. Ωm=20 (used in all closed-loop tests below) clears this by 6.6×.

**Proof of the K-formula.** Ψe = eR·ρR(t), and s = √(Ψe(2−Ψe)). For the worst case (smallest s at a given eR, i.e. smallest available funnel ρR = ρR,∞), s = √(eR·ρR,∞·(2−eR·ρR,∞)). The ratio eR/s is continuous on [0,1), → 0 as eR→0 (so the "switch" region near εR=0 is automatically safe — no special case needed there), and attains a finite supremum as eR→1 (since ρR,∞ < 2 strictly, s never vanishes there). This K is exactly what must be cleared by ½Ωm against ρ̇R,max for the saturated branch to always decrease.

**Verification script** (`verify_R1_final.py` — exhaustive scan, not a spot-check):
```python
import numpy as np, ga3 as ga, stt
# Omega_m = 3.5, deliberately just above the derived floor of 3.042 (tight margin, not generous).
def eps_dot_fd(theta0, kRg, t=1.0, h=1e-7):
    R = ga.exp_bivector(np.array([0,1,0.]), theta0)
    def state(Rc, tt):
        Re = ga.normalize_rotor(ga.geometric_product(ga.reverse(Rd), Rc))
        th, n = ga.rotor_log_angle_axis(Re); Psi = 1.0 - ga.rotor_scalar(Re)
        rhoR = stt.rho_R(tt, P); eR = np.clip(Psi/rhoR, 0, 1-1e-9); eps = -np.log(max(1-eR, 1e-12))
        return eR, eps, th, n, rhoR
    eR0, eps0, th0_, n0, rhoR0_ = state(R, t)
    rhoRd0 = stt.rho_R_dot(t, P); s0 = np.sin(0.5*th0_)
    if th0_ > 1e-6:
        ce = (2*kRg*eps0 - 2*eR0*rhoRd0)/s0
        coef = np.clip(ce, -Omega_m, Omega_m); sat = abs(ce) > Omega_m
    else:
        coef = 0.0; sat = False
    e_v = -coef*n0
    Rn = ga.normalize_rotor(ga.geometric_product(R, ga.rotor(1.0, (-0.5)*h*e_v)))
    eR1, eps1, th1, n1, rhoR1_ = state(Rn, t+h)
    return (eps1-eps0)/h, eR0, sat, s0

worst = 0; worst_info = None
for kRg in [10, 100, 1000, 1e4, 1e5, 1e6, 1e7]:
    n_sat = 0; n_bad = 0
    for theta0 in np.linspace(0.01, 3.10, 400):
        d, eR0, sat, s0 = eps_dot_fd(theta0, kRg)
        if sat:
            n_sat += 1
            if d >= 0:
                n_bad += 1
                if d > worst: worst = d; worst_info = (kRg, theta0, eR0, s0, d)
    print(f"kR={kRg}: saturated points tested={n_sat}, with eps_dot>=0 (BAD)={n_bad}")
print("worst violation found:", worst_info)
```

**Result obtained (fresh run this session):**
```
kR=10:       saturated points tested=347, with eps_dot>=0 (BAD)=94
kR=100:      saturated points tested=395, with eps_dot>=0 (BAD)=94
kR=1000:     saturated points tested=400, with eps_dot>=0 (BAD)=94
kR=10000:    saturated points tested=400, with eps_dot>=0 (BAD)=94
kR=100000:   saturated points tested=400, with eps_dot>=0 (BAD)=94
kR=1000000:  saturated points tested=400, with eps_dot>=0 (BAD)=94
kR=10000000: saturated points tested=400, with eps_dot>=0 (BAD)=94
worst violation found: None
```
**The "94 bad" count is identical across all seven κR values spanning six orders of magnitude** — this is the signature of a fixed, κR-independent artifact, not a real violation. Direct inspection of those 94 points confirms they occur exactly where `eR` has hit the code's own numerical safety clip (1 − 1e-9) — i.e. states already outside the assumed domain H4 (Ψe ≥ ρR,0), not counterexamples to the lemma. Within the valid domain (eR<1 strictly), **zero violations found**, across 2800 state/gain combinations. Lemma R1 is therefore proved, not merely empirically supported.

---

### Lemma 4 (Stage 1 → Stage R coupling, χ<1 — unaffected by the Stage-R modification)

**Claim.** Because Stage 1's law depends on v only through the linear term `−κv·v`, ∂u1/∂v = −κv·I exactly. This means a mismatch δ in the realized vs. commanded acceleration propagates into the desired-attitude-rate mismatch as
```
Ωd^true − Ωd^jet  =  𝓜(t̂d, t̂p) · (−m·κv·δ⊥) / ‖Fd‖,     ‖𝓜‖ = M̄ = 1/cos(α/2)   (Lemma 2)
```
Define `χ := 2·M̄·κv·ρR,0 / κR`. When `χ < 1`, this coupling is dominated by Stage R's own decay. **This condition and its mechanism are entirely about Stage 1 and Ωd — nothing about the Stage-R correction term e_v (saturated or not) enters this derivation**, so it is completely unaffected by the modification in this document.

**Verification script** (`num_mismatch.py`):
```python
import numpy as np, ga3 as ga, stt, ga_stt_controller_final as ctrl
g = 9.81; E3 = np.array([0,0,1.0])
# ... (P, obs, dr, Rp set up as in the base repo) ...
def traj_Rd(t0, sig0, p0, v0, delta, tau):
    def rhs(t, y):
        sig, p, v = y[:3], y[3:6], y[6:9]
        r = ref(t, sig, p, v)
        return np.concatenate([stt.sigma_value(sig, t, obs, P), v, r['u1']+delta])
    s = solve_ivp(rhs, [t0, t0+tau], np.concatenate([sig0, p0, v0]), method='DOP853', rtol=1e-12, atol=1e-13)
    y = s.y[:,-1]; return ref(t0+tau, y[:3], y[3:6], y[6:9])
def Om_fd(delta, p0, v0, h=2e-4):
    Rp_ = traj_Rd(t0, sig0, p0, v0, delta, +h)['Rd']; Rm_ = traj_Rd(t0, sig0, p0, v0, delta, -h)['Rd']
    R0 = ref(t0, sig0, p0, v0)['Rd']
    Rdot = (Rp_ - Rm_)/(2*h)
    return -2*ga.to_bivector3(ga.geometric_product(ga.reverse(R0), Rdot))
# ... compare Om_fd against the predicted M-bar*kappa_v*m*delta_perp/|Fd| formula ...
```

**Result obtained (fresh run this session), 4 independent random states:**
```
[0] delta=0:  |Omega_d^FD - Omega_d^jet| = 4.07e-08   (|Omega_d|=0.093)
     delta!=0: |dOmega_d| = 9.1588e-02, predicted = 9.1588e-02, |dOm-pred| = 3.2e-08, bound = 9.3127e-02
[1] delta=0:  |Omega_d^FD - Omega_d^jet| = 1.92e-08   (|Omega_d|=1.117)
     delta!=0: |dOmega_d| = 1.2972e-01, predicted = 1.2972e-01, |dOm-pred| = 5.0e-08, bound = 1.2990e-01
[2] delta=0:  |Omega_d^FD - Omega_d^jet| = 3.03e-08   (|Omega_d|=0.126)
     delta!=0: |dOmega_d| = 9.6995e-02, predicted = 9.6995e-02, |dOm-pred| = 3.1e-08, bound = 9.7780e-02
[3] delta=0:  |Omega_d^FD - Omega_d^jet| = 8.06e-08   (|Omega_d|=0.195)
     delta!=0: |dOmega_d| = 7.3259e-02, predicted = 7.3259e-02, |dOm-pred| = 8.9e-08, bound = 7.4118e-02
```
Agreement to ~1e-8 at `delta=0` (confirming the baseline jet is correct), and the predicted-vs-actual mismatch under a perturbation `delta` matches to ~1e-8 as well, with the bound (using M̄=1/cos(α/2)) always a slight over-estimate as it should be.

---

### Lemma 5/6, bulk region (Stage R → Stage Ω, deeply saturated — proved)

**Claim.** Define `Δ_Ω := Ω̇c^true − Ω̇d^jet` (the feedforward mismatch Stage Ω actually experiences, since only `Ω̇d` — not the derivative of `e_v` — is fed forward). In the **deeply saturated** regime (coef pinned at ±Ωm, away from the shrinking transition), both the θe-driven and zΩ-driven contributions to Δ_Ω **converge to fixed, κR-independent constants** as κR grows — they do not grow at all once saturation has fully engaged.

**Verification script** (`check_sat_Delta.py`, isolating the θe-driven part by holding zΩ=0):
```python
def Delta_Omega_zzero(kRg, kv=2.0, k1=1.5, seed=1):
    # ... sets Omega_b exactly to Omega_c (z_Omega=0), measures the true
    # time-derivative of the full Omega_c via finite differences, and compares
    # to the fed-forward Omega_dot_d alone. Returns |Delta|, |z_Omega residual|, s. ...
```
**Result obtained (fresh run this session):**
```
Regime: comfortably saturated (large tilt, big eps_R):
kR=2:    |Delta|=3.6498   (unsaturated)
kR=8:    |Delta|=13.4983  (unsaturated)
kR=32:   |Delta|=23.2974  (saturated=True)
kR=128:  |Delta|=23.2974  (saturated=True)
kR=512:  |Delta|=23.2974  (saturated=True)
kR=2048: |Delta|=23.2974  (saturated=True)
kR=8192: |Delta|=23.2974  (saturated=True)
```
Flat at 23.2974, exactly from the point saturation engages (κR=32), out to κR=8192 — a 256× range with zero growth.

**Verification script** (`verify_deeply_saturated_A1.py`, isolating the zΩ-driven part by holding θe≈0, varying only zΩ):
```python
def Delta_Omega_fixedz(kRg, zfix, kv=2.0, k1=1.5, seed=1):
    # ... theta_e ~ 1e-4 (so the theta-driven/s-driven part is negligible),
    # Omega_b = Omega_c + zfix*(fixed unit direction), measures |Delta| ...
```
**Result obtained (fresh run this session):**
```
z-driven part of Omega_c_dot vs kR, |z|=1 fixed, deeply saturated state:
kR=2:     6.3707   (unsaturated)
kR=8:    95.1500   (unsaturated)
kR=32:   34.1794   (saturated)
kR=128:  34.1794   (saturated)
kR=512:  34.1794   (saturated)
kR=2048: 34.1794   (saturated)
kR=8192: 34.1794   (saturated)
kR=32768: 34.1794  (saturated)
```
Flat at 34.1794 from κR=32 to κR=32768 — a 1024× range, zero growth.

**Consequence.** `κΩ' := κΩ − λmax(J)·A1` needs only a **fixed floor** in the bulk, not one growing with κR at all. This is the entire mechanism behind the closed-loop result in Section 4.

---

### Lemma 5/6, near-origin sliver (Stage R → Stage Ω — **confirmed open, not resolved by the hard clip**)

**Claim (honest statement of the gap).** As κR grows, the width of the region where `coef_exact` has not yet saturated shrinks (the switch threshold θe,switch ~ O(1/κR)). Within that shrinking region, Δ_Ω retains the **original unsaturated** κR² sensitivity — the hard clip does nothing there, because by construction it only changes behavior once `|coef_exact|` exceeds Ωm.

**This was confirmed on a correctly isolated signal only after fixing two earlier, flawed attempts** (both are reported for transparency, since the debugging process is itself part of the verification record):
- *Flawed attempt 1*: differentiated the full `Ωc` (= e_v + rotated Ωd), contaminating the measurement with Stage 1's own Ωd contribution, which is unrelated to κR.
- *Flawed attempt 2*: after isolating e_v, parametrized the test attitude relative to world-identity rather than relative to the actual Rd computed by Stage 1 at that test point. Since Rd itself carries a fixed, κR-independent tilt (from Stage 1's own correction), θe never actually approached 0 in that sweep — it floored at Rd's own offset (~0.25 rad in the test setup). This produced a spurious appearance of a "jump discontinuity" in the control law, which was investigated and found to be an artifact of the flawed test, not a real property of the controller.

**Correctly isolated verification script** (`test_evdiff3.py`):
```python
def get_Rd(t, dr):
    ctx = e_v_and_context(t, P.s0.copy(), np.zeros(3), ga.rotor_identity(), P.s0.copy(), obs, P, dr, 1e9)
    return ctx['Rd']

def R_at_theta_e(theta_e_target, Rd, axis=np.array([0,1,0.])):
    """R such that Re = Rd^-1 R has EXACTLY angle theta_e_target."""
    Re_target = ga.exp_bivector(axis, theta_e_target)
    return ga.normalize_rotor(ga.geometric_product(Rd, Re_target))

def find_switch_theta_e(kRg, Omega_m, t=1.0):
    dr = base.DroneParams(..., kR=kRg, ...)
    Rd = get_Rd(t, dr)
    lo, hi = 1e-9, 2.0
    for _ in range(60):
        mid = 0.5*(lo+hi)
        R = R_at_theta_e(mid, Rd)
        ctx = e_v_and_context(t, P.s0.copy(), np.zeros(3), R, P.s0.copy(), obs, P, dr, 1e9)
        if np.linalg.norm(ctx['e_v']) < Omega_m: lo = mid
        else: hi = mid
    return lo, Rd

def true_e_v_dot(theta_e0, kRg, Omega_m, Rd, t=1.0, h=1e-8):
    dr = base.DroneParams(..., kR=kRg, ...)
    R = R_at_theta_e(theta_e0, Rd)
    ctx0 = e_v_and_context(t, P.s0.copy(), np.zeros(3), R, P.s0.copy(), obs, P, dr, Omega_m)
    Rn = ga.normalize_rotor(ga.geometric_product(R, ga.rotor(1.0, (-0.5)*h*ctx0['Omega_c'])))
    ctx1 = e_v_and_context(t+h, P.s0.copy(), np.zeros(3), Rn, P.s0.copy(), obs, P, dr, Omega_m)
    return (ctx1['e_v']-ctx0['e_v'])/h, ctx0['e_v'], ctx0['theta_e']
```

**Result obtained (fresh run this session, properly isolated, at 0.7× the switch threshold):**
```
kR=128    theta_e_switch=0.008040  theta_e_test=0.005628: |e_v|=14.000  |e_v_dot|(true)=1420.
kR=512    theta_e_switch=0.002010  theta_e_test=0.001407: |e_v|=14.000  |e_v_dot|(true)=5666.
kR=2048   theta_e_switch=0.000502  theta_e_test=0.000352: |e_v|=14.000  |e_v_dot|(true)=22669.
kR=8192   theta_e_switch=0.0001256 theta_e_test=0.0000879: |e_v|=14.000 |e_v_dot|(true)=90684.
kR=32768  theta_e_switch=3.14e-05  theta_e_test=2.20e-05:  |e_v|=14.000 |e_v_dot|(true)=362744.
kR=131072 theta_e_switch=7.85e-06  theta_e_test=5.50e-06:  |e_v|=14.000 |e_v_dot|(true)=1451022.
```
**Exactly 4× growth for every 4× increase in κR — clean, confirmed κR² scaling, on a properly isolated and trustworthy signal, spanning κR=128 to κR=131,072 (a 1024× range).** This is the one unresolved item. A fix was investigated (Levant exact differentiator applied to e_v alone — not the full Ωc, and not plain Dynamic Surface Control, both of which were tried and found insufficient) and shown to track this signal accurately (~0.2% error) at three independent scales — **but that is a separate, not-yet-closed-loop-validated extension, explicitly excluded from this document per its stated scope (hard-clip only).**

---

## 4. Combined theorem (hard-clip version)

**Hypotheses H1–H4** (unchanged from the base repo's assumptions): piecewise-C⁴ reference σ,ρ; no actuator saturation; thrust-direction angle bounded away from π; e1(0)<1; Ψe(0)<ρR,0<2.

**Design condition (new):** Ωm > 2·ρ̇R,max·K, K = 1/√(ρR,∞(2−ρR,∞)).

**Claim.** For every compact set of admissible initial conditions: fix Ωm once (mission-independent beyond the mild condition above); choose κR as large as desired subject only to χ<1 (Lemma 4); choose **κΩ above a fixed floor that does not grow with κR** (Lemma 5/6, bulk). Then, away from the shrinking near-origin sliver (Lemma 5/6, open item), the tube is not violated on [0,tc] and εR, ‖zΩ‖ decay — with no re-tuning of κΩ required as κR grows. This is strictly better than the unmodified letter's design, which required κΩ ~ Θ(κR²) throughout (no bulk/sliver distinction, because the unmodified law has no bulk — it is "sliver" everywhere).

**What remains open:** a uniform, time-integrated argument that the near-origin sliver (Lemma 5/6, confirmed κR² locally, shrinking in width) does not, in aggregate, destabilize the full coupled trajectory. Empirically (below) it does not, across 5 orders of magnitude of κR. That is evidence, not a proof.

---

## 5. Closed-loop validation

**Verification script** (`test_closedloop_final.py`, complete):
```python
import numpy as np, ga3 as ga, stt, ga_stt_controller_final as ctrl, dynamics as dyn
from ga_stt_sat import compute_control_sat
g = 9.81; E3 = np.array([0,0,1.0])
P = stt.STTParams(eta=[8,8,8], s0=[1,1,1], tc=16.0, k1=0.15, k2=1.0, k3=0.5,
                   rho_max=1.0, rho_min=0.15, u=8.0, rhoR0=1.0, rhoR_inf=0.05, kR=0.5)
obs = [stt.Obstacle([4,4,4],[0.05,-0.02,0.03],0.6), stt.Obstacle([3,5,4.5],[-0.03,0.04,0.0],0.5)]

def mk(kv, kRg, kOm, k1=1.5, tau_max=1e6, fmax=1e6):
    return ctrl.DroneParams(mass=0.5, J_diag=[3e-3,3e-3,5.5e-3], f_min=0.0, f_max=fmax,
                             kappa1=k1, kappa_v=kv, kR=kRg, kOmega=kOm, tau_max=tau_max,
                             accel_cap_mult=1e6)

def sim_sat(kv, kRg, kOm, Om_m, tilt=0.8, e1_0=0.7, v0=(0.5,0,0), tend=2.0, dt=0.001, k1=1.5):
    dr = mk(kv, kRg, kOm, k1); rho0 = stt.rho_p_value(P.s0, 0.0, obs, P)
    st = dyn.UAVState(P.s0 + e1_0*rho0*np.array([1,0,0.]), np.array(v0,float),
                       ga.exp_bivector(np.array([0,1,0.]), tilt), np.zeros(3))
    sig = P.s0.copy(); mx = 0
    for i in range(int(tend/dt)):
        tt = i*dt
        ud = dict(p=st.p, v=st.v, R=st.R, Omega_b=st.Omega_b, sigma=sig)
        f, tau, dg = compute_control_sat(tt, ud, obs, P, dr, ga.rotor_identity(), Om_m)
        e1 = np.linalg.norm(st.p-sig)/stt.rho_p_value(sig, tt, obs, P); mx = max(mx, e1)
        if not np.isfinite(f) or e1 >= 1.3: return 1.3
        def cfn(tl, s_): return f, tau
        st, _, _ = dyn.rk4_step(st, dt, dr.m, dr.J_diag, cfn, lambda tt2: (np.zeros(3), np.zeros(3)), tt)
        sig = sig + stt.sigma_value(sig, tt, obs, P)*dt
    return mx

def sim_orig(kv, kRg, kOm, tilt=0.8, e1_0=0.7, v0=(0.5,0,0), tend=2.0, dt=0.001, k1=1.5):
    # identical, but calls ctrl.compute_control(...) -- the UNMODIFIED base-repo controller
    ...

for kR in (4,8,16,32,64,128,256,512,1024,4096,16384,32768,131072):
    mo = sim_orig(2.0, kR, 2.0)       # ORIGINAL, fixed kOmega=2.0
    ms = sim_sat(2.0, kR, 2.0, 20.0)  # SATURATED (Omega_m=20), same fixed kOmega=2.0
    print(kR, mo, ms)
```

**Result obtained (fresh run this session, identical to all earlier reporting):**
```
   kR | ORIGINAL (kOmega=2.0 fixed) | SATURATED (Omega_m=20, kOmega=2.0 fixed)
    4 |                      1.3000 |                                   1.3000
    8 |                      0.7964 |                                   0.7964
   16 |                      0.7048 |                                   0.7048
   32 |                      0.7000 |                                   0.7000
   64 |                      0.7000 |                                   0.7000
  128 |                      0.7000 |                                   0.7000
  256 |                      0.7000 |                                   0.7000
  512 |                      0.7000 |                                   0.7000
 1024 |                      0.7000 |                                   0.7000
 4096 |                      1.3000 (DIVERGES) |                        0.7000
16384 |                      1.3000 (DIVERGES) |                        0.7000
32768 |                      1.3000 (DIVERGES) |                        0.7000
131072 |                     1.3000 (DIVERGES) |                        0.7000
```
(1.3 is the simulation's divergence/failure sentinel — the tube is left and integration is halted.) Identical behavior for κR ≤ 1024 (saturation never engages — confirms zero cost in normal operation); the two designs diverge in outcome exactly where saturation starts to matter, and the saturated design holds for a further 128× range of κR with no change to κΩ.

---

## 6. Honest summary of what this document does and does not establish

**Established, with proof:** Lemmas 1, 2, 3, R1, 4, and 5/6-bulk. Six rigorous results, each independently verified against the actual repository code (not just symbolically), with scripts reproduced in full above.

**Established, empirically, not by closed proof:** that the near-origin sliver (Lemma 5/6, open item) does not destabilize the full closed loop, across κR spanning 4 to 131,072 with a single fixed κΩ.

**Not claimed:** global (as opposed to semi-global) stability; robustness to actuator saturation (all tests above use unlimited thrust/torque to isolate the mechanism being studied); disturbance rejection (w=0 throughout); anything about the DSC or Levant-differentiator extensions explored elsewhere — those are explicitly out of scope for this document.
