"""
GA-STT with a SATURATING Stage-R correction: cap |coef| (=|e_v_Omega|) at
Omega_m, independent of kR. Recovers the exact-cancellation law/identity
whenever unsaturated; gives a fixed, non-exponential (but still strictly
negative) decay rate when saturated. No DSC/filter needed -- this targets
the magnitude problem directly, which is what actually caused the wall.
"""
import sys; sys.path.insert(0,'/home/claude/ga_v5/ga_stt_py/lib')
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
        coef = np.clip(coef_exact, -Omega_m, Omega_m)     # <-- the only change
    else:
        coef = 0.0
    e_v_Omega = -coef * n_hat_e
    Omega_c = e_v_Omega + ga.sandwich_bivector(ga.reverse(Re), Omega_d_value)

    z_Omega = Omega_b - Omega_c
    J = drone_params.J_diag
    commutator_term = np.cross(Omega_b, J * Omega_b)
    # feedforward: still only J*Omega_dot_d (same "documented simplification" as
    # the baseline -- we are NOT relying on differentiating the saturated e_v at
    # all, consistent with it being piecewise-constant across saturation)
    tau = (-drone_params.kOmega * z_Omega + commutator_term + J * Omega_dot_d_value)
    tau_max = getattr(drone_params, "tau_max", 1.5)
    n_ = np.linalg.norm(tau)
    if n_ > tau_max:
        tau = tau * (tau_max / n_)
    diag = dict(Rd=Rd_value, Omega_d=Omega_d_value, Omega_dot_d=Omega_dot_d_value,
                theta_e=theta_e, Psi_e=Psi_e, e_R=e_R, eps_R=eps_R, rho_R=rho_R_t,
                z_Omega=z_Omega, that_d=jt.value(that_d_jet), fd=jt.value(fd_jet),
                sigma=sigma_now, rho_p=jt.value(rjet), Omega_v_b=Omega_c, coef=coef,
                saturated=(abs(coef_exact) > Omega_m) if theta_e>1e-6 else False)
    return f_command, tau, diag
