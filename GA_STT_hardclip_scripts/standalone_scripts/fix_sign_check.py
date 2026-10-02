"""
Re-derive eps_R_dot,nom directly from the ACTUAL controller code via finite
differences (not hand algebra), for BOTH the unsaturated and saturated
branches, to nail the correct sign before trusting any hand-derived formula.
"""
import sys; sys.path.insert(0,'/home/claude/ga_v5/ga_stt_py/lib')
import numpy as np, ga3 as ga, stt
E3=np.array([0,0,1.0]); g=9.81

P=stt.STTParams(eta=[8,8,8],s0=[1,1,1],tc=16.0,k1=0.15,k2=1.0,k3=0.5,rho_max=1.0,rho_min=0.15,u=8.0,rhoR0=1.0,rhoR_inf=0.05,kR=0.5)

def eps_R_nom_dot_fd(theta0, kR_gain, Omega_m, t=1.0, h=1e-6):
    """zOmega=0 nominal case: Omega_b is set EXACTLY to Omega_c (so eOmega=e_v only),
    advance R via the true rotor kinematics with that Omega_b, refetch eps_R."""
    ax=np.array([0.0,1.0,0.0])
    R=ga.exp_bivector(ax,theta0)
    # We don't need a real Rd/Stage-1 chain here -- just test the abstract Stage-R law
    # directly against a FIXED Rd=identity, so Re=R exactly, isolating Stage R.
    Rd=ga.rotor_identity()
    def eps_and_coef(Rcur):
        Re=ga.normalize_rotor(ga.geometric_product(ga.reverse(Rd),Rcur))
        theta_e,n_hat_e=ga.rotor_log_angle_axis(Re)
        Psi_e=1.0-ga.rotor_scalar(Re)
        rho_R_t=stt.rho_R(t,P); rho_R_dot_t=stt.rho_R_dot(t,P)
        e_R=np.clip(Psi_e/rho_R_t,0.0,1.0-1e-9); eps_R=-np.log(max(1.0-e_R,1e-12))
        sin_half=np.sin(0.5*theta_e)
        if theta_e>1e-6:
            coef_exact=(2.0*kR_gain*eps_R-2.0*e_R*rho_R_dot_t)/sin_half
            coef=np.clip(coef_exact,-Omega_m,Omega_m)
        else:
            coef_exact=0.0; coef=0.0
        e_v=-coef*n_hat_e
        Omega_c=e_v  # rotated(Omega_d)=0 since Omega_d=0 (Rd fixed=identity)
        return eps_R, Omega_c, coef_exact, coef, e_R, rho_R_t, rho_R_dot_t
    eps0,Oc0,ce0,c0,eR0,rhoR0_,rhoRd0=eps_and_coef(R)
    # advance with Omega_b = Omega_c exactly (zOmega=0 nominal)
    Rn=ga.normalize_rotor(ga.geometric_product(R, ga.rotor(1.0,(-0.5)*h*Oc0)))
    eps1,Oc1,ce1,c1,eR1,rhoR1_,rhoRd1=eps_and_coef(Rn)
    eps_dot_fd=(eps1-eps0)/h
    return eps_dot_fd, eps0, ce0, c0, eR0

print("Ground truth (finite-difference on actual code), UNSATURATED (Omega_m huge):")
for th in (0.3,0.8,1.5,2.5):
    for kRg in (2.0,8.0):
        d,eps,ce,c,eR=eps_R_nom_dot_fd(th,kRg,Omega_m=1e9)
        pred=-kRg*eps/(stt.rho_R(1.0,P)*(1-eR))
        print(f"  theta0={th:.2f} kR={kRg:5.1f}: eps_dot(FD)={d:10.5f}  predicted(-kR*eps/(rho(1-eR)))={pred:10.5f}  eps_R={eps:.4f}")
