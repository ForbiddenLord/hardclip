"""
Redo the Delta_Omega measurement (Omega_c_dot_true vs Omega_dot_d fed forward)
for the SATURATED law, separately in the saturated and unsaturated regimes.
"""
import sys; sys.path.insert(0,'/home/claude/ga_v5/ga_stt_py/lib')
import numpy as np, ga3 as ga, stt, ga_stt_controller_final as base
E3=np.array([0,0,1.0]); g=9.81
P=stt.STTParams(eta=[8,8,8],s0=[1,1,1],tc=16.0,k1=0.15,k2=1.0,k3=0.5,rho_max=1.0,rho_min=0.15,u=8.0,rhoR0=1.0,rhoR_inf=0.05,kR=0.5)
obs=[stt.Obstacle([4,4,4],[0.05,-0.02,0.03],0.6),stt.Obstacle([3,5,4.5],[-0.03,0.04,0.0],0.5)]

def compute_Omega_c_sat(t0,p,v,R,Omega_b,sigma_now,drone_params,Omega_m):
    import jet1d as jt, rotor_jet as rj
    sjet = stt.sigma_jet(sigma_now, t0, obs, P, base.JET_ORDER_SIGMA)
    rjet = stt.rho_p_jet(sjet, t0, obs, P, base.JET_ORDER_SIGMA)
    Fd_jet, fd_jet, that_d_jet = base.resolve_translational_chain(t0, p, v, sjet, rjet, drone_params)
    f=np.clip(jt.value(fd_jet),drone_params.f_min,drone_params.f_max)
    tp_hat=ga.sandwich(ga.rotor_identity(),E3)
    order_rot=jt.order_of(that_d_jet)
    tp_jet=jt.const_vector(tp_hat,order_rot)
    Ra_jet=rj.align(tp_jet,that_d_jet); Rp_jet=rj.const_from_value(ga.rotor_identity(),order_rot)
    Rd_jet=rj.mul(Ra_jet,Rp_jet); Rd_value=ga.normalize_rotor(rj.value_multivector(Rd_jet))
    Rd_rev_jet=rj.reverse(Rd_jet); Rd_dot_jet=rj.differentiate(Rd_jet)
    Omega_d_rotorjet=rj.mul(Rd_rev_jet,Rd_dot_jet)
    Omega_d_b=jt.scale(Omega_d_rotorjet.b,-2.0); Omega_d_value=jt.value(Omega_d_b)
    Re=ga.normalize_rotor(ga.geometric_product(ga.reverse(Rd_value),R))
    theta_e,n_hat_e=ga.rotor_log_angle_axis(Re); Psi_e=1.0-ga.rotor_scalar(Re)
    rho_R_t=stt.rho_R(t0,P); rho_R_dot_t=stt.rho_R_dot(t0,P)
    e_R=np.clip(Psi_e/rho_R_t,0.0,1.0-1e-9); eps_R=-np.log(max(1.0-e_R,1e-12))
    sin_half=np.sin(0.5*theta_e)
    if theta_e>1e-6:
        coef_exact=(2.0*drone_params.kR*eps_R-2.0*e_R*rho_R_dot_t)/sin_half
        coef=np.clip(coef_exact,-Omega_m,Omega_m); sat=abs(coef_exact)>Omega_m
    else:
        coef=0.0; sat=False
    e_v=-coef*n_hat_e
    Omega_c=e_v+ga.sandwich_bivector(ga.reverse(Re),Omega_d_value)
    return f,Omega_c,Omega_d_value,eps_R,sat,Rd_value,theta_e

def Delta_Omega_sat(kRg,Omega_m,regime,kv=2.0,k1=1.5,seed=1):
    dr=base.DroneParams(mass=0.5,J_diag=[3e-3,3e-3,5.5e-3],f_min=0.0,f_max=1e6,kappa1=k1,kappa_v=kv,kR=kRg,kOmega=1.0,tau_max=1e9,accel_cap_mult=1e6)
    rng=np.random.default_rng(seed); ax=rng.normal(size=3); ax/=np.linalg.norm(ax)
    tilt = 0.05 if regime=='unsat' else 0.9   # small tilt -> small eps_R (near threshold); large tilt -> saturated
    R=ga.exp_bivector(ax,tilt)
    p=P.s0+rng.normal(size=3)*0.1; v=rng.normal(size=3)*0.15; sig=P.s0.copy(); t=1.0
    f0,Oc0,Od0,eps0,sat0,Rd0,th0=compute_Omega_c_sat(t,p,v,R,None,sig,dr,Omega_m)
    h=1e-5
    Rn=ga.normalize_rotor(ga.geometric_product(R,ga.rotor(1.0,(-0.5)*h*np.zeros(3))))  # Omega_b irrelevant to Omega_c/Omega_d (state feedback only via p,v,R)
    pn=p+v*h; vn=v+(f0/dr.m*ga.sandwich(R,E3)-g*E3)*h
    f1,Oc1,Od1,eps1,sat1,Rd1,th1=compute_Omega_c_sat(t+h,pn,vn,Rn,None,sig,dr,Omega_m)
    Ocdot=(Oc1-Oc0)/h
    # feedforward actually fed (Omega_dot_d only, via same finite diff on Omega_d)
    Oddot=(Od1-Od0)/h
    Delta=Ocdot-Oddot
    return np.linalg.norm(Delta), eps0, sat0, th0

print("Regime: near eps_R~0, just below/above the saturation threshold (small tilt) -- unsaturated branch scaling")
for kR in (2,8,32,128,512,2048,8192):
    D,eps,sat,th=Delta_Omega_sat(kR,Omega_m=20.0,regime='unsat')
    print(f"  kR={kR:6d}: |Delta|={D:10.4f}  eps_R={eps:.5f} saturated={sat}  theta_e={th:.4f}")
print()
print("Regime: comfortably saturated (large tilt, big eps_R) -- saturated branch")
for kR in (2,8,32,128,512,2048,8192):
    D,eps,sat,th=Delta_Omega_sat(kR,Omega_m=20.0,regime='sat')
    print(f"  kR={kR:6d}: |Delta|={D:10.4f}  eps_R={eps:.5f} saturated={sat}  theta_e={th:.4f}")
