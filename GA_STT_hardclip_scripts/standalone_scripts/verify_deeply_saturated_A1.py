# Confirm A1 (z-driven part of Delta_Omega) is ALSO tamed by saturation, not just A2 (s-driven)
import sys; sys.path.insert(0,'/home/claude/ga_v5/ga_stt_py/lib')
import numpy as np, ga3 as ga, stt, ga_stt_controller_final as base
E3=np.array([0,0,1.0]); g=9.81
P=stt.STTParams(eta=[8,8,8],s0=[1,1,1],tc=16.0,k1=0.15,k2=1.0,k3=0.5,rho_max=1.0,rho_min=0.15,u=8.0,rhoR0=1.0,rhoR_inf=0.05,kR=0.5)
obs=[stt.Obstacle([4,4,4],[0.05,-0.02,0.03],0.6),stt.Obstacle([3,5,4.5],[-0.03,0.04,0.0],0.5)]
import jet1d as jt, rotor_jet as rj

def Omega_c_sat(t0,p,v,R,sigma_now,dr,Omega_m):
    sjet=stt.sigma_jet(sigma_now,t0,obs,P,base.JET_ORDER_SIGMA); rjet=stt.rho_p_jet(sjet,t0,obs,P,base.JET_ORDER_SIGMA)
    Fd_jet,fd_jet,that_d_jet=base.resolve_translational_chain(t0,p,v,sjet,rjet,dr)
    f=np.clip(jt.value(fd_jet),dr.f_min,dr.f_max)
    tp_hat=ga.sandwich(ga.rotor_identity(),E3); order_rot=jt.order_of(that_d_jet)
    tp_jet=jt.const_vector(tp_hat,order_rot); Ra_jet=rj.align(tp_jet,that_d_jet)
    Rp_jet=rj.const_from_value(ga.rotor_identity(),order_rot); Rd_jet=rj.mul(Ra_jet,Rp_jet)
    Rd_value=ga.normalize_rotor(rj.value_multivector(Rd_jet))
    Rd_rev_jet=rj.reverse(Rd_jet); Rd_dot_jet=rj.differentiate(Rd_jet)
    Omega_d_rotorjet=rj.mul(Rd_rev_jet,Rd_dot_jet)
    Omega_d_b=jt.scale(Omega_d_rotorjet.b,-2.0); Omega_d_value=jt.value(Omega_d_b)
    Re=ga.normalize_rotor(ga.geometric_product(ga.reverse(Rd_value),R))
    theta_e,n_hat_e=ga.rotor_log_angle_axis(Re); Psi_e=1.0-ga.rotor_scalar(Re)
    rho_R_t=stt.rho_R(t0,P); rho_R_dot_t=stt.rho_R_dot(t0,P)
    e_R=np.clip(Psi_e/rho_R_t,0.0,1.0-1e-9); eps_R=-np.log(max(1.0-e_R,1e-12))
    sin_half=np.sin(0.5*theta_e)
    if theta_e>1e-6:
        coefexact=(2.0*dr.kR*eps_R-2.0*e_R*rho_R_dot_t)/sin_half
        coef=np.clip(coefexact,-Omega_m,Omega_m)
    else: coef=0.0
    e_v=-coef*n_hat_e
    Omega_c=e_v+ga.sandwich_bivector(ga.reverse(Re),Omega_d_value)
    return f, Omega_c, Rd_value

def Delta_z_only(kRg,Omega_m,zval,kv=2.0,k1=1.5,seed=2):
    dr=base.DroneParams(mass=0.5,J_diag=[3e-3,3e-3,5.5e-3],f_min=0.0,f_max=1e6,kappa1=k1,kappa_v=kv,kR=kRg,kOmega=1.0,tau_max=1e9,accel_cap_mult=1e6)
    rng=np.random.default_rng(seed); ax=rng.normal(size=3); ax/=np.linalg.norm(ax)
    R=ga.exp_bivector(ax,0.9)   # comfortably in the saturated bulk (theta_e~0.75-ish per earlier)
    p=P.s0+rng.normal(size=3)*0.1; v=rng.normal(size=3)*0.15; sig=P.s0.copy(); t=1.0
    f0,Oc0,Rd0=Omega_c_sat(t,p,v,R,sig,dr,Omega_m)
    Omega_b_true = Oc0 + zval*np.array([0.3,0.5,0.7])/np.linalg.norm([0.3,0.5,0.7])   # actual Omega_b = Omega_c + z
    h=1e-6
    Rn=ga.normalize_rotor(ga.geometric_product(R,ga.rotor(1.0,(-0.5)*h*Omega_b_true)))
    pn=p+v*h; vn=v+(f0/dr.m*ga.sandwich(R,E3)-g*E3)*h
    f1,Oc1,Rd1=Omega_c_sat(t+h,pn,vn,Rn,sig,dr,Omega_m)
    Ocdot=(Oc1-Oc0)/h
    return np.linalg.norm(Ocdot)   # ~ the z-driven part of d/dt(Omega_c), since only Omega_b differs from Oc0

print("z-driven part of Omega_c_dot (i.e. contributes to A1|z_Omega|) vs kR, |z|=1 fixed, deeply saturated state:")
for kR in (2,8,32,128,512,2048,8192,32768):
    D=Delta_z_only(kR,20.0,1.0)
    print(f"  kR={kR:6d}: |dOmega_c/dt|_z-part = {D:10.4f}")
