"""
Final verification of the corrected Lemma R1: eps_R_dot,nom < 0 EVERYWHERE
in the saturated branch (both signs of coef_exact), for a wide range of kR,
with NO shrinking-boundary-layer exception -- checking directly on the
actual controller code via finite differences.
"""
import sys; sys.path.insert(0,'/home/claude/ga_v5/ga_stt_py/lib')
import numpy as np, ga3 as ga, stt
P=stt.STTParams(eta=[8,8,8],s0=[1,1,1],tc=16.0,k1=0.15,k2=1.0,k3=0.5,rho_max=1.0,rho_min=0.15,u=8.0,rhoR0=1.0,rhoR_inf=0.05,kR=0.5)
Rd=ga.rotor_identity()
Omega_m=3.5   # just above the derived floor of 3.0424, deliberately tight (not the generous 20 used before)

def eps_dot_fd(theta0, kRg, t=1.0, h=1e-7):
    R=ga.exp_bivector(np.array([0,1,0.]),theta0)
    def state(Rc,tt):
        Re=ga.normalize_rotor(ga.geometric_product(ga.reverse(Rd),Rc))
        th,n=ga.rotor_log_angle_axis(Re); Psi=1.0-ga.rotor_scalar(Re)
        rhoR=stt.rho_R(tt,P); eR=np.clip(Psi/rhoR,0,1-1e-9); eps=-np.log(max(1-eR,1e-12))
        return eR,eps,th,n,rhoR
    eR0,eps0,th0_,n0,rhoR0_=state(R,t)
    rhoRd0=stt.rho_R_dot(t,P); s0=np.sin(0.5*th0_)
    if th0_>1e-6:
        ce=(2*kRg*eps0-2*eR0*rhoRd0)/s0
        coef=np.clip(ce,-Omega_m,Omega_m); sat=abs(ce)>Omega_m
    else:
        coef=0.0; sat=False
    e_v=-coef*n0
    Rn=ga.normalize_rotor(ga.geometric_product(R,ga.rotor(1.0,(-0.5)*h*e_v)))
    eR1,eps1,th1,n1,rhoR1_=state(Rn,t+h)
    return (eps1-eps0)/h, eR0, sat, s0

print(f"Omega_m={Omega_m} (just above the derived floor 3.04). Scanning eps_R_dot,nom over eR in (0,1), across kR, SATURATED points only:")
worst=0; worst_info=None
for kRg in [10,100,1000,1e4,1e5,1e6,1e7]:
    n_sat=0; n_bad=0
    for theta0 in np.linspace(0.01, 3.10, 400):
        d,eR0,sat,s0 = eps_dot_fd(theta0, kRg)
        if sat:
            n_sat+=1
            if d>=0:
                n_bad+=1
                if d>worst: worst=d; worst_info=(kRg,theta0,eR0,s0,d)
    print(f"  kR={kRg:10.0f}: saturated points tested={n_sat:4d}, with eps_dot>=0 (BAD)={n_bad}")
print("worst violation found:", worst_info)
