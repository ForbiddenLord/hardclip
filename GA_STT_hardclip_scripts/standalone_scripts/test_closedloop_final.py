import sys; sys.path.insert(0,'/home/claude/ga_v5/ga_stt_py/lib')
import numpy as np, ga3 as ga, stt, ga_stt_controller_final as ctrl, dynamics as dyn
from ga_stt_sat import compute_control_sat
g=9.81; E3=np.array([0,0,1.0])
P=stt.STTParams(eta=[8,8,8],s0=[1,1,1],tc=16.0,k1=0.15,k2=1.0,k3=0.5,rho_max=1.0,rho_min=0.15,u=8.0,rhoR0=1.0,rhoR_inf=0.05,kR=0.5)
obs=[stt.Obstacle([4,4,4],[0.05,-0.02,0.03],0.6),stt.Obstacle([3,5,4.5],[-0.03,0.04,0.0],0.5)]
def mk(kv,kRg,kOm,k1=1.5,tau_max=1e6,fmax=1e6):
    return ctrl.DroneParams(mass=0.5,J_diag=[3e-3,3e-3,5.5e-3],f_min=0.0,f_max=fmax,kappa1=k1,kappa_v=kv,kR=kRg,kOmega=kOm,tau_max=tau_max,accel_cap_mult=1e6)

def sim_sat(kv,kRg,kOm,Om_m,tilt=0.8,e1_0=0.7,v0=(0.5,0,0),tend=2.0,dt=0.001,k1=1.5):
    dr=mk(kv,kRg,kOm,k1); rho0=stt.rho_p_value(P.s0,0.0,obs,P)
    st=dyn.UAVState(P.s0+e1_0*rho0*np.array([1,0,0.]),np.array(v0,float),ga.exp_bivector(np.array([0,1,0.]),tilt),np.zeros(3)); sig=P.s0.copy()
    mx=0
    for i in range(int(tend/dt)):
        tt=i*dt
        ud=dict(p=st.p,v=st.v,R=st.R,Omega_b=st.Omega_b,sigma=sig)
        f,tau,dg=compute_control_sat(tt,ud,obs,P,dr,ga.rotor_identity(),Om_m)
        e1=np.linalg.norm(st.p-sig)/stt.rho_p_value(sig,tt,obs,P); mx=max(mx,e1)
        if not np.isfinite(f) or e1>=1.3: return 1.3
        def cfn(tl,s_): return f,tau
        st,_,_=dyn.rk4_step(st,dt,dr.m,dr.J_diag,cfn,lambda tt2:(np.zeros(3),np.zeros(3)),tt)
        sig=sig+stt.sigma_value(sig,tt,obs,P)*dt
    return mx

def sim_orig(kv,kRg,kOm,tilt=0.8,e1_0=0.7,v0=(0.5,0,0),tend=2.0,dt=0.001,k1=1.5):
    dr=mk(kv,kRg,kOm,k1); rho0=stt.rho_p_value(P.s0,0.0,obs,P)
    st=dyn.UAVState(P.s0+e1_0*rho0*np.array([1,0,0.]),np.array(v0,float),ga.exp_bivector(np.array([0,1,0.]),tilt),np.zeros(3)); sig=P.s0.copy(); mx=0
    for i in range(int(tend/dt)):
        tt=i*dt
        ud=dict(p=st.p,v=st.v,R=st.R,Omega_b=st.Omega_b,sigma=sig)
        f,tau,dg=ctrl.compute_control(tt,ud,obs,P,dr,ga.rotor_identity())
        e1=np.linalg.norm(st.p-sig)/stt.rho_p_value(sig,tt,obs,P); mx=max(mx,e1)
        if not np.isfinite(f) or e1>=1.3: return 1.3
        def cfn(tl,s_): return f,tau
        st,_,_=dyn.rk4_step(st,dt,dr.m,dr.J_diag,cfn,lambda tt2:(np.zeros(3),np.zeros(3)),tt)
        sig=sig+stt.sigma_value(sig,tt,obs,P)*dt
    return mx

if __name__=="__main__":
    print(f"{'kR':>7} | {'ORIGINAL (kOmega=2.0 fixed)':>28} | {'SATURATED (Omega_m=20, kOmega=2.0 fixed)':>42}")
    for kR in (4,8,16,32,64,128,256,512,1024,4096,16384,32768,131072):
        mo=sim_orig(2.0,kR,2.0)
        ms=sim_sat(2.0,kR,2.0,20.0)
        print(f"{kR:7d} | {mo:28.4f} | {ms:42.4f}")
