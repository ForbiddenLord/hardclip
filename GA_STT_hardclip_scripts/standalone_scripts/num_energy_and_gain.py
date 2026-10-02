import sys; sys.path.insert(0,'../ga_v5/ga_stt_py/lib')
import numpy as np, ga3 as ga, stt, ga_stt_controller_final as ctrl, dynamics as dyn
g=9.81; E3=np.array([0,0,1.0])
def make(kv,kRg,kOm,k1=1.5):
    P=stt.STTParams(eta=[8,8,8],s0=[1,1,1],tc=16.0,k1=0.15,k2=1.0,k3=0.5,rho_max=1.0,rho_min=0.15,u=8.0,
                    rhoR0=0.28,rhoR_inf=0.05,kR=0.5)
    obs=[stt.Obstacle([4,4,4],[0.05,-0.02,0.03],0.6),stt.Obstacle([3,5,4.5],[-0.03,0.04,0.0],0.5)]
    dr=ctrl.DroneParams(mass=0.5,J_diag=[3e-3,3e-3,5.5e-3],f_min=0.0,f_max=12.0,kappa1=k1,kappa_v=kv,kR=kRg,kOmega=kOm,tau_max=1e3)
    return P,obs,dr
def run(kv,kRg,kOm,tend,dt=0.001,tilt=0.7,seed=0,k1=1.5,p_off=np.zeros(3)):
    P,obs,dr=make(kv,kRg,kOm,k1)
    rng=np.random.default_rng(seed); ax=rng.normal(size=3); ax/=np.linalg.norm(ax)
    st=dyn.UAVState(P.s0+p_off,np.zeros(3),ga.exp_bivector(ax,tilt),np.zeros(3)); sig=P.s0.copy()
    R=[]
    for i in range(int(tend/dt)+1):
        t=i*dt
        ud=dict(p=st.p,v=st.v,R=st.R,Omega_b=st.Omega_b,sigma=sig)
        f,tau,dg=ctrl.compute_control(t,ud,obs,P,dr,ga.rotor_identity())
        sj=stt.sigma_jet(sig,t,obs,P,ctrl.JET_ORDER_SIGMA); rj=stt.rho_p_jet(sj,t,obs,P,ctrl.JET_ORDER_SIGMA)
        rho,rhod,rhodd=rj[0],rj[1],2*rj[2]
        pt=st.p-sig; e1=np.linalg.norm(pt)/rho
        zv=st.v-sj[1]; q=pt/rho; qd=zv/rho-q*rhod/rho
        eps=np.log((1+e1)/(1-e1)) if e1<1 else np.inf
        En=0.5*qd@qd+k1/rho**2*0.5*eps**2
        tvec=ga.sandwich(st.R,E3); vdot=-g*E3+(f/dr.m)*tvec
        fd=dg['fd']; u1=fd*dg['that_d']/dr.m-g*E3; delta=vdot-u1
        Endot_formula=(-(2*rhod/rho+dr.kappa_v)*(qd@qd)-(rhodd/rho+dr.kappa_v*rhod/rho)*(q@qd)
                       +qd@delta/rho-2*rhod/rho*k1*0.5*eps**2/rho**2)
        Re=dg['Re']; th,n=ga.rotor_log_angle_axis(Re)
        # coupling ratio for rotational loop
        s=np.sin(th/2); rhoR=dg['rho_R']; eR=dg['e_R']; epsR=dg['eps_R']
        td=dg['that_d']; alpha=np.arccos(np.clip(td@E3,-1,1)); M=1/np.cos(alpha/2)
        R.append([t,En,Endot_formula,e1,th,epsR,np.linalg.norm(delta),M,rhoR,dg['fd'],f,np.linalg.norm(tau),np.linalg.norm(dg['z_Omega'])])
        def cfn(tl,s_): return f,tau
        st,_,_=dyn.rk4_step(st,dt,dr.m,dr.J_diag,cfn,lambda tt:(np.zeros(3),np.zeros(3)),t)
        sig=sig+stt.sigma_value(sig,t,obs,P)*dt
        if not np.isfinite(st.p).all() or np.linalg.norm(st.p-sig)>1e3: break
    return np.array(R)
if __name__=="__main__":
    A=run(2.0,8.0,0.2,tend=6.0)
    Endot_fd=np.gradient(A[:,1],A[:,0])
    k=slice(20,len(A)-20)
    print("E_n identity along coupled sim: max|Edot_formula - Edot_FD|/max|Edot_FD| =",
          np.max(np.abs(A[k,2]-Endot_fd[k]))/np.max(np.abs(Endot_fd[k])))
    print("  E_n(0)=%.4f max E_n=%.4f, max e1=%.3f"%(A[0,1],A[:,1].max(),A[:,3].max()))
