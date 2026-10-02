import sys; sys.path.insert(0,'../ga_v5/ga_stt_py/lib')
import numpy as np, ga3 as ga, stt, ga_stt_controller_final as ctrl
from scipy.integrate import solve_ivp
g=9.81; E3=np.array([0,0,1.0])
P=stt.STTParams(eta=[8,8,8],s0=[1,1,1],tc=16.0,k1=0.15,k2=1.0,k3=0.5,rho_max=1.0,rho_min=0.15,u=8.0,
                rhoR0=0.28,rhoR_inf=0.05,kR=0.5)
obs=[stt.Obstacle([4,4,4],[0.05,-0.02,0.03],0.6),stt.Obstacle([3,5,4.5],[-0.03,0.04,0.0],0.5)]
dr=ctrl.DroneParams(mass=0.5,J_diag=[3e-3,3e-3,5.5e-3],f_min=0.0,f_max=12.0,kappa1=1.5,kappa_v=2.0,kR=8.0,kOmega=0.2)
Rp=ga.rotor_identity()
def ref(t,sig,p,v):
    sj=stt.sigma_jet(sig,t,obs,P,ctrl.JET_ORDER_SIGMA); rj=stt.rho_p_jet(sj,t,obs,P,ctrl.JET_ORDER_SIGMA)
    Fd,fd,thd=ctrl.resolve_translational_chain(t,p,v,sj,rj,dr)
    st=dict(p=p,v=v,R=ga.rotor_identity(),Omega_b=np.zeros(3),sigma=sig)
    f,tau,dg=ctrl.compute_control(t,st,obs,P,dr,Rp)
    return dict(Fd=Fd[0],u1=Fd[0]/dr.m-g*E3,Rd=dg['Rd'],Om=dg['Omega_d'],Omdot=dg['Omega_dot_d'],td=thd[0],
                rho=stt.rho_p_value(sig,t,obs,P))
def traj_Rd(t0,sig0,p0,v0,delta,tau):
    def rhs(t,y):
        sig,p,v=y[:3],y[3:6],y[6:9]
        r=ref(t,sig,p,v)
        return np.concatenate([stt.sigma_value(sig,t,obs,P),v,r['u1']+delta])
    s=solve_ivp(rhs,[t0,t0+tau],np.concatenate([sig0,p0,v0]),method='DOP853',rtol=1e-12,atol=1e-13)
    y=s.y[:,-1]; return ref(t0+tau,y[:3],y[3:6],y[6:9])
t0=2.0
# sigma(t0)
s=solve_ivp(lambda t,y:stt.sigma_value(y,t,obs,P),[0,t0],P.s0,rtol=1e-12,atol=1e-13); sig0=s.y[:,-1]
rho0=stt.rho_p_value(sig0,t0,obs,P)
rng=np.random.default_rng(5)
def Om_fd(delta,p0,v0,h=2e-4):
    Rp_=traj_Rd(t0,sig0,p0,v0,delta,+h)['Rd']; Rm_=traj_Rd(t0,sig0,p0,v0,delta,-h)['Rd']
    R0=ref(t0,sig0,p0,v0)['Rd']
    Rdot=(Rp_-Rm_)/(2*h)
    return -2*ga.to_bivector3(ga.geometric_product(ga.reverse(R0),Rdot))
for k in range(4):
    p0=sig0+rng.normal(size=3)*0.25*rho0; v0=rng.normal(size=3)*0.3
    r0=ref(t0,sig0,p0,v0)
    Om0=Om_fd(np.zeros(3),p0,v0)
    print(f"[{k}] delta=0:  |Omega_d^FD - Omega_d^jet| = {np.linalg.norm(Om0-r0['Om']):.2e}   (|Omega_d|={np.linalg.norm(r0['Om']):.3f})")
    delta=rng.normal(size=3)*0.5
    Om1=Om_fd(delta,p0,v0)
    dOm=Om1-r0['Om']
    # prediction: Fd_dot mismatch = m * d u1/dv * delta = -m kv delta ;  td_dot mismatch = Pperp(.)/|Fd|
    Fd=r0['Fd']; fd=np.linalg.norm(Fd); td=r0['td']
    dF=-dr.m*dr.kappa_v*delta
    dtd=(dF-(dF@td)*td)/fd
    h=1e-6
    Rp_=ga.align_rotor(E3,(td+h*dtd)/np.linalg.norm(td+h*dtd)); Rm_=ga.align_rotor(E3,(td-h*dtd)/np.linalg.norm(td-h*dtd))
    Ra=ga.align_rotor(E3,td)
    pred=-2*ga.to_bivector3(ga.geometric_product(ga.reverse(Ra),(Rp_-Rm_)/(2*h)))
    alpha=np.arccos(np.clip(td@E3,-1,1)); M=1/np.cos(alpha/2)
    bound=M*dr.kappa_v*dr.m*np.linalg.norm(delta-(delta@td)*td)/fd
    print(f"     delta!=0: |dOmega_d| = {np.linalg.norm(dOm):.4e}, predicted = {np.linalg.norm(pred):.4e}, |dOm-pred| = {np.linalg.norm(dOm-pred):.1e}, bound M kv m|d_perp|/|Fd| = {bound:.4e}")
