import sys; sys.path.insert(0,'../ga_v5/ga_stt_py/lib')
import numpy as np, ga3 as ga
rng=np.random.default_rng(3)
def unit(x): return x/np.linalg.norm(x)
worst=0; worst_rel=0
for _ in range(4000):
    tp=unit(rng.normal(size=3)); td=unit(rng.normal(size=3))
    alpha=np.arccos(np.clip(tp@td,-1,1))
    if alpha>np.pi*0.97: continue
    w=rng.normal(size=3); h=1e-6
    td_p=unit(td+h*w); td_m=unit(td-h*w)
    Rp_=ga.align_rotor(tp,td_p); Rm_=ga.align_rotor(tp,td_m); R0=ga.align_rotor(tp,td)
    Rdot=(Rp_-Rm_)/(2*h)
    Om=-2*ga.to_bivector3(ga.geometric_product(ga.reverse(R0),Rdot))   # body bivector of Ra
    tdot=(td_p-td_m)/(2*h)
    ratio=np.linalg.norm(Om)/np.linalg.norm(tdot)
    bound=1/np.cos(alpha/2)
    worst_rel=max(worst_rel,ratio/bound)
print("max over samples of |Omega_a|/(|tdot_d| / cos(alpha/2)) =",worst_rel,"  (<=1 => M<=1/cos(alpha/2))")
# tightness: is it attained?  direction w in plane containing tp,td
alpha=1.2; tp=np.array([0,0,1.0]); td=np.array([np.sin(alpha),0,np.cos(alpha)])
best=0
for ang in np.linspace(0,np.pi,181):
    w=np.array([np.cos(ang),np.sin(ang),0]); w=w-(w@td)*td
    if np.linalg.norm(w)<1e-9: continue
    h=1e-6; td_p=unit(td+h*w); td_m=unit(td-h*w)
    Rdot=(ga.align_rotor(tp,td_p)-ga.align_rotor(tp,td_m))/(2*h)
    Om=-2*ga.to_bivector3(ga.geometric_product(ga.reverse(ga.align_rotor(tp,td)),Rdot))
    best=max(best,np.linalg.norm(Om)/np.linalg.norm((td_p-td_m)/(2*h)))
print("alpha=1.2: sup ratio =",best," 1/cos(alpha/2) =",1/np.cos(alpha/2))
