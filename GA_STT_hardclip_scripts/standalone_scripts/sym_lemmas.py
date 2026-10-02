import sympy as sp
t=sp.symbols('t',real=True)
k1,kv=sp.symbols('kappa1 kappa_v',positive=True)

# ---------- Lemma A: normalized-energy identity (3-D, generic rho(t), generic disturbance D) ----------
rho=sp.Function('rho',positive=True)(t)
P=sp.Matrix([sp.Function(f'P{i}')(t) for i in range(3)])      # p_tilde
D=sp.Matrix([sp.Function(f'D{i}')(t) for i in range(3)])      # lumped perturbation (thrust misalignment + w/m)
n=sp.sqrt((P.T*P)[0])
e=n/rho
eps=sp.log((1+e)/(1-e)); xi=2/(1-e**2)
g=eps*xi                                                       # U'(e), U=eps^2/2
phat=P/n
zv=P.diff(t)
# closed-loop: zv' = -kv zv - (k1/rho) eps xi phat + D
Pdd=-kv*zv-(k1/rho)*g*phat+D
q=P/rho
qd=q.diff(t)
En=sp.Rational(1,2)*(qd.T*qd)[0]+k1/rho**2*eps**2/2
dEn=En.diff(t)
subs={P[i].diff(t,2):Pdd[i] for i in range(3)}
dEn=dEn.subs(subs)
# claimed closed form
r,rd,rdd=rho,rho.diff(t),rho.diff(t,2)
claim=(-(2*rd/r+kv)*(qd.T*qd)[0]-(rdd/r+kv*rd/r)*(q.T*qd)[0]
       +(qd.T*D)[0]/r-2*(rd/r)*k1*eps**2/2/r**2)
import random
vals={}
def rnd(): return sp.Rational(random.randint(-9,9),7)
sub_num={}
for i in range(3):
    for d_ in range(0,3):
        sub_num[P[i].diff(t,d_) if d_ else P[i]]=sp.Rational(random.randint(-4,4),13)
for d_ in range(0,3):
    sub_num[rho.diff(t,d_) if d_ else rho]=sp.Rational(3,2) if d_==0 else sp.Rational(random.randint(-3,3),11)
for i in range(3): sub_num[D[i]]=sp.Rational(random.randint(-5,5),9)
sub_num[k1]=sp.Rational(23,10); sub_num[kv]=sp.Rational(17,10)
# make sure e<1
diff=(dEn-claim)
dd=diff.subs(sub_num)
print("A: |dE/dt - claimed| at random point =",sp.N(dd,20))
print("   (e =",sp.N(e.subs(sub_num),6),")")

# ---------- Lemma B: e_v is analytic: coefficient c(Psi) ----------
Psi,rR,rRd,kR=sp.symbols('Psi rho_R rhodot_R kappa_R',real=True)
eR=Psi/rR; epsR=-sp.log(1-eR)
s2=Psi*(2-Psi)                                     # sin^2(theta/2)
coef=(2*kR*epsR-2*eR*rRd)/s2                       # e_v = -coef * (n_hat*sin) ~ analytic multiple of <Re>_2
print("B: series of coefficient at Psi->0:",sp.series(sp.simplify(coef),Psi,0,3))
print("   limit:",sp.limit(coef,Psi,0))
print("   eR/sin^2 =",sp.simplify(eR/s2))

# ---------- Lemma C: xi_p correct (2/(1-e^2)) and dV1/dt exact identity (nominal D=0) ----------
E=sp.symbols('E',positive=True)
print("C: d eps/de - 2/(1-e^2) =",sp.simplify(sp.diff(sp.log((1+E)/(1-E)),E)-2/(1-E**2)))
