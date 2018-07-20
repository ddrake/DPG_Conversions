############################################################
# Solve Helmholtz equation with impedance boundary condition
#
#      - Delta u - k*k u = f    on Omega
#       n.grad u - i*k u = g    on bdry
#
# using the DPG variational formulation 
#
#   Y(e;v)    +  b(u,q; v)    = F(v)
#   b(w,r; e) +  c(w,r ; u,q) = conj(G(w,r))
#
# Here conj denotes complex conjugate,
#
#  Y(e,v)       = (grad e, grad v) + k*k (e,v)
#  b(u,q; v)    = (grad u, grad v) - k*k*(u,v) - <<q.n, v>> 
#  c(u,q; w,r)  = - <q.n - ik u, r.n - ik w> 
#  F(v)         = (f,v)
#  G(w,r,w)     = - <g, r.n - ik w> 
#
# where
#
# (.,.)   is sum over all (complex) element L2 inner products,
# <<.,.>> is sum over all (complex) element boundary L2 inner products,
# <.,.>   is global bdry (complex) L2 inner product.
#
# The new terms are contained in the form c, which is a 
# Hermitian positive addition to the whole system, which 
# imposes the impedance condition q.n - ik u = g.
############################################################

from ngsolve import *
from netgen.geom2d import SplineGeometry
from math import pi
from numpy import log
from ctypes import CDLL

libDPG = CDLL("../libDPG.so")

ngsglobals.msg_level = 1

geo = SplineGeometry("../pde/square4bdry.in2d")
mesh = Mesh("../pde/square4bdry4.vol.gz")

# Just set this if we need to?
SetHeapSize(int(1e7))

one = CoefficientFunction(1)
minus = CoefficientFunction(-1.0)

# approx number of waves in the (unit-sized) domain 
nwav = 2

# propagation angle 
theta = pi/16.0

# wavenumber
k  = 2.0*pi*nwav


kc = k*cos(theta)
ks = k*sin(theta)
ksqr = k*k 
minusksqr = -k*k 

# exact solution ex = exp(I*k.x)  for error computation
ex = CoefficientFunction(exp(1j*(kc*x + ks*y)))     


# Note that bc markers for the given square geometry are:
#              3
#        +-----------+
#        |           |
#        |           |
#      4 |           | 2
#        |           |
#        +-----------+
#              1
#  bc=1:  g = grad ex . [0,-1] - ik ex 
#  bc=2:  g = grad ex . [ 1,0] - ik ex  
#  bc=3:  g = grad ex . [ 0,1] - ik ex  
#  bc=4:  g = grad ex . [-1,0] - ik ex  

# -g
minusg = CoefficientFunction( [ 1j*(ks+k)*ex, -1j*(kc-k)*ex, 
				-1j*(ks-k)*ex, 1j*(kc+k)*ex] )

# -g * ik
minusgik = CoefficientFunction( [1j*k*1j*(ks+k)*ex,-1j*k*1j*(kc-k)*ex, 
			      -1j*k*1j*(ks-k)*ex, 1j*k*1j*(kc+k)*ex])


ik = CoefficientFunction(1j*k)
minusik = CoefficientFunction(-1j*k)

# Finite element spaces                      (p = 0,1,2,...)
fs1 = L2(mesh, order=4,complex=True) 		# e, v, deg p+2
fs2 = H1(mesh, order=3,complex=True)  		# u, w, deg p+1
fs3 = HDiv(mesh, order=2,complex=True, 
	orderinner=1) 		# q, r, deg p
fs = FESpace([fs1,fs2,fs3], complex=True)

lf = LinearForm(fs)
lf.components[2] += LFI("neumannhdiv", coef=minusg) # -<g,r.n>
lf.components[1] += LFI("neumann", coef=minusgik)   # +<g,ik*wh> = -<g*ik,wh>

#   LHS: We use standard and DPG integrators to make the
#   composite sesquilinear form
#    a( e,u,q,uh ; v,w,r,wh )  
#          = Y(e;v) + b(u,q; v)
#                   + conj( b(w,r; e) +  c(w,r ; u,q) ).

dpg = BilinearForm(fs, linearform=lf, symmetric=False, eliminate_internal=True)

dpg += BFI("gradgrad", coef=[2,1,one])
dpg += BFI("eyeeye", coef=[2,1,minusksqr])
dpg += BFI("flxtrc", coef=[3,1,minus])
dpg += BFI("flxflxbdry", coef=[3,3,minus])
dpg += BFI("flxtrcbdry", coef=[3,2,minusik])
dpg += BFI("trctrcbdry", coef=[2,2,minusksqr])
dpg.components[0] += BFI("laplace", coef=one)
dpg.components[0] += BFI("mass", coef=ksqr)

# Solve iteratively:
euq = GridFunction(fs)

#c = Preconditioner(dpg, type="direct")
#c = Preconditioner(dpg, type="local")
#c = Preconditioner(dpg, type="vertexschwarz", addcoarse=True)
c = Preconditioner(dpg, type="vertexschwarz")
c.Update()

# segfaults after constructor "cg solve for complex system"
# BVP didn't like innerproduct or solver kwargs
# maybe need to use CGSolver instead of BVP for this
#n2 = BVP(bf=dpg, lf=lf, gf=euq, pre=c, solver="cg", innerproduct="hermitean",
BVP(bf=dpg, lf=lf, gf=euq, pre=c, prec=1.e-10, maxsteps=1000).Do()

# Compute error
uu = GridFunction(fs2)
u = CoefficientFunction(euq.components[1])
absL2error = sqrt(Integrate( (u-ex) * Conj(u-ex), mesh, order=6 ))
print("absL2error: " , absL2error)

# Visualize
Draw(u*Conj(u),mesh, "absu2")

#numproc visualization see_real_part 
#        -scalarfunction=euq.2:1 -subdivision=4  -nolineartexture 

