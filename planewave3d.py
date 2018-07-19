# Solve Helmholtz equation with impedance boundary condition
#
#      - Delta u - k*k u = f    on Omega
#       n.grad u - i*k u = g    on bdry
#
# on a 3D domain. 

from ngsolve import *
from netgen.csg import CSGeometry
from math import pi
from numpy import log
from ctypes import CDLL

libDPG = CDLL("../libDPG.so")

#ngsglobals.msg_level = 1

geo = CSGeometry("../pde/cube6bc.geo")

mesh = Mesh("../pde/cube6bc4.vol.gz")

SetHeapSize(int(1e7))

one   = CoefficientFunction(1)
minus = CoefficientFunction(-1.0)
dd    = CoefficientFunction(1.0)
cc    = CoefficientFunction(1.0)

# number of waves in a unit-sized domain 
nwav = 2

# propagation angle 
theta = (pi/11.0)
phi   = (pi/3.0)

# wavenumber
k  = (2.0*pi*nwav)
k1 = (k*cos(theta)*sin(phi))
k2 = (k*sin(theta)*sin(phi))
k3 = (k*cos(phi))

ksqr  = (k*k) 
minusksqr = (-k*k) 

ik 		= CoefficientFunction(1j*k)
minusik 	= CoefficientFunction(-1j*k)
cik 		= CoefficientFunction(cc*1j*k)
minuscik 	= CoefficientFunction(-cc*1j*k)
diksqr 		= CoefficientFunction((dd*1j*k)*(dd*1j*k))
minusdiksqr 	= CoefficientFunction(-(dd*1j*k)*(dd*1j*k))


# exact solution ex = exp(I*k.x)  for error computation
ex = CoefficientFunction(exp(1j*(k1*x + k2*y + k3*z)))     

# grad ex = I * [k1,k2,k3] * exp( I * [k1,k2,k3] . [x,y,z] )

# We need to set b.c. using g = n.grad ex - ik ex  on each face:
#   
#  bc=1:  x=0,  g = grad ex . [-1, 0, 0] - ik ex 
#                 = -i (k1 + k)  ex
#  bc=2:  y=0,  g = grad ex . [ 0,-1, 0] - ik ex  
#                 = -i (k2 + k) ex
#  bc=3:  z=0,  g = grad ex . [ 0, 0,-1] - ik ex  
#                 = -i (k3 + k) ex 
#  bc=4:  x=1,  g = grad ex . [ 1, 0, 0] - ik ex  
#                 =  i (k1 - k) ex
#  bc=5:  y=1,  g = grad ex . [ 0, 1, 0] - ik ex  
#                 =  i (k2 - k) ex
#  bc=6:  z=1,  g = grad ex . [ 0, 0, 1] - ik ex  
#                 =  i (k3 - k) ex
#

# -g
minusg = CoefficientFunction( [	(1j*(k1+k)*ex), (1j*(k2+k)*ex), 
				(1j*(k3+k)*ex), (1j*(k-k1)*ex), 
				(1j*(k-k2)*ex), (1j*(k-k3)*ex) ])

# -g * ik
minusgik = CoefficientFunction([(1j*k*1j*(k1+k)*ex), (1j*k*1j*(k2+k)*ex), 
				(1j*k*1j*(k3+k)*ex), (1j*k*1j*(k-k1)*ex), 
				(1j*k*1j*(k-k2)*ex), (1j*k*1j*(k-k3)*ex)])

f = CoefficientFunction(0.0)

# finite element spaces                              (p = 0,1,2,...)
fs1 = L2(mesh, order=6,complex=True) 		# e, v, deg p+2
fs2 = H1(mesh, order=5,complex=True) 		# u, w, deg p+1
fs3 = HDiv(mesh, order=4,complex=True,
	flags={"orderinner":1})  		# q, r, deg p 
fs = FESpace([fs1,fs2,fs3], complex=True)

# forms 
lf = LinearForm(fs)
lf.components[0] += LFI("source", coef=f)
lf.components[2] += LFI("neumannhdiv", coef=minusg)  	# -<g,r.n>
lf.components[1] += LFI("neumann", coef=minusgik) 	# +<g,ik*wh> = -<g*ik,wh>

# We need to make the sesquilinearform
#  a( e,u,q,uh ; v,w,r,wh ) = 
#   Y(e;v)  +  b(u,q,uh; v) + conj( b(w,r,wh; e) +  c(w,r,wh ; u,q,uh) )

# The symmetric=False option is necessary for Hermitian too.
dpg = BilinearForm(fs, symmetric=False, linearform=lf, eliminate_internal=True)
dpg += BFI("gradgrad", 		coef=[2,1,one]) 	# (grad u, grad v)
dpg += BFI("eyeeye", 		coef=[2,1,minusksqr])	# - k*k*(u,v) 
dpg += BFI("flxtrc", 		coef=[3,1,minus])	# - <<q.n, v>> 
dpg += BFI("flxflxbdry", 	coef=[3,3,minus])	# - <q.n, r.n>
dpg += BFI("flxtrcbdry", 	coef=[3,2,minusik])	# + <q.n, ik w>
dpg += BFI("trctrcbdry", 	coef=[2,2,minusksqr])	# - <ik u, ik w> 
dpg.components[0] += BFI("laplace", coef=one)
dpg.components[0] += BFI("mass", coef=ksqr)

# solve:
euqf = GridFunction(fs)

#c = Preconditioner(dpg, type="direct")
#c = Preconditioner(dpg, type="local")
#c = Preconditioner(dpg, type="vertexschwarz", flags={"addcoarse":True})
c = Preconditioner(dpg, type="vertexschwarz")
c.Update()

# segfaults in BVP constructor
# BVP didn't like innerproduct or solver kwargs
# maybe need to use CGSolver instead of BVP for this
BVP(bf=dpg, lf=lf, gf=euqf, pre=c, prec=1.e-10, maxsteps=1000).Do()
#numproc bvp n2 -bilinearform=dpg -linearform=lf 
#        -gridfunction=euqf -preconditioner=c
#	-solver=cg -innerproduct=hermitean 
#	-prec=1.e-10 -maxsteps=1000


uu = GridFunction(fs2)
uu = euq.components[1]
absL2error = sqrt(Integrate( (uu-ex) * Conj(uu-ex), mesh, order=10))
print("absL2error: ", absL2error)
#gridfunction uu -fespace=fs2 -addcoef 
#numproc getcomp ngc3 -comp=2 -compoundgf=euqf -componentgf=uu
#coefficient err ( abs(uu-ex) )
#numproc integrate absL2error -coefficient=err

# Visualize
Draw(uu*Conj(uu),mesh, "absu2")
#numproc visualization see_real_part 
#        -scalarfunction=euqf.2:1 -subdivision=4  -nolineartexture 

