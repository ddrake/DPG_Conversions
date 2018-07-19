# Using adaptivity in DPG. A simple laplace equation example.

from ngsolve import *
from netgen.geom2d import SplineGeometry
from math import pi
from numpy import log
from ctypes import CDLL

libDPG = CDLL("../libDPG.so")

ngsglobals.msg_level = 1

geo = SplineGeometry("../pde/square.in2d")
mesh = Mesh("../pde/square2.vol.gz")
SetHeapSize(int(1e7))
one = CoefficientFunction(1)
minus = CoefficientFunction(-1.0)
lam = CoefficientFunction(1+1j)

# Source (exact solution unknown)
f = CoefficientFunction( (1+1j)*exp( -100.0*(x*x+y*y) ) )

# Compound finite element space:
fs1 = H1(mesh, order=4, dirichlet=[1], complex=True)# p+1
fs2 = HDiv(mesh, order=3, complex=True,
	flags={ "orderinner": 1})	# p
fs3 = L2(mesh, order=5, complex=True)	# p+2

fs = FESpace([fs1,fs2,fs3], complex=True)

# Forms:
dpg = BilinearForm(fs, eliminate_internal=True)
dpg += BFI("gradgrad", coef=[1,3,lam])  # (grad u, grad v) + Hermitian transpose 
dpg += BFI("flxtrc", coef=[2,3,minus])  # - << q.n, v >>   + Hermitian transpose
dpg.components[2] += BFI("laplace", coef=one)  	# (grad e, grad v)
dpg.components[2] += BFI("mass", coef=one)    	# (e,v) 

lf = LinearForm(fs)
lf.components[2] += LFI("source", coef=f)

#gridfunction uqe -fespace=fs
#numproc bvp n2 -bilinearform=dpg -linearform=lf 
#        -gridfunction=uqe -solver=direct

# Solve
# gridfunction uqe -fespace=fs
# numproc bvp n2 -bilinearform=dpg -linearform=lf 
#        -gridfunction=uqe -solver=direct

uqe = GridFunction(fs)

c = Preconditioner(dpg, type="direct")

# I think we want to do something like this from the Adaptive example in the docs
def SolveBVP():
	fes.Update()
	uqe.Update()
	dpg.Assemble()
	f.Assemble()
	inv = CGSolver(fes.mat, c.mat)
	uqe.vec.data = inv * f.vec

#def CalcError():
	# todo: ??

# Estimate & Mark
#dg0 = L2(mesh, order=0)
#eestim = GridFunction(dg0)
#numproc enormsc estimate_using_e_norm
#        -solution=uqe -estimator=eestim -fespace=fs -bilinearform=dpg
#	-yintegrators=[2,3] -yspaces=[3]
#numproc markelements mark_large_error_elements
#        -error=eestim -minlevel=1 -factor=0.5 


# Visualize imaginary part 
#numproc visualization see_solution
#        -scalarfunction=uqe.1:1 -subdivision=4  -nolineartexture
