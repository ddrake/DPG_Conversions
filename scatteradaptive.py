# Use the DPG method to adaptively compute a wave scattered 
# from a triangular scatterer. Adaptivity puts relatively more
# elements where the beam-like scattered wave is present.
#
# One can use any of the different possible implementation 
# techniques for the Helmholtz equation with impedance bc.
# (The results were observed not to depend on which.)
#
################################################################
# Compute scattered wave from a triangular scatterer. 
# The problem for utotal = uincident + uscattered is 
#
#  -Delta utotal - k*k utotal = 0,    outside scatterer
#                      utotal = 0     on scatterer boundary 
#                                     (sound-soft b.c.).
#
# Given uincident, we compute the scattered wave by solving:
#
#  -Delta uscattered - k*k uscattered = 0,  outside scatterer,
#                 uscattered = -uincident,  on scatterer boundary,
#   n.grad uscattered - ik uscattered = 0,  on rest of boundary.
#
# Press the Solve button repeatedly to proceed with successive 
# adaptive iterations.
################################################################
from ngsolve import *
from netgen.geom2d import SplineGeometry
from math import pi
from numpy import log
from ctypes import CDLL

libDPG = CDLL("../../libDPG.so")

ngsglobals.msg_level = 1

geo = SplineGeometry("../../pde/triangularscatterer.in2d")
mesh = Mesh("../../pde/triangularscatterer.vol.gz")

# Just set this if we need to?
SetHeapSize(int(1e7))

# Propagation angle 
theta = (pi/3.0)

# Wavenumber
k  = (5*pi)

# Sound-soft (Dirichlet) conditions by penalty
penalty = 1.e5

kc = (k*cos(theta))
ks = (k*sin(theta))
ksqr  = (k*k) 

# The incident wave
uincident = CoefficientFunction(exp(1j*(kc*x + ks*y)))

one 		= CoefficientFunction(1)
minus 		= CoefficientFunction(-1.0)
dd    		= CoefficientFunction(1.0)
cc    		= CoefficientFunction(1.0)
minusksqr 	= CoefficientFunction(-k*k) 

ik          = CoefficientFunction(1j*k)
minusik     = CoefficientFunction(-1j*k)
cik         = CoefficientFunction(cc*1j*k)
minuscik    = CoefficientFunction(-cc*1j*k)
diksqr      = CoefficientFunction((dd*1j*k)*(dd*1j*k))
minusdiksqr = CoefficientFunction(-(dd*1j*k)*(dd*1j*k))


# The boundary has 5 parts, the fifth being the scatterer boundary. 
# We impose outgoing impedance bc everywhere, except the scatterer 
# boundary, where sound-soft boundary condition is imposed. 

notsoft = CoefficientFunction([(-1.0), (-1.0), (-1.0), (-1.0), 0])

notsoftik = CoefficientFunction([ (-1j*k), (-1j*k), (-1j*k), (-1j*k), 0])

notsoftksqr = CoefficientFunction([(-k*k),(-k*k),(-k*k),(-k*k),0])

# Dirichlet bc imposed on scatterer boundary (bc=5) by penalty
soft = CoefficientFunction( [0,0,0,0,penalty])

inc = CoefficientFunction([ 0,0,0,0,-penalty * exp(1j*(kc*x + ks*y))])

# Finite element spaces                              (p = 0,1,2,...)
fs1 = L2(mesh, order=4, complex=True)  # e, v, deg p+2
fs2 = L2(mesh, order=3, complex=True)  # u, w, deg p+1
fs3 = HDiv(mesh, order=2, complex=True, 
		flags={"orderinner":True})  # q, r, deg p 
fs4 = H1(mesh, order=3, complex=True,
		flags={"orderinner":True})     # uh, wh, deg p+1
fs  = FESpace([fs1,fs2,fs3,fs4], flags={"complex":True})

# Forms 
#   RHS:
lf = LinearForm(fs)
lf.components[3] += LFI("neumann", coef=inc)  # - penalty <uincident, v> 

#   LHS:
dpg = BilinearForm(fs, flags={"linearform":lf, "nonsym":True, "eliminate_internal":True})
dpg += BFI("gradgrad", coef=[2,1,one])            # (grad u, grad v)
dpg += BFI("eyeeye", coef=[2,1,minusksqr])        # - k*k*(u,v) 
dpg += BFI("flxtrc", coef=[3,1,minus])            # - <<q.n, v>> 
dpg += BFI("trctrc", coef=[4,1,cik])              # + <<c * ik uh, v>>
dpg += BFI("trctrc", coef=[2,1,minuscik])         # - <<c * ik u,  v>>
dpg += BFI("trctrc", coef=[4,4,diksqr])           # - <<d * ik uh, d * ik wh>> 
dpg += BFI("trctrc", coef=[4,2,minusdiksqr])      # + <<d * ik uh, d * ik w >> 
dpg += BFI("trctrc", coef=[2,2,diksqr])           # - <<d * ik u,  d * ik w >> 
dpg += BFI("flxflxbdry", coef=[3,3,notsoft])      # - <q.n, r.n>
dpg += BFI("flxtrcbdry", coef=[3,4,notsoftik])    # + <q.n, ik wh>
dpg += BFI("trctrcbdry", coef=[4,4,notsoftksqr])  # - <ik uh, ik wh> 

dpg.components[3] += BFI("robin", coef=soft) # + penalty <u, v> 
dpg.components[0] += BFI("laplace", coef=one)
dpg.components[0] += BFI("mass", coef=ksqr)

# Solve:
euqf = GridFunction(fs)

#preconditioner c  -type=vertexschwarz -bilinearform=dpg 
#preconditioner c  -type=local -bilinearform=dpg 
c = Preconditioner(dpg, type="direct")

# I think we want to do something like this from the Adaptive example in the docs
def SolveBVP():
	dpg.Update()
	euqf.Update()
	dpg.Assemble()
	lf.Assemble()
	inv = CGSolver(dpg.mat, c.mat)
	euqf.vec.data = inv * lf.vec

#def CalcError():
	# todo: ??

#
#numproc bvp n2 -bilinearform=dpg -linearform=lf 
#        -gridfunction=euqf
#	-preconditioner=c
#	-solver=cg 
#	-innerproduct=hermitean 
#	-prec=1.e-10 -maxsteps=1000

# Estimate error & Mark elements for local refinement
dg0 = L2(mesh, order=0)
eestim = GridFunction(dg0)

#numproc enormsc estimate_using_e_norm -bilinearform=dpg -fespace=fs 
#        -solution=euqf 
#	-estimator=eestim      # output element error estimate values 
#	-yintegrators=[13,14]  # integrators forming Y-innerproduct 
#	-yspaces=[1]           # the Y-space

#numproc markelements mark_large_error_elements
#        -error=eestim -minlevel=1 -factor=0.25 


## Visualize
#numproc visualization see_real_part 
#        -scalarfunction=euqf.2:1 -subdivision=4  -nolineartexture 


