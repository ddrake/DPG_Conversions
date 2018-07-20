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
one = CoefficientFunction(1)
minus = CoefficientFunction(-1)
lam = CoefficientFunction(1+1j)

# Source (exact solution unknown)
f = CoefficientFunction( (1+1j)*exp( -100.0*(x*x+y*y) ) )

# Compound finite element space:
p = 3
fs1 = H1(mesh, order=p+1, dirichlet=[1], complex=True)
fs2 = HDiv(mesh, order=p, complex=True, orderinner=1)	
fs3 = L2(mesh, order=p+2, complex=True)	

fs = FESpace([fs1,fs2,fs3], complex=True)

# Forms:
dpg = BilinearForm(fs, eliminate_internal=True)
dpg += BFI("gradgrad", coef=[1,3,lam])          # (grad u, grad v) + Hermitian transpose 
dpg += BFI("flxtrc", coef=[2,3,minus])          # - << q.n, v >>   + Hermitian transpose
dpg.components[2] += BFI("laplace", coef=one)  	# (grad e, grad v)
dpg.components[2] += BFI("mass", coef=one)    	# (e,v) 

lf = LinearForm(fs)
lf.components[2] += LFI("source", coef=f)

uqe = GridFunction(fs)

c = Preconditioner(dpg, type="direct")

def Solve():
    fes.Update()
    uqe.Update()
    dpg.Assemble()
    lf.Assemble()
    inv = CGSolver(dpg.mat, c.mat, precision=1.e-10, maxsteps=1000)
    lf.vec.data += dpg.harmonic_extension_trans * lf.vec
    uqe.vec.data = inv * lf.vec
    uqe.vec.data += dpg.harmonic_extension * uqe.vec
    uqe.vec.data += dpg.inner_solve * lf.vec

l = []    # l = list of estimated total error

def CalcError():
    # compute the flux:
    space_flux.Update()      
    gf_flux.Update()
    flux = lam * grad(gfu)        
    gf_flux.Set(flux) 
    
    # TODO: Can we bring the enorms numproc into python?
    # compute estimator:
    err = 1/lam*(flux-gf_flux)*(flux-gf_flux)
    eta2 = Integrate(err, mesh, VOL, element_wise=True)
    maxerr = max(eta2)
    l.append ((fes.ndof, sqrt(sum(eta2))))
    print("ndof =", fes.ndof, " maxerr =", maxerr)
    
    # mark for refinement:
    for el in mesh.Elements():
        mesh.SetRefinementFlag(el, eta2[el.nr] > 0.25*maxerr)


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
