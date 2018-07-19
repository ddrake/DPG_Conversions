# Primal DPG method for real valued Laplace eq. in 2D
# 
##########################################################
# The BVP: 
#     -Delta u + u = f    on Omega 
#                u = 0    on bdry.
#
# The DPG weak formulation: 
#  
#   Y(e;v)       +  b(u,q; v)  = (f,v)
#   b(w,r; e)                  = 0
#
# where 
#
#  Y(e,v)         = (grad e, grad v) + (u,v)
#  b(u,q; v)      = (grad u, grad v) + (u,v) - <<q.n, v>> 
#
# and where
#
# (.,.)   is sum over all element L2 inner products,
# <<.,.>> is sum over all element boundary L2 inner products.
#
# The spaces: 
#     u, w     in H1
#     e, v     in broken H1
#     q, r     element boundary traces of H(div)
##########################################################

from ngsolve import *
from netgen.geom2d import SplineGeometry
from math import pi
from numpy import log
from ctypes import CDLL

libDPG = CDLL("../libDPG.so")

ngsglobals.msg_level = 1

geo = SplineGeometry("../pde/square.in2d")
mesh = Mesh("../pde/square2.vol.gz")
# Just set this if we need to?
#SetHeapSize(int(1e7))
one = CoefficientFunction(1)
minus = CoefficientFunction(-1.0)

# Exact solution for error computation
uex = CoefficientFunction( x*(1-x)*y*(1-y) )
graduex = CoefficientFunction( ( (1-2*x)*y*(1-y), x*(1-x)*(1-2*y) ) )
f = CoefficientFunction( 2*x*(1-x)+2*y*(1-y) + uex )

# DPG's compound finite element space of index p, where p=0,1,2,...
# has these component spaces:
p = 3
fs1 = H1(mesh, order=p+1, dirichlet=[1]) 		# p+1
fs2 = HDiv(mesh, order=p, orderinner=1)	                # p
fs3 = L2(mesh, order=p+2)				# p+2
fs = FESpace([fs1,fs2,fs3])

# Forms: Specify a dpg integrator operating on component 
# spaces I and J as:   <integrator name> <I> <J> <coeff>.
dpg = BilinearForm(fs, symmetric=True, eliminate_internal=True)
dpg += BFI("gradgrad", coef=[1,3,one])
dpg += BFI("flxtrc", coef=[2,3,minus])
dpg += BFI("eyeeye", coef=[1,3,one])
dpg.components[2] += BFI("laplace", coef=one)
dpg.components[2] += BFI("mass", coef=one)

lf = LinearForm(fs)
lf.components[2] += LFI("source", coef=f)

## Solve:  After static condensation, the DPG system is guaranteed 
## to be symmetric and positive definite, so we use CG to solve. 
uqe = GridFunction(fs)

c = Preconditioner(dpg,type="local") 
#c.Update()
# If you want to use a direct solver instead, use this:
# c = Preconditioner(dpg, type="direct")

# the output of this procedure matches the pde output exactly for 
# steps 0 - 18, but the outputs don't match for steps 19 - 21
#n2 = BVP(bf=dpg, lf = lf, gf = uqe, pre=c, prec=1.e-10, maxsteps=1000).Do()
dpg.Assemble()
lf.Assemble()
inv = CGSolver(dpg.mat, c.mat, precision=1.e-10, maxsteps=1000)
lf.vec.data += dpg.harmonic_extension_trans * lf.vec
uqe.vec.data = inv * lf.vec
uqe.vec.data += dpg.harmonic_extension * uqe.vec
uqe.vec.data += dpg.inner_solve * lf.vec
#u = GridFunction(fs1)
#q = GridFunction(fs2)

u = uqe.components[0]
q = uqe.components[1]

## Compute L2 and H1 errors in u:    Calculate || u - U ||.
L2err = sqrt(Integrate( (u - uex) * (u - uex), mesh, order=(p+2)*2))
H1err = sqrt(Integrate( (grad(u) - graduex) * (grad(u) - graduex), 
			mesh, order=(p+1)*2))

# These don't match, but are close. 
# The tablerror.out values generated by the pde
# are not rooted.  After rooting, the L2 error is 8.e-15 vs 2.e-14
# H1 error is 1.8e-13 vs 4.96e-13 (a little more than a factor of 2
# in both cases
print("L2err: ", L2err)
print("H1err: ", H1err)


# Compute approx H^(-1/2) norm of error in q:
RT = HDiv(mesh, order=6) # Space to extend numerical traces,
qRT = GridFunction(RT)   # of degree >= deg(fs2).

hdivipe = BilinearForm(RT) # H(div) inner product.
hdivipe += BFI("divdivhdiv", coef=one)
hdivipe += BFI("masshdiv", coef=one)

qex = GridFunction(RT)
dg0 = L2(mesh, order=0)
qerrsqr = GridFunction(dg0)

qex.Set(graduex) # Interp/proj exact Q=graduex.
qRT.Set(q) # Interp/proj q to RT space (here we need q treated as coefficient)

Draw(u)
#numproc fluxerr  calc_fluxerror_fracnorm  # Calculate ||q - Q||.
#	-exactq=qex -discreteq=qRT -extensionspace=RT 
#	-fespace=fs -hdivproduct=hdivipe -errorsquareq=qerrsqr  
#
## Write error values into file (one line per refinement & solve):
#numproc writefile tablulate_errors
#	-variables=[mesh.levels,fes.fs.ndof,fes.fs1.ndof,integrate.calcL2err.value,integrate.calcH1err.value,fluxerr.calc_fluxerror_fracnorm.value] 
#	-filename=tablerror.out
#
## Visualize
#numproc visualization see_real_part 
#        -scalarfunction=uqe.1 -subdivision=4  -nolineartexture 
