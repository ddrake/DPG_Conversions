from ngsolve import *
from netgen.csg import CSGeometry
from math import pi
from numpy import log
from ctypes import CDLL

libDPG = CDLL("../libDPG.so")

ngsglobals.msg_level = 1

# This mesh consists of a cylindrical magnet inside a bounding box
# I think it should be possible to recreate this mesh, adding boundary labels
geo = CSGeometry("../pde/magnet.geo")
mesh = Mesh("../pde/magnet.vol.gz")
print(mesh.GetBoundaries())
print(mesh.GetMaterials())
geometryorder = 3

# So far, haven't found documentation on pde file coefficent functions with this syntax
# I think it may represent a pair of vectors since it's used by the curledge integrator
# another thought is it might be a list of two vectors, one per domain or boundary condition?
#F = CoefficientFunction( (0,10,0), (0, 0, 0) )
#F = CoefficientFunction( (0,10,0,0, 0, 0), dims=(2,3))
#F = CoefficientFunction( ((0,10,0), (0, 0, 0)) )
#F = CoefficientFunction( [0,10,0, 0, 0, 0] )
F = CoefficientFunction( [(0,10,0), (0, 0, 0)] ) # this is the only construction that doesn't give any errors
# however, the solution is not visible and when I change the variable for viewing it seg faults

# TODO: Maybe we could make a new mesh with periodic boundaries
# or set the periodic boundaries of the existing mesh.
# we can pass periodic a list used_idnrs Identification numbers to be made periodic
#v = Periodic(HCurl(mesh, order=3, dirichlet=[1]), yends=[0,1], xends=[0,1])
v = FESpace("hcurlho_periodic", mesh, order=3, xends=[0,1], yends=[0,1] )
#define fespace v -type=hcurlho_periodic -order=3
#                 -yends=[0,1] -xends=[0,1] -dirichlet=[1]

u = GridFunction(v)
#define gridfunction u -fespace=v 

a = BilinearForm(v, symmetric=True, spd=True)
#define bilinearform a -fespace=v -symmetric -spd
a += BFI("curlcurledge",coef=(1.0))
a += BFI("massedge",coef=(0.001))

f = LinearForm(v)
f += LFI("curledge",coef=F) # I think this is a standard integrator
# for most formulations of F, gives RuntimeError: LFI for dimension3 not available

c = Preconditioner(a, type="direct")
c.Update()

BVP(bf=a, lf=f, gf=u, pre=c)

# non-assembled linear form for operator application only.
acurl = BilinearForm(v, symmetric=True, nonassemble=True)
acurl += BFI("curlcurledge", coef=(1.0))

DrawFlux(bf=acurl,gf=u,label="flux")

#numproc visualization npv1 -vectorfunction=flux -clipsolution=vector -subdivision=3  -clipvec=[0,0,-1] 
