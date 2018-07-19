from ngsolve import *
from netgen.csg import CSGeometry
from math import pi
from numpy import log
from ctypes import CDLL

libDPG = CDLL("../libDPG.so")

ngsglobals.msg_level = 1

geo = CSGeometry("../pde/magnet.geo")
mesh = Mesh("../pde/magnet.vol.gz")

geometryorder = 3

# I doubt if this is correct.. 
F = CoefficientFunction( (0,10,0), (0, 0, 0) )

# not correct yet..
# we can pass periodic a list used_idnrs Identification numbers to be made periodic
v = Periodic(HCurl(mesh, order=3, dirichlet=[1]))
#define fespace v -type=hcurlho_periodic -order=3
#                 -yends=[0,1] -xends=[0,1] -dirichlet=[1]

u = GridFunction(v)
#define gridfunction u -fespace=v 

a = BilinearForm(v, symmetric=True, symmetric=True, spd=True)
#define bilinearform a -fespace=v -symmetric -spd
a += BFI("curlcurledge",coef=(1.0))
a += BFI("massedge",coef=(0.001))

f = LinearForm(v)
f += LFI("curledge",coef=F)
# gives RuntimeError: LFI for dimension3 not available

c = Preconditioner(a, type="direct")
c.Update()

BVP(bf=a, lf=f, gf=u, pre=c)

acurl = BilinearForm(v, symmetric=True, nonassemble=True)
acurl += BFI("curlcurledge", coef=(1.0))

DrawFlux(bf=acurl,gf=u,label="flux")

# is this possible in python?
#numproc visualization npv1 -vectorfunction=flux -clipsolution=vector -subdivision=3  -clipvec=[0,0,-1] 
