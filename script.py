#run as python script.py path_to_spree_coordinates path_to_brandeburg_coordinates path_to_satellite_coordinates
import numpy as np
import os
from scipy.optimize import minimize
from scipy.stats import norm
from library import *
import sys


path_to_spree_file=sys.argv[1]
path_to_brandeburg_file=sys.argv[2]
path_to_satellite_file=sys.argv[3]

#earth radius
R=6371

###READ DATA FROM FILES
spree=[]
brandeburg=[]
satellite=[]

for x in open(path_to_spree_file,'r'):
    x=x.split(',')
    x=[float(xx) for xx in x]
    spree.append(x)

for x in open(path_to_brandeburg_file,'r'):
    x=x.split(',')
    x=[float(xx) for xx in x]
    brandeburg.append(x)

for x in open(path_to_satellite_file,'r'):
    x=x.split(',')
    x=[float(xx) for xx in x]
    satellite.append(x)

spree=np.array(spree)
satellite=np.array(satellite)
brandeburg=np.array(brandeburg[0])

###COORDINATE SYSTEM CHANGE
spree_km=np.array([latlon2xy(x) for x in spree])
brandeburg_km=np.array(latlon2xy(brandeburg))

#get the great circle
#get 3-d vectors corresponding to the paths'extremes
a=latlon2xyz(satellite[0][0],satellite[0][1])
b=latlon2xyz(satellite[1][0],satellite[1][1])
#get the angle between them
max_angle=angle(a,b)
#sample the angle parameter
alphas=[i*max_angle/50. for i in range(51)]
#get the vector tangent to the sphere and orthogonal to b
w=np.cross(np.cross(b,a),b)
#normalize vectors
w_n=w/np.linalg.norm(w)
b_n=b/np.linalg.norm(b)
#great circle in cartesian 3-d coordinates
circle_xyz=[R*b_n*np.cos(alpha)+R*w_n*np.sin(alpha) for alpha in alphas]
#great circle in lat-lon coordinates
circle_latlon=[xyz2latlon(P) for P in circle_xyz]
#great circle in the xy plane
satellite_km=np.array([latlon2xy(P) for P in circle_latlon])


###DERIVE DISTRIBUTIONS PARAMETERS
#get the sandard deviations of the normal distributions
res_spree = minimize(ob_spree,1, method='nelder-mead',options={'xtol': 1e-50, 'disp': False})   
res_sat = minimize(ob_sat,1, method='nelder-mead',options={'xtol': 1e-50, 'disp': False})   
sd_spree=res_spree.x[0]
sd_sat=res_sat.x[0]
#get location and scale of the lognormal distribution
mean=4.7
mode=3.877
lmean=np.log(mean)
lmode=np.log(mode)
scale=np.sqrt(2/3.*(lmean-lmode))
loc=lmean-.5*scale**2

#optimize the product of the density functions
res = minimize(objective,[0,0],args=(brandeburg_km,spree_km,satellite_km,sd_spree,sd_sat,loc,scale), method='nelder-mead',options={'xtol': 1e-50, 'disp': False})
print 'analyst: ',xy2latlon(res.x)
