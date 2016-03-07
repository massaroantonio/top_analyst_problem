#run as python script.py path_to_spree_coordinates path_to_brandeburg_coordinates path_to_satellite_coordinates
import numpy as np
import os
from scipy.optimize import minimize
from scipy.stats import norm
from library1 import *
import sys


path_to_spree_file=sys.argv[1]
path_to_brandeburg_file=sys.argv[2]
path_to_satellite_file=sys.argv[3]

#read the data from files
spree_f=open(path_to_spree_file,'r')
spree=[]
for x in spree_f:
    x=x.split(',')
    x=[float(xx) for xx in x]
    spree.append(x)
spree=np.array(spree)

brandeburg_f=open(path_to_brandeburg_file,'r')
brandeburg=[]
for x in brandeburg_f:
    x=x.split(',')
    x=[float(xx) for xx in x]
    brandeburg.append(x)
brandeburg=np.array(brandeburg[0])

satellite_f=open(path_to_satellite_file,'r')
satellite=[]
for x in satellite_f:
    x=x.split(',')
    x=[float(xx) for xx in x]
    satellite.append(x)
satellite=np.array(satellite)

#change the coordinate system
spree_km=np.array([to_xy(x) for x in spree])
satellite_km=np.array([to_xy(x) for x in satellite])
brandeburg_km=np.array(to_xy(brandeburg))

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
analyst=res.x
print 'analyst: ',to_latlon(res.x)

