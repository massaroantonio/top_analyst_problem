#run as python viz.py path_to_spree_coordinates path_to_brandeburg_coordinates path_to_satellite_coordinates
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.interpolate
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

from library import *
os.mkdir('viz')

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

plt.plot(spree_km[:,0],spree_km[:,1],marker='x', label='Spree',lw=4)
plt.plot(satellite_km[:,0],satellite_km[:,1],label='Satellite',marker='x',lw=4)
plt.scatter([brandeburg_km[0]],[brandeburg_km[1]],marker='o',s=80,label='Brandebourg gate',color='red')
plt.gcf().set_size_inches(10, 10)
plt.xticks(size=20)
plt.yticks(size=20)
plt.xlim(0,20)
plt.ylim(-5,15)
plt.legend(fontsize=20)
plt.savefig('viz/Stilyzed map.png')
plt.gcf().clear()

x=np.linspace(0,20,100)
y=np.linspace(-5,15,100)
xi,yi=np.meshgrid(x,y)

z=np.array([[sat_distr(sat_distance([xi[j,i],yi[j,i]],satellite_km),sd_sat) for i in range(len(x))] for j in range(len(y))])
z=(z-z.min())/(z.max()-z.min())
plt.imshow(z,vmin=z.min(),vmax=z.max(),origin='lower',extent=[x.min(),x.max(),y.min(),y.max()])
plt.plot(spree_km[:,0],spree_km[:,1],marker='x',label='Spree',color='green',lw=4)
plt.plot(satellite_km[:,0],satellite_km[:,1], label='Satellite',color='orange',lw=4)
plt.scatter([brandeburg_km[0]],[brandeburg_km[1]],s=80,marker='o',label='Brandebourg',color='red')
plt.legend(fontsize=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.xlim(0,20)
plt.ylim(-5,15)
plt.gcf().set_size_inches(10, 10)
plt.savefig('viz/Satellite_probability_2d.png')
plt.gcf().clear()

z=np.array([[brand_distr(brand_distance([xi[j,i],yi[j,i]],brandeburg_km),loc,scale) for i in range(len(x))] for j in range(len(y))])
z=(z-z.min())/(z.max()-z.min())
plt.imshow(z,vmin=z.min(),vmax=z.max(),origin='lower',extent=[x.min(),x.max(),y.min(),y.max()])
plt.plot(spree_km[:,0],spree_km[:,1],marker='x',label='Spree',color='green',lw=4)
plt.plot(satellite_km[:,0],satellite_km[:,1], label='Satellite',color='orange',lw=4)
plt.scatter([brandeburg_km[0]],[brandeburg_km[1]],s=80,marker='o',label='Brandebourg',color='red')
plt.legend(fontsize=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.xlim(0,20)
plt.ylim(-5,15)
plt.gcf().set_size_inches(10, 10)
plt.savefig('viz/Brandeburg_probability_2d.png')
plt.gcf().clear()

z=np.array([[spree_distr(spree_distance([xi[j,i],yi[j,i]],spree_km),sd_spree) for i in range(len(x))] for j in range(len(y))])
z=(z-z.min())/(z.max()-z.min())
plt.imshow(z,vmin=z.min(),vmax=z.max(),origin='lower',extent=[x.min(),x.max(),y.min(),y.max()])
plt.plot(spree_km[:,0],spree_km[:,1],marker='x',label='Spree',color='green',lw=4)
plt.plot(satellite_km[:,0],satellite_km[:,1], label='Satellite',color='orange',lw=4)
plt.scatter([brandeburg_km[0]],[brandeburg_km[1]],s=80,marker='o',label='Brandebourg',color='red')
plt.xlim(0,20)
plt.ylim(-5,15)
plt.gcf().set_size_inches(10, 10)
plt.xticks(size=20)
plt.yticks(size=20)
plt.legend(fontsize=20)
plt.savefig('viz/Spree_probability_2d.png')
plt.gcf().clear()

res = minimize(objective,[0,0],args=(brandeburg_km,spree_km,satellite_km,sd_spree,sd_sat,loc,scale), method='nelder-mead',options={'xtol': 1e-50, 'disp': False})
analyst=res.x

z=np.array([[-objective([xi[j,i],yi[j,i]],brandeburg_km,spree_km,satellite_km,sd_spree,sd_sat,loc,scale) for i in range(len(x))] for j in range(len(y))])
z=(z-z.min())/(z.max()-z.min())
plt.imshow(z,vmin=z.min(),vmax=z.max(),origin='lower',extent=[x.min(),x.max(),y.min(),y.max()])
plt.plot(spree_km[:,0],spree_km[:,1],marker='x',label='Spree',color='green',lw=4)
plt.plot(satellite_km[:,0],satellite_km[:,1], label='Satellite',color='orange',lw=4)
plt.scatter([brandeburg_km[0]],[brandeburg_km[1]],s=80,marker='o',label='Brandebourg',color='red')
plt.scatter([analyst[0]],[analyst[1]],s=80,marker='<',label='Analyst',color='orange')
plt.legend(fontsize=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.xlim(0,20)
plt.ylim(-5,15)
plt.gcf().set_size_inches(10, 10)
plt.savefig('viz/Joint_probability_2d')
plt.gcf().clear()


spree=[spree_distr(d/1000.,sd_spree) for d in range(-10000,10000)]
sat=[sat_distr(d/1000.,sd_sat) for d in range(-10000,10000)]
brand=[brand_distr(0.0011+d/1000.,loc,scale) if d>0 else brand_distr(-0.0011-d/1000.,loc,scale) for d in range(-15000,15000)]
plt.plot([d/1000. for d in range(-10000,10000)],spree,label='Spree')
plt.plot([d/1000. for d in range(-10000,10000)],sat,label='Satellite')
plt.plot([d/1000. for d in range(-15000,15000)],brand,label='Brandebourg')
plt.gcf().set_size_inches(10, 8)
plt.legend(fontsize=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.savefig('viz/densities.png')
plt.gcf().clear()


fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.arange(0, 19, 0.25)
Y = np.arange(-3, 15, 0.25)
X, Y = np.meshgrid(X, Y)

z=np.array([[brand_distr(brand_distance([X[i,j],Y[i,j]],brandeburg_km),loc,scale) for j in range(len(Y[i]))] for i in range(len (X[:,0]))])
surf = ax.plot_surface(X, Y, z, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
plt.xticks(size=25)
plt.yticks(size=25)
plt.gcf().set_size_inches(18, 10)
plt.savefig('viz/Brandeburg_probability_3d.png')
plt.gcf().clear()

fig = plt.figure()
ax = fig.gca(projection='3d')
z=np.array([[spree_distr(spree_distance([X[i,j],Y[i,j]],spree_km),sd_spree) for j in range(len(Y[i]))] for i in range(len (X[:,0]))])
surf = ax.plot_surface(X, Y, z, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
plt.xticks(size=25)
plt.yticks(size=25)
plt.gcf().set_size_inches(18, 10)
plt.savefig('viz/Spree_probability_3d.png')
plt.gcf().clear()

fig = plt.figure()
ax = fig.gca(projection='3d')
z=np.array([[sat_distr(sat_distance([X[i,j],Y[i,j]],satellite_km),sd_sat) for j in range(len(Y[i]))] for i in range(len (X[:,0]))])
surf = ax.plot_surface(X, Y, z, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
plt.xticks(size=25)
plt.yticks(size=25)
plt.gcf().set_size_inches(18, 10)
plt.savefig('viz/Satellite_probability_3d.png')
plt.gcf().clear()



fig = plt.figure()
ax = fig.gca(projection='3d')
z=np.array([[-objective([X[i,j],Y[i,j]],brandeburg_km,spree_km,satellite_km,sd_spree,sd_sat,loc,scale) for j in range(len(Y[i]))] for i in range(len (X[:,0]))])
surf = ax.plot_surface(X, Y, z, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
plt.xticks(size=25)
plt.yticks(size=25)
plt.gcf().set_size_inches(18, 10)
plt.savefig('viz/Joint_probability_3d.png')
plt.gcf().clear()