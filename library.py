import numpy as np
import os
from scipy.optimize import minimize
from scipy.stats import norm


#transformation from latitude-longitude to XY
def to_xy(p):
    SW_lat = 52.464011 
    SW_lon = 13.274099
    return [(p[1]-SW_lon)*np.cos(SW_lat * np.pi / 180)*111.323,(p[0]-SW_lat)*111.323]

#transformation from XY to latitude-longitude
def to_latlon(p):
    SW_lat = 52.464011 
    SW_lon = 13.274099
    lon=(p[0]/111.323)*(1/np.cos(SW_lat * np.pi / 180))+SW_lon
    lat=(p[1]/111.323)+SW_lat
    return [lat,lon]

#objective function to be minimized to determine the standard deviation of the probability distribution wrt
#the distance from the Spree river
def ob_spree(s):
    return np.abs(norm.cdf(2.73,0,s)-norm.cdf(-2.73,0,s)-0.95)

#objective function to be minimized to determine the standard deviation of the probability distribution wrt
#the distance from the satellite path
def ob_sat(s):
    return np.abs(norm.cdf(2.4,0,s)-norm.cdf(-2.4,0,s)-0.95)

#lognormal probability density function
def lognorm_pdf(x,mu,sigma):
    return 1/(x*sigma*np.sqrt(2*np.pi))*np.exp(-.5*((np.log(x)-mu)/sigma)**2)

#Probability density functions based on the distance
def brand_distr(d,loc_brand,scale_brand):
    return lognorm_pdf(d,loc_brand,scale_brand)
def spree_distr(d,sd_spree):
    return norm.pdf(d,0,sd_spree)
def sat_distr(d,sd_sat):
    return norm.pdf(d,0,sd_sat)

#returns the distance of point from line passing through line_point0 and line_point1
def point_line_distance(point,line_point0,line_point1):
    x=point[0]
    y=point[1]
    x0=line_point0[0]
    y0=line_point0[1]
    x1=line_point1[0]
    y1=line_point1[1]
    a=y1-y0
    b=x0-x1
    c=y0*(x1-x0)+x0*(y0-y1)
    return np.abs(a*x+b*y+c)/np.sqrt(a**2+b**2)
    
#returns the normal projection of point onto the line passing through line_poit0 and line_point1
def normal_projection_point_to_line(point, line_point0,line_point1):
    x=point[0]
    y=point[1]
    x0=line_point0[0]
    y0=line_point0[1]
    x1=line_point1[0]
    y1=line_point1[1]
    m=(y1-y0)/(x1-x0)
    q=y0-x0*(y1-y0)/(x1-x0)
    x_n=(m/(m**2+1))*(y+1/m*x-q)
    y_n=m*x_n+q
    return(np.array([x_n,y_n]))

#returns the distance between point and a segment defined by its extremal points segment_point0 and segment_point1
def point_segment_distance(point,segment_point0,segment_point1):
    N=normal_projection_point_to_line(point, segment_point0,segment_point1)
    min_x=min(segment_point0[0],segment_point1[0])
    max_x=max(segment_point0[0],segment_point1[0])
    min_y=min(segment_point0[1],segment_point1[1])
    max_y=max(segment_point0[1],segment_point1[1])
    if N[0]< min_x or N[0]> max_x or N[1]< min_y or N[1]> max_y:
        return min([np.linalg.norm(point-segment_point0),np.linalg.norm(point-segment_point1)])
    else:
        return point_line_distance(point,segment_point0,segment_point1)

#returns the distance of point from the river spree
def spree_distance(point,spree_km):
    return min([point_segment_distance(point,spree_km[i],spree_km[i+1]) for i in range(len(spree_km)-1)])

#returns the distance of point from the satellite path
def sat_distance(point,satellite_km):
    return point_segment_distance(point,satellite_km[0],satellite_km[1])

#returns the distance of a point from the Brandebourg gate
def brand_distance(point,brandeburg_km):
    point=np.array(point)
    return np.linalg.norm(point-brandeburg_km)

#joint probability to be optimized
def objective(point,brandeburg_km,spree_km,satellite_km,sd_spree,sd_sat,loc,scale):
	return -brand_distr(brand_distance(point,brandeburg_km),loc,scale)*spree_distr(spree_distance(point,spree_km),sd_spree)*sat_distr(sat_distance(point,satellite_km),sd_sat)
