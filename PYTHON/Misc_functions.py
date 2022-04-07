#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================================================
# === 0. Import Packages ===========================================================
# ==================================================================================
import glob
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.text as mtext
import matplotlib.colors as mcolors
import matplotlib as mpl
import matplotlib.cm as cmx

# to interpolate values
import scipy.interpolate as spi
import numpy as np
import netCDF4 as nc
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import datetime as dt

import geopandas

## - Define class LegendTitle to have subtitles in Legend
## - get_rolling_values(x,kernel_size)
## - get_scattering_info(x,kernel_size,type_metric=0)
## - filter_big_change_from_ref(X,Xref,thresh)
## - haversine(lat1, lon1, lat2, lon2)
## - interpolate_along_track(lon_mod,lat_mod,field_mod,lon_obs,lat_obs) 
## - get_contour(X_array,Y_array)
## - copy_NCfile(originalfile,targetfile,isdebug)
## - add_ocean_label(lon,lat)
## - init_map_cartopy(ax0,limx=(-180,180),limy=(-80,80))
## -
## -
## -
## -
'''
- read_offnadir_files(filenames):  Function that reads the offnadir files and appends the results outputs: time_swim,lat_offnadir,lon_offnadir,phi,phi_geo,seg_lat,seg_lon,modulation_spectra,fluctuation_spectra,wave_spectra,klin,inci,seg_inci,kwave,dk
- read_nadir_data(filename):time_alti0, lat_alti0, lon_alti0, swh_alti0: all values along track
#			- time_alti, lat_alti, lon_alti, swh_alti: filtered values (by availability)
- read_nadir_data_wind(filename): reads one nadir file at 5 Hz (native)
- read_nadir_data_wind_1_5Hz(filename): time_alti0,lat_alti0,lon_alti0,swh_alti_1hz0,wind_alti_1hz0,swh_alti_5hz0,wind_alti_5hz0,time_alti,lat_alti,lon_alti,swh_alti_1hz,wind_alti_1hz,swh_alti_5hz,wind_alti_5hz

- read_box_data_one_side(filename,iswest):
- read_Model_fields(filename): # 2D
- copy_NCfile(originalfile,targetfile,isdebug):
- apply_box_selection_to_track(lon,lat,time,field,lon_min,lon_max,lat_min,lat_max):
- draw_spectrum_macrocyle(axs,fluctu_spec_mm,phi_geo_mm,phi_loc_mm,freq,wvlmin,wvlmax,dphi,cNorm,cMap): 
- interpolate_along_track(lon_mod,lat_mod,field_mod,lon_obs,lat_obs):
- get_contour(X_array,Y_array):
- class LegendTitle(object):
- get_macrocycles(lon_offnadir,lat_offnadir,phi_loc,phi_geo,lon_min,lon_max,lat_min,lat_max):
- 
'''


# --------------------------------------------------------------------------------
# -- class LegendTitle to have subtitles in Legend
class LegendTitle(object):
	def __init__(self, text_props=None):
		self.text_props = text_props or {}
		super(LegendTitle, self).__init__()
	
	def legend_artist(self, legend, orig_handle, fontsize, handlebox):
		x0, y0 = handlebox.xdescent, handlebox.ydescent
		title = mtext.Text(x0, y0, r'\textbf{' + orig_handle + '}', usetex=True, **self.text_props)
		handlebox.add_artist(title)
		return title


# --------------------------------------------------------------------------------
# --- 1. Get values for a rolling windows ----------------------------------------
# --------------------------------------------------------------------------------
def get_rolling_values(x,kernel_size):
    lenx=len(x)
    rank_kernel=kernel_size//2
    B=np.zeros(lenx+2*rank_kernel)
    B[rank_kernel:-rank_kernel]=x
    
    indA = np.atleast_2d(np.arange(rank_kernel)) + np.atleast_2d(np.arange(lenx)).T
    
    return B[indA]
    
# --------------------------------------------------------------------------------
# --- 2. Get scattering info -----------------------------------------------------
# --------------------------------------------------------------------------------  
def get_scattering_info(x,kernel_size,type_metric=0):
    y=get_rolling_values(x,kernel_size)
    med = np.nanmedian(y,axis=1)

    if type_metric==0: # std
        r = np.std(y,axis=1)
    elif type_metric==1: # RMS from median
        r=np.sum((y-np.tile((med),(kernel_size//2,1)).T)**2,axis=1)
    elif type_metric==2: # Max-min
        r=(np.amax(y,axis=1)-np.amin(y,axis=1))/np.nanmedian(y,axis=1)
    return r    

# --------------------------------------------------------------------------------
# --- 3. Filter big changre from ref ---------------------------------------------
# --------------------------------------------------------------------------------     
def filter_big_change_from_ref(X,Xref,thresh):
    condition1= (np.isfinite(X))&(X!=0)&(np.isfinite(Xref))&(Xref!=0)
    ind = np.where(condition1)[0]
    
    X0 = np.inf*np.ones(X.shape)
    X0[ind]=np.exp(np.abs(np.log(X[ind]/Xref[ind])))
    
    Xlim=thresh*np.nanmedian(np.abs(X0[np.isfinite(X0)]))

    ind_suspect1 = np.where(np.abs(X0)>Xlim)[0]
    X1 = np.copy(X)
    X1[ind_suspect1]=np.nan
    
    return X1    
    
# -- function HAVERSINE ------------------------------------
# Calculates the distance [km] between 2 points on the globe
# using the Haversine formula
# inputs : lat1, lon1, lat2, lon2
# outputs: distance between the 2points in km
def haversine(lat1, lon1, lat2, lon2):
	# This code is contributed
	# by ChitraNayal
	# from https://www.geeksforgeeks.org/haversine-formula-to-find-distance-between-two-points-on-a-sphere/
	# distance between latitudes
	# and longitudes
	dLat = (lat2 - lat1) * np.pi / 180.0
	dLon = (lon2 - lon1) * np.pi / 180.0
	
	# convert to radians
	lat1 = lat1 * np.pi / 180.0
	lat2 = lat2 * np.pi / 180.0
	
	# apply formulae
	a = (pow(np.sin(dLat / 2), 2) +
	     pow(np.sin(dLon / 2), 2) *
	         np.cos(lat1) * np.cos(lat2));
	
	rad = 6371
	c = 2 * np.arcsin(np.sqrt(a))
	return rad * c	    

# ---  Linear ND Interpolator -----------------------------------------------
def interpolate_along_track(lon_mod,lat_mod,field_mod,lon_obs,lat_obs):
#  np.shape(field_mod)=(len(lat_mod),len(lon_mod))
	x, y = np.meshgrid(lon_mod, lat_mod)
	hs_interp = spi.LinearNDInterpolator(((x.flatten(),y.flatten())),field_mod.flatten())
	field_interp = hs_interp(lon_obs,lat_obs)
	
	return field_interp

def compute_dB(P,P0=10**-12):
    return 10*np.log10(P/P0)   
     
# -- function GET_CONTOUR ------------------------------------
# inputs : X_array,Y_array : 2D arrays which contours need to be extracted
# outputs: X_bound,Y_bound
def get_contour(X_array,Y_array):
	X_bound = np.zeros(2*np.size(X_array,0)+2*np.size(X_array,1))
	Y_bound = np.zeros(2*np.size(Y_array,0)+2*np.size(Y_array,1))
	
	count0 = np.size(X_array,0)
	X_bound[0:count0]=X_array[:,0]
	count1 = count0+np.size(X_array,1)
	X_bound[count0:count1]=X_array[-1,:]
	count0 = count1
	count1 = count1 + np.size(X_array,0)
	X_bound[count0:count1]=X_array[-1::-1,-1]
	count0 = count1
	count1 = count1 + np.size(X_array,1)
	X_bound[count0:count1]=X_array[0,-1::-1]
	
	count0 = np.size(Y_array,0)
	Y_bound[0:count0]= Y_array[:,0]
	count1 = count0+np.size(Y_array,1)
	Y_bound[count0:count1]=Y_array[-1,:]
	count0 = count1
	count1 = count1 + np.size(Y_array,0)
	Y_bound[count0:count1]=Y_array[-1::-1,-1]
	count0 = count1
	count1 = count1 + np.size(Y_array,1)
	Y_bound[count0:count1]=Y_array[0,-1::-1]
	
	return X_bound,Y_bound 

def get_contour_box_fromvectors(x,y):
	lenX = len(x)
	lenY = len(y)
	x_contour=np.zeros(2*lenX+2*lenY)
	y_contour=np.zeros(2*lenX+2*lenY)

	x_contour[0:lenX]=x
	y_contour[0:lenX]=np.ones(lenX)*y[0]
	x_contour[lenX:lenX+lenY]=x[-1]*np.ones(lenY)
	y_contour[lenX:lenX+lenY]=y
	x_contour[lenX+lenY:2*lenX+lenY]=x[-1::-1]
	y_contour[lenX+lenY:2*lenX+lenY]=y[-1]*np.ones(lenX)
	x_contour[2*lenX+lenY:2*(lenX+lenY)]=x[0]*np.ones(lenY)
	y_contour[2*lenX+lenY:2*(lenX+lenY)]=y[-1::-1]

	return x_contour,y_contour

	
# function to copy ncfiles in local path
# by reading to original file entierely and writing everything in a newfile
# use as:
# -  'copy_NCfile(file1,file2,0)' if you don't need log info
# -  'copy_NCfile(file1,file2,1)' in order to see the details of the processing
def copy_NCfile(originalfile,targetfile,isdebug):
	# -- 0. Open datasets -------------------
	originalDS = nc.Dataset(originalfile)
	targetDS = nc.Dataset(targetfile,mode='w')
	if isdebug:
		print('Reading datasets : done !')
	
	# -- 1. Read dimensions from original dataset and apply to target
	if isdebug:
		print('Reading + copying dimensions ...')
	for name, dim in originalDS.dimensions.items():
		targetDS.createDimension(name, len(dim) if not dim.isunlimited() else None)
	if isdebug:
		print('Reading + copying dimensions : done !')
	
	# -- 2. Read attributes and copy to target	
	if isdebug:
		print('Reading + copying attributes ...')
	targetDS.setncatts({a:originalDS.getncattr(a) for a in originalDS.ncattrs()})
	if isdebug:
		print('Reading + copying attributes : done !')
		
	# -- 3. For each variable: read and copy variable (with attributes)
	if isdebug:
		print('Reading + copying variables ...')
	count = 0
	for name, var in originalDS.variables.items():
		count = count +1
		if isdebug:
			print('Start of var '+str(count)+' over '+str(len(originalDS.variables.items())))
		targetDS.createVariable(name, var.dtype, var.dimensions)
		targetDS.variables[name].setncatts({a:var.getncattr(a) for a in var.ncattrs()})
		targetDS.variables[name][:] = originalDS.variables[name][:]
		if isdebug:
			print('Done for var '+str(count)+' over 49')
			
	if isdebug:
		print('Reading + copying variables : done !')
		
	# -- 4. close files
	targetDS.close()
	originalDS.close()

# ---- add ocean label ---------------------------------
def add_ocean_label(lon,lat):
    shape = geopandas.read_file("/home/mdecarlo/DATA/GOaS_v1/GOaS_v1_20211214/goas_v01.shp")
    ocean_label=np.zeros(np.shape(lon),dtype=int)-1 # set default = -1
    points_geo=geopandas.points_from_xy(lon,lat)
    order_k=range(10)
    ocean_names=[]
    ocean_names_short=[]
    ocean_names.append('Land') # set default to Land
    ocean_names_short.append('Land') # set default to Land
    for k in range(10):
        # --- get the name of oceans - both short and complete 
        A=shape.name[k].split(' ',-1)
        if len(A)<=3:
            if len(A)==3:
                A[0]=A[0][0]+'.'
            B = ' '.join(A[:-1])
        elif len(A)>3:
            B = A[-2]
        ocean_names.append(shape.name[k])
        ocean_names_short.append(B)
        # --- for each polygon check if points are inside ----
        if (shape.geometry[k]).geom_type=='MultiPolygon':
            for k_in in range(len(shape.geometry[k])):
                SG=shape.geometry[order_k[k]][k_in].convex_hull
                ind=points_geo.within(SG)
                ocean_label[ind]=order_k[k]
        else:
            SG=shape.geometry[order_k[k]].convex_hull
            ind=points_geo.within(SG)
            ocean_label[ind]=order_k[k]
    
    return ocean_label,ocean_names,ocean_names_short

def init_map_cartopy(ax0,limx=(-180,180),limy=(-80,80)):
    ax0.set_extent([limx[0],limx[1],limy[0],limy[1]],crs=ccrs.PlateCarree())
    # Grid Lines of the map ----------------------
    g1 = ax0.gridlines(ccrs.PlateCarree(), draw_labels=True)
    g1.xlabels_top = False
    g1.ylabels_right = False

    g1.xlabel_style = {'size': 18}
    g1.ylabel_style = {'size': 18}

    # --- Add land features ---- 
    # Definition ---
    land = cfeature.NaturalEarthFeature(
    category='physical',
    name='land',
    scale='50m',
    edgecolor='face',
    zorder=-10,
    facecolor=cfeature.COLORS['land'])
    # add the features ---
    ax0.add_feature(land)
    ax0.coastlines()
    
    return ax0, g1


def multiple_histo_groupby(DatasetGroupedby,xbins,var,ax,space=0.2,xticklabels=None,xticklabelsRot=90,xticks=None,ylabel=None,groupLabels=None):
# --- Histo of the variable 'var' from a dataset groupedby
	LenGroup=len(DatasetGroupedby)
	x=np.arange(len(xbins)-1)
	print(len(x))
	x0 = x - 0.5 + (space/2)
	width = (1-space)/LenGroup
	# x = 0.5*(bins[1:]+bins[0:-1])
	count_label=0
	for grlabel, group in DatasetGroupedby:
		counts, _ = np.histogram(group[var],bins=xbins)
		if groupLabels != None:
				grlabel=groupLabels[count_label]
		_= ax.bar(x0 + count_label*width, counts, width, label=grlabel)
		count_label=count_label+1
# ax.hist(ds_old.Ocean_flag,bins=np.arange(-2,11)+0.5,alpha=0.5)
	if xticks!=None:
		_=ax.set_xticks(xticks)
	else:
		_=ax.set_xticks(x)
	if ylabel!=None:
		_=ax.set_ylabel(ylabel)
	if xticklabels!=None:
		_=ax.set_xticklabels(xticklabels,rotation=xticklabelsRot)
	_=plt.legend()
	plt.grid()
    
    
def print_summary_xarray(ds):
	str1='Dimensions:                ('
	for idim in ds.dims:
		str1=str1+idim+': '+str(ds.dims[idim])+', '
	str1=str1[:-2]+')'
	print(str1)
	print(ds.coords)
	print(ds.data_vars)

	
	
def get_datetime_fromDOY(YYYY,DOY):
	return pd.to_datetime(dt.datetime(YYYY,1,1))+pd.Timedelta(days=DOY-1)
	

def get_nearest_model_interpolator(lon_lims=[-180., 180.],lat_lims=[-78.,83.],lon_step=0.5,lat_step=0.5):
    lon_mod = np.arange(lon_lims[0],lon_lims[1],lon_step)
    lat_mod = np.arange(lat_lims[0],lat_lims[1],lat_step)
    lon_interpolator=spi.interp1d(lon_mod,lon_mod,kind='nearest')
    lat_interpolator=spi.interp1d(lat_mod,lat_mod,kind='nearest')
    
    return lon_interpolator, lat_interpolator

def get_nearest_model_interpolator_index(lon_lims=[-180., 180.],lat_lims=[-78.,83.],lon_step=0.5,lat_step=0.5):
    lon_mod = np.arange(lon_lims[0],lon_lims[1],lon_step)
    lat_mod = np.arange(lat_lims[0],lat_lims[1],lat_step)
    ilon_interpolator=spi.interp1d(lon_mod,np.arange(len(lon_mod)),kind='nearest')
    ilat_interpolator=spi.interp1d(lat_mod,np.arange(len(lat_mod)),kind='nearest')
    
    return ilon_interpolator, ilat_interpolator
    
def read_Hs_model(date1,path=None):
	if path == None:
		path='/home/ref-ww3/GLOBMULTI_ERA5_GLOBCUR_01/GLOB-30M/'
	yr = date1.year
	T1str=date1.strftime('%Y%m')
	path_year=os.path.join(path,str(yr),'FIELD_NC')
	strfile='LOPS_WW3-GLOB-30M_'+T1str+'.nc'
	ds=xr.open_dataset(os.path.join(path_year,strfile))
	ds1=ds.sel(time=slice(date1-np.timedelta64(90,'m'), date1+np.timedelta64(90,'m')))
	return ds1
	
# ---  plot 2D model for one date 
def plot_2D_model_Hs(ax,date1,path=None,limx=(-180,180),limy=(-80,80),norm=None):
	ds=read_Hs_model(date1,path=None)
	if path == None:
		path='/home/ref-ww3/GLOBMULTI_ERA5_GLOBCUR_01/GLOB-30M/'
	if norm==None:
		swhNorm = mcolors.Normalize(vmin=5, vmax=12)	
	ds1=ds.sel(longitude=slice(limx[0],limx[1]),latitude=slice(limy[0],limy[1]))
	
	ax,g1=init_map_cartopy(ax,limx=limx,limy=limy)
	
	ax.pcolormesh(ds1.longitude,ds1.latitude,np.squeeze(ds1.hs),norm=norm)



#############################################################################
###  WAVES PHYSICS ##########################################################

def phase_speed_from_k(k,depth=None,g=9.81):
	if depth == 'None':
		# print("Deep water approximation")
		C=np.sqrt(g/k)
	else:
		# print("General case")
		C=np.sqrt(g*np.tanh(k*depth)/k)
	return C
			
def phase_speed_from_sig_k(sig,k):
	return sig/k
	
def group_speed_from_k(k,depth=None,g=9.81):
	C=phase_speed_from_k(k,depth=depth,g=g)
	if depth == 'None':
		# print("Deep water approximation")
		Cg=C/2
	else:
		# print("General case")
		Cg=C*(0.5+ ((k*depth)/(np.sinh(2*k*depth)) ))
	return Cg

def sig_from_k(k,D=None,g=9.81):
	if D=='None':
		# print("Deep water approximation")
		sig = np.sqrt(g*k)
	else:
		# print("General case")
		sig = np.sqrt(g*k*np.tanh(k*D))
	return sig

def f_from_sig(sig):
	return sig/(2*np.pi)

def sig_from_f(f):
	return 2*np.pi*f
	
def period_from_sig(sig):
	return (2*np.pi)/sig

def period_from_wvl(wvl,D=None):
	k=(2*np.pi)/wvl
	sig=sig_from_k(k,D=D)
	T = period_from_sig(sig)
	return T

def k_from_f(f,D=None,g=9.81):
	# inverts the linear dispersion relation (2*pi*f)^2=g*k*tanh(k*dep) to get 
	#k from f and dep. 2 Arguments: f and dep. 
	eps=0.000001
	sig=np.array(2*np.pi*f)
	if D=='None':
		# print("Deep water approximation")
		k=sig**2/g
	else:
		Y=D*sig**2/g
		X=np.sqrt(Y)
		I=1
		F=1.
		while (abs(np.max(F)) > eps):
			H=np.tanh(X)
			F=Y-(X*H)
			FD=-H-(X/(np.cosh(X)**2))
			X=X-(F/FD)

		k=X/D

	return k # wavenumber

def dfdk_from_k(k,D=None):
	Cg = group_speed_from_k(k,depth=D,g=9.81)
	return Cg/(2*np.pi)

def PM_spectrum_f(f,fm,g=9.81):
# There are 2 ways of writing the PM spectrum:
#  - eq 12 of Pierson and Moskowitz (1964) with exp(-0.74 * (f/fw)**-4) where fw=g*U10/(2*pi)
#  - eq of Hasselmann et al. 1973 with exp(-5/4 * (f/ fm)**-4) where fm is the max frequency  ...
# See Hasselmann et al. 1973 for the explanation 
	alpha=8.1*10**-3
	E = alpha*g**2*(2*np.pi)**-4*f**-5*np.exp((-5/4)*((fm/f)**4))
	return E

def PM_spectrum_k(k,fm,D=None,g=9.81):
# There are 2 ways of writing the PM spectrum:
#  - eq 12 of Pierson and Moskowitz (1964) with exp(-0.74 * (f/fw)**-4) where fw=g*U10/(2*pi)
#  - eq of Hasselmann et al. 1973 with exp(-5/4 * (f/ fm)**-4) where fm is the max frequency  ...
# See Hasselmann et al. 1973 for the explanation 
	alpha=8.1*10**-3
	f=sig_from_k(k,D=D)/(2*np.pi)
	Ef = alpha*g**2*(2*np.pi)**-4*f**-5*np.exp((-5/4)*((fm/f)**4))
	dfdk = dfdk_from_k(k,D=D)
	
	return Ef*dfdk

def define_spectrum_PM_cos2n(k,th,T0,thetam,n=4,D=None):
	Ek=PM_spectrum_k(k,1/T0,D=D)
	dth=th[1]-th[0]
	Eth=np.cos(th-thetam)**(2*n)
	II=np.where(np.cos(th-thetam) < 0)[0]
	Eth[II]=0
	sth=sum(Eth*dth)
	Ekth=np.broadcast_to(Ek,(len(th),len(k)))*np.broadcast_to(Eth,(len(k),len(th))).T /sth
	return Ekth,k,th

def define_Gaussian_spectrum_kxky(kX,kY,T0,theta_m,sk_theta,sk_k,D=None):
	if (len(kX.shape)==1) & (len(kY.shape)==1):
		kX,kY = np.meshgrid(kX,kY)
	elif (len(kX.shape)==1) | (len(kY.shape)==1):
		print('Error : kX and kY should either be: \n      - both vectors of shapes (nx,) and (ny,) \n  OR  - both matrices of shape (ny,nx)')
		print('/!\ Proceed with caution /!\ kX and kY have been flattened to continue running')
		kX = kX.flatten()
		kY = kY.flatten()
	kp = k_from_f(1/T0,D=D)
	# rotation of the grid => places kX1 along theta = theta_m
	kX1 = kX*np.cos(theta_m)+kY*np.sin(theta_m)
	kY1 = -kX*np.sin(theta_m)+kY*np.cos(theta_m)
	
	Z1_Gaussian =1/(2*np.pi*sk_theta*sk_k)* np.exp( - 0.5*((((kX1-kp)**2)/((sk_k)**2))+kY1**2/sk_theta**2))
	
	return Z1_Gaussian,kX,kY

### ----- Change variables from spectrum -----------------------------
def spectrum_from_fth_to_kth(Efth,f,th,D=None):
    shEfth = np.shape(Efth)
    if len(shEfth)<2:
        print('Error: spectra should be 2D')
    else:
        if shEfth[0]==shEfth[1]:
            print('Warning: same dimension for freq and theta.\n  Proceed with caution: The computation is done considering Efth = f(f,th)')
        elif (shEfth[1]==len(f)) &(shEfth[0]==len(th)):
            Efth = np.swapaxes(Efth,0,1)
        else:
            print('Error: Efth should have the shape : (f,th)')
    shEfth = np.shape(np.moveaxis(Efth,0,-1))
    k=k_from_f(f,D=D)
    dfdk=dfdk_from_k(k,D=D)
    Ekth = Efth*np.moveaxis(np.broadcast_to(dfdk,shEfth),-1,0)
    return Ekth, k, th

def spectrum_from_kth_to_kxky(Ekth,k,th):
    shEkth = np.shape(Ekth)
    if len(shEkth)<2:
        print('Error: spectra should be 2D')
    else:
        if shEkth[0]==shEkth[1]:
            print('Warning: same dimension for k and theta.\n  Proceed with caution: The computation is done considering Ekth = f(k,th)')
        elif (shEkth[1]==len(k)) &(shEkth[0]==len(th)):
            Ekth = np.swapaxes(Ekth,0,1)
        else:
            print('Error: Efth should have the shape : (k,th)')
    shEkth2 = np.shape(np.moveaxis(Ekth,0,-1)) # send k-axis to last -> in order to broadcast k along every dim
    shEkth2Dkth = Ekth.shape[0:2] # get only shape k,th for the broadcast of the dimensions kx,ky

    if np.max(th)>100:
        th=th*np.pi/180
    kx = np.moveaxis(np.broadcast_to(k,shEkth2Dkth[::-1]),-1,0) * np.cos(np.broadcast_to(th,shEkth2Dkth))
    ky = np.moveaxis(np.broadcast_to(k,shEkth2Dkth[::-1]),-1,0) * np.sin(np.broadcast_to(th,shEkth2Dkth))
    Ekxky = Ekth/np.moveaxis(np.broadcast_to(k,shEkth2),-1,0)
    return Ekxky, kx, ky

def spectrum_from_fth_to_kxky(Efth,f,th,D=None):
    shEfth = np.shape(Efth)
    if len(shEfth)<2:
        print('Error: spectra should be 2D')
    else:
        if shEfth[0]==shEfth[1]:
            print('Warning: same dimension for freq and theta.\n  Proceed with caution: The computation is done considering Efth = f(f,th)')
        elif (shEfth[1]==len(f)) &(shEfth[0]==len(th)):
            Efth = np.swapaxes(Efth,0,1)
        else:
            print('Error: Efth should have the shape : (f,th)')
    shEfth2 = np.shape(np.moveaxis(Efth,0,-1)) # send f-axis to last -> in order to broadcast f along every dim
    shEfth2Dfth = Efth.shape[0:2] # get only shape f,th for the broadcast of the dimensions kx,ky
    k=k_from_f(f,D=D)
    dfdk=dfdk_from_k(k,D=D)
    if np.max(th)>100:
        th=th*np.pi/180
    
    kx = np.moveaxis(np.broadcast_to(k,shEfth2Dfth[::-1]),-1,0) * np.cos(np.broadcast_to(th,shEkth2Dkth))
    ky = np.moveaxis(np.broadcast_to(k,shEfth2Dfth[::-1]),-1,0) * np.sin(np.broadcast_to(th,shEkth2Dkth))
    Ekxky = Efth * np.moveaxis(np.broadcast_to(dfdk /k,shEfth2),-1,0)
    return Ekxky, kx, ky

def spectrum_to_kxky(typeSpec,Spec,ax1,ax2,D=None):
    if typeSpec==0: # from f,th
        Ekxky, kx, ky = spectrum_from_fth_to_kxky(Spec,ax1,ax2,D=D)
    elif typeSpec==1: # from k,th
        Ekxky, kx, ky = spectrum_from_kth_to_kxky(Spec,ax1,ax2)
    else:
        print('Error ! typeSpec should be 0 = (f,th) or 1 = (k,th)')
        Ekxky = Spec
        kx = ax1
        ky = ax2
    return Ekxky, kx, ky


def surface_from_Z1kxky(Z1,kX,kY,nx=None,ny=None,dx=None,dy=None,dkx=None,dky=None):
	# usually when doing X,Y=np.meshgrid(x,y) with size(x)=nx and size(y)=ny => size(X)=size(Y)= (ny,nx)
	kX0 = np.unique(kX)
	kY0 = np.unique(kY)
	if nx==None:
		nx = Z1.shape[1]
	if ny==None:
		ny = Z1.shape[0]
	shx = np.floor(nx/2-1)
	shy = np.floor(ny/2-1)
	if (dx==None):
		if dkx == None:
			dx = 2*np.pi/((kX0[1] - kX0[0])*nx)
		else:
			dx = 2*np.pi/(dkx*nx)
		
	if (dy==None):
		if dky == None:
			dy = np.floor(2*np.pi/((kY0[1] - kY0[0])*ny))
		else:
			dy = 2*np.pi/(dky*ny)
	
	if (dkx==None):
		dkx = 2*np.pi/(dx*nx)
	if (dky==None):
		dky = 2*np.pi/(dy*ny)
	########################################################################################
	# SIDE NOTE :										#
	# obtain dkx, dky from dx and dy in order to account for eventual rounding of np.pi	#
	# considering that dkx has been defined according to the surface requisite (nx,dx)	#
	# Eg:											#
	# # initialisation to compute spectrum							#
	# nx = 205										#
	# dx = 10										#
	# dkx = 2*np.pi/(dx*nx)								#
	# kX0 = dkx*np.arange(-nx//2+1,nx//2+1)						#
	# 											#
	# # Compute from kX0 and nx (found from kX0.shape)					#
	# dkx2 = kX0[1] - kX0[0]								#
	# dxbis = (2*np.pi/(dkx2*nx))								#
	# dkx3 = 2*np.pi/(dxbis*nx)								#
	#											#
	# print('dkx = ',dkx)     		=> 0.0030649684425266273			#
	# print('dkx2 = ',dkx2) 		=> 0.0030649684425266277			#
	# print('dkx3 = ',dkx3) 		=> 0.0030649684425266273			#
	#											#
	########################################################################################
	
	rg = np.random.normal(0,1,(ny,nx))
	zhats=np.roll(np.sqrt(2*Z1*dkx*dky)*np.exp(1j*2*np.pi*rg),(-int(shy),-int(shx)),axis=(0,1))
	ky2D=np.roll(kY,(-int(shy),-int(shx)),axis=(0,1)) # checks that ky2D(1,1)=0 ... 
	kx2D=np.roll(kX,(-int(shy),-int(shx)),axis=(0,1)) # checks that kx2D(1,1)=0 ... 
	
	S1 = np.real(np.fft.ifft2(zhats))*(nx*ny)
	# S1 = np.real(np.fft.ifft2(zhats))*(nx**2)
	# # Bertrand's convolution... 
	# F2r = np.fft.fft2(rg) 
	# F2Z1 = np.roll(np.sqrt(2*Z1_Gaussian*dkx*dky),(-int(shy),-int(shx)),axis=(0,1))
	# # F2Z1 = np.fft.fft2(2*Z1_Gaussian*dkx*dky)
	# S1 = np.real(np.fft.ifft2(F2r*F2Z1))
	X = np.arange(0,nx*dx,dx) # from 0 to (nx-1)*dx with a dx step
	Y = np.arange(0,ny*dy,dy)
	
	return S1,X,Y
	
'''
def PM_spectrum_k(k,fm,g=9.81):
	pmofk(k,T0,H)
	alpha=8.1*10**-3
	
	w0=2*np.pi/T0
	w=np.sqrt(g*k*tanh(k*H))
	Cg=(0.5+k.*H/sinh(2.*k.*H)).*w./k;
	pmofk=0.008.*g.^2.*exp(-0.74.*(w./w0).^(-4))./(w.^5).*Cg+5.9;
	
	E = alpha*g**2*(2*np.pi)**-4*f**-5*np.exp((-5/4)*((fm/f)**4))
	return E
'''
