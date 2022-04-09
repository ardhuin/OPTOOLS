#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================================================
# === 0. Import Packages ===========================================================
# ==================================================================================
# from obspy import read, read_inventory,UTCDateTime
# -- to work with basic stuff
import glob
import os
import sys
import time
# to deal with .pickle
import pickle
import pandas as pd
# to interpolate values
import scipy.interpolate as spi
# to work with mathematical and numerical stuff
import numpy as np
# to read/write netCDF4 files
import netCDF4 as nc
from netCDF4 import stringtochar
# -- to work with dates
import datetime as dt
from datetime import datetime
# -- to work with argument parsing
import argparse

# XARRAY
import xarray as xr

import warnings
warnings.filterwarnings("ignore")

# -- to create plots
import matplotlib as mpl
import matplotlib.pyplot as plt
# to work with dates on plots
import matplotlib.dates as mdates
# to work with text on plots
import matplotlib.text as mtext
# to work with colors / colorbar
import matplotlib.colors as mcolors
import matplotlib.cm as cmx
# to work with "videos"
from IPython import display

mpl.rcParams.update({'font.size': 14,'savefig.facecolor':'white'})

from Misc_functions import *
from functions_colmap import *

# ==================================================================================
# === 1. Xarray Reading cfosat files ===============================================
# ==================================================================================
# -- 1.1. READ_OFFNADIR_FILES 
# Reads the offnadir l2s files (from OceanDataLab) and appends the results
# Of interest mostly when one is interested in various incidence angles
# OUPUT DATASET contains : 
# <xarray.Dataset>
# Dimensions:                (l2s_angle: 5, time0: 26019, k: 60)
# Coordinates:  	* k (k) float32, * l2s_angle (l2s_angle)
# Variables: 		
#		phi			('l2s_angle', 'time0')		float32
#		phi_geo 		('l2s_angle', 'time0')		float32
#		lat    		('l2s_angle', 'time0')		float32
#		lon    		('l2s_angle', 'time0')		float32
#		k    			('k',)   			float32
#		time_per_angle		('l2s_angle', 'time0')		datetime64[ns]
#		near_lon    		('l2s_angle', 'time0')		float32
#		near_lat    		('l2s_angle', 'time0')		float32
#		far_lon    		('l2s_angle', 'time0')		float32
#		far_lat    		('l2s_angle', 'time0')		float32
#		wave_spectra    	('l2s_angle', 'time0', 'k')	float32
#		dk    			('l2s_angle', 'k')   		float32
#		incidence    		('l2s_angle', 'time0')   	float32
#		l1a_availability_flag   ('l2s_angle', 'time0') 	float32
#		l2s_angle    		('l2s_angle',)   		int64
#		wavelength    		('l2s_angle', 'k')   		float32
#		wav_lon    		('l2s_angle', 'time0', 'k')	float64
#		wav_lat    		('l2s_angle', 'time0', 'k')	float32

def preprocess_offnadir(ds0):
	ds = ds0[["phi","phi_geo","lat","lon","k","time","near_lon","near_lat","far_lon","far_lat",
	      "wave_spectra","dk","incidence","l1a_availability_flag"]]
	ds = ds.assign({"l2s_angle":((ds.l2s_angle))})
	ds = ds.assign({"wavelength":("k",(2*np.pi/ds.k.data))})
	ds=ds.swap_dims({'time':'time0'})
	ds=ds.reset_coords('time')
	ds = ds.rename_vars({'time':'time_per_angle'})
	# -- prepare the lon_lat associated to the wave number for each ribbon -----
	nr_lon = ds.near_lon.values
	fr_lon = ds.far_lon.values
	# -- Deal with the 180/-180 discontinuity -----------------------
	ind_nr = np.where((nr_lon-fr_lon)>200)[0]
	nr_lon[ind_nr]=nr_lon[ind_nr]-360.
	ind_fr = np.where((fr_lon-nr_lon)>200)[0]
	fr_lon[ind_fr]=fr_lon[ind_fr]-360.

	lon1 = np.zeros(np.shape(nr_lon))
	lon2 = np.zeros(np.shape(nr_lon))
	ind_fr_min = np.where(fr_lon<=nr_lon)[0]
	ind_nr_min = np.where(fr_lon>=nr_lon)[0]
	lon1[ind_fr_min] = fr_lon[ind_fr_min]
	lon2[ind_fr_min] = nr_lon[ind_fr_min]
	lon1[ind_nr_min] = nr_lon[ind_nr_min]
	lon2[ind_nr_min] = fr_lon[ind_nr_min]
	lonlinspace=np.linspace(lon1,lon2,len(ds.k)).T
	# print('shape lonlinspace',np.shape(lonlinspace))
	lonlinspace[ind_fr_min]=np.flip(lonlinspace[ind_fr_min],axis=1)
	ind_lon_above = np.where(lonlinspace<-180)
	lonlinspace[ind_lon_above[0],ind_lon_above[1]]=lonlinspace[ind_lon_above[0],ind_lon_above[1]]+360.

	ds=ds.assign({"wav_lon":(("time0","k"),lonlinspace)})
	ds=ds.assign({"wav_lat":(("time0","k"),np.linspace(ds.near_lat,ds.far_lat,len(ds.k)).T)})
	return ds 

def read_l2s_offnadir_files(offnadir_files):
	ds_all_inci0=xr.open_mfdataset(offnadir_files,concat_dim="l2s_angle", combine="nested",decode_coords=False,
		data_vars='minimal',coords="all",compat='override',preprocess=preprocess_offnadir)
	return ds_all_inci0
	
# -- 1.2. READ_NADIR_DATA ------------------------------
# -- Nadir data can originate from CNES L2 files (1Hz and 5Hz)
#    or from CMEMS L3 files (1Hz)

# -- 3 different preprocessing functions : 
#    - preprocess_nadir_native(ds0): # 5 Hz
#    - preprocess_nadir_1Hz(ds0): # 1 Hz
#    - preprocess_nadir_CMEMS(ds0): # 1 Hz

# -- 2 read functions : 
#    - read_nadir_from_L2_CNES(nadir_files,flag_1Hz=1):
#    - read_nadir_from_L3_CMEMS(nadir_files):

def preprocess_nadir_native(ds0): # 5 Hz
	ds = ds0[["lat_l2anad_0","lon_l2anad_0","nadir_swh_native","nadir_wind_native",
		"nadir_swh_native_validity"]]
	time_alti1 = np.datetime64('2009-01-01') + ds0["time_nadir_native"].data[:,0]* np.timedelta64(1,'s') + ds0["time_nadir_native"].data[:,1]* np.timedelta64(1,'us')
	ds=ds.assign({"time_nadir_native":(("n_mcycles"),time_alti1)})
	ds=ds.rename_vars({"time_nadir_native":"time","lat_l2anad_0":"lat","lon_l2anad_0":"lon",
		"nadir_swh_native":"swh","nadir_swh_native_validity":"swh_flag","nadir_wind_native":"wind"})
	ds=ds.swap_dims({'n_mcycles':'time0'})
	return ds 

def preprocess_nadir_1Hz(ds0): # 1 Hz
	ds = ds0[["lat_nadir_1Hz","lon_nadir_1Hz","time_nadir_1Hz","nadir_swh_1Hz","nadir_wind_1Hz","flag_valid_wind_1Hz",
		"nadir_swh_1Hz_std","nadir_swh_1Hz_used_native","flag_valid_swh_1Hz"]]
	time_alti1 = np.datetime64('2009-01-01') + ds0["time_nadir_1Hz"].data[:]* np.timedelta64(1,'s')
	ds=ds.assign({"time_nadir_1Hz":(("n_nad_1Hz"),time_alti1)})
	ds=ds.rename_vars({"time_nadir_1Hz":"time","lat_nadir_1Hz":"lat","lon_nadir_1Hz":"lon",
		 "nadir_swh_1Hz":"swh","flag_valid_swh_1Hz":"swh_flag","nadir_wind_1Hz":"wind",
		 "flag_valid_wind_1Hz":"wind_flag",})
	ds=ds.swap_dims({'n_nad_1Hz':'time0'})
	return ds 

def preprocess_nadir_CMEMS(ds0): # 1 Hz
	ds = ds0[["latitude","longitude","time","VAVH","VAVH_UNFILTERED"]]
	ds=ds.rename_vars({"latitude":"lat","longitude":"lon",
		"VAVH":"swh"})
	ds=ds.swap_dims({'time':'time0'})	
	return ds

def read_nadir_from_L2_CNES(nadir_files,flag_1Hz=1):
# -- flag_1Hz == 1 : read 1Hz data || flag_1Hz == 0 : read 5 Hz data # n_nad_1Hz # n_mcycles
	if flag_1Hz==1:
		ds_nadir=xr.open_mfdataset(nadir_files,concat_dim="time0", combine="nested",decode_times=False,
			decode_coords=False,data_vars='minimal',coords="minimal",compat='override',
			preprocess=preprocess_nadir_1Hz)
	else:
		ds_nadir=xr.open_mfdataset(nadir_files,concat_dim="time0", combine="nested",decode_times=False,
			decode_coords=False,data_vars='minimal',coords="minimal",compat='override',
			preprocess=preprocess_nadir_native)
	return ds_nadir

def read_nadir_from_L3_CMEMS(nadir_files):
	ds_nadir=xr.open_mfdataset(nadir_files,concat_dim="time0", combine="nested",
		decode_coords=False,data_vars='minimal',coords="minimal",compat='override',
		preprocess=preprocess_nadir_CMEMS)
	return ds_nadir
	
## HERE for READ 
# -- 1.2. read Box data
def preprocess_nadir_boxes(ds0):
	#ds = ds0[["min_lat_l2","max_lat_l2","min_lon_l2","max_lon_l2",
	ds = ds0[["lat_spec_l2","lon_spec_l2","wave_param","wave_param_part","pp_mean","k_spectra","phi_vector","swh_ecmwf"]]
	time_alti1 = np.datetime64('2009-01-01') + ds0["time_l2"].data[:,:,0]* np.timedelta64(1,'s')+ ds0["time_l2"].data[:,:,1]* np.timedelta64(1,'us')
	ds=ds.assign({"time_box":(("n_posneg","n_box"),time_alti1)})
	
	# Prepare transformations (jacobian and slope to waves)
	omega=np.sqrt(ds["k_spectra"]*9.81)
	freq0=omega/(2*np.pi)
	ds=ds.assign({"freq":(freq0)})
	ds=ds.assign(wave_spectra=ds['pp_mean']*(ds['k_spectra']**-2)*(2*np.pi)/(0.5*(9.81/omega)))

	# df=np.nanmean(np.gradient(freq0)) #mean of diff(1/T)
	# domega=np.gradient(omega)
	# dk=np.gradient(WAVENUMBER[0:-4])
	# dtheta=15*np.pi/180 #Hauser et al. 2020
	da_concat=xr.concat([ds0["min_lat_l2"],ds0["max_lat_l2"],ds0["max_lat_l2"],ds0["min_lat_l2"],ds0["min_lat_l2"]],'n_corners')
	ds=ds.assign(lat_corners=da_concat)
	da_concat=xr.concat([ds0["min_lon_l2"],ds0["min_lon_l2"],ds0["max_lon_l2"],ds0["max_lon_l2"],ds0["min_lon_l2"]],'n_corners')
	ds=ds.assign(lon_corners=da_concat)
	ds=ds.rename_vars({"lat_spec_l2":"lat","lon_spec_l2":"lon",})
	ds=ds.swap_dims({'n_posneg':'iswest'})
	ds=ds.swap_dims({'n_box':'time0'})
	return ds 

def read_boxes_from_L2_CNES(nadir_files):
	ds_nadir=xr.open_mfdataset(nadir_files,concat_dim="time0", combine="nested",decode_times=False,
		decode_coords=False,data_vars='minimal',coords="minimal",compat='override',
		preprocess=preprocess_nadir_boxes)
	return ds_nadir

def read_box_data_one_side(filename,iswest):
	nc_swim = nc.Dataset(filename)
	
	# -- read dimensions 
	WAVENUMBER=nc_swim.variables['k_spectra'][:]
	PHI=nc_swim.variables['phi_vector'][:]
		
	# -- read lon/lat at center box
	lat_midrange = nc_swim.variables['lat_l2'][iswest,:] #latitude at mid-range (dim0 coordinate of boxes)
	lon_midrange = nc_swim.variables['lon_l2'][iswest,:]  #longitude at mid-range

	lat_spec=nc_swim.variables['lat_spec_l2'][iswest,:]#latitude spec
	lon_spec=nc_swim.variables['lon_spec_l2'][iswest,:] #longitude spec

	# -- read lon/lat box
	###########
	#Box L2A
	###########
	min_lon_l2_box=nc_swim.variables['min_lon_l2'][iswest,:]#(dim0 coordinate of boxes)
	min_lat_l2_box=nc_swim.variables['min_lat_l2'][iswest,:]
	max_lon_l2_box=nc_swim.variables['max_lon_l2'][iswest,:]
	max_lat_l2_box=nc_swim.variables['max_lat_l2'][iswest,:]
	
	###########
	# Param for each beam
	###########
	wave_param_part_06=nc_swim.variables['wave_param_part'][:,:,iswest,:,0]#(nparam, npartitions, n_posneg, n_box, n_beam_l2)
	wave_param_part_08=nc_swim.variables['wave_param_part'][:,:,iswest,:,1]#(nparam, npartitions, n_posneg, n_box, n_beam_l2)
	wave_param_part_10=nc_swim.variables['wave_param_part'][:,:,iswest,:,2]#(nparam, npartitions, n_posneg, n_box, n_beam_l2)
	
	###########
	# Wave spectra for each beam
	###########
	slope_spec_06deg=nc_swim.variables['pp_mean'][:,:,iswest,:,0] # wavenumber, phi, dim0,time,beam
	slope_spec_08deg=nc_swim.variables['pp_mean'][:,:,iswest,:,1]
	slope_spec_10deg=nc_swim.variables['pp_mean'][:,:,iswest,:,2]
	
	# Prepare transformations (jacobian and slope to waves)
	omega=np.sqrt(WAVENUMBER[0:-4]*9.81)
	freq0=omega/(2*np.pi)
	df=np.nanmean(np.gradient(freq0)) #mean of diff(1/T)
	domega=np.gradient(omega)
	dk=np.gradient(WAVENUMBER[0:-4])
	dtheta=15*np.pi/180 #Hauser et al. 2020
	
	# 3 axes : k,phi,time/box position
	WAVENUMBER_MAT = np.repeat(np.atleast_3d(np.repeat(np.atleast_2d(WAVENUMBER[0:-4]),len(PHI),axis=0).T),np.size(slope_spec_06deg,2),axis=2)
	FREQ = np.repeat(np.atleast_3d(np.repeat(np.atleast_2d(freq0),len(PHI),axis=0).T),np.size(slope_spec_06deg,2),axis=2)
	PHI_MAT = np.repeat(np.repeat(np.atleast_3d(PHI),len(WAVENUMBER[0:-4]),axis=0),np.size(slope_spec_06deg,2),axis=2)
	 
	
	#############
	#---slope spec to wave spec
	############
	wave_spec_06deg=slope_spec_06deg[0:-4,:,:]*WAVENUMBER_MAT[:,:,:]**-2
	wave_spec_08deg=slope_spec_08deg[0:-4,:,:]*WAVENUMBER_MAT[:,:,:]**-2
	wave_spec_10deg=slope_spec_10deg[0:-4,:,:]*WAVENUMBER_MAT[:,:,:]**-2

	#############
	#---Jacobian
	############

	wave_spec_06deg_fth=np.transpose(wave_spec_06deg)*(2*np.pi)/(0.5*(9.81/omega))
	wave_spec_08deg_fth=np.transpose(wave_spec_08deg)*(2*np.pi)/(0.5*(9.81/omega))
	wave_spec_10deg_fth=np.transpose(wave_spec_10deg)*(2*np.pi)/(0.5*(9.81/omega))

	
	nc_swim.close()
	
	return lat_midrange, lon_midrange, lat_spec, lon_spec,min_lon_l2_box, min_lat_l2_box, \
	 max_lon_l2_box, max_lat_l2_box,wave_spec_06deg_fth, wave_spec_08deg_fth,wave_spec_10deg_fth, \
	 WAVENUMBER_MAT, FREQ, PHI_MAT,wave_param_part_06,wave_param_part_08,wave_param_part_10



# -- 1.4. read Model Spectrum
# -- 1.4. read Model fields 
def read_Model_fields(filename):
	if os.path.exists(filename):
		nc_mod = nc.Dataset(filename)
		lat0 = nc_mod.variables['latitude'][:]
		lat0.data[lat0.mask]=0             
		lat_mod = lat0.data      

		lon0 = nc_mod.variables['longitude'][:]
		lon0.data[lon0.mask]=0
		lon_mod = lon0.data

		time_mod0 = nc_mod.variables['time'][:][:]
		time_mod = dt.datetime(1990,1,1) + time_mod0 * dt.timedelta(days=1)

		hs0 = nc_mod.variables['hs'][:]
		hs0.data[hs0.mask]=0
		hs_mod = hs0.data 

		t0m10 = nc_mod.variables['t0m1'][:]
		t0m10.data[t0m10.mask]=0
		t0m1_mod = t0m10.data

		dir_waves = nc_mod.variables['dir'][:]
		dir_waves.data[dir_waves.mask]=0
		dir_wv_mod = dir_waves.data

		uwnd0 = nc_mod.variables['uwnd'][:]
		uwnd0.data[uwnd0.mask]=0
		uwnd_mod = uwnd0.data

		vwnd0 = nc_mod.variables['vwnd'][:]
		vwnd0.data[vwnd0.mask]=0
		vwnd_mod = vwnd0.data
		
		nc_mod.close()
		
		return lat_mod,lon_mod,time_mod,hs_mod,t0m1_mod,dir_wv_mod,uwnd_mod,vwnd_mod

	else:
		print("ERROR : filename doesn't exist ! ")
    



# -- 1.4. Apply_box_selection to track
# this function applies the box selection to the cfosat track
def apply_box_selection_to_ds(ds,lon_min,lon_max,lat_min,lat_max,flag_coords):
	if flag_coords==0:
		indlonlat=np.where((ds.lat>=lat_min)&(ds.lat<=lat_max)&(ds.lon>=lon_min)&(ds.lon<=lon_max))[0]
		ds1=ds.isel(time0=indlonlat)
	elif flag_coords==1:
		indlonlat=np.where(((ds.wav_lat>=lat_min)&(ds.wav_lat<=lat_max)&
				    (ds.wav_lon>=lon_min)&(ds.wav_lon<=lon_max)).any('k'))[0]
		ds1=ds.isel(time0=indlonlat)
	elif flag_coords==2:
		indlonlat=np.where(((ds.wav_lat>=lat_min)&(ds.wav_lat<=lat_max)&
				    (ds.wav_lon>=lon_min)&(ds.wav_lon<=lon_max)).all('k'))[0]
		ds1=ds.isel(time0=indlonlat)    
	return ds1
	
# ==================================================================================	
# === 2. Drawing functions =========================================================
# ==================================================================================
# -- function DRAW_SPECTRUM_MACROCYCLE_NORTH -----------------------
# this function draws the spectrum for a given macrocycle, 
# with two half spectrum contours defined by the wavelength filter
# (wvlmin,wvlmax) and by the safety angle along cfosat track (dphi)
# inputs are:
#		- the axes for the plot to be on
#		- 'ds'		: the dataset with only one incidence angle + only one macrocycle
#		- 'wvlmin'	: the minimum wavelength to be acounted for
#               - 'wvlmax'	: the maximum wavelength to be accounted for
#		- 'dphi	: the safety angle along cfosat track
#		- 'raxis'	: flag to decide if radial axis is freq (=0) or wavelength (=1)
#               - 'isNorthward' : flag to decide if the 'north' of the spectrum corresponds to the geographic North (=1) of to the track (=0)
#		- 'isNE'	: flag to print only the 4 cardinal points (=0) or to also add the NE, NW, SE and SW (=1)
#		- 'cNorm'	: the normalization for cmap
#		- 'cMap'	: the colormap

def draw_spectrum_macrocyle_North(ax,ds,wvlmin=None,wvlmax=None,dphi=0,raxis=0,isNorthward=1,isNE=1,cNorm=None,cMap=None):
    # ds contains a macrocycle
	g= 9.81
	if wvlmin==None:
		wvlmin=ds.wavelength.min()
	if wvlmax==None:
		wvlmax=ds.wavelength.max()
	if cMap==None:
		nipynew = create_nipy_colmap()
		cMap=nipynew
        
	ds = ds.assign({"freq":(("k"),np.sqrt(ds.k.data*g)/(2*np.pi))})
	if isNE:
		ang_label=['N','NE','E','SE','S','SW','W','NW','N']
		ang_geo_ticks=np.arange(0,361,45)%360
	else:
		ang_label=['N','E','S','W','N']
		ang_geo_ticks=np.arange(0,361,90)%360
      
	if isNorthward==1:
		ds = ds.assign({"phi_geo_rad":(("time0"),(90-ds.phi_geo.data)*np.pi/180)})
		ds = ds.sortby('phi_geo')
		ang_ticks = (90-ang_geo_ticks)
		ang_ticks = ang_ticks-(ang_ticks.max()//360)*360
	else:
		ds = ds.assign({"phi_rad":(("time0"),(90-ds.phi.data)*np.pi/180)})
		ds = ds.sortby('phi')
		angl_offset = (ds.phi.data[0]-ds.phi_geo.data[0])
		ang_ticks = (90-ang_geo_ticks-angl_offset)
		ang_ticks = ang_ticks-(ang_ticks.compute().max()//360)*360

 
	ind_k = np.where((ds.wavelength<=wvlmax)&(ds.wavelength>=wvlmin))[0]
	ind_ang_tribord0 = np.where((ds.phi>=dphi)&(ds.phi<=180-dphi))[0]
	# -- Sort in order to have a convex polygon ---
	ind_ang_tribord00=np.argsort(ds.phi.data[ind_ang_tribord0])
	ind_ang_tribord=ind_ang_tribord0[ind_ang_tribord00]
	ind_ang_babord = np.where((ds.phi>=180+dphi)&(ds.phi<=360-dphi))[0]
    
	if raxis==0:
		if isNorthward==1:
			ax.pcolormesh(ds.phi_geo_rad,ds.freq,ds.wave_spectra.T,cmap=cMap)
			if len(ind_ang_tribord)>0:
					r_cont_tribord,theta_tribord=get_contour_box_fromvectors(ds.freq.data[ind_k],
									                 ds.phi_geo_rad.data[ind_ang_tribord])
			if len(ind_ang_babord)>0:
					r_cont_babord,theta_babord=get_contour_box_fromvectors(ds.freq.data[ind_k],
									                 ds.phi_geo_rad.data[ind_ang_babord])
		else:
			ax.pcolormesh(ds.phi_rad,ds.freq,ds.wave_spectra.T,cmap=cMap)
			if len(ind_ang_tribord)>0:
					r_cont_tribord,theta_tribord=get_contour_box_fromvectors(ds.freq.data[ind_k],
												 ds.phi_rad.data[ind_ang_tribord])
			if len(ind_ang_babord)>0:
					r_cont_babord,theta_babord=get_contour_box_fromvectors(ds.freq.data[ind_k],
												 ds.phi_rad.data[ind_ang_babord])
		if len(ind_ang_tribord)>0:
				ax.plot(theta_tribord,r_cont_tribord,'-g',linewidth=2)
		if len(ind_ang_babord)>0:
				ax.plot(theta_babord,r_cont_babord,'-r',linewidth=2)
		ax.set_rmax(ds.freq.max())
        
	else:
		if isNorthward==1:
			ax.pcolormesh(ds.phi_geo_rad,ds.wavelength,ds.wave_spectra.T,cmap=cMap)
			if len(ind_ang_tribord)>0:
				r_cont_tribord,theta_tribord=get_contour_box_fromvectors(ds.wavelength.data[ind_k],
												ds.phi_geo_rad.data[ind_ang_tribord])
			if len(ind_ang_babord)>0:
				r_cont_babord,theta_babord=get_contour_box_fromvectors(ds.wavelength.data[ind_k],
											     ds.phi_geo_rad.data[ind_ang_babord])
		else:
			ax.pcolormesh(ds.phi_rad,ds.wavelength,ds.wave_spectra.T,cmap=cMap)
			if len(ind_ang_tribord)>0:
				r_cont_tribord,theta_tribord=get_contour_box_fromvectors(ds.wavelength.data[ind_k],
											     ds.phi_rad.data[ind_ang_tribord])
			if len(ind_ang_babord)>0:
				r_cont_babord,theta_babord=get_contour_box_fromvectors(ds.wavelength.data[ind_k],
											     ds.phi_rad.data[ind_ang_babord])
		if len(ind_ang_tribord)>0:
				ax.plot(theta_tribord,r_cont_tribord,'-g',linewidth=2)
		if len(ind_ang_babord)>0:
				ax.plot(theta_babord,r_cont_babord,'-r',linewidth=2)
		ax.set_rmax(ds.wavelength.max())
    
	#ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
	ax.set_thetagrids(ang_ticks,labels=ang_label)
	ax.set_thetalim((-1.5*np.pi,0.5*np.pi))
	if (isNorthward==0) & ((ds.phi.data.compute().max() - ds.phi.data.compute().min())<200):
		ax.set_thetalim((ds.phi_rad.min(),ds.phi_rad.max()))
	ax.set_rlabel_position(90.)  # Move radial labels away from plotted line
	ax.grid(True)  



# ==================================================================================
# === 3. Comparison Obs-model ======================================================
# ==================================================================================


# ==================================================================================
# === X. Miscellaneous =============================================================



# -- function GET_MACROCYCLES --------------------------------------
# separates the trajectories of the x° beam into macrocycles 
# (at the 0-360 discontinuity of the azimuth)
# from a dataset containing only 1 incidence angle
# outputs a modified dataset with a new variable : 'macrocycle_label' (time0)

def get_macrocycles(ds):
# --- get entire cycle ----
	ind_end_macro0=np.where((ds.phi[1:]-ds.phi[0:-1])>100)[0]
	inds=np.zeros(len(ind_end_macro0)+2,dtype='int')-1
	inds[-1] = len(ds.phi)-1
	inds[1:-1]=ind_end_macro0
	len_macros=inds[1:]-inds[0:-1]
	macro_nb = np.repeat(np.arange(len(len_macros)),len_macros)

	ds = ds.assign({"macrocycle_label":(("time0"),macro_nb)})
	return ds

# --- read dates from namefile ---
def read_datesfrom_namefile(filename,typeTrack=0):
	nc_file=nc.Dataset(filename)
	if (typeTrack==0)| (typetrack == 1):
		T1=pd.to_datetime(nc_file.time_coverage_start)
		T2=pd.to_datetime(nc_file.time_coverage_end)
	elif typeTrack==2:
		T1=pd.to_datetime(nc_file.first_meas_time)
		T2=pd.to_datetime(nc_file.last_meas_time)
	nc_file.close()
	return T1,T2

# --- Get Tracks between 2 dates ------------------------
def get_tracks_between_dates(T1,T2,typeTrack=0,pathstorage=None,str_to_remove=None,inciangle=8):
	# typeTrack : 0 = nadir L2 CNES, 1 = offnadir ODL, 2 = nadir L3 CMEMS
	if pathstorage==None:
		if typeTrack==0:
			pathstorage='/home/datawork-cersat-public/provider/cnes/satellite/l2/cfosat/swim/swi_l2____/op05/5.1.2/'
		elif typeTrack==1:
			pathstorage='/home/ref-cfosat-public/datasets/swi_l2s/v0.4/'
		elif typeTrack==2:
			pathstorage='/home/ref-cmems-public/tac/wave/WAVE_GLO_WAV_L3_SWH_NRT_OBSERVATIONS_014_001/dataset-wav-alti-l3-swh-rt-global-cfo/'

#	if (T2-T1)>pd.Timedelta("7 days"):
#		print('To avoid too many data loading, your time span should not exceed 7 days')
	if (typeTrack == 0)|(typeTrack==1):
		date_ini_fold=get_datetime_fromDOY(T1.year,T1.day_of_year)
		
	else:	
		date_ini_fold=pd.to_datetime(dt.datetime(T1.year,T1.month,1))
		
	list_tracks = []
	while date_ini_fold<T2:
		yr = date_ini_fold.year
		doy = date_ini_fold.day_of_year
		month = date_ini_fold.month
		if (typeTrack == 0)|(typeTrack==1):
			path_date_ini = os.path.join(pathstorage,str(yr),f'{doy:03d}')
		else:
			path_date_ini = os.path.join(pathstorage,str(yr),f'{month:02d}')
		
		list_files=sorted(os.listdir(path_date_ini))
		
		for itrack in range(len(list_files)):
			name_track = list_files[itrack]
			if typeTrack == 1:
				name_track = name_track+'/'+name_track[:16]+f'{inciangle:02d}'+name_track[18:]+'.nc'
			
			d1 = pd.to_datetime(name_track[22:37])
			d2 = pd.to_datetime(name_track[38:53])
							
			if (name_track[-3:]=='.nc')&(d1<T2) & (d2>T1):
				file_track = os.path.join(path_date_ini,name_track)
				if str_to_remove==None:	
					list_tracks.append(file_track)
				else:
					if str_to_remove not in name_track:
						list_tracks.append(file_track)

		# -- final part of the while loop
		if (typeTrack == 0)|(typeTrack==1):
			date_ini_fold = date_ini_fold + pd.Timedelta(days=1)
		else:
			month=month+1
			date_ini_fold = pd.to_datetime(dt.datetime(yr+(month-1)//12,((month-1)%12)+1,1))
			
	
	return list_tracks
		

# --- Print tracks in a box  ----
def plot_nadir_tracks_in_a_box(list_tracks,lon_min,lon_max,lat_min,lat_max):
	ds_nadir=read_nadir_from_L2_CNES(list_tracks)
	ds_nadir_box=apply_box_selection_to_ds(ds_nadir,lon_min,lon_max,lat_min,lat_max,flag_coords=0)
	fig,ax=plt.subplots(figsize=(20,14),subplot_kw={'projection':ccrs.PlateCarree()})
	swhNorm = mcolors.Normalize(vmin=5, vmax=12)
	jet = cm = plt.get_cmap('jet')
	scalarSWHMap = cmx.ScalarMappable(norm=swhNorm, cmap='viridis')
	ax0, g1 = init_map_cartopy(ax,limx=(lon_min,lon_max),limy=(lat_min,lat_max))
	ax.scatter(ds_nadir_box.lon,ds_nadir_box.lat,c=np.arange(len(ds_nadir_box.time0)),s=100,cmap='jet')
	ax.scatter(ds_nadir_box.lon,ds_nadir_box.lat,c=ds_nadir_box.swh,s=36,cmap='viridis',norm=swhNorm)
	
	
def plot_2D_track_vs_model(ax,trackfile,lon_min,lon_max,lat_min,lat_max,pathWW3=None,isboxes=0,NbHs_min=100):
	ds_nadir=read_nadir_from_L2_CNES(trackfile)
	ds_nadir_box=apply_box_selection_to_ds(ds_nadir,lon_min,lon_max,lat_min,lat_max,flag_coords=0)
	if np.sum(1.*np.isfinite(ds_nadir_box.swh))>NbHs_min:
		T1 = pd.to_datetime(ds_nadir_box.time.data.compute()[0])
		T2 = pd.to_datetime(ds_nadir_box.time.data.compute()[-1])
		DT = T2-T1
		date1 = T1+DT
		swhNorm = mcolors.Normalize(vmin=5, vmax=12)
		scalarSWHMap = cmx.ScalarMappable(norm=swhNorm, cmap='viridis')
		plot_2D_model_Hs(ax,date1,path=pathWW3,limx=(lon_min,lon_max),limy=(lat_min,lat_max),norm=swhNorm)
		ax.scatter(ds_nadir_box.lon,ds_nadir_box.lat,s=60,c='m')
		ax.scatter(ds_nadir_box.lon,ds_nadir_box.lat,c=ds_nadir_box.swh,s=30,norm=swhNorm)
		plt.colorbar(scalarSWHMap,ax=ax)
		ax.set_title(date1.strftime('%Y/%m/%d %H:%M:%S'))
		return True
	else:
		plt.close()
		return False
	
def plot_2D_track_vs_model_boxes(ax,trackfile,lon_min,lon_max,lat_min,lat_max,pathWW3=None,isboxes=0,NbHs_min=100):
	ds_nadir=read_nadir_from_L2_CNES(trackfile)
	ds_boxes=read_boxes_from_L2_CNES(trackfile)
	ds_nadir_box=apply_box_selection_to_ds(ds_nadir,lon_min,lon_max,lat_min,lat_max,flag_coords=0)
	if np.sum(1.*np.isfinite(ds_nadir_box.swh))>NbHs_min:
		T1 = pd.to_datetime(ds_nadir_box.time.data.compute()[0])
		T2 = pd.to_datetime(ds_nadir_box.time.data.compute()[-1])
		DT = T2-T1
		date1 = T1+DT
		swhNorm = mcolors.Normalize(vmin=5, vmax=12)
		scalarSWHMap = cmx.ScalarMappable(norm=swhNorm, cmap='viridis')
		plot_2D_model_Hs(ax,date1,path=pathWW3,limx=(lon_min,lon_max),limy=(lat_min,lat_max),norm=swhNorm)
		ax.scatter(ds_nadir_box.lon,ds_nadir_box.lat,s=60,c='m')
		ax.scatter(ds_nadir_box.lon,ds_nadir_box.lat,c=ds_nadir_box.swh,s=30,norm=swhNorm)
		plt.colorbar(scalarSWHMap,ax=ax)
		strplt = ('-r','-b')
		#waveparams = []
		for isw in range(2):
			ds_boxes_boxW=apply_box_selection_to_ds(ds_boxes.isel(iswest=isw),lon_min,lon_max,lat_min,lat_max,flag_coords=0)
			ax.plot(ds_boxes_boxW.lon_corners,ds_boxes_boxW.lat_corners,strplt[isw],linewidth=2)
			#waveparams.append(ds_boxes_boxW.)
		#waveparams = 
		ax.set_title(date1.strftime('%Y/%m/%d %H:%M:%S'))
		return True
	else:
		plt.close()
		return False
		
def plot_1D_track_vs_model(ax,trackfile,lon_min,lon_max,lat_min,lat_max,pathWW3=None,NbHs_min=100):
	ds_nadir=read_nadir_from_L2_CNES(trackfile)
	ds_nadir_box=apply_box_selection_to_ds(ds_nadir,lon_min,lon_max,lat_min,lat_max,flag_coords=0)
	if np.sum(1.*np.isfinite(ds_nadir_box.swh))>NbHs_min:
		T1 = pd.to_datetime(ds_nadir_box.time.data.compute()[0])
		T2 = pd.to_datetime(ds_nadir_box.time.data.compute()[-1])
		DT = T2-T1
		date1 = T1+DT
		ds=read_Hs_model(date1,path=pathWW3)
		ds1=ds.sel(longitude=slice(lon_min,lon_max),latitude=slice(lat_min,lat_max))
		
		field_interp=interpolate_along_track(ds1.longitude.data,ds1.latitude.data,ds1.hs.data,ds_nadir_box.lon.data,ds_nadir_box.lat.data)

		ax.plot(ds_nadir_box.time,ds_nadir_box.swh,'.',label='1Hz nadir track')
		ax.plot(ds_nadir_box.time,field_interp,'.',label='interpolated model')
		ax.legend()
		ax.set_ylabel('Hs [m]')
		ax.grid(True)
		return True
	else:
		return False
			
def plot_1D_track_boxes(axs,trackfile,lon_min,lon_max,lat_min,lat_max,pathWW3=None):
	ds_boxes=read_boxes_from_L2_CNES(trackfile)
	#strplt = ('-r','-b')
	colors = plt.cm.coolwarm(np.linspace(0,1,8))
	strpltm = ('-*','-^','-o')
	waveparamstr = ('Hs','Peak wavelength','Peak direction')
	#waveparams = []
	for isw in range(2):
		ds_boxes_boxW=apply_box_selection_to_ds(ds_boxes.isel(iswest=isw),lon_min,lon_max,lat_min,lat_max,flag_coords=0)
		for iparam in range(len(ds_boxes.nparam.data)):
			for aa in range(len(ds_boxes.n_beam_l2.data)):
				inci=(aa+3)*2
				ordercol = isw*7+(-1)**isw*aa
				axs[iparam].plot(ds_boxes_boxW.time_box,ds_boxes_boxW.wave_param.isel(nparam=iparam,n_beam_l2=aa),strpltm[aa],linewidth=2,label='inci : '+f'{inci:02d}'+'°',color=colors[ordercol])
			axs[iparam].legend()
			axs[iparam].set_ylabel(waveparamstr[iparam])
			axs[iparam].grid(True)

