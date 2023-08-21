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
from wave_physics_functions import *
from functions_colmap import *
from functions_get_tracks_above_thresh import *

def decode_two_columns_time(coord,name_dim):
	factor = xr.DataArray([1e6, 1], dims=name_dim)
	time_us = (coord * factor).sum(dim=name_dim).assign_attrs(units="microseconds since 2009-01-01 00:00:00 0:00")
	#n_tim since 2009-01-01 00:00:00 0:00
	return time_us	

##### From CNES L2 file, get associated ODL L2S file (ribbons)
def file_L2S_from_file_L2(filetrack,PATH_ODL,v,nbeam,islocal=0):
	if islocal:
		indstart = 6
	else:
		indstart = 12

	YYYY = filetrack.split('/')[indstart]
	DDD = filetrack.split('/')[indstart+1]
	filenc = filetrack.split('/')[-1]
	filefolder = filenc.replace('L2_','L2S').replace('.nc','_'+v)
	filefile = filefolder.replace('L2S__','L2S'+f'{(3+nbeam)*2:02d}')+'.nc'
	file_ODL=PATH_ODL+YYYY+'/'+DDD+'/'+filefolder+'/'+filefile
	return file_ODL

def file_L2P_from_file_L2(file_L2,PATH_L2,PATH_L2P):
	file_L2P = ( file_L2[:len(PATH_L2)+5] + file_L2[len(PATH_L2)+9:] ).replace(PATH_L2,PATH_L2P).replace('____','PBOX')
	return file_L2P
	
def file_L2_from_file_L2P(file_L2P,PATH_L2P,PATH_L2):
	T1 = pd.Timestamp(file_L2P[-34:-19])
	file_L2 = file_L2P.replace('/CFO_','/'+f'{T1.day_of_year:03d}'+'/CFO_').replace(PATH_L2P,PATH_L2).replace('PBOX','____')

	return file_L2

def get_files_for_1_year(yea,nbeam=2,str_to_remove=['OPER','TEST'],PATH_L2S='/home/ref-cfosat-public/datasets/swi_l2s/v1.0/',\
			PATH_L2 = '/home/datawork-cersat-public/provider/cnes/satellite/l2/cfosat/swim/swi_l2____/op05/5.1.2/',\
			PATH_L2P = '/home1/datawork/mdecarlo/CFOSAT_L2P/swim_l2p_box_nrt/'):
	if yea==2020:
		T1=pd.to_datetime('2020-01-01')
		T2=pd.to_datetime('2021-01-01')
		# --- read files L2 and L2S between T1 and T2 ----------------------------------------
		list_L20,_ = get_tracks_between_dates(T1,T2,str_to_remove=str_to_remove,inciangle=(3+nbeam)*2,verbose=False)
		list_L2P0 = [file_L2P_from_file_L2(f,PATH_L2,PATH_L2P) for f in list_L20]
		
		list_L2S0,_ = get_tracks_between_dates(T1,T2,typeTrack=1,str_to_remove=str_to_remove,inciangle=(3+nbeam)*2)
	
	elif yea==2021:
		T1=pd.to_datetime('2021-01-01')
		T2=pd.to_datetime('2022-01-01')
		
		# -- get tracks between dates L2 v 5.1.2   -------------------
		list_L200,last_date_append = get_tracks_between_dates(T1,T2,str_to_remove=str_to_remove,inciangle=(3+nbeam)*2,verbose=False)
		# -- get tracks between dates L2 v 5.2.0   -------------------
		pathstorage='/home/datawork-cersat-public/provider/cnes/satellite/l2/cfosat/swim/swi_l2____/op05/5.2.0/'
		list_L201,_ = get_tracks_between_dates(last_date_append,T2,pathstorage=pathstorage,str_to_remove=str_to_remove,inciangle=(3+nbeam)*2)
		list_L2P00 = [file_L2P_from_file_L2(f,PATH_L2,PATH_L2P) for f in list_L200]
		list_L2P01 = [file_L2P_from_file_L2(f,pathstorage,PATH_L2P) for f in list_L201]

		list_L20 = list_L200+list_L201
		list_L2P0 = list_L2P00+list_L2P01

		list_L2S0,_ = get_tracks_between_dates(T1,T2,typeTrack=1,str_to_remove=str_to_remove,inciangle=(3+nbeam)*2)

	# --- change tracks L2 to L2S naming ------------------------------
	list_L2_as_L2S = [file_L2S_from_file_L2(fl,PATH_L2S,'1.0.0',nbeam,islocal=0) for fl in list_L20]
	# --- get tracks L2(as L2S) in L2S list ------------------------------
	list_L2S_intersec = [f for f in list_L2_as_L2S if f in list_L2S0]

	indfile = [list_L2_as_L2S.index(x) for x in list_L2S_intersec]
	list_L2_intersec = [list_L20[indi] for indi in indfile]
	list_L2P = [list_L2P0[indi] for indi in indfile]
	print('nb files intersection L2 and L2S :', len(list_L2_intersec),' over ',len(list_L20))
	
	return list_L2_intersec,list_L2S_intersec,list_L2P

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

def preprocess_offnadir_work_quiet(ds0):
	ds = ds0[["phi","lat","lon","k","time", "wave_spectra","dk"]]
	ds = ds.assign({"l2s_angle":((ds.l2s_angle))})
	ds = ds.assign({"wavelength":("k",(2*np.pi/ds.k.data))})
	ds = ds.assign({"wave_spectra_kth_hs":((ds.wave_spectra*ds.k**-1))})
	ds = ds.assign({"wave_spectra_kth": ((ds.wave_spectra*ds.k**-2))})
	omega=np.sqrt(ds["k"]*9.81)
	freq0=omega/(2*np.pi)
	ds=ds.assign({"freq":(freq0)})
	ds=ds.assign({"df":(("k"),np.gradient(ds['freq'].compute().data))})
	ds=ds.swap_dims({'time':'time0'})
	ds=ds.reset_coords('time')
	ds = ds.rename_vars({'time':'time_per_angle'})
	ds = ds.rename_vars({"k":"k_vector"})
	ds = ds.rename_vars({"phi":"phi_vector"})
	ds = ds.swap_dims({"k":"nk"})
	return ds
	
def read_l2s_offnadir_files_work_quiet(offnadir_files):
	ds_all_inci0=xr.open_mfdataset(offnadir_files,concat_dim="l2s_angle", combine="nested",decode_coords=False,autoclose=True,
		data_vars='minimal',coords="all",compat='override',preprocess=preprocess_offnadir_work_quiet)
	return ds_all_inci0
	
def preprocess_offnadir(ds0):
	ds = ds0[["phi","phi_geo","lat","lon","k","time","near_lon","near_lat","far_lon","far_lat",
	      "wave_spectra","dk","incidence","l1a_availability_flag"]]
	ds = ds.assign({"l2s_angle":((ds.l2s_angle))})
	ds = ds.assign({"wavelength":("k",(2*np.pi/ds.k.data))})
	ds = ds.assign({"wave_spectra_kth_hs":((ds.wave_spectra*ds.k**-1))})
	omega=np.sqrt(ds["k"]*9.81)
	freq0=omega/(2*np.pi)
	ds=ds.assign({"freq":(freq0)})
	ds=ds.assign({"df":(("k"),np.gradient(ds['freq'].compute().data))})
	ds=ds.swap_dims({'time':'time0'})
	ds=ds.reset_coords('time')
	ds = ds.rename_vars({'time':'time_per_angle'})
	ds = ds.rename_vars({"phi":"phi_vector"})
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
	ds_all_inci0=xr.open_mfdataset(offnadir_files,concat_dim="l2s_angle", combine="nested",decode_coords=False,autoclose=True,
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
	varnames = [n for n in ds0.variables]
	wantedvars = ["lat_l2anad_0","lon_l2anad_0","nadir_swh_native","nadir_wind_native",
		"nadir_swh_native_validity","nadir_swh_nsec","nadir_swh_nsec_std","wf_surf_flag",
		"wf_surf_ocean_index_nsec","flag_dep","time_nadir_native"]
	if all(var in varnames for var in wantedvars):
		ds = ds0[["lat_l2anad_0","lon_l2anad_0","nadir_swh_native","nadir_wind_native",
			"nadir_swh_native_validity","nadir_swh_nsec","nadir_swh_nsec_std","wf_surf_flag",
			"wf_surf_ocean_index_nsec","flag_dep"]]
		ds = ds.assign(time_nadir_native=decode_two_columns_time(ds0.time_nadir_native,"n_time"))
		ds=ds.rename_vars({"time_nadir_native":"time","lat_l2anad_0":"lat","lon_l2anad_0":"lon",
			"nadir_swh_native":"swh_native","nadir_swh_nsec":"swh","nadir_swh_nsec_std":"swh_std",
			"nadir_swh_native_validity":"swh_flag","nadir_wind_native":"wind"})
		ds = ds.assign(swh_filtered = ((ds0.wf_surf_flag==0)&(ds0.flag_dep<1)&(ds0.flag_availability==1)))
		#ds = ds.isel(n_mcycles=ind)
		ds=ds.swap_dims({'n_mcycles':'time0'})
		ds=ds.assign_attrs({"variables_ok":(True)})
		return xr.decode_cf(ds)
	else:
		print('missing variables for : ',ds0.encoding["source"])
		ds=ds0.assign_attrs({"variables_ok":(False)})
		return ds
	
def preprocess_nadir_native_v0(ds0): # 5 Hz
	ds = ds0
	time_alti1 = np.datetime64('2009-01-01') + ds0["time_nadir_native"].data[:,0]* np.timedelta64(1,'s') + ds0["time_nadir_native"].data[:,1]* np.timedelta64(1,'us')
	ds=ds.assign({"time_nadir_native":(("n_mcycles"),time_alti1)})
	ds=ds.rename_vars({"time_nadir_native":"time","lat_l2anad_0":"lat","lon_l2anad_0":"lon",
		"nadir_swh_native":"swh","nadir_swh_native_validity":"swh_flag","nadir_wind_native":"wind"})
	ds=ds.swap_dims({'n_mcycles':'time0'})
	return ds 
	
def preprocess_nadir_1Hz(ds0): # 1 Hz
	varnames = [n for n in ds0.variables]
	wantedvars = ["lat_nadir_1Hz","lon_nadir_1Hz","nadir_swh_1Hz","nadir_wind_1Hz","flag_valid_wind_1Hz",
		"nadir_swh_1Hz_std","nadir_swh_1Hz_used_native","flag_valid_swh_1Hz","time_nadir_1Hz"]
	if all(var in varnames for var in wantedvars):
		ds = ds0[["lat_nadir_1Hz","lon_nadir_1Hz","nadir_swh_1Hz","nadir_wind_1Hz","flag_valid_wind_1Hz",
			"nadir_swh_1Hz_std","nadir_swh_1Hz_used_native","flag_valid_swh_1Hz"]]
		ds = ds.assign(time_nadir_1Hz=decode_two_columns_time(ds0.time_nadir_1Hz,"n_tim"))
		ds=ds.rename_vars({"time_nadir_1Hz":"time","lat_nadir_1Hz":"lat","lon_nadir_1Hz":"lon",
			 "nadir_swh_1Hz":"swh","flag_valid_swh_1Hz":"swh_flag","nadir_wind_1Hz":"wind",
			 "flag_valid_wind_1Hz":"wind_flag","nadir_swh_1Hz_std":"swh_std",})
		ds = ds.assign(swh_filtered=CFOSAT_smooth_and_filter_track(ds["swh"],kernel_scatter=7,kernel_median=17,scatter_threshold=1))
		
		ds=ds.swap_dims({'n_nad_1Hz':'time0'})
		ds=ds.assign_attrs({"variables_ok":(True)})
		return xr.decode_cf(ds) 
	else:
		print('missing variables for : ',ds0.encoding["source"])
		ds=ds0.assign_attrs({"variables_ok":(False)})
		return ds

def preprocess_nadir_boxes(ds0): # boxes
	varnames = [n for n in ds0.variables]
	wantedvars = ["lat_spec_l2","lon_spec_l2","wave_param","wave_param_part","nadir_swh_box","nadir_swh_box_std","swh_ecmwf","time_l2",
			"min_lat_l2","max_lat_l2","min_lon_l2","max_lon_l2","flag_valid_swh_box"]
	if all(var in varnames for var in wantedvars):
		if ds0.n_beam_l2.size >=3:
			ds = ds0[["lat_spec_l2","lon_spec_l2","wave_param","wave_param_part","nadir_swh_box","nadir_swh_box_std","swh_ecmwf","flag_valid_swh_box"]]
			ds = ds.assign(time_box=decode_two_columns_time(ds0.time_l2,"n_tim"))
			# defines corners just in case (for plots)
			da_concat=xr.concat([ds0["min_lat_l2"],ds0["max_lat_l2"],ds0["max_lat_l2"],ds0["min_lat_l2"],ds0["min_lat_l2"]],'n_corners')
			ds=ds.assign(lat_corners=da_concat)
			da_concat=xr.concat([ds0["min_lon_l2"],ds0["min_lon_l2"],ds0["max_lon_l2"],ds0["max_lon_l2"],ds0["min_lon_l2"]],'n_corners')
			ds=ds.assign(lon_corners=da_concat)
			ds=ds.rename_vars({"lat_spec_l2":"lat","lon_spec_l2":"lon",
				 "nadir_swh_box":"swh","nadir_swh_box_std":"swh_std","flag_valid_swh_box":"swh_flag",})
			ds = ds.assign(swh_filtered=CFOSAT_smooth_and_filter_track(ds["swh"],kernel_scatter=7,kernel_median=17,scatter_threshold=1))
			ds=ds.swap_dims({'n_posneg':'isBabord'})
			ds=ds.swap_dims({'n_box':'time0'})	 
			ds=ds.assign_attrs({"variables_ok":(True)})
			return xr.decode_cf(ds) 
		else:
			print(os.path.basename(ds0.encoding["source"]),' size nbeaml2 = ',ds0.n_beam_l2.size)
			ds=ds0.assign_attrs({"variables_ok":(False)})
			return ds
	else:
		missing_v = [var for var in wantedvars if var not in varnames]
		print('missing variables :',missing_v)
		print(' for : ',os.path.basename(ds0.encoding["source"]))
		ds=ds0.assign_attrs({"variables_ok":(False)})
		return ds

def preprocess_nadir_CMEMS(ds0): # 1 Hz
	ds = ds0[["latitude","longitude","time","VAVH","VAVH_UNFILTERED"]]
	ds=ds.rename_vars({"latitude":"lat","longitude":"lon",
		"VAVH":"swh"})
	ds=ds.swap_dims({'time':'time0'})	
	return ds

def read_nadir_from_L2_CNES(nadir_files,flag_1Hz=1,flag_boxes=0):
# -- flag_1Hz == 1 : read 1Hz data || flag_1Hz == 0 : read 5 Hz data # n_nad_1Hz # n_mcycles
# -- flag boxes == 1 : read nadir swh and std from boxes !
	if flag_boxes == 0:
		if flag_1Hz==1:
			ds_nadir=xr.open_mfdataset(nadir_files,concat_dim="time0", combine="nested",decode_times=False,autoclose=True,
				decode_coords=False,data_vars='minimal',coords="minimal",compat='override',
				preprocess=preprocess_nadir_1Hz)
		else:
			ds_nadir=xr.open_mfdataset(nadir_files,concat_dim="time0", combine="nested",decode_times=False,autoclose=True,
				decode_coords=False,data_vars='minimal',coords="minimal",compat='override',
				preprocess=preprocess_nadir_native)
	else:
		ds_nadir=xr.open_mfdataset(nadir_files,concat_dim="time0", combine="nested",decode_times=False,autoclose=True,
			decode_coords=False,data_vars='minimal',coords="minimal",compat='override',
			preprocess=preprocess_nadir_boxes)
	return ds_nadir

def read_nadir_from_L3_CMEMS(nadir_files):
	ds_nadir=xr.open_mfdataset(nadir_files,concat_dim="time0", combine="nested",
		decode_coords=False,data_vars='minimal',coords="minimal",compat='override',autoclose=True,
		preprocess=preprocess_nadir_CMEMS)
	return ds_nadir


## HERE for READ 
# -- 1.2. read Box data
def preprocess_boxes_work_quiet(ds0):
	#ds = ds0[["min_lat_l2","max_lat_l2","min_lon_l2","max_lon_l2",
	ds = ds0[["lat_nadir_l2","lon_nadir_l2", "lat_spec_l2", "lon_spec_l2", "wave_param", "k_spectra" ,"phi_vector", "swh_ecmwf", "nadir_swh_box", "nadir_swh_box_std", "nadir_swh_box_used_native", "flag_valid_swh_box"]]
	ds = ds.assign(time_box=decode_two_columns_time(ds0.time_l2,"n_tim"))

	# Prepare transformations (jacobian and slope to waves)
	omega=np.sqrt(ds["k_spectra"]*9.81)
	freq0=omega/(2*np.pi)
	ds=ds.assign({"freq":(freq0)})
	ds=ds.assign(wave_spectra_kth_hs=ds0['pp_mean']*(ds['k_spectra']**-1))
	ds=ds.assign(wave_spectra_kth=ds0['pp_mean']*(ds['k_spectra']**-2))
	ds=ds.assign({"df":(("nk"),np.gradient(ds['freq'].compute().data))})
	ds=ds.assign({"dk":(("nk"),np.gradient(ds['k_spectra'].compute().data))})
	ds = ds.assign({"wavelength":(("nk"),(2*np.pi/ds['k_spectra'].compute().data))})
	
	ds = ds.rename_vars({"k_spectra":"k_vector"})

	da_concat=xr.concat([ds0["min_lat_l2"],ds0["max_lat_l2"],ds0["max_lat_l2"],ds0["min_lat_l2"],ds0["min_lat_l2"]],'n_corners')
	ds=ds.assign(lat_corners=da_concat)
	da_concat=xr.concat([ds0["min_lon_l2"],ds0["min_lon_l2"],ds0["max_lon_l2"],ds0["max_lon_l2"],ds0["min_lon_l2"]],'n_corners')
	ds=ds.assign(lon_corners=da_concat)
	ds=ds.rename_vars({"lat_spec_l2":"lat","lon_spec_l2":"lon",})
	ds=ds.swap_dims({'n_posneg':'isBabord'})
	ds=ds.swap_dims({'n_box':'time0'})
	return xr.decode_cf(ds)
	
def read_boxes_from_L2_CNES_work_quiet(nadir_files):
	ds_nadir=xr.open_mfdataset(nadir_files,concat_dim="time0", combine="nested",decode_times=False,
		decode_coords=False,data_vars='minimal',coords="minimal",compat='override',autoclose=True,
		preprocess=preprocess_boxes_work_quiet)
	return ds_nadir
	
def preprocess_boxes(ds0):
	#ds = ds0[["min_lat_l2","max_lat_l2","min_lon_l2","max_lon_l2",
	ds = ds0[["lat_spec_l2","lon_spec_l2","wave_param","wave_param_part","pp_mean","k_spectra","phi_vector","swh_ecmwf","nadir_swh_box","nadir_swh_box_std"]]
	time_alti1 = np.datetime64('2009-01-01') + ds0["time_l2"].data[:,:,0]* np.timedelta64(1,'s')+ ds0["time_l2"].data[:,:,1]* np.timedelta64(1,'us')
	ds=ds.assign({"time_box":(("n_posneg","n_box"),time_alti1)})
	
	# Prepare transformations (jacobian and slope to waves)
	omega=np.sqrt(ds["k_spectra"]*9.81)
	freq0=omega/(2*np.pi)
	ds=ds.assign({"freq":(freq0)})
	ds=ds.assign(wave_spectra_kth=ds['pp_mean']*(ds['k_spectra']**-2))
	ds=ds.assign(wave_spectra_kth_hs=ds['pp_mean']*(ds['k_spectra']**-1))
	ds=ds.assign(wave_spectra_fth=ds['pp_mean']*(ds['k_spectra']**-1)*(2*np.pi)/(0.5*(9.81/omega)))
	ds=ds.assign({"df":(("nk"),np.gradient(ds['freq'].compute().data))})
	ds=ds.assign({"dk":(("nk"),np.gradient(ds['k_spectra'].compute().data))})
	ds = ds.assign({"wavelength":(("nk"),(2*np.pi/ds['k_spectra'].compute().data))})
	# df=np.nanmean(np.gradient(freq0)) #mean of diff(1/T)
	# domega=np.gradient(omega)
	# dk=np.gradient(WAVENUMBER[0:-4])
	# dtheta=15*np.pi/180 #Hauser et al. 2020
	dphi = 15*np.pi/180
	# -- moments for directionnal spreading ------------------
	a0 = np.cos(ds.phi_vector*np.pi/180)*ds.wave_spectra_fth*dphi
	a1 = a0.sum(dim='n_phi')/(ds.wave_spectra_fth*dphi).sum(dim='n_phi')
	b0 = np.sin(ds.phi_vector*np.pi/180)*ds.wave_spectra_fth*dphi
	b1 = b0.sum(dim='n_phi')/(ds.wave_spectra_fth*dphi).sum(dim='n_phi')
	
	dir_spread_f = (2*(1-((a1)**2+(b1)**2)**(1/2)))**(1/2)
	ds=ds.assign(dir_spread_f=dir_spread_f)
	# -- moments for spectral shape ------------------
	Sf = (ds.wave_spectra_fth*dphi).sum(dim='n_phi')
	m0 = (Sf*ds.df).sum(dim='nk')
	m1 = (Sf*ds.freq*ds.df).sum(dim='nk')
	
	T01 = m0/m1
	parta = (1/m0)*(Sf*np.cos(2*np.pi*ds.freq*T01)*ds.df).sum(dim='nk')
	partb = (1/m0)*(Sf*np.sin(2*np.pi*ds.freq*T01)*ds.df).sum(dim='nk')
	
	spectral_shape = np.abs(parta)**2+ np.abs(partb)**2
	ds=ds.assign(spectral_shape=spectral_shape)
	ds=ds.assign(T01=T01)
	ds=ds.assign(fp=1/T01)
	#ds=ds.assign(dir_spread_fm=dir_spread_f.isel(nk=np.argmin(np.abs(ds.freq-(1/T01)))))
	
	da_concat=xr.concat([ds0["min_lat_l2"],ds0["max_lat_l2"],ds0["max_lat_l2"],ds0["min_lat_l2"],ds0["min_lat_l2"]],'n_corners')
	ds=ds.assign(lat_corners=da_concat)
	da_concat=xr.concat([ds0["min_lon_l2"],ds0["min_lon_l2"],ds0["max_lon_l2"],ds0["max_lon_l2"],ds0["min_lon_l2"]],'n_corners')
	ds=ds.assign(lon_corners=da_concat)
	ds=ds.rename_vars({"lat_spec_l2":"lat","lon_spec_l2":"lon",})
	ds=ds.swap_dims({'n_posneg':'isBabord'})
	ds=ds.swap_dims({'n_box':'time0'})
	return ds 

def read_boxes_from_L2_CNES(nadir_files):
	ds_nadir=xr.open_mfdataset(nadir_files,concat_dim="time0", combine="nested",decode_times=False,
		decode_coords=False,data_vars='minimal',coords="minimal",compat='override',autoclose=True,
		preprocess=preprocess_boxes)
	return ds_nadir

def read_box_data_one_side(filename,isBabord):
	nc_swim = nc.Dataset(filename)
	
	# -- read dimensions 
	WAVENUMBER=nc_swim.variables['k_spectra'][:]
	PHI=nc_swim.variables['phi_vector'][:]
		
	# -- read lon/lat at center box
	lat_midrange = nc_swim.variables['lat_l2'][isBabord,:] #latitude at mid-range (dim0 coordinate of boxes)
	lon_midrange = nc_swim.variables['lon_l2'][isBabord,:]  #longitude at mid-range

	lat_spec=nc_swim.variables['lat_spec_l2'][isBabord,:]#latitude spec
	lon_spec=nc_swim.variables['lon_spec_l2'][isBabord,:] #longitude spec

	# -- read lon/lat box
	###########
	#Box L2A
	###########
	min_lon_l2_box=nc_swim.variables['min_lon_l2'][isBabord,:]#(dim0 coordinate of boxes)
	min_lat_l2_box=nc_swim.variables['min_lat_l2'][isBabord,:]
	max_lon_l2_box=nc_swim.variables['max_lon_l2'][isBabord,:]
	max_lat_l2_box=nc_swim.variables['max_lat_l2'][isBabord,:]
	
	###########
	# Param for each beam
	###########
	wave_param_part_06=nc_swim.variables['wave_param_part'][:,:,isBabord,:,0]#(nparam, npartitions, n_posneg, n_box, n_beam_l2)
	wave_param_part_08=nc_swim.variables['wave_param_part'][:,:,isBabord,:,1]#(nparam, npartitions, n_posneg, n_box, n_beam_l2)
	wave_param_part_10=nc_swim.variables['wave_param_part'][:,:,isBabord,:,2]#(nparam, npartitions, n_posneg, n_box, n_beam_l2)
	
	###########
	# Wave spectra for each beam
	###########
	slope_spec_06deg=nc_swim.variables['pp_mean'][:,:,isBabord,:,0] # wavenumber, phi, dim0,time,beam
	slope_spec_08deg=nc_swim.variables['pp_mean'][:,:,isBabord,:,1]
	slope_spec_10deg=nc_swim.variables['pp_mean'][:,:,isBabord,:,2]
	
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

# ---- comparison between ribbon selection loop and dataset
def get_indices_box_for_ribbon(ds_boxes,ds_l2s):
    new_ds = xr.Dataset(
        {'lon_max': ds_boxes.lon_corners.isel(n_corners=2),
         'lon_min': ds_boxes.lon_corners.isel(n_corners=0),
         'lat_max': ds_boxes.lat_corners.isel(n_corners=2),
         'lat_min': ds_boxes.lat_corners.isel(n_corners=0),
         'lat_ribbon':ds_l2s.rename_dims({'time0':'time_l2s'}).lat,
         'phi_ribbon':ds_l2s.rename_dims({'time0':'time_l2s'}).phi_vector,
         'lon_ribbon':ds_l2s.rename_dims({'time0':'time_l2s'}).lon}
    ).stack(flattened=["isBabord", "time0"])#.reset_index("flattened")

    ind_time_box_2 = xr.where(
        ((
            (new_ds.lon_min<=new_ds.lon_max)&(
            (new_ds.lon_ribbon>=new_ds.lon_min)&(new_ds.lon_ribbon<=new_ds.lon_max))
        )|
        (
            (new_ds.lon_min>new_ds.lon_max)&(
            (new_ds.lon_ribbon>=new_ds.lon_min)|(new_ds.lon_ribbon<=new_ds.lon_max))
        ))&
        (
            (new_ds.lat_ribbon>=new_ds.lat_min)&
            (new_ds.lat_ribbon<=new_ds.lat_max)
        )&
        (
            (new_ds.phi_ribbon>=new_ds.isBabord*180)&
            (new_ds.phi_ribbon<=new_ds.isBabord*180+180)
        ),
             new_ds.time0,-1).max(dim='flattened')
    
    ind_west_box_2 = xr.where(
        ((
            (new_ds.lon_min<=new_ds.lon_max)&(
            (new_ds.lon_ribbon>=new_ds.lon_min)&(new_ds.lon_ribbon<=new_ds.lon_max))
        )|
        (
            (new_ds.lon_min>new_ds.lon_max)&(
            (new_ds.lon_ribbon>=new_ds.lon_min)|(new_ds.lon_ribbon<=new_ds.lon_max))
        ))&
        (
            (new_ds.lat_ribbon>=new_ds.lat_min)&
            (new_ds.lat_ribbon<=new_ds.lat_max)
        )&
        (
            (new_ds.phi_ribbon>=new_ds.isBabord*180)&
            (new_ds.phi_ribbon<=new_ds.isBabord*180+180)
        ),
             new_ds.isBabord,-1).max(dim='flattened')

    return ind_time_box_2,ind_west_box_2,new_ds

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
def get_tracks_between_dates(T1,T2,typeTrack=0,pathstorage=None,str_to_remove=None,inciangle=8,verbose=False):
	# typeTrack : 0 = nadir L2 CNES, 1 = offnadir ODL, 2 = nadir L3 CMEMS
	if pathstorage==None:
		if typeTrack==0:
			pathstorage='/home/datawork-cersat-public/provider/cnes/satellite/l2/cfosat/swim/swi_l2____/op05/5.1.2/'
		elif typeTrack==1:
			pathstorage='/home/ref-cfosat-public/datasets/swi_l2s/v1.0/'
		elif typeTrack==2:
			pathstorage='/home/ref-cmems-public/tac/wave/WAVE_GLO_WAV_L3_SWH_NRT_OBSERVATIONS_014_001/dataset-wav-alti-l3-swh-rt-global-cfo/'

#	if (T2-T1)>pd.Timedelta("7 days"):
#		print('To avoid too many data loading, your time span should not exceed 7 days')
	if (typeTrack == 0)|(typeTrack==1):
		date_ini_fold=get_datetime_fromDOY(T1.year,T1.day_of_year)
		
	else:	
		date_ini_fold=pd.to_datetime(dt.datetime(T1.year,T1.month,1))
		
	list_tracks = []
	exceptss = []
	exceptss_counts = []
	count = 0
	count_last = 0
	date_last_appended = date_ini_fold
	while date_ini_fold<T2:
		yr = date_ini_fold.year
		doy = date_ini_fold.day_of_year
		month = date_ini_fold.month
		if (typeTrack == 0)|(typeTrack==1):
			path_date_ini = os.path.join(pathstorage,str(yr),f'{doy:03d}')
		else:
			path_date_ini = os.path.join(pathstorage,str(yr),f'{month:02d}')
		try:
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
						date_last_appended = date_ini_fold
						count_last = count
					elif np.size(str_to_remove)==1:
						if str_to_remove not in name_track:
							list_tracks.append(file_track)
							date_last_appended = date_ini_fold
							count_last = count
					else:
						if all(st1 not in name_track for st1 in str_to_remove):
							list_tracks.append(file_track)
							date_last_appended = date_ini_fold
							count_last = count
		except Exception as inst:
			exceptss.append(str(inst)+'  '+str(date_ini_fold))
			exceptss_counts.append(count)
			
		# -- final part of the while loop
		if (typeTrack == 0)|(typeTrack==1):
			date_ini_fold = date_ini_fold + pd.Timedelta(days=1)
		else:
			month=month+1
			date_ini_fold = pd.to_datetime(dt.datetime(yr+(month-1)//12,((month-1)%12)+1,1))
		count = count +1
	
	if verbose==True:
		A=np.array(exceptss_counts)
		for ia,_ in enumerate(A[A<count_last]):
			print(exceptss[ia])
	
	return list_tracks,date_last_appended
		

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
			ds_boxes_boxW=apply_box_selection_to_ds(ds_boxes.isel(isBabord=isw),lon_min,lon_max,lat_min,lat_max,flag_coords=0)
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
		ds_boxes_boxW=apply_box_selection_to_ds(ds_boxes.isel(isBabord=isw),lon_min,lon_max,lat_min,lat_max,flag_coords=0)
		for iparam in range(len(ds_boxes.nparam.data)):
			for aa in range(len(ds_boxes.n_beam_l2.data)):
				inci=(aa+3)*2
				ordercol = isw*7+(-1)**isw*aa
				axs[iparam].plot(ds_boxes_boxW.time_box,ds_boxes_boxW.wave_param.isel(nparam=iparam,n_beam_l2=aa),strpltm[aa],linewidth=2,label='inci : '+f'{inci:02d}'+'°',color=colors[ordercol])
			axs[iparam].legend()
			axs[iparam].set_ylabel(waveparamstr[iparam])
			axs[iparam].grid(True)

