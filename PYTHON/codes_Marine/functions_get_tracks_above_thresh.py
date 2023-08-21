#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================================================
# === 0. Import Packages ===========================================================
# ==================================================================================
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.text as mtext
import matplotlib.colors as mcolors
import matplotlib as mpl
import matplotlib.cm as cmx

import glob
import os
import sys
import time
import pickle
#import nctoolkit as nctk
import scipy.interpolate as spi
import scipy.signal as scs

import numpy as np
import netCDF4 as nc
from netCDF4 import stringtochar
import datetime as dt
from datetime import datetime
import argparse
#from IPython import display

import multiprocessing as mp
from functions_colmap import *
from Misc_functions import *

##################################################################################
## CONTENTS :
# --------------------------------------------------------------------------------
# --- 1. Obtain list tracks ------------------------------------------------------
# --------------------------------------------------------------------------------
# - get_list_tracks(PATH_sat) ----------------------------------------------------
#   --- function to get the list of the tracks from an arborescence PATH_sat/YEAR/JULDAY/tracks.nc

# - get_list_tracks_nofold(PATH,SAT) ---------------------------------------------
#   --- function to get the list of the tracks from an arborescence PATH/SAT/tracks.nc

# --------------------------------------------------------------------------------
# --- 2. Reading function: -------------------------------------------------------
# --------------------------------------------------------------------------------
# - read_CCI_data(filetrack,is_rejectflag=0,is_swh_rms=0) ------------------------
#   --- Function that read CCI NC file 

# - read_CCI_data_40hz_v1(filetrack_v1) ------------------------------------------
#   --- Function that read sgdr CCI NC file

# - read_CCI_data_40hz_v2(filetrack_v2) ------------------------------------------
#   --- Function that read whales CCI NC file
 
# - read_CCI_data_model(filetrack) -----------------------------------------------
#   --- Function that read model from CCI NC file

# - read_CFOSAT_nadir_data(filename) ---------------------------------------------
#   --- Function that read nadir data from CFOSAT 1Hz

# - read_saved_track(filetrack) --------------------------------------------------
#   --- Function that read NC file saved (with or without ocean name) - 
#       OLD - OBSOLETE since ocean labelling is now done with geopandas

# - read_mod_hs(PATH_WW3,d1) -----------------------------------------------------
#   --- Function that read Hs model from model file

# ---------------------------------------------------------------------------------------------------------------------
# --- 3. process tracks (smoothing and thresholding): -----------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# - CFOSAT_smooth_and_filter_track(swh_alti,kernel_scatter=7,kernel_median=17,scatter_threshold=1) --------------------
#   --- function to remove spikes due to land

# - is_track_above_thresh(Path_storage,f2,time_alti,lat_alti,lon_alti,swh_alti,hs_thresh=10,nb_spikes=50,min_len=5) ---
#   --- if track is above threshold with given conditions => copy it in path_storage,f2

# - is_track_above_thresh_raw(time_alti,lat_alti,lon_alti,swh_alti,hs_thresh=10) --------------------------------------
#   --- raw because bool istrack_above is True is 1 point > thresh 
#	=> returns also, len_above and nb_spikes, along with info on Hs_max

# - get_v1_on_v2_results(i,filez) -------------------------------------------------------------------------------------
#   --- function to get Hs value for v1 on the position of the maximum Hs for v2  
# 	'filez' contains: the file name of v1 and the time where max is found in v2

# - get_model_on_v2_results(i,filez) ----------------------------------------------------------------------------------
#   --- function to get modeled Hs value (interpolated along the track) on the position of the maximum Hs for v2 
#       'filez' contains: the file name of track CCI and the time where max is found in v2

# --------------------------------------------------------------------------------
# --- 4. Histograms --------------------------------------------------------------
# --------------------------------------------------------------------------------
# - get_histo_track(i,filetrack,Hs_bins) -----------------------------------------
#   --- function to get an histogram for one track's values of Hs along with the year

# - get_histo_track_CCI_v1_v2_40hz(i,filetracks,Hs_bins) -------------------------
#   --- function to get 40 Hz histograms v1 and v2 (and combinations of valid/invalid data) 
#	for one track's values of Hs along with the year 
#	'filetracks' contains both names for v1 and v2

# - get_histo_track_CCI_v1_v2_1hz(i,filetracks,Hs_bins,isrms) --------------------
#   --- function to get 1 Hz histograms v1 and v2 (and combinations of valid/invalid data) 
#	for one track's values of Hs along with the year  
# 	'filetracks' contains both names for v1 and v2

# - get_histo_rejectionflag_track_1hz(i,filetrack,Hs_bins) -----------------------
#   --- get the rejection flags where v1 is valid  (=0) and v2 is not valid (=1), 
#	i.e. reasons of rejection


# --------------------------------------------------------------------------------
# --- 5. Main functions to investigate tracks ------------------------------------
# --------------------------------------------------------------------------------
# - investigate_track(i, filetrack,Path_storage) ---------------------------------
#   --- Main function to read CCI data and apply is_track_above_thresh function

# - investigate_track_raw(i, filetrack,Path_storage) -----------------------------
#   --- Main function to read CCI data and apply is_track_above_thresh_raw function 
#	/!\ Renamed from investigate_track_all

# - investigate_track_all_CFOSAT(i, filetrack) -----------------------------------
#   --- Main function to read CFOSAT nadir data, apply smoothing (spikes removal) and apply is_track_above_thresh_raw function 
#	/!\ Renamed from investigate_track_all

# - investigate_track_raw_2versions(i, filetracks) -------------------------------
#   --- Main function to read CCI data for 2 versions and apply is_track_above_thresh_raw function 
#	to both versions and then get the other version value at the max point 
# 	'filetracks' contains both names for v1 and v2 /!\ Renamed from investigate_track_all_2versions

# --------------------------------------------------------------------------------
# --- 6. plots -------------------------------------------------------------------
# --------------------------------------------------------------------------------
# - plot_high_tracks(i,filetrack,Path_images) ------------------------------------
#   --- function to plot the Hs time series for any track (if ocean_name is defined : color scatter the oceans)

# - plot_very_high_tracks_maps(i,filetrack,Path_images) --------------------------
#   --- function to plot the map of modeled Hs, and the Hs of a track saved, IF hs_max>17.5 m 





##################################################################################
## FUNCTIONS :
# --------------------------------------------------------------------------------
# --- 1. Obtain list tracks ------------------------------------------------------
# --------------------------------------------------------------------------------
def get_list_tracks(PATH_sat):
	#print(PATH_sat)
	list_year=sorted(os.listdir(PATH_sat))
	#print(list_year)
	if 'all_files.tar' in list_year:
		list_year.remove('all_files.tar')
	if 'all_files.tar.bz2' in list_year:
                list_year.remove('all_files.tar.bz2')
	if 'all.tar.bz2' in list_year:
                list_year.remove('all.tar.bz2')
	list_tracks=[]
	for iyr in range(len(list_year)):
		YR = list_year[iyr]
		PATH_year = os.path.join(PATH_sat,YR)
		list_jdays=sorted(os.listdir(PATH_year))
		for ijday in range(len(list_jdays)):
			JDAY = list_jdays[ijday]
			PATH_jday = os.path.join(PATH_year,JDAY)
			#print(PATH_jday)
			list_day_tracks = sorted(os.listdir(PATH_jday))
			for itrack in range(len(list_day_tracks)):
				name_track = list_day_tracks[itrack]
				if 'TEST' not in name_track:
					file_track = os.path.join(PATH_jday,name_track)
					#print(file_track)
					list_tracks.append(file_track) #got the list of all files
	
	return list_tracks
	
def  get_list_tracks_nofold(PATH,SAT):
	PATH_sat = os.path.join(PATH,SAT)
	list_files=sorted(os.listdir(PATH_sat))
	list_tracks = []
	for itrack in range(len(list_files)):
		if list_files[itrack][-3:]=='.nc':
			file_track = os.path.join(PATH_sat,list_files[itrack])
			list_tracks.append(file_track) #got the list of all files

	return list_tracks

# --------------------------------------------------------------------------------
# --- 2. Reading function: -------------------------------------------------------
# --------------------------------------------------------------------------------
## -- Function that reads CCI NC file 
# ---------------------------------------------------
def read_CCI_data(filetrack,is_rejectflag=0,is_swh_rms=0):
	nc_CCI = nc.Dataset(filetrack)
	lon0= nc_CCI.variables['lon'][:]
	lat0=nc_CCI.variables['lat'][:]
	time0=nc_CCI.variables['time'][:]
	# in version 2.0.6 the variable swh_denoised is the same as swh_adjusted_denoised in previous version 
	swh0=nc_CCI.variables['swh_denoised'][:]
	# Quality flag must be applied !!!! (Good : Quality_flg==3)
	swh_quality_flag = nc_CCI.variables['swh_quality'][:]
	if is_rejectflag:
		swh_rejection_flag0 = nc_CCI.variables['swh_rejection_flags'][:]
	if is_swh_rms:
		swh_rms0 = nc_CCI.variables['swh_rms'][:]

	ind_quality = np.where(np.logical_and(np.logical_not(swh0.mask),swh_quality_flag==3))[0]
	swh = swh0.data
	swh[swh0.mask]=-0.75
	swh[swh_quality_flag!=3]=-0.75
	lat = lat0.data
	lon = lon0.data
	time = time0.data
	if is_rejectflag:
		swh_rejection_flag = swh_rejection_flag0.data
	if is_swh_rms:
		swh_rms = swh_rms0.data

	if is_rejectflag & is_swh_rms:
		return time,lat,lon,swh,swh_quality_flag,swh_rejection_flag,swh_rms
	elif is_rejectflag:
		return time,lat,lon,swh,swh_quality_flag,swh_rejection_flag
	elif is_swh_rms:
		return time,lat,lon,swh,swh_quality_flag,swh_rms
	else:
		return time,lat,lon,swh,swh_quality_flag
		
# --------------------------------------------------------------------------------
## -- Function that reads CCI NC file V1 40Hz : SGDR
# ---------------------------------------------------	
def read_CCI_data_40hz_v1(filetrack_v1):
## -- time_alti,lat_alti,lon_alti,swh_alti,swh_quality_flag=read_CCI_data_40hz_v1(filev1)
	nc_CCI = nc.Dataset(filetrack_v1)
	lon0= nc_CCI.variables['lon_40hz'][:]
	lat0=nc_CCI.variables['lat_40hz'][:]
	time0=nc_CCI.variables['time_40hz'][:]
	swh0=nc_CCI.variables['swh_40hz'][:]
	# Quality flag must be applied !!!! (Good : Quality_flg==0)
	swh_quality_flag = nc_CCI.variables['swh_used_40hz'][:] #(0=yes,1 =no)
	swh = swh0.data
	swh[swh0.mask]=-0.75

	lat = lat0.data
	lon = lon0.data
	time = time0.data

	return time,lat,lon,swh,swh_quality_flag
	
# --------------------------------------------------------------------------------
## -- Function that reads CCI NC file V1 40Hz : WHALES
# ---------------------------------------------------		
def read_CCI_data_40hz_v2(filetrack_v2):
## -- time_alti,lat_alti,lon_alti,swh_alti,swh_quality_flag=read_CCI_data_40hz_v2(filev2)
	nc_CCI = nc.Dataset(filetrack_v2)
	lon0= nc_CCI.variables['lon_20hz'][:]
	lat0=nc_CCI.variables['lat_20hz'][:]
	time0=nc_CCI.variables['time_20hz'][:]
	swh0=nc_CCI.variables['swh_WHALES_20hz'][:]
	# Quality flag must be applied !!!! (Good : Quality_flg==0)
	swh_quality_flag = nc_CCI.variables['swh_WHALES_qual_20hz'][:] #(0=good,1 =bad)
	swh = swh0.data
	swh[swh0.mask]=-0.75

	lat = lat0.data
	lon = lon0.data
	time = time0.data

	return time,lat,lon,swh,swh_quality_flag	


## -----------------------------------------------------------------------------------------------
## Function that reads Hs model from CCI NC file
# ---------------------------------------------------
def read_CCI_data_model(filetrack):
	nc_CCI = nc.Dataset(filetrack)
	lon0= nc_CCI.variables['lon'][:]
	lat0=nc_CCI.variables['lat'][:]
	time0=nc_CCI.variables['time'][:]
	# in version 2.0.6 the variable swh_denoised is the same as swh_adjusted_denoised in previous version 
	swh0=nc_CCI.variables['swh_model'][:]
	# Quality flag must be applied !!!! (Good : Quality_flg==3)
	swh_quality_flag = nc_CCI.variables['swh_quality'][:]
	#print(swh_quality_flag)	
	#print(swh0.mask)
#	print(swh0.data)
	ind_quality = np.where(np.logical_and(np.logical_not(swh0.mask),swh_quality_flag==3))[0]
	swh = swh0.data
	swh[swh0.mask]=0
	swh[swh_quality_flag!=3]=0
	lat = lat0.data
	lon = lon0.data
	time = time0.data
	#swh1 = swh[ind_quality]
	#print(swh)
	return time,lat,lon,swh

## -----------------------------------------------------------------------------------------------
## Function that read nadir data from CFOSAT
# ---------------------------------------------------
def read_CFOSAT_nadir_data(filename):
	# -- 0. open file -----------------------------------------
	# ---------------------------------------------------------
	nc_swim_alti = nc.Dataset(filename)
	list_keys = list(nc_swim_alti.variables.keys())
	one_value_false=0
	# -- 1. latitude/longitude
	# ---------------------------------------------------------
	if 'lat_nadir_1Hz' in list_keys:
		lat_alti0 = nc_swim_alti.variables['lat_nadir_1Hz'][1:]
	else:
		print(filename,' no lat_nadir_1Hz')
		one_value_false=1
		lat_alti0 =[]
	if 'lon_nadir_1Hz' in list_keys:
		lon_alti0 = nc_swim_alti.variables['lon_nadir_1Hz'][1:]
	else:
		print(filename,' no lon_nadir_1Hz')
		lon_alti0 =[]
		one_value_false=1
	# -- 2. time at native (5Hz) ------------------------------
	# ---------------------------------------------------------
	# size:(nt,2) - from 01/01/2009 time[:,0]=seconds + time[:,1]=microseconds
	if 'time_nadir_1Hz' in list_keys:
		time_alti001 = nc_swim_alti.variables['time_nadir_1Hz'][:]
		time_alti0 = dt.datetime(2009,1,1) + time_alti001.data[1:]* dt.timedelta(seconds=1)# + time_alti001.data[1:,1]* dt.timedelta(microseconds=1)
	else:
		print(filename,' no time_nadir_1Hz')
		time_alti0 =[]
		one_value_false=1
	# -- 3. Significant Wave height at 1Hz --------------------
	# ---------------------------------------------------------
	if 'nadir_swh_1Hz' in list_keys:
		swh_alti0 = nc_swim_alti.variables['nadir_swh_1Hz'][1:]
	else:
		print(filename,' no nadir_swh_1Hz')
		swh_alti0 = []
		one_value_false=1
	
	# -- 4. flags for the data availability/validity ----------
	# ---------------------------------------------------------
	if 'flag_valid_swh_1Hz' in list_keys:
		flag_valid0 = nc_swim_alti.variables['flag_valid_swh_1Hz'][1:] #0:valid, 1:invalid
	else:
		print(filename,' no flag_valid_swh_1Hz')
		flag_valid0 = []
		one_value_false=1
	
	# -- 5. Apply filter flags  -------------------------------
	# ---------------------------------------------------------
	if one_value_false==0:
		ind_avail= np.where((flag_valid0==0))[0]
		if len(ind_avail)>0 & one_value_false==0:
			time_alti01 = time_alti001.data[ind_avail]
			time_alti = dt.datetime(2009,1,1) + time_alti01[:]* dt.timedelta(seconds=1) #+ time_alti01[:,1]* dt.timedelta(microseconds=1)
			lat_alti = lat_alti0[ind_avail]
			lon_alti = lon_alti0[ind_avail]
			swh_alti = swh_alti0[ind_avail]
		else:
			time_alti=[]
			lat_alti=[]
			lon_alti=[]
			swh_alti=[]	
	else:
		time_alti=[]
		lat_alti=[]
		lon_alti=[]
		swh_alti=[]
	# -- 6. close file ----------------------------------------
	# ---------------------------------------------------------
	nc_swim_alti.close()
	
	return time_alti0,lat_alti0,lon_alti0,swh_alti0,time_alti,lat_alti,lon_alti,swh_alti
	
## -----------------------------------------------------------------------------------------------
## Function that read NC file saved (with ocean name)
# ---------------------------------------------------
def read_saved_track(filetrack):
	nc_CCI = nc.Dataset(filetrack)
	lon= nc_CCI.variables['lon'][:]
	lat=nc_CCI.variables['lat'][:]
	time0=nc_CCI.variables['time'][:]
	swh=nc_CCI.variables['swh'][:]
#	swh_flag=nc_CCI.variables['swh_flag'][:]
	time1 = time0*dt.timedelta(seconds=1)+dt.datetime(1981,1,1)
	is_ocean_name = 0
	for var in nc_CCI.variables:
		if var=='ocean_label':
			is_ocean_name=1

	if is_ocean_name:
		ocean_name=nc_CCI.variables['ocean_label'][:]
	else:
		ocean_name=np.zeros(np.shape(swh))
	return time1,lat,lon,swh,ocean_name
## -----------------------------------------------------------------------------------------------
## Function that reads Hs model from model NC file for timestep d1
# --------------------------------------------------------------------
def read_mod_hs(PATH_WW3,d1):
	nc_HS=nc.Dataset(PATH_WW3)
	lon=nc_HS.variables['longitude'][:]
	lat=nc_HS.variables['latitude'][:]
	hs=nc_HS.variables['hs'][d1,:,:]
	
	lon_mod, lat_mod = np.meshgrid(lon,lat)
	return lon_mod,lat_mod,hs


# --------------------------------------------------------------------------------
# --- 3. Process tracks (smoothing and thresholding): ----------------------------
# --------------------------------------------------------------------------------
## Function to remove spikes (in CFOSAT) due to land
# --------------------------------------------------------------------
def CFOSAT_smooth_and_filter_track(swh_alti,kernel_scatter=7,kernel_median=17,scatter_threshold=1):
    isorder1 = 0
 
    if isorder1: 
        # - STEP 1 : median filtering (with kernel median) ------
        swh_alti1 = scs.medfilt(swh_alti,kernel_median)
        # - STEP 2 : filter Hs original if big changes compared to the median filtered timeserie ------
        swh_alti20 = filter_big_change_from_ref(swh_alti,swh_alti1,1.2)
	# - STEP 3 : removed values that are set to nan, are locally turned to 0 for the calculus of the scattering
        swh_alti200=np.copy(swh_alti20)
        swh_alti200[np.isnan(swh_alti200)]=0
        # - STEP 4 : Get the scattering metric based on the Hs where the big changes compared to ref are set to 0 ------
        scat2= get_scattering_info(swh_alti200,kernel_scatter,2)
        scat0= get_scattering_info(swh_alti200,kernel_scatter,0)
        scat1= get_scattering_info(swh_alti200,kernel_scatter,1)
                
        scatsfilt = scat0*scat1*scat2
        # - STEP 5 : apply filtering by applying a threshold to the scatter metric on the original 
        #            time series where big changes are set to nans --------------------
        indscat=np.where(scatsfilt>=scatter_threshold)[0]
        
        swh_alti2=np.copy(swh_alti20)
        swh_alti2[indscat]=-0.75#np.nan
        
    else:
        # - STEP 1 : Get the scattering metric based on the Hs original TS ------
        scat2= get_scattering_info(swh_alti,kernel_scatter,2)
        scat0= get_scattering_info(swh_alti,kernel_scatter,0)
        scat1= get_scattering_info(swh_alti,kernel_scatter,1)
                
        scatsfilt = scat0*scat1*scat2
        # - STEP 2 : Filter by applying a theshold to the scatter metric on the original Hs TS -----
        indscat=np.where(scatsfilt>=scatter_threshold)[0]
        
        swh_alti20=np.copy(swh_alti)
        swh_alti20[indscat]=np.nan
	# - STEP 3 : Create a copy where the removed values are set to 0 instead of nan ---------- 
        swh_alti200=np.copy(swh_alti20)
        swh_alti200[np.isnan(swh_alti200)]=0
        
        # - STEP 4 : median filtering of the copy with removed values = 0 -----------------
        swh_alti1=scs.medfilt(swh_alti200,kernel_median)
        # - Filter the already filtered data by comparing to the median filtered TS -----
        swh_alti2 = filter_big_change_from_ref(swh_alti200,swh_alti1,1.2)
        
    return  swh_alti2
    
## -----------------------------------------------------------------------------------------------
## Function that detects if the track is above the threshold
## -----------------------------------------------------------
def is_track_above_thresh(Path_storage,f2,time_alti,lat_alti,lon_alti,swh_alti,hs_thresh=10,nb_spikes=50,min_len=5):     
	# Path_storage,f2,time_alti,lat_alti,lon_alti,swh_alti,hs_thresh=10,nb_spikes=50,min_len=5):
	isabove_thresh = False
# define a binary line : 1 : > hs_thresh, 0 : <hs_thresh
	swh_BN = (swh_alti>hs_thresh)*1
# difference of successive binary values -1 : next = 0 and previous=1 (end of positive) / 1 : next =1 and previous =0 (beginning of positive)
	diff_hs_BN = swh_BN[1:] - swh_BN[0:-1]

	ind_idebut=np.where(diff_hs_BN==1)[0]+1
	ind_ifin = np.where(diff_hs_BN==-1)[0]
	if (len(ind_ifin)==0 and len(ind_idebut)==0):
		isabove_thresh = False
	else:
		if len(ind_idebut)==0:
			ind_debut=0
			ind_fin = ind_ifin
		elif len(ind_ifin)==0:
			ind_fin=len(swh_alti)
			ind_debut = ind_idebut
		else:
			# Deal with the issue arising if the Hs>thresh started before the beginning of the track
			if ind_ifin[0]<ind_idebut[0]:
				ind_debut=np.zeros(len(ind_idebut)+1)
				ind_debut[0]=0
				ind_debut[1:]=ind_idebut
			else:
				ind_debut=ind_idebut
			# Deal with the issue arising if the Hs>thresh ended after the end of the track
			if ind_ifin[-1]<ind_idebut[-1]:
				ind_fin=np.zeros(len(ind_ifin)+1)
				ind_fin[-1]=len(swh_alti)
				ind_fin[0:-1]=ind_ifin
			else:
				ind_fin=ind_ifin

		length_hs = ind_fin - ind_debut
		#print(length_hs)

		if length_hs.max()>min_len: 
			filencnew = os.path.join(Path_storage,f2)
			#if os.path.isfile(filencnew):
			#	os.remove(filencnew)
			dsnew = nc.Dataset(filencnew, 'w', format='NETCDF4')
			xtrack = dsnew.createDimension('t', len(time_alti))
			time_nc = dsnew.createVariable('time', 'f8', ('t',))
			lon_nc = dsnew.createVariable('lon', 'f4', ('t',))
			lat_nc = dsnew.createVariable('lat', 'f4', ('t',))
			swh_nc = dsnew.createVariable('swh', 'f8', ('t',))
			#swh_flag_nc = dsnew.createVariable('swh_flag','B',('t',))
			time_nc[:]=time_alti#(time_alti- dt.datetime(2018,12,31))/dt.timedelta(seconds=1)
			lon_nc[:]=lon_alti
			lat_nc[:]=lat_alti
			swh_nc[:]=swh_alti
			#swh_flag_nc[:]=swh_flag
			dsnew.close()
			isabove_thresh = True
		elif len(length_hs)>nb_spikes:# nb of spikes
			filencnew = os.path.join(Path_storage,f2)
			#if os.path.isfile(filencnew):
			#	os.remove(filencnew)
			dsnew = nc.Dataset(filencnew, 'w', format='NETCDF4')
			xtrack = dsnew.createDimension('t', len(time_alti))
			time_nc = dsnew.createVariable('time', 'f8', ('t',))
			lon_nc = dsnew.createVariable('lon', 'f4', ('t',))
			lat_nc = dsnew.createVariable('lat', 'f4', ('t',))
			swh_nc = dsnew.createVariable('swh', 'f8', ('t',))
			#swh_flag_nc = dsnew.createVariable('swh_flag','B',('t',))
			time_nc[:]=time_alti#(time_alti1- dt.datetime(2018,12,31))/dt.timedelta(seconds=1)
			lon_nc[:]=lon_alti
			lat_nc[:]=lat_alti
			swh_nc[:]=swh_alti
			#swh_flag_nc[:]=swh_flag
			dsnew.close()
			isabove_thresh = True
		else:
			isabove_thresh = False

	return isabove_thresh
	
# ----------------------------------------------------------------------------------------------------------------------------
## Function is track above thresh RAW ("raw" because if the track has 1 point > 10 it is above thresh
# ----------------------------------------------------------------------------------------------------------------------------
def is_track_above_thresh_raw(time_alti,lat_alti,lon_alti,swh_alti,hs_thresh=10):
	isabove_thresh = False
# define a binary line : 1 : > hs_thresh, 0 : <hs_thresh
	swh_BN = (swh_alti>hs_thresh)*1
	
	if sum(swh_BN)==0:
		isabove_thresh=False
		len_above=0
		nb_spikes = 0
	else:
# difference of successive binary values -1 : next = 0 and previous=1 (end of positive) / 1 : next =1 and previous =0 (beginning of positive)
		isabove_thresh=True
		swh_BN_p1 = np.zeros(len(swh_BN)+1)
		swh_BN_0 = np.zeros(len(swh_BN)+1)
		swh_BN_p1[0:-1]=swh_BN[:]
		swh_BN_0[1:]=swh_BN[:]
		# Here we add value equal to 0 before and after the time serie in order to get the info if the whole track is above thresh
		diff_hs_BN = swh_BN_p1 - swh_BN_0

		ind_debut=np.where(diff_hs_BN==1)[0]+1
		ind_fin = np.where(diff_hs_BN==-1)[0]

		length_hs = ind_fin - ind_debut
		#print(length_hs)
		
		len_above=length_hs.max()
		
		nb_spikes=len(length_hs)

	pos = np.argmax(swh_alti)
	Hs_max = swh_alti[pos]
	lon_max= lon_alti[pos]
	lat_max=lat_alti[pos]
	time_max=time_alti[pos]
	
	return isabove_thresh,len_above,nb_spikes,Hs_max,lon_max,lat_max,time_max
# ----------------------------------------------------------------------------------------------------------------------------
## Function to get Hs value for v1 on the position of the maximum Hs for v2. /!\ 'filez' = [filetrack_v1,timemax_v2]
# ----------------------------------------------------------------------------------------------------------------------------
def get_v1_on_v2_results(i,filez):
	filetrack = filez[0]
	timemax_v2 = filez[1]
	print(filetrack)
	time_alti,lat_alti,lon_alti,swh_alti,swh_quality=read_CCI_data(filetrack,is_rejectflag=0)
	if len(swh_alti)>0 and timemax_v2>0:
		ind=np.argmin(abs(time_alti-timemax_v2))
		hs_v1 = swh_alti[ind]
		time_v1 = time_alti[ind]
	else:
		hs_v1=0
		time_v1=0
	
	return i,filetrack,timemax_v2,hs_v1,time_v1

# ----------------------------------------------------------------------------------------------------------------------------
## Function to get Hs value for model (CCI) on the position of the maximum Hs for v2. /!\ 'filez' = [filetrack_v2,timemax_v2]
# ----------------------------------------------------------------------------------------------------------------------------
def get_model_on_v2_results(i,filez):
	filetrack = filez[0]
	timemax_v2 = filez[1]
	print(filetrack)
	time_alti,lat_alti,lon_alti,swh_alti=read_CCI_data_model(filetrack)
	if len(swh_alti)>0 and timemax_v2>0:
		ind=np.argmin(abs(time_alti-timemax_v2))
		hs_model = swh_alti[ind]
		time_model = time_alti[ind]
	else:
		hs_model=0
		time_model=0
	
	return i,filetrack,timemax_v2,hs_model,time_model

# --------------------------------------------------------------------------------
# --- 4. Histograms --------------------------------------------------------------
# --------------------------------------------------------------------------------
## Function to get histo for one track. returns year also
# --------------------------------------------------------------------
def get_histo_track(i,filetrack,Hs_bins):
	time_alti,lat_alti,lon_alti,swh_alti,swh_quality=read_CCI_data(filetrack)
	swh_alti[swh_quality!=3]=np.nan
	time1 = time_alti*dt.timedelta(seconds=1)+dt.datetime(1981,1,1)
	if len(swh_alti)>0:
		Hs_histo = np.histogram(swh_alti,Hs_bins)[0]
	else:
		Hs_histo = np.zeros(len(Hs_bins)-1)
	yy1 = time1[-1].year
	
	return i, Hs_histo, yy1
	
# --------------------------------------------------------------------------------
## Function to get one track's 40Hz histo for v1 and v2 + combinations valid/invalid
# --------------------------------------------------------------------
def get_histo_track_CCI_v1_v2_40hz(i,filetracks,Hs_bins):
	filev1 = filetracks[0]
	filev2 = filetracks[1]
	time_v10,lat_v1,lon_v1,swh_v10,swh_quality_flag_v10 = read_CCI_data_40hz_v1(filev1)
	time_v20,lat_v2,lon_v2,swh_v20,swh_quality_flag_v20 = read_CCI_data_40hz_v2(filev2)
	time_v1_sec = np.int64(np.floor(time_v10))
	time_v1_microsec = np.int64(np.floor((time_v10 - time_v1_sec)*10**6))
	time1 = time_v1_sec*np.timedelta64(1,'s')+time_v1_microsec*np.timedelta64(1,'us')+np.datetime64('2000-01-01')
	
	time_v2_sec = np.int64(np.floor(time_v20))
	time_v2_microsec = np.int64(np.floor((time_v20 - time_v2_sec)*10**6))
	time2 = time_v2_sec*np.timedelta64(1,'s')+time_v2_microsec*np.timedelta64(1,'us')+np.datetime64('2000-01-01')
	
	if len(swh_quality_flag_v10) < len(swh_quality_flag_v20):
		NS_v1 = np.shape(swh_quality_flag_v10)
		# find the first column without NaT value for the smallest timeserie
		iv1=0
		while iv1 < NS_v1[0]:
			# print(time_v10[ik,0]*np.timedelta64(1,'s')+np.datetime64('2000-01-01'))
			if np.isnat(time1[iv1,0]):
				iv1 +=1
			else:
				time_v1_ref=time_v10[iv1,0]
				break

		minind = np.argmin(np.abs(time_v20[:,0]-time_v1_ref))    

		iv2 = minind - np.floor(time_v20[minind,0]-time_v1_ref)
		indv2=np.arange(iv2-iv1,iv2-iv1+NS_v1[0]).astype(int)
		time_v2 = time_v20[indv2,:]
		swh_v2 = swh_v20[indv2,:]
		swh_quality_flag_v2 = swh_quality_flag_v20[indv2,:]
		
		time_v1 = time_v10[:,:]
		swh_v1 = swh_v10[:,:]
		swh_quality_flag_v1 = swh_quality_flag_v10[:,:]
	
	elif len(swh_quality_flag_v10) > len(swh_quality_flag_v20):
		NS_v2 = np.shape(swh_quality_flag_v20)
		# find the first column without NaT value for the smallest timeserie
		iv2=0
		while iv2 < NS_v2[0]:
			# print(time_v20[ik,0]*np.timedelta64(1,'s')+np.datetime64('2000-01-01'))
			if np.isnat(time2[iv2,0]):
				iv2 +=1
			else:
				time_v2_ref=time_v20[iv2,0]
				break
		minind = np.argmin(np.abs(time_v10[:,0]-time_v2_ref))    

		iv1 = minind - np.trunc(time_v10[minind,0]-time_v2_ref)
		indv1=np.arange(iv1-iv2,iv1-iv2+NS_v2[0]).astype(int)
		if indv1[-1]>=np.shape(time_v10)[0]:
			print('minind :',minind)
			print(time_v10[minind,0]-time_v2_ref)
			print(np.floor(time_v10[minind,0]-time_v2_ref))
			print('iv1 :',iv1)
			print(iv1-iv2+NS_v2[0])
			print('shape v2 time :',np.shape(time_v20))
			print('shape v1 time :',np.shape(time_v10))
		time_v1 = time_v10[indv1,:]
		swh_v1 = swh_v10[indv1,:]
		swh_quality_flag_v1 = swh_quality_flag_v10[indv1,:]
		
		time_v2 = time_v20[:,:]
		swh_v2 = swh_v20[:,:]
		swh_quality_flag_v2 = swh_quality_flag_v20[:,:]
	
	else:
		time_v1 = time_v10[:,:]
		swh_v1 = swh_v10[:,:]
		swh_quality_flag_v1 = swh_quality_flag_v10[:,:]
		time_v2 = time_v20[:,:]
		swh_v2 = swh_v20[:,:]
		swh_quality_flag_v2 = swh_quality_flag_v20[:,:]
	
			
	#list_tracks_v1_intersec = [f for f in list_tracks_v2_asv1 if f in list_tracks_v1]
	#indfile = [list_tracks_v2_asv1.index(x) for x in list_tracks_v1_intersec]
	# time1 = time_v1.max()*dt.timedelta(seconds=1)+dt.datetime(2000,1,1)
	if np.logical_and(len(swh_quality_flag_v1)>0,len(swh_quality_flag_v2)>0):
		# --- flag_v1 == 1 && flag_v2 ==1
		ind_v1_1_v2_1 = np.where(np.logical_and(swh_quality_flag_v1==1,swh_quality_flag_v2==1))
		if len(ind_v1_1_v2_1)>0:
			Hs_v1_1_v2_1,xbins,ybins=np.histogram2d(swh_v1[ind_v1_1_v2_1],swh_v2[ind_v1_1_v2_1],(Hs_bins,np.append(Hs_bins,500)))
		else:
			Hs_v1_1_v2_1 = np.zeros((len(Hs_bins)-1,len(Hs_bins)))
		# --- flag_v1 == 0 && flag_v2 ==1
		ind_v1_0_v2_1 = np.where(np.logical_and(swh_quality_flag_v1==0,swh_quality_flag_v2==1))
		if len(ind_v1_0_v2_1)>0:
			Hs_v1_0_v2_1,xbins,ybins=np.histogram2d(swh_v1[ind_v1_0_v2_1],swh_v2[ind_v1_0_v2_1],(Hs_bins,np.append(Hs_bins,500)))
		else:
			Hs_v1_0_v2_1 = np.zeros((len(Hs_bins)-1,len(Hs_bins)))
		# --- flag_v1 == 0 && flag_v2 == 0
		ind_v1_0_v2_0 = np.where(np.logical_and(swh_quality_flag_v1==0,swh_quality_flag_v2==0))
		if len(ind_v1_0_v2_0)>0:
			Hs_v1_0_v2_0,xbins,ybins=np.histogram2d(swh_v1[ind_v1_0_v2_0],swh_v2[ind_v1_0_v2_0],(Hs_bins,np.append(Hs_bins,500)))
		else:
			Hs_v1_0_v2_0 = np.zeros((len(Hs_bins)-1,len(Hs_bins)))
		# --- flag_v1 == 1 && flag_v2 == 0
		ind_v1_1_v2_0 = np.where(np.logical_and(swh_quality_flag_v1==1,swh_quality_flag_v2==0))
		if len(ind_v1_1_v2_0)>0:
			Hs_v1_1_v2_0,xbins,ybins=np.histogram2d(swh_v1[ind_v1_1_v2_0],swh_v2[ind_v1_1_v2_0],(Hs_bins,np.append(Hs_bins,500)))
		else:
			Hs_v1_1_v2_0 = np.zeros((len(Hs_bins)-1,len(Hs_bins)))
	else:
		Hs_v1_1_v2_1 = np.zeros((len(Hs_bins)-1,len(Hs_bins)))
		Hs_v1_0_v2_1 = np.zeros((len(Hs_bins)-1,len(Hs_bins)))
		Hs_v1_0_v2_0 = np.zeros((len(Hs_bins)-1,len(Hs_bins)))
		Hs_v1_1_v2_0 = np.zeros((len(Hs_bins)-1,len(Hs_bins)))
	#print(filev2)
	yy1 = int(filev2[-39:-35])
	
	return i,Hs_v1_1_v2_1, Hs_v1_0_v2_1, Hs_v1_0_v2_0, Hs_v1_1_v2_0, yy1

# --------------------------------------------------------------------------------
## Function to get one track's 1Hz histo for v1 and v2 + combinations valid/invalid
# --------------------------------------------------------------------
def get_histo_track_CCI_v1_v2_1hz(i,filetracks,Hs_bins,isrms):
	rms_bins = np.arange(0,4,0.1)
	filev1 = filetracks[0]
	filev2 = filetracks[1]
	if isrms:
		time_v10,lat_v1,lon_v1,swh_v10,swh_quality_flag_v10,swh_rms_v10 = read_CCI_data(filev1,is_swh_rms=1)
		time_v20,lat_v2,lon_v2,swh_v20,swh_quality_flag_v20,swh_rms_v20 = read_CCI_data(filev2,is_swh_rms=1)
	else:
		time_v10,lat_v1,lon_v1,swh_v10,swh_quality_flag_v10 = read_CCI_data(filev1)
		time_v20,lat_v2,lon_v2,swh_v20,swh_quality_flag_v20 = read_CCI_data(filev2)
	# time V1
	time_v1_sec = np.int64(np.floor(time_v10))
	time_v1_microsec = np.int64(np.floor((time_v10 - time_v1_sec)*10**6))
	time1 = time_v1_sec*np.timedelta64(1,'s')+time_v1_microsec*np.timedelta64(1,'us')+np.datetime64('1981-01-01')
	# time V2
	time_v2_sec = np.int64(np.floor(time_v20))
	time_v2_microsec = np.int64(np.floor((time_v20 - time_v2_sec)*10**6))
	time2 = time_v2_sec*np.timedelta64(1,'s')+time_v2_microsec*np.timedelta64(1,'us')+np.datetime64('1981-01-01')
	
	if len(swh_quality_flag_v10) < len(swh_quality_flag_v20):
		NS_v1 = np.shape(swh_quality_flag_v10)
		# find the first column without NaT value for the smallest timeserie
		iv1=0
		while iv1 < NS_v1:
			# print(time_v10[ik,0]*np.timedelta64(1,'s')+np.datetime64('2000-01-01'))
			if np.isnat(time1[iv1]):
				iv1 +=1
			else:
				time_v1_ref=time_v10[iv1]
				break

		minind = np.argmin(np.abs(time_v20[:]-time_v1_ref))    

		iv2 = minind - np.floor(time_v20[minind]-time_v1_ref)
		indv2=np.arange(iv2-iv1,iv2-iv1+NS_v1[0]).astype(int)
		time_v2 = time_v20[indv2]
		swh_v2 = swh_v20[indv2]
		swh_quality_flag_v2 = swh_quality_flag_v20[indv2]
		if isrms:
			swh_rms_v2 = swh_rms_v20[indv2]
		time_v1 = time_v10[:]
		swh_v1 = swh_v10[:]
		swh_quality_flag_v1 = swh_quality_flag_v10[:]
		if isrms:
			swh_rms_v1 = swh_rms_v10[:]
		
	elif len(swh_quality_flag_v10) > len(swh_quality_flag_v20):
		NS_v2 = np.shape(swh_quality_flag_v20)
		# find the first column without NaT value for the smallest timeserie
		iv2=0
		while iv2 < NS_v2[0]:
			# print(time_v20[ik,0]*np.timedelta64(1,'s')+np.datetime64('2000-01-01'))
			if np.isnat(time2[iv2]):
				iv2 +=1
			else:
				time_v2_ref=time_v20[iv2]
				break
		minind = np.argmin(np.abs(time_v10[:]-time_v2_ref))    

		iv1 = minind - np.trunc(time_v10[minind]-time_v2_ref)
		indv1=np.arange(iv1-iv2,iv1-iv2+NS_v2[0]).astype(int)
		if indv1[-1]>=np.shape(time_v10)[0]:
			print('minind :',minind)
			print(time_v10[minind]-time_v2_ref)
			print(np.floor(time_v10[minind]-time_v2_ref))
			print('iv1 :',iv1)
			print(iv1-iv2+NS_v2[0])
			print('shape v2 time :',np.shape(time_v20))
			print('shape v1 time :',np.shape(time_v10))
		time_v1 = time_v10[indv1]
		swh_v1 = swh_v10[indv1]
		swh_quality_flag_v1 = swh_quality_flag_v10[indv1]
		if isrms:
			swh_rms_v1 = swh_rms_v10[indv1]
		time_v2 = time_v20[:]
		swh_v2 = swh_v20[:]
		swh_quality_flag_v2 = swh_quality_flag_v20[:]
		if isrms:
			swh_rms_v2 = swh_rms_v20[:]
	else:
		time_v1 = time_v10[:]
		swh_v1 = swh_v10[:]
		swh_quality_flag_v1 = swh_quality_flag_v10[:]
		if isrms:
			swh_rms_v1 = swh_rms_v10[:]
		time_v2 = time_v20[:]
		swh_v2 = swh_v20[:]
		swh_quality_flag_v2 = swh_quality_flag_v20[:]
		if isrms:
			swh_rms_v2 = swh_rms_v20[:]
			
	if np.logical_and(len(swh_quality_flag_v1)>0,len(swh_quality_flag_v2)>0):
		# --- flag_v1 == 1 && flag_v2 ==1 -----------------------------------------------------------------
		ind_v1_1_v2_1 = np.where(np.logical_and(swh_quality_flag_v1!=3,swh_quality_flag_v2!=3))
		if len(ind_v1_1_v2_1)>0:
			if isrms:
				Hs_v1_rms_v1_ind_v1_1_v2_1,xbins,ybins=np.histogram2d(swh_v1[ind_v1_1_v2_1],swh_rms_v1[ind_v1_1_v2_1],(Hs_bins,rms_bins))
				Hs_v2_rms_v2_ind_v1_1_v2_1,xbins,ybins=np.histogram2d(swh_v2[ind_v1_1_v2_1],swh_rms_v2[ind_v1_1_v2_1],(Hs_bins,rms_bins))
				
			else:
				Hs_v1_1_v2_1,xbins,ybins=np.histogram2d(swh_v1[ind_v1_1_v2_1],swh_v2[ind_v1_1_v2_1],(Hs_bins,np.append(Hs_bins,500)))
		else:
			if isrms:
				Hs_v1_rms_v1_ind_v1_1_v2_1 = np.zeros((len(Hs_bins)-1,len(rms_bins)-1))
				Hs_v2_rms_v2_ind_v1_1_v2_1 = np.zeros((len(Hs_bins)-1,len(rms_bins)-1))
			else:
				Hs_v1_1_v2_1 = np.zeros((len(Hs_bins)-1,len(Hs_bins)))
		# --- flag_v1 == 0 && flag_v2 ==1 -----------------------------------------------------------------
		ind_v1_0_v2_1 = np.where(np.logical_and(swh_quality_flag_v1==3,swh_quality_flag_v2!=3))
		if len(ind_v1_0_v2_1)>0:
			if isrms:
				Hs_v1_rms_v1_ind_v1_0_v2_1,xbins,ybins=np.histogram2d(swh_v1[ind_v1_0_v2_1],swh_rms_v1[ind_v1_0_v2_1],(Hs_bins,rms_bins))
				Hs_v1_rms_v2_ind_v1_0_v2_1,xbins,ybins=np.histogram2d(swh_v1[ind_v1_0_v2_1],swh_rms_v2[ind_v1_0_v2_1],(Hs_bins,rms_bins))
				
			else:
				Hs_v1_0_v2_1,xbins,ybins=np.histogram2d(swh_v1[ind_v1_0_v2_1],swh_v2[ind_v1_0_v2_1],(Hs_bins,np.append(Hs_bins,500)))
		else:
			if isrms:
				Hs_v1_rms_v1_ind_v1_0_v2_1 = np.zeros((len(Hs_bins)-1,len(rms_bins)-1))
				Hs_v1_rms_v2_ind_v1_0_v2_1 = np.zeros((len(Hs_bins)-1,len(rms_bins)-1))
			else:
				Hs_v1_0_v2_1 = np.zeros((len(Hs_bins)-1,len(Hs_bins)))
		# --- flag_v1 == 0 && flag_v2 == 0 -----------------------------------------------------------------
		ind_v1_0_v2_0 = np.where(np.logical_and(swh_quality_flag_v1==3,swh_quality_flag_v2==3))
		if len(ind_v1_0_v2_0)>0:
			if isrms:
				Hs_v1_rms_v1_ind_v1_0_v2_0,xbins,ybins=np.histogram2d(swh_v1[ind_v1_0_v2_0],swh_rms_v1[ind_v1_0_v2_0],(Hs_bins,rms_bins))
				Hs_v2_rms_v2_ind_v1_0_v2_0,xbins,ybins=np.histogram2d(swh_v2[ind_v1_0_v2_0],swh_rms_v2[ind_v1_0_v2_0],(Hs_bins,rms_bins))
				
			else:
				Hs_v1_0_v2_0,xbins,ybins=np.histogram2d(swh_v1[ind_v1_0_v2_0],swh_v2[ind_v1_0_v2_0],(Hs_bins,np.append(Hs_bins,500)))
		else:
			if isrms:
				Hs_v1_rms_v1_ind_v1_0_v2_0 = np.zeros((len(Hs_bins)-1,len(rms_bins)-1))
				Hs_v2_rms_v2_ind_v1_0_v2_0 = np.zeros((len(Hs_bins)-1,len(rms_bins)-1))
			else:
				Hs_v1_0_v2_0 = np.zeros((len(Hs_bins)-1,len(Hs_bins)))
		# --- flag_v1 == 1 && flag_v2 == 0 -----------------------------------------------------------------
		ind_v1_1_v2_0 = np.where(np.logical_and(swh_quality_flag_v1!=3,swh_quality_flag_v2==3))
		if len(ind_v1_1_v2_0)>0:
			if isrms:
				Hs_v1_rms_v1_ind_v1_1_v2_0,xbins,ybins=np.histogram2d(swh_v1[ind_v1_1_v2_0],swh_rms_v1[ind_v1_1_v2_0],(Hs_bins,rms_bins))
				Hs_v2_rms_v2_ind_v1_1_v2_0,xbins,ybins=np.histogram2d(swh_v2[ind_v1_1_v2_0],swh_rms_v2[ind_v1_1_v2_0],(Hs_bins,rms_bins))
				
			else:
				Hs_v1_1_v2_0,xbins,ybins=np.histogram2d(swh_v1[ind_v1_1_v2_0],swh_v2[ind_v1_1_v2_0],(Hs_bins,np.append(Hs_bins,500)))
		else:
			if isrms:
				Hs_v1_rms_v1_ind_v1_1_v2_0 = np.zeros((len(Hs_bins)-1,len(rms_bins)-1))
				Hs_v2_rms_v2_ind_v1_1_v2_0 = np.zeros((len(Hs_bins)-1,len(rms_bins)-1))
			else:
				Hs_v1_1_v2_0 = np.zeros((len(Hs_bins)-1,len(Hs_bins)))
	else:
		if isrms:
			Hs_v1_rms_v1_ind_v1_1_v2_1 = np.zeros((len(Hs_bins)-1,len(rms_bins)-1))
			Hs_v2_rms_v2_ind_v1_1_v2_1 = np.zeros((len(Hs_bins)-1,len(rms_bins)-1))
			Hs_v1_rms_v1_ind_v1_0_v2_1 = np.zeros((len(Hs_bins)-1,len(rms_bins)-1))
			Hs_v1_rms_v2_ind_v1_0_v2_1 = np.zeros((len(Hs_bins)-1,len(rms_bins)-1))
			Hs_v1_rms_v1_ind_v1_0_v2_0 = np.zeros((len(Hs_bins)-1,len(rms_bins)-1))
			Hs_v2_rms_v2_ind_v1_0_v2_0 = np.zeros((len(Hs_bins)-1,len(rms_bins)-1))
			Hs_v1_rms_v1_ind_v1_1_v2_0 = np.zeros((len(Hs_bins)-1,len(rms_bins)-1))
			Hs_v2_rms_v2_ind_v1_1_v2_0 = np.zeros((len(Hs_bins)-1,len(rms_bins)-1))
		else:
			Hs_v1_1_v2_1 = np.zeros((len(Hs_bins)-1,len(Hs_bins)))
			Hs_v1_0_v2_1 = np.zeros((len(Hs_bins)-1,len(Hs_bins)))
			Hs_v1_0_v2_0 = np.zeros((len(Hs_bins)-1,len(Hs_bins)))
			Hs_v1_1_v2_0 = np.zeros((len(Hs_bins)-1,len(Hs_bins)))
	yy1 = int(filev2[-23:-19])
	if isrms:
		return i, Hs_v1_rms_v1_ind_v1_1_v2_1, Hs_v2_rms_v2_ind_v1_1_v2_1, Hs_v1_rms_v1_ind_v1_0_v2_1, Hs_v1_rms_v2_ind_v1_0_v2_1, Hs_v1_rms_v1_ind_v1_0_v2_0, Hs_v2_rms_v2_ind_v1_0_v2_0, Hs_v1_rms_v1_ind_v1_1_v2_0, Hs_v2_rms_v2_ind_v1_1_v2_0, yy1
	else:
		return i,Hs_v1_1_v2_1, Hs_v1_0_v2_1, Hs_v1_0_v2_0, Hs_v1_1_v2_0, yy1

# --------------------------------------------------------------------------------
## Function to get one track's 1Hz histo of rejection flag for v2 when v1==valid
# --------------------------------------------------------------------
def get_histo_rejectionflag_track_1hz(i,filetracks,Hs_bins):
	filev1 = filetracks[0]
	filev2 = filetracks[1]
	time_v10,lat_v1,lon_v1,swh_v10,swh_quality_flag_v10,rejection_flag_v10 = read_CCI_data(filev1,is_rejectflag=1)
	time_v20,lat_v2,lon_v2,swh_v20,swh_quality_flag_v20,rejection_flag_v20 = read_CCI_data(filev2,is_rejectflag=1)
	# time V1
	time_v1_sec = np.int64(np.floor(time_v10))
	time_v1_microsec = np.int64(np.floor((time_v10 - time_v1_sec)*10**6))
	time1 = time_v1_sec*np.timedelta64(1,'s')+time_v1_microsec*np.timedelta64(1,'us')+np.datetime64('1981-01-01')
	# time V2
	time_v2_sec = np.int64(np.floor(time_v20))
	time_v2_microsec = np.int64(np.floor((time_v20 - time_v2_sec)*10**6))
	time2 = time_v2_sec*np.timedelta64(1,'s')+time_v2_microsec*np.timedelta64(1,'us')+np.datetime64('1981-01-01')
	
	flag_bins=np.array([0,1.5,2.5,6,12,24,48,96,150])
	
	if len(swh_quality_flag_v10) < len(swh_quality_flag_v20):
		NS_v1 = np.shape(swh_quality_flag_v10)
		# find the first column without NaT value for the smallest timeserie
		iv1=0
		while iv1 < NS_v1:
			# print(time_v10[ik,0]*np.timedelta64(1,'s')+np.datetime64('2000-01-01'))
			if np.isnat(time1[iv1]):
				iv1 +=1
			else:
				time_v1_ref=time_v10[iv1]
				break

		minind = np.argmin(np.abs(time_v20[:]-time_v1_ref))    

		iv2 = minind - np.floor(time_v20[minind]-time_v1_ref)
		indv2=np.arange(iv2-iv1,iv2-iv1+NS_v1[0]).astype(int)
		time_v2 = time_v20[indv2]
		swh_v2 = swh_v20[indv2]
		swh_quality_flag_v2 = swh_quality_flag_v20[indv2]
		rejection_flag_v2 = rejection_flag_v20[indv2]
		
		time_v1 = time_v10[:]
		swh_v1 = swh_v10[:]
		swh_quality_flag_v1 = swh_quality_flag_v10[:]
		rejection_flag_v1 = rejection_flag_v10[:]
		
	elif len(swh_quality_flag_v10) > len(swh_quality_flag_v20):
		NS_v2 = np.shape(swh_quality_flag_v20)
		# find the first column without NaT value for the smallest timeserie
		iv2=0
		while iv2 < NS_v2[0]:
			# print(time_v20[ik,0]*np.timedelta64(1,'s')+np.datetime64('2000-01-01'))
			if np.isnat(time2[iv2]):
				iv2 +=1
			else:
				time_v2_ref=time_v20[iv2]
				break
		minind = np.argmin(np.abs(time_v10[:]-time_v2_ref))    

		iv1 = minind - np.trunc(time_v10[minind]-time_v2_ref)
		indv1=np.arange(iv1-iv2,iv1-iv2+NS_v2[0]).astype(int)
		if indv1[-1]>=np.shape(time_v10)[0]:
			print('minind :',minind)
			print(time_v10[minind]-time_v2_ref)
			print(np.floor(time_v10[minind]-time_v2_ref))
			print('iv1 :',iv1)
			print(iv1-iv2+NS_v2[0])
			print('shape v2 time :',np.shape(time_v20))
			print('shape v1 time :',np.shape(time_v10))
		time_v1 = time_v10[indv1]
		swh_v1 = swh_v10[indv1]
		swh_quality_flag_v1 = swh_quality_flag_v10[indv1]
		rejection_flag_v1 = rejection_flag_v10[indv1]
		
		time_v2 = time_v20[:]
		swh_v2 = swh_v20[:]
		swh_quality_flag_v2 = swh_quality_flag_v20[:]
		rejection_flag_v2 = rejection_flag_v20[:]
	else:
		time_v1 = time_v10[:]
		swh_v1 = swh_v10[:]
		swh_quality_flag_v1 = swh_quality_flag_v10[:]
		time_v2 = time_v20[:]
		swh_v2 = swh_v20[:]
		swh_quality_flag_v2 = swh_quality_flag_v20[:]
		rejection_flag_v2 = rejection_flag_v20[:]
			
	if np.logical_and(len(swh_quality_flag_v1)>0,len(swh_quality_flag_v2)>0):
		# --- flag_v1 == 0 && flag_v2 ==1
		ind_v1_0_v2_1 = np.where(np.logical_and(swh_quality_flag_v1==3,swh_quality_flag_v2!=3))
		if len(ind_v1_0_v2_1)>0:
			flag_v1_0_v2_1,xbins,ybins=np.histogram2d(swh_v1[ind_v1_0_v2_1],rejection_flag_v2[ind_v1_0_v2_1],(Hs_bins,flag_bins))
		else:
			flag_v1_0_v2_1 = np.zeros((len(Hs_bins)-1,len(flag_bins)-1))	
	else:
		flag_v1_0_v2_1 = np.zeros((len(Hs_bins)-1,len(flag_bins)-1))

	yy1 = int(filev2[-23:-19])
	
	return i,flag_v1_0_v2_1, yy1, flag_bins	
	
# --------------------------------------------------------------------------------
# --- 5. Main functions to investigate tracks ------------------------------------
# --------------------------------------------------------------------------------
## Main function to read CCI data and apply is_track_above_thresh (1 Hz)
# --------------------------------------------------------------------	
def investigate_track(i, filetrack,Path_storage):
	#print(filetrack)
	time_alti,lat_alti,lon_alti,swh_alti,swh_quality=read_CCI_data(filetrack,is_rejectflag=0)
	#print(filetrack)    
	if len(swh_alti)>0:
		#Path_storage = '/home1/datawork/mdecarlo/TEMPETES/SAT_extract/'
		f2 = os.path.basename(filetrack)
		isabove_thresh = is_track_above_thresh(Path_storage,f2,time_alti,lat_alti,lon_alti,swh_alti,hs_thresh=10,nb_spikes=50,min_len=5)
		hs_max = swh_alti.max()
	else:
		hs_max=0
		isabove_thresh=False
	
	return i,isabove_thresh,hs_max
	
# --------------------------------------------------------------------------------	
## Main function to read CCI data and apply is_track_above_thresh_raw (1 Hz)
# --------------------------------------------------------------------
def investigate_track_raw(i, filetrack):
	time_alti,lat_alti,lon_alti,swh_alti,swh_quality=read_CCI_data(filetrack,is_rejectflag=0)
	if len(swh_alti)>0:
		#Path_storage = '/home1/datawork/mdecarlo/TEMPETES/SAT_extract/'
		f2 = os.path.basename(filetrack)
		isabove_thresh,len_above,nb_spikes,hs_max,lon_max,lat_max,time_max=is_track_above_thresh_raw(time_alti,lat_alti,lon_alti,swh_alti,hs_thresh=10)
	else:
		hs_max=0
		lat_max=0
		lon_max=0
		len_above=0
		nb_spikes=0
		time_max=0
		isabove_thresh=False
	return i,isabove_thresh,len_above,nb_spikes,hs_max,lon_max,lat_max,time_max,filetrack
	
# --------------------------------------------------------------------------------	
## Main function to read CFOSAT nadir data, apply smoothing (spikes removal) and apply is_track_above_thresh_raw
# --------------------------------------------------------------------
def investigate_track_all_CFOSAT(i, filetrack):
	_,_,_,_,time_alti,lat_alti,lon_alti,swh_alti=read_CFOSAT_nadir_data(filetrack)
	if len(swh_alti)>0:
		f2 = os.path.basename(filetrack)
		# ---- Remove peaks --------------------------
		swh_alti2=CFOSAT_smooth_and_filter_track(swh_alti,kernel_scatter=7,kernel_median=17,scatter_threshold=1)
		swh_alti2[np.isnan(swh_alti2)]=-0.75
		isabove_thresh,len_above,nb_spikes,hs_max,lon_max,lat_max,time_max=is_track_above_thresh_raw(time_alti,lat_alti,lon_alti,swh_alti2,hs_thresh=10)
	else:
		hs_max=0
		lat_max=0
		lon_max=0
		len_above=0
		nb_spikes=0
		time_max=0
		isabove_thresh=False
	return i,isabove_thresh,len_above,nb_spikes,hs_max,lon_max,lat_max,time_max,filetrack

# --------------------------------------------------------------------------------	
## Main function to read CCI data for 2 versions and get the Hs_max for both and the Hs where the other version reaches its max
# --------------------------------------------------------------------
def investigate_track_raw_2versions(i, filetracks):
	filetrack1 = filetracks[0]
	filetrack2 = filetracks[1]
	# --- A - 2nd version on 1rst version results ---------------
	# -- Read 1st version and calculate hs_max and time_max --------
	time_alti,lat_alti,lon_alti,swh_alti,swh_quality=read_CCI_data(filetrack1,is_rejectflag=0)
	if len(swh_alti)>0:
		_,_,_,hs_max_v1,_,_,time_max_v1=is_track_above_thresh_raw(time_alti,lat_alti,lon_alti,swh_alti,hs_thresh=10)
		_,_,_,hs_v2_on_max_v1,_=investigate_v1_on_v2_results(i,(filetrack2,time_max_v1))
	else:
		hs_max_v1=0
		time_max_v1=0
		hs_v2_on_max_v1=0
	# --- B - 1rst version on 2nd version results ---------------
	# -- Read 2nd version and calculate hs_max and time_max --------
	time_alti,lat_alti,lon_alti,swh_alti,swh_quality=read_CCI_data(filetrack2,is_rejectflag=0)
	if len(swh_alti)>0:
		_,_,_,hs_max_v2,_,_,time_max_v2=is_track_above_thresh_raw(time_alti,lat_alti,lon_alti,swh_alti,hs_thresh=10)
		_,_,_,hs_v1_on_max_v2,_=investigate_v1_on_v2_results(i,(filetrack1,time_max_v2))
	else:
		hs_max_v2=0
		time_max_v2=0
		hs_v1_on_max_v2=0
	

	return i, hs_max_v1, time_max_v1, hs_v2_on_max_v1, hs_max_v2, time_max_v2, hs_v1_on_max_v2


# --------------------------------------------------------------------------------
# --- 6. plots -------------------------------------------------------------------
# --------------------------------------------------------------------------------
## function to plot the Hs time series for any track (if ocean_name is defined : color scatter the oceans)
# --------------------------------------------------------------------------------
def plot_high_tracks(i,filetrack,Path_images):
	time1,lat,lon,swh,ocean_name=read_saved_track(filetrack)
	base_filetrack = os.path.basename(filetrack)
	base_fileimage = base_filetrack[0:-3]
	
	if np.nansum(ocean_name)==0:
		islabeled = 0
		base_fileimage = base_fileimage+'_not_labeled'
	else:
		islabeled = 1

	base_fileimage = base_fileimage+'.png'
	fileimage = os.path.join(Path_images,base_fileimage)

	if islabeled:
		labels_oceans=['Southern Ocean','North Pacific Ocean','South Pacific Ocean','South Atlantic Ocean','North Atlantic Ocean','Indian Ocean','Arctic Ocean','Close Seas']
	
	plt.figure(figsize=(14,7))
	colmap = colmap_oceanic_bassins()
	cNorm  = mcolors.Normalize(vmin=0.5, vmax=8.5)
	OceanLabelMap = cmx.ScalarMappable(norm=cNorm, cmap=colmap)
	plt.plot(time1,swh,'-k',zorder=1)
	if islabeled:
		plt.scatter(time1,swh,s=16,c=ocean_name,cmap=colmap,norm=cNorm,zorder=10)

	plt.plot(time1,np.ones(np.shape(time1))*10,'--k',zorder=0)
	plt.plot(time1,np.ones(np.shape(time1))*12,'--k',zorder=0)
	plt.plot(time1,np.ones(np.shape(time1))*14,'--k',zorder=0)

	plt.grid(True)
	plt.xlim(time1[1],time1[-1])
	
	hs_max=swh.max()
	
	if hs_max<=17.5:
		plt.ylim(0,17.5)
	else:
		plt.plot(time1,np.ones(np.shape(time1))*17.5,'--k',zorder=0)
		plt.plot(time1,np.ones(np.shape(time1))*20,'--k',zorder=0)
		plt.ylim(0,hs_max+0.2)

	if islabeled:
		cbar=plt.colorbar(OceanLabelMap,ticks=range(1,9))
		cbar.ax.set_yticklabels(labels_oceans)

	plt.title("file nb = "+str(i)+' - '+base_filetrack[0:-3])
	plt.savefig(fileimage)
	time.sleep(4)
	plt.close()
	
	return i,hs_max
	
# --------------------------------------------------------------------------------
## function to plot the modelled map of Hs vs sat track IF Hs>17.5m
# --------------------------------------------------------------------------------
def plot_very_high_tracks_maps(i,filetrack,Path_images):
	time1,lat,lon,swh,ocean_name=read_saved_track(filetrack)
	base_filetrack = os.path.basename(filetrack)
	base_fileimage = base_filetrack[0:-3]
	fileimage = base_fileimage+'.png'
	
	hs_max=swh.max()
	ind_max=swh.argmax()
	time_max=time1.data[ind_max]
	jday_month=time_max.replace(second=0,microsecond=0,minute=0,hour=0,day=1)
	PATH_WW3='/home/mdecarlo/DATA/WW3_Hs_input/LOPS_WW3-GLOB-30M_'+time_max.strftime('%y%m')+'_hs.nc'

	timesteps_day=jday+range(0,250)*3*dt.timedelta(hours=1)
	d1 =np.argmin(np.abs(time_max-timesteps_day))
	
	if hs_max>=17.5: 
		lon_mod,lat_mod,hs=read_mod_hs(PATH_WW3,d1)
		fig = plt.figure(figsize=(20,10))
		m = Basemap(projection='mill',llcrnrlat=-90,urcrnrlat=90,\
			llcrnrlon=-180,urcrnrlon=180,resolution='c')
		m.drawcoastlines(linewidth=1.25)
		m.drawparallels(np.arange(-90.,91.,30.),labels=[1,0,0,0],fontsize=16)
		m.drawmeridians(np.arange(-180.,181.,60.),labels=[0,0,0,1],fontsize=16)
		HSNorm = mcolors.Normalize(vmin=0,vmax=hs_max)
		pc=m.colormesh(lon_mod,lat_mod,hs,latlon=True,cmap=plt.cm.jet,norm=HSNorm)
		m.scatter(lon,lat,s=20,c='k')
		m.scatter(lon,lat,s=16,c=swh,cmap=plt.cm.jet,norm=HSNorm)
		plt.title("file nb = "+str(i)+' - Hsmax='+str(hs_max)+' - '+base_filetrack[0:-3])
		plt.savefig(fileimage)
		time.sleep(4)
		plt.close()

	return i,hs_max


    
