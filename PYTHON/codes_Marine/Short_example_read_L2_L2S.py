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
# -- to work with dates
import datetime as dt
from datetime import datetime

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

mpl.rcParams.update({'font.size': 14,'savefig.facecolor':'white'})

twopi = 2* np.pi


def decode_two_columns_time(coord,name_dim):
	factor = xr.DataArray([1e6, 1], dims=name_dim)
	time_us = (coord * factor).sum(dim=name_dim).assign_attrs(units="microseconds since 2009-01-01 00:00:00 0:00")
	#n_tim since 2009-01-01 00:00:00 0:00
	return time_us	

def get_datetime_fromDOY(YYYY,DOY):
	return pd.to_datetime(dt.datetime(YYYY,1,1))+pd.Timedelta(days=DOY-1)


# ==================================================================================
# === 1. Functions to xr.open CFOSAT files =========================================
# ==================================================================================
def preprocess_boxes_env_work_quiet(ds0):
	ds = ds0[["lat_spec_l2", "lon_spec_l2", "wave_param", "k_spectra" ,"phi_vector","nadir_swh_box", "nadir_swh_box_std", "flag_valid_swh_box","indices_boxes"]] # Add the variables you consider useful (not to have 157 variables all along...)
	ds = ds.assign(time_box=decode_two_columns_time(ds0.time_l2,"n_tim"))
	ds=ds.assign({"dk":(("nk"),np.gradient(ds['k_spectra'].compute().data))})
	# Prepare transformations (jacobian and slope to waves)
	ds=ds.assign(wave_spectra_kth_hs=ds0['pp_mean']*(ds['k_spectra']**-1))
	# --- do some renaming ------
	ds = ds.rename_vars({"k_spectra":"k_vector"})
	
	
	return xr.decode_cf(ds)

	
def read_boxes_from_L2_CNES_env_work_quiet(L2):
	ds_L2=xr.open_mfdataset(L2,concat_dim="n_box", combine="nested",decode_times=False,
		decode_coords=False,data_vars='minimal',coords="minimal",compat='override',autoclose=True,
		preprocess=preprocess_boxes_env_work_quiet)
	return ds_L2

def preprocess_offnadir_env_work_quiet(ds0):
	ds = ds0[["phi","lat","lon","k","time", "wave_spectra","dk"]]
	ds = ds.assign({"wave_spectra_kth_hs":((ds.wave_spectra*ds.k**-1))})
	# --- do some renaming useful for concatenation afterwards ----
	ds = ds.swap_dims({'time':'time0'})
	ds = ds.reset_coords('time')
	ds = ds.rename_vars({'time':'time_per_angle'})
	# --- do some renaming to compare with L2 ----
	ds = ds.rename_vars({"k":"k_vector"})
	ds = ds.rename_vars({"phi":"phi_vector"})
	ds = ds.swap_dims({"k":"nk"})
	return ds
	
def read_l2s_offnadir_files_env_work_quiet(offnadir_files):
	ds_l2s = xr.open_mfdataset(offnadir_files,concat_dim="l2s_angle", combine="nested",decode_coords=False,autoclose=True,
		data_vars='minimal',coords="all",compat='override',preprocess=preprocess_offnadir_env_work_quiet)
	return ds_l2s


def get_indices_macrocycles(inds):
	# inds = ds_l20['indices_boxes'].isel(n_box=35,n_posneg=1,n_beam_l1a=3+nbeam)
	indssel = []
	for i in np.arange(0,inds.size,2):
		if np.isfinite(inds.isel(ni=i)):
			indssel.append(np.arange(inds.isel(ni=i),inds.isel(ni=i+1)+1,dtype='int'))
	return np.concatenate(indssel)
	
def get_bigcycles(ds):
# --- get entire cycle ----
	ind_end_0=np.where((ds.phi_vector[1:]-ds.phi_vector[0:-1])>100)[0]
	inds=np.zeros(len(ind_end_0)+2,dtype='int')-1
	inds[-1] = len(ds.phi_vector)-1
	inds[1:-1]=ind_end_0
	len_s=inds[1:]-inds[0:-1]
	big_nb = np.repeat(np.arange(len(len_s)),len_s)

	ds = ds.assign({"bigcycle_label":(("time0"),big_nb)})
	return ds
# ==================================================================================
# === 2. Function to work over a 2D spectrum =======================================
# ==================================================================================
def from_spec_CFOSAT_to_2Sided(ds):
	# ds_CNES_sel[['k_vector','phi_vector','wave_spectra_kth_hs','dk']]
	Spec_1 = ds.copy(deep=True)
	Spec_1["phi_vector"].values = (Spec_1["phi_vector"].compute().data+180.)%360
	Spec_2 = xr.concat([ds,Spec_1],dim="n_phi",data_vars='minimal')
	Spec_2['wave_spectra_kth_hs'].values = Spec_2['wave_spectra_kth_hs'].values/2
	# Spec_2['wave_spectra_kth_hs'].values[np.isnan(Spec_2['wave_spectra_kth_hs'].values)]=0

	Spec_2 = Spec_2.sortby('phi_vector')
	return Spec_2

def from_spec_CFOSAT_to_1Sided(ds):
	Spec_1 = ds.copy(deep=True)
	Spec_1["phi_vector"].values = (Spec_1["phi_vector"].compute().data+180.)%360
	Spec_1['wave_spectra_kth_hs'].values = 0*Spec_1['wave_spectra_kth_hs'].values
	Spec_2 = xr.concat([ds,Spec_1],dim="n_phi",data_vars='minimal')
	Spec_2['wave_spectra_kth_hs'].values = 1.*Spec_2['wave_spectra_kth_hs'].values
	# Spec_2['wave_spectra_kth_hs'].values[np.isnan(Spec_2['wave_spectra_kth_hs'].values)]=0

	Spec_2 = Spec_2.sortby('phi_vector')
	return Spec_2

	
# ==================================================================================
# === 3. Function to work over a track (only 2D) ===================================
# ==================================================================================
def function_one_track(file_L2,file_L2S,nbeam,isbabord):
	# --- read files CNES ------------ 
	ds_boxes = read_boxes_from_L2_CNES_env_work_quiet(file_L2)
	
	# --- read files ODL ------------ 
	# .isel(l2S_angle = 0) : because there is the posibility to concatenate over incidence angles when reading the file
	ds_l2s = read_l2s_offnadir_files_env_work_quiet(file_L2S).isel(l2s_angle=0)
	ds_l2S_2= get_bigcycles(ds_l2S)
	
	ntim = ds_boxes.dims['n_box']
	time_box = np.zeros((ntim))
	
	# # --- Loop over time steps --------------------------------
	# for it in range(ntim):
	# --- OR select one time step (i.e. one box)------------
	it = 35
		# print(it, 'over ',ntim,' -------------')
	ds_CNES_sel = ds_boxes.isel(n_posneg=isbabord,n_box=it,n_beam_l2=nbeam,n_beam_l1a=3+nbeam)
	# n_beam_l1a is useful to get_indices_macrocycles if I remember correctly
	
	if ds_CNES_sel['flag_valid_swh_box'] == 0:
		# ---- select indices for macrocycles L2S ------
		ind_L2S = get_indices_macrocycles(ds_CNES_sel['indices_boxes'])
		# ---- select macrocycles L2S from indices ------
		ds_ODL_sel = ds_l2s.isel(time0 = ind_L2S).swap_dims({'time0':'n_phi'}).set_coords('phi_vector')	
		# --- /!\ ------ /!\ ------ /!\ ------ /!\ ------ /!\ ------ /!\ ------
		# -- Once the selection is done, do not forget renaming the dim time0 into n_phi 
		# -- and making phi_vector a coord :
		# --  .swap_dims({'time0':'n_phi'}).set_coords('phi_vector')
		# --- /!\ ------ /!\ ------ /!\ ------ /!\ ------ /!\ ------ /!\ ------  
		
		# /!\ Watch out 'it2' is not SIMPLY related to 'it', better to check on a map (here I put the same for the example only)
		it2 = 35
		ind_big = np.where(ds_l2S_2['bigcycle_label'] == it2)[0]
		spec_L2S_bis = ds_l2S_2.isel(time0 = ind_big).swap_dims({'time0':'n_phi'}).set_coords('phi_vector')
		
		# DO Whatever you want 
		# e.g. 
		# spec_L2S = from_spec_CFOSAT_to_2Sided(ds_ODL_sel)
		# spec_L2 = from_spec_CFOSAT_to_2Sided(ds_CNES_sel)
		# fig,axs = plt.subplots(1,3,figsize=(14, 5),subplot_kw={'projection': 'polar'})
		# axs[0].pcolormesh((90-spec_L2.phi_vector)*np.pi/180,spec_L2.k_vector,
		# 		  spec_L2.wave_spectra_kth_hs.compute(),
		# 		  cmap='viridis')
		# axs[1].pcolormesh((90-spec_L2S.phi_vector)*np.pi/180,spec_L2S.k_vector,
		# 		  spec_L2S.wave_spectra_kth_hs.compute().T,
		# 		  cmap='viridis')
		# axs[2].pcolormesh((90-spec_L2S_bis.phi_vector)*np.pi/180,spec_L2S_bis.k_vector,
		# 		  spec_L2S_bis.wave_spectra_kth_hs.compute().T,
		# 		  cmap='viridis')

	return
