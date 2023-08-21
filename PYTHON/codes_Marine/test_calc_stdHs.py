# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================================================
# === 0. Import Packages ===========================================================
# ==================================================================================
import sys
# appending a path
# sys.path.append('/home1/datahome/mdecarlo/AltiProcessing/codes/functions_py/')
sys.path.append('/home/mdecarlo/Documents/TOOLS/optools_FA/optools/PYTHON/codes_Marine/')

import os, glob
import xarray as xr
import numpy as np
import pandas as pd
import time

import scipy.interpolate as spi
import scipy.integrate as spint
from scipy.ndimage import gaussian_filter
from scipy.signal import hilbert,hilbert2
import multiprocessing as mp

from functions_cfosat_env import *
from surface_simulation_functions import *
from matplotlib.dates import DateFormatter
# --- plotting and interactive stuff ----------------------
import matplotlib.pyplot as plt
# from matplotlib.ticker import AutoMinorLocator, FixedLocator

# --- read input to slect only part of files -----------------------
if len(sys.argv)>=3:
	nf1 = int(sys.argv[1])
	nf2 = int(sys.argv[2])
else:
	nf1=0
	nf2 = 4

print('dates : ',nf1,' to ',nf2)
start_time=time.time()

# ==================================================================================
# === 1. Parameters for the run ====================================================
# ==================================================================================
isbabord=1
yea = 2021
nbeam = 2

PATH_save = '/home1/datawork/mdecarlo/CFOSAT_Std_Hs/NEW'
# ==================================================================================
# === 2. Read files ================================================================
# ==================================================================================
PATH_L2S = '/home/ref-cfosat-public/datasets/swi_l2s/v1.0/'
PATH_L2 = '/home/datawork-cersat-public/provider/cnes/satellite/l2/cfosat/swim/swi_l2____/op05/5.1.2/'
PATH_L2P = '/home1/datawork/mdecarlo/CFOSAT_L2P/swim_l2p_box_nrt/'

str_to_remove = ['OPER','TEST']
	
list_L2,list_L2S,list_L2P = get_files_for_1_year(yea,nbeam=nbeam,str_to_remove=str_to_remove,PATH_L2S=PATH_L2S,\
								PATH_L2 = PATH_L2, PATH_L2P = PATH_L2P)

# ==================================================================================
# === 3. Create pool and run for selected files ====================================
# ==================================================================================
pool = mp.Pool(2)

results = []

results = pool.starmap_async(function_one_track_2D, [(i, files,nbeam,isbabord) for i, files in enumerate(zip(list_L2[nf1:nf2],list_L2S[nf1:nf2],list_L2P[nf1:nf2]))]).get()

pool.close()
print('done : before saving')

# ==================================================================================
# === 4. Read outputs and save them ================================================
# ==================================================================================
name_outputs = ['indfile','time_box', 'Hs_box', 'std_Hs_box',  'lat_box', 'lon_box', 'std_Hs_L2_2D', 'Hs_L2_2D', 'Lambda2_L2_2D', 'std_Hs_L2S_2D', 'Hs_L2S_2D','Lambda2_L2S_2D','flag_valid_L2P_swh_box', 'flag_valid_L2P_spec_box']#,'namefile']

for iout,outp in enumerate(name_outputs):
	namefile = os.path.join(PATH_save,'CFOSAT_BDD_'+str(yea)+'_'+f'{nf1:04d}'+'_'+f'{nf2:04d}'+'_'+outp+'_sorted_NEW')
	testvar = [result[iout] for result in results]
	if iout<4:
		print(outp,'--------------------')
		print(testvar)
	if iout==0:
		inds = np.argsort(np.array(testvar))
		print(inds)
	print(type(inds))
	print(type(inds[0]))
	testvar_sorted = np.array(testvar)[inds]
	np.save(namefile,testvar_sorted)

print('time to the end : ',time.time()-start_time)
	

