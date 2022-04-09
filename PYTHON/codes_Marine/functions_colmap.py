#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================================================
# === 0. Import Packages ===========================================================
# ==================================================================================
import glob
import os
import sys

# to work with mathematical and numerical stuff
import numpy as np

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


# ==================================================================================
# === 1. Functions       ===========================================================
# ==================================================================================

# -- function create_nipy_colmap
# create a colormap for spectra
def create_nipy_colmap():
	nipy_spectral = cmx.get_cmap('nipy_spectral', 256)
	indexes = np.linspace(0,1,256)
	COL = nipy_spectral(np.linspace(0,1,256))
	diff_col = COL[1:,:]- COL[0:-1,:]
	cdict ={}
	for kk in range(3):
		COLk = COL[:,kk]
		ind = np.where(abs(diff_col[1:,kk]-diff_col[0:-1,kk])>0.0005)[0]
		tup1= []
		if kk!=1:
			ind = ind[2:]
		if kk == 2:
			tup=(indexes[0],1,1)
		else:
			tup = (indexes[0],0.8,0.8)#COLk[0],COLk[0])
			
		tup1.append(tup)
		
		for iik in range(len(ind)):
			indik = ind[iik]+1
			tup = (indexes[indik],COLk[indik],COLk[indik])
			tup1.append(tup)
		if kk==0:
			tup = (indexes[-1],0.35,0.35)#COLk[-1],COLk[-1])
		else:
			tup = (indexes[-1],0.1,0.1)
		tup1.append(tup)
		if kk==0:
			cdict['red']=tup1   
		elif kk==1:
			cdict['green']=tup1 
		elif kk==2:
			cdict['blue']=tup1   
	
	newcmp = mcolors.LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)
	
	return newcmp
	
# -- Function generate colmap regularly space from number of colours --
# inputs need to be generated as:
# cols = []
# cols.append((R1,G1,B1))
# cols.append((R2,G2,B2))
def gen_colmap_linspace(cols):
	length_cols = np.size(cols,0)
	dcol = 1/length_cols
	tup_r = []
	tup_g = []
	tup_b = []
	#initialize
	tup_r.append((0,cols[0][0],cols[0][0]))
	tup_g.append((0,cols[0][1],cols[0][1]))
	tup_b.append((0,cols[0][2],cols[0][2]))
	
	for k in range(1,length_cols):
		tup_r.append((k*dcol,cols[k-1][0],cols[k][0]))
		tup_g.append((k*dcol,cols[k-1][1],cols[k][1]))
		tup_b.append((k*dcol,cols[k-1][2],cols[k][2]))
	
	tup_r.append((1,cols[-1][0],cols[-1][0]))
	tup_g.append((1,cols[-1][1],cols[-1][1]))
	tup_b.append((1,cols[-1][2],cols[-1][2]))
	
	cdict1 ={}
	cdict1['red']=tup_r
	cdict1['green']=tup_g
	cdict1['blue']=tup_b
	
	newcolmap = mcolors.LinearSegmentedColormap('testCmap', segmentdata=cdict1, N=256)
	
	return newcolmap

# -- function colmap Oceanic bassin ---
def colmap_oceanic_bassins():
	cols =[]
	cols.append((1,1,0.2))
	cols.append((0.6,0.3,0.65))
	cols.append((0.9,0.1,0.1))
	cols.append((0.65,0.85,0.32))
	cols.append((0.2,0.63,0.17))
	cols.append((1.,0.5,0.))
	cols.append((0.21,0.5,0.75))
	cols.append((0.6,0.6,0.6))
	
	newcolmap=gen_colmap_linspace(cols)
	
	return newcolmap
