# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================================================
# === 0. Import Packages ===========================================================
# ==================================================================================
from wave_physics_functions import *
import scipy.special as sps # function erf
import numpy as np


def generate_wvform_database(dr,nHs,edges_max=20,Hs_max=25,offset=10):
	# edges=np.linspace(0,edges_max,ne)    # range vector
	edges=np.arange(0,edges_max+dr,dr)   # range vector
	# dr=edges[1]-edges[0]   
	ne=len(edges)               
	Hsm=np.linspace(0,Hs_max,nHs)
	wfm=np.zeros((nHs,ne-1))
	for k in range(nHs):
		wfm[k,:]=0.5+0.5*sps.erf((edges[:-1]+0.5*dr-offset) / (0.25*np.sqrt(2)*Hsm[k]))

	return wfm, Hsm, edges	

def simple_retracking_process(wfm,edges,nHs=251,alti_sat=519000,dx=10,offset=10,index_calc=None,wfm_ref=None,Hsm_ref=None,ispolyfit=0):
	if (type(wfm_ref)==type(None)) | (type(Hsm_ref)==type(None)):
		if (type(wfm_ref)==type(None)) ^ (type(Hsm_ref)==type(None)):
			print("optional inputs 'wfm_ref' and 'Hsm_ref' are paired inputs i.e. in order to be applied they should be given together")
			print("As this is not the case here : the basic waveform database is computed")
		wfm_ref, Hsm_ref, edges = generate_wvform_database(len(edges),nHs,edges_max=edges[-1],offset=offset)
	
	dr = edges[1]-edges[0]
	Apix = np.pi*2*alti_sat*dr / (dx**2) # The area of a ring, in terms of pixels 

	testwf=np.broadcast_to(wfm,(nHs,len(wfm)))
	if type(index_calc)==type(None):
		dist=np.sum((Apix*wfm_ref-testwf)**2,axis=1)
	else:
		dist=np.sum((Apix*wfm_ref[:,index_calc]-testwf[:,index_calc])**2,axis=1)
	
	if ispolyfit:
		p = np.polyfit(Hsm_ref,dist,2)
		Hs = -p[1]/(2*p[0])
	else:
		Imin=np.argmin(dist)
		Hs = Hsm_ref[Imin]

	return Hs
	

def fly_over_track_v0(X,Y,S1,nsamp,nxa,di,wfm_ref,Hsm_ref,edges_ref,radi,radi1,radi2,alti_sat,range_shift):
	# radi = 4000     # radius used to compute waveform
	#radi1 = 900     # inner radius for Hs average
	#radi2 = 1200    # outer radius for Hs average

	ny_mid = len(np.unique(Y))//2
	Xalt = np.zeros((nsamp,1))
	Hs_retrack = np.zeros((nsamp,1))
	Hs_std = np.zeros((nsamp,1))
	Hs_stdbis = np.zeros((nsamp,1))
	Hs_std2 = np.zeros((nsamp,1))
	waveforms=np.zeros((nsamp,len(edges_ref)-1))

	# Footprint definition 
	dx = X[1]-X[0]
	dy = Y[1]-Y[0]
	footprint=np.ones((2*nxa+1,2*nxa+1))
	footprint1=np.ones((2*nxa+1,2*nxa+1))
	footprint2=np.ones((2*nxa+1,2*nxa+1))

	[Xa,Ya]=np.meshgrid(dx*np.arange(-nxa,nxa+1), dy*np.arange(-nxa,nxa+1))
	dist_ground = (Xa**2+Ya**2)
	footprint[dist_ground > radi **2]=0
	footprint1[dist_ground > radi1**2]=0
	footprint2[dist_ground > radi2**2]=0
	footprint2[dist_ground < radi1**2]=0

	for isamp in range(nsamp):
		ialt=(nxa+isamp*di).astype(int)
		Xalt[isamp] = X[ialt]
		surf=S1[ny_mid-nxa:ny_mid+nxa+1,ialt-nxa:ialt+nxa+1]*footprint
		surf1=S1[ny_mid-nxa:ny_mid+nxa+1,ialt-nxa:ialt+nxa+1]*footprint1
		surf2=S1[ny_mid-nxa:ny_mid+nxa+1,ialt-nxa:ialt+nxa+1]*footprint2
                # spatial averaging of Hs : disc < radi1 et annulus from radi1 to radi2
		Hs_std [isamp] = 4*np.std(surf1)/np.sqrt(np.mean(footprint1))
		surf1bis=np.nan*np.ones(surf1.shape)
		surf1bis[footprint1>0]=surf1[footprint1>0]
		Hs_stdbis [isamp] = 4*np.nanstd(surf1bis)
		Hs_std2[isamp] = 4*np.std(surf2)/np.sqrt(np.mean(footprint2))

		# r is distance to satellite = range + shift 
		r=np.sqrt(Xa**2+Ya**2+(alti_sat-surf)**2)-alti_sat+range_shift
		r[dist_ground > radi**2]=np.nan  # equivalent to multiplication by footprint

		counts,_=np.histogram(r,bins=edges_ref)
		Hs_retrack[isamp] = simple_retracking_process(counts,edges_ref,wfm_ref=wfm_ref,Hsm_ref=Hsm_ref,ispolyfit=0) 
		waveforms[isamp,:]=counts

	return Hs_std,Hs_stdbis,Hs_std2,Hs_retrack,Xalt,waveforms,surf1,footprint1

