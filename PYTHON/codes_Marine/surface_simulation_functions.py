#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================================================
# === 0. Import Packages ===========================================================
# ==================================================================================
from wave_physics_functions import *
import scipy.special as sps # function erf
import scipy.interpolate as spi # function griddata
import numpy as np

def surface_1D_from_Z1kx(Z1,kX,i,nx=None,dx=None,dkx=None):
	if nx is None:
		nx = len(Z1)

	if (dx is None):
		if dkx is None:
			dx = 2*np.pi/((kX[1] - kX[0])*nx)
		else:
			dx = 2*np.pi/(dkx*nx)
			
	if (dkx is None):
		dkx = 2*np.pi/(dx*nx)

	rng = np.random.default_rng(i)
	rg = rng.uniform(low=0.0, high=1.0, size=(nx))
	zhats=np.fft.ifftshift(np.sqrt(2*Z1*dkx)*np.exp(1j*2*np.pi*rg))
	kx0=np.fft.ifftshift(kX)
    
	S1_r = np.real(np.fft.ifft(zhats,norm="forward"))
	S1_i = np.imag(np.fft.ifft(zhats,norm="forward"))

	X = np.arange(0,nx*dx,dx) # from 0 to (nx-1)*dx with a dx step
	
	return S1_r,S1_i,X,dkx
	
def surface_2D_from_Z1kxky(Z1,kX,kY,i,nx=None,ny=None,dx=None,dy=None, dkx=None, dky=None, phase_type='uniform', verbose=False):
	# /!\ Watch out : shape(S) = (ny,nx)
	# usually when doing X,Y=np.meshgrid(x,y) with size(x)=nx and size(y)=ny => size(X)=size(Y)= (ny,nx)
	kX0 = np.unique(kX)
	kY0 = np.unique(kY)
	if nx is None:
		nx = len(kX0)
	if ny is None:
		ny = len(kY0)
	
	shx = np.floor(nx/2-1)
	shy = np.floor(ny/2-1)
	if verbose: 
		print('from vec kX0, dkx = ',(kX0[1] - kX0[0]))
		print('from vec kY0, dky = ',(kY0[1] - kY0[0]))
	if (dx is None):
		if dkx is None:
			dx = 2*np.pi/((kX0[1] - kX0[0])*nx)
		else:
			dx = 2*np.pi/(dkx*nx)

	if (dy is None):
		if dky is None:
			dy = (2*np.pi/((kY0[1] - kY0[0])*ny))
		else:
			dy = 2*np.pi/(dky*ny)
	if (dkx is None):
		dkx = 2*np.pi/(dx*nx)
	if (dky is None):
		dky = 2*np.pi/(dy*ny)

	if verbose:    
		print("variables : ")
		print(' dx = ',dx,' ; dy = ',dy,' ; nx = ',nx,' ; ny = ',ny)
		print('dkx = ',dkx,' ; dky = ', dky)
	########################################################################################
	# SIDE NOTE :                                                                          #
	# obtain dkx, dky from dx and dy in order to account for eventual rounding of np.pi	   #
	# considering that dkx has been defined according to the surface requisite (nx,dx)     #
	# Eg:	                                                                           #
	# # initialisation to compute spectrum                                                 #
	# nx = 205										#
	# dx = 10										#
	# dkx = 2*np.pi/(dx*nx)								#
	# kX0 = dkx*np.arange(-nx//2+1,nx//2+1)						#
	# 											#
	# # Compute from kX0 and nx (found from kX0.shape)					#
	# dkx2 = kX0[1] - kX0[0]								#
	# dxbis = (2*np.pi/(dkx2*nx)                                                           #
	# dkx3 = 2*np.pi/(dxbis*nx)								#
	#											#
	# print('dkx = ',dkx)     		=> 0.0030649684425266273			#
	# print('dkx2 = ',dkx2) 		=> 0.0030649684425266277			#
	# print('dkx3 = ',dkx3) 		=> 0.0030649684425266273                           #
	#                                                                                      #
	########################################################################################
	rng = np.random.default_rng(i)
	if phase_type=='uniform':
		rg = rng.uniform(low=0.0, high=1.0, size=(ny,nx))
	else:
		rg = rng.normal(0,1,(ny,nx))
	zhats=np.fft.ifftshift(np.sqrt(2*Z1*dkx*dky)*np.exp(1j*2*np.pi*rg))
	ky2D=np.fft.ifftshift(kY) 
	kx2D=np.fft.ifftshift(kX) 

	#     real part
	S2_r = np.real(np.fft.ifft2(zhats,norm="forward"))
	#     also computes imaginary part (useful for envelope calculations) 
	S2_i = np.imag(np.fft.ifft2(zhats,norm="forward"))

	X = np.arange(0,np.floor(nx*dx),dx) # from 0 to (nx-1)*dx with a dx step
	Y = np.arange(0,np.floor(ny*dy),dy)

	return S2_r,S2_i,X,Y,kX0,kY0,i,dkx,dky	

def surface_from_Efth(Efth,f_vec,th_vec,seed=0,nx=2048,ny=2048,dx=10,dy=10,D=None,iswvnb=0):
	import xarray as xr
	g=9.81
	#
	# Here we start by adding last and first value at the border of the spectrum in order to deal with the -180/180 gap
	spec = xr.DataArray(Efth,
		dims=['n_phi','nk'],
		coords={
			"phi_vector" : (["n_phi"], th_vec),
			"k_vector" : (["nk"], f_vec),
			},
		)
		
	spec_bis = xr.concat([spec.isel(n_phi=-1),spec,spec.isel(n_phi=0)],dim="n_phi")
	# --- change the first and last new values to have a 2pi revolution ---------------
	A = np.concatenate([[-360],np.zeros((spec.sizes['n_phi'])),[360]])
	factor = xr.DataArray(A, dims="n_phi")
	spec_bis['phi_vector'].values = spec_bis['phi_vector']+factor
	spec_bis = spec_bis.interpolate_na(dim='n_phi')

	# Then, turning the spectrum to cartesian coordinates in order 
	# to apply the Inverse Fourier to it and get the surface                   
	#
	# -- get cartesian spectrum ------
	if iswvnb:
		Ekxky0, kx, ky = spectrum_from_kth_to_kxky(np.squeeze(spec_bis.compute().data),  spec_bis['k_vector'].compute().data,spec_bis["phi_vector"].compute().data)
	else:
		Ekxky0, kx, ky = spectrum_from_fth_to_kxky(np.squeeze(spec_bis.compute().data),  spec_bis['k_vector'].compute().data,spec_bis["phi_vector"].compute().data)

	# -- GET THE INTERPOLATION GRID for the spectrum -----
	# -- get kx,ky values we want from the surface we want ---------
	dkx = 2*np.pi/(dx*nx)
	dky = 2*np.pi/(dy*ny)
	kX0 = np.fft.fftshift(np.fft.fftfreq(nx,d=dx))*2*np.pi
	kY0 = np.fft.fftshift(np.fft.fftfreq(ny,d=dy))*2*np.pi

	kX,kY = np.meshgrid(kX0,kY0)# , indexing='ij')

	# --- compute associated Kf, Phi(in deg) ---------
	kK = (np.sqrt(kX**2+kY**2))
	if iswvnb:
		kF = kK
	else:
		kF = f_from_k(kK,D=D)

	kPhi = np.arctan2(kY,kX)*180/np.pi
	kPhi[kPhi<0]=kPhi[kPhi<0]+360

	# create a dataArray with the new (i.e. wanted) values of F written in a cartesian array
	kF2 = xr.DataArray(kF, coords=[("ky", kY0), ("kx",kX0)])

	kPhi2 = xr.DataArray(kPhi, coords=[("ky", kY0), ("kx",kX0)])
	FPhi2s = xr.Dataset(
	{'kF': kF2,
	'kPhi': kPhi2}
	).stack(flattened=["ky", "kx"])

	Ekxky = xr.DataArray(Ekxky0, dims=("nf", "n_phi"), coords={"nf":f_vec , "n_phi":spec_bis['phi_vector']})
	B = Ekxky.interp(nf=FPhi2s.kF,n_phi=FPhi2s.kPhi,kwargs={"fill_value": 0})
	B.name='Ekxky_new'
	B0 = B.reset_coords(("nf","n_phi"))
	Ekxky_for_surf = B0.Ekxky_new.unstack(dim='flattened')

	S2_r,S2_i,X,Y,kX0,kY0,rg,dkx,dky = surface_2D_from_Z1kxky(Ekxky_for_surf,kX,kY,seed)

	return S2_r,S2_i,X,Y,rg,kX0,kY0,Ekxky_for_surf,dkx,dky

##################################################################################
def def_spectrumG_for_surface_1D(nx=2048,dx=10,T0=10,Hs=4,sk_k0=0.1,D=None,verbose=False):
	dkx = 2*np.pi/(dx*nx)

	kX = np.fft.fftshift(np.fft.fftfreq(nx,d=dx))*2*np.pi
	
	# --- only Gaussian -------------		
	Z1_Gaussian,kX,sk = Gaussian_1Dspectrum_kx(kX,T0,sk_k0,D=D)
	Z1 =(Hs/4)**2*Z1_Gaussian
	sumZ1=4*np.sqrt(sum(Z1.flatten()*dkx)) 
	if verbose:
		print('Hs for Gaussian : ',sumZ1)
	
	return Z1, kX,sk

def def_spectrumPM_for_surface_1D(nx=2048,dx=10,T0=10,D=None,verbose=False): #Hs=4,sk_k0=0.1
	dkx = 2*np.pi/(dx*nx)
	
	kX = np.fft.fftshift(np.fft.fftfreq(nx,d=dx))*2*np.pi
	
	# --- only PM -------------	
	Z1_PM = PM_spectrum_k(kX,1/T0,D=D)	
	# Z1_Gaussian,kX,sk = Gaussian_1Dspectrum_kx(kX,T0,sk_k0,D=D)
	# Z1 =(Hs/4)**2*Z1_Gaussian
	sumZ1=4*np.sqrt(sum(Z1_PM[np.isfinite(Z1_PM)].flatten()*dkx)) 
	if verbose:
		print('Hs for Gaussian : ',sumZ1)
	Z1_PM[np.isnan(Z1_PM)]=0
	return Z1_PM, kX
	
def def_spectrumJONSWAP_for_surface_1D(nx=2048,dx=10,T0=10,D=None,gammafac=3.3,sigA=0.07,sigB=0.09,verbose=False):
	dkx = 2*np.pi/(dx*nx)
	
	kX = np.fft.fftshift(np.fft.fftfreq(nx,d=dx))*2*np.pi
	fX = sig_from_k(kX,D=D)/(2*np.pi)
	Z1_PM = PM_spectrum_k(kX,1/T0,D=D)
	Z1_PM[np.isnan(Z1_PM)]=0
	fp = 1/T0
	sigAB = np.where(fX<fp,sigA,sigB)
	JSfactor = gammafac**np.exp((-(fX-fp)**2)/(2*sigAB**2*fp**2))
	Z1_JS = Z1_PM*JSfactor
	Z1_JS[np.isnan(Z1_JS)] = 0
	sumZ1=4*np.sqrt(sum(Z1_JS[np.isfinite(Z1_JS)].flatten()*dkx)) 
	if verbose:
		sumZ1=4*np.sqrt(sum(Z1_JS[np.isfinite(Z1_JS)].flatten()*dkx)) 
		print('Hs for Jonswap : ',sumZ1)
	return Z1_JS, kX

def def_spectrum_for_surface(nx=2048,ny=2048,dx=10,dy=10,theta_m=30,D=1000,T0=10,Hs=4,sk_theta=0.001,sk_k=0.001,\
nk=1001,nth=36,klims=(0.0002,0.2),n=4,typeSpec='Gaussian',verbose=False):
	dkx = 2*np.pi/(dx*nx)
	dky = 2*np.pi/(dy*ny)

	kX0 = np.fft.fftshift(np.fft.fftfreq(nx,d=dx))*2*np.pi
	kY0 = np.fft.fftshift(np.fft.fftfreq(ny,d=dy))*2*np.pi
	kX,kY = np.meshgrid(kX0, kY0)
	
		
	if typeSpec=='Gaussian':
		if verbose:
			print('Gaussian spectrum selected. Available options:\n - Hs, sk_theta, sk_k. \nWith (sk_k, sk_theta) the sigma values for the k-axis along the main direction and perpendicular to it respectively \n Other options (common to all spectrum types) are : nx, ny, dx, dy, T0, theta_m, D')
		
		Z1_Gaussian0,kX,kY = define_Gaussian_spectrum_kxky(kX,kY,T0,theta_m*np.pi/180,sk_theta,sk_k,D=D)
		
		Z1 =(Hs/4)**2*Z1_Gaussian0/np.sum(Z1_Gaussian0.flatten()*dkx*dky)
		sumZ1=4*np.sqrt(sum(Z1.flatten()*dkx*dky))
		if verbose: 
			print('Hs for Gaussian : ',sumZ1)
	
	elif typeSpec=='PM':
		if verbose:
			print('Pierson-Moskowitz* cos(theta)^(2*n) spectrum selected. Available options:\n - nk, nth, klims, n. \nWith n the exponent of the directional distribution: cos(theta)^(2*n)\n Other options (common to all spectrum types) are : nx, ny, dx, dy, T0, theta_m, D')
		k=np.linspace(klims[0],klims[1],nk)
		thetas=np.linspace(0,360*(nth-1)/nth,nth)*np.pi/180.

		Ekth,k,th = define_spectrum_PM_cos2n(k,thetas,T0,theta_m*np.pi/180.,D=D,n=n)
		Ekxky, kx, ky = spectrum_to_kxky(1,Ekth,k,thetas,D=D)

		Z1=spi.griddata((kx.flatten(),ky.flatten()),Ekxky.flatten(),(kX,kY),fill_value=0)
		sumZ1=4*np.sqrt(sum(Z1.flatten()*dkx*dky)) 
		if verbose:
			print('Hs for Pierson Moskowitz : ',sumZ1)

	return Z1, kX, kY,dkx,dky


