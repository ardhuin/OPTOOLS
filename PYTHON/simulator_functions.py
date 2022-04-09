# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================================================
# === 0. Import Packages ===========================================================
# ==================================================================================
from wave_physics_functions import *
import scipy.special as sps # function erf
import numpy as np


def surface_from_Z1kxky(Z1,kX,kY,nx=None,ny=None,dx=None,dy=None,dkx=None,dky=None):
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
	X = np.arange(0,nx*dx,dx) # from 0 to (nx-1)*dx with a dx step
	Y = np.arange(0,ny*dy,dy)
	
	return S1,X,Y 
	
def def_spectrum_for_surface(nx=2048,ny=2048,dx=10,dy=10,theta_m=30,D=1000,T0=10,Hs=4,sk_theta=0.001,sk_k=0.001,nk=1001,nth=36,klims=(0.0002,0.2),n=4,typeSpec='Gaussian'):
	dkx = 2*np.pi/(dx*nx)
	dky = 2*np.pi/(dy*ny)

	kX0 = dkx*np.arange(-nx//2+1,nx//2+1)
	kY0 = dky*np.arange(-ny//2+1,ny//2+1)
	kY,kX = np.meshgrid(kY0, kX0)
		
	if typeSpec=='Gaussian':
		print('Gaussian spectrum selected. Available options:\n - Hs, sk_theta, sk_k. \nWith (sk_k, sk_theta) the sigma values for the k-axis along the main direction and perpendicular to it respectively \n Other options (common to all spectrum types) are : nx, ny, dx, dy, T0, theta_m, D')
		
		Z1_Gaussian0,kX,kY=define_Gaussian_spectrum_kxky(kX,kY,T0,theta_m*np.pi/180,sk_theta,sk_k,D=D)
		Z1 =(Hs/4)**2*Z1_Gaussian0
		sumZ1=4*np.sqrt(sum(Z1.flatten()*dkx*dky)) 
		print('Hs for Gaussian : ',sumZ1)
	
	elif typeSpec=='PM':
		print('Pierson-Moskowitz* cos(theta)^(2*n) spectrum selected. Available options:\n - nk, nth, klims, n. \nWith n the exponent of the directional distribution: cos(theta)^(2*n)\n Other options (common to all spectrum types) are : nx, ny, dx, dy, T0, theta_m, D')
		k=np.linspace(klims[0],klims[1],nk)
		thetas=np.linspace(0,360*(nth-1)/nth,nth)*np.pi/180.

		Ekth,k,th = define_spectrum_PM_cos2n(k,thetas,T0,theta_m*np.pi/180.,D=D,n=n)
		Ekxky, kx, ky = spectrum_to_kxky(1,Ekth,k,thetas,D=D)

		Z1=spi.griddata((kx.flatten(),ky.flatten()),Ekxky.flatten(),(kX,kY),fill_value=0)
		sumZ1=4*np.sqrt(sum(Z1.flatten()*dkx*dky)) 
		print('Hs for Pierson Moskowitz : ',sumZ1)

	return Z1, kX, kY


