# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================================================
# === 0. Import Packages ===========================================================
# ==================================================================================
from Misc_functions import *
from wave_physics_functions import *
import scipy.special as sps # function erf
from IPython.display import clear_output

def surface_from_Z1kxky(Z1,kX,kY,i,nx=None,ny=None,dx=None,dy=None,dkx=None,dky=None):
# /!\ Watch out : shape(S) = (ny,nx)
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
	rng = np.random.default_rng(i)
	# rg = rng.normal(0,1,(ny,nx))
	rg = rng.uniform(low=0.0, high=1.0, size=(ny,nx))
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
	
def def_spectrum_for_surface(nx=2048,ny=2048,dx=10,dy=10,theta_m=30,D=1000,T0=10,Hs=4,sk_theta=0.001,sk_k=0.001,nk=1001,nth=36,klims=(0.0002,0.2),n=4,typeSpec='Gaussian'):
	dkx = 2*np.pi/(dx*nx)
	dky = 2*np.pi/(dy*ny)

	kX0 = np.fft.fftshift(np.fft.fftfreq(nx,d=dx))*2*np.pi
	kY0 = np.fft.fftshift(np.fft.fftfreq(ny,d=dy))*2*np.pi
	kX,kY = np.meshgrid(kX0, kY0)
		
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

def fly_over_track_v0(X,Y,S1,nsamp,nxa,di,wfm_ref,Hsm_ref,edges_ref,radi):
	ny_mid = len(np.unique(Y))//2
	Xalt = np.zeros((nsamp,1))
	Hs_retrack = np.zeros((nsamp,1))
	Hs_std = np.zeros((nsamp,1))
	waveforms=np.zeros((nsamp,len(edges_ref)-1))

	# Footprint definition 
	dx = X[1]-X[0]
	dy = Y[1]-Y[0]
	footprint=np.ones((2*nxa+1,2*nxa+1))
	[Xa,Ya]=np.meshgrid(dx*np.arange(-nxa,nxa+1), dy*np.arange(-nxa,nxa+1))
	dist_ground = (Xa**2+Ya**2)
	footprint[dist_ground > radi**2]=np.nan

	for isamp in range(nsamp):
		# clear_output(wait=True)
		print(isamp)
		ialt=(nxa+isamp*di).astype(int)
		Xalt[isamp] = X[ialt]
		surf=S1[ny_mid-nxa:ny_mid+nxa+1,ialt-nxa:ialt+nxa+1]*footprint
		# to have distance to satellite = range
		r=np.sqrt(Xa**2+Ya**2+(alti_sat-surf)**2)-alti_sat+10
		r[dist_ground > radi**2]=np.nan  # equivalent to multiplication by footprint

		counts,_=np.histogram(r,bins=edges_ref)
		Hs_retrack[isamp] = simple_retracking_process(counts,edges_ref,wfm_ref=wfm_ref,Hsm_ref=Hsm_ref) 
		waveforms[isamp,:]=counts
		Hs_std[isamp] = 4*np.nanstd(surf.flatten())#/np.sqrt(np.mean(footprint))

	return Hs_std,Hs_retrack,Xalt,waveforms


## investigate size ring for Hs calculation
def fly_over_track_v05_radius(i,X,Y,S1,nsamp,nxa,di,wfm_ref,Hsm_ref,edges_ref,offset_range,radi_wfm,radis1,radis2,alti_sat=519000):
	ny_mid = len(np.unique(Y))//2
	Xalt = np.zeros((nsamp,1))
	Hs_retrack = np.zeros((nsamp,1))
	Hs_std_disk = np.zeros((nsamp,len(radis2)))
	Hs_std_ring = np.zeros((nsamp,len(radis2),len(radis1)))
	waveforms=np.zeros((nsamp,len(edges_ref)-1))

	# Footprint definition 
	dx = X[1]-X[0]
	dy = Y[1]-Y[0]
	footprint=np.ones((2*nxa+1,2*nxa+1))
	[Xa,Ya]=np.meshgrid(dx*np.arange(-nxa,nxa+1), dy*np.arange(-nxa,nxa+1))
	dist_ground = (Xa**2+Ya**2)
	footprint[dist_ground > radi_wfm**2]=np.nan

	for isamp in range(nsamp):
		# define center point of satellite 
		ialt=(nxa+isamp*di).astype(int)
		Xalt[isamp] = X[ialt]
		for ir2 in range(len(radis2)):
			for ir1 in range(len(radis1)): 
				#clear_output(wait=True)
				#print(' i = ',i,'  - isamp = ',isamp,' / ',nsamp,' - ir1 = ',ir1,' / ',len(radis1),' - ir2 = ',ir2,' / ',len(radis2), '-----------')
				if radis1[ir1]>radis2[ir2]:
					Hs_std_ring[isamp,ir2,ir1] = np.nan
				else:
					# do calculus sqrt for ring 
					# --> define the corresponding footprint
					#print('hello')
					footprint_ring=np.ones((2*nxa+1,2*nxa+1))
					footprint_ring[dist_ground > radis2[ir2]**2]=np.nan
					footprint_ring[dist_ground < radis1[ir1]**2]=np.nan
					surf_ring=np.ones((2*nxa+1,2*nxa+1))
					surf_ring[:,:]=S1[ny_mid-nxa:ny_mid+nxa+1,ialt-nxa:ialt+nxa+1]*footprint_ring
					Hs_std_ring[isamp,ir2,ir1] = 4*np.nanstd(surf_ring.flatten())
					# end if else
			footprint_disk=np.ones((2*nxa+1,2*nxa+1))
			footprint_disk[dist_ground > radis2[ir2]**2]=np.nan
			surf_disk=np.ones((2*nxa+1,2*nxa+1))
			surf_disk[:,:]=S1[ny_mid-nxa:ny_mid+nxa+1,ialt-nxa:ialt+nxa+1]*footprint_disk
			Hs_std_disk[isamp,ir2] = 4*np.nanstd(surf_disk.flatten())
			# end for ir2
		surf=np.ones((2*nxa+1,2*nxa+1))
		surf[:,:]=S1[ny_mid-nxa:ny_mid+nxa+1,ialt-nxa:ialt+nxa+1]*footprint

		# to have distance to satellite = range
		r=np.sqrt(Xa**2+Ya**2+(alti_sat-surf)**2)-alti_sat+offset_range
		r[dist_ground > radi_wfm**2]=np.nan  # equivalent to multiplication by footprint

		counts,_=np.histogram(r,bins=edges_ref)
		Hs_retrack[isamp] = simple_retracking_process(counts,edges_ref,wfm_ref=wfm_ref,Hsm_ref=Hsm_ref) 
		waveforms[isamp,:]=counts
		# end for isamp
	return Hs_std_disk,Hs_std_ring,Hs_retrack,Xalt,waveforms
    

def process_investigateR_1surface(i,Z1,kX,kY):
#     random.seed(i)
#     np.random.default_rng(i)
	S1,X,Y= surface_from_Z1kxky(Z1,kX,kY,i)
	# plt.figure(figsize=(18,6))
	# im=plt.pcolormesh(X,Y,S1,cmap='seismic',norm = mcolors.Normalize(vmin=S1.min(), vmax=S1.max()))
	# plt.colorbar(im,label='$\zeta$ [m]')
	# _=plt.xlabel('X [m]')
	# _=plt.ylabel('Y [m]')
	# _=plt.title('Surface from spectrum')

	freq_satsampl=40 # freq for waveforms
	v_sat=7000 # satellite v
	alti_sat=519000 # altitude of satellite CFOSAT
	radi_wfm = 8000
	radis1 = np.arange(0,1000,250)
	radis2 = np.arange(250,1000,250)
	dist_ring_retrack = np.zeros((len(radis1),len(radis2)))
	dist_disk_retrack = np.zeros((len(radis2),1))

	# --- edges for range windows ------------
	dr = 0.375
	edges_max = 70
	kX0 = np.unique(kX)
	nx = Z1.shape[1]
	dx = 2*np.pi/((kX0[1] - kX0[0])*nx)
	nHs=251
	Hs_max = 25

	wfm_ref, Hsm_ref, edges_ref = generate_wvform_database(dr,nHs,edges_max=edges_max,Hs_max=Hs_max,offset=10)

	Apix = np.pi*2*alti_sat*dr / (dx**2)
	offset_range = 10

	nxa=np.floor(radi_wfm/dx).astype(int) # size of radius of footprint in pixel
	di=np.floor((v_sat/freq_satsampl)/dx).astype(int) # distance between footprint centers, in pixels (v_sat/freq_satsampl = dsitance in m)
	nsamp=np.floor((nx-2*nxa)/di).astype(int) # Nb of samples

	Hs_std_disk,Hs_std_ring,Hs_retrack,_,_ = fly_over_track_v05_radius(i,X,Y,S1,nsamp,nxa,di,wfm_ref,Hsm_ref,edges_ref,offset_range,radi_wfm,radis1,radis2,alti_sat=519000)
	print('shapes : Hs_std_disk = ',Hs_std_disk.shape,' - Hs_std_ring = ',Hs_std_ring.shape,' - Hs_retrack = ',Hs_retrack.shape)
	for ir1 in range(len(radis1)):
		radi1 = radis1[ir1]
		for ir2 in range(len(radis2)):
			# clear_output(wait=True)
			# print('PostProcess : i = ',i,'  - ir1 = ',ir1,' / ',len(radis1),' - ir2 = ',ir2,' / ',len(radis2), '-----------')
			radi2 = radis2[ir2]
			if radi1>=radi2:
				dist_ring_retrack[ir1,ir2]=np.nan
			else:
				dist_ring_retrack[ir1,ir2]=np.sum((np.squeeze(Hs_std_ring[:,ir2,ir1])-np.squeeze(Hs_retrack))**2)
				dist_disk_retrack[ir2]=np.sum((np.squeeze(Hs_std_disk[:,ir2])-np.squeeze(Hs_retrack))**2)

	return i,dist_disk_retrack,dist_ring_retrack,radis1,radis2,Hs_std_disk,Hs_std_ring,Hs_retrack

