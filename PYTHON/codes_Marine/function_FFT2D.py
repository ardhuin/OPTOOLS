import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def FFT2D(arraya,nxa,nya,dx,dy,n,isplot=0):
# function to do a FFT 2D 
# nxa, nya : size of arraya
# dx,dy : resolution of arraya
# n : number of tiles in each directions ... 
# 
# Eta is PSD of 1st image (arraya) 
# Etb is PSD of 2st image (arraya) 
	mspec=n**2+(n-1)**2

	nxtile=int(np.floor(nxa/n)) 
	nytile=int(np.floor(nya/n))
	print('nxtile : ',nxtile)
	print('nytile : ',nytile)

	dkxtile=2*np.pi/(dx*nxtile)   
	dkytile=2*np.pi/(dy*nytile)

	shx = int(nxtile//2)
	shy = int(nytile//2)

	### --- prepare wavenumber vectors -------------------------
	# wavenumbers starting at zero
	kx=np.linspace(0,(nxtile-1)*dkxtile,nxtile)
	ky=np.linspace(0,(nytile-1)*dkytile,nytile)
	# Shift wavenumbers to have zero in the middle
	kxs=np.roll(kx,-shx)
	kys=np.roll(ky,-shy)

	# change the first half to have negative wavenumber
	kxs[:shx+1]=kxs[:shx+1]-kx[-1]-dkxtile
	kys[:shy+1]=kys[:shy+1]-ky[-1]-dkytile

	kx2,ky2 = np.meshgrid(kxs,kys)
	if isplot:
		X = np.arange(0,nxa*dx,dx) # from 0 to (nx-1)*dx with a dx step
		Y = np.arange(0,nya*dy,dy)

	### --- prepare Hanning windows for performing fft and associated normalization ------------------------

	hanningx=(0.5 * (1-np.cos(2*np.pi*np.linspace(0,nxtile-1,nxtile)/(nxtile-1))))
	hanningy=(0.5 * (1-np.cos(2*np.pi*np.linspace(0,nytile-1,nytile)/(nytile-1))))
	# 2D Hanning window
	hanningxy=np.atleast_2d(hanningx)*np.atleast_2d(hanningy).T 

	wc2x=1/np.mean(hanningx**2);                              # window correction factor
	wc2y=1/np.mean(hanningy**2);                              # window correction factor

	normalization = (wc2x*wc2y)/(dkxtile*dkytile)

	### --- Initialize Eta = mean spectrum over tiles ---------------------

	Eta=np.zeros((nytile,nxtile))
	Eta_all=np.zeros((nytile,nxtile,mspec))
	if isplot:
		fig1,ax1=plt.subplots(figsize=(12,6))
		ax1.pcolormesh(X,Y,arraya)
		colors = plt.cm.seismic(np.linspace(0,1,mspec))

    ### --- Calculate spectrum for each tiles ----------------------------
	for m in range(mspec):
		### 1. Selection of tile ------------------------------
		if (m<n**2):
			i1=int(np.floor(m/n)+1)
			i2=int(m+1-(i1-1)*n)

			ix1 = nxtile*(i1-1)
			ix2 = nxtile*i1-1
			iy1 = nytile*(i2-1)
			iy2 = nytile*i2-1

			#                 array1=double(arraya(nx*(i1-1)+1:nx*i1,ny*(i2-1)+1:ny*i2));
			#        Select a 'tile' i.e. part of the surface : main loop ---------

			array1=np.double(arraya[iy1:iy2+1,ix1:ix2+1])
			if isplot:
				ax1.plot(X[[ix1,ix1,ix2,ix2,ix1]],Y[[iy1,iy2,iy2,iy1,iy1]],'-',color=colors[m],linewidth=2)
		else:
			#        # -- Select a 'tile' overlapping (50%) the main tiles ---
			#        %%%%%%%%%%%%%%% now shifted 50% , like Welch %%%%%%%%%%%%%%
			i1=int(np.floor((m-n**2)/(n-1))+1)
			i2=int(m+1-n**2-(i1-1)*(n-1))


			ix1 = nxtile*(i1-1)+shx 
			ix2 = nxtile*i1+shx-1
			iy1 = nytile*(i2-1)+shy
			iy2 = nytile*i2+shy-1

			array1=np.double(arraya[iy1:iy2+1,ix1:ix2+1])
			if isplot:
				ax1.plot(X[[ix1,ix1,ix2,ix2,ix1]],Y[[iy1,iy2,iy2,iy1,iy1]],'-',color=colors[m],linewidth=2)

		### 2. Work over 1 tile ------------------------------ 
		tile_centered=array1-np.mean(array1.flatten())
		tile_by_windows = (tile_centered)*hanningxy

		# 
		tileFFT = np.fft.fft(tile_by_windows,norm="forward")#/(nx*ny)
		tileFFT_shift = np.roll(tileFFT,(-shy,-shx),axis=(0,1))
		Eta_all[:,:,m] = (abs(tileFFT_shift)**2) *normalization
		Eta[:,:] = Eta[:,:] + (abs(tileFFT_shift)**2) *normalization #          % sum of spectra for all tiles

	return Eta/mspec,Eta_all,kx2,ky2

