# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================================================
# === 0. Import Packages ===========================================================
# ==================================================================================
import numpy as np

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt



def FFT2D_one_array(arraya,dx,dy,n,isplot=0):
# Welch-based 2D spectral analysis
# arraya: input array 
# dx,dy : resolution of arraya for dimensions 0,1
# n : number of tiles in each directions ... 
# 
# Eta is PSD of 1st image (arraya) 
    [nxa,nya]=np.shape(arraya)
    mspec=n**2+(n-1)**2
    nxtile=int(np.floor(nxa/n))
    nytile=int(np.floor(nya/n))

    dkxtile=1/(dx*nxtile)   
    dkytile=1/(dy*nytile)

    shx = int(nxtile//2)   # OK if nxtile is even number
    shy = int(nytile//2)

    ### --- prepare wavenumber vectors -------------------------
    kx=np.fft.fftshift(np.fft.fftfreq(nxtile, dx)) # wavenumber in cycles / m
    ky=np.fft.fftshift(np.fft.fftfreq(nytile, dy)) # wavenumber in cycles / m
    kx2,ky2 = np.meshgrid(kx,ky, indexing='ij')
    if isplot:
        X = np.arange(0,nxa*dx,dx) # from 0 to (nx-1)*dx with a dx step
        Y = np.arange(0,nya*dy,dy)

    ### --- prepare Hanning windows for performing fft and associated normalization ------------------------

    hanningx=(0.5 * (1-np.cos(2*np.pi*np.linspace(0,nxtile-1,nxtile)/(nxtile-1))))
    hanningy=(0.5 * (1-np.cos(2*np.pi*np.linspace(0,nytile-1,nytile)/(nytile-1))))
    # 2D Hanning window
    #hanningxy=np.atleast_2d(hanningx)*np.atleast_2d(hanningy).T 
    hanningxy=np.atleast_2d(hanningy)*np.atleast_2d(hanningx).T 

    wc2x=1/np.mean(hanningx**2);                              # window correction factor
    wc2y=1/np.mean(hanningy**2);                              # window correction factor

    normalization = (wc2x*wc2y)/(dkxtile*dkytile)

    ### --- Initialize Eta = mean spectrum over tiles ---------------------

    Eta=np.zeros((nxtile,nytile))
    Eta_all=np.zeros((nxtile,nytile,mspec))
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

            array1=np.double(arraya[ix1:ix2+1,iy1:iy2+1])
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

            array1=np.double(arraya[ix1:ix2+1,iy1:iy2+1])
            if isplot:
                ax1.plot(X[[ix1,ix1,ix2,ix2,ix1]],Y[[iy1,iy2,iy2,iy1,iy1]],'-',color=colors[m],linewidth=2)

        ### 2. Work over 1 tile ------------------------------ 
        tile_centered=array1-np.mean(array1.flatten())
        tile_by_windows = (tile_centered)*hanningxy

        # 
        tileFFT = np.fft.fft2(tile_by_windows,norm="forward")
        tileFFT_shift = np.fft.fftshift(tileFFT)
        Eta_all[:,:,m] = (abs(tileFFT_shift)**2) *normalization
        Eta[:,:] = Eta[:,:] + (abs(tileFFT_shift)**2) *normalization #          % sum of spectra for all tiles

    return Eta/mspec,Eta_all,kx2,ky2,dkxtile,dkytile



def FFT2D_two_arrays(arraya,arrayb,dx,dy,n,isplot=0):
# Welch-based 2D spectral analysis
# nxa, nya : size of arraya
# dx,dy : resolution of arraya
# n : number of tiles in each directions ... 
# 
# Eta is PSD of 1st image (arraya) 
# Etb is PSD of 2st image (arraya) 
    [nxa,nya]=np.shape(arraya)
    mspec=n**2+(n-1)**2
    nxtile=int(np.floor(nxa/n))
    nytile=int(np.floor(nya/n))

    dkxtile=1/(dx*nxtile)   
    dkytile=1/(dy*nytile)

    shx = int(nxtile//2)   # OK if nxtile is even number
    shy = int(nytile//2)

    ### --- prepare wavenumber vectors -------------------------
    # wavenumbers starting at zero
    kx=np.fft.fftshift(np.fft.fftfreq(nxtile, dx)) # wavenumber in cycles / m
    ky=np.fft.fftshift(np.fft.fftfreq(nytile, dy)) # wavenumber in cycles / m
    kx2,ky2 = np.meshgrid(kx,ky, indexing='ij')

    if isplot:
        X = np.arange(0,nxa*dx,dx) # from 0 to (nx-1)*dx with a dx step
        Y = np.arange(0,nya*dy,dy)

    ### --- prepare Hanning windows for performing fft and associated normalization ------------------------

    hanningx=(0.5 * (1-np.cos(2*np.pi*np.linspace(0,nxtile-1,nxtile)/(nxtile-1))))
    hanningy=(0.5 * (1-np.cos(2*np.pi*np.linspace(0,nytile-1,nytile)/(nytile-1))))
    # 2D Hanning window
    #hanningxy=np.atleast_2d(hanningx)*np.atleast_2d(hanningy).T 
    hanningxy=np.atleast_2d(hanningy)*np.atleast_2d(hanningx).T 

    wc2x=1/np.mean(hanningx**2);                              # window correction factor
    wc2y=1/np.mean(hanningy**2);                              # window correction factor

    normalization = (wc2x*wc2y)/(dkxtile*dkytile)

    ### --- Initialize Eta = mean spectrum over tiles ---------------------

    Eta=np.zeros((nxtile,nytile))
    Etb=np.zeros((nxtile,nytile))
    phase=np.zeros((nxtile,nytile))
    #Eta_all=np.zeros((nxtile,nytile,mspec))
    phases=np.zeros((nxtile,nytile,mspec))
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

            array1=np.double(arraya[ix1:ix2+1,iy1:iy2+1])
            array2=np.double(arrayb[ix1:ix2+1,iy1:iy2+1])
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

            array1=np.double(arraya[ix1:ix2+1,iy1:iy2+1])
            array2=np.double(arrayb[ix1:ix2+1,iy1:iy2+1])
            if isplot:
                ax1.plot(X[[ix1,ix1,ix2,ix2,ix1]],Y[[iy1,iy2,iy2,iy1,iy1]],'-',color=colors[m],linewidth=2)

        ### 2. Work over 1 tile ------------------------------ 
        tile_centered=array1-np.mean(array1.flatten())
        tile_by_windows = (tile_centered)*hanningxy

        # 
        tileFFT1 = np.fft.fft2(tile_by_windows,norm="forward")
        tileFFT1_shift = np.fft.fftshift(tileFFT1)
        #Eta_all[:,:,m] = (abs(tileFFT1_shift)**2) *normalization
        Eta[:,:] = Eta[:,:] + (abs(tileFFT1_shift)**2) *normalization #          % sum of spectra for all tiles

        tile_centered=array2-np.mean(array2.flatten())
        tile_by_windows = (tile_centered)*hanningxy

        tileFFT2 = np.fft.fft2(tile_by_windows,norm="forward")
        tileFFT2_shift = np.fft.fftshift(tileFFT2)#
        #Etb_all[:,:,m] = (abs(tileFFT2_shift)**2) *normalization
        Etb[:,:] = Etb[:,:] + (abs(tileFFT2_shift)**2) *normalization #          % sum of spectra for all tiles

        phase=phase+(tileFFT2_shift*np.conj(tileFFT1_shift))*normalization
        phases[:,:,m]=tileFFT2_shift*np.conj(tileFFT1_shift)/(abs(tileFFT2_shift)*abs(tileFFT1_shift)); 

# Now works with averaged spectra
   
    Eta=Eta/mspec
    Etb=Etb/mspec
    coh=abs((phase/mspec)**2)/(Eta*Etb)      # spectral coherence
    ang=np.angle(phase)
    crosr=np.real(phase)/mspec
    angstd=np.std(np.angle(phases),axis=2)

    return Eta,Etb,ang,angstd,coh,crosr,phases,kx2,ky2,dkxtile,dkytile
