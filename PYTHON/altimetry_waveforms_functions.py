# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================================================
# === 0. Import Packages ===========================================================
# ==================================================================================
from wave_physics_functions import *
import scipy.special as sps # function erf
import numpy as np
import scipy
from scipy import special
from scipy.optimize import minimize

def calc_footprint_diam(Hs,pulse_width = 1/(320*1e6),Rorbit=519*1e3,Rearth = 6370*1e3):
    clight= 299792458
    Airemax_div_pi = Rorbit*(clight*pulse_width + 2 * Hs)/(1+(Rorbit/Rearth))
    return 2*np.sqrt(Airemax_div_pi)


def  waveform_erf(incognita,data)  :
     """
     returns the least-square distance between the waveform data[0] and the simplest erf waveform
     only one unknown parameter: Hs 
     """
     ydata =data[0] # Waveform
     xdata =data[1] # ranges in m
     # wfm[k,:]=0.5+0.5*sps.erf((edges[:-1] - offset) / (0.25*np.sqrt(2)*Hsm[k]))
     fff = 0.5 * (  1+scipy.special.erf( xdata /(np.sqrt(2)*incognita[0])  ) ) 
     cy= (   ((ydata - fff) **2)).sum()
     return cy

def  waveform_brown_LS(incognita,data)  :
     """
     returns the least-square distance between the waveform data[0] and the theoretical 
     Brown-Hayne functional form, The unknown parameters in this version (17 Dec 2013) are Epoch, Sigma and Amplitude, where 
     sigma=( sqrt( (incognita(2)/(2*0.3)) ^2+SigmaP^2) ) is the rising time of the leading edge
     
     For the explanation of the terms in the equation, please check "Coastal Altimetry" Book
     
     """
     
     ydata =data[0] #Waveform coefficients
     Gamma =data[1]
     Zeta  =data[2]
     xdata =data[3]  #Epoch
     SigmaP=data[4]
     c_xi  =data[5]  #Term related to the slope of the trailing edge
     weights=data[6]  #Weights to apply to the residuals
         
     fff = ( incognita[2]/2*np.exp((-4/Gamma)*(np.sin(Zeta))**2) \
     * np.exp (-  c_xi*( (xdata-incognita[0])-c_xi*incognita[1]**2/2) ) \
     *   (  1+scipy.special.erf( ((xdata-incognita[0])-c_xi*incognita[1]**2)/((np.sqrt(2)*incognita[1]))  ) ) \
     )
    
     cy= (   weights *  ((ydata - fff) **2)).sum()
     
     return cy


def  waveform_brown_ML(incognita,data)  :
     """
     returns the ML distance between the waveform data[0] and the theoretical 
     Brown-Hayne functional form, The unknown parameters in this version (17 Dec 2013) are Epoch, Sigma and Amplitude, where 
     sigma=( sqrt( (incognita(2)/(2*0.3)) ^2+SigmaP^2) ) is the rising time of the leading edge
     
     For the explanation of the terms in the equation, please check "Coastal Altimetry" Book
     
     """
     
     ydata =data[0] #Waveform coefficients
     Gamma =data[1]
     Zeta  =data[2]
     xdata =data[3]  #Epoch
     SigmaP=data[4]
     c_xi  =data[5]  #Term related to the slope of the trailing edge
     weights=data[6]  #Weights to apply to the residuals
         
     fff = ( incognita[2]/2*np.exp((-4/Gamma)*(np.sin(Zeta))**2) \
     * np.exp (-  c_xi*( (xdata-incognita[0])-c_xi*incognita[1]**2/2) ) \
     *   (  1+scipy.special.erf( ((xdata-incognita[0])-c_xi*incognita[1]**2)/((np.sqrt(2)*incognita[1]))  ) ) \
     )
     ratio = ydata/fff 
     cy= ( ratio - np.log(ratio)).sum()
     
     return cy


############# using scipy.minimize , as in WHALES #################

def retracking_NM1D(wfm,edges,max_edg=25,nHs=251,alti_sat=519*1e3,\
                                  dx=10,offset=10,wfm_ref=None,Hsm_ref=None,ispolyfit=0,isepoch=0):
    dr = edges[1]-edges[0]
    Apix = np.pi*2*alti_sat*dr / (dx**2) # The area of a ring, in terms of pixels 
    wfn=wfm/Apix   # normalization: if wfm is area histogram then wfn should be in [0 1]
    
    xopt = minimize(waveform_erf, incognita, args=((wfn,edges),) ,method='Nelder-Mead',options={'disp': False})
    x=xopt.x
    if xopt.success == True:
       Hs=x[0]
       dist=waveform_erf(x,((wfn),(edges)))
       
    return Hs, dist


#     incognita=initial_conditions
        
#        c=3.0*(10**8) #Light speed
#        H=altitude
#        Ri=6378.1363*(10**3) #Earth radius

#        Gamma=0.5 * (1/math.log(2))*np.sin(Theta)*np.sin(Theta) # antenna beamwidth parameter
     
#        b_xi = np.cos (2*Zeta) - ((np.sin(2*Zeta))**2)/Gamma
#        a=( (4/Gamma)*(c/H) * 1/(1+H/Ri))
#        c_xi=b_xi* ( (4/Gamma)*(c/H) * 1/(1+H/Ri))
    
#        a=a/1000000000 #/ns
#        c_xi=c_xi/1000000000 #1/ns

############# 1D erf waveforms ####################################

def simple_retracking_process_v01(wfm,edges,max_edg=25,nHs=251,alti_sat=519*1e3,\
                                  dx=10,offset=10,wfm_ref=None,Hsm_ref=None,ispolyfit=0,isepoch=0):
    dr = edges[1]-edges[0]
    Apix = np.pi*2*alti_sat*dr / (dx**2) # The area of a ring, in terms of pixels 
    if isepoch:
        max_asymptote = np.mean(wfm[max_edg-2:max_edg+4])
        semiH = np.argmin(np.abs(wfm[0:max_edg]-(max_asymptote/2)))
        # offset pix
        offset_pix = int(offset/dr)
        testwfm0 = np.zeros(len(wfm))

        delt_off = int(np.abs(semiH - offset_pix))

        if offset_pix < semiH:
            testwfm0[:-delt_off]=wfm[delt_off:]
        elif offset_pix > semiH:
            testwfm0[delt_off:]=wfm[:-delt_off]
        else:
            testwfm0=wfm
        
        testwf=np.broadcast_to(testwfm0,(nHs,len(testwfm0)))
        dist=np.sum((Apix*wfm_ref[:,0:max_edg]-testwf[:,0:max_edg])**2,axis=1)
    else:
        testwf=np.broadcast_to(wfm,(nHs,len(wfm)))
        dist=np.sum((Apix*wfm_ref[:,0:max_edg]-testwf[:,0:max_edg])**2,axis=1)

    if ispolyfit:
        p = np.polyfit(Hsm_ref,dist,2)
        Hs = -p[1]/(2*p[0])
    else:
        Imin=np.nanargmin(dist)
        Hs = Hsm_ref[Imin]

    return Hs, dist

def generate_wvform_database(nHs,dr=None,ne=None,bandwidth=320*1e6,\
                             edges_max=25,Hs_max=25,offset=10):
    if (dr is None)&(ne is None):
        clight = 299792458
        dr = clight * 1/(2*bandwidth)
        edges = np.arange(0,edges_max+dr,dr) 
    elif (dr is None)&(ne is not None):
        edges=np.linspace(0,edges_max,ne)
    elif (ne is None)&(dr is not None):
        edges = np.arange(0,edges_max+dr,dr) 
    dr=edges[1]-edges[0]
    ne = len(edges)               
    Hsm=np.linspace(0,Hs_max,nHs)
    wfm=np.zeros((nHs,ne-1))

    for k in range(nHs):
#         wfm[k,:]=0.5+0.5*sps.erf((edges[:-1]+0.5*dr-offset) / (0.25*np.sqrt(2)*Hsm[k]))
        wfm[k,:]=0.5+0.5*sps.erf((edges[:-1] - offset) / (0.25*np.sqrt(2)*Hsm[k]))

    return wfm, Hsm, edges,dr   
    
def fly_over_track_only_retrack(X,Y,S1,nsamp,nxa0,nxa,di,wfm_ref,Hsm_ref,edges_ref,range_shift=10,\
                       alti_sat=519000,isepoch = 0):
    # ----- nxa0 : is the first offset --------------
    # ----- nxa : chelton diam ----------------------
    nHs = len(Hsm_ref)
    Xalt = np.zeros((nsamp))
    Yalt = np.zeros((nsamp-1))

    Hs_retrack = np.zeros((nsamp,nsamp-1))
    waveforms=np.zeros((nsamp,nsamp-1,len(edges_ref)-1))
    dist=np.zeros((nsamp,nsamp-1,nHs))
                    
    dx = X[1]-X[0]
    dy = Y[1]-Y[0]
      
    # --- Footprint definition For std(surface) --------------------
    [Xa0,Ya0]=np.meshgrid(dx*np.arange(-nxa,nxa+1), dy*np.arange(-nxa,nxa+1))
    dist_ground = (Xa0**2+Ya0**2)
    
    radi0 = nxa*dx
    rlim = np.sqrt((radi0/2)**2+(alti_sat)**2)-alti_sat+range_shift
    max_edg=np.argmax(edges_ref[edges_ref<=rlim])
    
    for isampx in range(nsamp):
        print('------------ ',isampx,' over ',nsamp-1,' ------------ ')
        for isampy in range(nsamp-1):
            ialtx=(nxa0+isampx*di).astype(int)
            ialty=(nxa0+isampy*di).astype(int)
            Xalt[isampx] = X[ialtx]
            Yalt[isampy] = Y[ialty]
             
            # --- get surface extract for altimeter ---------------------------
            surf1 = S1[ialty-nxa:ialty+nxa+1,ialtx-nxa:ialtx+nxa+1]
            # --- to have distance to satellite = range -------------------
            r=np.sqrt(Xa0**2+Ya0**2+(alti_sat-surf1)**2)-alti_sat+range_shift
            counts,_=np.histogram(r,bins=edges_ref)
            Hs_retrack[isampx,isampy],dist[isampx,isampy,:] = simple_retracking_process_v01(counts,edges_ref,max_edg=max_edg,
                                                                            dx=dx,nHs=nHs, wfm_ref=wfm_ref,
                                                                            offset = range_shift,Hsm_ref=Hsm_ref,
                                                                            alti_sat=alti_sat,isepoch=isepoch)
            waveforms[isampx,isampy,:]=counts

        
    return Hs_retrack,Xalt,Yalt,waveforms,dist

	

def fly_over_track_v0(X,Y,S1,nsamp,nxa,di,wfm_ref,Hsm_ref,edges_ref,radi,radi1,radi2,alti_sat,range_shift):
	# radi = 4000     # radius used to compute waveform
	#radi1 = 900     # inner radius for Hs average
	#radi2 = 1200    # outer radius for Hs average

	ny_mid = len(np.unique(Y))//2
	Xalt = np.zeros((nsamp,1))
	Hs_retrack = np.zeros((nsamp,1))
	dist       = np.zeros((nsamp,1))
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
#	   Hs_retrack[isamp]             = simple_retracking_process   (counts,edges_ref,wfm_ref=wfm_ref,Hsm_ref=Hsm_ref,ispolyfit=0) 
           Hs_retrack[isamp],dist[isamp] =simple_retracking_process_v01(wfm,edges,max_edg=25,nHs=251,alti_sat=519*1e3,\
                                  dx=10,offset=10,wfm_ref=None,Hsm_ref=None,ispolyfit=0,isepoch=0)
           waveforms[isamp,:]=counts

	return Hs_std,Hs_stdbis,Hs_std2,Hs_retrack,Xalt,waveforms,surf1,footprint1


############# 2D erf waveforms ####################################"

def simple_retracking_process_2params(wfm,edges,max_edg=25,nHs=251,nze=250,alti_sat=519*1e3,\
                                  dx=10,wfm_ref=None,Hsm_ref=None,ze_ref=None):
    dr = edges[1]-edges[0]
    Apix = np.pi*2*alti_sat*dr / (dx**2) # The area of a ring, in terms of pixels 
    
    testwf=np.broadcast_to(wfm,(nHs,nze,len(wfm)))
#     print('testwf shape = ',testwf.shape,' , wfm shape = ',wfm_ref.shape)
    dist=np.sum((Apix*wfm_ref[:,:,0:max_edg]-testwf[:,:,0:max_edg])**2,axis=-1)
    
#     print(np.sum(np.isnan(dist)),' / ',dist.size)
    
    hmin,zemin = np.unravel_index(np.nanargmin(dist,axis=None),dist.shape)
    
    Hs = Hsm_ref[hmin]
    ze = ze_ref[zemin]

    return Hs, ze, dist[hmin,zemin]

def generate_wvform_database_2D(nHs,nze,dr=None,ne=None,bandwidth=320*1e6,\
                             edges_max=25,Hs_max=25,ze_max=1,offset=10):
    if (dr is None)&(ne is None):
        clight = 299792458
        dr = clight * 1/(2*bandwidth)
        edges = np.arange(0,edges_max+dr,dr) 
    elif (dr is None)&(ne is not None):
        edges=np.linspace(0,edges_max,ne)
    elif (ne is None)&(dr is not None):
        edges = np.arange(0,edges_max+dr,dr) 
    dr=edges[1]-edges[0]
    ne = len(edges)               
    Hsm=np.linspace(0,Hs_max,nHs)
    zem = np.linspace(-ze_max,ze_max,nze)
    wfm=np.zeros((nHs,nze,ne-1))

    for k in range(nHs):
        for ize, ze in enumerate(zem):
#             wfm[k,ize,:]=0.5+0.5*sps.erf((edges[:-1]+ze+0.5*dr-offset) / (0.25*np.sqrt(2)*Hsm[k]))
            wfm[k,ize,:]=0.5+0.5*sps.erf(((edges[:-1]+ze) - offset) / (0.25*np.sqrt(2)*Hsm[k]))

    return wfm, Hsm, zem, edges, dr   

############################################
def fly_over_track_only_retrack_2D(X,Y,S1,nsamp,nxa0,nxa,di,wfm_ref,Hsm_ref,edges_ref,ze_ref,range_shift=10,\
                       alti_sat=519*1e3):
    # ----- nxa0 : is the first offset --------------
    # ----- nxa : chelton diam ----------------------
    nHs = len(Hsm_ref)
    nze = len(ze_ref)
    Xalt = np.zeros((nsamp))
    Yalt = np.zeros((nsamp-1))

    Hs_retrack = np.zeros((nsamp,nsamp-1))
    ze_retrack = np.zeros((nsamp,nsamp-1))
    waveforms=np.zeros((nsamp,nsamp-1,len(edges_ref)-1))
    dist=np.zeros((nsamp,nsamp-1))
                    
    dx = X[1]-X[0]
    dy = Y[1]-Y[0]
      
    # --- Footprint definition For std(surface) --------------------
    [Xa0,Ya0]=np.meshgrid(dx*np.arange(-nxa,nxa+1), dy*np.arange(-nxa,nxa+1))
    dist_ground = (Xa0**2+Ya0**2)
    
    radi0 = nxa*dx
    rlim = np.sqrt((radi0/2)**2+(alti_sat)**2)-alti_sat+range_shift
    max_edg=np.argmax(edges_ref[edges_ref<=rlim])
    
    for isampx in range(nsamp):
        print('------------ ',isampx,' over ',nsamp-1,' ------------ ')
        for isampy in range(nsamp-1):
            ialtx=(nxa0+isampx*di).astype(int)
            ialty=(nxa0+isampy*di).astype(int)
            Xalt[isampx] = X[ialtx]
            Yalt[isampy] = Y[ialty]
             
            # --- get surface extract for altimeter ---------------------------
            surf1 = S1[ialty-nxa:ialty+nxa+1,ialtx-nxa:ialtx+nxa+1]
            # --- to have distance to satellite = range -------------------
            r=np.sqrt(Xa0**2+Ya0**2+(alti_sat-surf1)**2)-alti_sat+range_shift
            # --- histogram counts the number of data points between edges ...
            #     modified by FA to center the bins on the edges.  
            dr2 = 0.5*(edges_ref[1]-edges_ref[2])
            counts,_=np.histogram(r,bins=edges_ref+dr2)
            Hs_retrack[isampx,isampy],ze_retrack[isampx,isampy],dist[isampx,isampy] = \
                                    simple_retracking_process_2params(counts,edges_ref,max_edg=max_edg,
                                                                    dx=dx,nHs=nHs,nze=nze, wfm_ref=wfm_ref,
                                                                    ze_ref=ze_ref,Hsm_ref=Hsm_ref,
                                                                    alti_sat=alti_sat)
            waveforms[isampx,isampy,:]=counts
#             print(Hs_retrack[isampx,isampy],ze_retrack[isampx,isampy],dist[isampx,isampy])

        
    return Hs_retrack,ze_retrack,Xalt,Yalt,waveforms,dist


