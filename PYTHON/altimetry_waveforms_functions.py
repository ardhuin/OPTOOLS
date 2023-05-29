# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================================================
# === 0. Import Packages ===========================================================
# ==================================================================================
from wave_physics_functions import *
import scipy.special as sps # function erf
import numpy as np



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

##################################################################
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
    
def fly_over_track_only_retrack_2D(X,Y,S1,nsamp,nxa0,nxa,di,wfm_ref,Hsm_ref,edges_ref,dr,ze_ref,range_shift=10,\
                       alti_sat=519*1e3):
    # ----- nxa0 : is the first offset --------------
    # ----- nxa : chelton diam ----------------------
    nHs = len(Hsm_ref)
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
            counts,_=np.histogram(r,bins=(edges_ref-dr*0.5))
            Hs_retrack[isampx,isampy],ze_retrack[isampx,isampy],dist[isampx,isampy] = \
                                    simple_retracking_process_2params(counts,edges_ref,max_edg=max_edg,
                                                                    dx=dx,nHs=nHs,nze=nze, wfm_ref=wfm_ref,
                                                                    ze_ref=ze_ref,Hsm_ref=Hsm_ref,
                                                                    alti_sat=alti_sat)
            waveforms[isampx,isampy,:]=counts
#             print(Hs_retrack[isampx,isampy],ze_retrack[isampx,isampy],dist[isampx,isampy])

        
    return Hs_retrack,ze_retrack,Xalt,Yalt,waveforms,dist


