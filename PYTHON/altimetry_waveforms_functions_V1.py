# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================================================
# === 0. Import Packages ===========================================================
# ==================================================================================
from wave_physics_functions import *
import scipy.special as sps # function erf
import numpy as np
import scipy
from scipy.optimize import minimize

def calc_footprint_diam(Hs,pulse_width = 1/(320*1e6),Rorbit=519*1e3,Rearth = 6370*1e3):
    clight= 299792458
    Airemax_div_pi = Rorbit*(clight*pulse_width + 2 * Hs)/(1+(Rorbit/Rearth))
    return 2*np.sqrt(Airemax_div_pi)


######################  Defines waveform theoretical models: most simple, 2 parameter erf 
def  wf_erf2D_eval(xdata,incognita,offset)  :
     print('size xdata:',np.shape(xdata))
     fff = 0.5 * (  1+scipy.special.erf( (xdata-incognita[0]-offset)/(0.25*np.sqrt(2)*incognita[1])  ) ) 
     return fff

def  wf_erf2D(incognita,data)  :
     """
     returns the least-square distance between the waveform data[0] and the simplest erf waveform
     two unknown parameter: (epoch,Hs)  both in meters
     """
     ydata =data[0] # Waveform
     xdata =data[1] # ranges in m
     hsat  =data[2]
     offset=data[5]
     costfun=data[6]
     #wfm[k,ize,:]=0.5+0.5*sps.erf(((edges[:-1]+ze) - offset) / (0.25*np.sqrt(2)*Hsm[k]))

     fff = 0.5 * (  1+sps.erf( (xdata+incognita[0]-offset)/(0.25*np.sqrt(2)*incognita[1])  ) ) 
     if costfun=='LS':
        cy= (   ((ydata - fff) **2)).sum()
     else:
        ratio = np.divide(ydata+1.e-5,fff+1.e-5) 
        cy= ( ratio - np.log(ratio)).sum()

     return cy

######################  Defines waveform theoretical models: brown 2D without mispoiting
def  wf_bro2D_eval(xdata,incognita,offset)  :
     print('size xdata:',np.shape(xdata))
     fff = 0.5 * (  1+scipy.special.erf( (xdata-incognita[0]-offset)/(0.25*np.sqrt(2)*incognita[1])  ) ) 
     return fff

def  wf_bro2D(incognita,data)  :
     """
     returns the least-square distance between the waveform data[0] and the simplest erf waveform
     two unknown parameter: (epoch,Hs)  both in meters
     """
     ydata =data[0] # Waveform
     xdata =data[1] # ranges in m
     hsat  =data[2]
     offset=data[3]
     costfun=data[4]
     #wfm[k,ize,:]=0.5+0.5*sps.erf(((edges[:-1]+ze) - offset) / (0.25*np.sqrt(2)*Hsm[k]))

     fff = 0.5 * (  1+sps.erf( (xdata+incognita[0]-offset)/(0.25*np.sqrt(2)*incognita[1])  ) ) 
     if costfun=='LS':
        cy= (   ((ydata - fff) **2)).sum()
     else:
        ratio = np.divide(ydata,fff) 
        cy= ( ratio - np.log(ratio)).sum()

     return cy

######################  Defines waveform theoretical models: brown from WHALES code 
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


############# A 2-parameter retracker using scipy.minimize , as in WHALES #################

def retracking_NM(wfm,discrete_ranges,wf_fun,alti_sat=519*1e3,Theta=1.,tau=1/(320*1e6),offset=10,costfun='LS',method='Nelder-Mead'):
    incognita=np.array([0.,10.0]) # initial conditions: could use previous waveform ... 
    xopt = minimize(wf_fun, incognita, args=((wfm,discrete_ranges,alti_sat,Theta,tau,offset,costfun),),\
                    bounds=((-4,4),(0.0,40)),method=method,options={'disp': False})
    x=xopt.x
    if xopt.success == True:
       Hs=x[1]
       epoch=x[0]
       dist=wf_fun(x,((wfm),(discrete_ranges),alti_sat,Theta,tau,offset,costfun))
    else:
       Hs=None
       epoch=None
       dist=None   
    return Hs, epoch, dist

############# A 2-parameter pyramid grid search #################
def retracking_pyramid3(wfm,discrete_ranges,wf_fun,alti_sat=519.e3,Theta=1.,tau=1./(320.e6),offset=10,costfun='LS'):
    nsteps=10
    a0=0.
    a1=20.
    b0=2.
    b1=10.
    for istep in range(nsteps):
        dist=np.zeros((5,5))
        for i0 in range(5):
           for i1 in range(5):
# Note that 9 out of 25 have already been computed at the previous step ... 
              incognita=np.array([a0+(i0-2)*b0,a1+(i1-2)*b1])
              dist[i0,i1]=eval(wf_fun)(incognita,(wfm,discrete_ranges,alti_sat,Theta,tau,offset,costfun)) 
              #print('      inds:',i0,i1,incognita,dist[i0,i1])       
    
        i0min,i1min = np.unravel_index(np.nanargmin(dist,axis=None),dist.shape)
        epoch=a0+(i0min-2)*b0              
        Hs   =a1+(i1min-2)*b1  
        dmin =dist[i0min,i1min] 
# Update of search interval ... 
        a0=a0+(i0min-2)*b0
        a1=a1+(i1min-2)*b1
        b0=b0/2.
        b1=b1/2.
        #print('step:',istep,epoch,Hs,dmin)       
    return Hs, epoch, dmin






############# 1D erf waveforms ####################################

def simple_retracking_process_v01(wfm,edges,max_edg=25,nHs=251,\
                                  offset=10,wfm_ref=None,Hsm_ref=None,ispolyfit=0,isepoch=0):
    dr = edges[1]-edges[0]
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
        dist=np.sum((wfm_ref[:,0:max_edg]-testwf[:,0:max_edg])**2,axis=1)
    else:
        testwf=np.broadcast_to(wfm,(nHs,len(wfm)))
        dist=np.sum((wfm_ref[:,0:max_edg]-testwf[:,0:max_edg])**2,axis=1)

    if ispolyfit:
        p = np.polyfit(Hsm_ref,dist,2)
        Hs = -p[1]/(2*p[0])
    else:
        Imin=np.nanargmin(dist)
        Hs = Hsm_ref[Imin]

    return Hs, Imin, dist

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

##################################
def retrack_waveforms(waveforms,discrete_ranges,max_range_fit,wfm_ref=None,Hsm_ref=None,ze_ref=None,\
                      min_method='bruteforce',wf_model='erf2D',costfun='LS',alti_sat=None,Theta=None,tau=None,range_offset=None):
    nxw,nyw,nr=np.shape(waveforms)
    print('size of waveforms:',nxw,nyw,nr)
    Hs_r=np.zeros((nxw,nyw))
    ze_r=np.zeros((nxw,nyw))
    di_r=np.zeros((nxw,nyw))
    if 'costfun'=='ML':
      min_range_fit=12
    else:
      min_range_fit=0

    
    for ix in range(nxw):
        print('Retracking waveforms',ix,' out of ',nxw,' ------------ ')

        for iy in range(nyw):
            wfm=waveforms[ix,iy,:]
            if min_method == 'bruteforce':
               Hs_r[ix,iy],ze_r[ix,iy],di_r[ix,iy]=simple_retracking_process_2params(wfm,\
                                  max_edg=max_range_fit,nHs=250,nze=251,wfm_ref=wfm_ref,Hsm_ref=Hsm_ref,ze_ref=ze_ref,costfun=costfun)
            elif min_method in [ 'Nelder-Mead','Newton-CG']:
               Hs_r[ix,iy],ze_r[ix,iy],di_r[ix,iy]=retracking_NM(wfm[min_range_fit:max_range_fit],\
                                                              discrete_ranges[min_range_fit:max_range_fit],wf_model,\
                           alti_sat=alti_sat,Theta=Theta,tau=tau,offset=range_offset,method=min_method,costfun=costfun)
            elif min_method == 'pyramid3':
               Hs_r[ix,iy],ze_r[ix,iy],di_r[ix,iy]=retracking_pyramid3(wfm[min_range_fit:max_range_fit],\
                                                              discrete_ranges[min_range_fit:max_range_fit],wf_model,\
                           alti_sat=alti_sat,Theta=Theta,tau=tau,offset=range_offset,costfun=costfun)
    return Hs_r,ze_r,di_r
    
##################################
def fly_over_track_only_retrack(X,Y,S1,nsamp,nxa0,nxa,di,wfm_ref,Hsm_ref,edges_ref,range_shift=10,\
                       alti_sat=519000,isepoch = 0):
    # ----- nxa0 : is the first offset --------------
    # ----- nxa : chelton diam ----------------------
    nHs = len(Hsm_ref)
    Xalt = np.zeros((nsamp))
    Yalt = np.zeros((nsamp-1))

    Hs_retrack = np.zeros((nsamp,nsamp-1))
    ind_retrack = np.zeros((nsamp,nsamp-1))
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
    
    dr = edges_ref[1]-edges_ref[0]
    Apix = np.pi*2*alti_sat*dr / (dx**2) # The area of a ring, in terms of pixels 

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
            waveform=counts/Apix
    
            Hs_retrack[isampx,isampy],ind_retrack[isampx,isampy],dist[isampx,isampy,:] = simple_retracking_process_v01(waveform,edges_ref,max_edg=max_edg,nHs=nHs, wfm_ref=wfm_ref,
                                                                            offset = range_shift,Hsm_ref=Hsm_ref,isepoch=isepoch)
            waveforms[isampx,isampy,:]=waveform

        
    return Hs_retrack,Xalt,Yalt,waveforms,dist

	

def fly_over_track_v0(X,Y,S1,nsamp,nxa,di,wfm_ref,Hsm_ref,edges_ref,radi,radi1,radi2,alti_sat,range_shift):
        # radi = 4000     # radius used to compute waveform
        #radi1 = 900     # inner radius for Hs average
        #radi2 = 1200    # outer radius for Hs average
        nHs    = len(Hsm_ref)
        ny_mid = len(np.unique(Y))//2
        Xalt = np.zeros((nsamp,1))
        Hs_retrack = np.zeros((nsamp,1))
        ind_retrack = np.zeros((nsamp,1))
        Hs_std = np.zeros((nsamp,1))
        Hs_stdbis = np.zeros((nsamp,1))
        Hs_std2 = np.zeros((nsamp,1))
        waveforms=np.zeros((nsamp,len(edges_ref)-1))
        dist     =np.zeros((nsamp,nHs))

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

           dr = edges_ref[1]-edges_ref[0]
           Apix = np.pi*2*alti_sat*dr / (dx**2) # The area of a ring, in terms of pixels 
           counts,_=np.histogram(r,bins=edges_ref)
           waveform=counts/Apix
#	   Hs_retrack[isamp]             = simple_retracking_process   (counts,edges_ref,wfm_ref=wfm_ref,Hsm_ref=Hsm_ref,ispolyfit=0) 
           Hs_retrack[isamp],ind_retrack[isamp],dist[isamp] =simple_retracking_process_v01(waveform,edges_ref,max_edg=25,nHs=251,\
                                  offset=10,wfm_ref=wfm_ref,Hsm_ref=Hsm_ref,ispolyfit=0,isepoch=0)
           waveforms[isamp,:]=waveform

#def simple_retracking_process_v01(wfm,edges,max_edg=25,nHs=251,alti_sat=519*1e3,\
#                                  dx=10,offset=10,wfm_ref=None,Hsm_ref=None,ispolyfit=0,isepoch=0):


        return Hs_std,Hs_stdbis,Hs_std2,Hs_retrack,ind_retrack,Xalt,waveforms,surf1,footprint1


############# 2D erf waveforms ####################################"

def simple_retracking_process_2params(wfm,edges=None,max_edg=25,nHs=251,nze=250,\
                                  wfm_ref=None,Hsm_ref=None,ze_ref=None,costfun='LS'):
    
    testwf=np.broadcast_to(wfm,(nHs,nze,len(wfm)))
#     print('testwf shape = ',testwf.shape,' , wfm shape = ',wfm_ref.shape)
    if costfun=='LS':
       dist=np.sum((wfm_ref[:,:,0:max_edg]-testwf[:,:,0:max_edg])**2,axis=-1)
    else:
       ratio = testwf[:,:,12:max_edg]/wfm_ref[:,:,12:max_edg] 
       dist=np.sum( ratio - np.log(ratio),axis=-1)
#     print(np.sum(np.isnan(dist)),' / ',dist.size)
    
    hmin,zemin = np.unravel_index(np.nanargmin(dist,axis=None),dist.shape)
    
    Hs = Hsm_ref[hmin]
    ze = ze_ref[zemin]

    return Hs, ze, dist[hmin,zemin]

##########################################
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
            wfm[k,ize,:]=0.5+0.5*sps.erf(((edges[:-1]+ze) - offset) / (0.25*np.sqrt(2)*Hsm[k]))

    return wfm, Hsm, zem, edges, dr   

##########################################
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
    #print('size :',max_edg,len(edges_ref))
    dr = edges_ref[1]-edges_ref[0]
    dr2 = 0.5*dr
    Apix = np.pi*2*alti_sat*dr / (dx**2) # The area of a ring, in terms of pixels 

    
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
            #print(isampx,isampy,ialtx,ialty,np.max(r),np.min(r))

            # --- histogram counts the number of data points between edges ...
            #     modified by FA to center the bins on the edges.  
            counts,_=np.histogram(r,bins=edges_ref-dr2)
            waveform=counts/Apix
            Hs_retrack[isampx,isampy],ze_retrack[isampx,isampy],dist[isampx,isampy] = \
                                    simple_retracking_process_2params(waveform,edges_ref,max_edg=max_edg,
                                                                    nHs=nHs,nze=nze, wfm_ref=wfm_ref,
                                                                    ze_ref=ze_ref,Hsm_ref=Hsm_ref)
            waveforms[isampx,isampy,:]=waveform
#             print(Hs_retrack[isampx,isampy],ze_retrack[isampx,isampy],dist[isampx,isampy])

        
    return Hs_retrack,ze_retrack,Xalt,Yalt,waveforms,dist

######### compute simulated waveforms #####################################
def simu_waveform_erf(X,Y,S1,nsampx,nsampy,nxa0,nxa,di,discrete_ranges,range_offset=10,\
                       alti_sat=519*1e3):
#
#  WARNING: RIGHT NOW THIS IS A FLAT EARTH APPROXIMATION ... 
#
#  input parameters:
#                   X,Y,S1         : x and y position 
#                   nxa0           : is the first offset, keep away from boundary
#                   nxa            : chelton diam 
#                   discrete_ranges: center value of the discrete ranges  
#                  

#  input parameters: maybe put some of this back. 
#          radi        : radius over which waveform is computed
#          radi1, radi2: are 2 different radii used to compute std(elevation) 
#
    Xalt = np.zeros((nsampx))
    Yalt = np.zeros((nsampy))

    Hs_retrack = np.zeros((nsampx,nsampy))
    ze_retrack = np.zeros((nsampx,nsampy))
    waveforms=np.zeros((nsampx,nsampy,len(discrete_ranges)-1))
                    
    dx = X[1]-X[0]
    dy = Y[1]-Y[0]
    ny=len(Y)
    stepy=1
    shifty1=0

      
    # --- Footprint definition For std(surface) --------------------
    [Xa0,Ya0]=np.meshgrid(dx*np.arange(-nxa,nxa+1), dy*np.arange(-nxa,nxa+1))
    dist_ground = (Xa0**2+Ya0**2)
    
    dr = discrete_ranges[1]-discrete_ranges[0]
    dr2 = 0.5*dr
    Apix = np.pi*2*alti_sat*dr / (dx**2) # The area of a ring, in terms of pixels 

    
    for isampx in range(nsampx):
        if nsampy > 1:
           print('Generating waveform',isampx,' over ',nsampx,' ------------ ')
        else:
           shifty1=ny/2-nxa0
           stepy=0
        for isampy in range(nsampy):
            ialtx=(nxa0+isampx*di).astype(int)
            ialty=(nxa0+isampy*di*stepy+shifty1).astype(int)
            Xalt[isampx] = X[ialtx]
            Yalt[isampy] = Y[ialty]
             
            # --- get surface extract for altimeter ---------------------------
            surf1 = S1[ialty-nxa:ialty+nxa+1,ialtx-nxa:ialtx+nxa+1]
            # --- to have distance to satellite = range -------------------
            r=np.sqrt(Xa0**2+Ya0**2+(alti_sat-surf1)**2)-alti_sat+range_offset
            #print(isampx,isampy,ialtx,ialty,np.max(r),np.min(r))
            # --- histogram counts the number of data points between edges ...
            #     modified by FA to center the range values
            counts,_=np.histogram(r,bins=discrete_ranges-dr2)
            waveform=counts/Apix
            waveforms[isampx,isampy,:]=waveform
    return Xalt,Yalt,waveforms



