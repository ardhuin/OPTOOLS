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
from scipy.signal import fftconvolve


def calc_footprint_diam(Hs,pulse_width = 1/(320*1e6),Rorbit=519*1e3,Rearth = 6370*1e3):
    clight= 299792458
    Airemax_div_pi = Rorbit*(clight*pulse_width + 2 * Hs)/(1+(Rorbit/Rearth))
    return 2*np.sqrt(Airemax_div_pi)


######################  Defines waveform theoretical models: most simple, 2 parameter erf 
######################                                      includes optionnal PTR 
def  wf_erf2D_eval(xdata,incognita,noise,Gamma,Zeta,c_xi,tau,PTR)  :
     ff0 = noise+ 0.5 * (  1+sps.erf( (xdata-incognita[0])/(np.sqrt(2)*incognita[1])  ) ) 
     if PTR[0] < 1:
        fff =fftconvolve(ff0,PTR,mode='same')
     else:
        fff=ff0
     return fff

     fff = noise+ 0.5 * (  1+sps.erf( (xdata-incognita[0])/(np.sqrt(2)*incognita[1])  ) ) 
     return fff

def  wf_erf2D(incognita,data)  :
     """
     returns the least-square distance between the waveform data[0] and the simplest erf waveform
     two unknown parameter: (epoch,Hs)  both in meters
     """

     ydata =data[0] # Waveform
     xdata =data[3] # times in ns 
     noise  =data[5]
     min_gate=data[6]
     max_gate=data[7]
     weights=data[8]
     costfun=data[9]
     PTR = data[10]
     ff0 = noise+ 0.5 *(  1+sps.erf( (xdata-incognita[0])/(np.sqrt(2)*incognita[1])  ) ) 
     if PTR[0] < 1:
        fff =fftconvolve(ff0,PTR,mode='same')
     else:
        fff=ff0
     if costfun=='LS':
        cy= (   ((ydata[min_gate:max_gate] - fff[min_gate:max_gate]) **2)).sum()
     else:
        ratio = np.divide(ydata[min_gate:max_gate]+1.e-5,fff[min_gate:max_gate]+1.e-5) 
        cy= ( ratio - np.log(ratio)).sum()

     return cy

######################  generalized erf with groups 
def  wf_erf4D_eval(xdata,incognita,noise,Gamma,Zeta,c_xi,tau,PTR)  :
    
     sig=incognita[1]
     da=incognita[3]*sig*sig
     ros=(xdata-incognita[0])/(np.sqrt(2)*sig)
     ro2=(xdata-incognita[0])/sig
     ro3=4*incognita[4]  # this is R0/Hs

     dw=np.exp(-0.5*(ro2-ro3)**2)*((ro2-ro3)**2-1)/(sig*sig)/np.sqrt(2*np.pi)
     dd=da*dw
     
     ff0 = noise+ 0.5 * (  1+sps.erf( ros  ) )+dd 
     if PTR[0] < 1:
        fff =fftconvolve(ff0,PTR,mode='same')
     else:
        fff=ff0

     return fff


def  wf_erf4D(incognita,data)  :
     """
     returns the least-square distance between the waveform data[0] and the simplest erf waveform
     two unknown parameter: (epoch,Hs)  both in meters
     """
     ydata  =data[0] # Waveform
     xdata  =data[3] # times in ns 

     noise  =data[5]
     min_gate=data[6]
     max_gate=data[7]
     weights=data[8]
     costfun=data[9]
     PTR = data[10]
     
     # Matlab version: dw=np.exp(-(R-r).^2/(2.*sig^2)).*((R-r).^2-sig.^2)./sig.^4/sqrt(2*pi);
     sig=incognita[1]
     da=incognita[3]*sig*sig
     ros=(xdata-incognita[0])/(np.sqrt(2)*sig)
     ro2=(xdata-incognita[0])/sig
     ro3=4*incognita[4]  # b is R0/(Hs), 4*b is R0/sig

     dw=np.exp(-0.5*(ro2-ro3)**2)*((ro2-ro3)**2-1)/(sig*sig)/np.sqrt(2*np.pi)
     dd=da*dw
 
     ff0 = noise+ 0.5 *( 1+sps.erf( ros ) ) +  dd 
     if PTR[0] < 1:
        fff =fftconvolve(ff0,PTR,mode='same')
     else:
        fff=ff0

     if costfun=='LS':
        cy= (   ((ydata[min_gate:max_gate] - fff[min_gate:max_gate]) **2)).sum()
     else:
        ratio = np.divide(ydata[min_gate:max_gate]+1.e-5,fff[min_gate:max_gate]+1.e-5) 
        cy= ( ratio - np.log(ratio)).sum()

     return cy





######################  Defines waveform theoretical models: brown from WHALES code 
def  wf_brown_eval(xdata,incognita,noise,Gamma,Zeta,c_xi,tau,PTR)  :
# This is Jean's MLE 
#A =  0.5*exp(-4*X/gamma)*sig0;
#if ordre==1
#    a1 = a*(1-2*X-4*X/gamma);
#else
#    a1= a*(1-2*X-2*X/gamma);
#end;
#u1 = (n - epoq-a1*sigC^2)/(sqrt(2)*sigC);
#v1 = a1.*(n - epoq - 0.5*a1*sigC^2);
#    modele1= A.*exp(-v1).*(1+erf(u1));
#    modele = modele1 + Bt;


     #ff0 = xdata*0
     ff0=noise+( incognita[2]/2*np.exp((-4/Gamma)*(np.sin(Zeta))**2) \
     * np.exp (-  c_xi*( (xdata-incognita[0])-c_xi*incognita[1]**2/2) ) \
     *   (  1+scipy.special.erf( ((xdata-incognita[0])-c_xi*incognita[1]**2)/((np.sqrt(2)*incognita[1]))  ) ) \
     )
     if PTR[0] < 1:
        fff =fftconvolve(ff0,PTR,mode='same')
     else:
        fff=ff0

     return fff

def  wf_brown(incognita,data)  :
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
     c_xi  =data[4]  #Term related to the slope of the trailing edge
     noise  =data[5]
     min_gate=data[6]
     max_gate=data[7]
     weights=data[8]
     costfun=data[9]
     PTR = data[10]
               
     ff0 = noise+( incognita[2]/2*np.exp((-4/Gamma)*(np.sin(Zeta))**2) \
     * np.exp (-  c_xi*( (xdata-incognita[0])-c_xi*incognita[1]**2/2) ) \
     *   (  1+scipy.special.erf( ((xdata-incognita[0])-c_xi*incognita[1]**2)/((np.sqrt(2)*incognita[1]))  ) ) \
     )
     if PTR[0] < 1:
        fff =fftconvolve(ff0,PTR,mode='same')
     else:
        fff=ff0
  
    
#        cy= (   weights *  ((ydata - fff) **2)).sum()
     if costfun=='LS':
        cy= (   ((ydata[min_gate:max_gate] - fff[min_gate:max_gate]) **2)).sum()
     else:
        ratio = np.divide(ydata[min_gate:max_gate]+1.e-5,fff[min_gate:max_gate]+1.e-5) 
        cy= ( ratio - np.log(ratio)).sum()
     
     return cy

######################  Defines waveform theoretical models: brown from WHALES code + wave groups
def  wf_bro4D_eval(xdata,incognita,noise,Gamma,Zeta,c_xi,tau,PTR)  :
     sig=incognita[1]
     da=incognita[3]*sig*sig
     ros=(xdata-incognita[0])/(np.sqrt(2)*sig)
     ro2=(xdata-incognita[0])/sig
     ro3=4*incognita[4]  # b is R0/(Hs), 4*b is R0/sig

     dw=np.exp(-0.5*(ro2-ro3)**2)*((ro2-ro3)**2-1)/(sig*sig)/np.sqrt(2*np.pi)
     dd=da*dw

#     ff0 = noise+ 0.5 *( 1+sps.erf( ros ) ) +  dd 
     ff0 = noise+( incognita[2]/2*np.exp((-4/Gamma)*(np.sin(Zeta))**2) \
     * np.exp (-  c_xi*( (xdata-incognita[0])-c_xi*incognita[1]**2/2) ) \
     *   (2*dd + 1+scipy.special.erf( ((xdata-incognita[0])-c_xi*incognita[1]**2)/((np.sqrt(2)*incognita[1]))  ) ) \
     )
     if PTR[0] < 1:
        fff =fftconvolve(ff0,PTR,mode='same')
     else:
        fff=ff0

     return fff

def  wf_bro4D(incognita,data)  :
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
     c_xi  =data[4]  #Term related to the slope of the trailing edge
     noise  =data[5]
     min_gate=data[6]
     max_gate=data[7]
     weights=data[8]
     costfun=data[9]
     PTR = data[10]
     
     sig=incognita[1]
     da=incognita[3]*sig*sig
     ros=(xdata-incognita[0])/(np.sqrt(2)*sig)
     ro2=(xdata-incognita[0])/sig
     ro3=4*incognita[4]  # b is R0/(Hs), 4*b is R0/sig

     dw=np.exp(-0.5*(ro2-ro3)**2)*((ro2-ro3)**2-1)/(sig*sig)/np.sqrt(2*np.pi)
     dd=da*dw

     ff0 = noise+( incognita[2]/2*np.exp((-4/Gamma)*(np.sin(Zeta))**2) \
     * np.exp (-  c_xi*( (xdata-incognita[0])-c_xi*incognita[1]**2/2) ) \
     *   (2*dd+  1+scipy.special.erf( ((xdata-incognita[0])-c_xi*incognita[1]**2)/((np.sqrt(2)*incognita[1]))  ) ) \
     )
     if PTR[0] < 1:
        fff =fftconvolve(ff0,PTR,mode='same')
     else:
        fff=ff0

    
     if costfun=='LS':
        cy= (   ((ydata[min_gate:max_gate] - fff[min_gate:max_gate]) **2)).sum()
     else:
        ratio = np.divide(ydata[min_gate:max_gate]+1.e-5,fff[min_gate:max_gate]+1.e-5) 
        cy= ( ratio - np.log(ratio)).sum()
     
     return cy


######################
def wf_eval(ranges,inputpar,clight,wf_model,tau=2.5,nominal_tracking_gate=30,noise=0.,alti_sat=None,mispointing=0.,theta3dB=1.,PTR_model='Gauss',PTR=([1.])):
    stonano=1000000000
    rtot=2*stonano/clight
    SigmaP=0.513*tau
    Ri=6378.1363*(10**3)      #Earth radius

    Zeta=mispointing
    Gamma =(np.sin(theta3dB))**2/(np.log(2)*2)
    clightn=clight/stonano

##############"""" WARNING: mispoiting below should be in radians ... hence Zeta ?? 

    b_xi = np.cos (2*mispointing) - ((np.sin(2*mispointing))**2)/Gamma
    c_xi=b_xi* ( (4/Gamma)*(clightn/alti_sat) * 1/(1+alti_sat/Ri))

    incognita=inputpar
    xdata=ranges*(2./clight)*stonano
    incognita[0]= inputpar[0]*rtot+nominal_tracking_gate*tau
    #incognita[2]= 1.
    if PTR_model == 'Gauss':
       incognita[1]= np.sqrt( (inputpar[1]*0.25*rtot)**2+SigmaP**2  )
    else:
       incognita[1]= inputpar[1]*0.25*rtot

    fff=eval(wf_model+'_eval')(xdata,incognita,noise,Gamma,Zeta,c_xi,tau,PTR)
    
    return fff



############# A 2-parameter retracker using scipy.minimize , as in WHALES #################

def retracking_NM(wfm,times,rtot,wf_fun,Gamma=1.,Zeta=0.,c_xi=0.,min_gate=0,max_gate=127,weights=1.,\
noise=0.,tau=2.5,costfun='LS',nominal_tracking_time=64*2.5,method='Nelder-Mead',PTR=([1])):

    Pu=None
    da=None
    R0=None 
    if wf_fun =='wf_erf2D':
       incognita=np.array([nominal_tracking_time,10.0*rtot,1.,0,0]) # initial conditions: could use previous waveform ... 
    elif wf_fun =='wf_brown':
      incognita=np.array([nominal_tracking_time,10.0*rtot,1.,0,0]) # initial conditions: could use previous waveform ... 
    elif wf_fun =='wf_erf4D':
      incognita=np.array([nominal_tracking_time,5.0*rtot,1.,0,0]) # initial conditions: could use previous waveform ... 
    elif wf_fun =='wf_bro4D':
      incognita=np.array([nominal_tracking_time,10.0*rtot,1e-6,0,0]) # initial conditions: could use previous waveform ... 

    xopt = minimize(eval(wf_fun), incognita, args=((wfm,Gamma,Zeta,times,c_xi,noise,min_gate,max_gate,weights,costfun,PTR),),\
                    method=method,options={'disp': False})
# bounds=((-4*rtot,4*rtot),(0.0,2.5*rtot)),
    x=xopt.x
    if xopt.success == True:
       Sigma=x[1]
       epoch=x[0]
       if wf_fun =='wf_brown':
          Pu=x[2]
       if wf_fun =='wf_bro4D':
          Pu=x[2]
          da=x[3]
          R0=x[4]
       if wf_fun =='wf_erf4D':
          da=x[3]
          R0=x[4]
       dist=eval(wf_fun)(x,((wfm),Gamma,Zeta,(times),c_xi,noise,min_gate,max_gate,weights,costfun,PTR))
    else:
       Sigma=-0.1
       epoch=-1.
       dist=-1.
    return Sigma, epoch, Pu, da, R0, dist

############# A 1-parameter pyramid grid search #################
def retracking_pyramid1(wfm,times,rtot,wf_fun,Gamma=1.,Zeta=0.,c_xi=0.,weights=1.,noise=0.,tau=2.5,costfun='LS',nominal_tracking_time=64*2.5):
    nsteps=12
    a1=15.0*rtot
    b1= 7.5*rtot
    
    for istep in range(nsteps):
        dist=np.zeros((5,1))
        for i1 in range(5):
# Note that 9 out of 25 have already been computed at the previous step ... 
              incognita=np.array([0.,a1+(i1-2)*b1])
              dist[i1]=eval(wf_fun)(incognita,(wfm,Gamma,Zeta,times,c_xi,weights,noise,tau,costfun,PTR)) 
              #print('      inds:',i0,i1,incognita,dist[i0,i1])       
    
        i1min = np.unravel_index(np.nanargmin(dist,axis=None),dist.shape)
        epoch=0.              
        Sigma=a1+(i1min-2)*b1  
        dmin =dist[i1min] 
# Update of search interval ... 
        a1=a1+(i1min-2)*b1
        b1=b1/2.
        #print('step:',istep,epoch,Hs,dmin)       
    return Sigma, epoch, dmin

############# A 2-parameter pyramid grid search #################
def retracking_pyramid2(wfm,times,rtot,wf_fun,Gamma=1.,Zeta=0.,c_xi=0.,weights=1.,noise=0.,tau=2.5,costfun='LS',nominal_tracking_time=64*2.5,PTR=([1.0])):
    nsteps=12
    a0= 0.0+nominal_tracking_time
    a1=15.0*rtot
    b0= 2.0*rtot
    b1= 7.5*rtot
    
    for istep in range(nsteps):
        dist=np.zeros((5,5))
        for i0 in range(5):
           for i1 in range(5):
# Note that 9 out of 25 have already been computed at the previous step ... 
              incognita=np.array([a0+(i0-2)*b0,a1+(i1-2)*b1])
              dist[i0,i1]=eval(wf_fun)(incognita,(wfm,Gamma,Zeta,times,c_xi,weights,noise,tau,costfun,PTR)) 
              #print('      inds:',i0,i1,incognita,dist[i0,i1])       
    
        i0min,i1min = np.unravel_index(np.nanargmin(dist,axis=None),dist.shape)
        epoch=a0+(i0min-2)*b0              
        Sigma=a1+(i1min-2)*b1  
        dmin =dist[i0min,i1min] 
# Update of search interval ... 
        a0=a0+(i0min-2)*b0
        a1=a1+(i1min-2)*b1
        b0=b0/2.
        b1=b1/2.
        #print('step:',istep,epoch,Hs,dmin)       
    return Sigma, epoch, dmin


############# A 3-parameter pyramid grid search #################
def retracking_pyramid3(wfm,times,rtot,wf_fun,noise=0.,Gamma=1.,Zeta=0.,c_xi=0.,min_gate=0,max_gate=127,weights=1.,tau=2.5,costfun='LS',nominal_tracking_time=64*2.5,PTR=([1.0])):
    nsteps=10
    a0=0.0+nominal_tracking_time
    a1=10.*rtot
    
    b0=2.0*rtot
    b1=5.0*rtot
    a2=1.0
    b2=0.5  # Pu 
    
    da=0.
    R0=0.
    
    for istep in range(nsteps):
        dist=np.zeros((5,5,5))
        for i0 in range(5):
         for i1 in range(5):
          for i2 in range(5):
# Note that 9 out of 25 have already been computed at the previous step ... 
              incognita=np.array([a0+(i0-2)*b0,a1+(i1-2)*b1,a2+(i2-2)*b2])
              dist[i0,i1,i2]=eval(wf_fun)(incognita,(wfm,Gamma,Zeta,times,c_xi,noise,min_gate,max_gate,weights,costfun,PTR)) 
              #print('      inds:',i0,i1,incognita,dist[i0,i1])       
    
        i0min,i1min,i2min = np.unravel_index(np.nanargmin(dist,axis=None),dist.shape)
        epoch=a0+(i0min-2)*b0              
        Sigma=a1+(i1min-2)*b1  
        Pu   =a2+(i2min-2)*b2  
        dmin =dist[i0min,i1min,i2min] 
# Update of search interval ... 
        a0=a0+(i0min-2)*b0
        a1=a1+(i1min-2)*b1
        a2=a2+(i2min-2)*b2
        b0=b0/2.
        b1=b1/2.
        b2=b2/2.
        #ddd=eval(wf_fun)(np.array((nominal_tracking_time,10.*rtot,1)),(wfm,Gamma,Zeta,times,c_xi,noise,min_gate,max_gate,weights,costfun,PTR)) 
          
        #print('step:',istep,i0min,i1min,i2min,4*Sigma/rtot, (epoch-nominal_tracking_time)/rtot, Pu, dmin,ddd,nominal_tracking_time)       
    #print('Last:',istep,4*Sigma/rtot, (epoch-nominal_tracking_time)/rtot, Pu, dmin,ddd)       
    return Sigma, epoch, Pu, da, R0, dmin

############# A 4-parameter pyramid grid search #################
def retracking_pyramid4(wfm,times,rtot,wf_fun,Gamma=1.,Zeta=0.,c_xi=0.,min_gate=0,max_gate=127,weights=1., \
                        noise=0.,tau=2.5,costfun='LS',nominal_tracking_time=64*2.5,PTR=([1.0])):
    nsteps=10
    a0=0.0+nominal_tracking_time
    a1=10.*rtot
    b0=2.0*rtot
    b1=5.0*rtot
    a2=0.2
    b2=0.1  # normalized waveform pertubation 
    a3=0.4
    b3=0.1  # factor 4 otherwise we get negative R0/Hs ... 

    Pu=1.0
    
    for istep in range(nsteps):
        dist=np.zeros((5,5,5,5))
        for i0 in range(5):
         for i1 in range(5):
          for i2 in range(5):
           for i3 in range(5):
# Note that 9 out of 25 have already been computed at the previous step ... 
              incognita=np.array([a0+(i0-2)*b0,a1+(i1-2)*b1,0,a2+(i2-2)*b2,a3+(i3-2)*b3])
              dist[i0,i1,i2,i3]=eval(wf_fun)(incognita,(wfm,Gamma,Zeta,times,c_xi,noise,min_gate,max_gate,weights,costfun,PTR)) 
              #print('      inds:',i0,i1,incognita,dist[i0,i1])       
    
        i0min,i1min,i2min,i3min = np.unravel_index(np.nanargmin(dist,axis=None),dist.shape)
        epoch=a0+(i0min-2)*b0              
        Sigma=a1+(i1min-2)*b1  
        da   =a2+(i2min-2)*b2  
        R0   =a3+(i3min-2)*b3  
        dmin =dist[i0min,i1min,i2min,i3min] 
# Update of search interval ... 
        a0=a0+(i0min-2)*b0
        a1=a1+(i1min-2)*b1
        a2=a2+(i2min-2)*b2
        a3=a3+(i3min-2)*b3
        b0=b0/2.
        b1=b1/2.
        b2=b2/2.
        b3=b3/2.
        #print('step:',istep,i2min,i3min,Sigma, epoch, Pu, da, R0, dmin)       
    return Sigma, epoch, Pu, da, R0, dmin




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
def retrack_waveforms(waveforms,ranges,max_range_fit,clight,mispointing=0.,theta3dB=1., min_range_fit=0,\
                      wfm_ref=None,Hsm_ref=None,ze_ref=None,\
                      min_method='gridsearch',wf_model='erf2D',PTR_model='Gauss',PTR=([1.]),\
                      costfun='LS',alti_sat=519*1e3,Theta=1.,tau=2.5,nominal_tracking_gate=30):
    #############################
    #  tau : duration of range gate in nsec 
    nxw,nyw,nr=np.shape(waveforms)
    
    print('size of waveforms array:',nxw,nyw,nr)
    Ri=6378.1363*(10**3)      #Earth radius
    stonano=1000000000
    rtot=(2./clight)*stonano  #Converts range to time

    times=ranges*rtot
    timeshift=tau*nominal_tracking_gate
    Hs_r=np.zeros((nxw,nyw))
    ze_r=np.zeros((nxw,nyw))
    Pu_r=np.zeros((nxw,nyw))+1.0
    da_r=np.zeros((nxw,nyw))
    R0_r=np.zeros((nxw,nyw))
    di_r=np.zeros((nxw,nyw))
    SigmaP=0.513*tau   # should use tax instead!
    Gamma =(np.sin(theta3dB))**2/(np.log(2)*2)
    clightn=clight/stonano
    in1=nominal_tracking_gate-40
    in2=nominal_tracking_gate-30
    noise=np.median(np.mean(waveforms[:,:,in1:in2],axis=2))
    if len(mispointing)<2:
       mispointing=mispointing+np.zeros((nxw,nyw))
    

    for ix in range(nxw):
        print('Retracking waveforms',ix,' out of ',nxw,' ------------ ')
        for iy in range(nyw):
            wfm=waveforms[ix,iy,:]
            b_xi = np.cos (2*mispointing[ix,iy]) - ((np.sin(2*mispointing[ix,iy]))**2)/Gamma
            c_xi=b_xi* ( (4/Gamma)*(clightn/alti_sat) * 1/(1+alti_sat/Ri))
            if min_method == 'gridsearch':
               Sigma,t0,di_r[ix,iy]=simple_retracking_process_2params(wfm,\
                                  max_edg=max_range_fit,nHs=250,nze=251,wfm_ref=wfm_ref,Hsm_ref=Hsm_ref,ze_ref=ze_ref,costfun=costfun)
            elif min_method in [ 'Nelder-Mead','Newton-CG']:
               Sigma,t0,Pu_r[ix,iy],da_r[ix,iy],R0_r[ix,iy],di_r[ix,iy]=retracking_NM(wfm, times,rtot,wf_model,
                                                              min_gate=min_range_fit,max_gate=max_range_fit,noise=noise,tau=tau, Gamma=Gamma,Zeta=mispointing[ix,iy], \
                                                              c_xi=c_xi,nominal_tracking_time=timeshift,method=min_method,costfun=costfun,PTR=PTR)
            elif min_method == 'pyramid2':
               Sigma,t0,di_r[ix,iy]=retracking_pyramid2(wfm[min_range_fit:max_range_fit],\
                                                              times[min_range_fit:max_range_fit],rtot,wf_model,noise=noise,tau=tau,Gamma=Gamma,Zeta=mispointing[ix,iy],\
                                                              c_xi=c_xi,nominal_tracking_time=timeshift,costfun=costfun,PTR=PTR)
            elif min_method == 'pyramid3':
               Sigma,t0,Pu_r[ix,iy],da_r[ix,iy],R0_r[ix,iy],di_r[ix,iy]=retracking_pyramid3(wfm, times,rtot,wf_model,rtot,wf_model,\
                                                    min_gate=min_range_fit,max_gate=max_range_fit,noise=noise,tau=tau,Gamma=Gamma,Zeta=mispointing[ix,iy],\
                                                    c_xi=c_xi,nominal_tracking_time=timeshift,costfun=costfun,PTR=PTR)
            elif min_method == 'pyramid4':
               Sigma,t0,Pu_r[ix,iy],da_r[ix,iy],R0_r[ix,iy],di_r[ix,iy]=retracking_pyramid4(wfm,times,rtot,wf_model,\
                                                    min_gate=min_range_fit,max_gate=max_range_fit, noise=noise,tau=tau,Gamma=Gamma,Zeta=mispointing[ix,iy],\
                                                    c_xi=c_xi,nominal_tracking_time=timeshift,costfun=costfun,PTR=PTR)
            if PTR_model == 'Gauss':              
               #print('TEST1:',Sigma*4/rtot,np.sqrt(Sigma**2- SigmaP**2)*4/rtot)
               if Sigma >= 0:
                  sigma_squared=( Sigma**2- SigmaP**2 )
                  if sigma_squared>=0 :    
                     Hs_r[ix,iy]=np.sqrt(sigma_squared)*4/rtot
                  else:
                     Hs_r[ix,iy]=0.
            else:
# No correction needed if waveform fitted with PTR 
               Hs_r[ix,iy]=Sigma*4/rtot   
               #print('TEST2:',Sigma*4/rtot,np.sqrt(Sigma**2- SigmaP**2)*4/rtot)
            Epoch=t0 - nominal_tracking_gate*tau;  
            ze_r[ix,iy]=Epoch/rtot  #m conversion from ns to meters
            #print('fit:',ix,iy,Sigma,SigmaP,SWH_squared,Hs_r[ix,iy])
            #print('tau:',tau,t0,t0 - nominal_tracking_gate*tau,ze_r[ix,iy])
            #print('da:',da_r[ix,iy],R0_r[ix,iy])
          
    return Hs_r,ze_r,Pu_r,da_r,R0_r,di_r
    
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
def simu_waveform_erf(X,Y,S1,nsampx,nsampy,nxa0,nxa,di,ranges,range_offset=10,\
                       alti_sat=519*1e3,Gamma=1.):
#
#  WARNING: RIGHT NOW THIS IS A FLAT EARTH APPROXIMATION ... 
#  
#
#  input parameters:
#                   X,Y,S1         : x and y position 
#                   nxa0           : is the first offset, keep away from boundary
#                   nxa            : chelton diam 
#                   ranges         : center value of the discrete ranges  
#                   Gamma          : antenna pattern parameter ... 
#                  

#  input parameters: maybe put some of this back. 
#          radi        : radius over which waveform is computed
#          radi1, radi2: are 2 different radii used to compute std(elevation) 
#
    Xalt = np.zeros((nsampx))
    Yalt = np.zeros((nsampy))

    Hs_retrack = np.zeros((nsampx,nsampy))
    ze_retrack = np.zeros((nsampx,nsampy))
    waveforms=np.zeros((nsampx,nsampy,len(ranges)-1))
                    
    dx = X[1]-X[0]
    dy = Y[1]-Y[0]
    ny=len(Y)
    stepy=1
    shifty1=0
    Ri=6378.1363*(10**3) #Earth radius
      
    # --- Footprint definition For std(surface) --------------------
    [Xa0,Ya0]=np.meshgrid(dx*np.arange(-nxa,nxa+1), dy*np.arange(-nxa,nxa+1))
    dist_ground = (Xa0**2+Ya0**2)
    
    dr = ranges[1]-ranges[0]
    dr2 = 0.5*dr
    Apix = np.pi*2*alti_sat*dr / (dx**2) # The area of a ring, in terms of pixels 

    r2=Xa0**2+Ya0**2
    # Uses 2-way antenna pattern to reduce backscattered power 
    power=np.exp(-4*(r2/alti_sat**2)*(1+alti_sat/Ri)/Gamma) 
            
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
            r=np.sqrt(r2+(alti_sat-surf1)**2)-alti_sat+range_offset
            #print(isampx,isampy,ialtx,ialty,np.max(r),np.min(r))
            # --- histogram counts the number of data points between edges ...
            #     modified by FA to center the range values
            counts,_=np.histogram(r,bins=ranges-dr2,weights=power)
            waveform=counts/Apix
            waveforms[isampx,isampy,:]=waveform
    return Xalt,Yalt,waveforms



