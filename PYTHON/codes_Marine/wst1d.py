#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 17:21:33 2022

@author: edouard
"""

import numpy as np
from numpy.fft import fft, ifft
import  proplot as pplt
import warnings
#------------------------------------------
# Functions
#------------------------------------------
def tilde_norm(S1,S2,var1,var2):
    S2t = S2/S1[...,None,:]
    var2t = S2t**2 * ((var1[...,None,:]/S1[...,None,:]**2) + (var2/S2**2))
    return S2t,var2t

def check_ndim(x):
    if x.ndim==1:
        x = x.reshape(1,-1)
    elif (x.ndim>2)|(x.ndim==0):
        raise TypeError('You are trying to apply the WST operator on 0d-array or >2d-array !\n data must be on the shape of (D,M) or (M).')
    return x

def nanmean(x,axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(x,axis=axis)
    return mean
#------------------------------------------
# WST 1D class
#------------------------------------------
class WST1D(object):
    """
    A class that implements the 1D wavelet scattering transform.

    Parameters
    ----------
    M : int
        Length of the input signal.
    J : int
        Number of wavelet scales.
    Jphi : int
        Number of low-pass scales. If not provided, defaults to J-2.

    Attributes
    ----------
    M : int
        Length of the input signal.
    J : int
        Number of wavelet scales.
    Jphi : int
        Number of low-pass scales.
    n : int
        Number of subsampled coefficients.
    psi : numpy array
        Wavelet filters.
    phi : numpy array
        Low-pass filters.
    xi : numpy array
        Central frequencies.
    sigma : numpy array
        Wavelet bandwidth
    """

    def __init__(self,M,J,Jphi=None):
        if Jphi!=None:
            if Jphi<J:
                raise TypeError('Jphi = {}, while J = {}.\nJphi must be greater than J.'.format(Jphi,J))
        if Jphi==None:
            Jphi = J
        if M/(3/2*2**(J-1))<2.:
            dj = 1
            while M/(3/2*2**(J-1-dj))<2.:
                dj += 1
            print('Oopsy ! To much Wavelet scales, J and Jphi have been changed.')
            print(' '*10+'   J : {} --> {}'.format(J,J-dj))
            print(' '*10+'Jphi : {} --> {}'.format(Jphi,Jphi-dj))
            J -= dj
            Jphi -= dj
        self.M = M
        self.J = J
        self.Jphi = Jphi
        self.n = self.M//2**self.Jphi
        # create filter bank :
        filters = Filters1D(self.M,self.J,self.Jphi).generate_filters()
        self.psi = filters['psi']
        self.phi = filters['phi']
        self.xi = filters['xi']
        self.sigma = filters['sigma']

    def get_coefs(self):
        """
        Returns the coefficients of the wavelet scattering transform.

        Returns
        -------
        Dict[str, numpy.ndarray]
            A dictionary containing the wavelet scattering transform coefficients.
            The keys are 'o0', 'o1' and 'o2' representing the 0-order, 1-order and 2-order coefficients respectively.
        """
        coefs_2order = {'c':self.S2}
        if hasattr(self,'S2t'):
            coefs_2order['t'] = self.S2t
        if hasattr(self,'S2d'):
            coefs_2order['d'] = self.S2d
        coefs = {'o0':self.S0,
                 'o1':self.S1,
                 'o2':coefs_2order
                }
        return coefs

    def get_noise_coefs(self):
        """
        Returns the noise coefficients of the wavelet scattering transform.

        Returns
        -------
        Dict[str, numpy.ndarray]
            A dictionary containing the noise coefficients of the wavelet scattering transform.
            The keys are 'o0', 'o1' and 'o2' representing the 0-order, 1-order and 2-order noise coefficients respectively.
        """
        coefs = {'o0':self.S0_noise,
                 'o1':self.S1_noise,
                 'o2':self.S2_noise
                }
        return coefs

    def get_sigmas(self):
        """
        Returns the standard deviation of the wavelet scattering transform.

        Returns
        -------
        Dict[str, numpy.ndarray]
            A dictionary containing the standard deviation of the wavelet scattering transform.
            The keys are 'o0', 'o1' and 'o2' representing the 0-order, 1-order and 2-order standard deviation respectively.
        """
        sigmas_2order = {'c':np.sqrt(self.var2)}
        if hasattr(self,'var2t'):
            sigmas_2order['t'] = np.sqrt(self.var2t)
        if hasattr(self,'var2d'):
            sigmas_2order['d'] = np.sqrt(self.var2d)
        sigmas = {'o0':np.sqrt(self.var0),
                  'o1':np.sqrt(self.var1),
                  'o2':sigmas_2order
                 }
        return sigmas

    def get_scales(self,ls=1):
        """
        Returns the physical scales.

        Parameters
        ----------
        ls : float
            The sampling length scale. Defaults to 1.

        Returns
        -------
        li : numpy.ndarray
            Array containing the physical scales.
        """
        ki = self.xi/ls # Wavelet Wavenumber
        li = 2*np.pi/ki # Wavelet Wavelength
        return li

    def average(self,x):
        """
        Computes the average of the signal by convolving with the low-pass filter.

        Parameters
        ----------
        x : numpy.ndarray
            The input signal.

        Returns
        -------
        numpy.ndarray
            The averaged signal.
        """
        x_ft = fft(x,axis=-1)
        return ifft(x_ft*self.phi[None,:],axis=-1).real


    def subsample(self,x):
        """
        This method is used to subsample the input data by averaging the values of each window of size 2^Jphi.
        The subsampling is performed along the last axis of the input data.

        Parameters
        ----------
        x : array_like
            The input data to be subsampled.
            It should be of shape (D, M) where D is the number of observations and M is the length of each observation.

        Returns
        -------
        y : array_like
            The subsampled data of shape (D, n, 2^Jphi) where n is the number of subsampled values of each observation.
        """

        if self.M % 2**self.Jphi != 0:
            N = self.n * 2**self.Jphi
            dn = (self.M-N)//2
            res = (self.M-N)%2
            x = x[:,dn:-(dn+res)]
        y = x.reshape(-1,self.n,2**self.Jphi)
        return y#.mean(-1)


    def compute(self, I0):
        """
        This method is used to compute the wavelet scattering coefficients of the input data.
        It computes the 0-order, 1-order and 2-order coefficients, as well as the associated variances.

        Parameters
        ----------
        I0 : array_like
            The input data to compute the scattering coefficients of.
            It should be of shape (D, M) where D is the number of observations and M is the length of each observation.

        Returns
        -------
        S0 : array_like
            The 0-order scattering coefficients of shape (D, n)
        S1 : array_like
            The 1-order scattering coefficients of shape (D, J, n)
        S2 : array_like
            The 2-order scattering coefficients of shape (D, J, J, n)
        var0 : array_like
            The variances of the 0-order scattering coefficients of shape (D, n)
        var1 : array_like
            The variances of the 1-order scattering coefficients of shape (D, J, n)
        var2 : array_like
            The variances of the 2-order scattering coefficients of shape (D, J, J, n)
        """
        # init :
        d,m = I0.shape
        if m != self.M:
            raise TypeError('Input data does not have the same length as filters.')
        S0 = np.zeros((d,self.n))*np.nan
        S1 = np.zeros((d,self.J,self.n))*np.nan
        S2 = np.zeros((d,self.J,self.J,self.n))*np.nan
        var0 = S0.copy()
        var1 = S1.copy()
        var2 = S2.copy()

        # O-order :
        U0 = self.average(I0)
        S0 = self.subsample(U0).mean(-1)
        var0 = self.subsample(U0).var(-1)
        #var0 = self.subsample((I0-U0)**2)

        # 1-order :
        for j1 in range(self.J):
            
            tmp1 = fft(I0,axis=-1) * self.psi[j1][None,:]
            I1 = abs(ifft(tmp1,axis=-1))
            U1 = self.average(I1)
            S1[:,j1,:] = self.subsample(U1).mean(-1)
            var1[:,j1,:] = self.subsample(U1).var(-1)
            #var1[:,j1,:] = self.subsample((I1-U1)**2)

            # 2-order :
            for j2 in range(self.J):
                if j2>j1:

                    tmp2 = fft(I1,axis=-1) * self.psi[j2][None,:]
                    I2 = abs(ifft(tmp2,axis=-1))
                    U2 = self.average(I2)
                    S2[:,j1,j2,:] = self.subsample(U2).mean(-1)
                    var2[:,j1,j2,:] = self.subsample(U2).var(-1)
                    #var2[:,j1,j2,:] = self.subsample((I2-U2)**2)
        
        return S0, S1, S2, var0, var1, var2

    def apply(self, data):
        """
        Apply the WST transform on a given data. 
        The data should be a list of 1D signals or a 2D numpy array with shape (D, M) 
        where D is the number of signals and M is the length of each signal.
        The result is stored as class attributes:
            - `self.data`: the input data
            - `self.D`: the number of signals in the input data
            - `self.S0`, `self.S1`, `self.S2`: the WST coefficients at order 0, 1 and 2 respectively
            - `self.var0`, `self.var1`, `self.var2`: the variances of the WST coefficients at order 0, 1 and 2 respectively
        """
        if type(data) is list:
            data = np.array(data)
        elif type(data) is not np.ndarray:
            raise TypeError('Input data type is not valid.')
        self.data = check_ndim(data)
        self.D = self.data.shape[0]
        self.S0,self.S1,self.S2,self.var0,self.var1,self.var2 = self.compute(self.data)


    def noise_realization(self, Nnoise=100):
        """
        Generates `Nnoise` realizations of white noise with the same power spectral 
        density as the input data and computes the WST coefficients on each realization.
        The result is stored as class attributes:
            - `self.Nnoise`: the number of noise realizations
            - `self.S0_noise`, `self.S1_noise`, `self.S2_noise`: the WST coefficients at order 0, 1 and 2 respectively for each noise realization
        """
        self.Nnoise = Nnoise
        flt = fft(self.data,axis=-1)
        flt[0] = 1
        wnoise = np.random.randn(Nnoise,self.D,self.M)
        wnoise_ft = fft(wnoise,axis=-1)
        cnoise = ifft(wnoise_ft*flt.conj()[None,...]).real
        cnoise = cnoise.reshape(-1,self.M)
        s0,s1,s2,v0,v1,v2 = self.compute(cnoise)
        self.S0_noise = nanmean(s0.reshape(Nnoise,self.D,self.n),axis=0)
        self.S1_noise = nanmean(s1.reshape(Nnoise,self.D,self.J,self.n),axis=0)
        self.S2_noise = nanmean(s2.reshape(Nnoise,self.D,self.J,self.J,self.n),axis=0)
        self.var0_noise = nanmean(v0.reshape(Nnoise,self.D,self.n),axis=0)
        self.var1_noise = nanmean(v1.reshape(Nnoise,self.D,self.J,self.n),axis=0)
        self.var2_noise = nanmean(v2.reshape(Nnoise,self.D,self.J,self.J,self.n),axis=0)   


    def coefs_normalization(self):
        """
        Normalize the second order wavelet coefficients.
        This method performs two types of normalization:
        1. Tilde normalization (S2t) - normalize the second order coefficients by the first order coefficients
        2. Dagger normalization (S2d) - the deviation from Gaussianity of the data.
        """
        self.S2t, self.var2t = tilde_norm(self.S1,self.S2,self.var1,self.var2) 

        try:
            self.S2t_noise, self.var2t_noise = tilde_norm(self.S1_noise,self.S2_noise,self.var1_noise,self.var2_noise)
        except Exception as e:
            print('The noise realization has not been done. Only the tilde normalization is available.')
        else:
            self.S2d = self.S2t / self.S2t_noise
            self.var2d = self.S2d**2 * ((self.var2t/self.S2t**2) + (self.var2t_noise/self.S2t_noise**2))
        

    def dispS1(self,ls=1,labels=None):
        sc = self.get_scales(ls)
        S1 = nanmean(self.S1,axis=-1)
        Err = nanmean(np.sqrt(self.var1),axis=-1)
        if labels is None:
            labels = np.arange(self.S1.shape[0])+1
        fig, ax = pplt.subplots(figsize=(6,5))
        
        for s1,err,lbl in zip(S1,Err,labels):
            ax.plot(sc,s1,lw=1,label=lbl)
            ax.fill_between(sc,s1-err,s1+err,alpha=.5,ec=None)
        ax.format(xlabel=r'scales',ylabel=r'$S_1$',
                    yscale='log',xscale='log', fontsize=15,
                    xformatter='log',yformatter='log',xreverse=True
                    )
        ax.legend(ncols=1)

        return fig, ax 

    def dispS2(self,ls=1,mode='classic',err=True,idx=None,refwidth=2.5,chl='w'):
    
        if mode=='classic':
            coef = self.S2
            sigma = np.sqrt(self.var2)
            ylabel = r'$S_2$'
        elif mode=='tilde':
            coef = self.S2t
            sigma = np.sqrt(self.var2t)
            ylabel = r'$\tilde{S}_2$'
        elif mode=='dagger':
            coef = self.S2d
            sigma = np.sqrt(self.var2d)
            ylabel = r'$S_2^\dagger$'
        
        sc = self.get_scales(ls)
        S2 = nanmean(coef,axis=-1)
        Err = nanmean(sigma,axis=-1)
        errnoise = nanmean(np.sqrt(self.var2t_noise),axis=(0,-1))

        if idx is not None:
            S2 = S2[idx]
            Err = Err[idx]

        wratios = np.arange(S2.shape[-2]-1,0,-1)
        wratios[-1] = 2
        fig, axs = pplt.subplots(nrows=1,ncols=S2.shape[-2]-1,wratios=wratios,wspace=0,ref=1,refwidth=refwidth)
        
        for i,ax in enumerate(axs):
            for ii,c in enumerate(S2[:,i]):
                if err==True:
                    ax.errorbar(sc,c,yerr=Err[ii,i])
                else:
                    ax.plot(sc,c,'.-')
            if mode=='dagger':
                ax.hlines(1,sc[i],sc[-1]+sc[-2],color=chl,lw=.5)
                ax.fill_between(sc,1-errnoise[i],1+errnoise[i],color=chl,ec=None,alpha=.5)
            ax.format(xlim=(sc[i],sc[-1]+sc[-2]),
                          title='{:.0e}'.format(sc[i])
                         )
        axs.format(xlabel=r'$L_{j_2}$',ylabel=ylabel,
                   xscale='log',yscale='log',xreverse=False,
                   xformatter='log',yformatter='log',
                   grid=False,suptitle=r'$L_{j_1}$'
                  )
        return fig, axs


#------------------------------------------
# Filter Bank Class
#------------------------------------------
def morlet_1d(M, xi, sigma, offset=0):
    """
        (from kymatio package) 
        Computes a 1D Morlet filter.
        A Morlet filter is the sum of a Gabor filter and a low-pass filter
        to ensure that the sum has exactly zero mean in the temporal domain.
        It is defined by the following formula in space:
        psi(u) = g_{sigma}(u) (e^(i xi^T u) - beta)
        where g_{sigma} is a Gaussian envelope, xi is a frequency and beta is
        the cancelling parameter.
        Parameters
        ----------
        sigma : float
            bandwidth parameter
        xi : float
            central frequency (in [0, 1])
        offset : int, optional
            offset by which the signal starts
        fft_shift : boolean
            if true, shift the signal in a numpy style
        Returns
        -------
        morlet_fft : ndarray
            numpy array of size (M)
    """
    wv = gabor_1d(M, xi, sigma, offset)
    wv_modulus = gabor_1d(M, 0, sigma, offset)
    K = np.sum(wv) / np.sum(wv_modulus)
    mor = wv - K * wv_modulus
    return mor

def gabor_1d(M, xi, sigma, offset=0):
    """
        (from kymatio package)
        Computes a 1D Gabor filter.
        A Gabor filter is defined by the following formula in space:
        psi(u) = g_{sigma}(u) e^(i xi^T u)
        where g_{sigma} is a Gaussian envelope and xi is a frequency.
        Parameters
        ----------
        sigma : float
            bandwidth parameter
        xi : float
            central frequency (in [0, 1])
        offset : int, optional
            offset by which the signal starts
        fft_shift : boolean
            if true, shift the signal in a numpy style
        Returns
        -------
        morlet_fft : ndarray
            numpy array of size (M, N)
    """
    curv = 1 / ( 2 * sigma**2)
    gab = np.zeros(M, np.complex128)
    xx = np.empty((2, M))
    
    for ii, ex in enumerate([-1, 0]):
        xx[ii] = np.arange(offset + ex * M, offset + M + ex * M)
    
    arg = -curv * xx * xx + 1.j * (xx * xi)
    gab = np.exp(arg).sum(0)
    norm_factor = 2 * np.pi * sigma**2
    gab = gab / norm_factor
    return gab

def normalize_filter(filter_ft):
    filter_ft /= abs(ifft(filter_ft)).sum()
    return filter_ft


class Filters1D(object):
    def __init__(self,M,J,Jphi):
        self.M = M
        self.J = J
        self.Jphi = Jphi
        self.xi = 3.*np.pi/4./2.**np.arange(J)
        self.sigma = .8*2**np.arange(J) 

    def generate_filters(self, precision='double', normalized=True):
        if precision=='double':
            psi = np.zeros((self.J, self.M), dtype=np.float64)
        if precision=='single':
            psi = np.zeros((self.J, self.M), dtype=np.float32)
        for j in range(0,self.J):
                wavelet = morlet_1d(self.M, self.xi[j], self.sigma[j])
                wavelet_ft = fft(wavelet)
                wavelet_ft[0] = 0
                if normalized==True:
                    wavelet_ft = normalize_filter(wavelet_ft)
                if precision=='double':
                    psi[j] = wavelet_ft.real
                if precision=='single':
                    psi[j] = wavelet_ft.real.astype(np.float32)
        gab_ft = fft(gabor_1d(self.M, 0, 0.8 * 2**(self.Jphi)))
        if normalized==True:
            gab_ft = normalize_filter(gab_ft)
        if precision=='double':
            phi = gab_ft.real
        if precision=='single':
            phi = gab_ft.real.astype(np.float32)
        filters = {'psi':psi, 'phi':phi,'xi':self.xi,'sigma':self.sigma}
        return filters

#------------------------------------------
# Older version
#------------------------------------------
''' 
class WST1D(object):
    def __init__(self,filters,Jphi):
        self.psi = filters['psi']
        self.phi = filters['phi']
        self.M = len(self.phi)
        self.J = len(self.psi)
        if Jphi >= self.J:
            raise TypeError('Jphi = {}, while J = {}.\nJphi must be smaller than J.'.format(Jphi,self.J))
        self.Jphi = Jphi
        self.n = self.M//2**self.Jphi
        self.S0 = np.zeros(self.n)*np.nan
        self.S1 = np.zeros((self.J,self.n))*np.nan
        self.S2 = np.zeros((self.J,self.J,self.n))*np.nan
        self.var0 = self.S0.copy()
        self.var1 = self.S1.copy()
        self.var2 = self.S2.copy()

    def getCoefs(self):
        coefs = {'o0':self.S0,
                 'o1':self.S1,
                 'o2':self.S2
                }
        return coefs

    def getSigmas(self):
        sigmas = {'o0':np.sqrt(self.var0),
                  'o1':np.sqrt(self.var1),
                  'o2':np.sqrt(self.var2)
                 }
        return sigmas

    def average(self,x):
        x_ft = fft(x)
        phi_ft = fft(self.phi)
        return ifft(x_ft*phi_ft).real/(.8*2**(self.Jphi+1))

    def subsample(self,x):
        if self.M % 2**self.Jphi != 0:
            N = self.n * 2**self.Jphi
            dn = (self.M-N)//2
            x = x[dn:-dn]
        y = x.reshape(self.n, 2**self.Jphi)
        return y.mean(-1)

    def compute(self, I0):
        if I0.shape[-1] != self.M:
            raise TypeError('Input data does not have the same length as filters.')

        # O-order :
        U0 = self.average(I0)
        self.S0 = self.subsample(U0)
        self.var0 = self.subsample((I0-U0)**2)

        # 1-order :
        for j1 in range(self.J):
            
            tmp1 = fft(I0) * self.psi[j1]
            I1 = abs(ifft(tmp1))
            U1 = self.average(I1)
            self.S1[j1,:] = self.subsample(U1)
            self.var1[j1,:] = self.subsample((I1-U1)**2)

            # 2-order :
            for j2 in range(self.J):
                if j2>j1:

                    tmp2 = fft(I1) * self.psi[j2]
                    I2 = abs(ifft(tmp2))
                    U2 = self.average(I2)
                    self.S2[j1,j2,:] = self.subsample(U2)
                    self.var2[j1,j2,:] = self.subsample((I2-U2)**2)

    def apply(self, data):
        self.compute(data)
        coefs = self.getCoefs()
        sigmas = self.getSigmas()
        return {'coefs':coefs, 'sigmas':sigmas}
'''
