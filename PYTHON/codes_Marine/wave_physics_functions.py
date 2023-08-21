# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================================================
# === 0. Import Packages ===========================================================
# ==================================================================================
# from Misc_functions import *
import numpy as np
import sys
#############################################################################
###  1. Dispersion relation and associated ##################################

def phase_speed_from_k(k,depth=None,g=9.81):
	if depth is None:
		# print("Deep water approximation")
		C=np.sqrt(g/k)
	else:
		# print("General case")
		C=np.sqrt(g*np.tanh(k*depth)/k)
	return C
			
def phase_speed_from_sig_k(sig,k):
	return sig/k
	
def group_speed_from_k(k,depth=None,g=9.81):
	C=phase_speed_from_k(k,depth=depth,g=g)
	if depth is None:
		# print("Deep water approximation")
		Cg=C/2
	else:
		# print("General case")
		Cg=C*(0.5+ ((k*depth)/(np.sinh(2*k*depth)) ))
	return Cg

def sig_from_k(k,D=None,g=9.81):
	if D is None:
		# print("Deep water approximation")
		sig = np.sqrt(g*k)
	else:
		# print("General case")
		sig = np.sqrt(g*k*np.tanh(k*D))
	return sig

def f_from_sig(sig):
	return sig/(2*np.pi)
	
def f_from_k(k,D=None,g=9.81):
	sig = sig_from_k(k,D=D,g=g)
	return sig/(2*np.pi)
	
def sig_from_f(f):
	return 2*np.pi*f
	
def period_from_sig(sig):
	return (2*np.pi)/sig

def period_from_wvl(wvl,D=None):
	k=(2*np.pi)/wvl
	sig=sig_from_k(k,D=D)
	T = period_from_sig(sig)
	return T

def k_from_f(f,D=None,g=9.81):
	# inverts the linear dispersion relation (2*pi*f)^2=g*k*tanh(k*dep) to get 
	#k from f and dep. 2 Arguments: f and dep. 
	eps=0.000001
	sig=np.array(2*np.pi*f)
	if D is None:
		# print("Deep water approximation")
		k=sig**2/g
	else:
		Y=D*sig**2/g
		X=np.sqrt(Y)
		I=1
		F=1.
		while (abs(np.max(F)) > eps):
			H=np.tanh(X)
			F=Y-(X*H)
			FD=-H-(X/(np.cosh(X)**2))
			X=X-(F/FD)

		k=X/D

	return k # wavenumber
	
#############################################################################
###  2. Get quick translation info (T0/k/f/L) ###############################
def infos_from_wvl(wvl,D=None):
	wvnb = 2*np.pi/wvl
	f = f_from_k(wvnb,D=D)
	P = 1/f

	print('From a wavelength of ',wvl, ' m : -----------------------')
	print('     - wavenumber k   =   '+f'{wvnb:.4f}'.rjust(6)+' rad/m')
	if D is None:
		print('   With the infinite depth approximation :')
	else:
		print('   With a depth of ',D,' m')
	print('     - frequency f    =   '+f'{f:.3f}'.rjust(6)+' Hz')
	print('     - period T       =   '+f'{P:.2f}'.rjust(6)+' s')
	print('--------------------------------------------------------')

def infos_from_wvnb(wvnb,D=None):
	wvl = 2*np.pi/wvnb
	f = f_from_k(wvnb,D=D)
	P = 1/f

	print('From a wavenumber of ',wvnb, ' rad/m : -----------------------')
	print('     - wavelength L   =   '+f'{wvl:.1f}'.rjust(6)+' m')
	if D is None:
		print('   With the infinite depth approximation :')
	else:
		print('   With a depth of ',D,' m')
	print('     - frequency f    =   '+f'{f:.3f}'.rjust(6)+' Hz')
	print('     - period T       =   '+f'{P:.2f}'.rjust(6)+' s')
	print('--------------------------------------------------------')

def infos_from_T0(P,D=None):
	f = 1/P
	wvnb = k_from_f(f,D=D)
	wvl = 2*np.pi/wvnb

	print('From a period of ',P, ' s : -----------------------')
	print('     - frequency f    =   '+f'{f:.3f}'.rjust(6)+' Hz')
	if D is None:
		print('   With the infinite depth approximation :')
	else:
		print('   With a depth of ',D,' m')
	print('     - wavelength L   =   '+f'{wvl:.1f}'.rjust(6)+' m')
	print('     - wavenumber k   =   '+f'{wvnb:.4f}'.rjust(6)+' rad/m')
	print('--------------------------------------------------------')

def infos_from_f(f,D=None):
	P = 1/f
	wvnb = k_from_f(f,D=D)
	wvl = 2*np.pi/wvnb

	print('From a frequency of ',f, ' Hz : -----------------------')
	print('     - period T       =   '+f'{P:.2f}'.rjust(6)+' s')
	if D is None:
		print('   With the infinite depth approximation :')
	else:
		print('   With a depth of ',D,' m')
	print('     - wavelength L   =   '+f'{wvl:.1f}'.rjust(6)+' m')
	print('     - wavenumber k   =   '+f'{wvnb:.4f}'.rjust(6)+' rad/m')
	print('--------------------------------------------------------')    



#############################################################################
###  3. Classical Wave spectra ##############################################

## ---- 3.1. 1D Wave spectra along f or k -----------------------------------
def PM_spectrum_f(f,fm,g=9.81):
# There are 2 ways of writing the PM spectrum:
#  - eq 12 of Pierson and Moskowitz (1964) with exp(-0.74 * (f/fw)**-4) where fw=g*U10/(2*pi)
#  - eq of Hasselmann et al. 1973 with exp(-5/4 * (f/ fm)**-4) where fm is the max frequency  ...
# See Hasselmann et al. 1973 for the explanation 
	alpha=8.1*10**-3
	E = alpha*g**2*(2*np.pi)**-4*f**-5*np.exp((-5/4)*((fm/f)**4))
	return E

def PM_spectrum_k(k,fm,D=None,g=9.81):
# There are 2 ways of writing the PM spectrum:
#  - eq 12 of Pierson and Moskowitz (1964) with exp(-0.74 * (f/fw)**-4) where fw=g*U10/(2*pi)
#  - eq of Hasselmann et al. 1973 with exp(-5/4 * (f/ fm)**-4) where fm is the max frequency  ...
# See Hasselmann et al. 1973 for the explanation 
	f=sig_from_k(k,D=D)/(2*np.pi)
	Ef = PM_spectrum_f(f,fm,g=g)
	dfdk = dfdk_from_k(k,D=D)
	
	return Ef*dfdk

def Gaussian_1Dspectrum_kx(kX,T0,sk_k0,D=None):
	kp = k_from_f(1/T0,D=D)
	sk_k = kp*sk_k0
	Z1D_Gaussian =1/(np.sqrt(2*np.pi)*sk_k)* np.exp( - 0.5*((kX-kp)**2)/(sk_k**2))
	
	return Z1D_Gaussian,kX,sk_k
	
## ---- 3.2. 2D Wave spectra ------------------------------------------------
def define_spectrum_PM_cos2n(k,th,T0,thetam,n=4,D=None):
	Ek=PM_spectrum_k(k,1/T0,D=D)
	dth=th[1]-th[0]
	Eth=np.cos(th-thetam)**(2*n)
	II=np.where(np.cos(th-thetam) < 0)[0]
	Eth[II]=0
	sth=sum(Eth*dth)
	Ekth=np.broadcast_to(Ek,(len(th),len(k)))*np.broadcast_to(Eth,(len(k),len(th))).T /sth
	return Ekth,k,th

def define_Gaussian_spectrum_kxky(kX,kY,T0,theta_m,sk_theta,sk_k,D=None):
	if (len(kX.shape)==1) & (len(kY.shape)==1):
		kX,kY = np.meshgrid(kX,kY)
	elif (len(kX.shape)==1) | (len(kY.shape)==1):
		print('Error : kX and kY should either be: \n      - both vectors of shapes (nx,) and (ny,) \n  OR  - both matrices of shape (ny,nx)')
		print('/!\ Proceed with caution /!\ kX and kY have been flattened to continue running')
		kX = kX.flatten()
		kY = kY.flatten()
	kp = k_from_f(1/T0,D=D)
	# rotation of the grid => places kX1 along theta = theta_m
	kX1 = kX*np.cos(theta_m)+kY*np.sin(theta_m)
	kY1 = -kX*np.sin(theta_m)+kY*np.cos(theta_m)
	
	Z1_Gaussian =1/(2*np.pi*sk_theta*sk_k)* np.exp( - 0.5*((((kX1-kp)**2)/((sk_k)**2))+kY1**2/sk_theta**2))
	
	return Z1_Gaussian,kX,kY


#############################################################################
###  4. Jacobians and variable changes ######################################

## ---- 4.1 Jacobians -------------------------------------------------------
def dfdk_from_k(k,D=None):
	Cg = group_speed_from_k(k,depth=D,g=9.81)
	return Cg/(2*np.pi)

## ----- 4.2 Change variables from spectrum ---------------------------------
def spectrum_from_fth_to_kth(Efth,f,th,D=None):
    shEfth = np.shape(Efth)
    # print(shEfth)    
    if len(shEfth)<2:
        print('Error: spectra should be 2D')
    else:
        if shEfth[0]==shEfth[1]:
            print('Warning: same dimension for freq and theta.\n  Proceed with caution: The computation is done considering Efth = f(f,th)')
        elif (shEfth[1]==len(f)) &(shEfth[0]==len(th)):
            Efth = np.swapaxes(Efth,0,1)
        elif (shEfth[1]==len(th)) &(shEfth[0]==len(f)):
            print('All good: Efth have the shape : (f,th)')    
        else:
            print('shEfth[1] : ',shEfth[1], ' vs ',len(f),'// shEfth[0] :',shEfth[0],' vs ',len(th))
            print('Error: Efth should have the shape : (f,th)')
    shEfth = np.shape(np.moveaxis(Efth,0,-1))
    k=k_from_f(f,D=D)
    dfdk=dfdk_from_k(k,D=D)
    Ekth = Efth*np.moveaxis(np.broadcast_to(dfdk,shEfth),-1,0)
    return Ekth, k, th

def spectrum_from_kth_to_kxky(Ekth,k,th):
	try:
		shEkth = np.shape(Ekth)
		if len(shEkth)<2:
			print('Error: spectra should be 2D')
		else:
			if shEkth[0]==shEkth[1]:
				print('Warning: same dimension for k and theta.\n  Proceed with caution: The computation is done considering Ekth = f(k,th)')
			elif ((shEkth[1]==len(k)) &(shEkth[0]==len(th))) | ((shEkth[1]==len(th)) &(shEkth[0]==len(k))):
				if (shEkth[1]==len(k)) &(shEkth[0]==len(th)):
					Ekth = np.swapaxes(Ekth,0,1)
			else:
				print('shEkth[1] : ',shEkth[1], ' vs ',len(k),'// shEkth[0] :',shEkth[0],' vs ',len(th))
				print('Error: Ekth should have the shape : (k,th)')
		shEkth2 = np.shape(np.moveaxis(Ekth,0,-1)) # send k-axis to last -> in order to broadcast k along every dim
		shEkth2Dkth = Ekth.shape[0:2] # get only shape k,th for the broadcast of the dimensions kx,ky
		
		if np.max(th)>100:
			th=th*np.pi/180
		kx = np.moveaxis(np.broadcast_to(k,shEkth2Dkth[::-1]),-1,0) * np.cos(np.broadcast_to(th,shEkth2Dkth))
		ky = np.moveaxis(np.broadcast_to(k,shEkth2Dkth[::-1]),-1,0) * np.sin(np.broadcast_to(th,shEkth2Dkth))
		Ekxky = Ekth/np.moveaxis(np.broadcast_to(k,shEkth2),-1,0)
		return Ekxky, kx, ky
	except Exception as inst:
		print('in spec to kxky : ',inst,', line number : ',sys.exc_info()[2].tb_lineno)

def spectrum_from_fth_to_kxky(Efth,f,th,D=None):
	shEfth = np.shape(Efth)
	#print(shEfth)
	if len(shEfth)<2:
		print('Error: spectra should be 2D')
	else:
		if shEfth[0]==shEfth[1]:
			print('Warning: same dimension for freq and theta.\n  Proceed with caution: The computation is done considering Efth = f(f,th)')
		elif ((shEfth[1]==len(f)) &(shEfth[0]==len(th))) | ((shEfth[1]==len(th)) &(shEfth[0]==len(f))):
			if (shEfth[1]==len(f)) &(shEfth[0]==len(th)):          	
				Efth = np.swapaxes(Efth,0,1)  
		else:
			print('Error: Efth should have the shape : (f,th)')
	shEfth2 = np.shape(np.moveaxis(Efth,0,-1)) # send f-axis to last -> in order to broadcast f along every dim
	shEfth2Dfth = Efth.shape[0:2] # get only shape f,th for the broadcast of the dimensions kx,ky
	k=k_from_f(f,D=D)
	dfdk=dfdk_from_k(k,D=D)
	if np.max(th)>100:
		th=th*np.pi/180

	kx = np.moveaxis(np.broadcast_to(k,shEfth2Dfth[::-1]),-1,0) * np.cos(np.broadcast_to(th,shEfth2Dfth))
	ky = np.moveaxis(np.broadcast_to(k,shEfth2Dfth[::-1]),-1,0) * np.sin(np.broadcast_to(th,shEfth2Dfth))
	Ekxky = Efth * np.moveaxis(np.broadcast_to(dfdk /k,shEfth2),-1,0)
	return Ekxky, kx, ky

def spectrum_to_kxky(typeSpec,Spec,ax1,ax2,D=None):
    if typeSpec==0: # from f,th
        Ekxky, kx, ky = spectrum_from_fth_to_kxky(Spec,ax1,ax2,D=D)
    elif typeSpec==1: # from k,th
        Ekxky, kx, ky = spectrum_from_kth_to_kxky(Spec,ax1,ax2)
    else:
        print('Error ! typeSpec should be 0 = (f,th) or 1 = (k,th)')
        Ekxky = Spec
        kx = ax1
        ky = ax2
    return Ekxky, kx, ky

def spectrum_f_to_k(Ef,f,D=None):
	shEf = np.array(np.shape(Ef))
	ind = np.where(shEf == len(f))[0]
	if len(ind)==0:
		print('Error: spectra should have an axis with same dimension as f')
	elif len(ind)>1:
		print('Warning: same dimension for different axes.\n  Proceed with caution: The computation is done considering Ef = f(...,f)')
		if ind[-1]<(len(shEf)-1):
			Ef=np.swapaxes(Ef,ind[-1],-1)
	elif len(ind)==1:
		Ef=np.swapaxes(Ef,ind,-1) # pass the f axis as last dim : to broadcast 
	
	k=k_from_f(f,D=D)
	dfdk=dfdk_from_k(k,D=D)
	shEf2 = np.shape(Ef)
	Ek = np.swapaxes(Ef*np.broadcast_to(dfdk,shEf2),-1,ind)
	return Ek, k

def spectrum_k_to_f(Ek,k,D=None):
	shEk = np.array(np.shape(Ek))
	ind = np.where(shEk == len(k))[0]
	if len(ind)==0:
		print('Error: spectra should have an axis with same dimension as k')
	elif len(ind)>1:
		print('Warning: same dimension for different axes.\n  Proceed with caution: The computation is done considering Ek = f(...,k)')
		if ind[-1]<(len(shEk)-1):
			ind0 = int(ind[-1])
			Ek=np.swapaxes(Ek,ind0,-1)
	elif len(ind)==1:
		ind0 = int(ind)
		Ek=np.swapaxes(Ek,ind0,-1) # pass the f axis as last dim : to broadcast 
	
	f=f_from_k(k,D=D)
	dfdk=dfdk_from_k(k,D=D)
	shEk2 = np.shape(Ek)
	Ef = np.swapaxes(Ek/np.broadcast_to(dfdk,shEk2),-1,ind0)
	return Ef, f
	
'''
def PM_spectrum_k(k,fm,g=9.81):
	pmofk(k,T0,H)
	alpha=8.1*10**-3
	
	w0=2*np.pi/T0
	w=np.sqrt(g*k*tanh(k*H))
	Cg=(0.5+k.*H/sinh(2.*k.*H)).*w./k;
	pmofk=0.008.*g.^2.*exp(-0.74.*(w./w0).^(-4))./(w.^5).*Cg+5.9;
	
	E = alpha*g**2*(2*np.pi)**-4*f**-5*np.exp((-5/4)*((fm/f)**4))
	return E
'''
