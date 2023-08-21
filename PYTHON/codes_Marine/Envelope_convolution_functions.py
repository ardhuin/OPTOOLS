# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================================================
# === 0. Import Packages ===========================================================
# ==================================================================================
# from Misc_functions import *
import numpy as np
import xarray as xr
import xrft
import scipy.interpolate as spi
import scipy.integrate as spint

from functions_cfosat_v1 import *
from surface_simulation_functions import *

def calc_footprint_diam(Hs,pulse_width = 1/(320*1e6),Rorbit=519*1e3,Rearth = 6370*1e3):
    clight= 299792458
    Airemax_div_pi = Rorbit*(clight*pulse_width + 2 * Hs)/(1+(Rorbit/Rearth))
    return 2*np.sqrt(Airemax_div_pi)

# ==================================================================================
# === 1. 1D Functions ==============================================================
# ==================================================================================
def myconv(x, h,axis=-1):
	assert np.shape(x) == np.shape(h), 'Inputs to periodic convolution '\
		               'must be of the same period, i.e., shape.'

	X = np.fft.fft(x,axis=axis)
	H = np.fft.fft(h,axis=axis)

	return np.real(np.fft.ifft(np.multiply(X, H),axis=axis))

def remove_part(x,h,dkvec):
	A = np.zeros(np.shape(x))
	for k in range(len(x)):
		if k<len(x)//2:
			A[k]=np.sum(x[k:len(x)//2]*h[np.arange((len(x)//2-1),(k-1),-1)]*dkvec[k:len(x)//2])
		else:
			A[k]=np.sum(x[len(x)//2:k]*h[k-1:len(x)//2-1:-1]*dkvec[len(x)//2:k])
	return A
    
def compute_env2_spec1D_complete(x0,kvec0,Nresampling=2501):
	if kvec0[0]>=0:
		kvec = np.concatenate([-kvec0[::-1],np.zeros(1),kvec0])#,np.zeros(1),k_vec])
		x = np.concatenate([np.zeros(len(kvec0)+1),x0])
	else:
		kvec = kvec0
		x = x0

	kvecnew = np.linspace(kvec[0],kvec[-1],Nresampling)
	f = spi.interp1d(kvec,x)
	xnew = f(kvecnew)

	# --- Define a pos/neg spectrum (real valued) -----
	lbd = 0.5
	x2 = lbd*xnew+(1-lbd)*xnew[::-1]
	# ----
	A = 8*np.fft.fftshift(myconv(x2,x2))*np.gradient(kvecnew)
	A2 = 8*remove_part(x2,x2,np.gradient(kvecnew))
	return A-A2, kvecnew
	
def compute_env2_spec1D_approx(x0,kvec0,Nresampling=2501):
	if kvec0[0]>=0:
		kvec = np.concatenate([-kvec0[::-1],np.zeros(1),kvec0])#,np.zeros(1),k_vec])
		x = np.concatenate([np.zeros(len(kvec0)+1),x0])
	else:
		kvec = kvec0
		x = x0

	kvecnew = np.linspace(kvec[0],kvec[-1],Nresampling)
	f = spi.interp1d(kvec,x)
	xnew = f(kvecnew)
	# --- TBD : verify that you have only positive frequencies in your life...
	# --- Work with the pos vs neg part of the spectrum 
	B2 = 8*2*np.fft.fftshift(myconv(0.5*xnew,0.5*np.flip(xnew)))*np.gradient(kvecnew)

# ==================================================================================
# === 2. 2D Functions ==============================================================
# ==================================================================================
def myconv2D(x, h):
    assert np.shape(x) == np.shape(h), 'Inputs to periodic convolution '\
                               'must be of the same period, i.e., shape.'

    X = np.fft.fft2(x)
    H = np.fft.fft2(h)
    
    nx = np.size(X,0)
    ny = np.size(X,1)

    return np.roll(np.real(np.fft.ifft2(np.multiply(X, H))),[nx//2+1,ny//2+1],axis=[0,1])

# --- functions from env2 to env ---------------------------------------------------
def from_env2_to_env(spec_env2,Hs):
    return spec_env2*2*(4-np.pi)/(Hs**2)

def from_env_to_spec_Hs(spec_env):
    return 8*spec_env
    
# ==================================================================================
# === 3. interpolation Functions ===================================================
# ==================================================================================  
# --- interp functions  ---------------------------------------------------
def prep_interp_grid(dkmin=0.0003734846,kmax=0.21):
	# --- define k interpolation vector (positive part) ----
	kX00 = np.arange(0.5*(dkmin),kmax,dkmin)
	# --- duplicate to have both positive and negative parts --------------
	kX0_origin = np.concatenate((kX00,-np.flip(kX00[:])))
	kX0 = np.fft.fftshift(kX0_origin)
	kx = kX0
	ky = kX0[:-2] 
	# --- generate 2D grid ------------------------
	kX,kY = np.meshgrid(kx,ky , indexing='ij')
	# --- compute associated K, Phi(in deg) ---------
	kK = np.sqrt(kX**2+kY**2)
	kPhi = np.arctan2(kY,kX)*180/np.pi
	kPhi[kPhi<0]=kPhi[kPhi<0]+360

	kK2 = xr.DataArray(kK, coords=[("kx", kx), ("ky",ky)])

	kPhi2 = xr.DataArray(kPhi, coords=[("kx", kx), ("ky",ky)])
	kKkPhi2s = xr.Dataset(
		{'kK': kK2,
		'kPhi': kPhi2}
		).stack(flattened=["kx", "ky"])
	return kKkPhi2s

def spec1D_from_spec_0360(spec):
	spec['wave_spectra_kth_hs'].values[np.isnan(spec['wave_spectra_kth_hs'].values)]=0
	# spec['wave_spectra_kth'].values[np.isnan(spec['wave_spectra_kth'].values)]=0
	spec_bis = xr.concat([spec.isel(n_phi=-1),spec,spec.isel(n_phi=0)],dim="n_phi",data_vars='minimal')
	
	# --- change the first and last new values to have a 2pi revolution ---------------
	A = np.concatenate([[-360],np.zeros((spec.dims['n_phi'])),[360]])
	factor = xr.DataArray(A, dims="n_phi")
	spec_bis['phi_vector'].values = spec_bis['phi_vector']+factor

	dphis = np.diff(spec_bis['phi_vector'].values)
	dphi = xr.DataArray(0.5*(dphis[0:-1]+dphis[1:]),dims='n_phi')*np.pi/180
	# divide by 2 pi if you want to have the mean value : circular spectrum
	Spec1D_v1 =  (spec['wave_spectra_kth_hs']*dphi).sum(dim='n_phi').data#/(2*np.pi)
	# Spec1D_v2 =  (spec['wave_spectra_kth']*dphi).sum(dim='n_phi').data/(2*np.pi)
	Hsnew = 4*np.sqrt((spec['wave_spectra_kth_hs']*dphi*spec['dk']).sum(dim=['n_phi','nk']).data)

	return Spec1D_v1, Hsnew#, Spec1D_v2

def interp_from_spec_0360(spec,kKkPhi2s):
	try:
		spec['wave_spectra_kth_hs'].values[np.isnan(spec['wave_spectra_kth_hs'].values)]=0
	except Exception as inst:
		print(inst)
	
	spec_bis = xr.concat([spec.isel(n_phi=-1),spec,spec.isel(n_phi=0)],dim="n_phi",data_vars='minimal')
	# --- change the first and last new values to have a 2pi revolution ---------------
	A = np.concatenate([[-360],np.zeros((spec.dims['n_phi'])),[360]])
	factor = xr.DataArray(A, dims="n_phi")
	spec_bis['phi_vector'].values = spec_bis['phi_vector']+factor

	dphis = np.diff(spec_bis['phi_vector'].values)
	dphi = xr.DataArray(0.5*(dphis[0:-1]+dphis[1:]),dims='n_phi')*np.pi/180

	Ekxky0, kx, ky = spectrum_to_kxky(1,np.squeeze(spec_bis['wave_spectra_kth_hs'].compute().data),  spec_bis['k_vector'].compute().data,spec_bis["phi_vector"].compute().data)

	# outputs kx and ky of the original spectrum are useless 
	# (as we work on polar coords, we only use them for plotting purposes) 

	# define the spectrum as a dataArray to apply the interp
	Speckxky = xr.DataArray(Ekxky0, dims=("nk", "n_phi"), coords={"nk":spec_bis['k_vector'].astype(np.float64) , "n_phi":spec_bis["phi_vector"]})
	# --- apply interpolation -----------------------------------
	B = Speckxky.interp(nk=kKkPhi2s.kK,n_phi=kKkPhi2s.kPhi,kwargs={"fill_value": 0})
	B.name='Ekxky_new'
	B0 = B.reset_coords(("nk","n_phi"))
	Ekxky_2 = B0.Ekxky_new.unstack(dim='flattened')
	# Ekxky_2 : -fmax : 0 :fmax

	Hsnew = 4*np.sqrt((spec['wave_spectra_kth_hs']*dphi*spec['dk']).sum(dim=['n_phi','nk']).data)

	return Ekxky_2, Hsnew   


# ==================================================================================
# === 4. std estimation ============================================================
# ==================================================================================  
# --- functions to estimate std(Hs)  ---------------------------------------------------
def estimate_stdHs_from_spec1D(ds_sel0,Nresampling=2501):
	# ---- 0. PREPARATION ---------------------------
	# ---- 0.1 turn spec to phi = [0:360] ---------------------------
	ds_sel1 = ds_sel0.copy(deep=True)
	ds_sel1["phi_vector"].values = (ds_sel1["phi_vector"].compute().data+180.)%360
	ds_sel = xr.concat([ds_sel0,ds_sel1],dim="n_phi",data_vars='minimal')
	ds_sel['wave_spectra_kth_hs'].values = ds_sel['wave_spectra_kth_hs'].values/2
	# ds_sel['wave_spectra_kth'].values = ds_sel['wave_spectra_kth'].values/2 # f o g (k,th), with f(kx,ky)
	ds_sel = ds_sel.sortby('phi_vector')

	# spec1D_v1, spec1D_v2,Hs_0 = spec1D_from_spec_0360(ds_sel)
	spec1D_v1, Hs_0 = spec1D_from_spec_0360(ds_sel)
	# ---- 0.2 prep grid for interp ---------------------------
	k_vecmax= ds_sel['k_vector'].compute().data.max()
	kvecnew = np.linspace(-k_vecmax,k_vecmax,Nresampling)
	# ---- 0.3 get klim from footprint ---------------------------
	Diam_chelton = calc_footprint_diam(Hs_0)
	klim = 2*np.pi/Diam_chelton
	
	# ------------------------------------------------------------------------------------
	# ---- 1. V1 : f(k,th) such that Hs= 4 * sqrt(sum(f(k,th)*dk*dth)) -------------------
	# ---- 1.1 Interp to new kx,ky grid ---------------------------
	f = spi.interp1d(ds_sel['k_vector'].compute().data,spec1D_v1,fill_value=0,bounds_error=False)
	xnew_v1 = f(kvecnew)

	# ---- 1.2 Convolution ---------------------------
	Spec1D_env2_fromconv1D_v1 = 8*2*np.fft.fftshift(myconv(0.5*xnew_v1, 0.5*np.flip(xnew_v1)))* np.gradient(kvecnew)
	Spec1D_Hs_from_conv1D_v1 = from_env_to_spec_Hs(from_env2_to_env(Spec1D_env2_fromconv1D_v1,Hs_0))
	# ---- 1.3 Integrate up to size footprint -------------------
	spec_Hs_1D_func_v1 = spi.interp1d(kvecnew,Spec1D_Hs_from_conv1D_v1)
	int_specHs_1D_v1 = spint.quad_vec(spec_Hs_1D_func_v1,-klim,klim)[0]

	var_env2_v1 = np.sum(Spec1D_env2_fromconv1D_v1*np.gradient(kvecnew))
	
	# ------------------------------------------------------------------------------------
	# ---- 2. V2 : f o g(k,th) such that Hs = 4 * sqrt(sum(f o g(k,th)*k*dk*dth)) --------------------
	# ---- 2.1 Interp to new kx,ky grid ---------------------------
	# f = spi.interp1d(ds_sel['k_vector'].compute().data,spec1D_v2,fill_value=0,bounds_error=False)
	# xnew_v2 = f(kvecnew)

	# ---- 2.2 Convolution ---------------------------
	# Spec1D_env2_fromconv1D_v2 = 8*2*np.fft.fftshift(myconv(0.5*xnew_v2, 0.5*np.flip(xnew_v2)))* np.gradient(kvecnew)
	# Spec1D_Hs_from_conv1D_v2 = from_env_to_spec_Hs(from_env2_to_env(Spec1D_env2_fromconv1D_v2,Hs_0))
	
	# ---- 2.3 Integrate up to size footprint -------------------
	# spec_Hs_1D_func_v2 = spi.interp1d(kvecnew,Spec1D_Hs_from_conv1D_v2)
	# int_specHs_1D_v2 = spint.quad_vec(spec_Hs_1D_func_v2,-klim,klim)[0]

	# var_env2_v2 = np.sum(Spec1D_env2_fromconv1D_v2*np.gradient(kvecnew))

	return np.sqrt(int_specHs_1D_v1),Hs_0, var_env2_v1#, Spec1D_env2_fromconv1D_v1 #,np.sqrt(int_specHs_1D_v2),var_env2_v2
    
    
def estimate_stdHs_from_spec2D_old(ds_sel0,kKkPhi2s):
	# ---- 1. Spec L2 ---------------------------
	# ---- 1.1 turn spec to phi = [0:360] ---------------------------
	ds_sel1 = ds_sel0.copy(deep=True)
	ds_sel1["phi_vector"].values = (ds_sel1["phi_vector"].compute().data+180.)%360
	ds_sel = xr.concat([ds_sel0,ds_sel1],dim="n_phi",data_vars='minimal')
	ds_sel['wave_spectra_kth_hs'].values = ds_sel['wave_spectra_kth_hs'].values/2
	ds_sel = ds_sel.sortby('phi_vector')

	# ---- 1.2 Interp to new kx,ky grid ---------------------------
	Ekxky,Hs_0 = interp_from_spec_0360(ds_sel,kKkPhi2s)
	dkx = np.gradient(Ekxky.kx)[0]
	dky = np.gradient(Ekxky.ky)[0]
	# ---- 1.3 Convolution ---------------------------
	Spec2D_env2_from_convol2D = 8*2*(myconv2D(0.5*Ekxky,0.5*np.flip(Ekxky)))*dkx*dky
	Spec2D_Hs_from_convol2D = from_env_to_spec_Hs(from_env2_to_env(Spec2D_env2_from_convol2D,Hs_0))
	# ---- 1.4 Integrate up to size footprint -------------------
	Diam_L2 = calc_footprint_diam(Hs_0)
	Rad_L2 = Diam_L2/2

	# -- check method 1 -----
	kxlim = 2*np.pi/Rad_L2
	kylim = 2*np.pi/Rad_L2
	#     print(np.shape(Ekxky.kx),np.shape(Ekxky.ky),np.shape(Spec2D_Hs_from_convol2D))
	int_specHs = spi.RectBivariateSpline(Ekxky.kx,Ekxky.ky, Spec2D_Hs_from_convol2D).integral(-kxlim, kxlim, -kylim, kylim)

	dkx2D,dky2D = np.meshgrid(np.gradient(Ekxky.kx),np.gradient(Ekxky.ky))
	var_env2 = np.sum(Spec2D_env2_from_convol2D.T*dkx2D*dky2D)

	return np.sqrt(int_specHs), Hs_0, var_env2   

def estimate_stdHs_from_spec2D(ds_sel0,kKkPhi2s,L1S,Hs=None):
	try:
		# ---- 1. Spec L2 ---------------------------
		# ---- 1.1 turn spec to phi = [0:360] ---------------------------
		ds_sel1 = ds_sel0.copy(deep=True)
		ds_sel1["phi_vector"].values = (ds_sel1["phi_vector"].compute().data+180.)%360
		ds_sel = xr.concat([ds_sel0,ds_sel1],dim="n_phi",data_vars='minimal')
		ds_sel['wave_spectra_kth_hs'].values = ds_sel['wave_spectra_kth_hs'].values/2
		ds_sel = ds_sel.sortby('phi_vector')

		# ---- 1.2 Interp to new kx,ky grid ---------------------------
		Ekxky,Hs_0 = interp_from_spec_0360(ds_sel,kKkPhi2s)
		if Hs is None:
			Hs = Hs_0
		dkx = np.gradient(Ekxky.kx)[0]
		dky = np.gradient(Ekxky.ky)[0]
		kx_c = Ekxky.kx.compute().data
		nkx_c = len(Ekxky.kx.compute().data)
		nky_c = len(Ekxky.ky.compute().data)
		dx_c = twopi/(nkx_c*dkx)
		dy_c = twopi/(nky_c*dky)
		Xa_c = dx_c*(np.arange(-nkx_c//2,nkx_c//2)+0.5)
		Ya_c = dy_c*(np.arange(-nky_c//2,nky_c//2)+0.5)
		[Xa_c2,Ya_c2] = np.meshgrid(Xa_c, Ya_c, indexing='ij')
		
		# -- Compute Lambda2 -------------------------------------------
		Lambda2 = Hs_0**4/(256*(Ekxky**2).sum*dkx*dky)
		# ---- 1.3 Convolution ---------------------------
		Spec2D_env2_from_convol2D = 8*(myconv2D(Ekxky,np.flip(Ekxky)))*dkx*dky
		Spec2D_Hs_from_convol2D = from_env_to_spec_Hs(from_env2_to_env(Spec2D_env2_from_convol2D,Hs))
		
		# ---- 1.4 Define filters ---------------------------------
		Diam_chelton = calc_footprint_diam(Hs)
		# --- integrate up to various k1 --------------------
		# L1S = [(7/5)*1e3,7*1e3,77*1e3,80*1e3]
		int_specHs= np.zeros((2,len(L1S)+1))
		
		L0_L2 = Diam_chelton/9
		for ifilt in range(2):
			if ifilt ==0:
			# ---- 1.4.a Filter Gaussian ------------------------------
				phi_x00 = np.exp(-0.5* Xa_c2**2 / (L0_L2)**2 )*np.exp(-0.5* Ya_c2**2 / (L0_L2)**2)
				phi_x0 = xr.DataArray(phi_x00/(L0_L2**2*twopi),#np.sum(phi_x00),#
							dims=['x','y'],
							coords={
							    "x" : Xa_c,
							    "y" : Ya_c,
							    },
							)
			elif ifilt==1:
			# ---- 1.4.b Good Filter (Annex A) ------------------------
				rc = Diam_chelton/2
				r0 = np.sqrt((Xa0)**2+(Ya0)**2)

				G_Lc20 = np.exp(-0.5* r0**2 / (rc)**2 )
				G_Lc2 = G_Lc20/(rc**2*twopi)#np.sum(G_Lc20*dx*dy)

				Id = np.zeros(np.shape(G_Lc2))
				Id[nkx_c//2,nky_c//2]=1/(dx_c*dy_c)

				Jr0 = (4*dx_c*dy_c/(np.pi*rc**2)) * (r0/rc)**2 * (6 - ((2*r0/rc)**4)) * np.exp(- 4 * r0**4 / rc**4)
				Jr0 = Jr0/np.sum(Jr0*dx_c*dy_c)

				Jr1 = fftconvolve((Id-G_Lc2),Jr0,mode='same')*dx_c*dy_c

				Filter_new = (G_Lc2+Jr1)
				Filter_new = Filter_new /np.sum(Filter_new*dx_c*dy_c)	
				
				phi_x0 = xr.DataArray(Filter_new,#np.sum(phi_x00),#
							dims=['x','y'],
							coords={
							    "x" : Xa_c,
							    "y" : Ya_c,
							    },
							)
			phi0_hat_ktild0 = xrft.power_spectrum(phi_x0, dim=["x", "y"]) # function of ktild = k/twopi
			phi0_hat_k = phi0_hat_ktild0*twopi*twopi	
				
			# ---- 1.5 Apply filters and integrate---------------------
			SpecHs_filt_div = phi0_hat_k*Spec2D_Hs_from_convol2D/(dkx*dky)
			# -- int over y ---------------------
			SpecHs_filt_div_inty = SpecHs_filt_div.sum('freq_y')*dky
			spec_Hs_funcx = spi.interp1d(kx_c,SpecHs_filt_div_inty)
			
			# -- store the total integral --------------
			int_specHs[ifilt,-1] = SpecHs_filt_div_inty.sum('freq_x')*dkx
			
			for i1,L1 in enumerate(L1S):
				k1 = twopi/(2*L1)
				int_specHs_remove = spint.quad_vec(spec_Hs_funcx,-k1,k1)[0]
				int_specHs[ifilt,i1] = int_specHs[ifilt,-1] - int_specHs_remove

		return np.sqrt(int_specHs), Hs_0  ,Lambda2 
	except Exception as inst:
		print('inside estimate_std function ',inst,'line bis :',sys.exc_info()[2].tb_lineno)
		return np.zeros((2,len(L1S)+1)), 0  ,0 


# ==================================================================================
# === 5.1 estimation for one track ( for-loop over different tracks ) ==============
# ==================================================================================      
def function_one_track(indf,files,nbeam,isbabord,is1D):
	print(indf,' start ')
	file_L2 = files[0]
	file_L2S = files[1]
	file_L2P = files[2]
	L1S = [(7/5)*1e3,7*1e3,77*1e3,80*1e3]
	print(file_L2S)
	try:
		kKkPhi2s = prep_interp_grid(dkmin=0.00006755)#dkmin=0.00016755)
		
		# --- read files CNES ------------ 
		ds_boxes = read_boxes_from_L2_CNES_work_quiet(file_L2)

		# --- read files ODL ------------ 
		ds_l2s = read_l2s_offnadir_files_work_quiet(file_L2S)
		ds_l2s = ds_l2s.isel(l2s_angle=0)
	
		# ---- Create a new dataset with indexes of the box for each ribbon ----
		ind_time_box_2,ind_babord_box_2,ds_l2s_new = get_indices_box_for_ribbon(ds_boxes,ds_l2s)
	
		ntim = ds_boxes.dims['time0']
		time_box = np.zeros((ntim))
		
		ds_L2P = xr.open_dataset(file_L2P)
		flag_valid_L2P_swh_box0 = ds_L2P['flag_valid_swh_box'].compute().data
		flag_valid_L2P_spec_box0 = 1*((ds_L2P['flag_valid_pp_mean'].isel(n_posneg=isbabord).sum(dim=['nk', 'n_phi']).compute().data)>1)                                          
		ds_L2P.close()
		
		print('ntim = ',ntim)
		std_Hs_L2_2D = np.zeros((ntim,2,len(L1S)+1))
		std_Hs_L2S_2D = np.zeros((ntim,2,len(L1S)+1))
		if is1D:
			std_Hs_L2_1D = np.zeros((ntim))
			std_Hs_L2S_1D = np.zeros((ntim))
	
		Hs_box = np.zeros((ntim))
		Hs_box_param = np.zeros((ntim))
		std_Hs_box = np.zeros((ntim))
		lat_box = np.zeros((ntim))
		lon_box = np.zeros((ntim))
		
		Hs_L2_2D = np.zeros((ntim))
		Hs_L2S_2D = np.zeros((ntim))
		Lambda2_L2_2D = np.zeros((ntim))
		Lambda2_L2S_2D = np.zeros((ntim))
		
		if is1D:
			Hs_L2_1D = np.zeros((ntim))
			Hs_L2S_1D = np.zeros((ntim))
			var_env2_L2_1D = np.zeros((ntim))
		var_env2_L2_2D = np.zeros((ntim))
	
		for it in range(ntim):
			print(it, 'over ',ntim,' -------------')
			ds_CNES_sel = ds_boxes.isel(isBabord=isbabord,time0=it,n_beam_l2=nbeam)
			# --- 0. Get global values from box ------------------------------
			# Hs_box_param[it] = ds_CNES_sel['wave_param'].isel(nparam=0).compute().data
			std_Hs_box[it] = ds_CNES_sel['nadir_swh_box_std'].compute().data
			Hs_box[it] = ds_CNES_sel['nadir_swh_box'].compute().data
			time_box[it] = ds_CNES_sel['time_box'].compute().data
			lat_box[it] = ds_CNES_sel['lat'].compute().data
			lon_box[it] = ds_CNES_sel['lon'].compute().data
		
			if ds_CNES_sel['flag_valid_swh_box'] == 1:
				# print('flag == 1')
				std_Hs_L2_2D[it,:,:] = np.nan
				Lambda2_L2_2D[it] = np.nan
				Hs_L2_2D[it] = np.nan
				#var_env2_L2_2D[it] = np.nan
				std_Hs_L2S_2D[it,:,:] = np.nan
				Lambda2_L2S_2D[it] = np.nan
				Hs_L2S_2D[it] = np.nan
				if is1D :
					std_Hs_L2_1D[it] = np.nan
					Hs_L2_1D[it] = np.nan
					var_env2_L2_1D[it] = np.nan
					std_Hs_L2S_1D[it] = np.nan
					Hs_L2S_1D[it] = np.nan
			else:
				# print('flag == 0')
				ds_ODL_sel = ds_l2s.isel(time0=np.where((ind_time_box_2==it)&(ind_babord_box_2==isbabord))[0])
	
				# ---- 1. Spec L2 ---------------------------
				# --- for 1D spec --------------------
				if is1D:
					std_Hs_L2_1D[it],Hs_L2_1D[it],var_env2_L2_1D[it] = estimate_stdHs_from_spec1D(ds_CNES_sel[['k_vector','phi_vector','wave_spectra_kth_hs','dk']])
				# --- for 2D spec --------------------
				std_Hs_L2_2D[it,:,:],Hs_L2_2D[it],Lambda2_L2_2D[it] = estimate_stdHs_from_spec2D(ds_CNES_sel[['k_vector','phi_vector','wave_spectra_kth_hs','dk']], kKkPhi2s,L1S)
	    
				# ---- 2. Spec L2S ---------------------------
				# --- for 1D spec --------------------
				if is1D:
					std_Hs_L2S_1D[it],Hs_L2S_1D[it],_ = estimate_stdHs_from_spec1D(ds_ODL_sel[['k_vector','phi_vector','wave_spectra_kth_hs','dk']].swap_dims({'time0':'n_phi'}))
				# --- for 2D spec --------------------
				std_Hs_L2S_2D[it,:,:],Hs_L2S_2D[it],Lambda2_L2S_2D[it] = estimate_stdHs_from_spec2D(ds_ODL_sel[['k_vector','phi_vector','wave_spectra_kth_hs','dk']].swap_dims({'time0':'n_phi'}),kKkPhi2s,L1S)

		print(indf,' end of work in file')
		if is1D:
			return indf,time_box, Hs_box, std_Hs_box, lat_box, lon_box, std_Hs_L2_1D, Hs_L2_1D, var_env2_L2_1D, std_Hs_L2_2D, Hs_L2_2D, var_env2_L2_2D, std_Hs_L2S_1D, Hs_L2S_1D, std_Hs_L2S_2D, Hs_L2S_2D, flag_valid_L2P_swh_box0, flag_valid_L2P_spec_box0,Lambda2_L2_2D,Lambda2_L2S_2D
		else:
			return indf,time_box, Hs_box, std_Hs_box, lat_box, lon_box, std_Hs_L2_2D, Hs_L2_2D, var_env2_L2_2D, std_Hs_L2S_2D, Hs_L2S_2D, flag_valid_L2P_swh_box0, flag_valid_L2P_spec_box0,Lambda2_L2_2D,Lambda2_L2S_2D

	except Exception as inst:
		print(inst,indf,'line :',sys.exc_info()[2].tb_lineno, file_L2S)
		if is1D:
			return 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0
		else:
			return 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0

# ==================================================================================
# === 5.2 estimation for one box ( for-loop over all boxes in 1 track ) ============
# ==================================================================================      
def function_one_box(it,ds_CNES_sel,ds_ODL_sel,is1D,kKkPhi2s):
	try:
		# --- 0. Get global values from box ------------------------------
		# Hs_box_param[it] = ds_CNES_sel['wave_param'].isel(nparam=0).compute().data
		std_Hs_box = ds_CNES_sel['nadir_swh_box_std'].compute().data
		Hs_box = ds_CNES_sel['nadir_swh_box'].compute().data
		time_box = ds_CNES_sel['time_box'].compute().data
			
		if ds_CNES_sel['flag_valid_swh_box'] == 1:
			std_Hs_L2_2D = np.nan
			Hs_L2_2D = np.nan
			var_env2_L2_2D = np.nan
			std_Hs_L2S_2D = np.nan
			Hs_L2S_2D = np.nan
			if is1D :
				std_Hs_L2_1D = np.nan
				Hs_L2_1D = np.nan
				var_env2_L2_1D = np.nan
				std_Hs_L2S_1D = np.nan
				Hs_L2S_1D = np.nan
		else:
			# ---- 1. Spec L2 ---------------------------
			if is1D:
				std_Hs_L2_1D,Hs_L2_1D,var_env2_L2_1D = estimate_stdHs_from_spec1D(ds_CNES_sel[['k_vector','phi_vector','wave_spectra_kth_hs','dk']])
			std_Hs_L2_2D,Hs_L2_2D,var_env2_L2_2D = estimate_stdHs_from_spec2D(ds_CNES_sel[['k_vector','phi_vector','wave_spectra_kth_hs','dk']], kKkPhi2s)

			# ---- 2. Spec L2S ---------------------------
			if is1D:
				std_Hs_L2S_1D,Hs_L2S_1,_ = estimate_stdHs_from_spec1D(ds_ODL_sel[['k_vector','phi_vector','wave_spectra_kth_hs','dk']].swap_dims({'time0':'n_phi'}))
			std_Hs_L2S_2D,Hs_L2S_2D,_ = estimate_stdHs_from_spec2D(ds_ODL_sel[['k_vector','phi_vector','wave_spectra_kth_hs','dk']].swap_dims({'time0':'n_phi'}),kKkPhi2s)

			print(indf,' end of work in file')
			if is1D:
				return time_box, Hs_box, std_Hs_box, std_Hs_L2_1D, Hs_L2_1D, var_env2_L2_1D, std_Hs_L2_2D, Hs_L2_2D, var_env2_L2_2D, std_Hs_L2S_1D, Hs_L2S_1D, std_Hs_L2S_2D, Hs_L2S_2D
			else:
				return time_box, Hs_box, std_Hs_box, std_Hs_L2_2D, Hs_L2_2D, var_env2_L2_2D, std_Hs_L2S_2D, Hs_L2S_2D

	except Exception as inst:
		print(inst,indf, file_L2S)
		return 0,0,0,0,0,0,0,0

