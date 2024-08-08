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
import struct


def read_one_WAVEX_file(filename,verbose=0):
    #f=open('MIR_20230405_000031501_POL.DF047')
    #A = np.fromfile(f, 'uint8')
    f=open(filename, 'rb')
    B = f.read() 
# File size is 248940 bytes
# 
# 0-9 bytes: DF-047-003           :  file format + version 
# 11-33 ??
# 34-58 : 2023-04-13 14:43:01.224Z  : date
# 58:61 : 3.2330  : vessel speed 
# 62:65 : 336.260 : vessel heading
# 66:69 : 337.8500 : vessel track 
# 70:73 : 4.50454521179199  longitude 
# 74:77 : 40.570690155   latitude 
# 78:81 : 6.94612264   wind speed   (-999 if missing)
# 82:85 : 323.0        wind dir     ( -999.98999023437 if missing) 
# 86:89 : 7.15   wind speed   (-999 if missing)
# 90:93 : 326.0        wind dir     ( -999.98999023437 if missing) 
# 94:97 :                           ( -999.98999023437 if missing) 
# 98:101 :                           ( -999.98999023437 if missing) 
# 103-200 : Furuno                                            FAR-2xx7, 24 RPM, 1.98 m, Atalante             '   
# 207:210:  1.98    : antenna length
# 211:214:  1.23    : antenna beamwidth 
# 751:752:  282     : number of ranges
# 755:758   249.6   : first range 
# 759:762   4.8     : range resolution 
# 763:764:  440     : number of azimuths
# 767:770:  258     : first azimuth 
# 771:774:  0.6     : angular resolution (deg)

#datetime=np. asarray(struct.unpack('24s',struct.pack('24s',B[34:58])))
    datetime=B[34:58].decode('UTF-8')

    [heading]=np.asarray(struct.unpack('<f',struct.pack('4s',B[62:66])))

    [lon]=np.asarray(struct.unpack('<f',struct.pack('4s',B[70:74])))
    [lat]=np.asarray(struct.unpack('<f',struct.pack('4s',B[74:78])))

    [nr,n1]=np.asarray(struct.unpack('<HH',struct.pack('4s',B[751:755])))
    [na,n2]=np.asarray(struct.unpack('<HH',struct.pack('4s',B[763:767])))


    [r0]=np.asarray(struct.unpack('<f',struct.pack('4s',B[755:759])))
    [dr]=np.asarray(struct.unpack('<f',struct.pack('4s',B[759:763])))


    [a0]=np.asarray(struct.unpack('<f',struct.pack('4s',B[767:771])))
    [da]=np.asarray(struct.unpack('<f',struct.pack('4s',B[771:775])))
 

#for i in range(780):
#   B2=struct.pack('4s',B[i:4+i])
#   print(i, 'B2:',B2, 'A:',A[i:i+4])
#   [x] =struct.unpack('<f', B2)
#   print('x:',x)
#   [x,y] =struct.unpack('<HH', B2)
#   print('x,y:',x,y)

    BB=np. asarray(struct.unpack('<124080H', B[780:248941]))
    mat2 = BB.reshape((na,nr))
    if verbose==1 :
        #print('Size of A array:',np.shape(A))
        print('Size of B array:',np.shape(B))
        print('datetime:',datetime)
        print('heading:',heading)
        print('lon,lat:',lon,lat)
        print('nr,na:',nr,na)
        print('first range, range resolution:',r0,dr)
        print('first azimuth, azimuth resolution:',a0,da)

    return lon,lat,datetime,nr,na,mat2,r0,dr,a0,da,heading


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


