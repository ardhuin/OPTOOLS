'''
  Software for the tracking of eddies in
  OFAM model output following Chelton et
  al., Progress in Oceanography, 2011.
'''
'''
  Modified to find Hs storms by Marine De Carlo
'''

# Load required modules

import numpy as np
import os
import sys
import matplotlib
import multiprocessing as mp
# Turn the followin on if you are running on storm sometimes - Forces matplotlib to not use any Xwindows backend.
#matplotlib.use('Agg')


from matplotlib import pyplot as plt
import storms_functions_v2 as storm

# Load parameters as input from code line

exec("from "+ sys.argv[1]+" import *")
#from params import *

is_mpi= 1
# initialize MPU
#

# Initialize lists

lon_storms_center = []
lat_storms_center = []
lon_storms_max = []
lat_storms_max = []
amp_storms_max = []
amp_storms_mean = []
area_storms = []
scale_storms = []
label_storms = []

r_i = []
r_lon_cent = []
r_lat_cent = []
r_lon_max = []
r_lat_max = []
r_amp_max = []
r_amp_mean = []
r_area = []
r_scale = []
r_labelStorms = []


# define the time vector
d1 = np.datetime64(date1)
d2 = np.datetime64(date2)
dday = np.timedelta64(dt,'h')

days_vec = np.arange(d1,d2,dday)
T = len(days_vec)

d1str = date1[0:10]
d2str = date2[0:10]

gen_name_labelstorm = 'storm_det_labels_'+d1str+'_'+d2str


if is_mpi:
    pool = mp.Pool(mp.cpu_count())
    results = pool.starmap_async(storm.investigate_detection_t, [(i,days_vec,filenameFormat,input_dir, cut_lon, cut_lat, swh_levels, Npix_min, amp_thresh, d_thresh_min, d_thresh_max,thresh_height_for_scale,data_dir, gen_name_labelstorm,0) for i in range(T)]).get()
# results : i,lon_cent, lat_cent, lon_max, lat_max , amp_max , amp_mean , area , scale , labelStorms
# close pool MPU
    pool.close()
 

    r_i = [result[0] for result in results]
    r_lon_cent = [result[1] for result in results]
    r_lat_cent = [result[2] for result in results]
    r_lon_max = [result[3] for result in results]
    r_lat_max = [result[4] for result in results]
    r_amp_max = [result[5] for result in results]
    r_amp_mean = [result[6] for result in results]
    r_area = [result[7] for result in results]
    r_scale = [result[8] for result in results]

else:
    for i in range(T):
        i,lon_cent, lat_cent, lon_max, lat_max , amp_max , amp_mean , area , scale , labelStorms = storm.investigate_detection_t(i,days_vec,filenameFormat,input_dir, cut_lon, cut_lat, swh_levels, Npix_min, amp_thresh, d_thresh_min, d_thresh_max,thresh_height_for_scale,data_dir, gen_name_labelstorm)
        r_i.append(i)
        r_lon_cent.append(lon_cent)
        r_lat_cent.append(lat_cent)
        r_lon_max.append(lon_max)
        r_lat_max.append(lat_max)
        r_amp_max.append(amp_max)
        r_amp_mean.append(amp_mean)
        r_area.append(area)
        r_scale.append(scale)
        r_labelStorms.append(labelStorms)
    


storms = storm.storms_list(r_i,days_vec,r_lon_cent, r_lat_cent, r_amp_max, r_amp_mean, r_area, r_scale, r_lon_max, r_lat_max)


filename = 'storm_det_'+d1str+'_'+d2str+'_v2-0'
filesave = os.path.join(data_dir,filename)

np.savez(filesave, storms=storms)

