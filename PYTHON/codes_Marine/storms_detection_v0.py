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
# Turn the followin on if you are running on storm sometimes - Forces matplotlib to not use any Xwindows backend.
#matplotlib.use('Agg')


from matplotlib import pyplot as plt
import storms_functions_v2 as storm

# Load parameters as input from code line

exec("from "+ sys.argv[1]+" import *")
#from params import *

# Initialize lists

lon_storms_center = []
lat_storms_center = []
lon_storms_max = []
lat_storms_max = []
amp_storms_max = []
amp_storms_mean = []
area_storms = []
scale_storms = []

# define the time vector
d1 = np.datetime64(date1)
d2 = np.datetime64(date2)
dday = np.timedelta64(dt,'h')

days_vec = np.arange(d1,d2,dday)
T = len(days_vec)

print('storm detection started')
print("number of time steps to loop over: ",T)
for tt in range(T):
    print("timestep: ",tt+1,". out of: ", T)
    # get time and filename
    datett = days_vec[tt]
    filename = storm.get_filename(filenameFormat,datett)
    # Load map of significant wave height (SWH)
    
    lon, lat, swh = storm.load_swh(input_dir,filename,datett)
    #storm.quick_plot(swh,findrange=True)
    # 
    ## Spatially filter SSH field
    # 
    swh_filt = storm.spatial_filter(swh, lon, lat, 0.5, cut_lon, cut_lat)
    #storm.quick_plot(swh_filt,findrange=True)
    # 
    ## Detect lon and lat coordinates of storms
    # lon_storms_center, lat_storms_center, lon_storm_max, lat_storm_max, amp_storm_max, amp_storm_mean, area_storm, scale_storm
    # (field, lon, lat, levels, Npix_min, amp_thresh, d_thresh):
    lon_cent, lat_cent, lon_max, lat_max, amp_max, amp_mean, area, scale, labelStorms = storm.detect_storms(swh_filt, lon, lat, swh_levels, Npix_min, amp_thresh, d_thresh_min, d_thresh_max,thresh_height_for_scale)
    #print("%d storms found (center) " % len(lon_cent))
    #print("%d storms found (max)" % len(lon_max))
    #print(scale)
    lon_storms_center.append(lon_cent)
    lat_storms_center.append(lat_cent)
    lon_storms_max.append(lon_max)
    lat_storms_max.append(lat_max)
    amp_storms_max.append(amp_max)
    amp_storms_mean.append(amp_mean)
    area_storms.append(area)
    scale_storms.append(scale)

    # Plot map of filtered SSH field

    storms_tt=(lon_storms_center[tt],lat_storms_center[tt])
    # storm.detection_plot(tt,lon,lat,swh,swh_filt,storms_tt,plot_dir,findrange=False)


# Combine eddy information from all days into a list

storms = storm.storms_list(lon_storms_center, lat_storms_center, amp_storms_max, amp_storms_mean, area_storms, scale_storms, lon_storms_max, lat_storms_max)

d1str = date1[0:10]
d2str = date2[0:10]
filename = 'storm_det_'+d1str+'_'+d2str+'_v4'# v3 : amp_thresh = 1/2, v4: amp_thresh = 1/5
filesave = os.path.join(data_dir,filename)

np.savez(filesave, storms=storms)
