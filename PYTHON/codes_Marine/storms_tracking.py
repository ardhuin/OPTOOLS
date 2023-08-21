'''
  Software for the tracking of eddies in
  OFAM model output following Chelton et
  al., Progress in Oceanography, 2011.
'''

# Load required modules
import os
import sys
import numpy as np
import storms_functions_v2 as storm

# Load parameters

exec("from "+ sys.argv[1]+" import *")
# from params import *

# Automated eddy tracking
d1str = date1[0:10]
d2str = date2[0:10]
filename = 'storm_det_'+d1str+'_'+d2str+'_v4.npz' # v2 is the old one, v3 is with amp threshold: 1/2, v4 : 1/5
fileload = os.path.join(data_dir,filename)
data = np.load(fileload,allow_pickle=True)

det_storms = data['storms'] # len(eddies) = number of time steps
T = len(det_storms)
# Initialize storms discovered at first time step

storms = storm.storms_init(det_storms)

# Stitch eddy tracks together at future time steps

print( 'storms tracking started')
print("number of time steps to loop over: ",T)


# start at 1 because 0 is the initialization
for tt in range(1, T):

    print( "timestep: " ,tt+1,". out of: ", T)

    # Track eddies from time step tt-1 to tt and update corresponding tracks and/or create new storms
    # there are 2 types of coordinates lon/lat (center of mass, 'lon_center',  and maximum 'lon_max'),
    # 'lon_max' and 'lat_max' are stated to select the type of coordinates used to track the storm
    storms = storm.track_storms(storms, det_storms, tt, dt, max_missed_dt, max_spd, storm_scale_min, storm_scale_max,'lon_max','lat_max','amp_max')
    # Save data incrementally

    if( np.mod(tt, dt_save)==0 ):
        filename = 'storm_track_'+d1str+'_'+d2str+'_v3-0'
        filesave = os.path.join(data_dir,filename)
        np.savez(filesave, storms=storms)

# Add keys for storm age and flag if storm was still in existence at end of run

for ed in range(len(storms)):
    storms[ed]['age'] = len(storms[ed]['lon_max'])

filename = 'storm_track_'+d1str+'_'+d2str+'_v5-4b' # v4-4 : det=v4 + is_possible: isnear*similaramp*similar area
filesave = os.path.join(data_dir,filename)
np.savez(filesave, storms=storms)
