'''
    A set of functions to accompany the eddy tracking software
'''

import os
import sys
import numpy as np
import scipy as sp
import numpy.linalg as linalg
import scipy.signal as signal
import scipy.ndimage as ndimage
import scipy.interpolate as interpolate
import glob
#import ipdb
import datetime as dtm
# from datetime import datetime
import matplotlib
# Turn the followin on if you are running on storm sometimes - Forces matplotlib to not use any Xwindows backend.
#matplotlib.use('Agg')

import matplotlib.pyplot as plt
#import pandas as pd

import netCDF4 as nc

from itertools import repeat

from area import area 

exec("from "+ sys.argv[1]+" import *")

import re

def mkdir(p):
    """make directory of path that is passed"""
    import os
    try:
       os.makedirs(p)
       print("output folder: "+p+ " does not exist, we will make one.")
    except OSError as exc: # Python >2.5
       import errno
       if exc.errno == errno.EEXIST and os.path.isdir(p):
          pass
       else: raise


def find_nearest(array, value):
    idx=(np.abs(array-value)).argmin()
    return array[idx], idx

def nanmean(array, axis=None):
    return np.mean(np.ma.masked_array(array, np.isnan(array)), axis)


def restrict_lonlat(lon, lat, lon1, lon2, lat1, lat2):
    '''
    Restricts latitude and longitude vectors given
    input limits.
    '''

    tmp, i1 = find_nearest(lon, lon1)
    tmp, i2 = find_nearest(lon, lon2)
    tmp, j1 = find_nearest(lat, lat1)
    tmp, j2 = find_nearest(lat, lat2)

    lon = lon[i1:i2+1]
    lat = lat[j1:j2+1]

    return lon, lat, i1, i2, j1, j2

def get_filename(filenameFormat,datei):
    '''
    Replace regular dateformat in filename with the actual date
    ss : sec
    mm : mins
    hh : hours
    DD : day
    MM : month
    YYYY/YY : year
    '''
    filename = filenameFormat
    # Check if ss, if mm, if hh, if DD, if MM, if YY or YYYY
    # /!\ Watch out ! mm is minutes and MM is month !
    outp = re.findall(r"s{2}",filenameFormat)
    if len(outp)>0:
        s = datei.astype(object).second
        filename = re.sub(r"s{2}",str(s).zfill(len(outp)),filename)
        
    outp = re.findall(r"m{2}",filenameFormat)
    if len(outp)>0:
        mins = datei.astype(object).minute
        filename = re.sub(r"m{2}",str(mins).zfill(2),filename)
    
    outp = re.findall(r"h{2}",filenameFormat)
    if len(outp)>0:
        hh = datei.astype(object).hour
        filename = re.sub(r"h{2}",str(hh).zfill(2),filename)   
        
    outp = re.findall(r"D{2,}",filenameFormat)
    if len(outp) > 0:
        if len(outp[0])>0:
            DD = datei.astype(object).day
            filename = re.sub(r"D{2,}",str(DD).zfill(len(outp[0])),filename)
    
    outp = re.findall(r"M{2}",filenameFormat)
    if len(outp)>0:
        MM = datei.astype(object).month
        filename = re.sub(r"M{2}",str(MM).zfill(2),filename)
        
    outp = re.findall(r"Y{1}",filenameFormat)
    if len(outp)==2:
        YY = datei.astype(object).year
        filename = re.sub(r"Y+",str(YY-2000).zfill(2),filename)   
    elif len(outp)==4:
        YY = datei.astype(object).year
        filename = re.sub(r"Y+",str(YY).zfill(4),filename) 
        
    return filename


def load_swh(path,filename,datei):
    '''
    Loads significant wave heigth from WW3 output
    Inputs: path + name of file, date1
    Outputs : lon, lat, swh, swh_miss
    '''

    filename_complete = os.path.join(path,filename)
    ds = nc.Dataset(filename_complete,'r')
    variables_nc = ds.variables.keys()
    
    # import latitude
    regex = re.compile(r"(l|L)(AT|at){1}(ITUDE|itude)?$")
    for var in variables_nc:
        if regex.match(var) is not None:
            latvar = var
    
    lat = ds.variables[latvar][:]        
            
    # import longitude
    regex = re.compile(r"(l|L)(on|ON){1}((g|G){1}(ITUDE|itude)?)?$")
    for var in variables_nc:
        if regex.match(var) is not None:
            lonvar = var
    
    lon = ds.variables[lonvar][:]       
            
    # import time
    regex = re.compile(r"((T|t){1}(( |ime|IME){1})|(D|d){1}( |ay|AY|ATE|ate){1})")
    for var in variables_nc:
        if regex.match(var) is not None:
            timevar = var
            print(var)

    print('selected var : '+timevar)    
    times = ds.variables[timevar][:]
    # times = ds.variables['time'][:]
    print(np.shape(times))
    # times is expressed in days from 1/1/1990
    times_datetime = np.datetime64('1990-01-01T00:00:00')+ (times*24).astype('timedelta64[h]')
    # times_offset = dtm.datetime(1990,1,1) 
    # print(dtm.datetime(1990,1,1))
    # times_delta = times * dtm.timedelta(days=1) 
    # times_datetime = times_offset+times_delta
    print(times_datetime[-1])
    print(datei)    
    # get the index of the current time step
    it = np.argmin(abs(times_datetime-datei))
    print('it = '+str(it))
    # extract significant wave height only for current time step
    # two ways of doing it : 
    # 1. by getting the Masked Array read: swh0, therefore a mask with the missing values is included in the output.
    # this occurs if there is nothing at the 'X' position between the variable name and the index : ds.variables['hs']X[i,:,:])
    print(np.shape(ds.variables['hs'][:]))
    swh0 = ds.variables['hs'][it,:,:]    
    swh = swh0.data
    swh[swh0.mask] = np.nan
    # 2. by selecting only the 'data' array : ds.variables['hs'][:][i,:,:] with a [:] at the previous 'X' position
    # swh = ds.variables['hs'][:][i,:,:]
    # swh_miss = ds.variables['hs']._FillValue
    # swh[swh==swh_miss] = np.nan
       
    # One may note that the variable has already been scaled according to the 'scale_factor' attribute /!\ Automatic with the variable reading
    ds.close()

    return lon, lat, swh

def quick_plot(field,title='field',findrange=False):
    '''
    Create quick interactive diagnostic plot to double check eddy_detection is doing what we want...
    '''
    y,x=np.meshgrid(np.arange(field.shape[1]),np.arange(field.shape[0]))
    plt.clf()

    if not findrange:
        plt.contourf(y, x, field)#, levels=np.arange(-2.5,2.5,0.05))
    else:
        if np.isnan(np.sum(field)):
            plotfield=np.nan_to_num(field)
            print('range of field (with nan) is:')
            print('min',np.min(plotfield))
            print('max',np.max(plotfield))

            plt.contourf(y, x, field,levels=np.linspace(np.min(plotfield),np.max(plotfield),50))

        else:
            print('range of field is:')
            print('min',np.min(field))
            print('max',np.max(field))

            plt.contourf(y, x, field,levels=np.linspace(np.min(field),np.max(field),50))
    plt.title(title)
    plt.colorbar()
    plt.show()
  #  ipdb.set_trace()
  
def quick_plot_storm(field,lon,lat,lon_storms1,lat_storms1,lon_storms2,lat_storms2,findrange=False):
    '''
    Create quick interactive diagnostic plot to double check eddy_detection is doing what we want...
    '''
    x,y = lat,lon
    # y,x=np.meshgrid(np.arange(field.shape[1]),np.arange(field.shape[0]))
    plt.clf()

    if not findrange:
        plt.contourf(y, x, field, levels=np.arange(-2.5,2.5,0.05))
    else:
        if np.isnan(np.sum(field)):
            plotfield=np.nan_to_num(field)
            print('range of field (with nan) is:')
            print('min',np.min(plotfield))
            print('max',np.max(plotfield))

            plt.contourf(y, x, field,levels=np.linspace(np.min(plotfield),np.max(plotfield),50))

        else:
            print('range of field is:')
            print('min',np.min(field))
            print('max',np.max(field))

            plt.contourf(y, x, field,levels=np.linspace(np.min(field),np.max(field),50))
    if len(lon_storms1)>0:
        for ij in range(len(lon_storms1)):
                plt.plot([lon_storms1[ij], lon_storms2[ij]],[lat_storms1[ij], lat_storms2[ij]],linestyle='-',marker='*',color='r')
        plt.plot(lon_storms2, lat_storms2,linestyle='none',marker='*',color='m')
        
    plt.title('diagnostic plot')
    plt.colorbar()
    plt.show()
  #  ipdb.set_trace()

def remove_missing(field, missing, replacement):
    '''
    Replaces all instances of 'missing' in 'field' with 'replacement'
    
    OBSOLETE as netCDF opening does everything
    '''

    field[field==missing] = replacement

    return field


def interp_nans(data, indices):
    '''
    Linearly interpolates over missing values (np.nan's) in data
    Data is defined at locations in vector indices.
    '''

    not_nan = np.logical_not(np.isnan(data))

    return np.interp(indices, indices[not_nan], data[not_nan])


def match_missing(data1, data2):
    '''
    Make all locations that are missing in data2 also missing in data1
    Missing values are assumed to be np.nan.
    '''

    data1[np.isnan(data2)] = np.nan
    return data1


def spatial_filter(field, lon, lat, res, cut_lon, cut_lat):
    '''
    Performs a spatial filter, removing all features with
    wavelenth scales larger than cut_lon in longitude and
    cut_lat in latitude from field (defined in grid given
    by lon and lat).  Field has spatial resolution of res
    and land identified by np.nan's
    '''

    field_filt = np.zeros(field.shape)

    # see Chelton et al, Prog. Ocean., 2011 for explanation of factor of 1/5
    sig_lon = (cut_lon/5.) / res
    sig_lat = (cut_lat/5.) / res

    land = np.isnan(field)
    field[land] = nanmean(field)

    # field_filt = field - ndimage.gaussian_filter(field, [sig_lat, sig_lon])
    field_filt = ndimage.gaussian_filter(field, [sig_lat, sig_lon])
    field_filt[land] = np.nan

    return field_filt

def area_matrix(lons,lats):
    '''Calculates the area defined by a polygon (in squared km) based on the area package that returns an area [m**2].
    Source: https://github.com/scisco/area.
    This fonction returns an area in squared km.'''

    X = len(lons)
    Y = len(lats)
    assert X == Y, 'area_matrix : lons and lats must have same number of elements'
    coords = [[]]
    for k in range(X):
        coords[0].append([])
        coords[0][k].append(lons[k])
        coords[0][k].append(lats[k])
        
    obj = {'type':'Polygon','coordinates':coords}
    area_km2 = area(obj)* 10**-6
    return area_km2

def distance_vect(lon0,lat0,lons,lats):
    '''Calculates the distances (in km) between (lon0,lat0) and any position in (lons,lats) based on the formulas
    c = sin(lati1)*sin(lati2)+cos(longi1-longi2)*cos(lati1)*cos(lati2)
    d = EARTH_RADIUS*Arccos(c)
    where EARTH_RADIUS is in km and the angles are in radians.
    Source: http://mathforum.org/library/drmath/view/54680.html
    This function returns the matrix.'''

    EARTH_RADIUS = 6378.1
    X = len(lons)
    Y = len(lats)
    assert X == Y, 'lons and lats must have same number of elements'

    d = np.zeros(X)
    
    c = np.sin(np.radians(lat0)) * np.sin(np.radians(lats)) + \
        np.cos(np.radians(lon0-lons)) * \
        np.cos(np.radians(lat0)) * np.cos(np.radians(lats))
    
    d[c<1] = EARTH_RADIUS * np.arccos(c[c<1])

    return d


def distance_matrix(lons,lats):
    '''Calculates the distances (in km) between any two cities based on the formulas
    c = sin(lati1)*sin(lati2)+cos(longi1-longi2)*cos(lati1)*cos(lati2)
    d = EARTH_RADIUS*Arccos(c)
    where EARTH_RADIUS is in km and the angles are in radians.
    Source: http://mathforum.org/library/drmath/view/54680.html
    This function returns the matrix.'''

    EARTH_RADIUS = 6378.1
    X = len(lons)
    Y = len(lats)
    assert X == Y, 'lons and lats must have same number of elements'

    d = np.zeros((X,X))

    #Populate the matrix.
    for i2 in range(len(lons)):
        lati2 = lats[i2]
        loni2 = lons[i2]
        c = np.sin(np.radians(lats)) * np.sin(np.radians(lati2)) + \
            np.cos(np.radians(lons-loni2)) * \
            np.cos(np.radians(lats)) * np.cos(np.radians(lati2))
        d[abs(c)<1,i2] = EARTH_RADIUS * np.arccos(c[abs(c)<1])

    return d

def get_regions_without_duplicate(regions,regions_origin,regions_duplicate):
    uregions_origin = np.unique(regions_origin)
    # print(uregions_origin)
    uregions_duplicate = np.unique(regions_duplicate)
    uregions = np.unique(regions)
    regions_duplicated_numbers = -100*np.ones(regions.shape)
    for k in range(1,len(uregions_origin)):
        iregion = uregions_origin[k]
        # --- 1.0.3 prepare index for duplicate field -----------------
        ind2 = np.zeros(np.shape(regions)).astype(bool)
        # --- 1.0.4 loop over the regions found in the duplicate ------

            # fieldnan[regions==iregion] = np.nan
        if sum(iregion == uregions_duplicate)==0:
        # --- if the label is not present on duplicate: there are two possibilities:  either the region is present with another name in duplicate (which leads to 2 possibilities again : total duplicate or 180/-180 wrapping) or the region is not present at all in duplicate
                # and the original region is called iregion
            nb_duplicate = np.unique(regions_duplicate[regions_origin==iregion])
            if nb_duplicate[0]==-100:
            # region is not present at all in duplicate
                regions_duplicated_numbers[regions==iregion]=iregion
            else:
                if sum(nb_duplicate[0] == uregions_origin)>0 : # the duplicated region also exists in the origin: -180/180 wrapping 
                # in regions : set the origin to -100
                    regions_duplicated_numbers[regions==iregion]=iregion
                    regions[regions==iregion] = -100 # the origin only label is set to -100
                    regions_duplicated_numbers[regions==nb_duplicate[0]]=iregion
                    regions[regions==nb_duplicate[0]]=iregion
                else: # the duplicated region is a total duplicate
                # in regions : set the duplicate to -100
                    regions_duplicated_numbers[regions==iregion]=iregion
                    regions_duplicated_numbers[regions==nb_duplicate[0]]=iregion  
                    regions[regions==nb_duplicate[0]]=-100
        else:
        # if the label is present in duplicate : 180/-180 wrapping
            regions_duplicated_numbers[regions==iregion]=iregion

    return regions, regions_duplicated_numbers
    
    
def store_storm(field, region, lon, lat, llon2, llat2, lon_storm_center, lat_storm_center, lon_storm_max, lat_storm_max, amp_storm_max, amp_storm_mean, area_storm, scale_storm):
    '''
    When a storm reaches the conditions it is stored 
    '''
    # --- 1. find centre of mass of storm
    storm_object_with_mass = field * region
    #     turning the nan to 0 is just a back-up, because there are not supposed to be nan in storm_object_with_mass
    storm_object_with_mass[np.isnan(storm_object_with_mass)] = 0
    j_cen, i_cen = ndimage.center_of_mass(storm_object_with_mass)
    lon_cen = np.interp(i_cen%720, range(0,len(lon)), lon)
    lat_cen = np.interp(j_cen, range(0,len(lat)), lat)
    lon_storm_center = np.append(lon_storm_center, lon_cen)
    lat_storm_center = np.append(lat_storm_center, lat_cen)
    # --- 2. find max of storm
    j_cen, i_cen = ndimage.maximum_position(storm_object_with_mass)
    lon_storm_max = np.append(lon_storm_max, lon[i_cen%720])
    lat_storm_max = np.append(lat_storm_max, lat[j_cen])
                
    # --- 3. assign (and calculate) amplitude
    interior = ndimage.binary_erosion(region)
    exterior = np.logical_xor(region, interior)
    
    amp_max = field[interior].max()
    amp_mean = field[region].mean()
    amp_storm_max = np.append(amp_storm_max, amp_max)
    amp_storm_mean = np.append(amp_storm_mean, amp_mean)
                
    # --- 4. assign (and calculate) area, and scale of eddies
    lon_ext = llon2[exterior]
    lat_ext = llat2[exterior]
    area_km2 = area_matrix(lon_ext, lat_ext)
    area_storm = np.append(area_storm, area_km2)
    d = distance_matrix(lon_ext, lat_ext)
    scale = 0.5*d # [km]
    scale_storm = np.append(scale_storm, scale)
                    
    return lon_storm_center, lat_storm_center, lon_storm_max, lat_storm_max, amp_storm_max, amp_storm_mean, area_storm, scale_storm
    
def modify_area_scale(label_regions_old_saved,lon_ext,lat_ext,area_storm,scale_storm,amp_storm_mean,amp_mean):
    area_km2 = area_matrix(lon_ext, lat_ext)
    area_storm[label_regions_old_saved] = area_km2
    d = distance_matrix(lon_ext, lat_ext)
    scale = 0.5*d.max() # [km]
    scale_storm[label_regions_old_saved] = scale
    amp_storm_mean[label_regions_old_saved] = amp_mean
    
    return area_storm, scale_storm, amp_storm_mean

def detect_storms(field, lon, lat, levels, Npix_min, amp_thresh, d_thresh_min, d_thresh_max,thresh_height_for_scale):
    '''
    Detect storms present in field 
    based on detect_eddies : which satisfy the criteria
    outlined in Chelton et al., Prog. ocean., 2011, App. B.2.
    Field is a 2D array specified on grid defined by lat and lon.
    ssh_crits is an array of ssh levels over which to perform
    eddy detection loop
    res is resolutin in degrees of field
    Npix_min, Npix_max, amp_thresh, d_thresh specify the constants
    used by the eddy detection algorithm (see Chelton paper for
    more details)
    cyc = 'cyclonic' or 'anticyclonic' [default] specifies type of
    eddies to be detected
    Function outputs lon, lat coordinates of detected eddies
    '''

    len_deg_lat = 111.325 # length of 1 degree of latitude [km]

    llon, llat = np.meshgrid(lon, lat)
    NX = len(lon)
    NY = len(lat)

    lon_storm_center = np.array([])
    lat_storm_center = np.array([])
    lon_storm_max = np.array([])
    lat_storm_max = np.array([])
    amp_storm_max = np.array([])
    amp_storm_mean = np.array([])
    area_storm = np.array([])
    scale_storm = np.array([])
    is_growing_storm = np.array([])
    
    # levels decreasing
    levels.sort()
    levels = np.flipud(levels)
    
    # Duplicate to avoid  problems due to wraping longitude
    llon2 = np.concatenate((llon,llon[:,range(int(NX/2))]+360.),axis=1)
    llat2 = np.concatenate((llat,llat[:,range(int(NX/2))]),axis=1)
    field2 = np.concatenate((field,field[:,range(int(NX/2))]),axis=1)
    
    # initialize the mask of already accounted storms
    fieldnan = np.ones(field2.shape)
    labelStorms = np.zeros(field2.shape) -100.
     
    count_storms=0
    # initialize land
    land = np.isnan(field2)
    field2[land] = 0.
    
    # fieldnan contains nan for continents and storms that have already (at previous levels) been categorised as storms
    fieldnan[land] = np.nan
    # regionsOld contains regions (=labels for storms) from previous step (= level)  
    regionsOld2 = np.zeros(field2.shape)-100.
    
    # --- plots to check (if needed)
 #   quick_plot(fieldnan,title='fieldnan')
    # quick_plot_storm(fieldnan,llon,llat,lon_storm_center,lat_storm_center,lon_storm_center,lat_storm_center,findrange=True)
    # loop over levels and remove interior pixels of detected eddies from subsequent loop steps
    for ilev in levels:
 
    # --- 1. Find all regions with Hs greater than ilev  ---------------
        # quick_plot(field2,title='field2',findrange=True)
        regions, nregions = ndimage.label( (field2>ilev).astype(int) )
        regions[regions==0]=-100
        # '-100' was set up to have an visible difference in the check-plot
        # for info: regions = lon (-180,360)
        # regionsOld1 = regions.copy()
        # --- 1.0 remove duplicate detections --------------------------
        # --- 1.0.1 separate regions into 2 parts: the origin [-180;180] and the duplication [180; 360] -----------------------------------------
        regions_origin = regions[:,range(NX)]
        regions_duplicate = -100*np.ones(np.shape(field))
        regions_duplicate[:,range(int(NX/2))] = regions[:,NX:]
        
        # --- 1.0.2 Apply function to select  only on region: origin or duplicate  -----------------------------------------
        # regions, regions_duplicated_numbers = 
        regions, regions_duplicated_numbers = get_regions_without_duplicate(regions,regions_origin,regions_duplicate)
        regionsOld1 = regions_duplicated_numbers.copy()
        uregions = np.unique(regions)

        for iiregion in range(1,len(uregions)):
            iregion = uregions[iiregion]
            # print(str(iregion)+' over '+str(nregions))

    # --- 2. Calculate number of pixels comprising detected region, reject if not greater than Npix_min ---------------------------------------------
            region = (regions==iregion)
            region_Npix = region.astype(int).sum()
            eddy_area_within_limits = (region_Npix > Npix_min)
    # --- 3. Look if the storm has already been stored (the field is turned to nan in fieldnan for a stored storm) -----------------------------       
            regions_nan_mask = np.zeros(field2.shape)
            regions_nan_mask[region] = fieldnan[region]
            has_nan_in = np.any(np.isnan(regions_nan_mask))
            # --- has_nan_in : if there is a nan in the region. Basically means that the region is already stored please go to next region! ... However, it can happen that the new region contains previous older regions, with only one saved. Therefore we look at the number of "old regions", not turned to nan, in the new region
            # print(region)
            
            if has_nan_in:
            # There is a nan in the region 
                # get the name of the old regions situated inside the "region" zone and that 
                regionsOld_nan_mask = np.zeros(field2.shape)-100
                regionsOld_nan_mask[region] = regionsOld2[region]
                uregions_old_not_saved = np.unique(regionsOld_nan_mask[np.logical_not(np.isnan(regions_nan_mask))])
                uregions_old_not_saved = uregions_old_not_saved[uregions_old_not_saved>=0] # remove  -100: corresponding to background
                
                label_regions_old_saved = np.unique(labelStorms[np.isnan(regions_nan_mask)]) 
                label_regions_old_saved = label_regions_old_saved[label_regions_old_saved>=0].astype(int)
                # print(is_growing_storm[label_regions_old_saved])
                has_storm_still_growing = np.any(is_growing_storm[label_regions_old_saved])
                has_old_storm_forgotten = len(uregions_old_not_saved)>1   # has_old_storm_forgotten : there was a storm that was not stored // if the other storm is expandin this storm won't be stored
                if has_storm_still_growing: # if has storm still growing : get the max of the Hs_max of the "saved storms" all others storm will be set to stop growing
                    if len(label_regions_old_saved)>1:
                    # the biggest will continue increasing, the other will stop
                        is_growing_storm[label_regions_old_saved]=0
                        ind = np.argmax(amp_storm_max[label_regions_old_saved])
                        is_growing_storm[label_regions_old_saved[ind]]=1
                        # Set the outside storm to be equal to the one with max amplitude
                        labelStorms[np.logical_and(regions_duplicated_numbers == iregion,regionsOld_nan_mask==-100)]=label_regions_old_saved[ind]
                        label_regions_old_saved = label_regions_old_saved[ind]
                        
                    if ilev>=thresh_height_for_scale: # increase the area of growing storm
                    # name the whole region as the growing storm
                        labelStorms[regions_duplicated_numbers == iregion] = label_regions_old_saved
                    # add the whole region as nan in the fieldnan
                        fieldnan[regions_duplicated_numbers == iregion] = np.nan
                    # compute the new scale/area and add them 
                        interior = ndimage.binary_erosion(region)
                        exterior = np.logical_xor(region, interior)
                        lon_ext = llon2[exterior]
                        lat_ext = llat2[exterior]
                        amp_mean = field2[region].mean()
                        area_storm, scale_storm, amp_storm_mean = modify_area_scale(label_regions_old_saved, lon_ext, lat_ext, area_storm, scale_storm, amp_storm_mean, amp_mean)
                    else:
                        is_growing_storm[label_regions_old_saved]=0      
                else:        
                    if has_old_storm_forgotten:
                    # --- region contains one storm at least that has been stored and one or more that have not been stored -----------------------------------------------------------------
                    # --- 3.2.0 recalculate info for the storm stored --------------------
                        max_area_old = np.max(area_storm[label_regions_old_saved])
                        for iregOld in range(len(uregions_old_not_saved)):
                            regionOld = (regionsOld_nan_mask == uregions_old_not_saved[iregOld])
                            interior_old = ndimage.binary_erosion(regionOld)
                            exterior_old = np.logical_xor(regionOld, interior_old)                        
                            
                            if interior_old.sum() == 0:
                                continue
                        
                            lon_old = llon2[exterior_old]
                            lat_old = llat2[exterior_old]
                            area_km2 = area_matrix(lon_old, lat_old)
                            
                            # is_medium_tall = 
                            # --- 3.2.3 Save if not too small ----------
                            if area_km2 >= max_area_old/10.:#np.logical_or(area_km2 >= max_area_old/10.,is_medium_tall) :
                                lon_storm_center, lat_storm_center, lon_storm_max, lat_storm_max, amp_storm_max, amp_storm_mean, area_storm, scale_storm = store_storm(field2, regionOld, lon, lat, llon2, llat2, lon_storm_center, lat_storm_center, lon_storm_max, lat_storm_max, amp_storm_max, amp_storm_mean, area_storm, scale_storm)
                                if ilev>=thresh_height_for_scale:
                                    is_growing_storm=np.append(is_growing_storm, 1)
                                else:
                                    is_growing_storm=np.append(is_growing_storm, 0)  
                                fieldnan[regionsOld2 == uregions_old_not_saved[iregOld]] = np.nan
                                labelStorms[regionsOld2 == uregions_old_not_saved[iregOld]] = count_storms # regions_duplicated_numbers from previous step 
                                count_storms = count_storms+1
                
                # get the residual to -100
                ind = np.logical_and([regions_duplicated_numbers == iregion], np.logical_not(np.isnan(fieldnan)))[0]
                fieldnan[ind] = np.nan
                labelStorms[ind] = -50
# ===============================================================================================
# ---- Has NOT nan in
# -----------------------------------------------------------------------------------------------
            else: # not has_nan_in      
                interior = ndimage.binary_erosion(region)
                exterior = np.logical_xor(region, interior)
                if interior.sum() == 0:
                    continue
                has_internal_max = field2[interior].max() > field2[exterior].max()
                if np.logical_not(has_internal_max):
                    continue

    # --- 5. Find amplitude of region: set amplitude limit for the external ring --------------------------------------
                amp = (field2[interior].max() - field2[exterior].mean()) / field2[interior].max()
                is_tall_storm = amp >= amp_thresh
                if np.logical_not(is_tall_storm):
                    continue
 
    # --- 6. Find maximum linear dimension of region, reject if < d_thresh ---------------
                lon_ext = llon2[exterior]
                lat_ext = llat2[exterior]
                d = distance_matrix(lon_ext, lat_ext)
            # print(str(ilev)+' m : '+str(d.max()))
            #quick_plot_storm(storm_object_with_mass,llon,llat,lon_storm_center,lat_storm_center,lon_storm_center,lat_storm_center,findrange=True)
            
                is_large_enough = np.logical_and((d.max() > d_thresh_min),(d.max() < d_thresh_max))
            # np.logical_and((d.max() > d_thresh_min),(d.max() < d_thresh_max))

    # --- 7. Save storm if conditions realized --------------------------------------------
                if eddy_area_within_limits * has_internal_max * is_tall_storm * is_large_enough:
                    lon_storm_center, lat_storm_center, lon_storm_max, lat_storm_max, amp_storm_max, amp_storm_mean, area_storm, scale_storm = store_storm(field2, region, lon, lat, llon2, llat2, lon_storm_center, lat_storm_center, lon_storm_max, lat_storm_max, amp_storm_max, amp_storm_mean, area_storm, scale_storm)
                    if ilev>=thresh_height_for_scale:
                        is_growing_storm=np.append(is_growing_storm, 1)
                    else:
                        is_growing_storm=np.append(is_growing_storm, 0)
                    
                    fieldnan[regions_duplicated_numbers==iregion]=np.nan
                    labelStorms[regions_duplicated_numbers==iregion] = count_storms # regions_duplicated_numbers from previous step 
                    count_storms = count_storms+1
        
    # --- 8. update regionsOld (at the end of the processing for one level) ---------------------------------------------
        # update field2 : where fieldnan is nan : field2 = 0.         
        # land = np.isnan(fieldnan)
        # field2[land] = 0.
        # update regionsOld
        regionsOld2 = regionsOld1.copy()
        # quick_plot(regions,title='regions at the end')
        # quick_plot(fieldnan,title='regions at the end')
    #field[land]=np.nan 
    
    labelStorms[labelStorms<0]=np.nan
    #quick_plot_storm(field,llon,llat,lon_storm_max,lat_storm_max,lon_storm_center,lat_storm_center,findrange=True)
    return lon_storm_center, lat_storm_center, lon_storm_max, lat_storm_max, amp_storm_max, amp_storm_mean, area_storm, scale_storm, labelStorms

def investigate_detection_t(i,days_vec,filenameFormat,input_dir, cut_lon, cut_lat, swh_levels, Npix_min, amp_thresh, d_thresh_min, d_thresh_max, thresh_height_for_scale,data_dir, gen_name_labelstorm,issave):
    datett = days_vec[i]
    print('time step : '+str(i))
    filename = get_filename(filenameFormat,datett)
    # Load map of significant wave height (SWH)
    lon, lat, swh = load_swh(input_dir,filename,datett)
    # 
    ## Spatially filter SSH field
    # 
    swh_filt = spatial_filter(swh, lon, lat, 0.5, cut_lon, cut_lat)
    #quick_plot(swh_filt,findrange=True)

    lon_cent, lat_cent, lon_max, lat_max, amp_max, amp_mean, area, scale, labelStorms = detect_storms(swh_filt, lon, lat, swh_levels, Npix_min, amp_thresh, d_thresh_min, d_thresh_max,thresh_height_for_scale)
    print('done')
    if issave:
        filesave0 = gen_name_labelstorm+'_part_'+str(i)# v3 : amp_thresh = 1/2, v4: amp_thresh = 1/5
        filesave = os.path.join(data_dir,filesave0)
        np.savez(filesave, labels=labelStorms,time=datett)
        print('LabelStorms saved')
    return i,lon_cent, lat_cent, lon_max, lat_max, amp_max, amp_mean, area, scale
    
    
    
def detection_plot(tt,lon,lat,eta,eta_filt,anticyc_eddies,cyc_eddies,ptype,plot_dir,findrange=True):
    """function to plot how the eddy detection alogirthm went
    
    :tt
    :lon
    :lat
    :eta
    :eta_filt
    :anticyc_eddies
    :cyc_eddies
    :ptype
    :plot_dir
    :findrange=True
    :returns: @todo
    """
    def plot_eddies():
        """@todo: Docstring for plot_eddies
        :returns: @todo
        """
        ax.plot(anticyc_eddies[0], anticyc_eddies[1], 'k^')
        ax.plot(cyc_eddies[0], cyc_eddies[1], 'kv')
    
        pass
    if ptype=='single':
        plt.close('all')
        fig=plt.figure()
        ax=fig.add_subplot(1, 1,1)

    elif ptype=='rawtoo':
        plt.close('all')
        fig=plt.figure()

        #width then height
        fig=plt.figure(figsize=(12.0,9.0))
        ax=fig.add_subplot(1, 2,1)

        #ecj range...
        #plt.contourf(lon, lat, eta_filt, levels=np.arange(-2.5,2.5,0.05))

        #cb NEMO range
        cs1=plt.contourf(lon, lat, eta_filt, levels=np.linspace(-.817,0.5,40))
        cbar=fig.colorbar(cs1,orientation='vertical')
        ax.set_title('day: ' + str(tt)+' filtered ssh')
        plot_eddies()
        
        ax=fig.add_subplot(1, 2,2)
        cs1=plt.contourf(lon, lat, eta, levels=np.linspace(-1.75,0.85,40))
        cbar=fig.colorbar(cs1,orientation='vertical')
        ax.set_title('day: ' + str(tt)+' raw ssh')
        plot_eddies()


        #determine range to plot 
        #if np.isnan(np.sum(eta_filt)):
            #plt.contourf(lon,lat, eta_filt,levels=np.linspace(np.min(np.nan_to_num(eta_filt)),np.max(np.nan_to_num(eta_filt)),50))
            #print np.min(np.nan_to_num(eta_filt))
            #print np.max(np.nan_to_num(eta_filt))
        #else:
            #plt.contourf(lon,lat, eta_filt,levels=np.linspace(np.min(eta_filt),np.max(eta_filt),50))

        #plt.clim(-0.5,0.5)
        plt.savefig(plot_dir+'eta_filt_' + str(tt).zfill(4) + '.png', bbox_inches=0)

    pass

def storms_list(it,datevec,lon_storms_center, lat_storms_center, amp_storms_max, amp_storms_mean, area_storms, scale_storms, lon_storms_max, lat_storms_max):
    '''
    Creates list detected eddies
    '''
    labels_tmp =[]
    time_tmp =[]
    storms = []
    print(it)
    for ied in range(len(it)):
        # print(ied)
        ed = np.where(np.array(it) == ied)[0][0].astype(int)
        # print(ed)
        storm_tmp = {}
        storm_tmp['lon_center'] = lon_storms_center[ed]
        storm_tmp['lat_center'] = lat_storms_center[ed]
        storm_tmp['lon_max'] = lon_storms_max[ed]
        storm_tmp['lat_max'] = lat_storms_max[ed]
        storm_tmp['amp_max'] = amp_storms_max[ed]
        storm_tmp['amp_mean'] = amp_storms_mean[ed]
        storm_tmp['area'] = area_storms[ed]
        storm_tmp['scale'] = scale_storms[ed]
        storm_tmp['time'] = datevec[ied]
        
        #if modulo_time!=(ied//150):
        #    filename = gen_name_labelstorm+'_part_'+str(modulo_time)# v3 : amp_thresh = 1/2, v4: amp_thresh = 1/5
        #    filesave = os.path.join(data_dir,filename)
        #    np.savez(filesave, labels=labels_tmp,time=time_tmp)
        #    labels_tmp =[]
        #    time_tmp =[]

        #labels_tmp.append(labelStorms[ed])
        #time_tmp.append(datevec[ied])
        #modulo_time=(ied//150)
        
        #if ied == len(it)-1:
        #    filename = 'gen_name_labelstorm_part'+str(modulo_time)# v3 : amp_thresh = 1/2, v4: amp_thresh = 1/5
        #    filesave = os.path.join(data_dir,filename)
        #    np.savez(filesave, labels=labels_tmp,time=time_tmp)
        #    labels_tmp =[]
        #    time_tmp =[]
            
        # N : number of storms per time step
        storm_tmp['N'] = len(storm_tmp['lon_center'])
        #storm_tmp['part'] = modulo_time
        storms.append(storm_tmp)

    return storms


def storms_init(det_storms):
    '''
    Initializes list of storms. The ith element of output is
    a dictionary of the ith storm containing information about
    position and size as a function of time, as well as type.
    '''

    storms = []

    for ed in range(det_storms[0]['N']):
        storm_tmp = {}
        storm_tmp['lon_center'] = np.array([det_storms[0]['lon_center'][ed]])
        storm_tmp['lat_center'] = np.array([det_storms[0]['lat_center'][ed]])
        storm_tmp['lon_max'] = np.array([det_storms[0]['lon_max'][ed]])
        storm_tmp['lat_max'] = np.array([det_storms[0]['lat_max'][ed]])
        storm_tmp['amp_max'] = np.array([det_storms[0]['amp_max'][ed]])
        storm_tmp['amp_mean'] = np.array([det_storms[0]['amp_mean'][ed]])
        storm_tmp['area'] = np.array([det_storms[0]['area'][ed]])
        storm_tmp['scale'] = np.array([det_storms[0]['scale'][ed]])
        storm_tmp['time'] = np.array([1])
        storm_tmp['exist_at_start'] = True
        storm_tmp['terminated'] = False
        storm_tmp['missed'] = 0
        storms.append(storm_tmp)

    return storms

def is_in_ellipse(x0, y0, dE, d, x, y):
    '''
    Check if point (x,y) is contained in ellipse given by the equation
      (x-x1)**2     (y-y1)**2
      ---------  +  ---------  =  1
         a**2          b**2
    where:
      a = 0.5 * (dE + dW)
      b = dE
      x1 = x0 + 0.5 * (dE - dW)
      y1 = y0
    '''

    dW = np.max([d, dE])

    b = dE
    a = 0.5 * (dE + dW)

    x1 = x0 + 0.5*(dE - dW)
    y1 = y0

    return (x-x1)**2 / a**2 + (y-y1)**2 / b**2 <= 1


def len_deg_lon(lat):
    '''
    Returns the length of one degree of longitude (at latitude
    specified) in km.
    '''

    R = 6371. # Radius of Earth [km]

    return (np.pi/180.) * R * np.cos( lat * np.pi/180. )



def track_storms(storms, det_storms, tt, dt, max_missed_dt, max_spd, storm_scale_min, storm_scale_max,lonvar,latvar,ampvar):
    '''
    Given a map of detected storms as a function of time (det_storms)
    this function will update tracks of individual storm at time step
    tt in variable storms
    '''

    # List of unassigned storms at time tt

    unassigned = np.arange(det_storms[tt]['N'])

    # For each existing storm (t<tt) loop through unassigned storms and assign to existing storm if appropriate

    for ed in range(len(storms)):
        # print(storms[ed])
        # print(storms[ed]['missed'])
        # Check if storm has already been terminated

        if not storms[ed]['terminated']:
            print(storms[ed]['missed'])
            # Define search region around centroid of existing storm ed at last known position
            print('ed = ', ed)   
            x0 = storms[ed][lonvar][-1] # [deg. lon]
            y0 = storms[ed][latvar][-1] # [deg. lat]
            dE = max_spd * dt * (storms[ed]['missed'] +1) + 0.5*storms[ed]['scale'][-1] # [km/h * h] = [km]
            # the limit depends on the missing values ( if storm is not detected during one time step, the following time step it may have travelled further away ) 
            
            # Find all storm centroids in search region at time tt
    
            is_near = is_in_ellipse(x0, y0, dE/len_deg_lon(y0), dE/len_deg_lon(y0), det_storms[tt][lonvar][unassigned], det_storms[tt][latvar][unassigned])
    
            # Check if storms' amp  and area are between 0.25 and 2.5 of original eddy
    
            amp = storms[ed][ampvar][-1]
            area = storms[ed]['area'][-1]
            is_similar_amp = (det_storms[tt][ampvar][unassigned] < amp*storm_scale_max) * (det_storms[tt][ampvar][unassigned] > amp*storm_scale_min)
            is_similar_area = (det_storms[tt]['area'][unassigned] < area*storm_scale_max) * (det_storms[tt]['area'][unassigned] > area*storm_scale_min)
    
           
            # Possible eddies are those which are near, of the right amplitude, and of the same type
    
            possibles = is_near * is_similar_amp # * is_similar_area
            if possibles.sum() > 0:
    
                # Of all found eddies, accept only the nearest one
    
                dist = distance_vect(x0,y0,det_storms[tt][lonvar][unassigned],det_storms[tt][latvar][unassigned])
                # np.sqrt((x0-det_storms[tt][lonvar][unassigned])**2 + (y0-det_storms[tt][latvar][unassigned])**2)
                nearest = dist == dist[possibles].min()
                # ind_unassigned is the index of next storm in unassigned
                ind_unassigned = np.where(nearest * possibles)[0][0]
                #next storm is index of storm
                next_storm = unassigned[ind_unassigned]
    
                # Add coordinatse and properties of accepted eddy to trajectory of eddy ed
    
                storms[ed]['lon_max'] = np.append(storms[ed]['lon_max'], det_storms[tt]['lon_max'][next_storm])
                storms[ed]['lat_max'] = np.append(storms[ed]['lat_max'], det_storms[tt]['lat_max'][next_storm])
                storms[ed]['lon_center'] = np.append(storms[ed]['lon_center'], det_storms[tt]['lon_center'][next_storm])
                storms[ed]['lat_center'] = np.append(storms[ed]['lat_center'], det_storms[tt]['lat_center'][next_storm])
                storms[ed]['amp_max'] = np.append(storms[ed]['amp_max'], det_storms[tt]['amp_max'][next_storm])
                storms[ed]['amp_mean'] = np.append(storms[ed]['amp_mean'], det_storms[tt]['amp_mean'][next_storm])
                storms[ed]['area'] = np.append(storms[ed]['area'], det_storms[tt]['area'][next_storm])
                storms[ed]['scale'] = np.append(storms[ed]['scale'], det_storms[tt]['scale'][next_storm])
                storms[ed]['time'] = np.append(storms[ed]['time'], tt+1)
    
                # Remove detected eddy from list of eddies available for assigment to existing trajectories
    
                #unassigned.remove(next_storm)
                unassigned = np.delete(unassigned,[ind_unassigned])
                storms[ed]['missed'] = 0 # the count for missed time step is reset each detected time step
            # Terminate eddy otherwise

            else:
                storms[ed]['missed'] = storms[ed]['missed'] + 1
                
                if storms[ed]['missed'] >= max_missed_dt/dt:
                    storms[ed]['terminated'] = True

    # Create "new storm" from list of storms not assigned to existing trajectories

    if len(unassigned) > 0:

        for un in unassigned:

            storm_tmp = {}
            storm_tmp['lon_center'] = np.array([det_storms[tt]['lon_center'][un]])
            storm_tmp['lat_center'] = np.array([det_storms[tt]['lat_center'][un]])
            storm_tmp['lon_max'] = np.array([det_storms[tt]['lon_max'][un]])
            storm_tmp['lat_max'] = np.array([det_storms[tt]['lat_max'][un]])
            storm_tmp['amp_mean'] = np.array([det_storms[tt]['amp_mean'][un]])
            storm_tmp['amp_max'] = np.array([det_storms[tt]['amp_max'][un]])
            storm_tmp['area'] = np.array([det_storms[tt]['area'][un]])
            storm_tmp['scale'] = np.array([det_storms[tt]['scale'][un]])
            storm_tmp['time'] = np.array([tt+1])
            storm_tmp['exist_at_start'] = False
            storm_tmp['terminated'] = False
            storm_tmp['missed'] = 0
            storms.append(storm_tmp)
            

    return storms
