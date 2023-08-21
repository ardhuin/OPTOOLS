import os
import sys
import numpy as np
import scipy as sp
import numpy.linalg as linalg
import scipy.signal as signal
import scipy.ndimage as ndimage
import scipy.interpolate as interpolate
import glob
import pandas as pd
import datetime as dtm
import xarray as xr
import matplotlib

import matplotlib.pyplot as plt



def lat_lon_cell_area(lat_lon_grid_cell):
    """
    Calculate the area of a cell, in meters^2, on a lat/lon grid.
    
    This applies the following equation from Santinie et al. 2010.
    
    S = (λ_2 - λ_1)(sinφ_2 - sinφ_1)R^2
    
    S = surface area of cell on sphere
    λ_1, λ_2, = bands of longitude in radians
    φ_1, φ_2 = bands of latitude in radians
    R = radius of the sphere
    
    Santini, M., Taramelli, A., & Sorichetta, A. (2010). ASPHAA: A GIS‐Based 
    Algorithm to Calculate Cell Area on a Latitude‐Longitude (Geographic) 
    Regular Grid. Transactions in GIS, 14(3), 351-377.
    https://doi.org/10.1111/j.1467-9671.2010.01200.x

    Parameters
    ----------
    lat_lon_grid_cell
        A shapely box with coordinates on the lat/lon grid

    Returns
    -------
    float
        The cell area in meters^2

    """
    
    # mean earth radius - https://en.wikipedia.org/wiki/Earth_radius#Mean_radius
    AVG_EARTH_RADIUS_METERS = 6371008.8
    
    west, south, east, north = lat_lon_grid_cell
    
    west = (west)*np.pi/180.
    east = (east)*np.pi/180.
    south = (south)*np.pi/180.
    north = (north)*np.pi/180.
    
    return (east - west) * (np.sin(north) - np.sin(south)) * (AVG_EARTH_RADIUS_METERS**2)

def spatial_filter(field, res, cut_lon, cut_lat):
    '''
    Performs a spatial filter, removing all features with
    wavelenth scales larger than cut_lon in longitude and
    cut_lat in latitude from field (defined in grid given
    by lon and lat).  Field has spatial resolution of res
    and land identified by np.nan's
    '''
    shp_field = field.shape
    field_filt = np.zeros(shp_field)

    # see Chelton et al, Prog. Ocean., 2011 for explanation of factor of 1/5
    sig_lon = (cut_lon/5.) / res
    sig_lat = (cut_lat/5.) / res

    land = np.isnan(field.flatten())
    field = field.flatten()
    field[land] = np.nanmean(field)
    
    field = np.reshape(field,shp_field)

    # field_filt = field - ndimage.gaussian_filter(field, [sig_lat, sig_lon])
    field_filt = ndimage.gaussian_filter(field, [sig_lat, sig_lon]).flatten()
    field_filt[land] = np.nan

    return np.reshape(field_filt,shp_field)

def remove_duplicates(regions,NX):
    uregions_origin = np.unique(regions[:,:NX])
    uregions_dupl = np.unique(regions[:,NX:])
    
    # create duplicate matrix from big one
    regions_dupl0 = np.zeros_like(regions[:,:NX])-100.
    regions_dupl0[:,:np.size(regions[:,NX:],1)] = regions[:,NX:]
    
    # --- get labels that intersect duplicate and non duplicate region ---- 
    intersec = [r for r in uregions_dupl if (r in uregions_origin)&(r>-1)]
    # print(intersec)
    regions_dupl = np.zeros_like(regions[:,NX:])-100.
    regions_orig = np.copy(regions[:,:NX])
    
    regions_interOnly_0 = np.zeros_like(regions[:,:NX])-100.
    for ir in intersec:
        regions_dupl[regions[:,NX:]==ir] = ir
        regions_orig[regions_dupl0==ir] = -100.
        regions_interOnly_0[regions_dupl0==ir] = ir
        regions_interOnly_0[regions_orig==ir] = ir

    regions_new = np.concatenate([regions_orig,regions_dupl],axis=1)
    regions_interOnly = np.concatenate([regions_interOnly_0,regions_dupl],axis=1)
    return regions_new,  regions_interOnly 

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


def read_WW3_HS_file(PATH,filename):
    ds0 = xr.open_dataset(os.path.join(PATH,filename))
    ds = ds0[['longitude','latitude','time','hs']]
    all_lats, all_lons = np.meshgrid(ds0['latitude'].data, ds0['longitude'].data, indexing='ij')
    side_length=0.5
    
    lat_lon_grid_cell = np.array([all_lons , all_lats - side_length/2 , all_lons + side_length , all_lats + side_length/2])
    areakm2 = lat_lon_cell_area(lat_lon_grid_cell) / 1e6
    
    return ds.assign({'areakm2' : (('latitude','longitude'),areakm2)})

# area_forgotten_ratio
# min_area = 500
# amp_thresh
# d_thresh_min
# d_thresh_max
# 
# 
def get_storm_info_from_savemap(ds):
    idm = ds.hs.idxmax()
    arsum = ds.areakm2.sum()
    hsmax = ds.hs.max()

    dsn = xr.Dataset({'lon_max' : (idm.longitude+180)%360 -180, 
                      'lat_max' : idm.latitude, 
                      'time' : ds.time, 
                      'hs_max' : hsmax,
                      'areastorm' : arsum, 
                     }
                    )
    return dsn

def get_storm_by_timestep(ds1,levels,Npix_min,amp_thresh, d_thresh_min, d_thresh_max, area_forgotten_ratio, min_area):
    # --- concat [-180: 360] ----
    ds2 = ds1.copy(deep=True).sel(longitude=slice(None,0))
    ds2['longitude'] = ds2['longitude']+360.
    dsTot = xr.concat((ds1,ds2),dim='longitude')
    
    swh_filt0 = spatial_filter(dsTot['hs'].data, 0.5, 4., 4.)
    swh_filt = dsTot['hs'].copy(data=swh_filt0)
    
    field20 = swh_filt
    count_storms=0

    area2 = dsTot['areakm2'].data
    field20 = field20.where(~np.isnan(field20),0)
    field2 = field20.data

    NX = ds1.sizes['longitude']

    llon2,llat2 = np.meshgrid(dsTot['longitude'],dsTot['latitude'])

    regions_old = np.zeros_like(field2.data,dtype='int')-100
    to_save = np.zeros_like(field2.data,dtype='int')-100
    to_save_intersec = np.zeros_like(field2.data,dtype='int')-100
    
    countst = 0

    for ilev, lev in enumerate(levels):
        # --- 1. Find all regions with Hs greater than ilev  ---------------
        regions, nregions = ndimage.label( (field2 > lev).astype(int) )
        regions[regions==0] =-100

        regions_new, regions_interOnly = remove_duplicates(regions,NX)

        uregions = np.unique(regions_new[regions_new>-100])

        for iir, ir in enumerate(uregions):
            # ---- 
            regionNB = (regions_new == ir)

            is_already_saved = np.any((to_save_intersec[regionNB]>-1))
            u_regions_saved_in = np.unique((to_save_intersec[(regionNB) & (to_save_intersec>-1)]))
            u_old_regions_in0 = np.unique((regions_old[(to_save_intersec<-2) &(regionNB) & (regions_old>-1)])) # regions that are not saved
            u_old_regions_out0 = np.unique(regions_old[(regionNB) & (to_save_intersec>-1)])
            u_old_regions_in = np.setdiff1d(u_old_regions_in0,u_old_regions_out0)

            if is_already_saved: # ---- inside the region, there is already a save storm:
                # --- case : there was also another "storm" detected at previous level that did not match all required flags for saving
                # -- get area of saved storms ---------
                area_old_max = 0
                for u_sav_in in u_regions_saved_in:
                    area_u = np.sum(area2[to_save==u_sav_in])
                    area_old_max = np.max((area_old_max,area_u))
                # --- for the forgotten storm, if area big enough : save -----     
                if len(u_old_regions_in) > 0:
                    for u_old in u_old_regions_in: # --- loop over "forgotten storm" 
                        region_old_u_old = regionNB & (regions_old==u_old)

                        interior = ndimage.binary_erosion(region_old_u_old)
                        exterior = np.logical_xor(region_old_u_old, interior)
                        if interior.sum() == 0:
                            continue
                        area_u_old = np.sum(area2[region_old_u_old])
                        # --- save if not too small ----------
                        if area_u_old >= area_old_max*area_forgotten_ratio:
                            to_save[region_old_u_old] = countst
                            to_save_intersec[region_old_u_old] = countst
                            if np.any(regions_interOnly == ir):
                                to_save_intersec[(regions_old==u_old)] = countst
                            # print(lev,'save in already saved', countst)
                            countst = countst + 1
                    # --- end of loop over "forgotten storm" 

            else: # --- not already saved
                # --- Flag number of pixel -------------------
                regionNb_Npix = regionNB.astype(int).sum()        
                # eddy_area_within_limits = (regionNb_Npix > Npix_min)
                area_reg = np.sum(area2[regionNB])
                eddy_area_within_limits = (area_reg > min_area)
                interior = ndimage.binary_erosion(regionNB)
                exterior = np.logical_xor(regionNB, interior)

                if interior.sum() == 0:
                    continue

                has_internal_max = np.max(field2[interior]) > field2[exterior].max()
                if np.logical_not(has_internal_max):
                    continue

                amp = (field2[interior].max() - field2[exterior].mean()) / field2[interior].max()
                is_tall_storm = (amp >= amp_thresh)
                if np.logical_not(is_tall_storm):
                    continue

                lon_ext = llon2[exterior]
                lat_ext = llat2[exterior]
                d = distance_matrix(lon_ext, lat_ext)

                is_large_enough = np.logical_and((d.max() > d_thresh_min),(d.max() < d_thresh_max))

                if eddy_area_within_limits * has_internal_max * is_tall_storm:# * is_large_enough:
                    to_save[regionNB] = countst
                    to_save_intersec[regionNB] = countst
                    if np.any(regions_interOnly == ir):
                        to_save_intersec[regions_interOnly == ir] = countst
                    countst = countst + 1

        # --- end of loop over labeled regions
        regions_old = regions_new.copy()
        regions_old[(regions_interOnly>-1)] = regions_interOnly[(regions_interOnly>-1)]        
    
    ds3=dsTot.assign({'regions' :(('latitude','longitude' ),to_save)})
    g=ds3.where(ds3.regions>-1.,np.nan).groupby('regions')
    res = g.map(get_storm_info_from_savemap).swap_dims({'regions':'x'})
    res = res.assign_coords(storms_by_t = xr.full_like(res.regions,fill_value=len(g),dtype=int))
           
    return res


