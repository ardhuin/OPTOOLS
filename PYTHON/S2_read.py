import numpy as np
import os, glob
import rasterio
from s2_and_sun_angs  import *
from rasterio.windows import Window

def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

def S2_read(S2path,boxi,bands):
    """
    reads sentinel-2 JPEG2000 optical image and saves it in 2D array
    Note that the imgs array is in the following geometry: channel 0 to N, Left to right, Bottom to top
    Args:
        S2file: path and filename for jp2 files 
        boxi : bounding box of sub-image: i1,i2,j1,j2
        bands: name of S2 bands such as "B02" ... 

    """
    arrs = []
    XML_File=find('MTD_TL.xml',S2path)
    print(XML_File)

    (tile_id, AngleObs, AngleSun)=get_angleobs(XML_File)
    #print('AngleObs:',AngleObs)
    #print('AngleSun:',AngleSun)

    for jp2_band in bands:
        files=glob.glob(os.path.join(S2path,'GRANULE/*/*/*'+jp2_band+'*.jp2'))
        S2file=files[0]
        print(S2file)
        #dataset = rasterio.open(S2file+'_'+jp2_band+'.jp2')
        dataset = rasterio.open(S2file)
        NX=dataset.width
        NY=dataset.height
        iystart=dataset.height-boxi[3]
        print(iystart)
        print(boxi[2])
        with rasterio.open(S2file) as image:
            w=image.read(1, window=Window(boxi[0]-1,iystart,  boxi[1]-boxi[0]+1,  boxi[3]-boxi[2]+1))
        print(w.shape)
        [nx,ny]=w.shape
        dx=10. 
        dy=10.  # will be updated later
        arrs.append(np.transpose(np.fliplr(w)))

    imgs = np.array(arrs, dtype=arrs[0].dtype)

    return imgs,NX,NY,nx,ny,dx,dy

