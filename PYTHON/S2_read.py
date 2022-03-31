import numpy as np
import os
import rasterio
from rasterio.windows import Window
def S2_read(S2file,boxi,bands):
    """
    reads sentinel-2 JPEG2000 optical image and saves it in 2D array
    
    Args:
        S2file: path and filename for jp2 files 
        boxi : bounding box of sub-image: i1,i2,j1,j2
        bands: name of S2 bands such as "B02" ... 

    """
    arrs = []
    print(S2file)
    for jp2_band in bands:
        dataset = rasterio.open(S2file+'_'+jp2_band+'.jp2')
        NX=dataset.width
        NY=dataset.height
        iystart=dataset.height-boxi[3]
        print(iystart)
        print(boxi[2])
        with rasterio.open(S2file+'_'+jp2_band+'.jp2') as image:
            w=image.read(1, window=Window(boxi[0]-1,iystart,  boxi[1]-boxi[0]+1,  boxi[3]-boxi[2]+1))
        print(w.shape)
        [nx,ny]=w.shape
        dx=10. 
        dy=10.  # will be updated later
        arrs.append(np.transpose(np.fliplr(w)))

    imgs = np.array(arrs, dtype=arrs[0].dtype)

    return imgs,NX,NY,nx,ny,dx,dy

