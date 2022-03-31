import numpy as np
import os
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
        with rasterio.open(S2file+'_'+jp2_band+'.jp2') as image:
            w=image.read(1, window=Window(boxi[2]-1, boxi[0]-1, boxi[3]-boxi[2]+1, boxi[1]-boxi[0]+1))
        print(w.shape)
        arrs.append(w)

    imgs = np.array(arrs, dtype=arrs[0].dtype)

    return imgs

