
####################################
#-------------MODULES
####################################
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os
#%matplotlib qt4


#import geopyspark as gps
#from pyspark import SparkContext


####################################
#-------------FUNCTION
####################################

def read_S2_JP2_image(image,bands):
    """
    image: image name "T10SDH_YYYYMMDDTHHMMSS_band.jp2"
    band=B02_10m,B03_10m,B04_10m,B08_10m in R10M (R20,R60 exist as well)
    
    This script is reading the sentinel-2 JPEG2000 optic image and save it in 2D array
    ###WARNING### the output 2D array is huge (10000x10000)-> PLOT SUB-IMAGE!!!
    g.marechal march 2021
    """
    arrs = []

    for jp2_band in bands:
        print(image,jp2_band)
        with rasterio.open(image+jp2_band+'.jp2') as f:
            arrs.append(f.read(1))
    data = np.array(arrs, dtype=arrs[0].dtype)
    return data
    
    
    

####################################
#-----------------SCRIPT
####################################    

#os.chdir('where/the/image/JPEG2000/is/STORED')
os.chdir('/export/home/perso/gmarecha/Bureau/PHD/data/S2_images/S2A_MSIL1C_20190722T153911_N0208_R011_T18SWF_20190722T192124.SAFE/GRANULE/L1C_T18SWF_A021315_20190722T154407/IMG_DATA/')


image="T18SWF_20190722T153911_"

bands=["B02"]
image=read_S2_JP2_image(image,bands)

plt.imshow((image[0,::20,::20]),vmin=800,vmax=1800,cmap='binary_r') #edit your vmin, vmax and cmap if you don't like greyscale colormap
plt.colorbar()
plt.show()



