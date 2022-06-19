import numpy as np
from numpy.linalg import norm
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
    
    (tile_id, AngleObs, AngleSun)=get_angleobs(XML_File)
    print('Reading XML file for view and sun angles:',XML_File)
    #print('AngleObs:',AngleObs)
    #print('AngleSun:',AngleSun)

    for jp2_band in bands:
        files=glob.glob(os.path.join(S2path,'GRANULE/*/*/*'+jp2_band+'*.jp2'))
        S2file=files[0]
        print('Reading file for band ',jp2_band,':',S2file)
        #dataset = rasterio.open(S2file+'_'+jp2_band+'.jp2')
        dataset = rasterio.open(S2file)
        NX=dataset.width
        NY=dataset.height
        iystart=dataset.height-boxi[3]
        with rasterio.open(S2file) as image:
            w=image.read(1, window=Window(boxi[0]-1,iystart,  boxi[1]-boxi[0]+1,  boxi[3]-boxi[2]+1))
        [nx,ny]=w.shape
        dx=10. 
        dy=10.  # will be updated later
        arrs.append(np.transpose(np.flipud(w)))

    imgs = np.array(arrs, dtype=arrs[0].dtype)

    nax= 23
    nay= 23
    nb=np.shape(bands)[0]
    xa0= float(AngleObs['ul_x'])
    ya0= float(AngleObs['ul_y'])
    dax=5000
    day=5000
    NY=10980
    dx=10
    obsvec=np.zeros((13,nay,nax,3))
    detector=np.zeros((13,nay,nax), dtype=int)
    sunvec=np.zeros((nay,nax,3))
    indexX=np.zeros((nay,nax))
    indexY=np.zeros((nay,nax))

    for sunrec in AngleSun['sun']:
       indax=int((sunrec[0]-xa0)/dax)
       inday=int((ya0-sunrec[1])/day)
       sunvec[inday,indax,0]=sunrec[4][0]
       sunvec[inday,indax,1]=sunrec[4][1]
       sunvec[inday,indax,2]=sunrec[4][2]
       indexX[inday,indax]=(sunrec[0]-xa0)/dx
       indexY[inday,indax]=(-ya0+sunrec[1])/dx+NY

    for viewrec in AngleObs['obs']:
       indax=int((viewrec[2]-xa0)/dax)
       inday=int((ya0-viewrec[3])/day)
       obsvec[viewrec[0],inday,indax,0]=viewrec[6][0]
       obsvec[viewrec[0],inday,indax,1]=viewrec[6][1]
       obsvec[viewrec[0],inday,indax,2]=viewrec[6][2]
       detector[viewrec[0],inday,indax]=viewrec[1]

    banddict={
        'B02': 1,
        'B03': 2,
        'B04': 3,
        'B08': 7
    }

# gets Sun angles in the middle of box
    xulc=xa0+dx*(0.5*(boxi[0]+boxi[1])) 
    yulc=ya0-dx*(NY-0.5*(boxi[2]+boxi[3])) 
    indy=round((NY*10/dx-0.5*(boxi[2]+boxi[3]))/(5000/dx))

    (latcenter, loncenter) =utm_inv(AngleObs['zone'],xulc,yulc)

    indx=round((0.5*(boxi[0]+boxi[1]))/(5000/dx)) #  %This takes the nearest cell in matrix 
    indy=round((NY*10/dx-0.5*(boxi[2]+boxi[3]))/(5000/dx))
    print('x and y indices in angles arrays:',indx,indy)
    sunvec1=np.squeeze(sunvec[indy,indx,:] )

    offspec=np.zeros(nb)
    phitrig=np.zeros(nb)
    thetav=np.zeros(nb)
# Now defines the view angle for each band
    jb=0
    for band in bands:
        obsvec1=np.squeeze(obsvec[banddict.get(band),indy,indx,:] )

        mid1=sunvec1+obsvec1  # vector that bisects the sun-target-sat angle 
        midn=norm(mid1)
        mid=mid1/midn
   
        offspec[jb]=np.degrees(np.arccos(mid[2]))  # off-specular angle
        phitrig[jb]=np.degrees(np.arctan2(mid[0],mid[1])) #azimuth of bistatic look
        thetav[jb]=np.degrees(np.arccos(obsvec1[2]))
        print('band ',band,' gives off-specular, azimuth, incidence:',offspec[jb],phitrig[jb],thetav[jb])
    
        jb += 1

    return imgs,NX,NY,nx,ny,dx,dy,offspec,phitrig,thetav,loncenter,latcenter,detector,indexX,indexY

