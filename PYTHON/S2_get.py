import matplotlib.pyplot as plt
import numpy as np
from S2_read import *
from s2_and_sun_angs import *
S2file='/home/ardhuin/ADMIN/PROPOSALS/EE11/S2/DATA/T11SMS_20160429T183252';
#S2file='/home/datawork-WW3/IMAGERY/S2/S2A_MSIL1C_20160429T183252_N0201_R027_T11SMS_20160429T184037.SAFE/GRANULE/L1C_T11SMS_A004457_20160429T184037/IMG_DATA/T11SMS_20160429T183252'
boxi=[5600,6400,3600,4400] # bounding box indices in image:  East West South North 
bands=['B04','B03','B02'];  
ibands=[3,2,1];  
nb=np.shape(bands)[0]
[imgs,NX,NY,nx,ny,dx,dy]=S2_read(S2file,boxi,bands)

# defines indices of where we are in coarse array of angles
# these indices are relative to top-left corner 
indx=int(np.round((0.5*(boxi[0]+boxi[1]))/(5000/dx)))
indy=int(np.round((NY*10/dx-0.5*(boxi[2]+boxi[3]))/(5000/dx)))
S2xml=S2file+'.xml'

#S2xml='/home/datawork-WW3/IMAGERY/S2/S2A_MSIL1C_20160429T183252_N0201_R027_T11SMS_20160429T184037.SAFE/GRANULE/L1C_T11SMS_A004457_20160429T184037/MTD_TL.xml'

(Tile_ID, AngleObs, AngleSun) = get_angleobs(S2xml)
ulx=AngleObs['ul_x']
uly=AngleObs['ul_y']
nz1=nz2=23
satg = numpy.zeros(shape=(nz1,nz2,13,3))
sung = numpy.zeros(shape=(nz1,nz2,5))
detector=numpy.zeros(shape=(nz1,nz2,13))

observe=AngleObs['obs']
sunall=AngleSun['sun']
sh1=np.shape(observe)
ncells=sh1[0]

for icell in range(0,ncells):
#  This takes the nearest cell in matrix 
   thiscell=observe[icell]
   iband=thiscell[0]
   ix=int((thiscell[2]-ulx)/5000)
   iy=int((uly-thiscell[3])/5000)
   detector[iy,ix,iband]=thiscell[1]
   satg[iy,ix,iband,0]=thiscell[6][0]
   satg[iy,ix,iband,1]=thiscell[6][1]
   satg[iy,ix,iband,2]=thiscell[6][2]

for isun in range(0,nz1*nz2):
   suncell=sunall[isun]
   ix=int((suncell[0]-ulx)/5000)
   iy=int((uly-suncell[1])/5000)
   sung[iy,ix,0]=suncell[2]
   sung[iy,ix,1]=suncell[3]
   sung[iy,ix,2]=suncell[4][0]
   sung[iy,ix,3]=suncell[4][1]
   sung[iy,ix,4]=suncell[4][2]


#observe = [bandId, detectorId, xcoord, ycoord, Sat, Gx]
#                        AngleObs['obs'].append(observe)

# these are the 3 components of the unit vector pointing to the sun
sun=sung[indy,indx,2:5]

thetav=np.zeros(shape=(nb,1))
phiv=np.zeros(shape=(nb,1))
offspec=np.zeros(shape=(nb,1))
phitrig=np.zeros(shape=(nb,1))
jj=0
for jb in ibands:
   # Unit vector from point in swath to sat
   sat=satg[indy,indx,jb,0:3]
   # vector that bisects the sun-target-sat angle 
   mid=sun+sat 
   midn=np.sqrt(mid[0]**2+mid[1]**2+mid[2]**2)
   mid=mid/midn;
   thetav[jj]=np.arccos(sat[2])*todeg    
   offspec[jj]=np.arccos(mid[2])*todeg            # off-specular angle
   phitrig[jj]=np.arctan2(mid[0],mid[1])*todeg   # azimuth of bistatic look
   jj=jj+1

#[V, VG, Height]=read_ground_velocity(DSxml, latcenter);



img1=imgs[0,:,:]
plt.imshow(np.fliplr(np.transpose(img1))) #edit your vmin, vmax and cmap if you don't like greyscale colormap
plt.colorbar()
plt.show()


#imgref=S2file+'_B02.jp2'
#XML_File='/home/ardhuin/Téléchargements/S2A_MSIL1C_20160429T183252_N0201_R027_T11SMS_20160429T184037.SAFE/GRANULE/L1C_T11SMS_A004457_20160429T184037/MTD_TL.xml'
#s2_sensor_angs(XML_File, imgref, '', '')

