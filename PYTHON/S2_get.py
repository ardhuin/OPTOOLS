import matplotlib.pyplot as plt
import numpy as np
from S2_read import *
from s2_angs import *
#S2file='/home/ardhuin/ADMIN/PROPOSALS/EE11/S2/DATA/T11SMS_20160429T183252';
S2file='/home/datawork-WW3/IMAGERY/S2/S2A_MSIL1C_20160429T183252_N0201_R027_T11SMS_20160429T184037.SAFE/GRANULE/L1C_T11SMS_A004457_20160429T184037/IMG_DATA/T11SMS_20160429T183252'
boxi=[5600,6400,3600,4400] # East West South North 
bands=['B04','B03','B02'];  
nb=np.shape(bands)[0]
[imgs,NX,NY,nx,ny,dx,dy]=S2_read(S2file,boxi,bands)

# defines indices of where we are in coarse array of angles
indx=int(np.round((NX*10/dx-0.5*(boxi[0]+boxi[1]))/(5000/dx)))
indy=int(np.round((NY*10/dx-0.5*(boxi[2]+boxi[3]))/(5000/dx)))
S2xml='/home/datawork-WW3/IMAGERY/S2/S2A_MSIL1C_20160429T183252_N0201_R027_T11SMS_20160429T184037.SAFE/GRANULE/L1C_T11SMS_A004457_20160429T184037/MTD_TL.xml'
(Tile_ID, AngleObs) = get_angleobs(S2file+'.xml')
nrows=AngleObs['nrows']
ncols=AngleObs['ncols']
ulx=AngleObs['ul_x']
uly=AngleObs['ul_y']
nz1=nz2=23
satg = numpy.zeros(shape=(nz1,nz2,13,3))
detector=numpy.zeros(shape=(nz1,nz2,13))

observe=AngleObs['obs']
sh1=np.shape(observe)
ncells=sh1[0]

for icell in range(0,ncells):
#  This takes the nearest cell in matrix 
   thiscell=observe[icell]
   iband=thiscell[0]
   ix=int((thiscell[2]-ulx)/5000)
   iy=int((thiscell[3]-uly)/5000)
   detector[ix,iy,iband]=thiscell[1]
   satg[ix,iy,iband,0]=thiscell[4][0]
   satg[ix,iy,iband,1]=thiscell[4][1]
   satg[ix,iy,iband,2]=thiscell[4][2]
   


#observe = [bandId, detectorId, xcoord, ycoord, Sat, Gx]
#                        AngleObs['obs'].append(observe)

# Unit vector from point in swath to sat
sat=np.zeros(shape=(nb,3));
# this is ok for B04 B03 B02 ... will have to be automated for other choices!
sat[0,0:3]=satg[indx,indy,3,0:3]
sat[1,:]=satg[indx,indy,2,:]
sat[2,:]=satg[indx,indy,1,:]


mid=sat
sun=np.zeros(1,3)
#thetasun=sunzen(indx,indy)
#phisun  =sunazi(indx,indy)

#sun(1,1)=sin(thetasun.*d2r).*cos(phisun.*d2r);
#sun(1,2)=sin(thetasun.*d2r).*sin(phisun.*d2r);
#sun(1,3)=cos(thetasun.*d2r);

#thetav=[ 0. 0. 0. 0.];
#phiv=thetav;
#offspec=thetav;
#phitrig=thetav;
for jb in range(0:nb):
   
#   mid(jb,:)=sun+sat(jb,:);  % vector that bisects the sun-target-sat angle 
#   midn=sqrt(mid(jb,1).^2+mid(jb,2).^2+mid(jb,3).^2);
#   mid(jb,:)=mid(jb,:)./midn;
   
#   offspec(jb)=acos(mid(jb,3))./d2r                % off-specular angle
#   phitrig(jb)=atan2(mid(jb,2),mid(jb,1))./d2r   % azimuth of bistatic look


#[V, VG, Height]=read_ground_velocity(DSxml, latcenter);



img1=imgs[0,:,:]
plt.imshow(np.fliplr(np.transpose(img1))) #edit your vmin, vmax and cmap if you don't like greyscale colormap
plt.colorbar()
plt.show()


#imgref=S2file+'_B02.jp2'
#XML_File='/home/ardhuin/Téléchargements/S2A_MSIL1C_20160429T183252_N0201_R027_T11SMS_20160429T184037.SAFE/GRANULE/L1C_T11SMS_A004457_20160429T184037/MTD_TL.xml'
#s2_sensor_angs(XML_File, imgref, '', '')

