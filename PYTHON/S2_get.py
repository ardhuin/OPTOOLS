import matplotlib.pyplot as plt
from S2_read import *
from s2_angs import *
S2file='/home/ardhuin/ADMIN/PROPOSALS/EE11/S2/DATA/T11SMS_20160429T183252';
S2angs=get_angleobs(S2file+'.xml');
boxi=[5600,6400,3600,4400] # East West South North 
bands=['B04','B03','B02'];  
imgs=S2_read(S2file,boxi,bands)
img1=imgs[0,:,:]
plt.imshow(np.fliplr(np.transpose(img1))) #edit your vmin, vmax and cmap if you don't like greyscale colormap
plt.colorbar()
plt.show()

