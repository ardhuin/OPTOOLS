import matplotlib.pyplot as plt
from S2_read import *
S2file='/home/ardhuin/ADMIN/PROPOSALS/EE11/S2/DATA/T11SMS_20160429T183252';
boxi=[5600,6400,3600,4400] 
bands=['B04','B03','B02'];
imgs=S2_read(S2file,boxi,bands)

plt.imshow((imgs[0,:,:])) #edit your vmin, vmax and cmap if you don't like greyscale colormap
plt.colorbar()
plt.show()

