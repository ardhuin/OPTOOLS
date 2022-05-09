from functions_alti_simulator import *
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")

if len(sys.argv)>=2:
	Hs = int(sys.argv[1])
	T0 = int(sys.argv[2])
	sk_k = np.float(sys.argv[3])
else:
	print('Define Hs, T0 and sk ')
	#return

# mp_investigate_radius(Hs,T0,sk_k):
nx = 2048
nx= 2048+400
ny = 2048
nx= 2048+400
dx = 10 # [m]
dy = 10 # [m]

## --- read_spectra
theta_m=30
D=1000

Z1, kX, kY = def_spectrum_for_surface(nx=nx,ny=ny,dx=dx,dy=dy,theta_m=theta_m,D=D,T0=T0,Hs=Hs,
	                 sk_theta=sk_k,sk_k=sk_k,typeSpec='Gaussian')
pool = mp.Pool(mp.cpu_count()-1)

results = []

results = pool.starmap_async(process_investigateR_1surface, [(i,Z1,kX,kY) for i in range(100)]).get()
# # where process_for_1_station is defined :
# # def process_for_1_station(i,stat,PATH,pathimage):
# #     blabla
# #     return whatever
pool.close()

ds=xr.Dataset(data_vars=dict(dist_disk_retrack=(["n","x2"],np.squeeze([result[1] for result in results])),
				dist_ring_retrack=(["n","x2","x1"],np.squeeze([result[2] for result in results])),
				Hs_retrack=(["n","nsamp"],np.squeeze([result[7] for result in results])),
				Hs_std_ring=(["n","nsamp","x2","x1"],np.squeeze([result[6] for result in results])),
				Hs_std_disk=(["n","nsamp","x2"],np.squeeze([result[5] for result in results])),
				),
				coords=dict(radi1=(["x1"],np.squeeze(results[0][3])),
				radi2=(["x2"],np.squeeze(results[0][4])),
				Nb_rand=(["n"],np.squeeze([result[0] for result in results])),
				Hs=Hs,
				T0=T0,
				sk_k=sk_k,
				),
				)
Hsi = int(Hs)
sk_ki = int(np.floor(1000*sk_k))
T0i = int(T0)
pathname = '/home1/datawork/mdecarlo/AltiProcessing/InvestigateRadius/distR_Hs_'+f'{Hsi:02d}'+'_T0_'+f'{T0i:02d}'+'_sk_'+f'{sk_ki:03d}'+'.nc'
ds.to_netcdf(pathname)

