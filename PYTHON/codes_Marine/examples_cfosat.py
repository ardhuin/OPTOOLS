#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================================================
# === 0. Import Packages + Define paths and files ==================================
# ==================================================================================

from functions_cfosat_v1 import *

# --- Example of paths and files ---
PATH_L2_CNES=os.path.join('/home/datawork-cersat-public/provider/cnes/satellite/l2/',
'cfosat/swim/swi_l2____/op05/5.1.2/')

L2_CNES_ExampleFiles=[os.path.join(PATH_L2_CNES,'2019/354/CFO_OP05_SWI_L2_____F_20191220T193316_20191220T210616.nc')]
L2_CNES_ExampleFiles.append(os.path.join(PATH_L2_CNES,'2019/354/CFO_OP05_SWI_L2_____F_20191220T210615_20191220T223901.nc')) 

PATHOffNadir='/home/ref-cfosat-public/datasets/swi_l2s/v0.4/'
PATHOffNadirExample = os.path.join(PATHOffNadir,
                    '2019/354/CFO_OPER_SWI_L2S____F_20191220T193316_20191220T210616_0.4.0/')
offnadirExampleFiles=[os.path.join(PATHOffNadirExample,'CFO_OPER_SWI_L2S02__F_20191220T193316_20191220T210616_0.4.0.nc')]
offnadirExampleFiles.append(os.path.join(PATHOffNadirExample,'CFO_OPER_SWI_L2S04__F_20191220T193316_20191220T210616_0.4.0.nc'))
offnadirExampleFiles.append(os.path.join(PATHOffNadirExample,'CFO_OPER_SWI_L2S06__F_20191220T193316_20191220T210616_0.4.0.nc'))
offnadirExampleFiles.append(os.path.join(PATHOffNadirExample,'CFO_OPER_SWI_L2S08__F_20191220T193316_20191220T210616_0.4.0.nc'))
offnadirExampleFiles.append(os.path.join(PATHOffNadirExample,'CFO_OPER_SWI_L2S10__F_20191220T193316_20191220T210616_0.4.0.nc'))

PATH_L3 = os.path.join('/home/ref-cmems-public/tac/wave/WAVE_GLO_WAV_L3_SWH_NRT_OBSERVATIONS_014_001/',
            'dataset-wav-alti-l3-swh-rt-global-cfo')
L3_CMEMS_ExampleFiles=[os.path.join(PATH_L3,'2019/12/global_vavh_l3_rt_cfo_20191220T180000_20191220T210000_20200710T103642.nc')]
L3_CMEMS_ExampleFiles.append(os.path.join(PATH_L3,'2019/12/global_vavh_l3_rt_cfo_20191220T210000_20191221T000000_20200710T103642.nc'))


def print_info_storage():
	print(" ___ CNES L2 (nadir data + boxes) __________________________________________________________ ")
	print('Path :')
	print(PATH_L2_CNES+'YYYY/DDD/')
	print("____________________________________________________________________________________________")
	print("")
	print(" ___ ODL L2S (ribbon product) ______________________________________________________________ ")
	print('Path :')
	print(PATHOffNadir+'YYYY/DDD/')
	print("____________________________________________________________________________________________")
	print("")
	print(" ___ CMEMS L3 (nadir data - calibrated against J3 +buoys) __________________________________")
	print('Path :')
	print(PATH_L3+'YYYY/MM/')

dphi_safe=6

lon_min = -34#-35
lon_max = -30.5#-30
lat_min = 42#38
lat_max = 45#50

def plot_info_apply_box_selection_to_ds(ds0):
	print('FUNCTION apply_box_selection_to_ds(ds,lon_min,lon_max,lat_min,lat_max,flag_coords)')
	print('     applies to a dataset for a selected incidence angle (i.e. dims = (time0, k)')
	print('     returns a dataset corresponding to the region')
	print('     flag_coords = {0,1,2}')
	print('                   * 0 = based on lon,lat at middle of ribbon')
	print('                   * 1 = any lon,lat of a ribbon is inside the box')
	print('                   * 2 = all lon,lat of a ribbon are inside the box')
	
	aa=3
	fig2, axs = plt.subplots(1,3,figsize=(24, 7),subplot_kw={'projection':ccrs.PlateCarree()},sharex=True,sharey=True)
	cNorm = mcolors.Normalize(vmin=0, vmax=2)
	jet = plt.get_cmap('jet')
	scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
	
	for fl in range(3):
		ax0, g1 =init_map_cartopy(axs[fl],limx=(lon_min-0.5,lon_max+0.5),limy=(lat_min-0.5,lat_max+0.5))
		ds=apply_box_selection_to_ds(ds0.isel(l2s_angle=aa),
						lon_min,lon_max,lat_min,lat_max,flag_coords=fl)

		axs[fl].scatter(ds.wav_lon, ds.wav_lat,s=18,marker='s',c=ds.wave_spectra,cmap=jet,norm=cNorm)
		_=axs[fl].plot((lon_min-0.5,lon_max+0.5),(lat_max,lat_max),'--r',linewidth=2)
		_=axs[fl].plot((lon_min-0.5,lon_max+0.5),(lat_min,lat_min),'--r',linewidth=2)
		_=axs[fl].plot((lon_min,lon_min),(lat_min-0.5,lat_max+0.5),'--r',linewidth=2)
		_=axs[fl].plot((lon_max,lon_max),(lat_min-0.5,lat_max+0.5),'--r',linewidth=2)

		_=axs[fl].set_title('flag_coords = '+str(fl))
	_=plt.suptitle('Impact of flag_coords on box selection')

def plot_info_get_macrocycles(ds0):
	print('FUNCTION get_macrocycles(ds)')
	print('     applies to a dataset for a selected incidence angle (i.e. dims = (time0, k)')
	print('     returns a dataset with a new variable "macrocycle_label" containing the number of the macrocycle')
	
	aa=3
	ds = get_macrocycles(ds0.isel(l2s_angle=aa))
	fig2, axs = plt.subplots(figsize=(15, 5),subplot_kw={'projection':ccrs.PlateCarree()})
	ax0, g1 =init_map_cartopy(axs,limx=(lon_min,lon_max),limy=(lat_min,lat_max))
	ds1=apply_box_selection_to_ds(ds,lon_min,lon_max,lat_min,lat_max,flag_coords=0)
	cols = plt.cm.Set1(np.linspace(0,1,9))
	for mm in np.unique(ds1.macrocycle_label):
		ind=np.where(ds1.macrocycle_label==mm)[0]
		axs.plot(ds1.isel(time0=ind).lon,ds1.isel(time0=ind).lat,'o',color=cols[mm%9],label=str(mm))
	_=plt.legend()

def plot_info_draw_macrocycles(ds0):
	print('FUNCTION draw_spectrum_macrocyle_North(ax,ds,args*)')
	print('     this function draws the spectrum for a given macrocycle or half macrocycle') 
	print('     with two half spectrum contours defined by the wavelength filter (wvlmin,wvlmax)')
	print('     and by the safety angle along cfosat track (dphi)')
	print('     inputs are:')
	print('	- the axes for the plot to be on')
	print("	- 'ds'		: the dataset with only one incidence angle + only one macrocycle")
	print("	- 'wvlmin'	: the minimum wavelength to be acounted for (default = min of wavelength vector)")
	print("	- 'wvlmax'	: the maximum wavelength to be accounted for (default = max of wavelength vector)")
	print("	- 'dphi	: the safety angle along cfosat track (default=0)")
	print("	- 'raxis'	: flag to decide if radial axis is freq (=0, default) or wavelength (=1)")
	print("	- 'isNorthward' : flag to decide if the 'north' of the spectrum corresponds to the geographic North (=1, default) of to the track (=0)")
	print("	- 'isNE'	: flag to print only the 4 cardinal points (=0) or to also add the NE, NW, SE and SW (=1, default)")
	print("	- 'cNorm'	: the normalization for cmap")
	print("	- 'cMap'	: the colormap")
	aa=3
	ds01 = get_macrocycles(ds0.isel(l2s_angle=aa))
	ds= ds01.isel(time0=np.where(ds01.macrocycle_label==433)[0])
	ds2 = ds.isel(time0=np.where(ds.phi<=180)[0])
	ds3 = ds.isel(time0=np.where(ds.phi>=180)[0])
	mpl.rcParams.update({'font.size': 14})
	
	
	fig2, axs = plt.subplots(1,2,figsize=(10, 5),subplot_kw={'projection': 'polar'})
	draw_spectrum_macrocyle_North(axs[0],ds,dphi=5,raxis=0,wvlmin=30,wvlmax=500)
	axs[0].set_title('raxis = 0 : freq')
	draw_spectrum_macrocyle_North(axs[1],ds,dphi=5,raxis=1,wvlmin=30,wvlmax=500)
	axs[1].set_title('raxis = 1 : wavelength')
	plt.suptitle('Northward')
	plt.tight_layout()

	fig2, axs = plt.subplots(1,2,figsize=(10, 5),subplot_kw={'projection': 'polar'})
	draw_spectrum_macrocyle_North(axs[0],ds,dphi=5,raxis=0,isNE=1,wvlmin=30,wvlmax=500,isNorthward=0)
	axs[0].set_title('isNE = 1')
	draw_spectrum_macrocyle_North(axs[1],ds,dphi=5,raxis=0,isNE=1,wvlmin=30,wvlmax=500,isNorthward=0)
	axs[1].set_title('isNE = 0')
	plt.suptitle('Trackward')
	plt.tight_layout()
	
	fig2, axs = plt.subplots(1,2,figsize=(10, 5),subplot_kw={'projection': 'polar'})
	draw_spectrum_macrocyle_North(axs[0],ds3,dphi=5,raxis=0,wvlmin=30,wvlmax=500,isNorthward=0)
	axs[0].set_title('Left side')
	draw_spectrum_macrocyle_North(axs[1],ds2,dphi=5,raxis=0,wvlmin=30,wvlmax=500,isNorthward=0)
	axs[1].set_title('Right side')
	plt.suptitle('Trackward')
	plt.tight_layout()

def plot_tracks_all_inci(ds0,ds_nadir,lon_min,lon_max,lat_min,lat_max):
	fig2, axs = plt.subplots(figsize=(14, 8),subplot_kw={'projection':ccrs.PlateCarree()},sharex=True,sharey=True)
	ax0, g1 =init_map_cartopy(axs,limx=(lon_min,lon_max),limy=(lat_min,lat_max))
	colors = plt.cm.cool(np.linspace(0,1,5))
	ds_nadir_box=apply_box_selection_to_ds(ds_nadir,
                                          lon_min,lon_max,lat_min,lat_max,flag_coords=0)
	axs.plot(ds_nadir_box.lon,ds_nadir_box.lat,'*k',label='nadir',zorder=0)
	for aa in range(len(ds0.l2s_angle)):
		ds_inci0=ds0.isel(l2s_angle=aa)
		ds_all_inci=apply_box_selection_to_ds(ds_inci0,
                                          lon_min,lon_max,lat_min,lat_max,flag_coords=0)
		axs.plot(ds_all_inci.lon,ds_all_inci.lat,'o',label=f'{ds_inci0.l2s_angle.data:02d}',color=colors[aa],zorder=6-aa)
	plt.legend()	
	return fig2
	
def plot_tracks_all_inci_CNES_box(ds0,ds_nadir,ds_boxes,lon_min,lon_max,lat_min,lat_max):
	fig2, axs = plt.subplots(figsize=(14, 8),subplot_kw={'projection':ccrs.PlateCarree()},sharex=True,sharey=True)
	ax0, g1 =init_map_cartopy(axs,limx=(lon_min,lon_max),limy=(lat_min,lat_max))
	colors = plt.cm.cool(np.linspace(0,1,5))
	ds_nadir_box=apply_box_selection_to_ds(ds_nadir,lon_min,lon_max,lat_min,lat_max,flag_coords=0)
	axs.plot(ds_nadir_box.lon,ds_nadir_box.lat,'*k',label='nadir',zorder=0)
	strplt = ('-r','-b')
	for isw in range(2):
		ds_boxes_boxW=apply_box_selection_to_ds(ds_boxes.isel(iswest=isw),lon_min,lon_max,lat_min,lat_max,flag_coords=0)
		loncorners=ds_boxes_boxW.lon_corners
		latcorners=ds_boxes_boxW.lat_corners
		axs.plot(loncorners.data.compute(),latcorners.data.compute(),strplt[isw],linewidth=2)

	for aa in range(len(ds0.l2s_angle)):
		ds_inci0=ds0.isel(l2s_angle=aa)
		ds_all_inci=apply_box_selection_to_ds(ds_inci0,
                                          lon_min,lon_max,lat_min,lat_max,flag_coords=0)
		axs.plot(ds_all_inci.lon,ds_all_inci.lat,'o',label=f'{ds_inci0.l2s_angle.data:02d}',color=colors[aa],zorder=6-aa)
	plt.legend()	
	return fig2
	
def plot_ribbons_all_inci_in_box(ds0):
	mpl.rcParams.update({'font.size': 18})
	dphi_safe=5
	fig2, axs = plt.subplots((len(ds0.l2s_angle)//3)+1,3,figsize=(24, 14),subplot_kw={'projection':ccrs.PlateCarree()},sharex=True,sharey=True)
	jet = cm = plt.get_cmap('jet')
	cNorm  = mcolors.Normalize(vmin=0, vmax=2)
	scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

	for aa in range(len(ds0.l2s_angle)):
		ds_aa=ds0.isel(l2s_angle=aa)
		ds_aa_box=apply_box_selection_to_ds(ds_aa,
					          lon_min,lon_max,lat_min,lat_max,flag_coords=1)
		ax0, g1 =init_map_cartopy(axs[aa//3,aa%3],limx=(lon_min,lon_max),limy=(lat_min,lat_max))
		phi_aa = ds_aa_box.phi

		ind_SWIM_safe = np.where(((phi_aa >=(0 + dphi_safe)) & (phi_aa <=(180 - dphi_safe)))
				     |((phi_aa>= (180 + dphi_safe)) & (phi_aa <=(360 - dphi_safe))))[0]

		ind_SWIM_blind=np.setdiff1d(np.arange(len(ds_aa_box.time0)),ind_SWIM_safe)
		if len(ind_SWIM_blind)>0:
			axs[aa//3,aa%3].scatter(ds_aa_box.isel(time0=ind_SWIM_blind).wav_lon,
				       ds_aa_box.isel(time0=ind_SWIM_blind).wav_lat,s=18,marker='s',
				       color=(0.5, 0.5, 0.5),zorder=0)
		axs[aa//3,aa%3].scatter(ds_aa_box.isel(time0=ind_SWIM_safe).wav_lon,
			   ds_aa_box.isel(time0=ind_SWIM_safe).wav_lat,s=18,marker='s',
			   c=ds_aa_box.isel(time0=ind_SWIM_safe).wave_spectra,cmap=jet,norm=cNorm)
		_=axs[aa//3,aa%3].set_title('angle inci :'+str(ds_aa_box.l2s_angle.data))

	_=plt.colorbar(scalarMap,label='Wave spectrum',ax=axs[:,:])
	_=plt.suptitle('blind along track')
	
	
def plot_macrocycles_nadir(ds_offnad0,ds_nadir0):
	swhNorm = mcolors.Normalize(vmin=5, vmax=13)
	jet = cm = plt.get_cmap('jet')
	scalarSWHMap = cmx.ScalarMappable(norm=swhNorm, cmap=jet)
	
	aa=2
	lon_min = -35
	lon_max = -30
	lat_min = 38
	lat_max = 50
	ds_offnad01=get_macrocycles(ds_offnad0.isel(l2s_angle=aa))
	ds_offnad=apply_box_selection_to_ds(ds_offnad01,lon_min,lon_max,lat_min,lat_max,flag_coords=1)
	ds_nadir=apply_box_selection_to_ds(ds_nadir0,lon_min,lon_max,lat_min,lat_max,flag_coords=0)
	
	fig2, ax = plt.subplots(figsize=(16, 12),subplot_kw={'projection':ccrs.PlateCarree()})
	ax0, g1 =init_map_cartopy(ax,limx=(lon_min,lon_max),limy=(lat_min,lat_max))
	
	_=ax.scatter(ds_nadir.lon,ds_nadir.lat,s=1000,c=ds_nadir.swh,cmap=jet,norm=swhNorm)
	_=plt.colorbar(scalarSWHMap,label='SWH [m]')
	cols = plt.cm.Set1(np.linspace(0,1,9))
	for mm in np.unique(ds_offnad.macrocycle_label):
		ind=np.where(ds_offnad.macrocycle_label==mm)[0]
		ax.plot(ds_offnad.isel(time0=ind).lon,ds_offnad.isel(time0=ind).lat,'o',color=cols[mm%9],label=str(mm))
	
def plot_WW3_mesh_resolution(ds_offnad0,ds_nadir0):
	spa_res = [6,15,30]
	aa=2
	lon_min = -35
	lon_max = -30
	lat_min = 38
	lat_max = 49
	# --- define color maps --------------------------
	distNorm = mcolors.Normalize(vmin=0, vmax=100)
	cool = cm = plt.get_cmap('cool') #pt_to_use
	distanceMap = cmx.ScalarMappable(norm=distNorm, cmap=cool)
	reds = cm = plt.get_cmap('hot_r')
	BNNorm = mcolors.Normalize(vmin=-0.1, vmax=1.1) 
	distanceBNMap = cmx.ScalarMappable(norm=BNNorm, cmap=reds)

	fig2, ax = plt.subplots(1,3,figsize=(24, 10),subplot_kw={'projection':ccrs.PlateCarree()},sharex=True,sharey=True)
	ds_offnad=apply_box_selection_to_ds(ds_offnad0.isel(l2s_angle=aa),lon_min,lon_max,lat_min,lat_max,flag_coords=1)
	ds_nadir=apply_box_selection_to_ds(ds_nadir0,lon_min,lon_max,lat_min,lat_max,flag_coords=0)

	dks_offnad = np.zeros(len(ds_offnad.time0))


	for ioffnad in range(len(ds_offnad.time0)):
		dk = haversine(ds_offnad.isel(time0=ioffnad).lat,ds_offnad.isel(time0=ioffnad).lon,ds_nadir.lat,ds_nadir.lon)
		dks_offnad[ioffnad]=dk.min()

	for res in range(len(spa_res)):
		lon_res = np.arange(-40.,-25.,(spa_res[res]/60))
		lat_res = np.arange(35.,52.,(spa_res[res]/60))
		lon_res2D, lat_res2D = np.meshgrid(lon_res,lat_res)
		dist_res_nad=np.zeros(np.shape(lon_res2D))
		#dist_mesh =np.zeros((len(lon_res),len(lat_res)))
		distMeshMax=0
		print(np.shape(lon_res2D))
		for io in range(len(lon_res)):
			print('io = ',io,' over ',len(lon_res))
			for ia in range(len(lat_res)):
				dk = haversine(lat_res[ia],lon_res[io],ds_nadir.lat,ds_nadir.lon)
				dist_res_nad[ia,io] = dk.min()
				dk = haversine(lat_res[ia],lon_res[io],lat_res2D,lon_res2D).flatten()
				distMeshMax=np.maximum(distMeshMax,dk[dk>0].min())
				#dist_mesh[io,ia] = dk[dk>0].min()

		distOffnadMax = dks_offnad.max()
		dist_res_NB=np.zeros(np.shape(dist_res_nad),dtype=bool)
		ind=np.where(dist_res_nad<=(distOffnadMax+distMeshMax))

		dist_res_NB[ind[0],ind[1]]=True

		# --- plot ------------------------------------------
		ax0, g1 = init_map_cartopy(ax[res],limx=(lon_min,lon_max),limy=(lat_min,lat_max))
		ax[res].scatter(ds_offnad.lon,ds_offnad.lat,s=30,c=dks_offnad,cmap=cool,norm=distNorm,zorder=4)
		ax[res].plot(ds_offnad.lon,ds_offnad.lat,'*k',markersize=3,zorder=5)
		#ax[res].scatter(lon_res_2D,lat_res_2D,s=40,c=dist_res_nad,cmap=jet,norm=distNorm)
		ax[res].scatter(lon_res2D,lat_res2D,s=spa_res[res],c=dist_res_NB,cmap=reds,norm=BNNorm,zorder=3)
		ax[res].plot(ds_nadir.lon,ds_nadir.lat,'ok',zorder=2)
		_=ax[res].set_title('resolution '+f'{spa_res[res]:02d}'+'°')
		
	_=plt.colorbar(distanceMap,label='Distance from track [km]',ax=ax[:])

def plot_Nadir_track_vs_model_frame(T1,T2,lon_min,lon_max,lat_min,lat_max,NbHs_min=100):
	list_tracks=get_tracks_between_dates(T1,T2,typeTrack=0,str_to_remove='TEST')
	for filetrack in list_tracks:
		fig=plt.figure(figsize=(20,8))
		ax1=fig.add_subplot(1, 2, 1,projection=ccrs.PlateCarree())
		boolplot = plot_2D_track_vs_model(ax1,filetrack,lon_min,lon_max,lat_min,lat_max,pathWW3=None,NbHs_min=100)
		if boolplot:
			ax2=fig.add_subplot(1,2,2)
			_=plot_1D_track_vs_model(ax2,filetrack,lon_min,lon_max,lat_min,lat_max,pathWW3=None)		
			_=plt.tight_layout()
			
def plot_Nadir_track_vs_model_frame_boxes(T1,T2,lon_min,lon_max,lat_min,lat_max,NbHs_min=100):
	list_tracks=get_tracks_between_dates(T1,T2,typeTrack=0,str_to_remove='TEST')
	for filetrack in list_tracks:
		fig=plt.figure(figsize=(20,8))
		ax1=fig.add_subplot(3, 2, (1,3),projection=ccrs.PlateCarree())
		boolplot = plot_2D_track_vs_model_boxes(ax1,filetrack,lon_min,lon_max,lat_min,lat_max,pathWW3=None,NbHs_min=100)
		if boolplot:
			ax2=fig.add_subplot(3,2,5)
			_=plot_1D_track_vs_model(ax2,filetrack,lon_min,lon_max,lat_min,lat_max,pathWW3=None)	
			ax3=fig.add_subplot(3,2,2)
			ax4=fig.add_subplot(3,2,4)
			ax5=fig.add_subplot(3,2,6)
			plot_1D_track_boxes([ax3,ax4,ax5],filetrack,lon_min,lon_max,lat_min,lat_max,pathWW3=None)
			#_=plt.tight_layout()	

def plot_usage_get_nearest_model_interpolator(ds0):
	print('FUNCTION get_nearest_model_interpolator(args*)')
	print('     this function obtains interpolators for the lon and lat of the model') 
	print('     i.e. doing lat_interpolator(lat_CFOSAT) returns the model latitudes ')
	print('     closest to the track latitudes.')
	print('     Optionnal inputs are:')
	print('	- lon_lims : the model limits of lon (default = [-180, 180])')
	print('	- lat_lims : the model limits of lat (default = [-78, 83])')
	print('	- lon_step : the model step in lon (default = 0.5)')
	print('	- lat_step : the model step in lat (default = 0.5)')
	print('.....')
	print('associated function : ')
	print('FUNCTION get_nearest_model_interpolator_index(args*)')	
	print("     takes the same inputs as 'get_nearest_model_interpolator'")
	print('     this function obtains interpolators for the INDICES of lon and lat of the model') 
	print('     i.e. doing lat_interpolator(lat_CFOSAT) returns the model latitude INDICES')

	lon_interpolator, lat_interpolator=get_nearest_model_interpolator()
	ds = ds0.isel(iswest=1)
	plt.figure(figsize=(12,8))
	plt.plot(ds.lon,ds.lat,'.',label='CFOSAT points W',zorder=1)
	plt.plot(lon_interpolator(ds.lon),lat_interpolator(ds.lat),'*',label='interpolated over model W',zorder=2)
	for k in range(len(ds.lon)):
		X=[ds.isel(time0=k).lon,lon_interpolator(ds.isel(time0=k).lon)]
		Y=[ds.isel(time0=k).lat,lat_interpolator(ds.isel(time0=k).lat)]
		plt.plot(X,Y,'-k',zorder=0)

	ds = ds0.isel(iswest=0)
	plt.plot(ds.lon,ds.lat,'.',label='CFOSAT points E',zorder=1)
	plt.plot(lon_interpolator(ds.lon),lat_interpolator(ds.lat),'*',label='interpolated over model E',zorder=2)
	for k in range(len(ds.lon)):
		X=[ds.isel(time0=k).lon,lon_interpolator(ds.isel(time0=k).lon)]
		Y=[ds.isel(time0=k).lat,lat_interpolator(ds.isel(time0=k).lat)]
		plt.plot(X,Y,'-k',zorder=0)   

	plt.grid(True)
	plt.legend()




	
