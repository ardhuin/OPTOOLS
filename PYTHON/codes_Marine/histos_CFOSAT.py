from functions_cfosat_v1 import *

# def set_histo_to_zero

### ---------------------------------------------------------------------------------------------------------------
### --- HISTO ( = functions for 1 file) -----------------------------------------------------------------------------
### ---------------------------------------------------------------------------------------------------------------	
def histo_std_swh_nadir(i,filetrack, Hs_bins,is1Hz=1,isfiltered=0):
	ds_1Hz = read_nadir_from_L2_CNES(filetrack,flag_1Hz=is1Hz)
	if ds_1Hz.variables_ok==True:
		if isfiltered:
			ds_1Hz = ds_1Hz.isel(time0=np.where((ds_1Hz.swh_flag==0)&(ds_1Hz.swh_filtered>0))[0])
		else:
			ds_1Hz = ds_1Hz.isel(time0=np.where((ds_1Hz.swh_flag==0))[0])
		counts0,bins=np.histogram(ds_1Hz.swh_std,bins=Hs_bins)
		is_good_track = 1
	else:
		is_good_track = 0
		if np.size(Hs_bins)==1:
			counts0 = np.zeros((Hs_bins))
			bins = np.zeros((Hs_bins+1))
		else:
			counts0= np.zeros(len(Hs_bins)-1)
			bins= Hs_bins
	return i,is_good_track,counts0,bins


def histo_swh_nadir(i,filetrack, Hs_bins,is1Hz = 1,isfiltered=0):
	ds_1Hz = read_nadir_from_L2_CNES(filetrack,flag_1Hz=is1Hz)
	if ds_1Hz.variables_ok==True:
		if isfiltered:
			ds_1Hz = ds_1Hz.isel(time0=np.where((ds_1Hz.swh_flag==0)&(ds_1Hz.swh_filtered>0))[0])
		else:
			ds_1Hz = ds_1Hz.isel(time0=np.where((ds_1Hz.swh_flag==0))[0])
		counts0,bins=np.histogram(ds_1Hz.swh,bins=Hs_bins)
		is_good_track = 1
	else:
		is_good_track = 0
		if np.size(Hs_bins)==1:
			counts0 = np.zeros((Hs_bins))
			bins = np.zeros((Hs_bins+1))
		else:
			counts0= np.zeros(len(Hs_bins)-1)
			bins= Hs_bins
		
	return i,is_good_track,counts0,bins



def histo_swh_vs_std_nadir(i,filetrack,Hs_bins,Hs_std_bins,is1Hz,isinbox,isfiltered=0):
	ds_1Hz = read_nadir_from_L2_CNES(filetrack,flag_1Hz=is1Hz,flag_boxes=isinbox)
	if ds_1Hz.variables_ok==True:
		if isfiltered:
			ds_1Hz = ds_1Hz.isel(time0=np.where((ds_1Hz.swh_flag==0)&(ds_1Hz.swh_filtered==True))[0])
		else:
			ds_1Hz = ds_1Hz.isel(time0=np.where((ds_1Hz.swh_flag==0))[0])
		#print('len = ',ds_1Hz.dims['time0'])
		#if ds_1Hz.dims['time0']>1:
		try:
			#print(i,' -------- ',ds_1Hz.swh)
			counts0,Hs_edges,std_Hs_edges=np.histogram2d(ds_1Hz.swh,ds_1Hz.swh_std,bins=[Hs_bins,Hs_std_bins])
			is_good_track = 1
		except Exception as inst:
			print(' i = ', i)
			print(ds_1Hz.swh)
			print(inst)
			return i, 0, 0, 0, 0

	else:
		is_good_track = 0
		if np.size(Hs_bins)==1:
			Hs_edges = np.zeros((Hs_bins+1))
			if np.size(Hs_std_bins)==1:
				counts0 = np.zeros((Hs_bins,Hs_std_bins))
				std_Hs_edges = np.zeros((Hs_std_bins+1))
			else:
				counts0 = np.zeros((Hs_bins,len(Hs_std_bins)-1))
				std_Hs_edges = Hs_std_bins
		else:
			Hs_edges = Hs_bins
			if np.size(Hs_std_bins)==1:
				counts0= np.zeros((len(Hs_bins)-1,Hs_std_bins))
				std_Hs_edges = np.zeros((Hs_std_bins+1))
			else:
				counts0= np.zeros((len(Hs_bins)-1,len(Hs_std_bins)-1))
				std_Hs_edges = Hs_std_bins
		
	return i,is_good_track, counts0, Hs_edges, std_Hs_edges


def histo_ratio_vs_wvl(i,filetrack,ratio_bins,wvnb_bins,iwest,inbeam,isfiltered=0):
	ds_boxes = read_nadir_from_L2_CNES(filetrack,flag_1Hz=1,flag_boxes=1)
	if ds_boxes.variables_ok==True:
		if isfiltered:
			ds_boxes = ds_boxes.isel(time0=np.where((ds_boxes.swh_flag==0)&(ds_boxes.swh_filtered>0))[0])
		else:
			ds_boxes = ds_boxes.isel(time0=np.where((ds_boxes.swh_flag==0))[0])
		ratio = ds_boxes.swh_std/ds_boxes.swh
		wavl = ds_boxes.wave_param.isel(nparam=2, iswest=iwest,n_beam_l2=inbeam)
		counts0,ratio_edges,wvnb_edges = np.histogram2d(ratio,wavl,bins=[ratio_bins,wvnb_bins])
		is_good_track=1
	else:
		if np.size(ratio_bins)==1:
			ratio_edges = np.zeros((ratio_bins+1))
		else:
			ratio_edges = ratio_bins
		
		if np.size(wvnb_bins)==1:
			wvnb_edges = np.zeros((wvnb_bins+1))
		else:
			wvnb_edges = wvnb_bins
		
		counts0 = np.zeros((len(ratio_edges)-1,len(wvnb_edges)-1))	
		is_good_track = 0
	return i,is_good_track, counts0 , ratio_edges, wvnb_edges

def histo_ratio_vs_Hs_vs_wvl(i,filetrack,ratio_bins,Hs_bins,wvnb_bins,iwest,inbeam):
	ds_boxes = read_nadir_from_L2_CNES(filetrack,flag_1Hz=1,flag_boxes=1)
	if ds_boxes.variables_ok==True:
		ds_boxes = ds_boxes.isel(time0=np.where(ds_boxes.swh_flag==0)[0])
		ratio = ds_boxes.swh_std/ds_boxes.swh
		wavl = ds_boxes.wave_param.isel(nparam=1, iswest=iwest,n_beam_l2=inbeam)
		#print('size first =',np.shape(np.array([ratio,ds_boxes.swh,wavl])),' , bins = ',np.shape(np.array([ratio_bins,Hs_bins,wvnb_bins])))
		counts0,edges_vec=np.histogramdd((ratio,ds_boxes.swh,wavl),bins=np.array([ratio_bins,Hs_bins,wvnb_bins]))
		ratio_edges = edges_vec[0]
		Hs_edges = edges_vec[1]
		wvnb_edges = edges_vec[2]
		is_good_track = 1
	else:
		if np.size(ratio_bins)==1:
			ratio_edges = np.zeros((ratio_bins+1))
		else:
			ratio_edges = ratio_bins
		if np.size(Hs_bins)==1:
			Hs_edges = np.zeros((Hs_bins+1))
		else:
			Hs_edges = Hs_bins
		if np.size(wvnb_bins)==1:
			wvnb_edges = np.zeros((wvnb_bins+1))
		else:
			wvnb_edges = wvnb_bins
		
		counts0 = np.zeros((len(ratio_edges)-1,len(Hs_edges)-1,len(wvnb_edges)-1))	
		is_good_track = 0
		
	return i,is_good_track, counts0 #, ratio_edges, Hs_edges, wvnb_edges
### ---------------------------------------------------------------------------------------------------------------
### --- GET HISTOS (= functions going through various files) -------------------------------------------------------
### ---------------------------------------------------------------------------------------------------------------

def get_histo_std_swh_nadir(T1,T2,Hs_bins,is1Hz=1):
	list_tracks= get_tracks_between_dates(T1,T2,typeTrack=0,str_to_remove=['TEST','OPER'])
	
	if np.size(Hs_bins)==1:
		counts= np.zeros((Hs_bins))
	else:
		counts= np.zeros(len(Hs_bins)-1)
	for il,trackname in enumerate(list_tracks):
		if il==0:
			bins = Hs_bins
		_,counts0,bins = histo_std_swh_nadir(0,trackname, Hs_bins,is1Hz)
		counts = counts + counts0
	return counts, bins


def get_histo_swh_nadir(T1,T2,Hs_bins,is1Hz = 1):
	list_tracks= get_tracks_between_dates(T1,T2,typeTrack=0,str_to_remove=['TEST','OPER'])
	if np.size(Hs_bins)==1:
		counts= np.zeros((Hs_bins))
	else:
		counts= np.zeros(len(Hs_bins)-1)
	for il,trackname in enumerate(list_tracks):
		if il==0:
			bins = Hs_bins
		_,counts0,bins=histo_swh_nadir(0,trackname, Hs_bins,is1Hz = is1Hz)
		counts = counts + counts0
	return counts, bins


def get_histo_swh_vs_std_nadir(T1,T2,Hs_bins,Hs_std_bins,isinbox=1):
	list_tracks = get_tracks_between_dates(T1,T2,typeTrack=0,str_to_remove=['TEST','OPER'])
	if np.size(Hs_bins)==1:
		if np.size(Hs_std_bins)==1:
			counts = np.zeros((Hs_bins,Hs_std_bins))
		else:
			counts = np.zeros((Hs_bins,len(Hs_std_bins)-1))
	else:
		if np.size(Hs_std_bins)==1:
			counts= np.zeros((len(Hs_bins)-1,Hs_std_bins))
		else:
			counts= np.zeros((len(Hs_bins)-1,len(Hs_std_bins)-1))
	for il,trackname in enumerate(list_tracks):
		if il==0:
			_,counts0,Hs_edges,std_Hs_edges=histo_swh_vs_std_nadir(i,trackname,Hs_bins,Hs_std_bins,isinbox)
		else:
			_,counts0,Hs_edges,std_Hs_edges=histo_swh_vs_std_nadir(i,trackname,Hs_edges,std_Hs_edges,isinbox)
		counts = counts + counts0
	return counts, Hs_edges, std_Hs_edges


def get_histo_ratio_vs_wvnb(T1,T2,ratio_bins,Hs_bins,wvnb_bins,iwest,inbeam):
	list_tracks = get_tracks_between_dates(T1,T2,typeTrack=0,str_to_remove=['TEST','OPER'])
	if np.size(ratio_bins)==1:
		if np.size(Hs_bins)==1:
			if np.size(wvnb_bins)==1:
				counts = np.zeros((ratio_bins,Hs_bins,wvnb_bins))
			else:
				counts = np.zeros((ratio_bins,Hs_bins,len(wvnb_bins)-1))
		else:
			if np.size(wvnb_bins)==1:
				counts= np.zeros((ratio_bins,len(Hs_bins)-1,wvnb_bins))
			else:
				counts= np.zeros((ratio_bins,len(Hs_bins)-1,len(wvnb_bins)-1))
	else:
		if np.size(Hs_bins)==1:
			if np.size(wvnb_bins)==1:
				counts = np.zeros((len(ratio_bins)-1,Hs_bins,wvnb_bins))
			else:
				counts = np.zeros((len(ratio_bins)-1,Hs_bins,len(wvnb_bins)-1))
		else:
			if np.size(wvnb_bins)==1:
				counts= np.zeros((len(ratio_bins)-1,len(Hs_bins)-1,wvnb_bins))
			else:
				counts= np.zeros((len(ratio_bins)-1,len(Hs_bins)-1,len(wvnb_bins)-1))

	for il,trackname in enumerate(list_tracks):
		if il==0:
			_, counts0, ratio_edges, Hs_edges, wvnb_edges = histo_ratio_vs_wvnb(i,filetrack,ratio_bins,Hs_bins,wvnb_bins,iwest,inbeam)
		else:
			_, counts0, ratio_edges, Hs_edges, wvnb_edges = histo_ratio_vs_wvnb(i,filetrack,ratio_edges,Hs_edges,wvnb_edges,iwest,inbeam)
		
		counts = counts + counts0
	return counts, ratio_edges, Hs_edges, wvnb_edges

def filetrack_ODL_from_CNES_boxes(filetrack,PATH_ODL,v,nbeam):
	YYYY = filetrack.split('/')[12]
	DDD = filetrack.split('/')[13]
	filenc = filetrack.split('/')[-1]
	filefolder = filenc.replace('L2_','L2S').replace('.nc','_'+v)
	filefile = filefolder.replace('L2S__','L2S'+f'{(3+nbeam)*2:02d}')+'.nc'
	file_ODL=PATH_ODL+YYYY+'/'+DDD+'/'+filefolder+'/'+filefile
	return file_ODL

def histo_Hs_ribbon_vs_boxes(i,filetrack_box,Hs_bins,Hs_bins2,iw,nbeam):
	PATH_ODL='/home/ref-cfosat-public/datasets/swi_l2s/v1.0/'
	is_track_good=1
	try:
		ds_boxes = read_boxes_from_L2_CNES_work_quiet(filetrack_box)
	except Exception as inst:
		print(inst)
		return i, 0, 0, 0, 0
	filetrack_l2s=filetrack_ODL_from_CNES_boxes(filetrack_box,PATH_ODL,'1.0.0',nbeam)
	try:
		ds_l2s = read_l2s_offnadir_files_work_quiet(filetrack_l2s)
	except Exception as inst:
		print(inst,filetrack_l2s)
		is_track_good=0
		
	if is_track_good==1:
		try:
			ds_l2s = ds_l2s.isel(l2s_angle=0)
			ds_l2s = ds_l2s.rename({'time0':'time01'})#.swap_dims({'time0':'time01'})
			# ind_time_box_2,_,new_ds = get_indices_box_for_ribbon(ds_boxes,ds_l2s)
			ds_boxes = ds_boxes.isel(iswest=iw,n_beam_l2=nbeam)
			Hs_ribbon = np.zeros((len(ds_boxes.time0)))
			for itime in range(len(ds_boxes.time0)):
				ind_time_box_2 = np.where(
					((
					    (ds_boxes.lon_corners.isel(n_corners=0)<=ds_boxes.lon_corners.isel(n_corners=2))&(
					    (ds_l2s.lon>=ds_boxes.lon_corners.isel(n_corners=0))&
					    (ds_l2s.lon<=ds_boxes.lon_corners.isel(n_corners=2)))
					)|
					(
					    (ds_boxes.lon_corners.isel(n_corners=0)> ds_boxes.lon_corners.isel(n_corners=2))&(
					    (ds_l2s.lon>=ds_boxes.lon_corners.isel(n_corners=0))|
					    (ds_l2s.lon<=ds_boxes.lon_corners.isel(n_corners=2)))
					))&
					(
					    (ds_l2s.lat>=ds_boxes.lat_corners.isel(n_corners=0))&
					    (ds_l2s.lat<=ds_boxes.lat_corners.isel(n_corners=2))
					)&
					(
					    (ds_l2s.phi_vector>=iw*180)&
					    (ds_l2s.phi_vector<=iw*180+180)
					))[0]
				
				ribbons_for_spec = ds_l2s.isel(time01=ind_time_box_2)
				ribbons_for_spec2 = ribbons_for_spec.sortby('phi_vector')
				
				ribbons_for_spec2 = ribbons_for_spec2.assign(dphi=np.gradient(ribbons_for_spec2.phi_vector))
				RR = ribbons_for_spec2['wave_spectra_kth_hs']*ribbons_for_spec2['dk']*ribbons_for_spec2['dphi']*np.pi/180
				Hs_ribbon[itime] = 4*np.sqrt(RR.sum(dim=['time01','nk']))					          
					
			counts0,Hs_edges,Hs_edges2=np.histogram2d(ds_boxes["wave_param"].isel(nparam=0),Hs_ribbon,bins=[Hs_bins,Hs_bins2])
		except Exception as inst:
			print(inst,filetrack_l2s)
			is_track_good=0
	
	if is_track_good==1:
		return i, 1,counts0, Hs_edges,Hs_edges2
	else:
		return i, 0, 0, 0, 0

