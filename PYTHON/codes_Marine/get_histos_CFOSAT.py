# --- get histos CFOSAT ---------------------
from histos_CFOSAT import *
import multiprocessing as mp

PATH_storage = '/home1/datawork/mdecarlo/AltiProcessing/Histos_CFOSAT/'
isfiltered = 0
if len(sys.argv)>=2:
	type_hist = int(sys.argv[1])
else:
	type_hist = 0
if type_hist<3:
	if len(sys.argv)>=3:
		is1Hz = int(sys.argv[2])
	else:
		is1Hz = 0
else:
	if len(sys.argv)>=3:
		iwest = int(sys.argv[2])
		inbeam = int(sys.argv[3])
	else:
		iwest = 0
		inbeam = 0

if isfiltered :
	suff='_filtered'
else:
	suff=''
T1=pd.to_datetime('2019-06-13T12:00:00')
#T2=pd.to_datetime('2019-06-15')
T2=pd.to_datetime('2021-01-06')
#indi = 0
list_tracks= get_tracks_between_dates(T1,T2,typeTrack=0,str_to_remove=['TEST','OPER'])#[indi:]

pool = mp.Pool(1)

results = []
Hs_bins = np.append(np.append(-1,np.linspace(0,30.5,200)),100)
std_Hs_bins = np.append(np.append(-1,np.logspace(np.log10(10**-2),np.log10(1.5),150)),100)
ratio_bins = np.logspace(np.log10(10**-2),np.log10(65),101)
wvnb_bins = np.logspace(np.log10(30),np.log10(600),150)

if type_hist==0:  # histo SWH for 1hz or 5hz
	results = pool.starmap_async(histo_swh_nadir, [(i, filetrack,Hs_bins,is1Hz,isfiltered) for i, filetrack in enumerate(list_tracks)]).get()
elif type_hist==1: # histo STD(SWH) for 1hz or 5hz
	results = pool.starmap_async(histo_std_swh_nadir, [(i, filetrack,std_Hs_bins,is1Hz,isfiltered) for i, filetrack in enumerate(list_tracks)]).get()
elif type_hist==2: # SWH vs STD(SWH) for 1hz or 5hz
	results = pool.starmap_async(histo_swh_vs_std_nadir, [(i, filetrack,Hs_bins,std_Hs_bins,is1Hz,0,isfiltered) for i, filetrack in enumerate(list_tracks)]).get()
elif type_hist==3: # SWH vs STD(SWH) for boxes !!!!! 
	results = pool.starmap_async(histo_swh_vs_std_nadir, [(i, filetrack,Hs_bins,std_Hs_bins,1,1,isfiltered) for i, filetrack in enumerate(list_tracks)]).get()
elif type_hist==4: # STD(SWH)/SWH vs HS vs wvlength /!\ wvnb in the function name is a mistake /!\ HEAVY CALCULATION
	results = pool.starmap_async(histo_ratio_vs_Hs_vs_wvl,[(i,filetrack,ratio_bins,Hs_bins,wvnb_bins,iwest,inbeam) for i,filetrack in enumerate(list_tracks)]).get()
elif type_hist==5:
	results = pool.starmap_async(histo_ratio_vs_wvl,[(i,filetrack,ratio_bins,wvnb_bins,iwest,inbeam,isfiltered) for i,filetrack in enumerate(list_tracks)]).get()
elif type_hist==6:
# histo_Hs_ribbon_vs_boxes(i,filetrack_box,Hs_bins,Hs_bins2,iw,nbeam)
	results = pool.starmap_async(histo_Hs_ribbon_vs_boxes,[(i,filetrack,Hs_bins,Hs_bins[0:-2],iwest,inbeam) for i,filetrack in enumerate(list_tracks)]).get()

pool.close()

# i,is_good_track, counts0, Hs_edges, std_Hs_edges
r_i = np.array([result[0] for result in results])

r_is_good_track = np.array([result[1] for result in results])
ind = np.where(r_is_good_track==1)[0][0]

r_histo  = np.zeros(results[ind][2].shape)

for result in results:
	if result[1]==1: # if is_good_track==1
		r_histo = r_histo + result[2]
if type_hist!=4:
	r_bins0 = results[ind][3]
	if type_hist>=2:
		r_bins1 = results[ind][4]

if type_hist==0:
	if is1Hz:
		savename = PATH_storage+'Hs_histo_CFOSAT'+suff
	else:
		savename = PATH_storage+'Hs_histo_5Hz_CFOSAT'+suff
	np.savez(savename,Hs_histo = r_histo,Hs_bins = r_bins0,num_tracks=r_i.max(),num_tracks_good=np.sum(r_is_good_track))
elif type_hist==1:
	if is1Hz:
		savename = PATH_storage+'Hs_std_histo_CFOSAT'+suff
	else:
		savename = PATH_storage+'Hs_std_histo_5Hz_CFOSAT'+suff	
	np.savez(savename,Hs_std_histo = r_histo,Hs_std_bins = r_bins0,num_tracks=r_i.max(),num_tracks_good=np.sum(r_is_good_track))
elif type_hist==2:
	if is1Hz:
		savename = PATH_storage+'Hs_vs_std_1Hz_histo_CFOSAT'+suff
	else:
		savename = PATH_storage+'Hs_vs_std_5Hz_histo_CFOSAT'+suff
	np.savez(savename,Hs_vs_std_histo = r_histo,Hs_bins = r_bins0,Hs_std_bins = r_bins1,num_tracks=r_i.max(),num_tracks_good=np.sum(r_is_good_track))
elif type_hist==3:
	savename = PATH_storage+'Hs_vs_std_boxes_histo_CFOSAT'+suff
	np.savez(savename,Hs_vs_std_histo = r_histo,Hs_bins = r_bins0,Hs_std_bins = r_bins1,num_tracks=r_i.max(),num_tracks_good=np.sum(r_is_good_track))
elif type_hist==4:
	savename = PATH_storage+'ratio_vs_Hs_vs_wvl_boxes_histo_CFOSAT_'+str(iwest)+'_'+str(inbeam)+suff
	np.savez(savename,ratio_vs_Hs_vs_wvl_histo = r_histo,ratio_bins = ratio_bins,Hs_bins = Hs_bins,wvnb_bins=wvnb_bins,num_tracks=r_i.max(),num_tracks_good=np.sum(r_is_good_track))
elif type_hist==5:
	savename = PATH_storage+'ratio_vs_wvl_boxes_histo_CFOSAT_'+str(iwest)+'_'+str(inbeam)+suff
	np.savez(savename,ratio_vs_wvl_histo = r_histo,ratio_bins = ratio_bins,wvnb_bins=wvnb_bins,num_tracks=r_i.max(),num_tracks_good=np.sum(r_is_good_track))
elif type_hist==6:
	savename = PATH_storage+'Hs_ribbon_vs_boxes_histo_CFOSAT_'+str(iwest)+'_'+str(inbeam)+suff
	np.savez(savename,Hs_ribbon_vs_boxes_histo = r_histo,Hs_bins = r_bins0,Hs_bins2=r_bins1,num_tracks=r_i.max(),num_tracks_good=np.sum(r_is_good_track))

