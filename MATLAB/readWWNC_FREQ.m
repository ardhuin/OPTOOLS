
function [lat,lon,freq,df,time,ef,th1m,sth1m,th2m,sth2m]=readWWNC_FREQ(filename)
%
% Reads all or a subset of a NetCDF file, with a frequency dimension.
% If date1 is specified: takes only the closest dates in the file
% if lon1 is specificied: takes only the closest longitude ...

%
% 1. Opens file and gets dimensions of arrays
%
finfo = ncinfo(filename);
varlist = {finfo.Variables.Name};

fid=netcdf.open(filename,'NC_NOWRITE');
[ndims,nvars,ngatts,unlimdimid]=netcdf.inq(fid);

dimstat = netcdf.inqDimID(fid,'station');
dimtime = netcdf.inqDimID(fid,'time');
dimf = netcdf.inqDimID(fid,'frequency');
[d0,ns]=netcdf.inqDim(fid,dimstat);
[d3,nt]=netcdf.inqDim(fid,dimtime);
[d4,nf]=netcdf.inqDim(fid,dimf);

varf = netcdf.inqVarID(fid,'frequency');
varf1 = netcdf.inqVarID(fid,'frequency1');
varf2 = netcdf.inqVarID(fid,'frequency2');
varlon = netcdf.inqVarID(fid,'longitude');
varlat = netcdf.inqVarID(fid,'latitude');
varef = netcdf.inqVarID(fid,'ef');
varth1m = netcdf.inqVarID(fid,'th1m');
varsth1m = netcdf.inqVarID(fid,'sth1m');
varth2m = netcdf.inqVarID(fid,'th2m');
varsth2m = netcdf.inqVarID(fid,'sth2m');

freq=netcdf.getVar(fid,varf);
freq1=netcdf.getVar(fid,varf1);
freq2=netcdf.getVar(fid,varf2);
df=freq2-freq1;
lon=netcdf.getVar(fid,varlon);
lat=netcdf.getVar(fid,varlat);
%
% We assume that the date reference is 1 Jan 1990.
% This is normally written in the time attributes
%
time0=datenum(1990,1,1);
vartime = netcdf.inqVarID(fid,'time');
time=netcdf.getVar(fid,vartime)+time0;

ef=double(netcdf.getVar(fid,varef));
th1m=double(netcdf.getVar(fid,varth1m));
sth1m=double(netcdf.getVar(fid,varsth1m));
th2m=double(netcdf.getVar(fid,varth2m));
sth2m=double(netcdf.getVar(fid,varsth2m));

netcdf.close(fid);
%    pcolor(lon1,lat1,squeeze(val1(:,:,1))');shading flat;

