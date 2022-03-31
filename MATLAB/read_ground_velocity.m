function [V, VG, Height]=read_ground_velocity(xDoc, lat0);


% % FIND XML FILE
% if numb=='00000',
% fol=dir([dirgen(1:end-42),'DATASTRIP\DS*']); fol=fol.name;
% gdir=[dirgen(1:end-42),'DATASTRIP\',fol,'\']; gname='MTD_DS.xml';
% else
% fol=dir([dirdata(1:end-80),'DATASTRIP\*DS*']); fol=fol.name;
% gdir=[dirdata(1:end-80),'DATASTRIP\',fol,'\']; gname=dir([gdir,'*xml']); gname=gname.name;
% end
% xDoc=xmlread([gdir,gname]);

% READ POSITION AND VELOCITY (in m)
p0=xDoc.getElementsByTagName('GPS_Points_List').item(0).getElementsByTagName('GPS_Point');
P=[]; V=[]; T=[];
for ip=0:p0.getLength-1
p=p0.item(ip).getElementsByTagName('POSITION_VALUES'); 
P(ip+1,:)=str2num(p.item(0).getFirstChild.getData)./1e3;
v=p0.item(ip).getElementsByTagName('VELOCITY_VALUES'); 
V(ip+1,:)=str2num(v.item(0).getFirstChild.getData)/1e3;
t=p0.item(ip).getElementsByTagName('GPS_TIME'); 
%T(ip+1)=datenum(char(t.item(0).getFirstChild.getData),'yyyy-mm-ddTHH:MM:SS');
end

V=sqrt(V(:,1).^2+V(:,2).^2+V(:,3).^2);
X=P(:,1); Y=P(:,2); Z=P(:,3);

% % the closest time point
% v=find(abs(T-datenum(time(1:18),'yyyy-mm-ddTHH:MM:SS'))==min(abs(T-datenum(time(1:18),'yyyy-mm-ddTHH:MM:SS'))));
% %v=round(length(T)/2);
% %v=150;
% X=X(v); Y=Y(v); Z=Z(v); V=V(v);



lat=atan(Z./sqrt(X.^2+Y.^2));
v=find( abs(lat*180/pi-lat0)==min(abs(lat*180/pi-lat0))  ); v=v(1);
X=X(v); Y=Y(v); Z=Z(v); V=V(v); lat=lat(v);

% EARTH RADIUS from ELLIPSOID ERS (2003)
a=6378136.6; b=6356751.9; 	
R=1./sqrt(cos(lat).^2./a^2+sin(lat).^2./b^2); %earth radius

% GROUND VELOCITY
VG=V.*R./sqrt(X.^2+Y.^2+Z.^2); 
Height=sqrt(X.^2+Y.^2+Z.^2)-R;

%disp([VG Height])
%disp(lat*180/pi);

% figure, plot(VG,'g.');
% figure, plot(Height,'r.')
%figure, plot(lat*180/pi,'k.')



