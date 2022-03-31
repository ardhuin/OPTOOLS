% Playing with S2 data
% Version 1.0 by F. Ardhuin, Nov. 2020

d2r=pi/180;


% defines grey color scale
swim=linspace(0,256,257)/256;
   %swim3=repmat([0. 0.60 0.60],257,1)+swim'*[0. 0.25 0.40];
   swim3=swim'*[1. 1. 1];
 
%
% Choice of image and sub-images: the actual reading of the JPEG2000 files is done by S2_read ... you can replace this ... 
% 
choice = 3
bands=[4 3 8 2 ];
bands=[4  3  2 ];
%bands=[12 11 5 ]; %all 20 m
%bands=[12 11 2 ];  %20 and 10 m
boxi=[5600 6400  3600 4400];
S2file='T11SMS_20160429T183252';
[imgs,S2xml,DSxml,S2txt,nx,ny,dx,dy,x,y,boxi]=S2_read(S2file,boxi,bands); 

% now grabs the view angles from the xml structure ... 
%      <Geoposition resolution="10">
%        <ULX>399960</ULX>
%        <ULY>3700020</ULY>



zonestring=S2xml.Children(4).Children(2).Children(2).Children(1).Data;
sizestring1=S2xml.Children(4).Children(2).Children(6).Children(2).Children(1).Data;
sizestring2=S2xml.Children(4).Children(2).Children(6).Children(4).Children(1).Data;
posstring1=S2xml.Children(4).Children(2).Children(12).Children(2).Children(1).Data;
posstring2=S2xml.Children(4).Children(2).Children(12).Children(4).Children(1).Data;
xc=[ boxi(1) boxi(2) boxi(2) boxi(1)].*dx;
yc=[ boxi(3) boxi(3) boxi(4) boxi(4)].*dy;
zones=strsplit(zonestring,' ');
UTMzone=sscanf(char(zones(end)),'%d');
X0=sscanf(char(posstring1),'%d');
NX=sscanf(char(sizestring1),'%d');
NY=sscanf(char(sizestring2),'%d');
Y0=sscanf(char(posstring2),'%d');
[lat,lon]=utm2ll(xc+X0,Y0-NY*dy+yc,UTMzone)  % to be verified

latcenter=0.5.*(lat(2)+lat(3));

loncenter=0.5.*(lon(1)+lon(2));


geom=S2xml.Children(4);
S2szen=geom.Children(4).Children(2).Children(2).Children(6);
S2sazi=geom.Children(4).Children(2).Children(4).Children(6);

S2view=geom.Children(4).Children(6);
ntables=length(S2txt);

mean(std(double(img1)))./mean(mean(double(img1)))

nzen1=(length(S2szen.Children)-1)/2;
for i=1:nzen1
   A=S2szen.Children(i*2+1);
   A=S2szen.Children(2).Children.Data;
   B=cell2mat(textscan(char(A),'%f'));
   nzen2=length(B);
   A=S2sazi.Children(2).Children.Data;
   C=cell2mat(textscan(char(A),'%f'));
   if i==1
     sunzen=zeros(nzen2,nzen1);
     sunazi=zeros(nzen2,nzen1);
     viewzen=zeros(nzen2,nzen1,13,12)+NaN;
     viewazi=zeros(nzen2,nzen1,13,12)+NaN;
   end
   sunzen(:,i)=B;
   sunazi(:,i)=C;
   for j=1:ntables
   i1=S2txt(j,1)+1; % band index
   i2=S2txt(j,2);   % detector index
   A=geom.Children(4).Children(j*2+4).Children(2).Children(6).Children(i*2).Children(1).Data;
   B=cell2mat(textscan(char(A),'%f'));
   viewzen(:,i,i1,i2)=flipud(B);
   A=geom.Children(4).Children(j*2+4).Children(4).Children(6).Children(i*2).Children(1).Data;
   C=cell2mat(textscan(char(A),'%f'));
   viewazi(:,i,i1,i2)=flipud(C);
   end
end

% Warning : this is only correct for  <COL_STEP unit="m">5000</COL_STEP>


indx=1+round((NX*10/dx-0.5*(boxi(1)+boxi(2)))/(5000/dx));  % This takes the nearest cell in matrix 
indy=1+round((NY*10/dx-0.5*(boxi(3)+boxi(4)))/(5000/dx));



% Unit vector from point in swath to sat
sat=zeros(2,3);
mid=sat;
sun=zeros(1,3);
thetasun=sunzen(indx,indy);
phisun  =sunazi(indx,indy);

sun(1,1)=sin(thetasun.*d2r).*cos(phisun.*d2r);
sun(1,2)=sin(thetasun.*d2r).*sin(phisun.*d2r);
sun(1,3)=cos(thetasun.*d2r);

thetav=[ 0. 0. 0. 0.];
phiv=thetav;
offspec=thetav;
phitrig=thetav;
for jb=1:4;
   if jb <= length(bands) 
   ib=bands(jb);
   else
   ib=bands(end);
   end
   if ib > 8
    ib=ib+1;
   end
  ib=ib*1
   thetasat=max(viewzen(indx,indy,ib,:));
   phisat=max(viewazi(indx,indy,ib,:));
   thetav(jb)=thetasat;
   phiv(jb)=phisat;
   
   sat(jb,1)=sin(thetasat.*d2r).*cos(phisat.*d2r);
   sat(jb,2)=sin(thetasat.*d2r).*sin(phisat.*d2r);
   sat(jb,3)=cos(thetasat.*d2r);

   mid(jb,:)=sun+sat(jb,:);  % vector that bisects the sun-target-sat angle 
   midn=sqrt(mid(jb,1).^2+mid(jb,2).^2+mid(jb,3).^2);
   mid(jb,:)=mid(jb,:)./midn;
   
   offspec(jb)=acos(mid(jb,3))./d2r                % off-specular angle
   phitrig(jb)=atan2(mid(jb,2),mid(jb,1))./d2r   % azimuth of bistatic look
end

[V, VG, Height]=read_ground_velocity(DSxml, latcenter);

RE=4E7/(2*pi); % Earth radius
%H=786000;
%inclination=98.5;

% 1) law of sines in a triangle in the plane Earth center (C) - satellite (S)  - point observed (O). gamma is the off-nadir angle at the satellite: 
% hence in the triangle SCO, the side OC has length RE, side CO has length RE+Height and we get 
% sin (gamma) / RE = sin (pi-incidence) / (RE + Height) 

gamma=asin(RE.*sin(pi-thetav(:).*d2r)./(RE+Height));
a=(thetav(:).*d2r-gamma(:)); % angle at center of the Earth between nadir and point observed  

% 2) now applies law of cosines on the sphere ( https://en.wikipedia.org/wiki/Spherical_law_of_cosines ) 
D1nr=RE.*acos(cos(a(1)).*cos(a(:))+sin(a(1)).*sin(a(:)).*cos(phiv(1).*d2r-phiv(:).*d2r))

D1n=Height.*sqrt(tan(thetav(1).*d2r).^2+tan(thetav(:).*d2r).^2-2.*tan(thetav(1).*d2r).*tan(thetav(:).*d2r).*cos(phiv(1).*d2r-phiv(:).*d2r));

%Vsat=sqrt(9.81*RE^2/(H+RE)); % WARNING: need Earth rotation too... 

%v1=-Vsat.*sin(inclination.*d2r); % northward velocity on ground
%v2=Vsat.*cos(inclination.*d2r)+4E7./86400.*cos(mean(lat).*d2r); % eastward velocity on ground 
%vn=sqrt(v1.^2+v2.^2);

% WARNING: this gives positive times ... not correct for odd detectors on S2
% B04 is only before for even detectors
% should be generalized
imgtimes=D1nr./VG;


 save S2img img1 img2 img3 img4 imgtimes x y nx ny dx dy sat sun mid offspec phitrig thetav


