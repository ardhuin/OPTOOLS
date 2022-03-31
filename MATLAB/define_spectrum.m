function [Efth,freq,dir2]=define_spectrum;
% Just a quick dirty code to define the wave spectrum from buoy data ... 
% probably need to add a few more options (e.g. Elfouhaily spectrum) in the future
% F. Ardhuin, 2022/03/03

% This particular choice corresponds to the case in Kudryavtsev et al. (2017) also looked at in Ardhuin et al. (2021)
fileNC='CDIP46258_201604_spectrum.nc';
datenow=datenum(2016,04,29,18,32,52);


[lat,lon,freq,df,time,efi,th1mi,sth1mi,th2mi,sth2mi]=readWWNC_FREQ(fileNC);

I=find(time > datenow);
I(1)
ndates=I(1);
nf=size(efi,1);
%dth=360/nth;
efnow=efi(:,1,ndates);
th1now=th1mi(:,1,ndates);
sth1now=sth1mi(:,1,ndates);
th2now=th2mi(:,1,ndates);
sth2now=sth2mi(:,1,ndates);
d2r=pi/180;
%%% NARROW: 
%sth1now=sth1mi(:,1,ndates)./10;
%sth2now=sth1mi(:,1,ndates)./10;

m1=abs(1.-0.5*(sth1now(:)*d2r).^2);
m2=abs(1.-2*(sth2now(:)*d2r).^2);



a1=cos(th1now*d2r).*m1;
b1=sin(th1now*d2r).*m1;
a2=cos(2.*th2now*d2r).*m2;
b2=sin(2.*th2now*d2r).*m2;
nth=72;

% Computes directional spectrum from moments using MEM (Lygre & Krogstad, JPO 1986)

[SD,Efth] = MEM_calc(a1,a2,b1,b2,efnow,nth); % input to MEM are line vectors
Efth=SD.*repmat(efnow',nth,1)';
x1=0:(360/nth):360-360/nth; % These use nautical, from
dir=x1'.*d2r;


%
% Changes dir from nautical (0 is North, clockwise) to trigonometric convention, from
%
dirtrig=pi*1.5+dir; % DO NOT ASK ME WHY ... 
%dir=pi/2.-dir;
%
datevec(time(ndates))
%

%-------------------------------------------------------------
% 2. Plots the spectrum in polar coordinates
%-------------------------------------------------------------
% 2.1. Repeats the 1st direction at the end to 
%      wrap around nicely 
dirr=[dirtrig' dirtrig(1)]';
dir2=repmat(dirr',nf,1);
freq2=repmat(freq,1,nth+1);
Efth2=zeros(nf,nth+1);
Efth2(:,1:nth)=Efth;
Efth2(:,nth+1)=Efth(:,1);
x2=cos(dir2).*freq2;  
y2=sin(dir2).*freq2;  
%
% 2.2. plots 2D spectrum
%
figure(10);
clf;
set(gca,'FontSize',15);
pcolor(x2,y2,Efth2);axis equal;shading flat;
colorbar;
hold on;
for i=1:7 
    plot(0.1*i*cos(linspace(0,2*pi,25)),0.1*i*sin(linspace(0,2*pi,25)))
end
title('Spectrum estimated from MEM method');
%pause 

%
% 2.3. plots 1D spectrum in frequency only: E(f) 
%
figure(2);
clf
dth=2.*pi./real(nth);

Ef=sum(Efth,2)*dth;
plot(freq,Ef,'k-+','LineWidth',2);
set(gca,'FontSize',15);
xlabel('Frequency (Hz)');
ylabel('E(f) (m^2/Hz)');
%pause


% Calcul des directions moyennes ... 
dir2=repmat(dirtrig',nf,1);
a1c=sum(Efth.*cos(dir2),2).*dth./Ef;
b1c=sum(Efth.*sin(dir2),2).*dth./Ef;

m1c=sqrt(a1c.^2+b1c.^2);
sth1mc=sqrt(2*(1-m1c))*180/pi;

figure(5);
clf
th1mc=mod(90-atan2(b1c,a1c)./d2r,360); 
plot(freq,th1mc,'k-',freq,th1now,'ro','LineWidth',2);
set(gca,'FontSize',15);
xlabel('Frequency (Hz)');
ylabel('Mean direction th1m (deg)');
legend('After MEM','before MEM')

figure(6);
clf
plot(freq,sth1mc,'k-',freq,sth1mi(:,1,ndates),'ro','LineWidth',2);
set(gca,'FontSize',15);
xlabel('Frequency (Hz)');
ylabel('Spread sth1m (deg)');
legend('After MEM','before MEM')



  
