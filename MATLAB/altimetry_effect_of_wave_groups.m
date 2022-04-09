% Uses a simple surface simulator and waveform tracker to investigate
% effects of wave groups on retrieved wave heights. 
% First created in Matlab by F. Ardhuin     2021/09/15

% In this first example we define the wave spectrum direcly on the kx,ky grid 
% that is the Fourier transform of the x,y grid we want for the sea surface

% Here are the main steps:
% 1. define wave spectrum 
% 2. generate map of sea surface elevation from spectrum
% 3. sample surface like an altimeter (assuming geometrical optics, uniform NRCS ...)
% 4. retrack waveform to find Hs (brute force distance minimization) 

clear all 
close all

% 1. Defining wave spectrum 

nx=2048;
ny=2048;
dx=10;dy=10;  % grid spacing in meters

noise = 0; %1E-3;

Tp=16 % peak period in seconds
kp=(2*pi/Tp).^2/9.81;
sx=kp*0.1; % spectral width in peak dir
sy=sx;     % spectral width in perpendicular dir

% defines grid of wavenumbers  (WARNING: maybe shift X and Y ...)
dkx=2*pi/(dx*nx);  
dky=2*pi/(dy*ny);  
[X,Y] = meshgrid(dkx*(-nx/2+1:nx/2), dky*(-ny/2+1:ny/2));

% rotation (trigonometric convention: 0 is waves to the east)
phi=0;
X1 = X*cosd(phi)+Y*sind(phi);
Y1 =-X*sind(phi)+Y*cosd(phi);



% defines the spectrum
Z1 =1/(2.*pi.*sx.*sy).* exp( - 0.5*((X1-kp).^2/sx.^2+Y1.^2/sy.^2)) + noise ;



% checks on normalization
sumZ1=4.*sqrt(sum(Z1(:).*dkx.*dky))  % variance should be 1. for Gaussian


figure(1)
pcolor(X,Y,Z1)
xlabel('k_x (rad/m')
ylabel('k_y (rad/m')
shading flat
colorbar


rg = randn(ny, nx);

% Now shifts spectrum for fft: 
shx=floor(nx/2-1);
shy=floor(ny/2-1);
zhats=circshift(sqrt(2.*Z1.*dkx.*dky).*exp(j*2.*pi.*rg),[ -shx -shy]);
ky2D=circshift(Y,[ -shx -shy]); % checks that ky2D(1,1)=0 ... 
kx2D=circshift(X,[ -shx -shy]); % checks that kx2D(1,1)=0 ... 



% 2. generate map of sea surface elevation from spectrum
S1 = real(ifft2(zhats)).*(nx.^2);


x=linspace(0,dx*(nx-1),nx);
y=linspace(0,dy*(ny-1),ny);


% Shows surface map (elevation, sigma0 ...)
figure(2)
pcolor(x,y,S1)
shading flat
colorbar
axis equal;




% rough estimate of altimeter sampling: disc of diameter 4 km.
altif=40; % frequency for waveforms
v=7000;
h=560000; %altitude of satellite
dr=0.4;   % width of range gate
npas=2*h*dr/dx^2*pi;

radi2=1200;
radi1=900;
radi=4000;
nxa=floor(radi/dx);
footprint=ones(2*nxa+1,2*nxa+1);
footprint1=footprint;

[Xa,Ya]=meshgrid(dx*(-nxa:nxa), dx*(-nxa:nxa));
II1=find(Xa.^2+Ya.^2 > radi2^2);
KK1=find(Xa.^2+Ya.^2 > radi1^2);
JJ1=find(Xa.^2+Ya.^2 < radi1^2);
II=find(Xa.^2+Ya.^2 > radi^2);
JJ=find(Xa.^2+Ya.^2 <= radi^2);

footprint1(KK1)=0.;
footprint2=footprint;
footprint2(II1)=0;
footprint2(JJ1)=0;
footprint(II)=0.;

di=floor((v/altif)/dx); % spacing between footprint centers, in pixels
nsamp=floor((nx-2*nxa)/di);
Hsalt=zeros(nsamp,1);
Xalt=Hsalt;
Hsalt2=zeros(nsamp,1);
Hsalt3=zeros(nsamp,1);
Hsalt4=zeros(nsamp,1);

% Defines theoretical waveforms 
Hsmax=20; % max value of Hs to be tested
Hsmaxt=ceil(Hsmax/dr)*dr
ne=ceil(Hsmaxt/dr)+1;
edges=linspace(0,Hsmaxt,ne);

nHs=251;
Hsm=linspace(0,25,nHs);
wfm=zeros(nHs,ne-1);
for i=1:nHs
  wfm(i,:)=0.5+erf((edges(1:end-1)+dr/2-10)./(0.25*sqrt(2).*Hsm(i)))/2;
end


% computes waveforms and performs retracking 
waveforms=zeros(nsamp,ne-1);

for i=1:nsamp
  ialt=1+nxa+(i-1)*di;
  Xalt(i)=(ialt-1)*dx;
  surf1=S1(ny/2-nxa:ny/2+nxa,ialt-nxa:ialt+nxa).*footprint1;
  surf2=S1(ny/2-nxa:ny/2+nxa,ialt-nxa:ialt+nxa).*footprint2;
  surf =S1(ny/2-nxa:ny/2+nxa,ialt-nxa:ialt+nxa).*footprint;
  Hsalt(i)=4.*std(surf1(:))./sqrt(mean(footprint1(:)));
  Hsalt2(i)=4.*std(surf2(:))./sqrt(mean(footprint2(:)));
  r=sqrt(Xa.^2+Ya.^2+(h-surf).^2)-h+10;
 

  hc=histcounts(r(JJ),edges);
  testwf=repmat(hc,nHs,1);
  dist=sum((npas.*wfm-testwf).^2,2);
  [dmin,Imin]=min(dist);
  Hsalt3(i)=Hsm(Imin);
  KK=find(edges > 10-Hsalt3(i)/2 & edges < 10+Hsalt3(i)/2);
  dist2=sum((npas.*wfm(:,KK)-testwf(:,KK)).^2,2);
  [dmin,Imin]=min(dist2);
  [p,S] = polyfit(Hsm(Imin-2:Imin+2),dist2(Imin-2:Imin+2)',2);
  Hsalt4(i)=-p(2)/(2*p(1));
  waveforms(i,:)=hc;
end
figure(3)
plot(edges(1:end-1)+dr/2,waveforms(:,:));
xlabel('range');
xlabel('normalized power');



figure(6)
plot(Xalt/1000.,Hsalt,'ko-',Xalt/1000.,Hsalt2,'rx-',Xalt/1000., ...
                Hsalt3,'b+-',Xalt/1000.,Hsalt4,'g^','LineWidth',2);
legend('4 km disc','1.8-2.4 km annulus','Retracked')
set(gca,'FontSize',16,'fontname','Helvetica','LineWidth',1)
xlabel('x (km)');
ylabel('Hs (m)');
axis([Xalt(1)/1000. Xalt(end)/1000 min(Hsalt4) max(Hsalt4)]);
grid on

figure(7)
pcolor(Xalt/1000.,edges(1:end-1)+dr/2,waveforms(:,:)'); shading flat;
colorbar
xlabel('x (km)');
ylabel('range (m)');
set(gca,'FontSize',16,'fontname','Helvetica','LineWidth',1)
axis([Xalt(1)/1000. Xalt(end)/1000 6 16]);






