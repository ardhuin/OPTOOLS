% Playing with S2 data: forward model
% Version 1.1 by F. Ardhuin, Jan. 2021
% e.g. 
% S2 CASE : 
%[img1, img2, img3, img4, phitrig, nx, ny, x, y, dx, dy  ] =   S2_simu(Efth,freq,dir2,10,801 ,1000,0.  ,0.,0.15,5,0.25,1); 
% STREAM CASE: 
%[img1, img2, img3, img4,  phitrig, nx, ny, x, y, dx, dy  ] =   S2_simu(Efth,freq,dir2, [148.1901  148.8061  149.1342  149.4561], 5 ,1601,1000,0.  ,0.,0.15,5,2   ,1); 
function [imgs, nx, ny, x, y, dx, dy ] = S2_simu(Efth,freq,dir2,U10,Udir,Ux,Uy,imgtimes, offspec,phitrig,theta, dx,nx ,fac ,facr,na,nt,plotb);
%  times : times of acquisition of images in seconds
%  dx is pixel size       (NB: dy=dx)
%  nx is number of pixels (NB: nx=ny)
%  fac is the mean image value  
%  facr is the reflection coefficient used to impose reflections: if zero, full 2D spectrum is used, if negative  the "reflected" part is set to zero
%  na: additive noise parameter
%  nt: multiplicative noise parameter
% dti: time step between images
% plotb: option for intermediate plots;

ntime=length(imgtimes);

% defines constants
d2r=pi/180;
j=sqrt(-1);


nxp=(nx-1)/8+1;
dy=dx;ny=nx;



% defines sun and sensor angles  (these should be parameters of the function in future versions) 




%-------------------------------------------------------------
% 3. Computes maps of surface elevation
%-------------------------------------------------------------

% c. defines surface size
nkx=nx*2;

% a. remaps the spectrum to nx*nx k-spectrum


dkx=2*pi/(dx*nkx);   
kmax=pi./(dx); %nyquist
kx=linspace(0,(nkx-1)*dkx,nkx)-kmax+dkx;
ky2=repmat(kx,nkx,1);
kx2=ky2';
g=9.81;

sig=2*pi.*freq;
ks=sig.^2./g;   %     this is the deep water dispersion relation 

% dks=df.*2.*pi.*C= df .*2.pi*g/k

nth=size(Efth,2);
ks2=repmat(ks,1,nth);
Ja=g./(ks.*4.*pi.*sig);
% Ja is jacobian 
Ja2=repmat(Ja,1,nth);
ktx2=-cos(dir2).*ks2;  % the - transforms "direction from" in "direction to"
kty2=-sin(dir2).*ks2;  % the - transforms "direction from" in "direction to"



F=scatteredInterpolant(double(ktx2(:)),double(kty2(:)),double(Efth(:).*Ja2(:)));
Ekk=F(kx2,ky2);
kn=sqrt(kx2.^2+ky2.^2);
Inoz=find(kn>0.1);


% Here is a short try at some isotropic wave spectrum ... 
%Ekk(:)=0;
%Ekk(Inoz)=0.1./kn(Inoz).^2;
sig2=sqrt(sqrt(kx2.^2+ky2.^2)*g);
I=find(kx2 < 0);
if (facr < 0) 
  Ekk(I)=0;  % removes waves in opposing direction ... 
end
Ekkr=flipud(fliplr(Ekk)); 
if (facr > 0) 
Ekk(I)=facr.*Ekkr(I);  % test reflection
end

figure(1)
colormap default;
set(gcf, 'Renderer', 'painters');
title('Wave spectrum interpolated in kx,ky plane');
clf
pcolor(-kx2./(2*pi)*1000,-ky2./(2*pi)*1000,10.*log10(Ekk.*1E6./(3.*pi^2)));shading flat;colorbar; % plotting convention is direction from
hold on;
for i=1:7 
    plot(10*i*cos(linspace(0,2*pi,49)),10*i*sin(linspace(0,2*pi,49)),'k-','LineWidth',1)
end
plot([-60 60],[0 0],'k-','LineWidth',2);
plot([0 0],[-60 60],'k-','LineWidth',2);
axis equal;
axis([-50 50 -50 50]);
set(gca,'Fontsize',16)
xlabel('k_x / 2 \pi (counts per km)');
caxis([20 70]);

figure(11)
clf
Efths=circshift(Efth,[0 36]);
Hspec=4.*Efth.*Efths./((Efth+Efths).^2);
set(gcf, 'Renderer', 'painters');
pcolor(ktx2./(2*pi)*1000,kty2./(2*pi)*1000,Hspec);shading flat; colorbar;
title('H spectrum');
hold on;
for i=1:7 
    plot(10*i*cos(linspace(0,2*pi,49)),10*i*sin(linspace(0,2*pi,49)),'k-','LineWidth',1)
end
plot([-60 60],[0 0],'k-','LineWidth',2);
plot([0 0],[-60 60],'k-','LineWidth',2);
axis equal;
axis([0 50 -50 50]);
set(gca,'Fontsize',16)
xlabel('k_x / 2 \pi (counts per km)');
caxis([0 1]);


% b. random draw of phases ... 
phases=rand(nkx,nkx)*2*pi;
%save phases phases
%load phases

zhat=sqrt(Ekk.*dkx.^2).*exp(j.*phases); %+fliplr(flipud(sqrt(Ekk.*dkx.^2./2))).*exp(-j.*phases);

shx=floor(nkx/2-1);
kxs=circshift(kx',-shx);
I=find(kxs < 0);
kxsc=kxs;
kxsc(I)=kxs(I)+2*pi/dx;

zhats=circshift(zhat,[ -shx -shx]);
kx2s=circshift(kx2,[ -shx -shx]);
ky2s=circshift(ky2,[ -shx -shx]);
si2s=circshift(sig2,[ -shx -shx]);
kns=sqrt(kx2s.^2+ky2s.^2);

xp=linspace(0,dx*nxp,nxp);
y2=repmat(xp',1,nxp);
x2=repmat(xp,nxp,1);
yp=xp;

colormap(gray);

nfig=30;
figure(nfig);
clf
set(nfig,'Position',[1 1 3*nxp+240 3*nxp+200])
M=struct([]);
mov = VideoWriter('simuS2.avi');
open(mov);
mov2 = VideoWriter('simuS2_B.avi');
open(mov2);



allB=zeros(nx,nx,ntime);
allz=zeros(nx,nx,ntime);
imgs=zeros(ntime,nx,nx);

mssx=0.001+0.00316.*U10;
mssy=0.003+0.00185.*U10;
mss=mssx+mssy;




figure(101);
clf

x=linspace(0,dx*(nx-1),nx);
y=linspace(0,dy*(nx-1),nx);

allind=(nx*2)*10+10+linspace(1,(nx)^2,(nx)^2);
for ii=1:nx-1
   allind(1+ii*nx:end)=allind(1+ii*nx:end)+nx;
end
choppy=0;
%%%%%%%%%% Loop on time %%%%%%%%%%%%%%%%%ù
for ii=1:ntime

% adjust geometry of view
  phip=phitrig(ii)*d2r % azimuth of specular slope
  beta=offspec(ii)*d2r
  thetav=theta(ii)*d2r;
  sx0=-tan(beta).*sin(phip);  % slope without long wave effect 
  sy0=-tan(beta).*cos(phip);



  t1=imgtimes(ii);
  % computes z and sx, sy for full fft window
  phasor=exp(-j.*(si2s-kx2s.*Ux-ky2s.*Uy).*t1);
  zeta1=real(ifft2(zhats.*phasor)).*(nkx.^2);
  if choppy==1 % adds a horizontal displacement similar to choppy model by Nouguier et al. (JGR 2009).
    Dx=-real(ifft2(zhats.*j.*kx2s.*phasor./kns)).*(nkx.^2)./dx; % normalized displacement
    Dy=-real(ifft2(zhats.*j.*ky2s.*phasor./kns)).*(nkx.^2)./dx;
    iDx=floor(Dx);
    iDy=floor(Dy);
    wDx=Dx-iDx;
    wDy=Dy-iDy;
    zetachoppy=(1-wDx(allind)).*( zeta1(allind+  iDx(allind)   *(nx*2)+iDy(allind)  ).*(1-wDy(allind)) ...
                                 +zeta1(allind+  iDx(allind)   *(nx*2)+iDy(allind)+1).*   wDy(allind)  ) ...
                 +wDx(allind) .*( zeta1(allind+ (iDx(allind)+1)*(nx*2)+iDy(allind)  ).*(1-wDy(allind)) ...
                                 +zeta1(allind+ (iDx(allind)+1)*(nx*2)+iDy(allind)+1).*   wDy(allind)  );


  % keeps the part of interest (region 1:nx, 1:nx) 
    zeta=reshape(zetachoppy,nx,nx); % zeta1(1:nx,1:nx);
  else
   zeta=zeta1(1:nx,1:nx);
  end
  sx1=real(ifft2(zhats.*j.*kx2s.*phasor)).*(nkx.^2);
  sy1=real(ifft2(zhats.*j.*ky2s.*phasor)).*(nkx.^2);
  sx=sx1(1:nx,1:nx);
  sy=sy1(1:nx,1:nx);

% Adds slope of bistatic look direction + rotate in wind direction
  sxt=(sx+sx0).*cos(Udir)+(sy+sy0).*sin(Udir);
  syt=(sy+sy0).*cos(Udir)-(sx+sx0).*sin(Udir);
  sx0t=(sx0).*cos(Udir)+(sy0).*sin(Udir);
  sy0t=(sy0).*cos(Udir)-(sx0).*sin(Udir);

  norma=(2.*pi.*sqrt(mssx*mssy));
  B=exp(-0.5.*(sxt.^2/mssx+syt.^2/mssy))./(cos(beta).^4.*cos(thetav)); %/(2.*pi.*sqrt(mssx*mssy));
  B0=exp(-0.5.*(sx0t.^2/mssx+sy0t.^2/mssy)); 

  allz(:,:,ii)=zeta;
  allB(:,:,ii)=B./B0;
  allB(:,:,ii)=zeta;
  
  % something to check on: 
   if ii==1 
   mean(std(B))./mean(mean(B))
   end

  if plotb == 1  & ii == 1
figure(101) 
hold on
plot(x,zeta(:,1),'k-',x,zeta1(11:nx+10,11),'r-');

  figure(nfig);
  set(nfig,'Position',[1 1 3*nxp+300 3*nxp+200])
  imagesc(xp(1:nxp),yp(1:nxp),fliplr(zeta(1:nxp,1:nxp))');
  caxis([-1,1]);
  %axis equal;
  axis([0 1000 0 1000]);
  set(nfig,'Position',[1 1 3*nxp+300 3*nxp+200])
  colorbar;
  xlabel('x (m)');
  ylabel('y (m)');
  F=getframe(gcf);
  writeVideo(mov,F);
  end

  if plotb == 1 % & ii == 1
  % Plots B
  figure(nfig+2);
  clf
  colormap(gray);
  imagesc(xp(1:nxp),yp(1:nxp),fliplr(B(1:nxp,1:nxp)./B0)');
 
%  imagesc(xp(1:nxp),yp(1:nxp),fac.*fliplr(squeeze(allB(:,:,ii))'));
  caxis([0. 1.5]);
  axis equal;
  axis([0 1000 0 1000]);
  colorbar;
  xlabel('x (m)');
  ylabel('y (m)');
  %pause;
  set(nfig+2,'Position',[1000 1000 3*nxp+300 3*nxp+200])
  F=getframe(gcf);
  writeVideo(mov2,F);
  end
% Comment this out to remove the MTF
%allB(:,:,:)=1+0.1.*allz; 
  thisimg=round((nt.*rand(nx,nx)+1-nt./2).*fac.*(squeeze(allB(:,:,ii)).*(1-na./2)+na.*rand(nx,nx)));
  imgs(ii,:,:)=thisimg;
end




close(mov);close(mov2);

