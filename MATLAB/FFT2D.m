function [Eta Etb ang angstd coh kxs kys phases]=FFT2D(arraya,arrayb,nxa,nya,dx,dy,n);

% Eta is PSD of 1st image (arraya) 
% Etb is PSD of 2st image (arraya) 
nx=floor(nxa/n);
ny=floor(nya/n);

hammingx=transpose(0.54-0.46.*cos(2*pi*linspace(0,nx-1,nx)/(nx-1)));
hanningx=transpose(0.5 * (1-cos(2*pi*linspace(0,nx-1,nx)/(nx-1))));
hanningy=transpose(0.5 * (1-cos(2*pi*linspace(0,ny-1,ny)/(ny-1))));

wc2x=1/mean(hanningx.^2);                              % window correction factor
wc2y=1/mean(hanningy.^2);                              % window correction factor
dkx=2*pi/(dx*nx);   
dky=2*pi/(dy*ny);   
% wavenubmers starting at zero
kx=linspace(0,(nx-1)*dkx,nx);
ky=linspace(0,(ny-1)*dky,ny);
hanningxy=repmat(hanningx,1,ny).*repmat(hanningy,1,nx)';


shx=floor(nx/2);
shy=floor(ny/2);
% Shifted wavenumbers to have zero in the middle
kxs=circshift(kx',shx);
kxs(1:shx)=kxs(1:shx)-kx(nx)-dkx;
kys=circshift(ky',shy);
kys(1:shy)=kys(1:shy)-ky(ny)-dky;

E=zeros(nx,ny);

%[xx yy] = meshgrid(x,y);
%xv=reshape(xx,nx*ny,1);
%yv=reshape(yy,nx*ny,1);
kx2=repmat(kxs,1,ny);
ky2=repmat(kys',nx,1);
theta2=atan2(ky2,kx2);
knorm=sqrt(kx2.^2+ky2.^2);
Eta=E;
Etb=E;
phase=E;
phases=zeros(nx,ny,n^2+(n-1)^2);
coh=E;

nspec=0;

nx1=floor(nx/2);
ny1=floor(ny/2);

normalization = (wc2x*wc2y)./(dkx*dky);
mspec=n^2+(n-1)^2;
for m=1:mspec
  if (m<=n^2)
    i1=floor((m-1)/n)+1;
    i2=m-(i1-1)*n;
%%%%%%%%%%%%%%% Start of main loop on samples %%%%%%%%%%%%%%
%
    array1=double(arraya(nx*(i1-1)+1:nx*i1,ny*(i2-1)+1:ny*i2));
    array2=double(arrayb(nx*(i1-1)+1:nx*i1,ny*(i2-1)+1:ny*i2));

  else
%%%%%%%%%%%%%%% now shifted 50% , like Welch %%%%%%%%%%%%%%
    i1=floor((m-n^2-1)/(n-1))+1;
    i2=m-n^2-(i1-1)*(n-1);
    array1=double(arraya(nx*(i1-1)+1+nx1:nx*i1+nx1,ny*(i2-1)+1+ny1:ny*i2+ny1));
    array2=double(arrayb(nx*(i1-1)+1+nx1:nx*i1+nx1,ny*(i2-1)+1+ny1:ny*i2+ny1));
  end

    z2a=array1-mean(mean(array1));
    zb(:,:)=(z2a).*hanningxy;
    zca=circshift(fftn(zb)./(nx*ny),[ shx shy]);
    Eta=Eta+(abs(zca).^2).*normalization;           % sum of spectra for all tiles

    z2a=array2-mean(mean(array2));
    zb(:,:)=(z2a).*hanningxy;
    zcb=circshift(fftn(zb)./(nx*ny),[ shx shy]);
    Etb=Etb+(abs(zcb).^2).*normalization;           % sum of spectra for all tiles 

    phase=phase+(zcb.*conj(zca)).*normalization;    % coherent sum of co-spectra estimates
    nspec=nspec+1;
    phases(:,:,nspec)=zcb.*conj(zca)./(abs(zca)*abs(zcb));    % keeps all co-spectra
end

 
% rotates phases around the mean phase to be able to compute std
for m=1:nspec
phases(:,:,m)=phases(:,:,m)./phase;
end



Eta=Eta./nspec;
Etb=Etb./nspec;

coh=abs((phase./nspec) .^2)./(Eta.*Etb);      % spectral coherence


ang=angle(phase);
angstd=std(angle(phases),0,3);

%E=E+(abs(zf).^2).*(wc2x*wc2y*wc2/(dkx*dky*df));



%Ekk=squeeze(sum(E(:,:,nfft/2+1:nfft)));

%Efth=zeros(nfft/2,36);

