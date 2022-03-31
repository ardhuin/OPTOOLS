function [E1 E2 E3 U U2 Uall EA EB nU coh12 coh23 coh31 ang12 ang23 ang31 kxs kys angstd phases eps2s]= ...
          FFT2D3(array1,array2,array3,imgtimes,nxa,nya,dx,dy,n,Umin,Umax);

nx=floor(nxa/n);
ny=floor(nya/n);
Umid=0.5.*(Umin+Umax);

% define windows 
% 1D windows
hammingx=transpose(0.54-0.46.*cos(2*pi*linspace(0,nx-1,nx)/(nx-1)));
hanningx=transpose(0.5 * (1-cos(2*pi*linspace(0,nx-1,nx)/(nx-1))));
hanningy=transpose(0.5 * (1-cos(2*pi*linspace(0,ny-1,ny)/(ny-1))));
% 2D window
hanningxy=repmat(hanningx,1,ny).*repmat(hanningy,1,nx)';

wc2x=1/mean(hanningx.^2);                              % window correction factor
wc2y=1/mean(hanningy.^2);                              % window correction factor
dkx=2*pi/(dx*nx);   
dky=2*pi/(dy*ny);   
kx=linspace(0,(nx-1)*dkx,nx);
ky=linspace(0,(ny-1)*dky,ny);


shx=floor(nx/2);
shy=floor(ny/2);
kxs=circshift(kx',shx);
kxs(1:shx)=kxs(1:shx)-kx(nx)-dkx;
kys=circshift(ky',shy);
kys(1:shy)=kys(1:shy)-ky(ny)-dky;

E=zeros(nx,ny);



kx2=repmat(kxs,1,ny);
ky2=repmat(kys',nx,1);

nkx=length(kxs);
nky=length(kys);
kxs2=repmat(kxs',nky,1)';
kys2=repmat(kys',nkx,1);
kn=sqrt(kxs2.^2+kys2.^2);
sig=sqrt(9.81.*kn);

theta2=atan2(ky2,kx2);

E1=E;E2=2;E3=E;
F1=E;F2=E;F3=E;
U=E;U2=E;
EA=E;EB=E;
Etb=E;Unow=E;eps2now=E;
phase12=E;phase23=E;phase31=E;
coh12=E;coh23=E;coh31=E;
phases=zeros(nx,ny,n^2+(n-1)^2);
Uall=zeros(nx,ny,n^2+(n-1)^2);
eps2s=zeros(nx,ny,n^2+(n-1)^2);

nspec=0;
nU=E;
UOK=E;

r1=(wc2x*wc2y)/(dkx*dky);
r2=sqrt(r1);
options = optimset('TolX',2E-3,'Display','off','FunValCheck','off');

nx1=floor(nx/2)
ny1=floor(ny/2)

mspec=n^2+(n-1)^2;
for m=1:mspec
  if (m<=n^2)
    i1=floor((m-1)/n)+1;
    i2=m-(i1-1)*n;
%%%%%%%%%%%%%%% Start of main loop on samples %%%%%%%%%%%%%%
%
    tile1=double(array1(nx*(i1-1)+1:nx*i1,ny*(i2-1)+1:ny*i2));
    tile2=double(array2(nx*(i1-1)+1:nx*i1,ny*(i2-1)+1:ny*i2));
    tile3=double(array3(nx*(i1-1)+1:nx*i1,ny*(i2-1)+1:ny*i2));

  else
%%%%%%%%%%%%%%% now shifted 50% , like Welch %%%%%%%%%%%%%%
    i1=floor((m-n^2-1)/(n-1))+1;
    i2=m-n^2-(i1-1)*(n-1);
    tile1=double(array1(nx*(i1-1)+1+nx1:nx*i1+nx1,ny*(i2-1)+1+ny1:ny*i2+ny1));
    tile2=double(array2(nx*(i1-1)+1+nx1:nx*i1+nx1,ny*(i2-1)+1+ny1:ny*i2+ny1));
    tile3=double(array3(nx*(i1-1)+1+nx1:nx*i1+nx1,ny*(i2-1)+1+ny1:ny*i2+ny1));
  end

    z2a=tile1-mean(mean(tile1));
    zb(:,:)=(z2a).*hanningxy;
    zc1=circshift(fftn(zb)./(nx*ny),[ shx shy]);
    E1=E1+(abs(zc1).^2).*r1;

    z2a=tile2-mean(mean(tile2));
    zb(:,:)=(z2a).*hanningxy;
    zc2=circshift(fftn(zb)./(nx*ny),[ shx shy]);
    E2=E2+(abs(zc2).^2).*r1;

    z2a=tile3-mean(mean(tile3));
    zb(:,:)=(z2a).*hanningxy;
    zc3=circshift(fftn(zb)./(nx*ny),[ shx shy]);
    E3=E3+(abs(zc3).^2).*r1;
    
    Z1a=angle(zc1);
    % WARNING DOES NOT WORK IF NOT SQUARED IMAGE
    F1=F1+zc1*exp(-j.*Z1a);
    F2=F2+zc2*exp(-j.*Z1a);
    F3=F3+zc3*exp(-j.*Z1a);
    phase12=phase12+(zc2.*conj(zc1).*r1);  % these are not yet phases, these are cospectra
    phase23=phase23+(zc3.*conj(zc2).*r1);
    phase31=phase31+(zc3.*conj(zc1).*r1);
  
    np=nx.*ny;
    UOK(:)=0.;
    for jj=1:np % THIS LOOP IS TAKING A LOONG TIME
		% Half spectrum is enough. Also, could be recoded in Fortran ... 
       if (abs(zc1(jj)) > 1E-8 & kn(jj) > 1E-3 & kxs2(jj) >= 0) 
%          Ufun=@(Uc) ULS(Uc,kn(jj),sig(jj),imgtimes,[ zc1(jj) zc2(jj) zc3(jj)]);
%          Uc=fzero(Ufun,0.);
%%% second solution method: fminbnd... this is generally faster
%Ufun=@(Uc) abs(ULS(Uc,kn(jj),sig(jj),imgtimes,[ zc1(jj) zc2(jj) zc3(jj)]));
%[Uc,fval,exitflag,output] =fminbnd(Ufun,Umin-0.1,Umax+0.1,options);
%%% and for large phase differences: global minimum!
Ufun=@(Uc) abs(ULSmin(Uc,kn(jj),sig(jj),imgtimes,[ zc1(jj) zc2(jj) zc3(jj)]));
[Uc,fval,exitflag,output] =fminbnd(Ufun,Umin-0.1,Umax+0.1,options);
%          [Ufval A B eps2]=ULS(Uc,kn(jj),sig(jj),imgtimes,[ zc1(jj) zc2(jj) zc3(jj)]);
          [eps2 A B]=ULSmin(Uc,kn(jj),sig(jj),imgtimes,[ zc1(jj) zc2(jj) zc3(jj)]);
          if (abs (Uc-Umid) < abs(Umax-Umid))
             nU(jj)=nU(jj)+1.;
             UOK(jj)=1.;
             eps2now(jj)=sum(eps2)./(abs(zc1(jj)).^2+abs(zc2(jj)).^2+abs(zc3(jj)).^2);
             Unow(jj)=Uc;
             EA(jj)=EA(jj)+(abs(A).^2).*r1;
             EB(jj)=EB(jj)+(abs(B).^2).*r1;
          else
             eps2now(jj)=-1.;
          end
       end
    end
    U=U+UOK.*Unow;
    nspec=nspec+1
    Uall(:,:,nspec)=Unow;
    eps2s(:,:,nspec)=eps2now;
    phases(:,:,nspec)=zc3.*conj(zc1)./(abs(zc3)*abs(zc1));    % keeps all co-spectra
end

% rotates phases around the mean phase to be able to compute std
for m=1:nspec
phases(:,:,m)=phases(:,:,m)./phase31;
end

E1=E1./nspec;
E2=E2./nspec;
E3=E3./nspec;
U=median(Uall,3); % More accurate than the average
size(Uall)
size(U)
EA=EA./nU;
EB=EB./nU;

coh12=abs((phase12./nspec) .^2)./(E1.*E2);      % spectral coherence
coh23=abs((phase23./nspec) .^2)./(E2.*E3);      % spectral coherence
coh31=abs((phase31./nspec) .^2)./(E3.*E1);      % spectral coherence

% try to estimate current from the summed spectra: does not work !
np=nkx.*nky;
UOK(:)=0.;
for jj=1:np
  if (abs(zc1(jj)) > 1E-8 & kn(jj) > 1E-3) 
   UOK(jj)=1.;
   Ufun=@(Uc) ULS(Uc,kn(jj),sig(jj),imgtimes,[ F1(jj) F2(jj) F3(jj)]);
   U2(jj)=fzero(Ufun,0.);
   [Ufval Ac Bc eps2]=ULS(U(jj),kn(jj),sig(jj),imgtimes,[ F1(jj) F2(jj) F3(jj)]);
   end
end
ang12=angle(phase12);   % now compute the phases 
ang23=angle(phase23);
ang31=angle(phase31);

ang=angle(phase31);
angstd=std(angle(phases),0,3);
