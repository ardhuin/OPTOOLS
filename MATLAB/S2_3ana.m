%
%  Script for analyzing phase speed and presence of reflection in sequence of 
%  3 images
%

col=load('doppler_modified_land_2.ct');
col(:,254)=col(:,253);
col(:,255)=col(:,253);
col(:,256)=col(:,253);
d2r=pi/180;
dt=imgtimes(end)-imgtimes(1);
img1=squeeze(imgs(1,:,:));
img2=squeeze(imgs(2,:,:));
img3=squeeze(imgs(3,:,:));
redChannel = fliplr(img1)';
greenChannel = fliplr(img2)';
blueChannel = fliplr(img3)';
% Recombine separate color channels into an RGB image.
rgbImage = cat(3, redChannel, greenChannel, blueChannel);

col=load('doppler_modified_land_2.ct');
col(:,254)=col(:,253);
col(:,255)=col(:,253);
col(:,256)=col(:,253);

figure(1)
%clf
%imshow(rgbImage(1:201,1:201,:)./30);




figure(2)
pcolor(x./1000,y./1000,img1');shading flat;colorbar;
colormap(gray)
s1=std(double(img1(:)));
axis equal
axis([min(x) max(x) min(y) max(y)]/1000.);
caxis([mean(img1(:))-2*s1 mean(img1(:))+2.*s1]);
set(gca,'FontSize',18)  
xlabel('x (km)');
ylabel('y (km)');

rm=mean(img1(:));
gm=mean(img2(:));
bm=mean(img3(:));
rs=std(double(img1(:)-rm));
gs=std(double(img2(:)-gm));
bs=std(double(img3(:)-bm));

img1a=(img1-rm)./rs;
img2a=(img2-gm)./gs;
img3a=(img3-bm)./bs;
check=std(double(img4a(:)))

NSX=round(16*10/dx);  % the larger, the slower ... 
NSX=16;
%NSX=6; % number of tiles in each dimension. giving 2*NSX^2 degrees of freedom

Umin=-5;Umax=5; % number of tiles in each dimension. giving 2*NSX^2 degrees of freedom
[E1 E2 E3 U U2 Uall EA EB nU coh12 coh23 coh31  ang12 ang23 ang31 kxs kys angstd phases eps2s] = ...
          FFT2D3(img1,img2,img3,[imgtimes(1) imgtimes(2) imgtimes(3)],nx,ny,dx,dy,NSX,Umin,Umax);
%[E1 E2 E3 U U2 Uall EA EB nU coh12 coh23 coh31  ang12 ang23 ang31 kxs kys angstd phases eps2s] = ...
%          FFT2D3U(img1,img2,img4,[imgtimes(1) imgtimes(2) imgtimes(4)],nx,ny,dx,dy,NSX,-1.3 ,0.);
%[E1 E2 E3 U U2 Uall EA EB nU coh12 coh23 coh31  ang12 ang23 ang31 kxs kys angstd phases eps2s] = ...
%          FFT2D3U(img1,img2,img4,[imgtimes(1) imgtimes(2) imgtimes(4)],nx,ny,dx,dy,NSX,[-1 -1.8 -1.],[0. 0. 0.]);

dirC1=108;dirC2=118;
%dirC1=45;dirC2=55;
%dirC1=110;dirC2=120;
curmax=1.5; % for plots

nkx=length(kxs);
nky=length(kys);

kxs2=repmat(kxs',nky,1)';
kys2=repmat(kys',nkx,1);
kn=sqrt(kxs2.^2+kys2.^2);
kncpk=kn./(2*pi)*1000;
dir2=atan2(kxs2,kys2)./d2r;
% For plotting: uses k./2*pi*1000: counts per km
kxp=kxs./(2.*pi).*1000;
kyp=kys./(2.*pi).*1000;
knp=kn./(2.*pi).*1000;
kN=round(kxp(end)+kxp(end)-kxp(end-1)); % Nyquist: = 50 for Sentinel 2 bands B02 ... 

%%%%%%%%%%%%%%%%% Sanity check: values of directions as used in the direction selection 
figure(1)
clf;
set(gcf, 'Renderer', 'painters');
set(gca,'FontSize',16)  
imagesc(kxp,kyp,(dir2)');shading flat; colorbar;
set(gca,'YDir','normal')
caxis([-180 180]);
colormap(jet)
axis equal;
axis([-kN kN -kN kN])
xlabel('k_x (cycles per km)');
ylabel('k_y (cycles per km)');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Image PSD
figure(3)
clf
colormap('default');
imagesc(kxp,kyp,10.*log10(E1'));shading flat; colorbar;
set(gca,'FontSize',16)  
set(gca,'YDir','normal')
axis equal;
axis([0 kN -kN kN])
xlabel('k_x (cycles per km)');
ylabel('k_y (cycles per km)');
title('PSD of 1st band');
phiblind=phitrig(1)+90.;
phiblind3=phitrig(3)+90.;
hold on
plot([-2.*kN.*sin(phiblind.*d2r) 2.*kN.*sin(phiblind.*d2r)],[-2.*kN.*cos(phiblind.*d2r) 2.*kN.*cos(phiblind.*d2r)],'k--','LineWidth',2);
caxis([24 64]);
hold on
plot([0 kN.*sin(dirC1.*d2r)],[0 kN.*cos(dirC1.*d2r)],'k-','LineWidth',2);
plot([0 kN.*sin(dirC2.*d2r)],[0 kN.*cos(dirC2.*d2r)],'k-','LineWidth',2);

%%%%%%%%%%%%%%%%%% plot phase img 3 - phase img 1
figure(4)
clf;
set(gcf, 'Renderer', 'painters');
dkx=0; %(kxs(2)-kxs(1))/(4*pi);
imagesc(kxp,kyp,ang31'./d2r);shading flat; colorbar;
set(gca,'YDir','normal')
caxis([-180 180]);
colormap(jet)
axis equal;
axis([-kN kN -kN kN])
set(gca,'FontSize',16)  
title('Phase of co-spectrum');
xlabel('k_x (cycles per km)');
ylabel('k_y (cycles per km)');
hold on
plot([-2.*kN.*sin(phiblind.*d2r) 2.*kN.*sin(phiblind.*d2r)],[-2.*kN.*cos(phiblind.*d2r) 100.*cos(phiblind.*d2r)],'k--','LineWidth',2);
plot([-2.*kN.*sin(phiblind3.*d2r) 2.*kN.*sin(phiblind3.*d2r)],[-2.*kN.*cos(phiblind3.*d2r) 2.*kN.*cos(phiblind3.*d2r)],'w--','LineWidth',2);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% std of phase , related to coherence
figure(5)
clf;
set(gcf, 'Renderer', 'painters');
set(gca,'FontSize',16)  
title('std of phase across FFT tiles');
dkx=0; %(kxs(2)-kxs(1))/(4*pi);
imagesc(kxp,kyp,angstd'./d2r);shading flat; colorbar;
set(gca,'YDir','normal')
caxis([-180 180]);
colormap(jet)
axis equal;
axis([-0.025 0.025 -0.025 0.025])
axis([0 50 -50 50])
xlabel('k_x / 2 \pi (m^{-1})');
ylabel('k_y / 2 \pi (m^{-1})');

%%%%%%%%%%%%% Plots coherence (this is symmetric, half plane is enough)
figure(6)
clf;
set(gcf, 'Renderer', 'painters');
set(gca,'FontSize',16)  

imagesc(kxp,kyp,sqrt(1-coh31'));shading flat; colorbar;
title('(1-Coh)^{0.5}');
set(gca,'YDir','normal')
caxis([0 1]);
axis equal;
axis([0 kN -kN kN])
xlabel('k_x (cycles per km)');
ylabel('k_y (cycles per km)');

hold on
plot([0 kN.*sin(dirC1.*d2r)],[0 kN.*cos(dirC1.*d2r)],'k-','LineWidth',2);
plot([0 kN.*sin(dirC2.*d2r)],[0 kN.*cos(dirC2.*d2r)],'k-','LineWidth',2);
plot([-2.*kN.*sin(phiblind.*d2r) 2.*kN.*sin(phiblind.*d2r)],[-2.*kN.*cos(phiblind.*d2r) 2.*kN.*cos(phiblind.*d2r)],'k--','LineWidth',2);



%%%%%%%%% std of normalized residuals
figure(8)
clf;
set(gcf, 'Renderer', 'painters');
set(gca,'FontSize',16)  
dkx=0; %(kxs(2)-kxs(1))/(4*pi);
colormap('default');
Je=find(eps2s > 0);
ebin=eps2s.*0+1;
ebin(Je)=0;
ebin2=eps2s.*0;
ebin2(Je)=eps2s(Je);
%ep=median(eps2s,3);
ep=sqrt(median(ebin+ebin2,3));

imagesc(kxp,kyp,ep');shading flat; colorbar;
set(gca,'YDir','normal')


axis equal;
set(gca,'FontSize',16)  
axis([ 0 kN -kN kN])

caxis([0 0.5]);

hold on
plot([0 kN.*sin(dirC1.*d2r)],[0 kN.*cos(dirC1.*d2r)],'k-','LineWidth',2);
plot([0 kN.*sin(dirC2.*d2r)],[0 kN.*cos(dirC2.*d2r)],'k-','LineWidth',2);
plot([-2.*kN.*sin(phiblind.*d2r) 2.*kN.*sin(phiblind.*d2r)],[-2.*kN.*cos(phiblind.*d2r) 2.*kN.*cos(phiblind.*d2r)],'k--','LineWidth',2);

title('LS residuals');
xlabel('k_x (cycles per km)');
ylabel('k_y (cycles per km)');


%%%%%%%%%%%%% prepares for phase unrolling (for large time lags)
Njump=kxs2.*0;
f2=sqrt(9.81.*kn)./(2.*pi);

for ij=1:6
 I=find(f2.*dt > double(ij)+0.5);
  Njump(I)=Njump(I)+1;
end
figure(43)
clf;
set(gcf, 'Renderer', 'painters');
set(gca,'FontSize',16)  
imagesc(kxp,kyp,Njump');shading flat; colorbar;
axis([-kN kN -kN kN])
axis equal;


J=find(  dir2 < dirC2 & dir2 > dirC1 & angstd./d2r < 60 ) ; 
J2=find(  dir2 < dirC2 & dir2 > dirC1 ); 
J3=find(dir2 < dirC2 & dir2 > dirC1 & ep < 0.3   );
J4=find(ep > 0.3  ); 
J5=find( coh31 < 0.64); 
J6=find(  dir2 < dirC2 & dir2 > dirC1 & angstd./d2r < 60 & knp > 30 & knp < 40 ) ; 




%%%%%%%%% Now current speed 
figure(7)
clf;
set(gcf, 'Renderer', 'painters');
set(gca,'FontSize',16)  
dkx=0; %(kxs(2)-kxs(1))/(4*pi);
colormap(col'./255);

Uguess=-1;
Vguess=0;
Clin=sqrt(9.81./kn);
philinp= mod(Clin.*kn.*dt+Uguess.*kxs2.*dt+pi,2*pi)-pi;
philinm= mod(-Clin.*kn.*dt+Uguess.*kxs2.*dt+pi,2*pi)-pi;
Cimg0=(ang31-philinm)./kn./dt;
Cimg2=(ang31-philinp)./kn./dt;
signchoice=find(abs(Cimg2) < abs(Cimg0));
Cimg0(signchoice)=Cimg2(signchoice);
Cimgp=Cimg0;
Cimgp(J5)=NaN;

Cstd=angstd./kn./dt;
CUR=(Cimgp-Clin.*sign(ang31) )';
JJ=find(isnan(CUR)==0);
mean(CUR(JJ))
imagesc(kxs./(2.*pi).*1000,kys./(2.*pi).*1000,Cimgp');shading flat; colorbar;
set(gca,'YDir','normal')
caxis([-curmax curmax]);
axis equal;
set(gca,'FontSize',16)  
axis([ 0 kN -kN kN])
hold on
plot([0 kN.*sin(dirC1.*d2r)],[0 kN.*cos(dirC1.*d2r)],'k-','LineWidth',2);
plot([0 kN.*sin(dirC2.*d2r)],[0 kN.*cos(dirC2.*d2r)],'k-','LineWidth',2);
plot([-2.*kN.*sin(phiblind.*d2r) 2.*kN.*sin(phiblind.*d2r)],[-2.*kN.*cos(phiblind.*d2r) 2.*kN.*cos(phiblind.*d2r)],'k--','LineWidth',2);

title('Current from phase shift');
xlabel('k_x (cycles per km)');
ylabel('k_y (cycles per km)');



%%%%%%%%% Now current speed from Least Squares
figure(71)
clf;
set(gcf, 'Renderer', 'painters');
set(gca,'FontSize',16)  
dkx=0; %(kxs(2)-kxs(1))/(4*pi);
colormap(col'./255);
Up=U;
Up(J4)=NaN;
imagesc(kxp,kyp,Up');shading flat; colorbar;
set(gca,'YDir','normal')
caxis([-curmax curmax]);
axis equal;
set(gca,'FontSize',16)  
axis([ 0 kN -kN kN])
hold on
plot([0 kN.*sin(dirC1.*d2r)],[0 kN.*cos(dirC1.*d2r)],'k-','LineWidth',2);
plot([0 kN.*sin(dirC2.*d2r)],[0 kN.*cos(dirC2.*d2r)],'k-','LineWidth',2);
plot([-2.*kN.*sin(phiblind.*d2r) 2.*kN.*sin(phiblind.*d2r)],[-2.*kN.*cos(phiblind.*d2r) 2.*kN.*cos(phiblind.*d2r)],'k--','LineWidth',2);

title('Current from 3 img. LS');
xlabel('k_x (cycles per km)');
ylabel('k_y (cycles per km)');


%%%%%%%%% opposition spectrum
figure(9)
clf;
set(gcf, 'Renderer', 'painters');
set(gca,'FontSize',16)  
opposition=4.*(EA.*EB)./((EA+EB).^2);
imagesc(kxp,kyp,opposition');
hold on;
colorbar;
set(gca,'YDir','normal')
title('H(k,\phi)=4 (E(k,\phi) x E(k,\phi+\pi))/((E(k,\phi) + E(k,\phi+\pi))^2');
caxis([0 1]);
axis equal;
axis([0 kN -kN kN])
xlabel('k_x (cycles per km)');
ylabel('k_y (cycles per km)');
plot([0 kN.*sin(dirC1.*d2r)],[0 kN.*cos(dirC1.*d2r)],'k-','LineWidth',2);
plot([0 kN.*sin(dirC2.*d2r)],[0 kN.*cos(dirC2.*d2r)],'k-','LineWidth',2);
plot([-2.*kN.*sin(phiblind.*d2r) 2.*kN.*sin(phiblind.*d2r)],[-2.*kN.*cos(phiblind.*d2r) 2.*kN.*cos(phiblind.*d2r)],'k--','LineWidth',2);


% Linear dispersion : theoretical phase speed with no current
dispt=sqrt(9.81./kn(J));
dispt2=sqrt(9.81./kn(J2));
% Measured phase speed (if the phase did not jump over pi ...) 
Cimg=(ang31(J))./kn(J)./dt;
Cimg2=(ang31(J2))./kn(J2)./dt;





%%%%%%%%%%%%%%%%%%%%%%%% Plots phase speeds for chosen directions (J2) 
figure(12)
clf
hold on
set(gcf, 'Renderer', 'painters');
e1=errorbar(kn(J2)./(2*pi)*1000,abs(Cimg2),Cstd(J2)./NSX,'ro','LineWidth',2);
e2=errorbar(kn(J)./(2*pi)*1000,abs(Cimg),Cstd(J)./NSX,'bo','LineWidth',2);
%plot(kn(J)./(2*pi)*1000,abs(Cimg),'o','Color',[1 0.7 0.7]);
%plot(kn(J)./(2*pi)*1000,abs(Cimg),'o','Color',[0.4 1 0.4]);
%plot(kn(J)./(2*pi)*1000,abs(Cimg),'o','Color',[0.5 0.5 1]);
hold on

plot(kn(J2)./(2*pi)*1000,dispt2,'kx','LineWidth',3)
%axis([50 100 3.5 5.5]);set(gca,'FontSize',14,'XTick',linspace(0,100,11)) ;
%axis([2 5 0 25])
axis([8 kN 4 14]); set(gca,'FontSize',14,'XTick',linspace(0,50,6)) 
set(gcf, 'Renderer', 'painters');   
xlabel('k / 2 \pi (km^{-1})');
ylabel('Phase speed (m/s)');
legend('C from S2 phase','std(phase) < 60°)','Theory, zero current');
%legend('theory','dt=1s','dt=4s','dt=10s')
grid on


%%%%%%%%%%%%%%%%%%%%%%%% Plots current velocity for chosen directions (J2) 
Cimg=Cimg0(J); 
Cimg2=Cimg0(J2);
Ustd=0.*Cimg0;
Ufstd=0.*Cimg0;
Uf=0.*Cimg0+NaN;
taille=size(Uall);
for ii=1:nkx
  for jj=1:nky;
     UU=Uall(ii,jj,:);
     ee=eps2s(ii,jj,:);
     JJ=find(ee > -0.5 );
     II=find(ee < 0.04 & ee >=0);
     %Ustd(ii,jj)=std(UU(JJ))./sqrt(length(JJ)./2);
     %Ustd(ii,jj)=100.*(20./knp(ii,jj)).*ep(ii,jj)./sqrt(length(JJ)./2);
     Ustd(ii,jj)=(Clin(ii,jj)./dt).^2.*ep(ii,jj)./sqrt(length(JJ)./2);
     %Ufstd(ii,jj)=std(UU(II))./sqrt(length(II)./2);
     %Ufstd(ii,jj)=100.*(20./knp(ii,jj)).*sqrt(median(ee(II)))./sqrt(length(II)./2);
     Ufstd(ii,jj)=(Clin(ii,jj)./dt).^2.*sqrt(median(ee(II)))./sqrt(length(II)./2);
     Uf(ii,jj)=median(UU(II));
  end
end

figure(13)
clf
hold on
set(gcf, 'Renderer', 'painters');
e1=errorbar(knp(J2),Cimg2+Uguess.*sin(dir2(J2).*d2r),Cstd(J2)./NSX,'ro','LineWidth',2);
e2=errorbar(knp(J),Cimg+Uguess.*sin(dir2(J).*d2r),Cstd(J)./NSX,'bo','LineWidth',2);
%e3=errorbar(knp(J3),U(J3),Ustd(J3),'ko','LineWidth',2);
%e3=errorbar(knp(J3),Uf(J3),Ufstd(J3),'go','LineWidth',2);
e3=plot(knp(J3),U(J3),'ko','LineWidth',2);
e3=plot(knp(J3),Uf(J3),'go','LineWidth',2);
Ump=mean(Uguess.*sin(dir2(J).*d2r));
plot([0 kN],[Ump Ump],'-','Color',[0.5 0.5 0.5],'LineWidth',2)
axis([8 kN -1.5 1.5]); set(gca,'FontSize',14,'XTick',linspace(0,50,6)) 
set(gcf, 'Renderer', 'painters');   
xlabel('k / 2 \pi (km^{-1})');
ylabel('Current (m/s)');
legend('U :S2 phase','std(phase) < 60°)','U : 3-LS','U : 3-LS, \epsilon_r < 0.2','input','Interpreter','latex');
%legend('theory','dt=1s','dt=4s','dt=10s')
grid on


i1=35;j1=22;
%i1=22;j1=10;
 figure(14)
ep1=squeeze(eps2s(i1,j1,:));
II=find(ep1 >=0 ); 
%plot(sqrt(ep1(II)),abs(squeeze(Uall(i1,j1,II))+1),'k+')
plot(sqrt(ep1(II)),abs(squeeze(Uall(i1,j1,II))-Ump),'k+')
std(Uall(i1,j1,II))./sqrt(length(II))

figure(15)
edges = [Umin:0.02:Umax];
histogram(Uall(i1,j1,II),edges);


