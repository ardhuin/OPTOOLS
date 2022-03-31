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

NSX=16; % number of tiles in each dimension. giving 2*NSX^2 degrees of freedom

% Computes FFT, co-spectra, etc ... 
[Eta Etb phase angstd coh kxs kys phases]=FFT2D(img1,img3,nx,ny,dx,dy,NSX);

dirC1=45;dirC2=55;
%dirC1=10;dirC2=20;
%dirC1=130;dirC2=150; %swell Oahu
%dirC1=140;dirC2=150;
dirC1=110;dirC2=120;


nkx=length(kxs);
nky=length(kys);

 kxs2=repmat(kxs',nky,1)';
 kys2=repmat(kys',nkx,1);
kn=sqrt(kxs2.^2+kys2.^2);
kncpk=kn./(2*pi)*1000;
dir2a=atan2(kxs2,kys2)./d2r;
% For plotting: uses k./2*pi*1000: counts per km
kxp=kxs./(2.*pi).*1000;
kyp=kys./(2.*pi).*1000;
kN=round(kxp(end)+kxp(end)-kxp(end-1)); % Nyquist: = 50 for Sentinel 2 bands B02 ... 


%%%%%%%%%%%%%%%%% Sanity check: values of directions as used in the direction selection 
figure(1)
clf;
set(gcf, 'Renderer', 'painters');
set(gca,'FontSize',16)  
imagesc(kxp,kyp,(dir2a)');shading flat; colorbar;
set(gca,'YDir','normal')
caxis([-180 180]);
colormap(jet)
axis equal;
axis([-kN kN -kN kN])
xlabel('k_x (cycles per km)');
ylabel('k_y (cycles per km)');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Image PSD %%%%%%%%%%%%%%%%%%%%%
figure(3)  % plots spectrum
clf
colormap('default');
imagesc(kxp,kyp,10.*log10(Eta'));shading flat; colorbar;
set(gca,'YDir','normal')
axis equal;
axis([0 50 -50 50])
xlabel('k_x (cycles per km)');
ylabel('k_y (cycles per km)');
title('PSD of 1st band');
phiblind=phitrig(1)+90.;
phiblind3=phitrig(3)+90.;
hold on
plot([-2.*kN.*sin(phiblind.*d2r) 2.*kN.*sin(phiblind.*d2r)],[-2.*kN.*cos(phiblind.*d2r) 100.*cos(phiblind.*d2r)],'k--','LineWidth',2);
caxis([24 64]);
hold on
plot([0 kN.*sin(dirC1.*d2r)],[0 kN.*cos(dirC1.*d2r)],'k-','LineWidth',2);
plot([0 kN.*sin(dirC2.*d2r)],[0 kN.*cos(dirC2.*d2r)],'k-','LineWidth',2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% cospectrum phase
figure(4) 
clf;
set(gcf, 'Renderer', 'painters');
dkx=0; %(kxs(2)-kxs(1))/(4*pi);
imagesc(1000.*kxs./(2.*pi),1000.*kys./(2.*pi),phase'./d2r);shading flat; colorbar;
set(gca,'YDir','normal')
caxis([-180 180]);
colormap(jet)
axis equal;
axis([-50 50 -50 50])
set(gca,'FontSize',16)  
title('Phase of co-spectrum');
xlabel('k_x (cycles per km)');
ylabel('k_y (cycles per km)');
hold on
plot([-100.*sin(phiblind.*d2r) 100.*sin(phiblind.*d2r)],[-100.*cos(phiblind.*d2r) 100.*cos(phiblind.*d2r)],'k--','LineWidth',2);
plot([-100.*sin(phiblind3.*d2r) 100.*sin(phiblind3.*d2r)],[-100.*cos(phiblind3.*d2r) 100.*cos(phiblind3.*d2r)],'w--','LineWidth',2);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% std of phase , related to coherence
figure(56)
clf;
set(gcf, 'Renderer', 'painters');
set(gca,'FontSize',16)  
title('std of phase across FFT tiles');
imagesc(kxs./(2.*pi),kys./(2.*pi),angstd'./d2r);shading flat; colorbar;
set(gca,'YDir','normal')
caxis([0 110]);
colormap(jet)
axis equal;
axis([-0.025 0.025 -0.025 0.025])
axis([0 0.05 -0.05 0.05])
xlabel('k_x / 2 \pi (m^{-1})');
ylabel('k_y / 2 \pi (m^{-1})');

%%%%%%%%%%%%% Plots coherence (this is symmetric, half plane is enough)
figure(6)
clf;
set(gcf, 'Renderer', 'painters');
set(gca,'FontSize',16)  
imagesc(1000.*kxs./(2.*pi),1000.*kys./(2.*pi),coh');shading flat; colorbar;
title('Coherence');
set(gca,'YDir','normal')
caxis([0 1]);
axis equal;
axis([0 50 -50 50])
xlabel('k_x (cycles per km)');
ylabel('k_y (cycles per km)');

hold on
plot([0 kN.*sin(dirC1.*d2r)],[0 kN.*cos(dirC1.*d2r)],'k-','LineWidth',2);
plot([0 kN.*sin(dirC2.*d2r)],[0 kN.*cos(dirC2.*d2r)],'k-','LineWidth',2);
plot([-2.*kN.*sin(phiblind.*d2r) 2.*kN.*sin(phiblind.*d2r)],[-2.*kN.*cos(phiblind.*d2r) 2.*kN.*cos(phiblind.*d2r)],'k--','LineWidth',2);


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
imagesc(kxs./(2.*pi).*1000,kys./(2.*pi).*1000,Njump');shading flat; colorbar;
axis([-50 50 -50 50])
axis equal;




I=find(coh < 0.64);
J=find( coh > 0.01 & dir2a < dirC2 & dir2a > dirC1 & angstd./d2r < 60) ; 
% & kn > 2.*(2.*pi)./1000 & kn < 5.*(2.*pi)./1000);
J2=find( coh > 0.01 & dir2a < dirC2 & dir2a > dirC1 ); 
%& kn > 2.*(2.*pi)./1000 & kn < 5.*(2.*pi)./1000);
%J3=find( angstd./d2r > 60); % | dir2a < 0); % | dir2a > 90);
J3=find( coh < 0.64); % | dir2a < 0); % | dir2a > 90);
%J2=find(coh > 0.2);


%%%%%%%%% Now current speed 
figure(7)
clf;
set(gcf, 'Renderer', 'painters');
dkx=0; %(kxs(2)-kxs(1))/(4*pi);
colormap(col'./255);

Clin=sqrt(9.81./kn);
philinp= mod(Clin.*kn.*dt+pi,2*pi)-pi;
philinm= mod(-Clin.*kn.*dt+pi,2*pi)-pi;
Cimg0=(phase-philinm)./kn./dt;
Cimg2=(phase-philinp)./kn./dt;
signchoice=find(abs(Cimg2) < abs(Cimg0));
Cimg0(signchoice)=Cimg2(signchoice);
Cimgp=Cimg0;
Cimgp(J3)=NaN;

Cstd=angstd./kn./dt;
CUR=(Cimgp-Clin.*sign(phase) )';
JJ=find(isnan(CUR)==0);
mean(CUR(JJ))
imagesc(kxs./(2.*pi).*1000,kys./(2.*pi).*1000,Cimgp');shading flat; colorbar;
set(gca,'YDir','normal')
caxis([-3 3]);
axis equal;
set(gca,'FontSize',16)  
axis([ 0 50 -50 50])
hold on
plot([0 50.*sin(dirC1.*d2r)],[0 50.*cos(dirC1.*d2r)],'k-','LineWidth',2);
plot([0 50.*sin(dirC2.*d2r)],[0 50.*cos(dirC2.*d2r)],'k-','LineWidth',2);
plot([-100.*sin(phiblind.*d2r) 100.*sin(phiblind.*d2r)],[-100.*cos(phiblind.*d2r) 100.*cos(phiblind.*d2r)],'k--','LineWidth',2);

title('Current velocity in direction phi');
xlabel('k_x (cycles per km)');
ylabel('k_y (cycles per km)');




figure(40)
clf;
set(gcf, 'Renderer', 'painters');
set(gca,'FontSize',16)  
dkx=0; %(kxs(2)-kxs(1))/(4*pi);
imagesc(1000.*kxs./(2.*pi),1000.*kys./(2.*pi),angstd'./d2r);shading flat; colorbar;
set(gca,'YDir','normal')
caxis([0 110]);
axis equal;
axis([0 50 -50 50 ])
hold on
plot([0 50.*sin(dirC1.*d2r)],[0 50.*cos(dirC1.*d2r)],'k-','LineWidth',2);
plot([0 50.*sin(dirC2.*d2r)],[0 50.*cos(dirC2.*d2r)],'k-','LineWidth',2);
plot([-100.*sin(phiblind.*d2r) 100.*sin(phiblind.*d2r)],[-100.*cos(phiblind.*d2r) 100.*cos(phiblind.*d2r)],'k--','LineWidth',2);
xlabel('k_x (cycles per km)');
ylabel('k_y (cycles per km)');



% Linear dispersion : theoretical phase speed with no current
dispt=sqrt(9.81./kn(J));
dispt2=sqrt(9.81./kn(J2));
% Measured phase speed (if the phase did not jump over pi ...) 
Cimg=(phase(J))./kn(J)./dt;
Cimg2=(phase(J2))./kn(J2)./dt;

%%%%%%%%%%%%%%%%%%%%%%%% Plots phase speeds for chosen directions (J2) 
figure(22)
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
axis([8 50 4 14]); set(gca,'FontSize',14,'XTick',linspace(0,50,6)) 
set(gcf, 'Renderer', 'painters');   
xlabel('k / 2 \pi (km^{-1})');
ylabel('Phase speed (m/s)');
legend('C from S2 phase','std(phase) < 60°)','Theory, zero current');
%legend('theory','dt=1s','dt=4s','dt=10s')
grid on


%%%%%%%%%%%%%%%%%%%%%%%% Plots current velocity for chosen directions (J2) 
Cimg=Cimg0(J); 
Cimg2=Cimg0(J2);
figure(13)
clf
hold on
set(gcf, 'Renderer', 'painters');
e1=errorbar(kn(J2)./(2*pi)*1000,Cimg2,Cstd(J2)./NSX,'ro','LineWidth',2);
e2=errorbar(kn(J)./(2*pi)*1000,Cimg,Cstd(J)./NSX,'bo','LineWidth',2);
axis([8 50 -1.2 1.2]); set(gca,'FontSize',14,'XTick',linspace(0,50,6)) 
set(gcf, 'Renderer', 'painters');   
xlabel('k / 2 \pi (km^{-1})');
ylabel('Current (m/s)');
legend('U from S2 phase','std(phase) < 60°)');
%legend('theory','dt=1s','dt=4s','dt=10s')
grid on



%%%%%%%%%%%%%%%%%%%%%%%% Plots phase speeds for chosen directions (J2) 
phase_o = dt.*sqrt(9.81.*kn); % phase from linear theory (deep water)
phase_diff = phase-phase_o;
% phase_diff should be equal to kx.Ux + ky.Uy ... 
k_min_fit = 10; k_max_fit =40; std_max=60; %in cpk
id_fit = find( angstd./d2r<std_max & kncpk >= k_min_fit & kncpk <= k_max_fit  );  % & phase>0
figure(6)
hold on
plot(kxs2(id_fit).*1000./(2.*pi),kys2(id_fit).*1000./(2.*pi),'ko')
weight= 1./(angstd(id_fit).^2);
K_x_fit = kxs2(id_fit); K_y_fit = kys2(id_fit); Y = phase_diff(id_fit)./dt;
    B = [K_x_fit(:) K_y_fit(:) ones(size(K_x_fit(:)))] \ Y;  
A=[K_x_fit(:) K_y_fit(:) ];
C=diag(weight);
X =  Y'/ A';
Y2=A*X';
D=A'*C*A;
sigU=sqrt(abs(inv(D)))

np=length(Y); 

nj=100;
B2j=zeros(nj,2);
inds=1+round(rand(nj,1)*np-1);
for jk=1:nj
   II=[[1:1:inds(jk)-1] [inds(jk)+1:1:np]];
   B2j(jk,:) =  Y(II)' / [K_x_fit(II) K_y_fit(II)]';
end
std(B2j(:,1))*sqrt(np-1)
std(B2j(:,2))*sqrt(np-1)

KU_plane = reshape([kxs2(:), kys2(:), ones(size(kxs2(:)))] * B, numel(kxs), []);


