% Very simple test of least-square estimation from 3 1D images at 3 different times
% using exactly periodic signal (no windowing , leaking ...) 
L=50;g=9.81;
k=1*pi./L;
sig=sqrt(g*k);
T=2*pi/sig;
C0=sqrt(g/k);

nx=100;
dx=10;
x=linspace(0,(nx-1)*dx,nx);

t1=0;
t2=0.5;
t3=1.0;
imgtimes=[t1 t2 t3];

%%% Warning: this phase unwrapping should be corrected to take into account current ... 
philinp= mod(C0.*k.*t3+pi,2*pi)-pi;   % linear zero-current phase with phase wrapping
philinm= mod(-C0.*k.*t3+pi,2*pi)-pi;
sig=sqrt(g.*k);

U=0.2;
Aa=1;                 % complex amplitude of rightward propagating waves
Ba=0.05;               % comples amplitude of leftward propagating waves
noise_amplitude=0.05;


NMC=5000;             % number of Monte Carlo realisations
%NMC=50000;            better convergence
MMC=1;
Um1=zeros(MMC,1);
Us1=zeros(MMC,1);
Us3=zeros(MMC,1);
Um2=zeros(MMC,1);
Um3=zeros(MMC,1);
Um4=zeros(MMC,1);

for jMC=1:MMC
jMC
Ucall=zeros(NMC,1);
Aall=zeros(NMC,1);
Ball=zeros(NMC,1);
Upall=zeros(NMC,1);
indOK=zeros(NMC,1);
eps2all=zeros(NMC,1);
EA=0;
EB=0;



F3sum=0;
F2sum=0;
F1sum=0;
options = optimset('TolX',2E-4,'Display','off','FunValCheck','off');
% loop for Monte Carlo noise simulation
Nok=0;
for iMC=1:NMC


phia=rand(1).*2*pi;
phib=rand(1).*2*pi;
A=Aa.*exp(j.*phia);
B=Ba.*exp(j.*phib);
na=noise_amplitude.*(rand(3,nx)+j.*rand(3,nx)); % Noise
nt=noise_amplitude.*(rand(3,nx)); % Noise

f1=real(A*exp(j.*(k.*x-(sig-k.*U).*t1))+B*exp(j.*(k.*x+(sig+k.*U).*t1))).*(1-0.5.*noise_amplitude+nt(1,:));
f2=real(A*exp(j.*(k.*x-(sig-k.*U).*t2))+B*exp(j.*(k.*x+(sig+k.*U).*t2))).*(1-0.5.*noise_amplitude+nt(2,:));
f3=real(A*exp(j.*(k.*x-(sig-k.*U).*t3))+B*exp(j.*(k.*x+(sig+k.*U).*t3))).*(1-0.5.*noise_amplitude+nt(3,:));

ifig=0;
if ifig > 0 | iMC == 1
figure(1)
title('example of 3 measured 1D images')
clf
plot(x,f1,'r-',x,f2,'g-',x,f3,'b-','LineWidth',2);
xlabel(' x (m)');
ylabel(' pixel value');
legend('time = 0','time= t2','time= t3');
set(gca,'FontSize',16)
grid on;
end


% Takes FFT of the "measurements" times 2 to simplify notations
f1hat=fft(f1)./nx; 
[fmax,Imax]=max(abs(f1hat));
F1=2.*f1hat(Imax);
f2hat=fft(f2)./nx;
F2=2.*f2hat(Imax);
f3hat=fft(f3)./nx;
F3=2.*f3hat(Imax);

%coh=abs((phase./nspec) .^2)./(Eta.*Etb);  

%%% First solution method: fzero matlab function 
%Ufun=@(Uc) Usum(Uc,k,sig,imgtimes,[F1 F2 F3]);
%Uc=fzero(Ufun,0.);

%%% second solution method: fminbnd... this is generally faster
Ufun=@(Uc) abs(ULS(Uc,k,sig,imgtimes,[F1 F2 F3]));
[Uc,fval,exitflag,output] =fminbnd(Ufun,-5.1,5.1,options);

[U2 Ac Bc eps2]=ULS(Uc,k,sig,imgtimes,[F1 F2 F3]);

  if (abs (Uc) < 5)   % test when using fzero
     Nok=Nok+1;
     eps2all(iMC)=sum(eps2);
     EA=EA+(abs(Ac).^2);
     EB=EB+(abs(Bc).^2);
     Aall(iMC)=Ac;
     Ball(iMC)=Bc;
     indOK(iMC)=1;
     Ucall(iMC)=Uc;
  end
  F1a=angle(F1);
  F1sum=F1sum+F1*exp(-j.*F1a);
  F2sum=F2sum+F2*exp(-j.*F1a);
  F3sum=F3sum+F3*exp(-j.*F1a);
  Upall(iMC)=(angle(F3.*conj(F1))-philinm)./k./t3;
end % end of loop for Monte Carlo

% Now also computes a current from the sum spectrum (does not work with real images) ... 
Ufun=@(Uc) ULS(Uc,k,sig,imgtimes,[F1sum F2sum F3sum]);
Um2(jMC)=fzero(Ufun,0.);
[Ufval Ac Bc eps2]=ULS(Um2(jMC),k,sig,imgtimes,[F1sum F2sum F3sum]);

J=find(indOK==1);
EM=mean(eps2all(J));
Um1(jMC)=median(Ucall(J));
Us1(jMC)=std(Ucall(J));
UW=-((C0+U)*A-B*(C0-U)-C0*(A+B))/(A+B);
Um3(jMC)=mean(Upall);
Us3(jMC)=std(Upall);
Um4(jMC)=(angle(F3sum.*conj(F1sum))-philinm)./k./t3
H=4.*EA.*EB/(EA+EB)^2
H2=4.*abs(Ac)^2.*abs(Bc)^2/(abs(Ac)^2+abs(Bc)^2).^2
Hin=4.*abs(A)^2.*abs(B)^2/(abs(A)^2+abs(B)^2).^2
end 



figure(2)
clf
set(gcf, 'Renderer', 'painters');
set(gca,'FontSize',16,'fontname','Helvetica','LineWidth',1)
dU=0.02;
edges = [-5:dU:5];
histogram(Ucall(J),edges);
title('Distribution of current from least squares');
sigULS=std(Ucall(J));
gp=NMC.*exp(-0.5.*(edges-U).^2./sigULS.^2)./(sigULS.*sqrt(2*pi)).*dU;
sig2=0.2;
a=0.15;
gp2=NMC.*a./((edges-U).^2+a.^2)./(pi).*dU;
gp3=NMC.*exp(-0.5.*(edges-U).^2./sig2.^2)./(sig2.*sqrt(2*pi)).*dU;
hold on;
plot(edges,gp,'k--',edges,gp2,'b--','LineWidth',2);
legend('data','Gaussian','Cauchy');
xlabel('U (m/s)');
ylabel('count');
stdU_LS=std(Ucall)
stdU_phi2=std(Upall)
axis([-1.2 1.6 0 1200]);
%axis([-0.4 0.8 0 600]);

figure(21)
clf
set(gcf, 'Renderer', 'painters');
set(gca,'FontSize',16,'fontname','Helvetica','LineWidth',1)
dU=0.005;
edges = [0.8:dU:1.2];
histogram(abs(Aall(J)),edges);
xlabel('A (m)');
ylabel('count');
axis([0.8 1.2 0 10000]);
sigA=std(abs(Aall(J)));
gp=NMC.*exp(-0.5.*(edges-mean(abs(Aall(J)))).^2./sigA.^2)./(sigA.*sqrt(2*pi)).*dU;
hold on;
plot(edges,gp,'k--','LineWidth',2);
legend('data','Gaussian');

figure(3)
clf
set(gcf, 'Renderer', 'painters');
set(gca,'FontSize',16,'fontname','Helvetica','LineWidth',1)
dU=0.02;
edges = [0:dU:0.4];
histogram(Um2,edges);
title('Distribution of current from mean of least squares');
sigULS=std(Um1);
gp=MMC.*exp(-0.5.*(edges-U).^2./sigULS.^2)./(sigULS.*sqrt(2*pi)).*dU;

hold on;
plot(edges,gp,'k--','LineWidth',2);
legend('data','Gaussian');
xlabel('U (m/s)');
ylabel('count');
stdU_LS=std(Ucall)
stdU_phi2=std(Upall)
%axis([-0.4 0.8 0 600]);


figure(6)
Y=abs(Ucall-U);
X=sqrt(eps2all);
  obs_rms=sum(X.^2);
   obs_mean=mean(X);
   obs_scat=sqrt(sum((X-obs_mean).^2));
   mod_mean=mean(Y);
   mod_scat=sqrt(sum((Y-mod_mean).^2));
   nrmse=sqrt(sum((X-Y).^2)./obs_rms);
   corr=sum((X-obs_mean).*(Y-mod_mean))./(obs_scat*mod_scat);
set(gcf, 'Renderer', 'painters');
set(gca,'FontSize',16,'fontname','Helvetica','LineWidth',1)
plot(X,Y,'k+');
xlabel('rms of  residual');
ylabel('error on U');
std(Ucall);
std(Upall);


figure(4)
title('Distribution of current from phase difference');
set(gcf, 'Renderer', 'painters');
set(gca,'FontSize',16,'fontname','Helvetica','LineWidth',1)
edges = [-1.4:0.02:1.8];
axis([-1.2 1.6 0 1200]);
histogram(Upall(J),edges);
xlabel('U (m/s)');
ylabel('count');

     

