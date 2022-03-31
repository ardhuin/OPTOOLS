% performs a series of simulations to test convergence
[Efth,freq,dir2]=define_spectrum;

U10=6;
Udir=40.*d2r; % direction to, trig. convention 
Ux=0;
Uy=0;


% these are the real angles from S2 image
phitrig =[  148.1901  148.8061  149.4561 ];
offspec=[8.9740    9.0674    9.1693    ];
theta=[6.2804    6.2413    6.2114    ];
%phitrig(:)=phitrig(2);
%offspec(:)=offspec(2);
%theta(:)=theta(2);

imgtimes=[0 0.5 1 ];

nMC=100;
Cimgall=zeros(nMC,50,50);
Cstdall=zeros(nMC,50,50);
phaseall=zeros(nMC,50,50);
% loop on realizations 
for iMC = 1:nMC
   iMC
   % simulation with no noise and no waves in opposite directions 
   [imgs,  nx, ny, x, y, dx, dy  ] =      S2_simu(Efth,freq,dir2,U10,Udir,Ux,Uy,imgtimes,offspec,phitrig,theta,   10,801 ,1000,-1.,0.,0.0,1);

   S2_analysis  % standard 2-image phase difference method

   Cimgall(iMC,:,:)=Cimg0; 
   Cstdall(iMC,:,:)=Cstd; 
   phaseall(iMC,:,:)=phase_diff;
end

Cimgm=squeeze(mean(Cimgall,1));
Cstdm=squeeze(mean(Cstdall,1));
phasem=squeeze(mean(phaseall,1));
Cimg=Cimgm(J); 
Cimg2=Cimgm(J2)

figure(413)
clf
hold on
set(gcf, 'Renderer', 'painters');
e1=errorbar(kn(J2)./(2*pi)*1000,Cimg2,Cstdm(J2)./NSX./sqrt(nMC),'ro','LineWidth',2);
e2=errorbar(kn(J)./(2*pi)*1000,Cimg,Cstdm(J)./NSX./sqrt(nMC),'bo','LineWidth',2);
axis([8 50 -0.2 0.2]); set(gca,'FontSize',14,'XTick',linspace(0,50,6)) 
set(gcf, 'Renderer', 'painters');   
xlabel('k / 2 \pi (km^{-1})');
ylabel('Current (m/s)');
legend('U from S2 phase','std(phase) < 60°)');
%legend('theory','dt=1s','dt=4s','dt=10s')
grid on


%%%%%%%%%%%%%%%%%%%%%%%% Plots phase speeds for chosen directions (J2) 
phase_o = dt.*sqrt(9.81.*kn); % phase from linear theory (deep water)
phase_diff = phasem;
% phase_diff should be equal to kx.Ux + ky.Uy ... 
k_min_fit = 10; k_max_fit =40; std_max=60; %in cpk
id_fit = find( angstd./d2r<std_max & kncpk >= k_min_fit & kncpk <= k_max_fit  );  % & phase>0
figure(6)
hold on
plot(kxs2(id_fit).*1000./(2.*pi),kys2(id_fit).*1000./(2.*pi),'ko')
weight= nMC./(angstd(id_fit).^2);
K_x_fit = kxs2(id_fit); K_y_fit = kys2(id_fit); Y = phase_diff(id_fit)./dt;
    B = [K_x_fit(:) K_y_fit(:) ones(size(K_x_fit(:)))] \ Y;  
A=[K_x_fit(:) K_y_fit(:) ];
C=diag(weight);
X =  Y'/ A';
Y2=A*X';
D=A'*C*A;
sigU=sqrt(abs(inv(D)))

save MC_no_ref_move_cam Cimgall Cstdall phaseall Cimgm Cstdm phasem
