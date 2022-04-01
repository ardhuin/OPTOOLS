function [B]=tony_omni_L(k,u10,omega)

km=363;
g=9.81;

c=sqrt(g./k.*(1+(k/km).^2));
cp=u10/omega;
cm=sqrt(2*g/km);


k0=g./u10.^2;
kp=k0.*omega.^2;

%Peak enhancement according to Jonswap
sigma=0.08.*(1+4./omega.^3);
gamma=1.7;
if (omega>1)
   gamma=1.7+6*log10(omega);
end   
LPM=exp(-(5/4)*(kp./k).^2);

Jp=gamma.^(exp(-(sqrt(k./kp)-1).^2/2/sigma/sigma));


%Curvature spectrum of Long waves
alpha_p=6.0e-3*sqrt(omega);

Fp=LPM.*Jp.*exp(-omega/sqrt(10)*(sqrt(k./kp)-1));

Bl=0.5*alpha_p*cp./c.*Fp;


B=(Bl);



return


