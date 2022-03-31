function [Usum Ac Bc eps2]  = ULS(U,k,sig,imgtimes,FS)
%  ULS: least square estimate of current velocity from Fourier transform of images
%
%  CALL:  [Usum Ac Bc eps2] = ULS(U,k,sig,imgtimes,FS);
% input parameters: 
%                   - U  : value of current to be tested
%                   - k  : wavenumber
%                   - sig: radian frequency
%                   - imgtimes: vector of times of the image acquisitin
%                   - FS : vector of observed complex amplitudes
%
% output parameters: 
%                   - Usum : value of the function that goes through O for U = true current
%                   - Ac   : complex amplitude of waves propagating in direction of k
%                   - Bc   : complex amplitude of waves propagating opposite to k
%                   - eps2 : vector of squares for each image 
%
% Least square estimation of wave complex amplitudes Ac and Bc and current 
% This is an adaptation of the Mansard & Funke 3-probe method, 
% with the addition of a current. 
% WARNING: lengths of imgtimes and FS arrays should be identical, and equal to the number
%          of images used. 
%
% What we are solving for each time imgtimes(i)  is : 
%  Ac.*exp(j.*(-sig+k.*U).*imgtimes(i))+Bc.*exp(j.*(sig+k.*U).*imgtimes(i)) -F(i) = eps(i)
%  In 1D this corresponds to the Fourier transform of: 
%  A*exp(j.*(k.*x-sig+k.*U).*t)+B*exp(j.*(k.*x+sig+k.*U).*t)-F = eps
%    or in 2D: 
%  A*exp(j.*(kx.*x+ky.*y-sig+k.*U).*t)+B*exp(j.*(kx.*x+ky.*y+sig+k.*U).*t)-F = eps
%
%  
%
% F. Ardhuin. V 1.0. 20/02/2021
%
% using nt images or nt time series that give the complex fourier measurements FS(1:nt)

nt=length(imgtimes);

RA1=1;RB1=1;RF1=FS(1);RC1=1;RF2=FS(1);
for i=2:nt
  RA1=RA1+       exp(j.*(-2.*(sig-k.*U).*imgtimes(i)));
  RB1=RB1+       exp(j.*( 2.*(    k.*U).*imgtimes(i)));
  RF1=RF1+FS(i).*exp(j.*(   -(sig-k.*U).*imgtimes(i)));
  RC1=RC1+       exp(j.*( 2.*(sig+k.*U).*imgtimes(i)));
  RF2=RF2+FS(i).*exp(j.*(    (sig+k.*U).*imgtimes(i))); 
end
RAs=exp(j.*(-(sig-k.*U).*imgtimes));
RBs=exp(j.*( (sig+k.*U).*imgtimes));
Bc= (RF2-RF1.*RB1./RA1)./(RC1-RB1.^2./RA1);
Ac= (RF1 - Bc.*RB1)./RA1;

Usum=imag(sum(imgtimes(2:nt).*(RAs(2:nt).*Ac+RBs(2:nt).*Bc)...
                            .*(RAs(2:nt).*Ac+RBs(2:nt).*Bc-FS(2:nt)) ));
% Computes residual vector
eps2=sum(abs(Ac.*RAs+Bc.*RBs-FS).^2);
end 
