function pmofk=pmofk(k,T0,H)
   w0=2*pi/T0;
   g=9.81;
   w=sqrt(g.*k.*tanh(k.*H));
   Cg=(0.5+k.*H/sinh(2.*k.*H)).*w./k;
   pmofk=0.008.*g.^2.*exp(-0.74.*(w./w0).^(-4))./(w.^5).*Cg+5.9;
  %pmofk=0.008.*g.^2.*exp(-0.74.*(w./w0).^(-4))./(w.^5).*Cg ...
  %      + 0.0004.*g.^2.*exp(-0.74.*(w0./w0).^(-4))./(w0.^5).*Cg ;
   
