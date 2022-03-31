function [imgs,S2xml,DSxml,S2txt,nx,ny,dx,dy,x,y,boxi]=read_S2(S2file,boxi,bands);
% Reads pieces from 3 or 4 images using data from 3 or 4 different bands ... 
nb=length(bands);

for ib=1:nb
   bis=sprintf('%02.f',bands(ib));
   filename=[S2file '_B' bis '.jp2'];
   info = imfinfo(filename)
   nxf=info.Width;
   nyf=info.Width;
   if ib==1
     nxf1=nxf;
     nyf1=nyf;
     if bands(1)==4 | bands(1) == 3 | bands(1) == 2
       dx=10;
       dy=10;
       boxi2=boxi;  % boxi2 are the indices for output imgs
     else
      dx=20;dy=20;
      mem=boxi;
      boxi2=boxi/2;
     end
     nx=boxi2(2)-boxi2(1)+1;
     ny=boxi2(4)-boxi2(3)+1;
     imgs=zeros(nb,nx,ny);
   end
%
% Now reads ...
% the fliplr and transpose commands  re-order the matrix as a plot: 
% first dimension is x, increasing to the left, second is y, increasing up
   

   if (nxf < nxf1)  % this happens if a 20 m channel is read after a 10 m one.
      w=fliplr(imread(filename, ...
       'PixelRegion',{[nyf-boxi2(4)+1 nyf-boxi2(3)+1],[boxi2(1) boxi2(2)]}))'; 
   elseif (nxf > nxf1) %  if a 10 m channel is read after a 20 m one.
      I4=fliplr(imread(filename, ...
       'PixelRegion',{[nyf-boxi(4)+1 nyf-boxi(3)+1],[boxi(1) boxi(2)]}))'; 

    % First test: "box averaging"
    %img4=0.25.*( I4(boxi(1)  :2:boxi(2),  boxi(3):2:boxi(4))+I4(boxi(1)  :2:boxi(2),  boxi(3)+1:2:boxi(4)+1) ...
    %            +I4(boxi(1)+1:2:boxi(2)+1,boxi(3):2:boxi(4))+I4(boxi(1)+1:2:boxi(2)+1,boxi(3)+1:2:boxi(4)+1));
    % 2nd test: centered
    img4=0.125.*(4.*I4(boxi(1)  :2:boxi(2),  boxi(3):2:boxi(4))+I4(boxi(1)  :2:boxi(2),  boxi(3)+1:2:boxi(4)+1) ...
                   +I4(boxi(1)+1:2:boxi(2)+1,boxi(3):2:boxi(4))+I4(boxi(1)  :2:boxi(2),  boxi(3)-1:2:boxi(4)-1) ...
                   +I4(boxi(1)-1:2:boxi(2)-1,boxi(3):2:boxi(4))   );
    % 3nd test: box-1
    %img4=0.25.*( I4(boxi(1)  :2:boxi(2),  boxi(3):2:boxi(4))+I4(boxi(1)  :2:boxi(2),  boxi(3)-1:2:boxi(4)-1) ...
    %            +I4(boxi(1)+1:2:boxi(2)+1,boxi(3):2:boxi(4))+I4(boxi(1)-1:2:boxi(2)-1,boxi(3)-1:2:boxi(4)-1));
    % 4th test: box-1,+1
    img4a=0.25.*( I4(boxi(1)  :2:boxi(2),  boxi(3):2:boxi(4))+I4(boxi(1)  :2:boxi(2),  boxi(3)-1:2:boxi(4)-1) ...
                 +I4(boxi(1)-1:2:boxi(2)-1,boxi(3):2:boxi(4))+I4(boxi(1)-1:2:boxi(2)-1,boxi(3)-1:2:boxi(4)-1));
    img4b=0.25.*( I4(boxi(1)+1:2:boxi(2)+1,boxi(3):2:boxi(4))+I4(boxi(1)+1:2:boxi(2)+1,boxi(3)-1:2:boxi(4)-1) ...
                 +I4(boxi(1)  :2:boxi(2)  ,boxi(3):2:boxi(4))+I4(boxi(1)  :2:boxi(2)  ,boxi(3)-1:2:boxi(4)-1));
    
    img4c=0.25.*( I4(boxi(1)  :2:boxi(2),  boxi(3)-1:2:boxi(4)-1)+I4(boxi(1)  :2:boxi(2),  boxi(3)  :2:boxi(4)  ) ...
                 +I4(boxi(1)-1:2:boxi(2)-1,boxi(3)-1:2:boxi(4)-1)+I4(boxi(1)-1:2:boxi(2)-1,boxi(3)  :2:boxi(4)  ));
    img4d=0.25.*( I4(boxi(1)+1:2:boxi(2)+1,boxi(3)-1:2:boxi(4)-1)+I4(boxi(1)+1:2:boxi(2)+1,boxi(3)  :2:boxi(4)  ) ...
                 +I4(boxi(1)  :2:boxi(2)  ,boxi(3)-1:2:boxi(4)-1)+I4(boxi(1)  :2:boxi(2)  ,boxi(3)  :2:boxi(4)  ));
    
    w=1.*(0.9*img4a+0.1*img4b); %+0.1*(0.9.*img4c+0.1.*img4d);

   else
      w=fliplr(imread(filename, ...
       'PixelRegion',{[nyf-boxi(4)+1 nyf-boxi(3)+1],[boxi(1) boxi(2)]}))'; 
   end   

   imgs(ib,:,:)=w;

end


    filexml=[file '.xml'];
    filexml2=[file '_DS.xml'];
    filetxt=[file '.txt'];

    S2xml = parseXML(filexml);
    DSxml = xmlread(filexml2);
    S2txt = load(filetxt);

x=linspace(0,nx-1,nx).*dx; 
y=linspace(0,ny-1,ny).*dy;


