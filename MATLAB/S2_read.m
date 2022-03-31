function [imgs,S2xml,DSxml,S2txt,nx,ny,dx,dy,x,y,boxi]=read_S2(file,boxi,bands);
% Reads pieces from 3 or 4 images using data from 3 or 4 different bands ... 
nb=length(bands);

%
% Now reads ...
% the flip command is there to re-order the matrix as a plot: 
% first dimension is x, increasing to the left, second is y, increasing up

bi1=sprintf('%02.f',bands(1));
bi2=sprintf('%02.f',bands(2));
bi4=sprintf('%02.f',bands(3));
I1=fliplr(imread(['DATA/' file '_B' bi1 '.jp2'])'); %'PixelRegion',{[1 100],[4 500]}
I2=fliplr(imread(['DATA/' file '_B' bi2 '.jp2'])'); 
I4=fliplr(imread(['DATA/' file '_B' bi4 '.jp2'])'); 
I3=I4;

if nb > 3
   bi4=sprintf('%02.f',bands(4));
   I3=fliplr(imread(['DATA/' file '_B' bi3 '.jp2'])'); 
end

    filexml=['DATA/' file '.xml'];
    filexml2=['DATA/' file '_DS.xml'];
    filetxt=['DATA/' file '.txt'];

    S2xml = parseXML(filexml);
    DSxml = xmlread(filexml2);
    S2txt = load(filetxt);

if bands(1)==4
  dx=10;
  dy=10;
  img1=I1(boxi(1):boxi(2),boxi(3):boxi(4));
  img2=I2(boxi(1):boxi(2),boxi(3):boxi(4));
  img3=I3(boxi(1):boxi(2),boxi(3):boxi(4));
  img4=I4(boxi(1):boxi(2),boxi(3):boxi(4));
else
  dx=20;dy=20;
  mem=boxi;
  boxi=boxi/2;
  img1=I1(boxi(1):boxi(2),boxi(3):boxi(4));
  img2=I2(boxi(1):boxi(2),boxi(3):boxi(4));


%boxi=mem;
%  img1=0.25.*(   I1(boxi(1)  :2:boxi(2),  boxi(3):2:boxi(4))+I1(boxi(1)  :2:boxi(2),  boxi(3)-1:2:boxi(4)-1) ...
%                +I1(boxi(1)-1:2:boxi(2)-1,boxi(3):2:boxi(4))+I1(boxi(1)-1:2:boxi(2)-1,boxi(3)-1:2:boxi(4)-1));
%  img2=0.25.*(   I2(boxi(1)  :2:boxi(2),  boxi(3):2:boxi(4))+I2(boxi(1)  :2:boxi(2),  boxi(3)-1:2:boxi(4)-1) ...
%                +I2(boxi(1)-1:2:boxi(2)-1,boxi(3):2:boxi(4))+I2(boxi(1)-1:2:boxi(2)-1,boxi(3)-1:2:boxi(4)-1));


  if bands(end)==2
    boxi=mem;
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
    
    img4=1.*(0.9*img4a+0.1*img4b); %+0.1*(0.9.*img4c+0.1.*img4d);
    boxi=boxi/2;
 end
  if length(bands)==3
    img3=img4;
  end
end
 

[nx ny]=size(img1);
x=linspace(0,nx-1,nx).*dx; 
y=linspace(0,ny-1,ny).*dy;


clear I1 I2 I3 I4;
imgs=zeros(nb,nx,ny);
for ib=1:nb
    eval(['imgs('  num2str(ib) ',:,:)=img' num2str(ib) '(:,:);']);
end
