switch choice
    case 1
    file='T11SMS_20200520T181919';
    boxi=[7000 7800 4000 4800 ]; % La Jolla including coast.
    boxi=[6900 7300 4600 5000 ];
    case 2
    file='T11SMS_20160419T184432';
    boxi=[7000 7800 4000 4800 ]; % La Jolla including coast.
    boxi=[5500 6000 300 800 ];
    case 3 % this is the image in Kudry & al. 2017 figs 2-9 
    file='T11SMS_20160429T183252';
    boxi=[7000 7800 4000 4800 ]; % La Jolla including coast.
    boxi=[5600 6400  3600 4400]; % Box for paper on opposing waves
    case 4
    file='T31TFH_20190619T103031';
    boxi=[3000 3800  1000 1800]; % Calanques
    boxi=[9000 9800  200 1000];
    boxi=[3000 3800  9000 9800]; % sud embouchure du rhone
    case 5 % Mississipi 
    file='T16RBT_20200711T162901';
    boxi=[2000 2800 2000 2800]; % 
    case 6 % Etoile campaing
    file='T30TWP_20170716T110649';
    boxi=[7000 7800 8400 9200]; % 
    case 7 % Sud Bretagne low red signal, whitecaps ... 
    file='T30TVT_20210101T111451';
    boxi=[2200 3000 7200 8000]; % 
    case 8 % Sud Bretagne a little more green in the plume 
    file='T30TVT_20210101T111451';
    boxi=[1100 1500 7900 8300]; % 
    case 9 % Waimea
    file='T04QEK_20190320T211911';
    boxi=[6200 7000 100 900]; % 
    boxi=[5000 5800 5000 5200]; % 
    boxi=[5000 5800 5000 5200]; % 
    boxi=[4600 5400 200 1000 ]; %  should be no reflection here 
    case 10 % Waimera with stronger glitter
    file='T04QEK_20200523T211921';
    boxi=[8200 9000 400 1200]; %  right in front of Waimea, water depth > 200 m , box #2
    %boxi=[5000 5800 5000 5200]; % 
    %boxi=[5000 5800 5000 5200]; % 
    %boxi=[4600 5400 100 900 ]; %  NW Oahu should be no reflection here 
    case 11 % Waimea with stronger glitter
    file='T04QEJ_20200523T211921';
    boxi=[7600 8000 8100 8500]; %  SW of Oahu 
    boxi=[4400 5200 4700 5500]; %  SW of Oahu 
    case 12 % Waimea with stronger glitter
    file='T04QFK_20200523T211921';
    boxi=[1200 1600 800 1200]; %  NE of Oahu 
    case 13 % S. Georgia
    file='T25FDU_20210203T115651';
    boxi=[4500 4800 7800 8100]; %  S Georgia iceberg1 
    case 14 % Arcic
    file='T38XNM_20190323T110721';
    boxi=[1300 1700 9700 10100]; %  waves in ice
    boxi=[2000 2400 8700 9100]; %  waves in ice
    case 15 % Arcic
    file='T39XVH_20190323T110721';
    boxi=[7900 8300 5400 5800]; %  waves in ice
    %boxi=[9400 9800 8600 9000]; %  waves in ice?
    case 16
    file='T37XCE_20210312T110841';
    boxi=[10200       10600        9800       10200]; %  waves in ice
    boxi=[8400       8800        9800       10200]; %  waves in ice
    case 16
    file='T37XCF_20210312T110841';
    boxi=[10200       10600        9800       10200]; %  waves in ice
    boxi=[8400       8800        9800       10200]; %  waves in ice
    case 17 
    file='T33XXE_20210312T110841';
    boxi=[5100       5500       1400     1800]; %  waves in ice
    case 18
    file='T33XXD_20210312T110841';
    boxi=[5600        6200        8600        9200]; %
    boxi=[5400        6600        8400        9600]; %
    boxi=[6300        6600        8500        8800]; %
    end
