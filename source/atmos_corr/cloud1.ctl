dset ^./src/grid_CM/agg_cloud_data/cloud_2015210_1406.dat
options template
title soil moisture
undef -9999. 
xdef 3750 linear -15.0 .004
ydef 3750 linear 30.0 .004
zdef 1 levels 1 1
tdef 1 linear 0z01jan2002 1dy
vars 1
soil 0 0 soil 
endvars

