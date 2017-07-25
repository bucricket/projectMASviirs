if ( dtime(i,j).le.1.8499988.and.modis_dtrad(i,j).le.7.7208467) then
 latn(i,j) = (-2.00403+(0.333*modis_dtrad(i,j))+(0.011*day(i,j))+(0.0009*terrain(i,j))+(-0.2*fc(i,j)))
endif
if ( dtime(i,j).le.1.6687537.and.modis_dtrad(i,j).gt.7.7208467) then
 latn(i,j) = (8.0725188+(0.206*modis_dtrad(i,j))+(-0.025*day(i,j))+(2.1*fc(i,j))+(0.0002*dtime(i,j)))
endif
if ( day(i,j).le.283.74112.and.dtime(i,j).le.2.3499992.and.dtime(i,j).gt.1.6687537) then
 latn(i,j) = (-7.8942329+(0.199*modis_dtrad(i,j))+(0.042*day(i,j))+(-1.5*fc(i,j))+(0.0553*dtime(i,j)))
endif
if ( modis_dtrad(i,j).le.7.7208467.and.dtime(i,j).gt.1.8499988) then
 latn(i,j) = (-1.1500314+(0.646*modis_dtrad(i,j))+(-0.012*day(i,j))+(0.0053*terrain(i,j))+(1.5*fc(i,j))+(1.8224*dtime(i,j)))
endif
if ( dtime(i,j).le.2.3499992.and.day(i,j).gt.283.74112.and.modis_dtrad(i,j).gt.7.7208467.and.dtime(i,j).gt.1.6687537) then
 latn(i,j) = (27.0092857+(0.263*modis_dtrad(i,j))+(-0.111*day(i,j))+(-0.0043*terrain(i,j))+(4.4648*dtime(i,j)))
endif
if ( terrain(i,j).le.1.8872331.and.dtime(i,j).gt.3.1208341.and.day(i,j).le.303.73444.and.dtime(i,j).le.3.625001) then
 latn(i,j) = (-16.3071504+(0.635*modis_dtrad(i,j))+(0.004*day(i,j))+(0.6216*terrain(i,j))+(1.6*fc(i,j))+(4.4836*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.1437504.and.dtime(i,j).le.3.216666.and.day(i,j).le.303.73444.and.modis_dtrad(i,j).gt.7.7208467) then
 latn(i,j) = (-123.9988942+(0.218*modis_dtrad(i,j))+(-0.054*day(i,j))+(0.0162*terrain(i,j))+(2*fc(i,j))+(45.4454*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.1208341.and.dtime(i,j).le.3.1437504.and.day(i,j).le.303.73444) then
 latn(i,j) = (160.1543705+(0.421*modis_dtrad(i,j))+(-0.113*day(i,j))+(-0.0018*terrain(i,j))+(-39.451*dtime(i,j)))
endif
if ( terrain(i,j).le.3.8671637.and.dtime(i,j).le.2.6520841.and.dtime(i,j).gt.2.3499992.and.day(i,j).le.303.73444.and.modis_dtrad(i,j).gt.7.7208467) then
 latn(i,j) = (-23.7383371+(0.46*modis_dtrad(i,j))+(0.289*terrain(i,j))+(-0.2*fc(i,j))+(10.306*dtime(i,j)))
endif
if ( dtime(i,j).gt.2.6520841.and.dtime(i,j).le.3.1208341.and.day(i,j).le.298.08633.and.terrain(i,j).le.46.655537.and.modis_dtrad(i,j).gt.7.7208467) then
 latn(i,j) = (-28.1587162+(0.259*modis_dtrad(i,j))+(0.083*day(i,j))+(0.0148*terrain(i,j))+(0.7*fc(i,j))+(3.4107*dtime(i,j)))
endif
if ( dtime(i,j).le.3.1208341.and.day(i,j).le.303.73444.and.dtime(i,j).gt.2.6520841.and.day(i,j).gt.298.08633.and.terrain(i,j).le.46.655537.and.modis_dtrad(i,j).gt.7.7208467) then
 latn(i,j) = (70.4540089+(0.453*modis_dtrad(i,j))+(-0.235*day(i,j))+(0.0315*terrain(i,j))+(-2.2*fc(i,j))+(1.7201*dtime(i,j)))
endif
if ( dtime(i,j).le.2.6520841.and.dtime(i,j).gt.2.3499992.and.terrain(i,j).gt.3.8671637.and.modis_dtrad(i,j).gt.7.7208467) then
 latn(i,j) = (-7.4974045+(0.35*modis_dtrad(i,j))+(-0.057*day(i,j))+(-0.0024*terrain(i,j))+(-1.6*fc(i,j))+(12.0058*dtime(i,j)))
endif
if ( dtime(i,j).le.3.1208341.and.terrain(i,j).gt.46.655537.and.dtime(i,j).gt.2.6520841.and.day(i,j).le.303.73444.and.modis_dtrad(i,j).gt.7.7208467) then
 latn(i,j) = (-23.0420975+(0.468*modis_dtrad(i,j))+(0.085*day(i,j))+(-1.9*fc(i,j))+(1.1588*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.216666.and.dtime(i,j).le.3.625001.and.day(i,j).le.303.73444.and.terrain(i,j).gt.1.8872331.and.modis_dtrad(i,j).gt.7.7208467) then
 latn(i,j) = (-21.2151245+(0.434*modis_dtrad(i,j))+(0.054*day(i,j))+(-0.0038*terrain(i,j))+(3.0208*dtime(i,j)))
endif
if ( dtime(i,j).le.3.1091135.and.dtime(i,j).gt.2.8583345.and.day(i,j).gt.303.73444) then
 latn(i,j) = (-54.9101071+(0.311*modis_dtrad(i,j))+(0.205*day(i,j))+(0.0103*terrain(i,j))+(-4.1*fc(i,j)))
endif
if ( dtime(i,j).gt.3.375001.and.dtime(i,j).le.3.625001.and.day(i,j).gt.303.73444) then
 latn(i,j) = (44.7703831+(0.706*modis_dtrad(i,j))+(-0.087*day(i,j))+(-0.0115*terrain(i,j))+(1.8*fc(i,j))+(-5.0644*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.625001.and.day(i,j).gt.295.60349) then
 latn(i,j) = (39.9200907+(0.484*modis_dtrad(i,j))+(-0.097*day(i,j))+(-0.0022*terrain(i,j))+(-1.6*fc(i,j))+(-1.109*dtime(i,j)))
endif
if ( dtime(i,j).le.3.375001.and.dtime(i,j).gt.3.1091135.and.day(i,j).gt.303.73444.and.modis_dtrad(i,j).gt.7.7208467) then
 latn(i,j) = (-72.2210978+(0.303*modis_dtrad(i,j))+(0.288*day(i,j))+(0.012*terrain(i,j))+(-3.1*fc(i,j))+(-2.4245*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.625001.and.day(i,j).le.295.60349) then
 latn(i,j) = (-36.3558734+(0.424*modis_dtrad(i,j))+(0.162*day(i,j))+(-0.0045*terrain(i,j))+(0.2*fc(i,j))+(-1.1044*dtime(i,j)))
endif
if ( day(i,j).gt.303.73444.and.dtime(i,j).le.2.8583345) then
 latn(i,j) = (-40.1119286+(0.411*modis_dtrad(i,j))+(0.08*day(i,j))+(0.006*terrain(i,j))+(-1.2*fc(i,j))+(7.5753*dtime(i,j)))
endif
