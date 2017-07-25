if ( dtime(i,j).le.1.6750011) then
 latn(i,j) = (1.756658+(0.209*modis_dtrad(i,j))+(3.1*fc(i,j))+(0.35*dtime(i,j)))
endif
if ( modis_dtrad(i,j).le.19.578287.and.dtime(i,j).le.2.4333327.and.dtime(i,j).gt.1.6750011) then
 latn(i,j) = (-3.048915+(0.344*modis_dtrad(i,j))+(-2.1*fc(i,j))+(3.29*dtime(i,j)))
endif
if ( fc(i,j).gt.0.14358249.and.dtime(i,j).le.3.0749989.and.day(i,j).le.312.91287) then
 latn(i,j) = (8.016057+(0.371*modis_dtrad(i,j))+(-0.05*day(i,j))+(-4.1*fc(i,j))+(4.95*dtime(i,j)))
endif
if ( modis_dtrad(i,j).le.19.578287.and.dtime(i,j).gt.2.4333327) then
 latn(i,j) = (5.974614+(0.646*modis_dtrad(i,j))+(-0.036*day(i,j))+(-2*fc(i,j))+(2.4*dtime(i,j)))
endif
if ( dtime(i,j).le.3.0791659.and.fc(i,j).le.0.14358249.and.modis_dtrad(i,j).gt.19.578287) then
 latn(i,j) = (-15.119496+(0.334*modis_dtrad(i,j))+(0.031*day(i,j))+(-0.006*terrain(i,j))+(4.1*fc(i,j))+(4.48*dtime(i,j)))
endif
if ( dtime(i,j).le.3.0749989.and.day(i,j).gt.312.91287) then
 latn(i,j) = (55.110956+(0.209*modis_dtrad(i,j))+(-0.163*day(i,j))+(-0.009*terrain(i,j))+(-8*fc(i,j))+(3.11*dtime(i,j)))
endif
if ( dtime(i,j).le.3.1416659.and.dtime(i,j).gt.3.0749989.and.fc(i,j).gt.0.14358249.and.modis_dtrad(i,j).gt.19.578287) then
 latn(i,j) = (78.340632+(0.436*modis_dtrad(i,j))+(-0.142*day(i,j))+(0.006*terrain(i,j))+(-5.4*fc(i,j))+(-9.07*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.3333325.and.dtime(i,j).le.3.4593737) then
 latn(i,j) = (-49.689891+(0.406*modis_dtrad(i,j))+(-0.154*day(i,j))+(-6*fc(i,j))+(31.2*dtime(i,j)))
endif
if ( day(i,j).le.310.91605.and.dtime(i,j).gt.3.1416659.and.fc(i,j).gt.0.2402212) then
 latn(i,j) = (28.782796+(0.399*modis_dtrad(i,j))+(-0.083*day(i,j))+(-0.007*terrain(i,j))+(-2.1*fc(i,j))+(1.35*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.0791659.and.dtime(i,j).le.3.1416659.and.fc(i,j).le.0.14358249.and.modis_dtrad(i,j).gt.19.578287) then
 latn(i,j) = (-50.935569+(0.292*modis_dtrad(i,j))+(-0.098*day(i,j))+(-0.01*terrain(i,j))+(-8.6*fc(i,j))+(29.75*dtime(i,j)))
endif
if ( fc(i,j).gt.0.2402212.and.day(i,j).gt.310.91605.and.dtime(i,j).le.4.5833287.and.dtime(i,j).gt.3.1416659) then
 latn(i,j) = (-6.456294+(0.423*modis_dtrad(i,j))+(0.015*terrain(i,j))+(-4.9*fc(i,j))+(3.66*dtime(i,j)))
endif
if ( dtime(i,j).gt.4.5833287.and.fc(i,j).gt.0.2402212.and.modis_dtrad(i,j).gt.19.578287) then
 latn(i,j) = (-57.697198+(0.167*modis_dtrad(i,j))+(0.107*day(i,j))+(0.014*terrain(i,j))+(-7.8*fc(i,j))+(8.8*dtime(i,j)))
endif
if ( day(i,j).gt.322.18536.and.modis_dtrad(i,j).le.32.871368.and.dtime(i,j).gt.3.1416659.and.dtime(i,j).le.4.5687456) then
 latn(i,j) = (-1.36774+(0.323*modis_dtrad(i,j))+(-0.008*day(i,j))+(-0.01*terrain(i,j))+(-8.1*fc(i,j))+(4.2*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.1416659.and.dtime(i,j).le.3.3333325.and.fc(i,j).le.0.2402212) then
 latn(i,j) = (-17.644788+(0.346*modis_dtrad(i,j))+(-0.057*day(i,j))+(-6.8*fc(i,j))+(13.94*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.4593737.and.fc(i,j).le.0.2402212.and.day(i,j).le.322.18536.and.dtime(i,j).le.4.5583334) then
 latn(i,j) = (12.658102+(0.411*modis_dtrad(i,j))+(-0.052*day(i,j))+(-0.005*terrain(i,j))+(-5.8*fc(i,j))+(3.4*dtime(i,j)))
endif
if ( dtime(i,j).le.3.6958313.and.day(i,j).gt.322.18536.and.modis_dtrad(i,j).gt.32.871368.and.fc(i,j).le.0.2402212.and.dtime(i,j).gt.3.1416659) then
 latn(i,j) = (-22.796084+(0.629*modis_dtrad(i,j))+(0.009*terrain(i,j))+(-8.3*fc(i,j))+(6.16*dtime(i,j)))
endif
if ( dtime(i,j).gt.4.5583334.and.fc(i,j).le.0.2402212.and.day(i,j).le.322.18536.and.modis_dtrad(i,j).gt.19.578287) then
 latn(i,j) = (-19.100847+(0.024*modis_dtrad(i,j))+(0.068*day(i,j))+(-0.007*terrain(i,j))+(-8.5*fc(i,j))+(4.45*dtime(i,j)))
endif
if ( dtime(i,j).le.3.756249.and.dtime(i,j).gt.3.6958313.and.modis_dtrad(i,j).gt.32.871368.and.fc(i,j).le.0.2402212) then
 latn(i,j) = (-227.399137+(0.084*modis_dtrad(i,j))+(0.293*day(i,j))+(-0.05*terrain(i,j))+(-1.6*fc(i,j))+(41.54*dtime(i,j)))
endif
if ( modis_dtrad(i,j).gt.32.871368.and.dtime(i,j).gt.3.756249.and.dtime(i,j).le.4.5687456) then
 latn(i,j) = (-20.407408+(0.596*modis_dtrad(i,j))+(-0.01*terrain(i,j))+(-7.6*fc(i,j))+(5.81*dtime(i,j)))
endif
if ( dtime(i,j).gt.4.5687456.and.day(i,j).gt.322.18536) then
 latn(i,j) = (-49.427965+(0.195*modis_dtrad(i,j))+(0.212*day(i,j))+(-0.014*terrain(i,j))+(-11*fc(i,j)))
endif
