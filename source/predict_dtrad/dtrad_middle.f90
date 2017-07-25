if ( terrain(i,j).gt.140.7823.and.fc(i,j).le.0.1961928.and.dtime(i,j).le.3.2333336.and.day(i,j).le.320.4877) then
 latn(i,j) = (161.207154+(1.326*modis_dtrad(i,j))+(-0.671*day(i,j))+(0.1*terrain(i,j))+(4.61*dtime(i,j)))
endif
if ( terrain(i,j).gt.150.26923.and.fc(i,j).le.0.1961928.and.day(i,j).le.323.53998) then
 latn(i,j) = (-100.3729458+(0.578*modis_dtrad(i,j))+(0.592*terrain(i,j)))
endif
if ( dtime(i,j).le.2.3708344.and.fc(i,j).le.0.1225492) then
 latn(i,j) = (-3.5400357+(0.31*modis_dtrad(i,j))+(-0.01*day(i,j))+(5.1*fc(i,j))+(4.89*dtime(i,j)))
endif
if ( dtime(i,j).le.2.3708344.and.fc(i,j).gt.0.1225492.and.modis_dtrad(i,j).le.33.786167) then
 latn(i,j) = (6.5991133+(0.339*modis_dtrad(i,j))+(-0.042*day(i,j))+(-3*fc(i,j))+(4.67*dtime(i,j)))
endif
if ( terrain(i,j).gt.19.648546.and.fc(i,j).gt.0.1961928.and.dtime(i,j).le.3.354166.and.dtime(i,j).gt.2.3708344.and.modis_dtrad(i,j).le.33.786167) then
 latn(i,j) = (0.3462191+(0.343*modis_dtrad(i,j))+(-0.008*day(i,j))+(0.004*terrain(i,j))+(-4.5*fc(i,j))+(3.33*dtime(i,j)))
endif
if ( dtime(i,j).le.2.8499999.and.dtime(i,j).gt.2.3708344.and.fc(i,j).le.0.1961928.and.terrain(i,j).le.140.7823) then
 latn(i,j) = (4.2518165+(0.365*modis_dtrad(i,j))+(-0.023*day(i,j))+(0.002*terrain(i,j))+(3*dtime(i,j)))
endif
if ( dtime(i,j).gt.2.3708344.and.dtime(i,j).le.3.3583329.and.terrain(i,j).le.19.648546.and.day(i,j).le.315.34042.and.modis_dtrad(i,j).le.33.786167) then
 latn(i,j) = (19.1843215+(0.452*modis_dtrad(i,j))+(-0.081*day(i,j))+(0.017*terrain(i,j))+(-2.9*fc(i,j))+(3.33*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.2333336.and.modis_dtrad(i,j).le.22.827059.and.fc(i,j).le.0.1961928.and.day(i,j).le.323.53998.and.terrain(i,j).le.150.26923) then
 latn(i,j) = (8.2323226+(0.472*modis_dtrad(i,j))+(-0.044*day(i,j))+(-0.019*terrain(i,j))+(19.8*fc(i,j))+(2.75*dtime(i,j)))
endif
if ( dtime(i,j).le.3.0749991.and.dtime(i,j).gt.2.8499999.and.terrain(i,j).le.140.7823) then
 latn(i,j) = (-15.3198202+(0.344*modis_dtrad(i,j))+(-0.005*terrain(i,j))+(-4.7*fc(i,j))+(8.11*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.1406248.and.dtime(i,j).le.3.2333336.and.modis_dtrad(i,j).le.33.786167.and.day(i,j).le.320.4877) then
 latn(i,j) = (-30.33703+(0.361*modis_dtrad(i,j))+(-0.071*day(i,j))+(-6.1*fc(i,j))+(18.98*dtime(i,j)))
endif
if ( day(i,j).gt.320.4877.and.dtime(i,j).le.3.2333336.and.fc(i,j).le.0.1961928) then
 latn(i,j) = (46.294863+(0.332*modis_dtrad(i,j))+(-0.157*day(i,j))+(-7.9*fc(i,j))+(3.98*dtime(i,j)))
endif
if ( dtime(i,j).le.3.1406248.and.dtime(i,j).gt.3.0749991) then
 latn(i,j) = (155.1485899+(0.514*modis_dtrad(i,j))+(-0.218*day(i,j))+(0.002*terrain(i,j))+(-4.4*fc(i,j))+(-26.89*dtime(i,j)))
endif
if ( day(i,j).gt.315.34042.and.fc(i,j).gt.0.1961928.and.terrain(i,j).le.19.648546.and.modis_dtrad(i,j).le.33.786167) then
 latn(i,j) = (-7.0260374+(0.342*modis_dtrad(i,j))+(0.022*day(i,j))+(0.077*terrain(i,j))+(-8.9*fc(i,j))+(2.68*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.3583329.and.day(i,j).le.315.34042.and.terrain(i,j).le.19.648546) then
 latn(i,j) = (39.8434606+(0.5*modis_dtrad(i,j))+(-0.139*day(i,j))+(0.048*terrain(i,j))+(-5.4*fc(i,j))+(2.43*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.2333336.and.fc(i,j).le.0.1961928.and.terrain(i,j).le.8.65277.and.modis_dtrad(i,j).le.33.786167.and.day(i,j).le.323.53998) then
 latn(i,j) = (30.5888354+(0.367*modis_dtrad(i,j))+(-0.099*day(i,j))+(0.082*terrain(i,j))+(-6.2*fc(i,j))+(2.8*dtime(i,j)))
endif
if ( terrain(i,j).gt.19.648546.and.fc(i,j).gt.0.1961928.and.dtime(i,j).gt.3.354166) then
 latn(i,j) = (19.2364636+(0.316*modis_dtrad(i,j))+(-0.047*day(i,j))+(-7.9*fc(i,j))+(2.1*dtime(i,j)))
endif
if ( day(i,j).gt.323.53998.and.modis_dtrad(i,j).le.33.786167.and.dtime(i,j).gt.3.2333336) then
 latn(i,j) = (27.184061+(0.338*modis_dtrad(i,j))+(-0.076*day(i,j))+(-0.01*terrain(i,j))+(-12.9*fc(i,j))+(2.52*dtime(i,j)))
endif
if ( dtime(i,j).gt.4.5697899) then
 latn(i,j) = (48.1359255+(0.327*modis_dtrad(i,j))+(-0.113*day(i,j))+(-12.4*fc(i,j))+(0.5*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.2333336.and.fc(i,j).le.0.1961928.and.terrain(i,j).gt.8.65277.and.day(i,j).le.323.53998.and.modis_dtrad(i,j).gt.22.827059.and.terrain(i,j).le.150.26923) then
 latn(i,j) = (14.710721+(0.364*modis_dtrad(i,j))+(-0.048*day(i,j))+(-0.009*terrain(i,j))+(2.81*dtime(i,j)))
endif
if ( modis_dtrad(i,j).gt.33.786167.and.dtime(i,j).le.4.5697899) then
 latn(i,j) = (13.7015374+(0.435*modis_dtrad(i,j))+(-0.066*day(i,j))+(-0.013*terrain(i,j))+(-9.5*fc(i,j))+(4.27*dtime(i,j)))
endif
