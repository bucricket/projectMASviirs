if ( dtime(i,j).le.1.6166668) then
 latn(i,j) = (-17.5078859+(0.157*modis_dtrad(i,j))+(0.074*day(i,j)))
endif
if ( modis_dtrad(i,j).le.8.1104765) then
 latn(i,j) = (-20.2467188+(0.718*modis_dtrad(i,j))+(0.071*day(i,j))+(0.0019*terrain(i,j))+(1.7*fc(i,j)))
endif
if ( day(i,j).le.291.34402.and.dtime(i,j).le.2.3645842.and.dtime(i,j).gt.1.6166668.and.modis_dtrad(i,j).gt.8.1104765) then
 latn(i,j) = (-30.3077934+(0.166*modis_dtrad(i,j))+(0.104*day(i,j))+(-0.0061*terrain(i,j))+(-4.2*fc(i,j))+(3.7586*dtime(i,j)))
endif
if ( dtime(i,j).le.2.3645842.and.day(i,j).gt.291.34402.and.modis_dtrad(i,j).gt.8.1104765) then
 latn(i,j) = (12.0230043+(0.289*modis_dtrad(i,j))+(-0.071*day(i,j))+(-0.0014*terrain(i,j))+(-3.1*fc(i,j))+(6.6909*dtime(i,j)))
endif
if ( dtime(i,j).le.2.9416656.and.dtime(i,j).gt.2.3645842.and.day(i,j).le.303.37073.and.modis_dtrad(i,j).gt.8.1104765) then
 latn(i,j) = (-22.7056189+(0.309*modis_dtrad(i,j))+(0.065*day(i,j))+(-3.2*fc(i,j))+(4.1177*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.85833.and.day(i,j).le.301.23203.and.dtime(i,j).le.5.0583344.and.modis_dtrad(i,j).gt.8.1104765.and.fc(i,j).gt.0.14021321) then
 latn(i,j) = (4.9784885+(0.45*modis_dtrad(i,j))+(0.014*day(i,j))+(0.003*terrain(i,j))+(-2.9*fc(i,j))+(-0.3891*dtime(i,j)))
endif
if ( terrain(i,j).le.1.7700616.and.dtime(i,j).gt.3.625001.and.dtime(i,j).le.4.5749989.and.fc(i,j).gt.0.14021321.and.modis_dtrad(i,j).gt.8.1104765) then
 latn(i,j) = (24.2913078+(0.485*modis_dtrad(i,j))+(-0.054*day(i,j))+(1.5838*terrain(i,j))+(-2.2*fc(i,j))+(-0.8878*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.625001.and.dtime(i,j).le.3.85833.and.day(i,j).le.301.23203.and.modis_dtrad(i,j).gt.8.1104765) then
 latn(i,j) = (-28.6255914+(0.309*modis_dtrad(i,j))+(0.172*day(i,j))+(-0.0027*terrain(i,j))+(-5.2*fc(i,j))+(-2.9856*dtime(i,j)))
endif
if ( dtime(i,j).le.3.2843757.and.dtime(i,j).gt.2.9416656) then
 latn(i,j) = (-22.3601227+(0.343*modis_dtrad(i,j))+(0.055*day(i,j))+(0.0056*terrain(i,j))+(-3*fc(i,j))+(4.1109*dtime(i,j)))
endif
if ( dtime(i,j).le.3.625001.and.dtime(i,j).gt.3.2843757.and.fc(i,j).gt.0.19698051) then
 latn(i,j) = (-13.3128123+(0.351*modis_dtrad(i,j))+(0.076*day(i,j))+(0.001*terrain(i,j))+(-6.4*fc(i,j))+(0.0377*dtime(i,j)))
endif
if ( dtime(i,j).le.2.8270838.and.day(i,j).gt.303.37073.and.dtime(i,j).gt.2.3645842.and.fc(i,j).gt.0.1078644) then
 latn(i,j) = (31.6200937+(0.418*modis_dtrad(i,j))+(-0.173*day(i,j))+(0.0104*terrain(i,j))+(-4.4*fc(i,j))+(10.1697*dtime(i,j)))
endif
if ( dtime(i,j).le.2.9416656.and.dtime(i,j).gt.2.8270838.and.fc(i,j).gt.0.1078644.and.modis_dtrad(i,j).gt.8.1104765) then
 latn(i,j) = (-22.6457859+(0.383*modis_dtrad(i,j))+(-0.012*day(i,j))+(-2.4*fc(i,j))+(11.4048*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.8645823.and.dtime(i,j).le.4.5749989.and.day(i,j).le.309.66037.and.fc(i,j).gt.0.14021321.and.modis_dtrad(i,j).gt.8.1104765) then
 latn(i,j) = (12.4218864+(0.424*modis_dtrad(i,j))+(-3.7*fc(i,j))+(-0.9347*dtime(i,j)))
endif
if ( dtime(i,j).gt.5.0583344) then
 latn(i,j) = (87.6384564+(0.782*modis_dtrad(i,j))+(-0.32*day(i,j))+(-0.0063*terrain(i,j))+(1.9231*dtime(i,j)))
endif
if ( fc(i,j).le.0.1078644.and.dtime(i,j).gt.2.3645842.and.dtime(i,j).le.2.9416656.and.modis_dtrad(i,j).gt.8.1104765) then
 latn(i,j) = (-14.7505907+(0.343*modis_dtrad(i,j))+(0.033*day(i,j))+(-0.0044*terrain(i,j))+(12*fc(i,j))+(3.7619*dtime(i,j)))
endif
if ( dtime(i,j).gt.4.5749989.and.day(i,j).gt.301.23203.and.dtime(i,j).le.5.0583344.and.fc(i,j).gt.0.14021321) then
 latn(i,j) = (0.4728077+(0.343*modis_dtrad(i,j))+(0.063*day(i,j))+(-0.0023*terrain(i,j))+(-7.2*fc(i,j))+(-1.5034*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.625001.and.dtime(i,j).le.3.8645823.and.day(i,j).gt.301.23203.and.day(i,j).le.309.66037.and.terrain(i,j).gt.1.7700616.and.fc(i,j).gt.0.14021321) then
 latn(i,j) = (46.6331432+(0.315*modis_dtrad(i,j))+(-0.011*terrain(i,j))+(-5.6*fc(i,j))+(-9.0781*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.2843757.and.fc(i,j).le.0.19698051.and.dtime(i,j).le.3.625001.and.modis_dtrad(i,j).gt.8.1104765) then
 latn(i,j) = (17.2468131+(0.299*modis_dtrad(i,j))+(0.041*day(i,j))+(-0.0149*terrain(i,j))+(-28.3*fc(i,j))+(-4.4746*dtime(i,j)))
endif
if ( day(i,j).gt.309.66037.and.dtime(i,j).gt.3.625001.and.dtime(i,j).le.4.5749989.and.terrain(i,j).gt.1.7700616.and.fc(i,j).gt.0.14021321) then
 latn(i,j) = (15.2625462+(0.38*modis_dtrad(i,j))+(-5*fc(i,j))+(-1.2988*dtime(i,j)))
endif
if ( fc(i,j).le.0.14021321.and.dtime(i,j).gt.3.625001) then
 latn(i,j) = (28.2513252+(0.621*modis_dtrad(i,j))+(-0.073*day(i,j))+(-0.0181*terrain(i,j))+(-19.3*fc(i,j))+(0.0235*dtime(i,j)))
endif
