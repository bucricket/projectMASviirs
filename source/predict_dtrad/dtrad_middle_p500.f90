if ( dtime(i,j).le.1.5666664.and.modis_dtrad(i,j).le.14.333391) then
 latn(i,j) = (-10.1172069+(0.099*modis_dtrad(i,j))+(0.05*day(i,j))+(-0.005*terrain(i,j))+(0.5*fc(i,j)))
endif
if ( dtime(i,j).le.1.6458384.and.modis_dtrad(i,j).gt.14.333391) then
 latn(i,j) = (-33.7260704+(0.138*modis_dtrad(i,j))+(0.132*day(i,j)))
endif
if ( dtime(i,j).le.1.7687515.and.dtime(i,j).gt.1.5666664) then
 latn(i,j) = (-8.9760558+(0.158*modis_dtrad(i,j))+(-0.006*terrain(i,j))+(7.7641*dtime(i,j)))
endif
if ( modis_dtrad(i,j).le.14.333391.and.dtime(i,j).gt.1.7687515.and.dtime(i,j).le.2.5999999) then
 latn(i,j) = (-6.0488545+(0.286*modis_dtrad(i,j))+(0.006*day(i,j))+(-2.5*fc(i,j))+(4.3457*dtime(i,j)))
endif
if ( dtime(i,j).le.2.3291681.and.modis_dtrad(i,j).gt.14.333391.and.dtime(i,j).gt.1.6458384) then
 latn(i,j) = (-20.8916886+(0.234*modis_dtrad(i,j))+(0.061*day(i,j))+(-0.004*terrain(i,j))+(-2.6*fc(i,j))+(4.3235*dtime(i,j)))
endif
if ( modis_dtrad(i,j).le.14.333391.and.dtime(i,j).gt.2.5999999) then
 latn(i,j) = (-3.2000537+(0.576*modis_dtrad(i,j))+(0.008*day(i,j))+(0.009*terrain(i,j))+(-1.2*fc(i,j))+(1.2707*dtime(i,j)))
endif
if ( day(i,j).le.296.32886.and.dtime(i,j).gt.2.3291681.and.dtime(i,j).le.3.2041662.and.fc(i,j).gt.0.099774703) then
 latn(i,j) = (-27.6182198+(0.213*modis_dtrad(i,j))+(0.112*day(i,j))+(-0.006*terrain(i,j))+(-4.1*fc(i,j))+(1.5605*dtime(i,j)))
endif
if ( dtime(i,j).le.2.8083334.and.day(i,j).gt.296.32886.and.modis_dtrad(i,j).gt.14.333391) then
 latn(i,j) = (-19.4747+(0.348*modis_dtrad(i,j))+(0.033*day(i,j))+(-2.5*fc(i,j))+(6.113*dtime(i,j)))
endif
if ( fc(i,j).le.0.099774703.and.dtime(i,j).gt.2.3291681.and.dtime(i,j).le.2.8916676.and.modis_dtrad(i,j).gt.14.333391) then
 latn(i,j) = (-21.7913334+(0.255*modis_dtrad(i,j))+(0.052*day(i,j))+(-0.007*terrain(i,j))+(-0.6*fc(i,j))+(5.6417*dtime(i,j)))
endif
if ( dtime(i,j).le.2.9916666.and.dtime(i,j).gt.2.8916674.and.fc(i,j).gt.0.099774703) then
 latn(i,j) = (119.1887771+(0.408*modis_dtrad(i,j))+(0.041*day(i,j))+(-0.007*terrain(i,j))+(-3*fc(i,j))+(-42.1144*dtime(i,j)))
endif
if ( dtime(i,j).le.2.8916674.and.dtime(i,j).gt.2.8083334.and.fc(i,j).gt.0.099774703.and.modis_dtrad(i,j).gt.14.333391) then
 latn(i,j) = (-49.9743535+(0.289*modis_dtrad(i,j))+(0.032*day(i,j))+(-0.006*terrain(i,j))+(-6.3*fc(i,j))+(17.4634*dtime(i,j)))
endif
if ( dtime(i,j).le.3.2041662.and.dtime(i,j).gt.2.9916666.and.modis_dtrad(i,j).gt.14.333391) then
 latn(i,j) = (37.3755468+(0.355*modis_dtrad(i,j))+(0.011*day(i,j))+(-0.003*terrain(i,j))+(-4.9*fc(i,j))+(-10.4766*dtime(i,j)))
endif
if ( dtime(i,j).gt.4.2833338.and.fc(i,j).gt.0.2488669) then
 latn(i,j) = (45.1926396+(0.439*modis_dtrad(i,j))+(-0.081*day(i,j))+(0.006*terrain(i,j))+(-6.5*fc(i,j))+(-2.3874*dtime(i,j)))
endif
if ( terrain(i,j).le.2.4648063.and.dtime(i,j).gt.3.2041662.and.fc(i,j).gt.0.099774703.and.modis_dtrad(i,j).gt.14.333391) then
 latn(i,j) = (7.8977438+(0.45*modis_dtrad(i,j))+(-0.027*day(i,j))+(1.197*terrain(i,j))+(-5.3*fc(i,j))+(1.5055*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.3322911.and.dtime(i,j).le.3.8177083.and.day(i,j).le.319.33286.and.fc(i,j).gt.0.099774703.and.terrain(i,j).gt.2.4648063) then
 latn(i,j) = (-15.6814407+(0.328*modis_dtrad(i,j))+(0.047*day(i,j))+(-0.007*terrain(i,j))+(-5*fc(i,j))+(3.4801*dtime(i,j)))
endif
if ( dtime(i,j).le.3.3322911.and.dtime(i,j).gt.3.2041662.and.terrain(i,j).gt.2.4648063) then
 latn(i,j) = (-34.6459961+(0.172*modis_dtrad(i,j))+(0.064*day(i,j))+(-0.017*terrain(i,j))+(-7.5*fc(i,j))+(9.5506*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.8177083.and.dtime(i,j).le.4.2833338.and.day(i,j).le.319.33286.and.fc(i,j).gt.0.099774703.and.terrain(i,j).gt.2.4648063) then
 latn(i,j) = (-12.8867856+(0.326*modis_dtrad(i,j))+(0.028*day(i,j))+(-0.004*terrain(i,j))+(-6.7*fc(i,j))+(4.1068*dtime(i,j)))
endif
if ( fc(i,j).le.0.099774703.and.dtime(i,j).gt.2.8916676.and.dtime(i,j).le.4.2833338) then
 latn(i,j) = (-16.5686158+(0.399*modis_dtrad(i,j))+(0.03*day(i,j))+(-0.003*terrain(i,j))+(9*fc(i,j))+(4.1221*dtime(i,j)))
endif
if ( day(i,j).gt.319.33286.and.fc(i,j).gt.0.099774703.and.dtime(i,j).le.4.2833338.and.dtime(i,j).gt.3.2041662) then
 latn(i,j) = (19.6101441+(0.384*modis_dtrad(i,j))+(-0.059*day(i,j))+(-0.011*terrain(i,j))+(-9.4*fc(i,j))+(2.7591*dtime(i,j)))
endif
if ( dtime(i,j).gt.4.2833338.and.fc(i,j).le.0.2488669.and.modis_dtrad(i,j).gt.14.333391) then
 latn(i,j) = (34.2715956+(0.468*modis_dtrad(i,j))+(-0.045*day(i,j))+(0.009*terrain(i,j))+(-15.7*fc(i,j))+(-2.134*dtime(i,j)))
endif
