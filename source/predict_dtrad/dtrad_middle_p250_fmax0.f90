if ( dtime(i,j).le.2.1291697) then
 latn(i,j) = (-1.5058084+(0.326*dtrad(i,j))+(-0.031*day(i,j))+(0.004*terrain(i,j))+(-3*fc(i,j))+(7.2*dtime(i,j)))
endif
if ( dtrad(i,j).le.19.283241.and.dtime(i,j).gt.2.1291697.and.day(i,j).le.308.46628.and.terrain(i,j).gt.2.1578522) then
 latn(i,j) = (34.7074234+(0.744*dtrad(i,j))+(-0.159*day(i,j))+(0.008*terrain(i,j))+(4.04*dtime(i,j)))
endif
if ( terrain(i,j).le.2.1578522.and.dtrad(i,j).le.25.145691.and.dtime(i,j).gt.2.1291697) then
 latn(i,j) = (20.7685623+(0.514*dtrad(i,j))+(-0.104*day(i,j))+(0.934*terrain(i,j))+(3.99*dtime(i,j)))
endif
if ( dtime(i,j).le.2.2312512.and.dtrad(i,j).gt.25.145691) then
 latn(i,j) = (2.7537362+(0.347*dtrad(i,j))+(-0.062*day(i,j))+(0.004*terrain(i,j))+(-2*fc(i,j))+(9.39*dtime(i,j)))
endif
if ( day(i,j).gt.308.46628.and.dtrad(i,j).le.25.145691.and.dtime(i,j).le.2.8999996) then
 latn(i,j) = (-18.4623143+(0.621*dtrad(i,j))+(0.007*day(i,j))+(-0.006*terrain(i,j))+(4*fc(i,j))+(5.94*dtime(i,j)))
endif
if ( dtrad(i,j).le.25.145691.and.dtime(i,j).gt.2.1291697.and.day(i,j).le.308.46628.and.dtrad(i,j).gt.19.283241.and.terrain(i,j).gt.2.1578522) then
 latn(i,j) = (2.2203937+(0.419*dtrad(i,j))+(-0.036*day(i,j))+(4.95*dtime(i,j)))
endif
if ( dtime(i,j).le.2.4302106.and.dtrad(i,j).gt.25.145691.and.dtime(i,j).gt.2.2312512) then
 latn(i,j) = (-17.3888195+(0.265*dtrad(i,j))+(0.049*day(i,j))+(-0.012*terrain(i,j))+(7*fc(i,j))+(3.66*dtime(i,j)))
endif
if ( dtrad(i,j).le.25.145691.and.dtime(i,j).le.3.3270812.and.dtime(i,j).gt.2.8999996.and.day(i,j).gt.308.46628) then
 latn(i,j) = (5.9043888+(0.849*dtrad(i,j))+(-0.117*day(i,j))+(-0.004*terrain(i,j))+(10*fc(i,j))+(8.35*dtime(i,j)))
endif
if ( dtrad(i,j).le.24.930624.and.dtime(i,j).gt.3.3270812) then
 latn(i,j) = (-1.9510256+(0.778*dtrad(i,j))+(-0.032*day(i,j))+(2.98*dtime(i,j)))
endif
if ( terrain(i,j).le.1.0779178.and.dtrad(i,j).gt.25.145691.and.dtime(i,j).le.3.3270812) then
 latn(i,j) = (-7.2960723+(0.38*dtrad(i,j))+(-0.006*day(i,j))+(0.573*terrain(i,j))+(-2*fc(i,j))+(5.24*dtime(i,j)))
endif
if ( dtime(i,j).le.3.1416662.and.dtrad(i,j).gt.25.145691.and.dtime(i,j).gt.2.4302106.and.day(i,j).le.319.43118.and.terrain(i,j).gt.1.0779178) then
 latn(i,j) = (-6.2571866+(0.367*dtrad(i,j))+(-0.01*terrain(i,j))+(4.7*dtime(i,j)))
endif
if ( dtime(i,j).le.2.9781265.and.day(i,j).gt.319.43118.and.dtrad(i,j).gt.25.145691) then
 latn(i,j) = (2.7103592+(0.389*dtrad(i,j))+(-0.125*day(i,j))+(-0.015*terrain(i,j))+(3*fc(i,j))+(15.74*dtime(i,j)))
endif
if ( terrain(i,j).le.1.6753062.and.dtime(i,j).gt.3.3270812.and.day(i,j).le.323.87332) then
 latn(i,j) = (-2.2932401+(0.693*dtrad(i,j))+(1.922*terrain(i,j)))
endif
if ( dtime(i,j).le.3.3270812.and.day(i,j).gt.319.43118.and.dtime(i,j).gt.2.9781265.and.dtrad(i,j).gt.25.145691) then
 latn(i,j) = (54.8089916+(0.342*dtrad(i,j))+(-0.222*day(i,j))+(-0.02*terrain(i,j))+(-7*fc(i,j))+(8.41*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.1416662.and.dtime(i,j).le.3.3270812.and.day(i,j).le.319.43118.and.terrain(i,j).gt.1.0779178) then
 latn(i,j) = (-37.8011425+(0.326*dtrad(i,j))+(-0.013*terrain(i,j))+(14.91*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.4833314.and.dtime(i,j).le.3.9395823.and.day(i,j).gt.314.12427.and.day(i,j).le.323.87332) then
 latn(i,j) = (-38.3435842+(0.426*dtrad(i,j))+(0.103*day(i,j))+(-0.014*terrain(i,j))+(4*fc(i,j))+(3.52*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.4833314.and.day(i,j).le.314.12427.and.dtime(i,j).le.3.9395823) then
 latn(i,j) = (59.4476842+(0.511*dtrad(i,j))+(-0.14*day(i,j))+(-0.011*terrain(i,j))+(-10*fc(i,j))+(-2.48*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.3270812.and.dtime(i,j).le.3.4833314.and.dtrad(i,j).gt.24.930624.and.day(i,j).le.323.87332) then
 latn(i,j) = (-26.8224619+(0.432*dtrad(i,j))+(-0.048*day(i,j))+(-0.015*terrain(i,j))+(-9*fc(i,j))+(14.96*dtime(i,j)))
endif
if ( day(i,j).gt.323.87332.and.dtime(i,j).le.3.9395823.and.dtime(i,j).gt.3.3270812) then
 latn(i,j) = (4.7738615+(0.397*dtrad(i,j))+(-0.024*day(i,j))+(-0.02*terrain(i,j))+(-3*fc(i,j))+(3.52*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.9395823) then
 latn(i,j) = (-2.7715737+(0.483*dtrad(i,j))+(0.017*day(i,j))+(-0.008*terrain(i,j))+(-11*fc(i,j))+(1.62*dtime(i,j)))
endif
