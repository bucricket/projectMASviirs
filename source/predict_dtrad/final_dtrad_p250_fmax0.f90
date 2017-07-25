program build_cubist_model

integer, parameter :: dx=3750, dy=3750
real :: latn(dx,dy), dtrad(dx,dy), day(dx,dy), night(dx,dy)
real :: dtime(dx,dy), lai(dx,dy), terrain(dx,dy)
character(len=4) :: arg1,arg3
character(len=3) :: arg2
character(len=255) :: arg4, basedir
!character (len=:), allocatable :: basedir
integer :: year, doy, yyyyddd, riseddd
character(len=255) :: ifile1, ifile2, ifile3, ifile4, ifile5, ifile6, ofile1
real :: predict(dx,dy)
real :: precip(dx,dy), fmax(dx,dy)
real :: fc(dx,dy)
integer :: laiddd, nweek1, cday, offset, rday

call getarg(1,arg1)
call getarg(2,arg2)
call getarg(3,arg3)
call getarg(4,arg4)
read(arg1,'(i4)') year
read(arg2,'(i3)') doy
yyyyddd=year*1000+doy
basedir=trim(arg4)

nweek1=(doy-1)/7.
cday=nweek1*7.
offset=int((doy-cday)/7.)
rday=int(((offset+nweek1)*7)+1)
riseddd=2014*1000+rday

write(ifile1,'(a,a,a,a)') basedir,'/STATIC/PRECIP/tiles/PRECIP_',arg3,'.dat'
write(ifile2,'(a,a,a,a)') basedir,'/STATIC/FMAX/tiles/FMAX_',arg3,'.dat'
open(10,file=ifile1,form='unformatted',access='direct',recl=dx*dy*4)
open(11,file=ifile2,form='unformatted',access='direct',recl=dx*dy*4)
read(10,rec=1) precip
read(11,rec=1) fmax
close(10)
close(11)

nweek1=(doy-1)/7.
cday=nweek1*8.
offset=int((doy-cday)/7.)
rday=int(((offset+nweek1)*8)+1)
laiddd=year*1000+rday

write(6,*) laiddd, riseddd
write(ifile1,'(a,a,a,a)') basedir,'/STATIC/TERRAIN_SD/tiles/TERRAIN_',arg3,'.dat'
write(ifile2,'(a,a,a,a,I7,a,a,a)') basedir,'/TILES/',arg3,'/FINAL_DAY_LST_',yyyyddd,'_',arg3,'.dat'
write(ifile3,'(a,a,a,a,I7,a,a,a)') basedir,'/TILES/',arg3,'/FINAL_NIGHT_LST_',yyyyddd,'_',arg3,'.dat'
write(ifile4,'(a,a,I7,a,a,a)') basedir,'/STATIC/LAI/tiles/MLAI_',laiddd,'_',arg3,'.dat'
write(ifile5,'(a,a,I7,a,a,a)') basedir,'/STATIC/DTIME/tiles/DTIME_',riseddd,'_',arg3,'.dat'

open(10,file=ifile1,form='unformatted',access='direct',recl=dx*dy*4)
open(11,file=ifile2,form='unformatted',access='direct',recl=dx*dy*4)
open(12,file=ifile3,form='unformatted',access='direct',recl=dx*dy*4)
open(13,file=ifile4,form='unformatted',access='direct',recl=dx*dy*4)
open(14,file=ifile5,form='unformatted',access='direct',recl=dx*dy*4)
read(10,rec=1) terrain
read(11,rec=1) day
read(12,rec=1) night
read(13,rec=1) lai
read(14,rec=1) dtime
close(10)
close(11)
close(12)
close(13)
close(14)

latn(:,:) = -9999.
do j = 1, dy
do i = 1, dx

 if (precip(i,j).gt.0.0.and.precip(i,j).lt.250.and.fmax(i,j).ge.0.0.and.fmax(i,j).lt.0.25) then 
 if (day(i,j).ne.-9999.and.night(i,j).ne.-9999.) then
 if (terrain(i,j).ne.-9999.and.lai(i,j).ne.-9999.) then 

  dtrad(i,j) = day(i,j)-night(i,j)
  fc(i,j) = 1.0-exp(-0.5*lai(i,j))
!  write(6,*) dtrad(i,j), fc(i,j), dtime(i,j), lai(i,j), fmax(i,j)
!  write(6,*) day(i,j), night(i,j), precip(i,j), terrain(i,j)
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
if ( dtime(i,j).gt.3.2395823) then
 latn(i,j) = (-2.7715737+(0.483*dtrad(i,j))+(0.017*day(i,j))+(-0.008*terrain(i,j))+(-11*fc(i,j))+(1.62*dtime(i,j)))
endif
 endif
 endif
 endif

enddo
enddo

write(ifile1,'(a,a,a,a)') basedir,'/PROCESSING/DTRAD_PREDICTION/comp1_',arg3,'.dat'
open(10,file=ifile1,form='unformatted',access='direct',recl=dx*dy*4)
write(10,rec=1) latn
close(10)

end program
