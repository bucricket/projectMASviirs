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

 if (precip(i,j).gt.0.0.and.precip(i,j).lt.250.and.fmax(i,j).ge.0.25.and.fmax(i,j).lt.1.0) then 
 if (day(i,j).ne.-9999.and.night(i,j).ne.-9999.) then
 if (terrain(i,j).ne.-9999.and.lai(i,j).ne.-9999.) then 

  dtrad(i,j) = day(i,j)-night(i,j)
  fc(i,j) = 1.0-exp(-0.5*lai(i,j))

if ( dtime(i,j).le.1.6750011) then
 latn(i,j) = (1.756658+(0.209*dtrad(i,j))+(3.1*fc(i,j))+(0.35*dtime(i,j)))
endif
if ( dtrad(i,j).le.19.578287.and.dtime(i,j).le.2.4333327.and.dtime(i,j).gt.1.6750011) then
 latn(i,j) = (-3.048915+(0.344*dtrad(i,j))+(-2.1*fc(i,j))+(3.29*dtime(i,j)))
endif
if ( fc(i,j).gt.0.14358249.and.dtime(i,j).le.3.0749989.and.day(i,j).le.312.91287) then
 latn(i,j) = (8.016057+(0.371*dtrad(i,j))+(-0.05*day(i,j))+(-4.1*fc(i,j))+(4.95*dtime(i,j)))
endif
if ( dtrad(i,j).le.19.578287.and.dtime(i,j).gt.2.4333327) then
 latn(i,j) = (5.974614+(0.646*dtrad(i,j))+(-0.036*day(i,j))+(-2*fc(i,j))+(2.4*dtime(i,j)))
endif
if ( dtime(i,j).le.3.0791659.and.fc(i,j).le.0.14358249.and.dtrad(i,j).gt.19.578287) then
 latn(i,j) = (-15.119496+(0.334*dtrad(i,j))+(0.031*day(i,j))+(-0.006*terrain(i,j))+(4.1*fc(i,j))+(4.48*dtime(i,j)))
endif
if ( dtime(i,j).le.3.0749989.and.day(i,j).gt.312.91287) then
 latn(i,j) = (55.110956+(0.209*dtrad(i,j))+(-0.163*day(i,j))+(-0.009*terrain(i,j))+(-8*fc(i,j))+(3.11*dtime(i,j)))
endif
if ( dtime(i,j).le.3.1416659.and.dtime(i,j).gt.3.0749989.and.fc(i,j).gt.0.14358249.and.dtrad(i,j).gt.19.578287) then
 latn(i,j) = (78.340632+(0.436*dtrad(i,j))+(-0.142*day(i,j))+(0.006*terrain(i,j))+(-5.4*fc(i,j))+(-9.07*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.3333325.and.dtime(i,j).le.3.4593737) then
 latn(i,j) = (-49.689891+(0.406*dtrad(i,j))+(-0.154*day(i,j))+(-6*fc(i,j))+(31.2*dtime(i,j)))
endif
if ( day(i,j).le.310.91605.and.dtime(i,j).gt.3.1416659.and.fc(i,j).gt.0.2402212) then
 latn(i,j) = (28.782796+(0.399*dtrad(i,j))+(-0.083*day(i,j))+(-0.007*terrain(i,j))+(-2.1*fc(i,j))+(1.35*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.0791659.and.dtime(i,j).le.3.1416659.and.fc(i,j).le.0.14358249.and.dtrad(i,j).gt.19.578287) then
 latn(i,j) = (-50.935569+(0.292*dtrad(i,j))+(-0.098*day(i,j))+(-0.01*terrain(i,j))+(-8.6*fc(i,j))+(29.75*dtime(i,j)))
endif
if ( fc(i,j).gt.0.2402212.and.day(i,j).gt.310.91605.and.dtime(i,j).le.4.5833287.and.dtime(i,j).gt.3.1416659) then
 latn(i,j) = (-6.456294+(0.423*dtrad(i,j))+(0.015*terrain(i,j))+(-4.9*fc(i,j))+(3.66*dtime(i,j)))
endif
if ( dtime(i,j).gt.4.5833287.and.fc(i,j).gt.0.2402212.and.dtrad(i,j).gt.19.578287) then
 latn(i,j) = (-57.697198+(0.167*dtrad(i,j))+(0.107*day(i,j))+(0.014*terrain(i,j))+(-7.8*fc(i,j))+(8.8*dtime(i,j)))
endif
if ( day(i,j).gt.322.18536.and.dtrad(i,j).le.32.871368.and.dtime(i,j).gt.3.1416659.and.dtime(i,j).le.4.5687456) then
 latn(i,j) = (-1.36774+(0.323*dtrad(i,j))+(-0.008*day(i,j))+(-0.01*terrain(i,j))+(-8.1*fc(i,j))+(4.2*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.1416659.and.dtime(i,j).le.3.3333325.and.fc(i,j).le.0.2402212) then
 latn(i,j) = (-17.644788+(0.346*dtrad(i,j))+(-0.057*day(i,j))+(-6.8*fc(i,j))+(13.94*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.4593737.and.fc(i,j).le.0.2402212.and.day(i,j).le.322.18536.and.dtime(i,j).le.4.5583334) then
 latn(i,j) = (12.658102+(0.411*dtrad(i,j))+(-0.052*day(i,j))+(-0.005*terrain(i,j))+(-5.8*fc(i,j))+(3.4*dtime(i,j)))
endif
if ( dtime(i,j).le.3.6958313.and.day(i,j).gt.322.18536.and.dtrad(i,j).gt.32.871368.and.fc(i,j).le.0.2402212.and.dtime(i,j).gt.3.1416659) then
 latn(i,j) = (-22.796084+(0.629*dtrad(i,j))+(0.009*terrain(i,j))+(-8.3*fc(i,j))+(6.16*dtime(i,j)))
endif
if ( dtime(i,j).gt.4.5583334.and.fc(i,j).le.0.2402212.and.day(i,j).le.322.18536.and.dtrad(i,j).gt.19.578287) then
 latn(i,j) = (-19.100847+(0.024*dtrad(i,j))+(0.068*day(i,j))+(-0.007*terrain(i,j))+(-8.5*fc(i,j))+(4.45*dtime(i,j)))
endif
if ( dtime(i,j).le.3.756249.and.dtime(i,j).gt.3.6958313.and.dtrad(i,j).gt.32.871368.and.fc(i,j).le.0.2402212) then
 latn(i,j) = (-227.399137+(0.084*dtrad(i,j))+(0.293*day(i,j))+(-0.05*terrain(i,j))+(-1.6*fc(i,j))+(41.54*dtime(i,j)))
endif
if ( dtrad(i,j).gt.32.871368.and.dtime(i,j).gt.3.756249.and.dtime(i,j).le.4.5687456) then
 latn(i,j) = (-20.407408+(0.596*dtrad(i,j))+(-0.01*terrain(i,j))+(-7.6*fc(i,j))+(5.81*dtime(i,j)))
endif
if ( dtime(i,j).gt.4.5687456.and.day(i,j).gt.322.18536) then
 latn(i,j) = (-49.427965+(0.195*dtrad(i,j))+(0.212*day(i,j))+(-0.014*terrain(i,j))+(-11*fc(i,j)))
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
