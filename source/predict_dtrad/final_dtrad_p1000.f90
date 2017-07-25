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

 if (precip(i,j).gt.750.0.and.precip(i,j).lt.1000.and.fmax(i,j).ge.0.0.and.fmax(i,j).lt.1.0) then 
 if (day(i,j).ne.-9999.and.night(i,j).ne.-9999.) then
 if (terrain(i,j).ne.-9999.and.lai(i,j).ne.-9999.) then 

  dtrad(i,j) = day(i,j)-night(i,j)
  fc(i,j) = 1.0-exp(-0.5*lai(i,j))

if ( dtime(i,j).le.1.5812508.and.dtrad(i,j).gt.10.253759) then
 latn(i,j) = (9.1260084+(0.053*dtrad(i,j))+(-0.023*day(i,j))+(2.3*fc(i,j)))
endif
if ( dtime(i,j).le.2.2000029.and.dtrad(i,j).le.10.253759) then
 latn(i,j) = (-2.4121588+(0.485*dtrad(i,j))+(0.01*day(i,j))+(0.0019*terrain(i,j)))
endif
if ( day(i,j).le.282.60614.and.dtrad(i,j).le.10.253759.and.dtime(i,j).gt.2.2000029) then
 latn(i,j) = (-34.3258905+(0.696*dtrad(i,j))+(0.127*day(i,j))+(0.0047*terrain(i,j))+(2*fc(i,j)))
endif
if ( dtime(i,j).le.2.2083328.and.day(i,j).le.288.59763.and.dtrad(i,j).gt.10.253759.and.dtime(i,j).gt.1.5812508) then
 latn(i,j) = (-27.5413735+(0.293*dtrad(i,j))+(0.097*day(i,j))+(-0.0029*terrain(i,j))+(-2*fc(i,j))+(2.0497*dtime(i,j)))
endif
if ( dtrad(i,j).le.10.253759.and.day(i,j).gt.282.60614.and.dtime(i,j).gt.2.2000029) then
 latn(i,j) = (16.2826677+(0.921*dtrad(i,j))+(-0.068*day(i,j))+(-0.0016*terrain(i,j))+(0.7*fc(i,j))+(1.1698*dtime(i,j)))
endif
if ( dtime(i,j).le.2.2083328.and.day(i,j).gt.288.59763.and.dtrad(i,j).gt.10.253759) then
 latn(i,j) = (28.1334984+(0.232*dtrad(i,j))+(-0.132*day(i,j))+(-0.0031*terrain(i,j))+(-4*fc(i,j))+(8.1167*dtime(i,j)))
endif
if ( dtime(i,j).le.3.5635402.and.dtime(i,j).gt.3.4249992.and.day(i,j).le.305.84937.and.day(i,j).gt.292.24063) then
 latn(i,j) = (-60.2911139+(0.301*dtrad(i,j))+(-0.011*day(i,j))+(0.0134*terrain(i,j))+(-3*fc(i,j))+(20.2109*dtime(i,j)))
endif
if ( day(i,j).le.292.24063.and.dtime(i,j).gt.2.2083328.and.dtime(i,j).le.3.5635402.and.dtrad(i,j).gt.10.253759) then
 latn(i,j) = (-27.7986106+(0.251*dtrad(i,j))+(0.105*day(i,j))+(-0.0094*terrain(i,j))+(1.1*fc(i,j))+(1.6148*dtime(i,j)))
endif
if ( dtime(i,j).le.3.4249992.and.dtime(i,j).gt.3.1489587.and.day(i,j).le.305.84937.and.day(i,j).gt.292.24063) then
 latn(i,j) = (23.3552798+(0.393*dtrad(i,j))+(-0.039*day(i,j))+(-0.7*fc(i,j))+(-1.8214*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.9833345.and.fc(i,j).gt.0.62520468.and.dtime(i,j).le.4.458334.and.day(i,j).gt.295.75366) then
 latn(i,j) = (60.5823165+(0.427*dtrad(i,j))+(-0.203*day(i,j))+(0.0043*terrain(i,j))+(-5.1*fc(i,j))+(2.3574*dtime(i,j)))
endif
if ( dtime(i,j).le.2.8229182.and.day(i,j).gt.292.24063.and.dtime(i,j).gt.2.2083328.and.dtrad(i,j).gt.10.253759) then
 latn(i,j) = (4.0393105+(0.372*dtrad(i,j))+(-0.043*day(i,j))+(0.0048*terrain(i,j))+(-3.3*fc(i,j))+(5.7282*dtime(i,j)))
endif
if ( dtime(i,j).le.3.1489587.and.dtime(i,j).gt.2.8229182.and.day(i,j).le.305.84937.and.day(i,j).gt.292.24063.and.dtrad(i,j).gt.10.253759) then
 latn(i,j) = (9.0047961+(0.35*dtrad(i,j))+(0.035*day(i,j))+(0.0039*terrain(i,j))+(-1.8*fc(i,j))+(-4.4002*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.9833345.and.dtime(i,j).le.4.140626.and.terrain(i,j).le.16.027388.and.day(i,j).gt.295.75366) then
 latn(i,j) = (50.7107678+(0.277*dtrad(i,j))+(-0.013*day(i,j))+(0.1151*terrain(i,j))+(-2.4*fc(i,j))+(-9.3654*dtime(i,j)))
endif
if ( dtime(i,j).gt.4.140626.and.dtime(i,j).le.4.3052063.and.terrain(i,j).le.16.027388.and.dtrad(i,j).gt.10.253759.and.day(i,j).gt.295.75366) then
 latn(i,j) = (49.9680445+(0.457*dtrad(i,j))+(-0.194*day(i,j))+(0.0521*terrain(i,j))+(-5.7*fc(i,j))+(4.0975*dtime(i,j)))
endif
if ( dtime(i,j).gt.4.458334.and.dtrad(i,j).gt.10.253759) then
 latn(i,j) = (-19.2819226+(0.312*dtrad(i,j))+(0.025*day(i,j))+(-0.0128*terrain(i,j))+(-1.7*fc(i,j))+(4.596*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.9833345.and.day(i,j).le.295.75366.and.dtime(i,j).le.4.458334.and.dtrad(i,j).gt.10.253759) then
 latn(i,j) = (-50.0465354+(0.295*dtrad(i,j))+(0.178*day(i,j))+(-0.0082*terrain(i,j))+(1.7963*dtime(i,j)))
endif
if ( day(i,j).gt.305.84937.and.dtime(i,j).le.3.5635402.and.dtime(i,j).gt.2.8229182) then
 latn(i,j) = (-34.7450221+(0.418*dtrad(i,j))+(0.102*day(i,j))+(0.0037*terrain(i,j))+(-3.1*fc(i,j))+(3.0138*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.5635402.and.dtime(i,j).le.3.9833345) then
 latn(i,j) = (-10.6676846+(0.462*dtrad(i,j))+(0.065*day(i,j))+(-0.0139*terrain(i,j))+(-3.6*fc(i,j))+(-0.1185*dtime(i,j)))
endif
if ( terrain(i,j).gt.16.027388.and.dtime(i,j).gt.3.5635402.and.day(i,j).gt.295.75366.and.dtrad(i,j).gt.10.253759) then
 latn(i,j) = (18.2051643+(0.363*dtrad(i,j))+(-0.026*day(i,j))+(-0.0023*terrain(i,j))+(-5.5*fc(i,j))+(0.1302*dtime(i,j)))
endif
if ( dtime(i,j).gt.4.3052063.and.fc(i,j).le.0.62520468.and.dtime(i,j).le.4.458334.and.day(i,j).gt.295.75366.and.terrain(i,j).le.16.027388.and.dtrad(i,j).gt.10.253759) then
 latn(i,j) = (100.7223231+(0.336*dtrad(i,j))+(-0.072*day(i,j))+(0.0026*terrain(i,j))+(-5.6*fc(i,j))+(-15.5303*dtime(i,j)))
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
