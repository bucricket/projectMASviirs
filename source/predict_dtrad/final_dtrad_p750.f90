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

 if (precip(i,j).gt.500.0.and.precip(i,j).lt.750.and.fmax(i,j).ge.0.0.and.fmax(i,j).lt.1.0) then 
 if (day(i,j).ne.-9999.and.night(i,j).ne.-9999.) then
 if (terrain(i,j).ne.-9999.and.lai(i,j).ne.-9999.) then 

  dtrad(i,j) = day(i,j)-night(i,j)
  fc(i,j) = 1.0-exp(-0.5*lai(i,j))

if ( dtime(i,j).le.1.6166668) then
 latn(i,j) = (-17.5078859+(0.157*dtrad(i,j))+(0.074*day(i,j)))
endif
if ( dtrad(i,j).le.8.1104765) then
 latn(i,j) = (-20.2467188+(0.718*dtrad(i,j))+(0.071*day(i,j))+(0.0019*terrain(i,j))+(1.7*fc(i,j)))
endif
if ( day(i,j).le.291.34402.and.dtime(i,j).le.2.3645842.and.dtime(i,j).gt.1.6166668.and.dtrad(i,j).gt.8.1104765) then
 latn(i,j) = (-30.3077934+(0.166*dtrad(i,j))+(0.104*day(i,j))+(-0.0061*terrain(i,j))+(-4.2*fc(i,j))+(3.7586*dtime(i,j)))
endif
if ( dtime(i,j).le.2.3645842.and.day(i,j).gt.291.34402.and.dtrad(i,j).gt.8.1104765) then
 latn(i,j) = (12.0230043+(0.289*dtrad(i,j))+(-0.071*day(i,j))+(-0.0014*terrain(i,j))+(-3.1*fc(i,j))+(6.6909*dtime(i,j)))
endif
if ( dtime(i,j).le.2.9416656.and.dtime(i,j).gt.2.3645842.and.day(i,j).le.303.37073.and.dtrad(i,j).gt.8.1104765) then
 latn(i,j) = (-22.7056189+(0.309*dtrad(i,j))+(0.065*day(i,j))+(-3.2*fc(i,j))+(4.1177*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.85833.and.day(i,j).le.301.23203.and.dtime(i,j).le.5.0583344.and.dtrad(i,j).gt.8.1104765.and.fc(i,j).gt.0.14021321) then
 latn(i,j) = (4.9784885+(0.45*dtrad(i,j))+(0.014*day(i,j))+(0.003*terrain(i,j))+(-2.9*fc(i,j))+(-0.3891*dtime(i,j)))
endif
if ( terrain(i,j).le.1.7700616.and.dtime(i,j).gt.3.625001.and.dtime(i,j).le.4.5749989.and.fc(i,j).gt.0.14021321.and.dtrad(i,j).gt.8.1104765) then
 latn(i,j) = (24.2913078+(0.485*dtrad(i,j))+(-0.054*day(i,j))+(1.5838*terrain(i,j))+(-2.2*fc(i,j))+(-0.8878*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.625001.and.dtime(i,j).le.3.85833.and.day(i,j).le.301.23203.and.dtrad(i,j).gt.8.1104765) then
 latn(i,j) = (-28.6255914+(0.309*dtrad(i,j))+(0.172*day(i,j))+(-0.0027*terrain(i,j))+(-5.2*fc(i,j))+(-2.9856*dtime(i,j)))
endif
if ( dtime(i,j).le.3.2843757.and.dtime(i,j).gt.2.9416656) then
 latn(i,j) = (-22.3601227+(0.343*dtrad(i,j))+(0.055*day(i,j))+(0.0056*terrain(i,j))+(-3*fc(i,j))+(4.1109*dtime(i,j)))
endif
if ( dtime(i,j).le.3.625001.and.dtime(i,j).gt.3.2843757.and.fc(i,j).gt.0.19698051) then
 latn(i,j) = (-13.3128123+(0.351*dtrad(i,j))+(0.076*day(i,j))+(0.001*terrain(i,j))+(-6.4*fc(i,j))+(0.0377*dtime(i,j)))
endif
if ( dtime(i,j).le.2.8270838.and.day(i,j).gt.303.37073.and.dtime(i,j).gt.2.3645842.and.fc(i,j).gt.0.1078644) then
 latn(i,j) = (31.6200937+(0.418*dtrad(i,j))+(-0.173*day(i,j))+(0.0104*terrain(i,j))+(-4.4*fc(i,j))+(10.1697*dtime(i,j)))
endif
if ( dtime(i,j).le.2.9416656.and.dtime(i,j).gt.2.8270838.and.fc(i,j).gt.0.1078644.and.dtrad(i,j).gt.8.1104765) then
 latn(i,j) = (-22.6457859+(0.383*dtrad(i,j))+(-0.012*day(i,j))+(-2.4*fc(i,j))+(11.4048*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.8645823.and.dtime(i,j).le.4.5749989.and.day(i,j).le.309.66037.and.fc(i,j).gt.0.14021321.and.dtrad(i,j).gt.8.1104765) then
 latn(i,j) = (12.4218864+(0.424*dtrad(i,j))+(-3.7*fc(i,j))+(-0.9347*dtime(i,j)))
endif
if ( dtime(i,j).gt.5.0583344) then
 latn(i,j) = (87.6384564+(0.782*dtrad(i,j))+(-0.32*day(i,j))+(-0.0063*terrain(i,j))+(1.9231*dtime(i,j)))
endif
if ( fc(i,j).le.0.1078644.and.dtime(i,j).gt.2.3645842.and.dtime(i,j).le.2.9416656.and.dtrad(i,j).gt.8.1104765) then
 latn(i,j) = (-14.7505907+(0.343*dtrad(i,j))+(0.033*day(i,j))+(-0.0044*terrain(i,j))+(12*fc(i,j))+(3.7619*dtime(i,j)))
endif
if ( dtime(i,j).gt.4.5749989.and.day(i,j).gt.301.23203.and.dtime(i,j).le.5.0583344.and.fc(i,j).gt.0.14021321) then
 latn(i,j) = (0.4728077+(0.343*dtrad(i,j))+(0.063*day(i,j))+(-0.0023*terrain(i,j))+(-7.2*fc(i,j))+(-1.5034*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.625001.and.dtime(i,j).le.3.8645823.and.day(i,j).gt.301.23203.and.day(i,j).le.309.66037.and.terrain(i,j).gt.1.7700616.and.fc(i,j).gt.0.14021321) then
 latn(i,j) = (46.6331432+(0.315*dtrad(i,j))+(-0.011*terrain(i,j))+(-5.6*fc(i,j))+(-9.0781*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.2843757.and.fc(i,j).le.0.19698051.and.dtime(i,j).le.3.625001.and.dtrad(i,j).gt.8.1104765) then
 latn(i,j) = (17.2468131+(0.299*dtrad(i,j))+(0.041*day(i,j))+(-0.0149*terrain(i,j))+(-28.3*fc(i,j))+(-4.4746*dtime(i,j)))
endif
if ( day(i,j).gt.309.66037.and.dtime(i,j).gt.3.625001.and.dtime(i,j).le.4.5749989.and.terrain(i,j).gt.1.7700616.and.fc(i,j).gt.0.14021321) then
 latn(i,j) = (15.2625462+(0.38*dtrad(i,j))+(-5*fc(i,j))+(-1.2988*dtime(i,j)))
endif
if ( fc(i,j).le.0.14021321.and.dtime(i,j).gt.3.625001) then
 latn(i,j) = (28.2513252+(0.621*dtrad(i,j))+(-0.073*day(i,j))+(-0.0181*terrain(i,j))+(-19.3*fc(i,j))+(0.0235*dtime(i,j)))
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
