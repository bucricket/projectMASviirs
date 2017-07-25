program build_cubist_model

integer, parameter :: dx=3750, dy=3750
real :: latn(dx,dy), dtrad(dx,dy), day(dx,dy), night(dx,dy)
real :: dtime(dx,dy), lai(dx,dy), terrain(dx,dy)
character(len=4) :: arg1,arg3
character(len=3) :: arg2
integer :: year, doy, yyyyddd, riseddd
character(len=255) :: ifile1, ifile2, ifile3, ifile4, ifile5, ifile6, ofile1
real :: predict(dx,dy)
real :: precip(dx,dy), fmax(dx,dy)
real :: fc(dx,dy)
integer :: laiddd, nweek1, cday, offset, rday

call getarg(1,arg1)
call getarg(2,arg2)
call getarg(3,arg3)
read(arg1,'(i4)') year
read(arg2,'(i3)') doy
yyyyddd=year*1000+doy
riseddd=2014*1000+doy

write(ifile1,'(a,a,a)') '/data/VIIRS_GLOBAL_PROCESS/PRECIP/tiles/PRECIP_',arg3,'.dat'
write(ifile2,'(a,a,a)') '/data/VIIRS_GLOBAL_PROCESS/FMAX/tiles/FMAX_',arg3,'.dat'
open(10,file=ifile1,form='unformatted',access='direct',recl=dx*dy*4)
open(11,file=ifile2,form='unformatted',access='direct',recl=dx*dy*4)
read(10,rec=1) precip
read(11,rec=1) fmax
close(10)
close(11)

nweek1=(iday-1)/7.
cday=nweek1*8.
offset=int((iday-cday)/7.)
rday=int(((offset+nweek1)*8)+1)
laiddd=year*1000+rday

write(ifile1,'(a,a,a)') '/data/VIIRS_GLOBAL_PROCESS/TERRAIN_SD/tiles/TERRAIN_',arg3,'.dat'
write(ifile2,'(a,a,a,I7,a)') '/data/VIIRS_GLOBAL_PROCESS/tiles/',arg3,'/FINAL_DAY_LST_',yyyyddd,'.dat'
write(ifile3,'(a,a,a,I7,a)') '/data/VIIRS_GLOBAL_PROCESS/tiles/',arg3,'/FINAL_NIGHT_LST_',yyyyddd,'.dat' 
write(ifile4,'(a,I7,a,a,a)') '/data/VIIRS_GLOBAL_PROCESS/LAI/tiles/MLAI_',laiddd,'_',arg3,'.dat'
write(ifile5,'(a,I7,a,a,a)') '/data/VIIRS_GLOBAL_PROCESS/DTIME/tiles/DTIME_',riseddd,'_',arg3,'.dat'

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

 if (precip(i,j).gt.0.0.and.precip(i,j).lt.250.and.fmax(i,j).ge.0.0.and.fmax(i,j).lt.1.0) then 
 if (day(i,j).ne.-9999.and.night(i,j).ne.-9999.) then
 if (terrain(i,j).ne.-9999.and.lai(i,j).ne.-9999.) then 

  dtrad(i,j) = day(i,j)-night(i,j)
  fc(i,j) = 1.0-exp(-0.5*lai(i,j))

