program merge_dtrad

integer, parameter :: dx=3750, dy=3750
character(len=255) :: f1, f2, f3, f4, f5, f6, o1
real :: dtrad1(dx,dy), dtrad2(dx,dy), dtrad3(dx,dy)
real :: dtrad4(dx,dy), dtrad5(dx,dy), dtrad6(dx,dy)
real :: final_dtrad(dx,dy)
character(len=4) :: arg1, arg3
character(len=3) :: arg2
character(len=255) :: arg4, basedir
!character (len=:), allocatable :: basedir
integer :: iyear, iday
integer :: yyyyddd

call getarg(1,arg1)
call getarg(2,arg2)
call getarg(3,arg3)
call getarg(3,arg3)
read(arg1,'(i4)') year
read(arg2,'(i3)') doy
basedir=trim(arg4)

write(f1,'(a,a,a,a)') basedir,'/PROCESSING/DTRAD_PREDICTION/comp1_',arg3,'.dat'
write(f2,'(a,a,a,a)') basedir,'/PROCESSING/DTRAD_PREDICTION/comp2_',arg3,'.dat'
write(f3,'(a,a,a,a)') basedir,'/PROCESSING/DTRAD_PREDICTION/comp3_',arg3,'.dat'
write(f4,'(a,a,a,a)') basedir,'/PROCESSING/DTRAD_PREDICTION/comp4_',arg3,'.dat'
write(f5,'(a,a,a,a)') basedir,'/PROCESSING/DTRAD_PREDICTION/comp5_',arg3,'.dat'
write(f6,'(a,a,a,a)') basedir,'/PROCESSING/DTRAD_PREDICTION/comp6_',arg3,'.dat'

open(10,file=f1,form='unformatted',access='direct',recl=dx*dy*4)
open(11,file=f2,form='unformatted',access='direct',recl=dx*dy*4)
open(12,file=f3,form='unformatted',access='direct',recl=dx*dy*4)
open(13,file=f4,form='unformatted',access='direct',recl=dx*dy*4)
open(14,file=f5,form='unformatted',access='direct',recl=dx*dy*4)
open(15,file=f6,form='unformatted',access='direct',recl=dx*dy*4)
read(10,rec=1) dtrad1
read(11,rec=1) dtrad2
read(12,rec=1) dtrad3
read(13,rec=1) dtrad4
read(14,rec=1) dtrad5
read(15,rec=1) dtrad6
close(10)
close(11)
close(12)
close(13)
close(14)
close(15)

final_dtrad(:,:) = -9999.
do j = 1, dy
do i = 1, dx
 if (dtrad1(i,j).ne.-9999.) then
  final_dtrad(i,j) = dtrad1(i,j) 
 endif
 if (dtrad2(i,j).ne.-9999.) then
  final_dtrad(i,j) = dtrad2(i,j)
 endif
 if (dtrad3(i,j).ne.-9999.) then
  final_dtrad(i,j) = dtrad3(i,j)
 endif
 if (dtrad4(i,j).ne.-9999.) then
  final_dtrad(i,j) = dtrad4(i,j)
 endif
 if (dtrad5(i,j).ne.-9999.) then
  final_dtrad(i,j) = dtrad5(i,j)
 endif
 if (dtrad6(i,j).ne.-9999.) then
  final_dtrad(i,j) = dtrad6(i,j)
 endif
enddo
enddo

yyyyddd=iyear*1000+iday
write(o1,'(a,a,a,a,I7,a,a,a)') basedir,'/TILES/',arg3,'/FINAL_DTRAD_',yyyyddd,'_',arg3,'.dat'
open(10,file=o1,form='unformatted',access='direct',recl=dx*dy*4)
write(10,rec=1) final_dtrad
close(20)

end program
