program merge_dtrad

integer, parameter :: dx=3750, dy=3750
character(len=400) :: f1, f2, f3, f4, f5, f6, out_fn
real :: dtrad1(dx,dy), dtrad2(dx,dy), dtrad3(dx,dy)
real :: dtrad4(dx,dy), dtrad5(dx,dy), dtrad6(dx,dy)
real :: final_dtrad(dx,dy)


call getarg(1,f1)
call getarg(2,f2)
call getarg(3,f3)
call getarg(4,f4)
call getarg(5,f5)
call getarg(6,f6)
call getarg(7,out_fn)


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

open(10,file=out_fn,form='unformatted',access='direct',recl=dx*dy*4)
write(10,rec=1) final_dtrad
close(20)

end program
