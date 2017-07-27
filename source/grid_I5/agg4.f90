program read_modis_1km

integer, parameter :: dx=15000, dy=15000
real :: lai(dx,dy)
real :: ilai(dx,dy)
real :: counter(dx,dy)
character(len=400) :: file2, file1, ofile
integer, parameter :: ax=3750, ay=3750
!integer (kind=1) :: agg(ax,ay)
real :: agg(ax,ay)
integer :: x1, x2, y1, y2
real :: sum1, count1

call getarg(1,file1)
call getarg(2,file2)
call getarg(3,ofile)


open(10,file=file1,form='unformatted',access='direct',recl=dx*dy*4)
read(10,rec=1) ilai
close(10)
open(10,file=file2,form='unformatted',access='direct',recl=dx*dy*4)
read(10,rec=1) counter
close(10)

write(6,*) "processing ...", file1
agg(:,:) = -9999.
do j = 1, ay
do i = 1, ax
 sum1=0.0
 count1=0.0
 x1=(i-1)*4+1
 x2=(i-1)*4+5
 y1=(j-1)*4+1
 y2=(j-1)*4+5
 do m = y1, y2
 do k = x1, x2
  if (ilai(k,m).ne.-9999.and.ilai(k,m).gt.0.0) then
   sum1=sum1+((ilai(k,m)/counter(k,m))*0.00351+150.0)
   count1=count1+1
  endif
 enddo
 enddo
 if (count1.ge.1) then 
  agg(i,j) = sum1/count1
 endif
enddo
enddo


!write(file1,'(a,a,a)') arg3
open(20,file=ofile,form='unformatted',access='direct',recl=ax*ay*4)
write(20,rec=1) agg 
close(20)


end program
