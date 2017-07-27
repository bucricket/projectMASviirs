program mask_water_cloud

integer, parameter :: ax=3750, ay=3750
character(len=400) :: file1, file2, file3, file4
real :: bt(ax,ay), cloud(ax,ay), water(ax,ay)

call getarg(1,file1)
call getarg(2,file2)
call getarg(3,file3)
call getarg(4,file4)


open(10,file=file1,form='unformatted',access='direct',recl=ax*ay*4)
read(10,rec=1) bt
close(10)
open(10,file=file2,form='unformatted',access='direct',recl=ax*ay*4)
read(10,rec=1) cloud 
close(10)
open(10,file=file3,form='unformatted',access='direct',recl=ax*ay*4)
read(10,rec=1) water 
close(10)

do j = 1, ay
do i = 1, ax
 if (water(i,j).gt.99.or.cloud(i,j).eq.1) then
  bt(i,j) = -9999.
 endif
enddo
enddo

!write(file1,'(a,a,a)') '/raid1/sport/people/chain/VIIRS_PROCESS/bt_flag_',arg4,'.dat'
open(10,file=file4,form='unformatted',access='direct',recl=ax*ay*4)
write(10,rec=1) bt
close(10)

end program
