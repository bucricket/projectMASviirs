program calc_offset_correction
!character (len=*), allocatable :: basedir
character(len=4) :: tile_num
integer, parameter :: dx=3750, dy=3750
character(len=400) :: dayfile, outdayfile, dtimefile, ntimefile
character(len=400) :: avgfile, dztimefile, nztimefile
character(len=255) :: arg4, basedir
character(len=4) :: arg1, arg3
character(len=3) :: arg2
integer :: iyear, iday, time
integer :: yyyyddd, avgddd
real :: day_lst(dx,dy), night_lst(dx,dy), dtrad_avg(dx,dy)
integer(kind=1) :: day_time(dx,dy), night_time(dx,dy)
real :: day_minus_coeff(6), day_plus_coeff(6), night_minus_coeff(6), night_plus_coeff(6)
real :: day_minus_b(6), day_plus_b(6), night_minus_b(6), night_plus_b(6)
real :: dztime(dx,dy), nztime(dx,dy)
real :: day_corr(dx,dy), night_corr(dx,dy)
integer :: tindex1, tindex2
real :: w1, w2, c1, c2
integer :: nweek1, cday, offset, rday

day_minus_coeff(:) = (/0.0504,1.384,2.415,3.586,4.475,4.455/)
day_plus_coeff(:) = (/0.280,1.51,3.034,3.234,1.905,1.049/)
night_minus_coeff(:) = (/0.114,0.261,0.423,0.621,0.836,1.178/)
night_plus_coeff(:) = (/0.081,0.011,-0.042,0.468,2.426,3.989/)

day_minus_b(:) = (/-0.023,0.003,0.088,0.221,0.397,0.606/)
day_plus_b(:) = (/0.058,0.143,0.262,0.439,0.634,0.751/)
night_minus_b(:) = (/-0.024,-0.052,-0.085,-0.122,-0.166,-0.229/)
night_plus_b(:) = (/0.0158,0.036,0.052,0.0213,-0.1399,-0.3673/)

call getarg(1,arg1)
call getarg(2,arg2)
call getarg(3,arg3)
call getarg(4,tile_num)
call getarg(5,dztimefile)
call getarg(6,dayfile)
call getarg(7,avgfile)
call getarg(8,outdayfile)
read(arg1,'(i4)') iyear
read(arg2,'(i3)') iday
read(arg3,'(i4)') time
!basedir=trim(arg4)

write(6,*) iyear, iday, time, tile_num, basedir
yyyyddd=iyear*1000+iday
avgddd=2014*1000+iday

ctime=time/100.
nweek1=(iday-1)/7.
cday=nweek1*7.
offset=int((iday-cday)/7.)
rday=int(((offset+nweek1)*7)+1)
write(6,*) yyyyddd, rday

!write(dztimefile,'(a,a,a,a)') basedir,'/PROCESSING/overpass_corr/CURRENT_DAY_ZTIME_',tile_num,'.dat'
open(10,file=dztimefile,form='unformatted',access='direct',recl=dx*dy*4)
read(10,rec=1) dztime
close(10)

!write(dayfile,'(a,a,a,a)') basedir,'/PROCESSING/overpass_corr/RAW_TRAD1_',tile_num,'.dat'
!write(avgfile,'(a,a,a,a)') basedir,'/PROCESSING/overpass_corr/CURRENT_DTRAD_AVG_',tile_num,'.dat'

open(10,file=dayfile,form='unformatted',access='direct',recl=dx*dy*4)
open(14,file=avgfile,form='unformatted',access='direct',recl=dx*dy*4)
read(10,rec=1) day_lst
read(14,rec=1) dtrad_avg
close(10)
close(14)

day_corr(:,:) = -9999.
do j = 1, dy 
do i = 1, dx 
if (day_lst(i,j).ne.-9999..and.dtrad_avg(i,j).ne.-9999.) then 
 tdiff_day=(ctime)-(dztime(i,j))
 tindex1=int(abs(tdiff_day))
 tindex2=tindex1+1
 w2=(abs(tdiff_day)-tindex1)
 w1=(1.0-w2)
 if (tindex1.gt.0.0) then
  c1=day_minus_coeff(tindex1)+(day_minus_b(tindex1)*dtrad_avg(i,j))
  c2=day_minus_coeff(tindex2)+(day_minus_b(tindex2)*dtrad_avg(i,j))
  day_corr(i,j) = day_lst(i,j)+(c1*w1+c2*w2)
 endif
 if (tindex1.eq.0.0) then
  c1=day_minus_coeff(tindex1)+(day_minus_b(tindex1)*dtrad_avg(i,j))
  c2=day_minus_coeff(tindex2)+(day_minus_b(tindex2)*dtrad_avg(i,j))
  day_corr(i,j) = day_lst(i,j)+c2*w2
 endif
endif

enddo
enddo

!write(dayfile,'(a,a,a)') basedir,'/PROCESSING/overpass_corr/TRAD1_',tile_num
open(10,file=outdayfile,form='unformatted',access='direct',recl=dx*dy*4)
write(10,rec=1) day_corr
close(10)

end program

