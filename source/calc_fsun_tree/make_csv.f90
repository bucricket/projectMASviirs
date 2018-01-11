program make_csv

integer, parameter :: dx=1600, dy=900
character(len=255) :: i1, i2, i3, i4, i5, i6, i7, o1
character(len=4) :: arg1, arg2, tile 
character(len=3) :: arg3, arg4
real :: corr(dx,dy), dthr(dx,dy), trad2(dx,dy), rnet(dx,dy), mlai(dx,dy), fsun(dx,dy), precip(dx,dy), fmax(dx,dy)
integer :: counter
integer :: ndays
integer :: iyear, eyear, iday, eday
integer :: yyyyddd
character(len=10) :: arg5, arg6, arg7, arg8, arg9, arg10
integer :: f1, f2, p1, p2, v1, v2
real :: vegt(dx,dy)

call getarg(1,arg1)
call getarg(2,arg2)
call getarg(3,arg3)
call getarg(4,arg4)
call getarg(5,arg5)
call getarg(6,arg6)
call getarg(7,arg7)
call getarg(8,arg8)
call getarg(9,arg9)
call getarg(10,arg10)

read(arg1,'(i4)') iyear
read(arg2,'(i4)') eyear
read(arg3,'(i3)') iday
read(arg4,'(i3)') eday
read(arg5,'(i6)') p1
read(arg6,'(i6)') p2
read(arg7,'(i6)') f1
read(arg8,'(i6)') f2
read(arg9,'(i2)') v1
read(arg10,'(i2)') v2
write(6,*) iyear, eyear, iday, eday
write(6,*) p1, p2, f1, f2, v1, v2
counter=0

i1='/data/PROCESS_VIIRS/STATIC/5KM/FMAX/FMAX.dat'
i2='/data/PROCESS_VIIRS/STATIC/5KM/PRECIP/PRECIP.dat'
open(10,file=i1,form='unformatted',access='direct',recl=dx*dy*4)
open(11,file=i2,form='unformatted',access='direct',recl=dx*dy*4)
read(10,rec=1) fmax
read(11,rec=1) precip
close(10)
close(11)

ndays=(eday-iday)+1
do k = 1, ndays
 yyyyddd=iyear*1000+iday
 write(i1,'(a,I7,a)') '/data/PROCESS_VIIRS/PROCESSING/FSUN_TREES/DTHR',yyyyddd,'.dat'
 write(i2,'(a,I7,a)') '/data/PROCESS_VIIRS/PROCESSING/FSUN_TREES/TRD2',yyyyddd,'.dat'
 write(i4,'(a,I7,a)') '/data/PROCESS_VIIRS/PROCESSING/FSUN_TREES/MLAI',yyyyddd,'.dat'
 write(i5,'(a,I7,a)') '/data/PROCESS_VIIRS/PROCESSING/FSUN_TREES/FSUN',yyyyddd,'.dat'
 write(i6,'(a,I7,a)') '/data/PROCESS_VIIRS/PROCESSING/FSUN_TREES/VEGT',yyyyddd,'.dat'
 write(i7,'(a,i7,a)') '/data/PROCESS_VIIRS/PROCESSING/FSUN_TREES/CORR',yyyyddd,'.dat'
 open(10,file=i1,form='unformatted',access='direct',recl=dx*dy*4)
 open(11,file=i2,form='unformatted',access='direct',recl=dx*dy*4)
 open(13,file=i4,form='unformatted',access='direct',recl=dx*dy*4)
 open(14,file=i5,form='unformatted',access='direct',recl=dx*dy*4)
 open(15,file=i6,form='unformatted',access='direct',recl=dx*dy*4)
 open(16,file=i7,form='unformatted',access='direct',recl=dx*dy*4)
 read(10,rec=1) dthr
 read(11,rec=1) trad2 
 read(13,rec=1) mlai 
 read(14,rec=1) fsun 
 read(15,rec=1) vegt
 read(16,rec=1) corr
 close(10)
 close(11)
 close(13)
 close(14)
 close(15)
 close(16)
 counter=counter+1
 iday=iday+1
enddo

counter=0
write(o1,'(a)') '/data/PROCESS_VIIRS/PROCESSING/FSUN_TREES/tiles_ctl/fsun.data'
open(50,file=o1)
do j = 1, dy
do i = 1, dx
 fmax(i,j) = 1.0-exp(-0.5*fmax(i,j))
 if (precip(i,j).ge.p1.and.precip(i,j).lt.p2.and.fmax(i,j).gt.(f1/100.).and.fmax(i,j).lt.(f2/100.)) then 
 if (vegt(i,j).ge.v1.and.vegt(i,j).le.v2) then
 if (corr(i,j).ne.-9999.and.dthr(i,j).gt.0.0.and.trad2(i,j).gt.0.0.and.fsun(i,j).gt.0.0.and.mlai(i,(dy-j)+1).gt.0.0) then 
!  write(50,'(f9.3,a1,f9.3,a1,f9.3,a1,f9.3,a1,f9.3)') fsun(i,j), ',', dthr(i,j), ',',  rnet(i,j)/dthr(i,j), ',',  rnet(i,j),',', mlai(i,(dy-j)+1)
  write(50,'(f9.3,a1,f9.3,a1,f9.3,a1,f9.3)') fsun(i,j), ',', dthr(i,j)*corr(i,j), ',',  mlai(i,(dy-j)+1), ',', trad2(i,j)
  counter=counter+1
  write(*,*) counter
 endif
 endif
 endif
enddo
enddo

close(50)


end program
