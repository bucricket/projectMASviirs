program set_grid

real :: lat, lon
integer :: ilat, ilon
integer, parameter :: nx=15000, ny=15000
real :: test(nx,ny)
real :: counter(nx,ny)
character(len=255) :: latfile, lonfile
character(len=255) :: goesfile, glatfile, glonfile,sumfile,countfile
integer :: gx, gy, status
real, allocatable :: temp(:,:)
real, allocatable :: new_temp(:,:)
real, allocatable :: glat(:,:), glon(:,:)
real, allocatable :: glat2(:,:), glon2(:,:)
integer :: ipoint1, ipoint2, jpoint1, jpoint2
integer :: channel, isat
character(len=255) :: i1,i2,arg5, arg6, arg7
character(len=6) :: arg1, arg2
character(len=255) :: arg3, arg4
real :: x1, x2, y1, y2
integer :: pt1, pt2
integer :: m, n 

call getarg(1,arg1)
call getarg(2,arg2)
read(arg1,'(i6)') ilat
read(arg2,'(i6)') ilon
lat=ilat*1.0
lon=ilon*1.0
write(6,*) lat, lon

gx=6400
gy=6144
allocate(temp(gx,gy),stat=status)
allocate(glat(gx,gy),stat=status)
allocate(glon(gx,gy),stat=status)
allocate(glat2(gx,gy),stat=status)
allocate(glon2(gx,gy),stat=status)
call getarg(3,arg3)
call getarg(4,arg4)
call getarg(5,arg5)
call getarg(6,arg6)
call getarg(7,arg7)
goesfile=trim(arg3)
glatfile=trim(arg4)
glonfile=trim(arg5)
sumfile=trim(arg6)
countfile=trim(arg7)

write(6,*) "processing ....", goesfile
open(20,file=goesfile,form='unformatted',access='direct',recl=gx*gy*4)
open(21,file=glatfile,form='unformatted',access='direct',recl=gx*gy*4)
open(22,file=glonfile,form='unformatted',access='direct',recl=gx*gy*4)
read(20,rec=1) temp
read(21,rec=1) glat
read(22,rec=1) glon
close(20)
close(21)
close(22)

!new_temp(:,:) = -9999.
npix=0.0

do j = 2, gy-2 
do i = 2, gx-2
 if (glat(i,j+1).gt.ilat.and.glat(i,j).gt.ilat.and.glon(i+1,j).gt.ilon.and.glon(i,j).gt.ilon) then
  ipoint1=0
  ipoint2=0
  jpoint1=0
  jpoint2=0
  y1=(glat(i,j-1)+glat(i,j))/2.
  y2=(glat(i,j)+glat(i,j+1))/2.
  x1=(glon(i-1,j)+glon(i,j))/2.
  x2=(glon(i,j)+glon(i+1,j))/2.

  jpoint1=nint((glat(i,j)-ilat)/0.001)
  ipoint1=nint((glon(i,j)-ilon)/0.001)
  jpoint2=nint((glat(i,j+1)-ilat)/0.001)
  ipoint2=nint((glon(i+1,j)-ilon)/0.001)
  diff1=glon(i,j)-ilon
  diff2=glat(i,j+1)-ilat
  diff3=glon(i+1,j)-ilon
  diff4=glat(i,j)-ilat
  if (ipoint1.gt.1.and.ipoint2.gt.1.and.jpoint1.gt.1.and.jpoint2.gt.1) then
  if (ipoint1.le.nx-1.and.ipoint2.le.nx-1.and.jpoint1.le.ny-1.and.jpoint2.le.ny-1) then
   do n = jpoint2, jpoint1
   do m = ipoint1+1, ipoint2+1
    if (temp(i,j).lt.65000) then 
    test(m,n) = test(m,n)+temp(i,j)
    counter(m,n) = counter(m,n)+1
    npix=npix+1.0
    endif
   enddo
   enddo  
  endif
  endif
 endif
enddo
enddo

write(6,*) "pixels in scene ", npix
if (npix.gt.0) then
open(50,file=sumfile,form='unformatted',access='direct',recl=nx*ny*4)
write(50,rec=1) test !trad
close(50)
open(50,file=countfile,form='unformatted',access='direct',recl=nx*ny*4)
write(50,rec=1) counter !trad
close(50)
endif


end program
