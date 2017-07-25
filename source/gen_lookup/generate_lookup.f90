program regridder

integer, parameter :: ax=3750, ay=3750
integer, parameter :: gx=2880, gy=1200
real :: mask(gx,gy), alat(ax,ay), alon(ax,ay)
real :: goes(gx,gy)
real :: glat(gx,gy), glon(gx,gy)
integer :: alexi_goes_i(ax,ay), alexi_goes_j(ax,ay)
character(len=255) :: maskfile,goes_file, alatfile, alonfile
character(len=255) :: alexi_lon_file
character(len=255) :: goes_lat, goes_lon
character(len=255) :: outfile1, outfile2
real :: ilat, ilon
real :: dlat
real :: lat1, lon1
real :: diflon, diflat, dist
integer :: regrid_i, regrid_j
real :: min_dist
real :: alexi_goes_data(ax,ay)
character(len=8) arg1, arg2
character(len=4) arg3

call getarg(1,arg1)
call getarg(2,arg2)
call getarg(3,arg3)
read(arg1,'(f8.0)') ilat
read(arg2,'(f8.0)') ilon
write(6,*) "ILAT ILON = ", ilat, ilon

write(outfile1,'(a,a,a)') 'CFSR_',arg3,'_lookup_icoord.dat'
write(outfile2,'(a,a,a)') 'CFSR_',arg3,'_lookup_jcoord.dat'

write(6,*) outfile1, outfile2

do j = 1, ay
do i = 1, ax
 alat(i,j) = ilat
 if (i.eq.1) then
  alon(i,j) = ilon
 endif
 if (i.gt.1) then
  alon(i,j) = alon(i-1,j)+0.004
 endif
enddo
 ilat=ilat+0.004
enddo

write(6,*) alat(1,1), alat(ax,ay)
write(6,*) alon(1,1), alon(ax,ay)

ilat=-60.0
ilon=-180.00
do j = 1, gy
do i = 1, gx
 glat(i,j) = ilat
 if (i.eq.1) then
  glon(i,j) = ilon
 endif
 if (i.gt.1) then
  glon(i,j) = glon(i-1,j)+0.125
 endif
enddo
 ilat=ilat+0.125
 ilon=-180.00
enddo



alexi_goes_data(:,:) = -9999.
alexi_goes_i(:,:) = -9999.
alexi_goes_j(:,:) = -9999.

! write(6,*) alon(1,1), alat(1,1), glat(160,100), glon(160,100)
 
do j = 1, ay ! Value representative of GOES-EAST Sounder 
do i = 1, ax ! Lat = 24.29 to 52.18; Lon = -124.24 to -66.53

 if (i.eq.1) then
  check=0
 endif
 
 lat1=alat(i,j)
 lon1=alon(i,j) 
 regrid_i=0
 regrid_j=0
 min_dist=0.6

!############################################################
!
 if (check.eq.0) then

  do n = 1, gy
  do m = 1, gx

   diflat=abs(glat(m,n)-lat1)
   diflon=abs(glon(m,n)-lon1)
   dist=sqrt(diflat*diflat+diflon*diflon)
   if (dist.lt.min_dist) then
    min_dist=dist
    regrid_i=m
    regrid_j=n
    prior_i = m
    prior_j = n
   endif

  enddo
  enddo

  if (min_dist.lt.0.50.and.regrid_i.ne.0.and.regrid_j.ne.0) then
   alexi_goes_i(i,j) = regrid_i
   alexi_goes_j(i,j) = regrid_j
   check=1
   prior_i = regrid_i
   prior_j = regrid_j
!   write(20,*) i, j, alexi_goes_i(i,j), alexi_goes_j(i,j), min_dist
!   write(21,*) i, j, lat1, glat(regrid_i,regrid_j), lon1, glon(regrid_i,regrid_j)
  endif


! endif
!#############################################################

!#############################################################
!
 else if (check.ne.0) then 

!  prior_i = alexi_goes_i(i-1,j)
!  prior_j = alexi_goes_j(i-1,j) 

  min_dist=0.6
  do n = prior_j-4, prior_j+4
  do m = prior_i-4, prior_i+4
  
   diflat=abs(glat(m,n)-lat1)
   diflon=abs(glon(m,n)-lon1)
   dist=sqrt(diflat*diflat+diflon*diflon)
   if (dist.lt.min_dist) then
    min_dist=dist
    regrid_i=m
    regrid_j=n
    prior_i = m
    prior_j = n
   endif
  
  enddo
  enddo

  if (min_dist.lt.0.5.and.regrid_i.ne.0.and.regrid_j.ne.0) then
   alexi_goes_i(i,j) = regrid_i
   alexi_goes_j(i,j) = regrid_j
   check=1
   prior_i = regrid_i
   prior_j = regrid_j
!   write(21,*) i, j, lat1, glat(regrid_i,regrid_j), lon1, glon(regrid_i,regrid_j)
  endif

 endif
!##############################################################


enddo
enddo


open(50,file=outfile1,form='unformatted',access='direct',recl=ax*ay*4)
open(51,file=outfile2,form='unformatted',access='direct',recl=ax*ay*4)
write(50,rec=1) alexi_goes_i
write(51,rec=1) alexi_goes_j
close(50)
close(51)


end program
