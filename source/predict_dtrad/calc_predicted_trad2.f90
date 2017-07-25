program calc_predicted_dtrad

integer, parameter :: dx=3750, dy=3750
integer :: year, eyear
integer :: doy, eday
integer :: i, j

! FMAX < 0.10
real, parameter :: c1=-9.6463
real, parameter :: c1x1=-0.183506 ! DAY-NIGHT
real, parameter :: c1x2=1.04281    ! DAY
real, parameter :: c1x3=-0.0529513

character(len=255) :: risefile, rise2file
character(len=255) :: snowfile, viewfile, dtradfile, laifile, fmaxfile, modelfile, dayfile 
character(len=15) :: cmd, zip
character(len=255) :: dir, cmd1, cmd2, cmd3, cmd4, omd1, omd2, omd3, omd4, omd5
character(len=255) :: arg4, basedir
!character (len=:), allocatable :: basedir
integer :: yyyyddd
integer :: rise
integer :: riseddd
logical :: exists1, exists2, exists3, exists4
real :: lai(dx,dy), fmax(dx,dy), day(dx,dy), fcov(dx,dy)
real :: night(dx,dy), model(dx,dy), dtime(dx,dy)
integer :: ipt, jpt
character(len=3) :: arg2
character(len=4) :: arg1, arg3
integer :: laiddd, nweek1, cday, offset, rday

cmd='gunzip -ff '
zip='gzip -ff '

call getarg(1,arg1)
call getarg(2,arg2)
call getarg(3,arg3)
call getarg(3,arg3)
read(arg1,'(i4)') year
read(arg2,'(i3)') doy
basedir=trim(arg4)

yyyyddd=year*1000+doy 
riseddd=2014*1000+doy
nweek1=(iday-1)/7.
cday=nweek1*8.
offset=int((iday-cday)/7.)
rday=int(((offset+nweek1)*8)+1)
laiddd=year*1000+rday
write(6,*) yyyyddd

write(dtradfile,'(a,a,a,a,I7,a,a,a)') basedir,'/TILES/',arg3,'/FINAL_DAY_LST_',yyyyddd,'_',arg3,'.dat' 
write(dayfile,'(a,a,a,a,I7,a,a,a)') basedir,'/TILES',arg3,'/FINAL_NIGHT_LST_',yyyyddd,'_',arg3,'.dat'
write(laifile,'(a,a,I7,a,a,a)') basedir,'/LAI/tiles/MLAI_',laiddd,'_',arg3,'.dat' 

inquire(file=dayfile,exist=exists1)
inquire(file=dtradfile,exist=exists2)
inquire(file=laifile,exist=exists3)

 write(6,*) exists1, exists2, exists3
 if (exists1.eqv..TRUE..and.exists2.eqv..TRUE..and.exists3.eqv..TRUE.) then
  open(10,file=dayfile,form='unformatted',access='direct',recl=dx*dy*4)
  open(11,file=dtradfile,form='unformatted',access='direct',recl=dx*dy*4)
  open(12,file=laifile,form='unformatted',access='direct',recl=dx*dy*4)
  read(10,rec=1) night 
  read(11,rec=1) day 
  read(12,rec=1) lai 
  close(10)
  close(11)
  close(12)
 endif
 if (exists1.eqv..FALSE..or.exists2.eqv..FALSE..or.exists3.eqv..FALSE.) then
  day(:,:) = -9999.
  night(:,:) = -9999.
 endif

 write(6,*) day(251,251), night(251,251), lai(251,251)
 model(:,:) = -9999.
 do j = 1, dy
 do i = 1, dx
  if (day(i,j)-night(i,j).gt.0.and.day(i,j).gt.0.and.night(i,j).gt.0.and.lai(i,j).ge.0) then 
    model(i,j) = c1+(c1x1*(day(i,j)-night(i,j)))+(c1x2*day(i,j))+(c1x3*(lai(i,j)))
  endif
 enddo
 enddo

 write(6,*) model(251,251) 
 write(modelfile,'(a,a,a,a,I7,a,a,a)') basedir,'/TILES/',arg3,'/FINAL_DAY_LST_TIME2_',yyyyddd,'_',arg3,'.dat' 
 open(50,file=modelfile,form='unformatted',access='direct',recl=dx*dy*4)
 write(50,rec=1) model
 close(50)

 doy=doy+1


end program
