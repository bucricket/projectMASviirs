program calc_predicted_dtrad

integer, parameter :: dx=3750, dy=3750
integer :: i, j
real, parameter :: c1=-9.6463
real, parameter :: c1x1=-0.183506 ! DAY-NIGHT
real, parameter :: c1x2=1.04281    ! DAY
real, parameter :: c1x3=-0.0529513

character(len=400) :: nightfile, laifile, dayfile, outfile 
logical :: exists1, exists2, exists3
real :: lai(dx,dy), day(dx,dy)
real :: night(dx,dy), model(dx,dy)

call getarg(1,nightfile)
call getarg(2,dayfile)
call getarg(3,laifile)
call getarg(4,outfile)

inquire(file=dayfile,exist=exists1)
inquire(file=nightfile,exist=exists2)
inquire(file=laifile,exist=exists3)

 write(6,*) exists1, exists2, exists3
 if (exists1.eqv..TRUE..and.exists2.eqv..TRUE..and.exists3.eqv..TRUE.) then
  open(10,file=nightfile,form='unformatted',access='direct',recl=dx*dy*4)
  open(11,file=dayfile,form='unformatted',access='direct',recl=dx*dy*4)
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
 open(50,file=outfile,form='unformatted',access='direct',recl=dx*dy*4)
 write(50,rec=1) model
 close(50)


end program
