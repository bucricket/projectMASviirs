program merge_day_overpass

integer, parameter :: dx=3750, dy=3750
!character (len=:), allocatable :: dir, arg6
character(len=450) :: lstfiles,viewfiles
character(len=450) :: outlstfile, outviewfile
character(len=4) :: arg2
character(len=3) :: arg3
character(len=1) :: arg4
integer :: nfiles, doy, year, yyyyddd
!character(len=450), allocatable :: lstfiles(:), viewfiles(:)
integer :: status1
real, allocatable :: lst(:,:,:), view(:,:,:)
character(len=450) :: lstfile, viewfile
real :: temp1(dx,dy), temp2(dx,dy)
real :: final_lst(dx,dy)
real :: final_view(dx,dy)
integer :: test(1), ind
integer :: min_index
real :: min_view

call getarg(1,lstfiles)
call getarg(2,viewfiles)
call getarg(3,arg4)
call getarg(4,outlstfile)
call getarg(5,outviewfile)
read(arg4,'(i1)') nfiles



!allocate(lstfiles(nfiles),stat=status1)
!allocate(viewfiles(nfiles),stat=status1)
allocate(lst(dx,dy,nfiles),stat=status1)
allocate(view(dx,dy,nfiles),stat=status1)


open(10,file=lstfiles,form='unformatted',access='direct',recl=dx*dy*4*nfiles)
open(11,file=viewfiles,form='unformatted',access='direct',recl=dx*dy*4*nfiles)
read(10,rec=nfiles) lst
read(11,rec=nfiles) view
close(10)
close(11)

write(6,*) lst(2500,1500,:), view(2500,1500,:)
write(6,*) minloc(view(2500,1500,:),mask=view(2500,1500,:).ne.-9999.)
final_lst(:,:) = -9999.
final_view(:,:) = -9999.
do j = 1, dy
do i = 1, dx
 min_view=90.
 min_index=0
 do k = 1, nfiles
  if (view(i,j,k).lt.min_view.and.view(i,j,k).ne.-9999.) then
   min_view=view(i,j,k)
   min_index=k
  endif
 enddo
 if (min_index.gt.0.and.min_view.le.60.) then  
  final_lst(i,j) = lst(i,j,min_index)
  final_view(i,j) = view(i,j,min_index)
 endif 
enddo
enddo
write(6,*) final_lst(2500,1500), final_view(2500,1500)

 !write(lstfile,'(a,a,a,I7,a,a,a)') trim(dir),arg4,'/FINAL_DAY_LST_',yyyyddd,'_',arg4,'.dat'
 !write(viewfile,'(a,a,a,I7,a,a,a)') trim(dir),arg4,'/FINAL_DAY_VIEW_',yyyyddd,'_',arg4,'.dat'
 open(10,file=outlstfile,form='unformatted',access='direct',recl=dx*dy*4)
 open(11,file=outviewfile,form='unformatted',access='direct',recl=dx*dy*4)
 write(10,rec=1) final_lst
 write(11,rec=1) final_view
 close(10)
 close(11)

end program
