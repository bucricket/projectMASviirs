program merge_day_overpass

integer, parameter :: dx=3750, dy=3750
character(len=255) :: dir, list_files
character(len=4) :: arg2, arg4
character(len=3) :: arg3
integer :: year, doy, yyyyddd
character(len=1) :: arg5
integer :: nfiles
character(len=4), allocatable :: times(:)
integer :: status1
real, allocatable :: lst(:,:,:), view(:,:,:)
character(len=255) :: lstfile, viewfile
real :: temp1(dx,dy), temp2(dx,dy)
real :: final_lst(dx,dy)
real :: final_view(dx,dy)
integer :: test(1), ind
integer :: min_index
real :: min_view

call getarg(1,list_files)
call getarg(2,arg2)
call getarg(3,arg3)
call getarg(4,arg4)
call getarg(5,arg5)
read(arg2,'(i4)') year
read(arg3,'(i3)') doy
read(arg5,'(i1)') nfiles
yyyyddd=year*1000+doy
write(6,*) yyyyddd, nfiles

allocate(times(nfiles),stat=status1)
allocate(lst(dx,dy,nfiles),stat=status1)
allocate(view(dx,dy,nfiles),stat=status1)

open(10,file=trim(list_files)) 
read(10,*) times
close(10)

dir='/raid1/sport/people/chain/VIIRS_PROCESS/TILES/'
do k = 1, nfiles
 write(lstfile,'(a,a,a,I7,a,a,a,a,a)') trim(dir),arg4,'/lst_',yyyyddd,'_',arg4,'_',times(k),'.dat' 
 write(viewfile,'(a,a,a,I7,a,a,a,a,a)') trim(dir),arg4,'/view_angle_',yyyyddd,'_',arg4,'_',times(k),'.dat'
 open(10,file=lstfile,form='unformatted',access='direct',recl=dx*dy*4)
 open(11,file=viewfile,form='unformatted',access='direct',recl=dx*dy*4)
 read(10,rec=1) temp1
 read(11,rec=1) temp2
 close(10)
 close(11)
 write(6,*) lstfile, k
 lst(:,:,k) = temp1(:,:)
 view(:,:,k) = temp2(:,:)
enddo

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

 write(lstfile,'(a,a,a,I7,a,a,a)') trim(dir),arg4,'/FINAL_NIGHT_LST_',yyyyddd,'_',arg4,'.dat'
 write(viewfile,'(a,a,a,I7,a,a,a)') trim(dir),arg4,'/FINAL_NIGHT_VIEW_',yyyyddd,'_',arg4,'.dat'
 open(10,file=lstfile,form='unformatted',access='direct',recl=dx*dy*4)
 open(11,file=viewfile,form='unformatted',access='direct',recl=dx*dy*4)
 write(10,rec=1) final_lst
 write(11,rec=1) final_view
 close(10)
 close(11)

end program
