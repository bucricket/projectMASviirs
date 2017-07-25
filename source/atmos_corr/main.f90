program main

use corr_parms
use corr_arrays

character(len=7) :: yyyyddd
character(len=4) :: arg4, arg2, tile_num
integer :: time, year
integer :: switch
character (len=:), allocatable :: basedir
character(len=255) :: iptfile, jptfile, arg5, outfile, arg6
character(len=255) :: tprof, qprof, tsfcfile, presfile, qsfcfile
integer :: tindex
character(len=255) :: ifile

call getarg(1,tile_num)
call getarg(2,arg2)
call getarg(3,yyyyddd)
call getarg(4,arg4)
call getarg(5,arg5)
call getarg(6,arg6)
read(arg2,'(i4)') time
read(arg4,'(i4)') year
basedir=trim(arg5)
outfile=trim(arg6)

pres(:) = (/1000,975,950,925,900,850,800,750,700,650,600,550,500,450,400,350,300,250,200,150,100/)

tindex=int(((time/300)+1)*300)
if (tindex.eq.2400) then
 tindex=2100
endif

if (tindex.ge.0.and.tindex.lt.1000) then
 write(tprof,'(a,a,I4,a,a,a)') basedir,'/PROCESSING/CFSR/output/',year,'/temp_profile_0',yyyyddd,'_0000.dat'
 write(qprof,'(a,a,I4,a,a,a)') basedir,'/PROCESSING/CFSR/output/',year,'/spfh_profile_0',yyyyddd,'_0000.dat'
 write(tsfcfile,'(a,a,I4,a,a,a)') basedir,'/PROCESSING/CFSR/output/',year,'/sfc_temp_0',yyyyddd,'_0000.dat'
 write(presfile,'(a,a,I4,a,a,a)') basedir,'/PROCESSING/CFSR/output/',year,'/sfc_pres_0',yyyyddd,'_0000.dat'
 write(qsfcfile,'(a,a,I4,a,a,a)') basedir,'/PROCESSING/CFSR/output/',year,'/sfc_spfh_0',yyyyddd,'_0000.dat'
endif
if (tindex.ge.0.and.tindex.lt.1000) then
 write(tprof,'(a,a,I4,a,a,a,I3,a)') basedir,'/PROCESSING/CFSR/output/',year,'/temp_profile_',yyyyddd,'_0',int(tindex),'.dat'
 write(qprof,'(a,a,I4,a,a,a,I3,a)') basedir,'/PROCESSING/CFSR/output/',year,'/spfh_profile_',yyyyddd,'_0',int(tindex),'.dat'
 write(tsfcfile,'(a,a,I4,a,a,a,I3,a)') basedir,'/PROCESSING/CFSR/output/',year,'/sfc_temp_',yyyyddd,'_0',int(tindex),'.dat'
 write(presfile,'(a,a,I4,a,a,a,I3,a)') basedir,'/PROCESSING/CFSR/output/',year,'/sfc_pres_',yyyyddd,'_0',int(tindex),'.dat'
 write(qsfcfile,'(a,a,I4,a,a,a,I3,a)') basedir,'/PROCESSING/CFSR/output/',year,'/sfc_spfh_',yyyyddd,'_0',int(tindex),'.dat'
endif
if (tindex.gt.1000) then
 write(tprof,'(a,a,I4,a,a,a,I4,a)') basedir,'/PROCESSING/CFSR/output/',year,'/temp_profile_',yyyyddd,'_',int(tindex),'.dat'
 write(qprof,'(a,a,I4,a,a,a,I4,a)') basedir,'/PROCESSING/CFSR/output/',year,'/spfh_profile_',yyyyddd,'_',int(tindex),'.dat'
 write(tsfcfile,'(a,a,I4,a,a,a,I4,a)') basedir,'/PROCESSING/CFSR/output/',year,'/sfc_temp_',yyyyddd,'_',int(tindex),'.dat'
 write(presfile,'(a,a,I4,a,a,a,I4,a)') basedir,'/PROCESSING/CFSR/output/',year,'/sfc_pres_',yyyyddd,'_',int(tindex),'.dat'
 write(qsfcfile,'(a,a,I4,a,a,a,I4,a)') basedir,'/PROCESSING/CFSR/output/',year,'/sfc_spfh_',yyyyddd,'_',int(tindex),'.dat'
endif
write(6,*) presfile
open(10,file=tprof,form='unformatted',access='direct',recl=ci*cj*kz*4)
open(11,file=qprof,form='unformatted',access='direct',recl=ci*cj*kz*4)
open(12,file=tsfcfile,form='unformatted',access='direct',recl=ci*cj*4)
open(13,file=presfile,form='unformatted',access='direct',recl=ci*cj*4)
open(14,file=qsfcfile,form='unformatted',access='direct',recl=ci*cj*4)
read(10,rec=1) theta
read(11,rec=1) spfh
read(12,rec=1) tsfc
read(13,rec=1) psfc
read(14,rec=1) qsfc
close(10)
close(11)
close(12)
close(13)
close(14)

write(iptfile,'(a,a,a,a)') basedir,'/PROCESSING/CFSR/viirs_tile_lookup_tables/CFSR_',tile_num,'_lookup_icoord.dat'
write(jptfile,'(a,a,a,a)') basedir,'/PROCESSING/CFSR/viirs_tile_lookup_tables/CFSR_',tile_num,'_lookup_jcoord.dat'
open(10,file=iptfile,form='unformatted',access='direct',recl=ilg*jlg*4)
open(11,file=jptfile,form='unformatted',access='direct',recl=ilg*jlg*4)
read(10,rec=1) lookup_i
read(11,rec=1) lookup_j
close(10)
close(11)

write(6,*) lookup_i(3001,3001), lookup_j(3001,3001)
write(6,*) tsfc(lookup_i(3001,3001), lookup_j(3001,3001))

write(ifile,'(a,a,a)') basedir,'/overpass_corr/TRAD1_',tile_num 
open(10,file=ifile,form='unformatted',access='direct',recl=ilg*jlg*4)
read(10,rec=1) radini
close(10)

write(ifile,'(a,a,a)') basedir,'/overpass_corr/VIEW_ANGLE_',tile_num
open(20,file=ifile,form='unformatted',access='direct',recl=ilg*jlg*4)
read(20,rec=1) satang
close(20)
write(6,*) radini(3001,3001), satang(3001,3001)

tc15(:,:) = -9999.
tc55(:,:) = -9999.
lst(:,:) = -9999.
!call fsin
call wndo

do j = 1, jlg
do i = 1, ilg
 if (radini(i,j).ne.-9999.and.tc15(i,j).ne.-9999.) then
  lst(i,j) = radini(i,j)+tc15(i,j)
 endif
enddo
enddo
write(6,*) tc15(3001,3001), lst(3001,3001)
!write(ifile,'(a,a,a)') '/raid1/sport/people/chain/VIIRS_PROCESS/overpass_corr/lst_',tile_num,'.dat'
open(50,file=outfile,form='unformatted',access='direct',recl=ilg*jlg*4)
write(50,rec=1) lst 
close(50)

end program
