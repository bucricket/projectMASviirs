program build_cubist_model

integer, parameter :: dx=3750, dy=3750
real :: latn(dx,dy), dtrad(dx,dy), day(dx,dy), night(dx,dy)
real :: dtime(dx,dy), lai(dx,dy), terrain(dx,dy)
character(len=400) :: precip_fn, fmax_fn, terrain_fn, daylst_fn, nightlst_fn
character(len=400) :: lai_fn, dtime_fn, out_fn
real :: predict(dx,dy)
real :: precip(dx,dy), fmax(dx,dy)
real :: fc(dx,dy)


call getarg(1,precip_fn)
call getarg(2,fmax_fn)
call getarg(3,terrain_fn)
call getarg(4,daylst_fn)
call getarg(5,nightlst_fn)
call getarg(6,lai_fn)
call getarg(7,dtime_fn)
call getarg(8,out_fn)


open(10,file=precip_fn,form='unformatted',access='direct',recl=dx*dy*4)
open(11,file=fmax_fn,form='unformatted',access='direct',recl=dx*dy*4)
read(10,rec=1) precip
read(11,rec=1) fmax
close(10)
close(11)


open(10,file=terrain_fn,form='unformatted',access='direct',recl=dx*dy*4)
open(11,file=daylst_fn,form='unformatted',access='direct',recl=dx*dy*4)
open(12,file=nightlst_fn,form='unformatted',access='direct',recl=dx*dy*4)
open(13,file=lai_fn,form='unformatted',access='direct',recl=dx*dy*4)
open(14,file=dtime_fn,form='unformatted',access='direct',recl=dx*dy*4)
read(10,rec=1) terrain
read(11,rec=1) day
read(12,rec=1) night
read(13,rec=1) lai
read(14,rec=1) dtime
close(10)
close(11)
close(12)
close(13)
close(14)

latn(:,:) = -9999.
do j = 1, dy
do i = 1, dx

 if (precip(i,j).gt.1000.and.precip(i,j).lt.20000.and.fmax(i,j).ge.0.0.and.fmax(i,j).lt.1.0) then 
 if (day(i,j).ne.-9999.and.night(i,j).ne.-9999.) then
 if (terrain(i,j).ne.-9999.and.lai(i,j).ne.-9999.) then 

  dtrad(i,j) = day(i,j)-night(i,j)
  fc(i,j) = 1.0-exp(-0.5*lai(i,j))

if ( dtime(i,j).le.1.8499988.and.dtrad(i,j).le.7.7208467) then
 latn(i,j) = (-2.00403+(0.333*dtrad(i,j))+(0.011*day(i,j))+(0.0009*terrain(i,j))+(-0.2*fc(i,j)))
endif
if ( dtime(i,j).le.1.6687537.and.dtrad(i,j).gt.7.7208467) then
 latn(i,j) = (8.0725188+(0.206*dtrad(i,j))+(-0.025*day(i,j))+(2.1*fc(i,j))+(0.0002*dtime(i,j)))
endif
if ( day(i,j).le.283.74112.and.dtime(i,j).le.2.3499992.and.dtime(i,j).gt.1.6687537) then
 latn(i,j) = (-7.8942329+(0.199*dtrad(i,j))+(0.042*day(i,j))+(-1.5*fc(i,j))+(0.0553*dtime(i,j)))
endif
if ( dtrad(i,j).le.7.7208467.and.dtime(i,j).gt.1.8499988) then
 latn(i,j) = (-1.1500314+(0.646*dtrad(i,j))+(-0.012*day(i,j))+(0.0053*terrain(i,j))+(1.5*fc(i,j))+(1.8224*dtime(i,j)))
endif
if ( dtime(i,j).le.2.3499992.and.day(i,j).gt.283.74112.and.dtrad(i,j).gt.7.7208467.and.dtime(i,j).gt.1.6687537) then
 latn(i,j) = (27.0092857+(0.263*dtrad(i,j))+(-0.111*day(i,j))+(-0.0043*terrain(i,j))+(4.4648*dtime(i,j)))
endif
if ( terrain(i,j).le.1.8872331.and.dtime(i,j).gt.3.1208341.and.day(i,j).le.303.73444.and.dtime(i,j).le.3.625001) then
 latn(i,j) = (-16.3071504+(0.635*dtrad(i,j))+(0.004*day(i,j))+(0.6216*terrain(i,j))+(1.6*fc(i,j))+(4.4836*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.1437504.and.dtime(i,j).le.3.216666.and.day(i,j).le.303.73444.and.dtrad(i,j).gt.7.7208467) then
 latn(i,j) = (-123.9988942+(0.218*dtrad(i,j))+(-0.054*day(i,j))+(0.0162*terrain(i,j))+(2*fc(i,j))+(45.4454*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.1208341.and.dtime(i,j).le.3.1437504.and.day(i,j).le.303.73444) then
 latn(i,j) = (160.1543705+(0.421*dtrad(i,j))+(-0.113*day(i,j))+(-0.0018*terrain(i,j))+(-39.451*dtime(i,j)))
endif
if ( terrain(i,j).le.3.8671637.and.dtime(i,j).le.2.6520841.and.dtime(i,j).gt.2.3499992.and.day(i,j).le.303.73444.and.dtrad(i,j).gt.7.7208467) then
 latn(i,j) = (-23.7383371+(0.46*dtrad(i,j))+(0.289*terrain(i,j))+(-0.2*fc(i,j))+(10.306*dtime(i,j)))
endif
if ( dtime(i,j).gt.2.6520841.and.dtime(i,j).le.3.1208341.and.day(i,j).le.298.08633.and.terrain(i,j).le.46.655537.and.dtrad(i,j).gt.7.7208467) then
 latn(i,j) = (-28.1587162+(0.259*dtrad(i,j))+(0.083*day(i,j))+(0.0148*terrain(i,j))+(0.7*fc(i,j))+(3.4107*dtime(i,j)))
endif
if ( dtime(i,j).le.3.1208341.and.day(i,j).le.303.73444.and.dtime(i,j).gt.2.6520841.and.day(i,j).gt.298.08633.and.terrain(i,j).le.46.655537.and.dtrad(i,j).gt.7.7208467) then
 latn(i,j) = (70.4540089+(0.453*dtrad(i,j))+(-0.235*day(i,j))+(0.0315*terrain(i,j))+(-2.2*fc(i,j))+(1.7201*dtime(i,j)))
endif
if ( dtime(i,j).le.2.6520841.and.dtime(i,j).gt.2.3499992.and.terrain(i,j).gt.3.8671637.and.dtrad(i,j).gt.7.7208467) then
 latn(i,j) = (-7.4974045+(0.35*dtrad(i,j))+(-0.057*day(i,j))+(-0.0024*terrain(i,j))+(-1.6*fc(i,j))+(12.0058*dtime(i,j)))
endif
if ( dtime(i,j).le.3.1208341.and.terrain(i,j).gt.46.655537.and.dtime(i,j).gt.2.6520841.and.day(i,j).le.303.73444.and.dtrad(i,j).gt.7.7208467) then
 latn(i,j) = (-23.0420975+(0.468*dtrad(i,j))+(0.085*day(i,j))+(-1.9*fc(i,j))+(1.1588*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.216666.and.dtime(i,j).le.3.625001.and.day(i,j).le.303.73444.and.terrain(i,j).gt.1.8872331.and.dtrad(i,j).gt.7.7208467) then
 latn(i,j) = (-21.2151245+(0.434*dtrad(i,j))+(0.054*day(i,j))+(-0.0038*terrain(i,j))+(3.0208*dtime(i,j)))
endif
if ( dtime(i,j).le.3.1091135.and.dtime(i,j).gt.2.8583345.and.day(i,j).gt.303.73444) then
 latn(i,j) = (-54.9101071+(0.311*dtrad(i,j))+(0.205*day(i,j))+(0.0103*terrain(i,j))+(-4.1*fc(i,j)))
endif
if ( dtime(i,j).gt.3.375001.and.dtime(i,j).le.3.625001.and.day(i,j).gt.303.73444) then
 latn(i,j) = (44.7703831+(0.706*dtrad(i,j))+(-0.087*day(i,j))+(-0.0115*terrain(i,j))+(1.8*fc(i,j))+(-5.0644*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.625001.and.day(i,j).gt.295.60349) then
 latn(i,j) = (39.9200907+(0.484*dtrad(i,j))+(-0.097*day(i,j))+(-0.0022*terrain(i,j))+(-1.6*fc(i,j))+(-1.109*dtime(i,j)))
endif
if ( dtime(i,j).le.3.375001.and.dtime(i,j).gt.3.1091135.and.day(i,j).gt.303.73444.and.dtrad(i,j).gt.7.7208467) then
 latn(i,j) = (-72.2210978+(0.303*dtrad(i,j))+(0.288*day(i,j))+(0.012*terrain(i,j))+(-3.1*fc(i,j))+(-2.4245*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.625001.and.day(i,j).le.295.60349) then
 latn(i,j) = (-36.3558734+(0.424*dtrad(i,j))+(0.162*day(i,j))+(-0.0045*terrain(i,j))+(0.2*fc(i,j))+(-1.1044*dtime(i,j)))
endif
if ( day(i,j).gt.303.73444.and.dtime(i,j).le.2.8583345) then
 latn(i,j) = (-40.1119286+(0.411*dtrad(i,j))+(0.08*day(i,j))+(0.006*terrain(i,j))+(-1.2*fc(i,j))+(7.5753*dtime(i,j)))
endif
 endif
 endif
 endif

enddo
enddo

open(10,file=out_fn,form='unformatted',access='direct',recl=dx*dy*4)
write(10,rec=1) latn
close(10)

end program
