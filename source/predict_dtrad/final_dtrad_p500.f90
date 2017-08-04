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

write(6,*) day(3001,2501), night(3001,2501)
latn(:,:) = -9999.
do j = 1, dy
do i = 1, dx

 if (precip(i,j).gt.250.and.precip(i,j).lt.500.and.fmax(i,j).ge.0.0.and.fmax(i,j).lt.1.0) then 
 if (day(i,j).ne.-9999.and.night(i,j).ne.-9999.) then
 if (terrain(i,j).ne.-9999.and.lai(i,j).ne.-9999.) then 

  dtrad(i,j) = day(i,j)-night(i,j)
  fc(i,j) = 1.0-exp(-0.5*lai(i,j))

if ( dtime(i,j).le.1.5666664.and.dtrad(i,j).le.14.333391) then
 latn(i,j) = (-10.1172069+(0.099*dtrad(i,j))+(0.05*day(i,j))+(-0.005*terrain(i,j))+(0.5*fc(i,j)))
endif
if ( dtime(i,j).le.1.6458384.and.dtrad(i,j).gt.14.333391) then
 latn(i,j) = (-33.7260704+(0.138*dtrad(i,j))+(0.132*day(i,j)))
endif
if ( dtime(i,j).le.1.7687515.and.dtime(i,j).gt.1.5666664) then
 latn(i,j) = (-8.9760558+(0.158*dtrad(i,j))+(-0.006*terrain(i,j))+(7.7641*dtime(i,j)))
endif
if ( dtrad(i,j).le.14.333391.and.dtime(i,j).gt.1.7687515.and.dtime(i,j).le.2.5999999) then
 latn(i,j) = (-6.0488545+(0.286*dtrad(i,j))+(0.006*day(i,j))+(-2.5*fc(i,j))+(4.3457*dtime(i,j)))
endif
if ( dtime(i,j).le.2.3291681.and.dtrad(i,j).gt.14.333391.and.dtime(i,j).gt.1.6458384) then
 latn(i,j) = (-20.8916886+(0.234*dtrad(i,j))+(0.061*day(i,j))+(-0.004*terrain(i,j))+(-2.6*fc(i,j))+(4.3235*dtime(i,j)))
endif
if ( dtrad(i,j).le.14.333391.and.dtime(i,j).gt.2.5999999) then
 latn(i,j) = (-3.2000537+(0.576*dtrad(i,j))+(0.008*day(i,j))+(0.009*terrain(i,j))+(-1.2*fc(i,j))+(1.2707*dtime(i,j)))
endif
if ( day(i,j).le.296.32886.and.dtime(i,j).gt.2.3291681.and.dtime(i,j).le.3.2041662.and.fc(i,j).gt.0.099774703) then
 latn(i,j) = (-27.6182198+(0.213*dtrad(i,j))+(0.112*day(i,j))+(-0.006*terrain(i,j))+(-4.1*fc(i,j))+(1.5605*dtime(i,j)))
endif
if ( dtime(i,j).le.2.8083334.and.day(i,j).gt.296.32886.and.dtrad(i,j).gt.14.333391) then
 latn(i,j) = (-19.4747+(0.348*dtrad(i,j))+(0.033*day(i,j))+(-2.5*fc(i,j))+(6.113*dtime(i,j)))
endif
if ( fc(i,j).le.0.099774703.and.dtime(i,j).gt.2.3291681.and.dtime(i,j).le.2.8916676.and.dtrad(i,j).gt.14.333391) then
 latn(i,j) = (-21.7913334+(0.255*dtrad(i,j))+(0.052*day(i,j))+(-0.007*terrain(i,j))+(-0.6*fc(i,j))+(5.6417*dtime(i,j)))
endif
if ( dtime(i,j).le.2.9916666.and.dtime(i,j).gt.2.8916674.and.fc(i,j).gt.0.099774703) then
 latn(i,j) = (119.1887771+(0.408*dtrad(i,j))+(0.041*day(i,j))+(-0.007*terrain(i,j))+(-3*fc(i,j))+(-42.1144*dtime(i,j)))
endif
if ( dtime(i,j).le.2.8916674.and.dtime(i,j).gt.2.8083334.and.fc(i,j).gt.0.099774703.and.dtrad(i,j).gt.14.333391) then
 latn(i,j) = (-49.9743535+(0.289*dtrad(i,j))+(0.032*day(i,j))+(-0.006*terrain(i,j))+(-6.3*fc(i,j))+(17.4634*dtime(i,j)))
endif
if ( dtime(i,j).le.3.2041662.and.dtime(i,j).gt.2.9916666.and.dtrad(i,j).gt.14.333391) then
 latn(i,j) = (37.3755468+(0.355*dtrad(i,j))+(0.011*day(i,j))+(-0.003*terrain(i,j))+(-4.9*fc(i,j))+(-10.4766*dtime(i,j)))
endif
if ( dtime(i,j).gt.4.2833338.and.fc(i,j).gt.0.2488669) then
 latn(i,j) = (45.1926396+(0.439*dtrad(i,j))+(-0.081*day(i,j))+(0.006*terrain(i,j))+(-6.5*fc(i,j))+(-2.3874*dtime(i,j)))
endif
if ( terrain(i,j).le.2.4648063.and.dtime(i,j).gt.3.2041662.and.fc(i,j).gt.0.099774703.and.dtrad(i,j).gt.14.333391) then
 latn(i,j) = (7.8977438+(0.45*dtrad(i,j))+(-0.027*day(i,j))+(1.197*terrain(i,j))+(-5.3*fc(i,j))+(1.5055*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.3322911.and.dtime(i,j).le.3.8177083.and.day(i,j).le.319.33286.and.fc(i,j).gt.0.099774703.and.terrain(i,j).gt.2.4648063) then
 latn(i,j) = (-15.6814407+(0.328*dtrad(i,j))+(0.047*day(i,j))+(-0.007*terrain(i,j))+(-5*fc(i,j))+(3.4801*dtime(i,j)))
endif
if ( dtime(i,j).le.3.3322911.and.dtime(i,j).gt.3.2041662.and.terrain(i,j).gt.2.4648063) then
 latn(i,j) = (-34.6459961+(0.172*dtrad(i,j))+(0.064*day(i,j))+(-0.017*terrain(i,j))+(-7.5*fc(i,j))+(9.5506*dtime(i,j)))
endif
if ( dtime(i,j).gt.3.8177083.and.dtime(i,j).le.4.2833338.and.day(i,j).le.319.33286.and.fc(i,j).gt.0.099774703.and.terrain(i,j).gt.2.4648063) then
 latn(i,j) = (-12.8867856+(0.326*dtrad(i,j))+(0.028*day(i,j))+(-0.004*terrain(i,j))+(-6.7*fc(i,j))+(4.1068*dtime(i,j)))
endif
if ( fc(i,j).le.0.099774703.and.dtime(i,j).gt.2.8916676.and.dtime(i,j).le.4.2833338) then
 latn(i,j) = (-16.5686158+(0.399*dtrad(i,j))+(0.03*day(i,j))+(-0.003*terrain(i,j))+(9*fc(i,j))+(4.1221*dtime(i,j)))
endif
if ( day(i,j).gt.319.33286.and.fc(i,j).gt.0.099774703.and.dtime(i,j).le.4.2833338.and.dtime(i,j).gt.3.2041662) then
 latn(i,j) = (19.6101441+(0.384*dtrad(i,j))+(-0.059*day(i,j))+(-0.011*terrain(i,j))+(-9.4*fc(i,j))+(2.7591*dtime(i,j)))
endif
if ( dtime(i,j).gt.4.2833338.and.fc(i,j).le.0.2488669.and.dtrad(i,j).gt.14.333391) then
 latn(i,j) = (34.2715956+(0.468*dtrad(i,j))+(-0.045*day(i,j))+(0.009*terrain(i,j))+(-15.7*fc(i,j))+(-2.134*dtime(i,j)))
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
