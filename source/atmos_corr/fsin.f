      SUBROUTINE FSIN

      use corr_parms
      use corr_arrays

      parameter (kzp1=kz+1)
      dimension zt(kzp1),zth(kzp1)
      PARAMETER (mkx=37)
      CHARACTER*3 MONTH(12)
      CHARACTER*256 f1, f2, f3, f4
      CHARACTER*68 CAPSHN
      LOGICAL LNEW,LCCL

      f1='/data/data123/chain/4KM/ATMOS_CORR_TNOON/temp_profile.bin'
      f2='/data/data123/chain/4KM/ATMOS_CORR_TNOON/height_profile.bin'
      f3='/data/data123/chain/4KM/ATMOS_CORR_TNOON/spfh_profile.bin'
      f4='/data/data123/chain/4KM/ATMOS_CORR_TNOON/pressure_profile.bin'

      open(10,file=f1,form='unformatted',
     &          access='direct',recl=ci*cj*kz*4)
      open(11,file=f2,form='unformatted',
     &          access='direct',recl=ci*cj*kz*4)
      open(12,file=f3,form='unformatted',
     &          access='direct',recl=ci*cj*kz*4)
      open(13,file=f4,form='unformatted',
     &          access='direct',recl=ci*cj*kz*4)

      read(10,rec=1) theta
      read(11,rec=1) height
      read(12,rec=1) spfh 
      read(13,rec=1) pres 

      close(10)
      close(11)
      close(12)
      close(13)

      
      do k = 1, kz
      write(6,*) theta(700,300,k),spfh(361,149,k)*1000.,pres(361,149,k)
      enddo


c  Next, compute the column variables from the analysis fields.
c  Form column variables:
      kmax=kz
      kmm1=kmax-1
      g=9.806
c      r=287.0
c
      print*,'Data check ...'
      do j=1,jlg
        do i=1,ilg
!          zps=p(i,j,kz)
!          prsmd(i,j)=zps
!          do k=1,kmax                                        !kz/kmax is surface
!            ztan(k,i,j)=theta(i,j,k)
!            zt(k)=ztan(k,i,j)
!            zran(k,i,j)=r_t1(i,j,k)                   !mixing ratio
!            zqan(k,i,j)=p(i,j,k)                          !sigma levels


!          enddo

        enddo
      enddo
c
      close(IDATE)
      goto 400
c
 99   WRITE(6,399)IDATA
 399  FORMAT(' ***E*** FSIN: OPEN ERROR ON INIT=',I2)
      STOP 99
 400  CONTINUE
c
      RETURN
      END

