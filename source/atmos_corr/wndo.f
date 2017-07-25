c  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
c
c
C     * * * * * * * * * * * * * * * * * * * * * * * * * * * *
C     *                 PROGRAM WINDOW                      *
C     *   COMPUTE ATMOSPHERIC EFFECT IN THE THERMAL IR WINDOW
C             WRITTEN BY DR. JOHN C. PRICE - U.S.DEPT. OF
C      AGRICULTURE. REFERENCE - REMOTE SENSING OF ENVIRONMENT
C      1983,VOL 13, PP353-361                               *
C     *                                                     *
C     * ABSORPTION FROM KNEIZYS-LOWTRAN                     *
C     *    EM = MEAN MOLECULAR WEIGHT OF AIR                *
C     *    GO = ACCELERATION DUE TO GRAVITY M/SEC-SEC       *
C     *    R = GAS CONSTANT JOULES/(0 K-KG MOLE)            *
C     *    C1 = CONSTANT IN PLANK'S FUNCTION (MW-CM**4)/(CM**2-SR)
C     *    C2 = CONSTANT IN PLANK'S FUNCTION CM- K          *
C     *                                                     *
C     *      P = PRESSURE IN MILLIBARS                      *
C     *      T = TEMPERATURE IN DEGREES CELSIUS             *
C     *      E = VAPOR PRESSURE IN MILLIBARS                *
C     * * * * * * * * * * * * * * * * * * * * * * * * * * * *
C
C     DIMENSIONS FOR INPUT DATA
C  -------------------------------------------------------------
C  Inputs from CRAS model:  TA - column air temp in degrees C
C                           PI - pressure in millibars
C                           EI - mixing ratio (g/kg)
C                        ANGLE - view angle in degrees
C
C  We perform correction on input surface brightness temperatures
C  'radini' (T(1)) and 'radfin' (T(2)) on a per-point basis.
C
C  Outputs: A new 'raddts' array from corrected (radfin-radini).
C
C  -------------------------------------------------------------
C  Heavily modified by John Mecikalski
C  10/02/96
C  Corrections to original Price code added by Martha Anderson
C  2/19/02
C
      subroutine wndo

      use corr_parms
      use corr_arrays

      DIMENSION TI(kz),EI(kz),PI(kz),TA(kz)
C     DIMENSIONS FOR INTERPOLATED DATA, LAYER DEPTHS, ABSORPTION
C
      DIMENSION E(kz),TT(kz),EMIS(kz)    !TT was T, but used in CRAS model
C
      INTEGER pindex
      DIMENSION PLEV(6),DPRES(6),SURT(2)
      DIMENSION TAU(kz),P(kz),HI(kz),TVI(kz),SKY(kz),STR(kz)
      character*42 wndofile
      DATA PLEV/100.,200.,300.,500.,700.,900./
      DATA DPRES /10.,20.,30.,50.,75.,100./
      DATA NSURT,NANGLE /1,1/    !jrm
C     ABSORPTION COEFFICIENTS, PHYSICAL CONSTANTS
      DATA EM,GO,R,C1,C2/28.967,9.78,8314.34,1.191E-9,1.439/
      DATA BADATA/999./
      INTEGER tpt, ipt, jpt
      INTEGER start, counter
C
C  Setting constants based on CRAS Model inputs:
      NLEV=kz-1
      IVAP=4
cmca      EPSLN=1.00
      EPSLN=0.98
      A180PI=180.0/3.141592654
      B=R/(GO*EM)

C     DEFINE WAVELENGTH OF SPECTRAL INTERVAL
C
C     ANV=10000/(WAVELENGTH IN MICROMETERS)
C     HCMM 870. ;   TIROS-N 912. ;   NOAA6 912.
C     NOAA7    CH4=927., CH5=841.
C
      ANV=873.6           !GOES-I IMAGER @10.7MICRONS
   10 FORMAT(20X,'WAVENUMBER=',F8.1,'  CM-1')
      C1P = C1*ANV**3
      C2P = C2*ANV
C
C ********************************************************************
C
C 1) READ NUMBER OF CASES (SOUNDINGS) TO BE DONE FOR EACH CASE
C 2) READ HEADER INFORMATION, # OF SOUNDING LEVELS,
C 3) SET TYPE OF HUMIDITY DATA, VIEWING ANGLE, SURFACE EMISSIVITY
C     HUMIDITY INDEX(1=REL.HUM.,2=ABS.HUM.,3=MIXING RATIO,4=SPEC.HUM.,
C     5=DEWPOINT DEPRESSION 6=VAPOR PRESSURE-MB,7=DEWPOINT(K),8=DEW
C     POINT(C))
C     MIXING RATIO IN GM/KG
C 4) SET VIEWING ANGLE
C
C ********************************************************************
ccc   WRITE(6,180)
ccc   WRITE(6,10) ANV
C
C  LOOP FOR NUMBER OF SOUNDINGS; Loop over entire analysis grid.
C
       do ja=1,jlg
!        write(6,*) ja
       do ia=1,ilg

        SURT(1)=radini(ia,ja)
        if (SURT(1).eq.-9999.) then
         tc15(ia,ja)=BADATA
         tc55(ia,ja)=BADATA
        goto 299
        endif

        ipt=lookup_i(ia,ja)
        jpt=lookup_j(ia,ja)
        if (ipt.eq.-9999.or.jpt.eq.-9999.) then
         ipt=1
         jpt=1
        endif

        sfpr= psfc(ipt,jpt)/100.       !P in millibars
        if (sfpr.eq.0.0) goto 299

        pindex=0
        do i = 1, kz
         ka=i
         if (sfpr.ge.pres(ka).and.pindex.eq.0) then 
          pindex = i
         endif
        enddo
       
        if (sfpr.gt.pres(1)) then
         pindex=1
        endif  

        start=(21-pindex)+1
        counter=1
        do i=pindex,21  !kz
          ka=i
!          PI(i)=pres((kz-ka)+1)                 !millibars
          PI(counter)=pres(ka)
          TA(counter)=theta(ipt,jpt,ka)/(1000/PI(counter))**0.286
!          TA(i)=sum(theta(ipt-1:ipt+1,jpt-1:jpt+1,(kz-ka)+1))/9.
          EI(counter)=spfh(ipt,jpt,ka)
!          EI(i)=sum(spfh(ipt-1:ipt+1,jpt-1:jpt+1,(kz-ka)+1))/9.
          if (ia.eq.1001.and.ja.eq.2001) then
          write(6,*)sfpr,pres(1),PI(counter),EI(counter),TA(counter)
          endif
          counter=counter+1
        enddo
        ANGLE=acos(satang(ia,ja))*A180PI
        ANGLE=satang(ia,ja)
        if (ANGLE.gt.90.0) ANGLE=80.0

      DO 140 I=1,counter-1 !NLEV
      TI(I)=TA(I)
      STR(I)=EI(I)
  140 CONTINUE
c
  160 CONTINUE
  180 FORMAT(2X,'PROGRAM *WINDOW* COMPUTES THE ATMOSPHERIC CORRECTION',
     X/10x,'  DUE TO WATER VAPOR IN THE 10-12 MICROMETER REGION ',//22X,
     X'JOHN C. PRICE',//8x,' AGRICULTURAL RESEARCH SERVICE   USDA'/)
c      WRITE(6,220) NLEV
c      WRITE(6,240) IVAP
c      WRITE(6,260) ANGLE
c      WRITE(6,280) EPSLN
  220 FORMAT(12X,'NUMBER OF SOUNDING LEVELS',20X,I2)
  240 FORMAT(12X,'HUMIDITY OBSERVATION TYPE',20X,I2)
  260 FORMAT(12X,'VIEWING ANGLE (=1.0 GIVES 0,10,20,30)',3X,F7.1)
  280 FORMAT(12X,'SURFACE EMISSIVITY',23X,F6.2)
      HI(1) = 0.
      SUMPR = 0.0
      PRECIP=0.0
c      WRITE(6,320)
  320 FORMAT(/' P (MB)',5X,'T (C)',5X,'MOISTURE INPUT',5X,'E (MB)'
     1,5X,'H (M)',5X,'PR. WATER (CM)'/)
C  CREATE VAPOR PRESSURE(MB) FROM INPUT DATA
      DO 560 I = 1,counter-1
      GO TO (340,360,380,400,420,440,460,480), IVAP
  340 EXPT=7.5*(TI(I)-273.15)/(237.3+TI(I)-273.15)
      EI(I)=EI(I)/100*6.11*10.**EXPT
      GO TO 500
  360 EI(I)=EI(I)*R*TI(I)/18.02
      GO TO 500
  380 IF(EI(1).GT..07) EI(I)=EI(I)!*0.001  !Use this for g/kg input (IVAP=3)
C  CONVERT TO KG/KG FOR PROGRAM USE ONLY IF INPUT IS GM/KG
      EI(I)=EI(I)*PI(I)/(.622+EI(I))
      GO TO 500
  400 EI(I)=EI(I)*PI(I)/(.378*EI(I)+.622)
ccc   GO TO 100                           !This looks wrong; 100?
      GO TO 500
  420 TD=TI(I)-EI(I)-273.15
      EXPT=7.5*TD/(237.3+TD)
      EI(I)=6.11*10.**EXPT
  440 CONTINUE
      GO TO 500
  460 TD=EI(I)-273.15
      EXPT=7.5*TD/(237.3+TD)
      EI(I)=6.11*10**EXPT
      GO TO 500
  480 TD=EI(I)                            !Use this -- input Td (IVAP=8)
      EXPT=7.5*TD/(237.3+TD)
      EI(I)=6.11*10.**EXPT
  500 IF (EI(I).LT.0.01) EI(I)=0.01
      TVI(I)=TI(I)/(1.0-.378*EI(I)/PI(I))
      IF(I.EQ.1) GO TO 520
C  GEOMETRIC HEIGHT
      HI(I) = HI(I-1) + B*(TVI(I)-TVI(I-1))*ALOG(PI(I)/PI(I-1))/
     *(ALOG(TVI(I-1)/TVI(I)))
C  PRECIPITABLE WATER IN LAYER
      PRECIP = -.622/(GO)*10.*
     X((EI(I-1)*PI(I)-EI(I)*PI(I-1))*
     XALOG(PI(I)/PI(I-1))/(PI(I)-PI(I-1)) + EI(I)-EI(I-1))
c      SUMPR=2.681
      SUMPR=SUMPR+PRECIP
  520 CONTINUE
c      WRITE(6,540) PI(I),TA(I),STR(I)*1000.,EI(I),HI(I),PRECIP
c  540 FORMAT(F6.0,4X,F6.1,6X,F10.3,9X,F5.2,4X,F6.0,10X,F5.2)
  560 CONTINUE
  600 CONTINUE
C
C  INSERT INTERMEDIATE LEVELS FOR RADIATIVE TRANSFER INTEGRATION
C  (I don't understand why we do this -- JRM)
C  (Improves the accuracy of the integral -- MCA)
C
      E(1) = EI(1)
      P(1) = PI(1)
      TT(1) = TI(1)
      J = 1
      MLEV = 1
      DPP = 0.
      OPTD=0.
      EGR=(EI(2)-EI(1))/(PI(2)-PI(1))
      TGR=(TI(2)-TI(1))/(PI(2)-PI(1))

      DO 660 I=1,kz-1
      M=I+1
      ILP = ((PI(J)-DPP)-PI(J+1))/DPRES(MLEV) + 0.999
      IF(ILP.LT.1) ILP = 1
       DLP = ((PI(J)-DPP)-PI(J+1))/ILP
      P(I+1)=P(I)-DLP
      E(I+1)=E(I)+EGR*(P(I+1)-P(I))
      TT(I+1)=TT(I)+TGR*(P(I+1)-P(I))
C  LAYER OPTICAL DEPTH, EMISSION AT GIVEN LEVELS
      EMIS(I)=0.5*(PLANK(TT(I),anv)+PLANK(TT(I+1),anv))
C  SIMPSON'S RULE INTEGRATION (3 POINTS)
C
      TAU(I)=(P(I)-P(I+1))*(DTAUDP(ANV,TT(I),E(I),P(I))+
     xDTAUDP(ANV,TT(I+1),E(I+1),P(I+1)) + 4.*
     xDTAUDP(ANV,(TT(I)+TT(I+1))/2.,(E(I)+E(I+1))/2.,(P(I)+P(I+1))/2.)
     x)/6.
      OPTD=OPTD+TAU(I)
      IF (ABS(P(I+1)-PI(J+1)).GT..01) GO TO 620
      J = J +1
      EGR=(EI(J+1)-EI(J))/(PI(J+1)-PI(J))
      TGR=(TI(J+1)-TI(J))/(PI(J+1)-PI(J))
      DPP = 0.
  620 CONTINUE
      IF ((PI(1)-P(I+1)).LT.PLEV(MLEV)) GO TO 640
      MLEV = MLEV + 1
      DPP=PI(J)-P(I+1)
  640 CONTINUE
      IF (ABS(P(I+1)-PI(NLEV)).LT.0.1) GO TO 680
660    CONTINUE
680    CONTINUE
  700 FORMAT(/7X,'PRECIP. WATER(CM)',F6.2,10X,'TOTAL OPTICAL DEPTH=',
     XF7.3,9X,'SKY RADIANCE',E12.3,//12X,'GROUND RADIANCE',E12.3)
C
C  RADIATIVE TRANSFER INTEGRATION
C
      CS=COS(ANGLE/57.29)
c      if (ia.eq.426 .and. ja.eq.108) then
c       print*,'CS = ',cs, 'Angle = ', angle
c      endif
      TRANS=EXP(-OPTD/CS)
C     720 LOOP COMPUTES UPWARD SKY RADIANCE AT EACH LEVEL
      SKY(2)=EMIS(1)*(1.0-EXP(-TAU(1)/CS))
!      if (ia.eq.391 .and. ja.eq.114) then
!        print*,'SKY(2) EMIS(1) TAU(1) ',sky(1),emis(1),tau(1),
!     + EXP(-TAU(1)/CS)
!      endif
      DO 720 I=3,M
      SKY(I)=EMIS(I-1)+(SKY(I-1)-EMIS(I-1))*EXP(-TAU(I-1)/CS)
  720 CONTINUE
      GRRAD=EPSLN*PLANK(TT(1),anv)
!      if (ia.eq.251 .and. ja.eq.251) then
!       WRITE(6,700) SUMPR,OPTD,SKY(M),GRRAD
!      endif
C
C  SLOPE FOR LINEAR EQUATION
C  COMPUTED AT TT(LOWEST LEVEL AIR TEMP.)+5C
C
      R1=EPSLN*PLANK(TT(1)+6.0,anv)*TRANS+SKY(M)
      R2=EPSLN*PLANK(TT(1)+4.0,anv)*TRANS+SKY(M)
      SLOPE=(APLANK(R1,anv)-APLANK(R2,anv))/2.
      RSLOPE=1.0/SLOPE
C
C  OFFSET FOR LINEAR EQUATION
C  COMPUTED AT TT(LOWEST LEVEL AIR TEMP.) +20 C
C
      R1=EPSLN*PLANK(TT(1)+20.0,anv)*TRANS+SKY(M)
      OFFSET=APLANK(R1,anv)-SLOPE*(TT(1)+20.0)
      ROFF=-OFFSET/SLOPE
C
C  760 LOOP FOR SURFACE TEMPERATURES
cmca    This is wrong - below the computation is corrected.

cmca      DO 760 I=1,NSURT
cmca      R1=EPSLN*PLANK(SURT(I),anv)*TRANS
cmca      TRAD=APLANK(R1+SKY(M),anv)
cmca      CORR=SURT(I)-TRAD

      DO 760 I=1,NSURT
      grndrad=(plank(surt(i),anv)-sky(m)*(1.0+trans*(1.0-epsln)))
     &         /trans
      if (trans.le.0.001) grndrad=0.0
      if (grndrad.le.0.0) then
!        iflag(ia,ja)=2                      !cloudy point
        goto 299
      endif
      trad=aplank(grndrad/epsln,anv)
      corr=trad-surt(i)
      if (ia.eq.1001  .and. ja.eq.2001) then
        print*,'trad corr epsln grndrad ',trad,corr,radini(ia,ja)
      endif
C
      if (I.eq.1) then
cmca    tc15(ia,ja)=SURT(I)-TRAD
        tc15(ia,ja)=TRAD-SURT(I)
!        write(6,*) "PW = ", tc15(ia,ja)
!        pw(ia,ja)=SUMPR
ccc     radini(ia,ja)=radini(ia,ja)+CORR    !Apply correction (radini)
c        radini(ia,ja)=radini(ia,ja)+CORR            !Apply correction (radini)
      endif
      if (I.eq.2) then
cmca    tc55(ia,ja)=SURT(I)-TRAD      
        tc55(ia,ja)=TRAD-SURT(I)
cc     radfin(ia,ja)=radfin(ia,ja)+CORR    !Apply correction (radfin)
c        rdfin=radfin(ia,ja)+CORR            !Apply correction (radfin)
      endif
c      if (ia.eq.391  .and. ja.eq.114) then
c      if (I.eq.2)
c     1 print*,'Tcorr: SR+1.5h  SR+5.5h ',tc15(ia,ja),tc55(ia,ja),ia,ja
c      endif
C
C      if (ia.eq.1876.and.ja.eq.501) then
c       WRITE(6,740) SURT(I),TRAD,CORR,ANGLE
c  740 FORMAT(/2X,'  SURFACE T.',F8.2,'     SATELLITE',F8.2,
c     X'      SURF.-SAT. TEMP',
c     X F6.2,10X,'ANGLE',F8.1)
c       endif
  760 CONTINUE

c      if (ia.eq.391.and.ja.eq.114) then
c
c       WRITE(6,800) SLOPE,OFFSET
c  800 FORMAT(/12X,'T(SATELLITE)=',F10.3,'  T(GROUND) + ',F10.2)
c       WRITE(6,860) RSLOPE,ROFF
c  860 FORMAT(/12X,'T(GROUND)=',F10.3,'  T(SATELLITE) + ',F10.2)
c       endif
C
C  Return to calling program with corrected 'radini(ia,ja)' and
C  'radfin(ia,ja)', which are then subtracted from one another to
C  get 'raddts(ia,ja)'.
ccc       raddts(ia,ja)=radfin(ia,ja)-radini(ia,ja)       !NEW raddts
c          rddts=rdfin-rdini                               !NEW raddts
C
C  Write out corrections, satellite angle and precipitable water to
C  ASCII file for plots and analysis.
C          write(52,*)tc15(ia,ja),tc55(ia,ja),sumpr
C
C  End of main ia, ja loop
 299      continue
        enddo
      enddo
C
c      close(20,status='keep')
       close(20)
      RETURN
      END
C
C  From top of program wndo  -----------------------------------------
C  TEMPERATURE TO RADIANCE AND INVERSE
C
C  UNITS OF PLANCK FUNCTION AND SKY RADIANCE ARE (MW-CM)/(CM**2-SR)
C  THIS IS A NON STANDARD PLANCK FUNCTION BUT IT WAS SELECTED IN ORDER
C  TO MATCH (ALMOST) THE NOAA AVHRR DOCUMENTATION WHICH IS FOR UNITS
C  (MW-CM)/(M**2-SR)
C
C  THESE UNITS DROP OUT COMPLETELY IN TEMPERATURE CALCULATIONS
C
      FUNCTION PLANK(X,anv)
        C1=1.191E-9
        C2=1.439
cmca        ANV=870.		! Don't want this hardcoded to a
        C1P = C1*ANV**3		! different wavelength!!
        C2P = C2*ANV
        PLANK = C1P/(EXP(C2P/X)-1.)
      END
      FUNCTION APLANK(X,anv)
        C1=1.191E-9
        C2=1.439
cmca        ANV=870.
        C1P = C1*ANV**3
        C2P = C2*ANV
        APLANK = C2P/ALOG(1.+C1P/X)
      END
C
cmca  This function considers absorption in atmospheric column
cmca  only due to water vapor.
C  FUNCTION DTAUDP(WAVENUMBER,TEMPERATURE,VAPOR PRESSURE,PRESSURE)
cmca      FUNCTION DTAUDP(W,X,Y,Z)
cmca        GO=9.78
cmca        DTAUDP=(4.18+5578.*EXP((-7.87E-3)*
cmca     XW))*EXP(1800.*(1.0/X-1.0/296.))*(Y/Z+.002)*(.622/(101.3*GO))*Y
cmca      END
      
cmca  This function contains an empirical correction for absorption
cmca  by non-water-vapor constituents.  It was developed in comparison
cmca  with a series of MODTRAN experiments.     
C  FUNCTION DTAUDP(WAVENUMBER,TEMPERATURE,VAPOR PRESSURE,PRESSURE)
      FUNCTION DTAUDP(W,X,Y,Z)
        GO=9.78
        DTAUDP1=(4.18+5578.*EXP((-7.87E-3)*
     XW))*EXP(1800.*(1.0/X-1.0/296.))*(Y/Z+.002)*(.622/(101.3*GO))*Y
        DTAUDP=DTAUDP1+(0.00004+Y/60000.)
      END
      
