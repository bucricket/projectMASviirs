#!/usr/bin/perl

$homedir="/usr/people/chain/viirs_process/predict_dtrad";
$tiledir="/raid1/sport/people/chain/VIIRS_PROCESS/TILES";

$year=@ARGV[0];
$doy=@ARGV[1];
$tile=@ARGV[2];

 $yyyyddd=$year*1000+$doy;
 `gunzip -ff $tiledir/$tile/FINAL_DAY_LST_${yyyyddd}_${tile}.dat.gz`;
 `gunzip -ff $tiledir/$tile/FINAL_NIGHT_LST_${yyyyddd}_${tile}.dat.gz`;
 `$homedir/final_dtrad_p250_fmax0 $year $doy $tile`;
 `$homedir/final_dtrad_p250_fmax20 $year $doy $tile`;
 `$homedir/final_dtrad_p500 $year $doy $tile`;
 `$homedir/final_dtrad_p750 $year $doy $tile`;
 `$homedir/final_dtrad_p1000 $year $doy $tile`;
 `$homedir/final_dtrad_p2000 $year $doy $tile`;
 `$homedir/merge $year $doy $tile`;
 `$homedir/calc_predicted_trad2 $year $doy $tile`;
 `gzip -ff $tiledir/$tile/FINAL_DAY_LST_${yyyyddd}_${tile}.dat`;
 `gzip -ff $tiledir/$tile/FINAL_NIGHT_LST_${yyyyddd}_${tile}.dat`;
 `gzip -ff $tiledir/$tile/FINAL_DAY_LST_TIME2_${yyyyddd}_${tile}.dat`;
 `gzip -ff $tiledir/$tile/FINAL_DTRAD_${yyyyddd}_${tile}.dat`;


