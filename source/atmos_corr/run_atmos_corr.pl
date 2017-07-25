#! /usr/bin/perl

$homedir="/usr/people/chain/viirs_process/atmos_corr";
$overdir="/raid1/sport/people/chain/VIIRS_PROCESS/overpass_corr";
$tiledir="/raid1/sport/people/chain/VIIRS_PROCESS/TILES";

$year=@ARGV[0];
$iday=@ARGV[1];
$eday=@ARGV[1];
$tile=@ARGV[2];
#$yyyyddd=$year*1000+$doy;

for ($doy=$iday; $doy<=$eday; $doy++){
$ayear=2014;
$yyyyddd=$year*1000+$doy;

use integer;
$nweek=($doy-1)/7;
$cday=$nweek*7;
$offset=($doy-$cday)/7;
$rday=(($offset+$nweek)*7)+1;
$avgddd=$ayear*1000+$rday;
print "$avgddd\n";
system("cp $overdir/dtrad_avg/DTRAD_${tile}_${avgddd}.dat.gz $overdir/CURRENT_DTRAD_AVG_$tile.dat.gz");
system("gunzip -ff $overdir/CURRENT_DTRAD_AVG_$tile.dat.gz");
system("cp $overdir/nominal_overpass_time_tiles/DAY_ZTIME_${tile}.dat.gz $overdir/CURRENT_DAY_ZTIME_$tile.dat.gz");
system("gunzip -ff $homedir/ATMOS_CORR/CURRENT_DAY_ZTIME_$tile.dat.gz");

@dayfiles=`ls $tiledir/$tile/day_bt_flag_${yyyyddd}*$tile*`;
foreach $dayfile (@dayfiles){
 chomp($dayfile);
 print "$dayfile\n";
 $time = substr($dayfile,-11,4);
 print "$time\n";
 $viewfile="$tiledir/$tile/view_angle_${yyyyddd}_${tile}_${time}.dat.gz";
 `cp $dayfile $overdir/RAW_TRAD1_$tile.gz`;
 `cp $viewfile $overdir/VIEW_ANGLE_$tile.gz`;
 `gunzip -ff $overdir/RAW_TRAD1_$tile.gz`;
 `gunzip -ff $overdir/VIEW_ANGLE_$tile.gz`;
 `$homedir/calc_offset_correction $year $doy $time $tile`;
 `$homedir/run_correction $tile $time $yyyyddd $year`;
 `mv $overdir/lst_$tile.dat $tiledir/$tile/lst_${yyyyddd}_${tile}_${time}.dat`;
 `gzip -ff $tiledir/$tile/lst_${yyyyddd}_${tile}_${time}.dat`;
 `rm $overdir/TRAD1_$tile`;
 `rm $overdir/VIEW_ANGLE_$tile`;
 `rm $overdir/RAW_TRAD1_$tile`;
 die;
}

@nightfiles=`ls $tiledir/$tile/night_bt_flag_${yyyyddd}*$tile*`;
foreach $nightfile (@nightfiles){
 chomp($nightfile);
 print "$nightfile\n";
 $time = substr($nightfile,-11,4);
 $viewfile="$tiledir/$tile/view_angle_${yyyyddd}_${tile}_${time}.dat.gz";
 `cp $nightfile $overdir/TRAD1_$tile.gz`;
 `cp $viewfile $overdir/VIEW_ANGLE_$tile.gz`;
 `gunzip -ff $overdir/TRAD1_$tile.gz`;
 `gunzip -ff $overdir/VIEW_ANGLE_$tile.gz`;
 `$homedir/run_correction $tile $time $yyyyddd $year`;
 `mv $overdir/lst_$tile.dat $tiledir/$tile/lst_${yyyyddd}_${tile}_${time}.dat`;
 `gzip -ff $tiledir/$tile/lst_${yyyyddd}_${tile}_${time}.dat`;
 `rm $overdir/TRAD1_$tile`;
 `rm $overdir/VIEW_ANGLE_$tile`;
}

}

`rm $overdir/CURRENT_DAY_ZTIME_$tile.dat`;
`rm $overdir/CURRENT_DTRAD_AVG_$tile.dat`;
