#! /usr/bin/perl

$iyear=2015;
$eyear=2015;
$iday=156;
$eday=162;
$tile_num="063";

for ($year=$iyear; $year<=$eyear; $year++){
for ($doy=$iday; $doy<=$eday; $doy++){
 $yyyyddd=$year*1000+$doy;
 print "$yyyyddd\n";
 @files=`ls /data/VIIRS_GLOBAL_PROCESS/tiles/T${tile_num}/night_bt_flag_${yyyyddd}*`;

 open ($fh,'>',"lst_files_T${tile_num}.txt");
 $nfiles=0;
 foreach $file (@files){
  print "$file\n";
  $nfiles++;
  $time=substr($file,-12,4);
  print "$time $nfiles\n";
  print $fh "$time\n";
  $lstfile="/data/VIIRS_GLOBAL_PROCESS/tiles/T${tile_num}/night_bt_flag_${yyyyddd}_${time}.dat.gz";
  $viewfile="/data/VIIRS_GLOBAL_PROCESS/tiles/T${tile_num}/view_angle_${yyyyddd}_${time}.dat.gz";
  `gunzip -ff $lstfile`;
  `gunzip -ff $viewfile`;
 }
 close($fh);

 # Run Merge Code
 `/data/VIIRS_GLOBAL_PROCESS/MERGE_DAY_NIGHT/merge_night_overpass_bt lst_files_T${tile_num}.txt $year $doy T$tile_num $nfiles`;
 `gzip -ff /data/VIIRS_GLOBAL_PROCESS/tiles/T${tile_num}/FINAL_NIGHT_BT_$yyyyddd.dat`;
 `gzip -ff /data/VIIRS_GLOBAL_PROCESS/tiles/T${tile_num}/FINAL_NIGHT_VIEW_$yyyyddd.dat`;
 
 foreach $file (@files){
  print "$file\n";
  $time=substr($file,-12,4);
  print "$time\n";
  print $fh "$time\n";
  $lstfile="/data/VIIRS_GLOBAL_PROCESS/tiles/T${tile_num}/night_bt_flag_${yyyyddd}_${time}.dat";
  $viewfile="/data/VIIRS_GLOBAL_PROCESS/tiles/T${tile_num}/view_angle_${yyyyddd}_${time}.dat";
  `gzip -ff $lstfile`;
  `gzip -ff $viewfile`;
 }


}
}

