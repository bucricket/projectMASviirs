#! /usr/bin/perl

$homedir="/usr/people/chain/viirs_process/merge_day_night";
$tiledir="/raid1/sport/people/chain/VIIRS_PROCESS/TILES";

$year=@ARGV[0];
$doy=@ARGV[1];
$tile_num=@ARGV[2];

$yyyyddd=$year*1000+$doy;
 print "$yyyyddd\n";
 @files=`ls $tiledir/${tile_num}/night_bt_flag_${yyyyddd}_$tile_num*`;

 open ($fh,'>',"$homedir/lst_files_${tile_num}.txt");
 $nfiles=0;
 foreach $file (@files){
  print "$file\n";
  $nfiles++;
  $time=substr($file,-12,4);
  print "$time $nfiles\n";
  print $fh "$time\n";
  $lstfile="$tiledir/${tile_num}/lst_${yyyyddd}_${tile_num}_${time}.dat.gz";
  $viewfile="$tiledir/${tile_num}/view_angle_${yyyyddd}_${tile_num}_${time}.dat.gz";
  `gunzip -ff $lstfile`;
  `gunzip -ff $viewfile`;
 }
 close($fh);

 # Run Merge Code
 `$homedir/merge_night_overpass $homedir/lst_files_${tile_num}.txt $year $doy $tile_num $nfiles`;
 `gzip -ff $tiledir/${tile_num}/FINAL_NIGHT_LST_${yyyyddd}_${tile_num}.dat`;
 `gzip -ff $tiledir/${tile_num}/FINAL_NIGHT_VIEW_${yyyyddd}_${tile_num}.dat`;
 
 foreach $file (@files){
  print "$file\n";
  $time=substr($file,-12,4);
  print "$time\n";
  print $fh "$time\n";
  $lstfile="$tiledir/${tile_num}/lst_${yyyyddd}_${tile_num}_${time}.dat";
  $viewfile="$tiledir/${tile_num}/view_angle_${yyyyddd}_${tile_num}_${time}.dat";
  `gzip -ff $lstfile`;
  `gzip -ff $viewfile`;
 }


