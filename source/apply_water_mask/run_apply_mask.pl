#! /usr/bin/perl

$homedir="/usr/people/chain/viirs_process";
$datadir="/raid1/sport/people/chain/VIIRS_PROCESS";

$year=@ARGV[0];
$doy=@ARGV[1];
$lat1=@ARGV[2];
$lon1=@ARGV[3];

$lat2=$lat1+15;
$lon2=$lon1+15;

$yyyyddd=$year*1000+$doy;
# calculate tile number
$tile=abs(((75)-$lat2)/15)*24+abs(((-180)-$lon2)/15);
print "Processing Tile Number: $tile\n";
print "Lat Bounds: $lat1 $lat2\n";
print "Lon Bounds: $lon1 $lon2\n";


if ($tile < 10) {
 $waterfile="$datadir/WATER_MASK/WATER_T00$tile.dat";
 $tile_str = "T00$tile";
}
if ($tile >= 10 and $tile < 100){
 $waterfile="$datadir/WATER_MASK/WATER_T0$tile.dat";
 $tile_str = "T0$tile";
}
if ($tile >= 100){
 $waterfile="$datadir/WATER_MASK/WATER_T$tile.dat";
 $tile_str = "T$tile";
}

#system("/usr/bin/perl $homedir/grid_CM/run_cloud_extract.pl $year $doy $lat1 $lat2 $lon1 $lon2 $tile_str");
#system("/usr/bin/perl $homedir/grid_I5/run_i5_sdr_extract.pl $year $doy $lat1 $lat2 $lon1 $lon2 $tile_str");

@btfile=`ls $datadir/grid_I5/agg_i5_data/*bt11*$tile_str*`;
print "@btfile\n";
foreach $btfile (@btfile){
 chomp($btfile);
 print "$btfile\n";
 $time = substr($btfile,-8,4);
 $daynight = substr($btfile,-30,1);
 print "$daynight\n";
 $cloudfile="$datadir/grid_CM/agg_cloud_data/cloud_${yyyyddd}_${tile_str}_${time}.dat";
 print "cloud = $cloudfile $time\n";
 system("$homedir/apply_water_mask/mask_cloud_water $btfile $cloudfile $waterfile $tile_str");
 if ($daynight eq 'd') {
  `mv $datadir/bt_flag_$tile_str.dat $datadir/TILES/$tile_str/day_bt_flag_${yyyyddd}_${tile_str}_${time}.dat`;
 }
 if ($daynight eq 'g') {
  `mv $datadir/bt_flag_$tile_str.dat $datadir/TILES/$tile_str/night_bt_flag_${yyyyddd}_${tile_str}_${time}.dat`;
 }
}

@btfile=`ls $datadir/grid_I5/agg_i5_data/*view*$tile_str*`;
foreach $btfile (@btfile){
 chomp($btfile);
 print "$btfile\n";
 $time = substr($btfile,-8,4);
 $cloudfile="$datadir/grid_CM/agg_cloud_data/cloud_${yyyyddd}_${tile_str}_${time}.dat";
 print "cloud = $cloudfile $time\n";
 system("$homedir/apply_water_mask/mask_cloud_water $btfile $cloudfile $waterfile $tile_str");
 `mv $datadir/bt_flag_$tile_str.dat $datadir/TILES/$tile_str/view_angle_${yyyyddd}_${tile_str}_${time}.dat`;
}

#system("rm /data/VIIRS_GLOBAL_PROCESS/src/grid_I5/agg_i5_data/*$tile_str*.dat");
#system("rm /data/VIIRS_GLOBAL_PROCESS/src/grid_I5/grid_i5_data/*$tile_str*.dat");
#system("rm /data/VIIRS_GLOBAL_PROCESS/src/grid_I5/temp_i5_data/*$tile_str*.dat");
#system("rm /data/VIIRS_GLOBAL_PROCESS/src/grid_CM/agg_cloud_data/*$tile_str*.dat");
#system("rm /data/VIIRS_GLOBAL_PROCESS/src/grid_CM/grid_cloud_data/*$tile_str*.dat");
#system("rm /data/VIIRS_GLOBAL_PROCESS/src/grid_CM/temp_cloud_data/*$tile_str*.dat");
#system("gzip -ff /data/VIIRS_GLOBAL_PROCESS/tiles/$tile_str/*${yyyyddd}*.dat");
