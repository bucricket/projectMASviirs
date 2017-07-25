#! /usr/bin/perl

$dir="/usr/people/chain/viirs_process/grid_CM";
$temp="/raid1/sport/people/chain/VIIRS_PROCESS/grid_CM";
$data1="/raid1/sport/people/chain/VIIRS_RT/CONUS";
$data2="/raid1/sport/people/chain/VIIRS_RT/VICMO";

print "@ARGV\n";
$year=@ARGV[0];
$doy=@ARGV[1];
$ilat1=@ARGV[2];
$ilat2=@ARGV[3];
$ilon1=@ARGV[4];
$ilon2=@ARGV[5];
$tile=@ARGV[6];
$ymid=($ilat1+$ilat2)/2.;
$xmid=($ilon1+$ilon2)/2.;

print "$ilat1 $ilat2 $ilon1 $ilon2\n";
print "$xmid $ymid\n";
&GetDate;
$yyyyddd=$year*1000+$doy;
print "$date3\n";

open BBOX, "</raid1/sport/people/chain/VIIRS_PROCESS/bounding_box/CURRENT/bounding_box_${yyyyddd}.dat" or die;
@files = <BBOX>;
foreach $files (@files){
 chomp($files);
 ($fn,$y1,$y2,$x1,$x2,$t) = split ' ', $files;
 if ($ymid >= $y1-5.0 and $ymid <= $y2+5.0 and $xmid >= $x1-5.0 and $xmid <= $x2+5.0) {
  print "$fn\n";
  $tag1 = substr($fn,0,25);
  $tag2 = substr($fn,-68,18);
  print "$tag1 $tag2\n";
  $geo = `ls $data1/$year/*GMTCO*${tag2}*`;
  $cld = `ls $data2/*VICMO*${tag2}*`;
  chomp($geo);
  chomp($cld);
  print "$ymid $xmid\n";
  print "$y1 $y2 $x1 $x2 $t\n";
  print "$cld $geo\n";
  print "tile = $tile\n";
  system("/raid1/sport/people/chain/alexi_specific_libs/anaconda2/bin/python2.7 $dir/read_cm.py $cld $geo $tile");
 }
}

@pfiles=`ls $temp/temp_cloud_data/cloud_lat_${tile}*`;
foreach $pfiles (@pfiles) {
 chomp($pfiles);
 print "$pfiles\n";
 $time = substr($pfiles,-8,4);
 print "time = $time\n";
 $file1="$temp/temp_cloud_data/cloud_${tile}_$time.dat";
 $latfile="$temp/temp_cloud_data/cloud_lat_${tile}_$time.dat";
 $lonfile="$temp/temp_cloud_data/cloud_lon_${tile}_$time.dat";
 print "$file1 $latfile $lonfile\n"; 
 system("$dir/grid_cloud_day $ilat1 $ilon1 $tile $file1 $latfile $lonfile");
 system("mv $temp/cloud_sum1_$tile.dat $temp/grid_cloud_data/cloud_sum1_${tile}_$time.dat");
 system("mv $temp/cloud_count1_$tile.dat $temp/grid_cloud_data/cloud_count1_${tile}_$time.dat");
 system("$dir/grid_cloud_night $ilat1 $ilon1 $tile $file1 $latfile $lonfile");
 system("mv $temp/night_cloud_sum1_$tile.dat $temp/grid_cloud_data/night_cloud_sum1_${tile}_$time.dat");
 system("mv $temp/night_cloud_count1_$tile.dat $temp/grid_cloud_data/night_cloud_count1_${tile}_$time.dat");
 if (-e "$temp/grid_cloud_data/cloud_sum1_${tile}_$time.dat") {
  system("$dir/agg_cloud $temp/grid_cloud_data/cloud_sum1_${tile}_$time.dat $temp/grid_cloud_data/cloud_count1_${tile}_$time.dat $tile");
  system("mv $temp/agg_$tile.dat $temp/agg_cloud_data/cloud_${yyyyddd}_${tile}_$time.dat");
 }
 if (-e "$temp/grid_cloud_data/night_cloud_sum1_${tile}_$time.dat") {
  print "exists\n";
  system("$dir/agg_cloud $temp/grid_cloud_data/night_cloud_sum1_${tile}_$time.dat $temp/grid_cloud_data/night_cloud_count1_${tile}_$time.dat $tile");
  system("mv $temp/agg_$tile.dat $temp/agg_cloud_data/cloud_${yyyyddd}_${tile}_$time.dat");
 }

}

#system("rm $dir/temp_cloud_data/*.dat");
#system("rm $dir/grid_cloud_data/*.dat");




#-----------------------------------------------------------------------
#                                                               GET_DATE
sub GetDate {

@month=(Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec);
if ( $year % 4 != 0 ) {
  @nday=(31,28,31,30,31,30,31,31,30,31,30,31);
} else {
  @nday=(31,29,31,30,31,30,31,31,30,31,30,31);
}

$isum=0;
for ($i=0; $i<=12; $i++) {
  $istr[$i]=$isum;
  last if ($istr[$i]>=$doy);
  $isum+=$nday[$i];
}

$im=$i-1;
$id=$doy-$istr[$im];

if ($id<10) {
  $dom = "0$id";
} else { 
  $dom = "$id";
}

$imm=$im+1;
if ($imm<10) {
  $moy = "0$imm";
} else { 
  $moy = "$imm";
}

$yr = $year % 100;
if ($yr<10) {
  $yr = "0$yr";
}

$date3 = "$year$moy$dom";
 
}



