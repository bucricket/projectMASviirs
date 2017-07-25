#! /usr/bin/perl
use FindBin qw($Bin);
use File::Basename qw(dirname);
use File::Spec::Functions qw(catdir);

$temp = catdir(dirname($Bin),'grid_I5');
$data1 = catdir(dirname($Bin),'..','VIIRS_RT');

#! $dir="/usr/people/chain/viirs_process/grid_I5";
#! $temp="/raid1/sport/people/chain/VIIRS_PROCESS/grid_I5";
#! $data1="/raid1/sport/people/chain/VIIRS_RT/CONUS";
#! $data2="/raid1/sport/people/chain/VIIRS_RT/VICMO";

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
 ($fn,$y1,$y2,$x1,$x2) = split ' ', $files;
 if ($ymid >= $y1-5.0 and $ymid <= $y2+5.0 and $xmid >= $x1-5.0 and $xmid <= $x2+5.0) {
  $tag1 = substr($fn,0,25);
  $tag2 = substr($fn,-68,18);
  $geo = `ls $data1/$year/*GITCO*${tag2}*`;
  print "$ymid $xmid\n";
  print "$y1 $y2 $x1 $x2\n";
  print "$fn $geo\n";
  print "tile = $tile\n";
  $geo=trim($geo);
  $fn=trim($fn);
  system("/raid1/sport/people/chain/alexi_specific_libs/anaconda2/bin/python2.7 $dir/read_i5_sdr.py $fn $geo $tile");
 }
}

@pfiles=`ls $temp/temp_i5_data/bt11_lat_${tile}*`;
foreach $pfiles (@pfiles) {
 chomp($pfiles);
 print "$pfiles\n";
 $time = substr($pfiles,-8,4);
 print "$time\n";
 $file1="$temp/temp_i5_data/bt11_${tile}_$time.dat";
 $latfile="$temp/temp_i5_data/bt11_lat_${tile}_$time.dat";
 $lonfile="$temp/temp_i5_data/bt11_lon_${tile}_$time.dat";
 $viewfile="$temp/temp_i5_data/view_${tile}_$time.dat";
 print "$file1 $latfile $lonfile $viewfile\n";
 system("$dir/grid_I5_SDR $ilat1 $ilon1 $tile $file1 $latfile $lonfile");
 system("mv $temp/trad_sum1_$tile.dat $temp/grid_i5_data/bt11_sum1_${tile}_$time.dat");
 system("mv $temp/trad_count1_$tile.dat $temp/grid_i5_data/bt11_count1_${tile}_$time.dat");
 system("$dir/grid_I5_SDR_night $ilat1 $ilon1 $tile $file1 $latfile $lonfile");
 system("mv $temp/night_trad_sum1_$tile.dat $temp/grid_i5_data/night_bt11_sum1_${tile}_$time.dat");
 system("mv $temp/night_trad_count1_$tile.dat $temp/grid_i5_data/night_bt11_count1_${tile}_$time.dat");
 system("$dir/grid_I5_SDR $ilat1 $ilon1 $tile $viewfile $latfile $lonfile");
 system("mv $temp/trad_sum1_$tile.dat $temp/grid_i5_data/view_sum1_${tile}_$time.dat");
 system("mv $temp/trad_count1_$tile.dat $temp/grid_i5_data/view_count1_${tile}_$time.dat");
 system("$dir/grid_I5_SDR_night $ilat1 $ilon1 $tile $viewfile $latfile $lonfile");
 system("mv $temp/night_trad_sum1_$tile.dat $temp/grid_i5_data/view_sum1_${tile}_$time.dat");
 system("mv $temp/night_trad_count1_$tile.dat $temp/grid_i5_data/view_count1_${tile}_$time.dat");

 if (-e "$temp/grid_i5_data/bt11_sum1_${tile}_$time.dat") {
  system("$dir/agg4 $temp/grid_i5_data/bt11_sum1_${tile}_$time.dat $temp/grid_i5_data/bt11_count1_${tile}_$time.dat $tile");
  system("mv $temp/agg_${tile}.dat $temp/agg_i5_data/day_bt11_${yyyyddd}_${tile}_$time.dat");
  system("$dir/agg_view $temp/grid_i5_data/view_sum1_${tile}_$time.dat $temp/grid_i5_data/view_count1_${tile}_$time.dat $tile");
  system("mv $temp/agg_${tile}.dat $temp/agg_i5_data/view_${yyyyddd}_${tile}_$time.dat");
 }
 if (-e "$temp/grid_i5_data/night_bt11_sum1_${tile}_$time.dat") {
  system("$dir/agg4 $temp/grid_i5_data/night_bt11_sum1_${tile}_$time.dat $temp/grid_i5_data/night_bt11_count1_${tile}_$time.dat $tile");
  system("mv $temp/agg_$tile.dat $temp/agg_i5_data/night_bt11_${yyyyddd}_${tile}_$time.dat");
  system("$dir/agg_view $temp/grid_i5_data/view_sum1_${tile}_$time.dat $temp/grid_i5_data/view_count1_${tile}_$time.dat $tile");
  system("mv $temp/agg_$tile.dat $temp/agg_i5_data/view_${yyyyddd}_${tile}_$time.dat");

 }
}

#system("rm $dir/temp_i5_data/*.dat");
#system("rm $dir/grid_i5_data/*.dat");




#-----------------------------------------------------------------------
#                                                               GET_DATE

sub trim($)
{
        my $string = shift;
        $string =~ s/^\s+//;
        $string =~ s/\s+$//;
        return $string;
}


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



