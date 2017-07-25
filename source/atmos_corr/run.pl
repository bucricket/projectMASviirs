#! /usr/bin/perl

$year=2015;
$iday=156;
$eday=162;

for ($doy=$iday; $doy<=$eday; $doy++){
 print "$year $doy\n";
 system("/usr/bin/perl run_atmos_corr.pl $year $doy T063");
}

