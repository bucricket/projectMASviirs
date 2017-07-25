import h5py
import sys
import re

from numpy import array

filename=sys.argv[1]
geofile=sys.argv[2]
tile=sys.argv[3]
start=filename.find('_t')
time1=filename[start+2:start+6]
print tile
print (filename[start+2:start+6])
print (filename)
print (geofile)

f=h5py.File(filename,'r')
g=h5py.File(geofile,'r')
data_array = f['/All_Data/VIIRS-CM-EDR_All/QF1_VIIRSCMEDR'][()]
lat_array = g['/All_Data/VIIRS-MOD-GEO-TC_All/Latitude'][()]
lon_array = g['/All_Data/VIIRS-MOD-GEO-TC_All/Longitude'][()]

lat=array(lat_array,'float32')
lon=array(lon_array,'float32')
data=array(data_array,'uint8')

lat_filename='/raid1/sport/people/chain/VIIRS_PROCESS/grid_CM/temp_cloud_data/cloud_lat_'+tile+'_'+time1+'.dat'
lon_filename='/raid1/sport/people/chain/VIIRS_PROCESS/grid_CM/temp_cloud_data/cloud_lon_'+tile+'_'+time1+'.dat'
data_filename='/raid1/sport/people/chain/VIIRS_PROCESS/grid_CM/temp_cloud_data/cloud_'+tile+'_'+time1+'.dat'
output_lat_file=open(lat_filename,'wb')
lat.tofile(output_lat_file)
output_lat_file.close()
output_lon_file=open(lon_filename,'wb')
lon.tofile(output_lon_file)
output_lon_file.close()
output_data_file=open(data_filename,'wb')
data.tofile(output_data_file)
output_data_file.close()

