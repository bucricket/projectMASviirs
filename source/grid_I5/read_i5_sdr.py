import h5py
import sys

from numpy import array
print sys.argv
filename=sys.argv[1]
geofile=sys.argv[2]
tile=sys.argv[3]
start=filename.find('_t')
time1=filename[start+2:start+6]

f=h5py.File(filename,'r')
g=h5py.File(geofile,'r')
data_array = f['/All_Data/VIIRS-I5-SDR_All/BrightnessTemperature'][()]
lat_array = g['/All_Data/VIIRS-IMG-GEO-TC_All/Latitude'][()]
lon_array = g['/All_Data/VIIRS-IMG-GEO-TC_All/Longitude'][()]
view_array = g['/All_Data/VIIRS-IMG-GEO-TC_All/SatelliteZenithAngle'][()]

lat=array(lat_array,'float32')
lon=array(lon_array,'float32')
data=array(data_array,'float32')
view=array(view_array,'float32')

lat_filename='/raid1/sport/people/chain/VIIRS_PROCESS/grid_I5/temp_i5_data/bt11_lat_'+tile+'_'+time1+'.dat'
lon_filename='/raid1/sport/people/chain/VIIRS_PROCESS/grid_I5/temp_i5_data/bt11_lon_'+tile+'_'+time1+'.dat'
data_filename='/raid1/sport/people/chain/VIIRS_PROCESS/grid_I5/temp_i5_data/bt11_'+tile+'_'+time1+'.dat'
view_filename='/raid1/sport/people/chain/VIIRS_PROCESS/grid_I5/temp_i5_data/view_'+tile+'_'+time1+'.dat'
output_lat_file=open(lat_filename,'wb')
lat.tofile(output_lat_file)
output_lat_file.close()
output_lon_file=open(lon_filename,'wb')
lon.tofile(output_lon_file)
output_lon_file.close()
output_data_file=open(data_filename,'wb')
data.tofile(output_data_file)
output_data_file.close()
output_view_file=open(view_filename,'wb')
view.tofile(output_view_file)
output_view_file.close()


