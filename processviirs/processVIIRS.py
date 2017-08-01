#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 12:35:59 2017

@author: mschull
"""
import os
import subprocess
import h5py
import numpy as np
import csv
import pandas as pd
import glob
import datetime
import gzip
import shutil
from osgeo import gdal,osr
import argparse
import urllib2, base64



def folders(base):
    data_path = os.path.abspath(os.path.join(base,os.pardir,'VIIRS_DATA'))
    if not os.path.exists(data_path):
        os.makedirs(data_path) 
    processing_path = os.path.join(base,"PROCESSING")
    if not os.path.exists(processing_path):
        os.makedirs(processing_path) 
    static_path = os.path.join(base,"STATIC")
    if not os.path.exists(static_path):
        os.makedirs(static_path) 
    tile_base_path = os.path.join(base,"TILES")
    if not os.path.exists(tile_base_path):
        os.makedirs(tile_base_path) 
    grid_I5_path = os.path.join(processing_path,'grid_i5_data')
    if not os.path.exists(grid_I5_path):
        os.makedirs(grid_I5_path) 
    grid_I5_temp_path = os.path.join(grid_I5_path,'temp_i5_data')
    if not os.path.exists(grid_I5_temp_path):
        os.makedirs(grid_I5_temp_path) 
    agg_I5_path = os.path.join(processing_path,'agg_i5_data')
    if not os.path.exists(agg_I5_path):
        os.makedirs(agg_I5_path) 
    cloud_grid = os.path.join(processing_path,'grid_CM')
    if not os.path.exists(cloud_grid):
        os.makedirs(cloud_grid) 
    agg_cloud_path = os.path.join(cloud_grid,'agg_cloud_data')
    if not os.path.exists(agg_cloud_path):
        os.makedirs(agg_cloud_path) 
    temp_cloud_data = os.path.join(cloud_grid,'temp_cloud_data')
    if not os.path.exists(temp_cloud_data):
        os.makedirs(temp_cloud_data) 
    overpass_correction_path = os.path.join(processing_path,"overpass_corr")   
    if not os.path.exists(overpass_correction_path):
        os.makedirs(overpass_correction_path)
    CFSR_path = os.path.join(processing_path,"CFSR")   
    if not os.path.exists(CFSR_path):
        os.makedirs(CFSR_path)
    out = {'grid_I5_path':grid_I5_path,'grid_I5_temp_path':grid_I5_temp_path,
           'agg_I5_path':agg_I5_path,'data_path':data_path,
           'cloud_grid': cloud_grid,'temp_cloud_data':temp_cloud_data,
           'agg_cloud_path':agg_cloud_path,'processing_path':processing_path,
           'static_path':static_path,'tile_base_path':tile_base_path,
           'overpass_correction_path':overpass_correction_path,
           'CFSR_path':CFSR_path}
    return out

base = os.getcwd()
Folders = folders(base)  
grid_I5_path = Folders['grid_I5_path']
grid_I5_temp_path = Folders['grid_I5_temp_path']
agg_I5_path = Folders['agg_I5_path']
data_path = Folders['data_path']
cloud_grid = Folders['cloud_grid']
cloud_temp_path = Folders['temp_cloud_data']
agg_cloud_path = Folders['agg_cloud_path']
processing_path = Folders['processing_path']
static_path = Folders['static_path']
tile_base_path = Folders['tile_base_path']
overpass_corr_path = Folders['overpass_correction_path']
CFSR_path = Folders['CFSR_path']


def gunzip(fn, *positional_parameters, **keyword_parameters):
    inF = gzip.GzipFile(fn, 'rb')
    s = inF.read()
    inF.close()
    if ('out_fn' in keyword_parameters):
        outF = file(keyword_parameters['out_fn'], 'wb')
    else:
        outF = file(fn[:-3], 'wb')
          
    outF.write(s)
    outF.close()
    
def gzipped(fn):
    with open(fn, 'rb') as f_in, gzip.open(fn+".gz", 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(fn)


def writeArray2Tiff(data,res,UL,inProjection,outfile,outFormat):

    xres = res[0]
    yres = res[1]

    ysize = data.shape[0]
    xsize = data.shape[1]

    ulx = UL[0] #- (xres / 2.)
    uly = UL[1]# - (yres / 2.)
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(outfile, xsize, ysize, 1, outFormat)
    #ds = driver.Create(outfile, xsize, ysize, 1, gdal.GDT_Int16)
    
    srs = osr.SpatialReference()
    
    if isinstance(inProjection, basestring):        
        srs.ImportFromProj4(inProjection)
    else:
        srs.ImportFromEPSG(inProjection)
        
    ds.SetProjection(srs.ExportToWkt())
    
    gt = [ulx, xres, 0, uly, 0, -yres ]
    ds.SetGeoTransform(gt)
    
    ds.GetRasterBand(1).WriteArray(data)
    #ds = None
    ds.FlushCache()  
    
def convertBin2tif(inFile,inUL,shape,res):
    inProj4 = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'
    outFormat = gdal.GDT_UInt16
    outFormat = gdal.GDT_Float32
    read_data = np.fromfile(inFile, dtype=np.float32)
    dataset = np.flipud(read_data.reshape([shape[0],shape[1]]))
    dataset = read_data.reshape([shape[0],shape[1]])
#    dataset = np.array(dataset*1000,dtype='uint16')
    dataset = np.array(dataset,dtype='Float32')
    outTif = inFile[:-4]+".tif"
    writeArray2Tiff(dataset,res,inUL,inProj4,outTif,outFormat) 
    
    
def get_VIIRS_bounds(fn):
    f = h5py.File(fn, 'r')
    east = []
    west = []
    north = []
    south = []
    for i in range(4):
        east = np.append(east,f['Data_Products']['VIIRS-I5-SDR']['VIIRS-I5-SDR_Gran_%d' % i ].attrs['East_Bounding_Coordinate'][0][0])
        west = np.append(west,f['Data_Products']['VIIRS-I5-SDR']['VIIRS-I5-SDR_Gran_%d' % i ].attrs['West_Bounding_Coordinate'][0][0])
        north = np.append(north,f['Data_Products']['VIIRS-I5-SDR']['VIIRS-I5-SDR_Gran_%d' % i ].attrs['North_Bounding_Coordinate'][0][0])
        south = np.append(south,f['Data_Products']['VIIRS-I5-SDR']['VIIRS-I5-SDR_Gran_%d' % i ].attrs['South_Bounding_Coordinate'][0][0])
    east = east.min()
    west = west.max()
    north = north.max()
    south = south.min()
    date = f['Data_Products']['VIIRS-I5-SDR']['VIIRS-I5-SDR_Gran_0'].attrs['Beginning_Date'][0][0]
    N_Day_Night_Flag = f['Data_Products']['VIIRS-I5-SDR']['VIIRS-I5-SDR_Gran_0'].attrs['N_Day_Night_Flag'][0][0]
    bounds = {'filename':fn,'east':east,'west':west,'north':north,'south':south,
              'N_Day_Night_Flag':N_Day_Night_Flag,'date':date}
    database = os.path.join(data_path,'I5_database.csv')
    if not os.path.exists(database):
        with open(database, 'wb') as f:  # Just use 'w' mode in 3.x
            w = csv.DictWriter(f, bounds.keys())
            w.writeheader()
            w.writerow(bounds)
    else:
        print("writing")
        with open(database, 'ab') as f:  # Just use 'w' mode in 3.x
            w = csv.DictWriter(f, bounds.keys())
            w.writerow(bounds)
            
    return 

def getTile(LLlat,LLlon):
    URlat = LLlat+15
    URlon = LLlon+15
    return np.abs(((75)-URlat)/15)*24+np.abs(((-180)-(URlon))/15)

def tile2latlon(tile):
    row = tile/24
    col = tile-(row*24)
    # find lower left corner
    lat= (75.-row*15.)-15.
    lon=(col*15.-180.)-15. 
    return [lat,lon]


def read_i5_sdr(tile,year,doy):
    lat,lon = tile2latlon(tile)
    print lat,lon
    # 
    latmid = lat+7.5
    lonmid = lon+7.5
    db = pd.read_csv(os.path.join(data_path,'I5_database.csv'))
    db = pd.DataFrame.drop_duplicates(db)
    files = db[(db['south']-5 <= latmid) & (db['north']+5 >= latmid) & 
               (db['west']-5 <= lonmid) & (db['east']+5 >= lonmid) & 
               (db['year'] == year) & (db['doy'] == doy)]
    filenames = files['filename']
    night_flags = files['N_Day_Night_Flag']
    
    for i in range(len(filenames)):    
        filename = filenames.iloc[i]
        night_flag = night_flags.iloc[i]
        folder = os.sep.join(filename.split(os.sep)[:-1])
        search_geofile = os.path.join(folder,'GITCO'+filename.split(os.sep)[-1][5:-34])
        geofile = glob.glob(search_geofile+'*')[0]
        
        start=filename.find('_t')
        time1=filename[start+2:start+6]
        
        f=h5py.File(filename,'r')
        g=h5py.File(geofile,'r')
        data_array = f['/All_Data/VIIRS-I5-SDR_All/BrightnessTemperature'][()]
        lat_array = g['/All_Data/VIIRS-IMG-GEO-TC_All/Latitude'][()]
        lon_array = g['/All_Data/VIIRS-IMG-GEO-TC_All/Longitude'][()]
        view_array = g['/All_Data/VIIRS-IMG-GEO-TC_All/SatelliteZenithAngle'][()]
        
        lat=np.array(lat_array,'float32')
        lon=np.array(lon_array,'float32')
        data=np.array(data_array,'float32')
        view=np.array(view_array,'float32')
        
        lat_filename=os.path.join(grid_I5_temp_path,'bt11_lat_%03d_%s_%s.dat' % (tile,time1,night_flag))
        lon_filename=os.path.join(grid_I5_temp_path,'bt11_lon_%03d_%s_%s.dat' % (tile,time1,night_flag))
        data_filename=os.path.join(grid_I5_temp_path,'bt11_%03d_%s_%s.dat' % (tile,time1,night_flag))
        view_filename=os.path.join(grid_I5_temp_path,'view_%03d_%s_%s.dat' % (tile,time1,night_flag))
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

def read_cloud(tile,year,doy):
    
    lat,lon = tile2latlon(tile)
    print lat,lon
    # 
    latmid = lat+7.5
    lonmid = lon+7.5
    db = pd.read_csv(os.path.join(data_path,'I5_database.csv'))
    db = pd.DataFrame.drop_duplicates(db)
    files = db[(db['south']-5 <= latmid) & (db['north']+5 >= latmid) & 
               (db['west']-5 <= lonmid) & (db['east']+5 >= lonmid) & 
               (db['year'] == year) & (db['doy'] == doy)]
    filenames = files['filename']
    night_flags = files['N_Day_Night_Flag']
    for i in range(len(filenames)):    
        filename = filenames.iloc[i]
        night_flag = night_flags.iloc[i]
        folder = os.sep.join(filename.split(os.sep)[:-1])
        search_geofile = os.path.join(folder,'GMTCO'+filename.split(os.sep)[-1][5:-34])
        search_cloudfile = os.path.join(folder,'IICMO'+filename.split(os.sep)[-1][5:-34])
        geofile = glob.glob(search_geofile+'*')[0]
        cloudfile = glob.glob(search_cloudfile+'*')[0]
        start=filename.find('_t')
        time1=filename[start+2:start+6]
        
        f=h5py.File(cloudfile,'r')
        g=h5py.File(geofile,'r')
        data_array = f['/All_Data/VIIRS-CM-IP_All/QF1_VIIRSCMIP'][()]
        lat_array = g['/All_Data/VIIRS-MOD-GEO-TC_All/Latitude'][()]
        lon_array = g['/All_Data/VIIRS-MOD-GEO-TC_All/Longitude'][()]
        
        lat=np.array(lat_array,'float32')
        lon=np.array(lon_array,'float32')
        data=np.array(data_array,'uint8')

        lat_filename=os.path.join(cloud_temp_path,'cloud_lat_%03d_%s_%s.dat' % (tile,time1,night_flag))
        lon_filename=os.path.join(cloud_temp_path,'cloud_lon_%03d_%s_%s.dat' % (tile,time1,night_flag))
        data_filename=os.path.join(cloud_temp_path,'cloud_%03d_%s_%s.dat' % (tile,time1,night_flag))
        output_lat_file=open(lat_filename,'wb')
        lat.tofile(output_lat_file)
        output_lat_file.close()
        output_lon_file=open(lon_filename,'wb')
        lon.tofile(output_lon_file)
        output_lon_file.close()
        output_data_file=open(data_filename,'wb')
        data.tofile(output_data_file)
        output_data_file.close()
        
    
    
        
def regrid_I5(tile,year,doy):
    grid_I5_SDR = "grid_I5_SDR"
    grid_I5_SDR_night = "grid_I5_SDR_night"
    agg4 = "agg4"
    agg_view = "agg_view"
    
    #find LL corner based on 15x15 deg tile numbers
    lat,lon = tile2latlon(tile)     
    print lat, lon
    # convert all available data from HDF to binary
    read_i5_sdr(tile,year,doy)

    filelist = glob.glob(os.path.join(grid_I5_temp_path,'bt11_lat_%03d*' % tile))
    for i in range(len(filelist)):
        fn = filelist[i]
        time = fn.split(os.sep)[-1].split("_")[3]
        night_flag = fn.split(os.sep)[-1].split("_")[4].split(".")[0]
        date = "%d%03d" % (year,doy)
        i5_data = os.path.join(grid_I5_temp_path,"bt11_%03d_%s_%s.dat" % (tile,time,night_flag))
        latfile = os.path.join(grid_I5_temp_path,"bt11_lat_%03d_%s_%s.dat" % (tile,time,night_flag))
        lonfile = os.path.join(grid_I5_temp_path,"bt11_lon_%03d_%s_%s.dat" % (tile,time,night_flag))
        viewfile = os.path.join(grid_I5_temp_path,"view_%03d_%s_%s.dat" % (tile,time,night_flag))
#        '/raid1/sport/people/chain/VIIRS_PROCESS/grid_I5/trad_sum1_',arg3,'.dat'
#        '/raid1/sport/people/chain/VIIRS_PROCESS/grid_I5/trad_count1_',arg3,'.dat'
        #grid day I5 data
        if night_flag == "Day":
            trad_sum_fn = os.path.join(grid_I5_path,'bt11_sum1_%03d_%s.dat' % (tile,time))
            trad_count_fn = os.path.join(grid_I5_path,'bt11_count1_%03d_%s.dat' % (tile,time))
            out = subprocess.check_output(["%s" % grid_I5_SDR, "%d" % lat, "%d" %  lon,
                                     "%s" % i5_data, "%s" % latfile,
                                     "%s" % lonfile, "%s" % trad_sum_fn, "%s" % trad_count_fn])
            print out
            #grid day view data
            view_sum_fn = os.path.join(grid_I5_path,'view_sum1_%03d_%s.dat' % (tile,time))
            view_count_fn = os.path.join(grid_I5_path,'view_count1_%03d_%s.dat' % (tile,time))
            subprocess.check_output(["%s" % grid_I5_SDR, "%d" % lat, "%d" %  lon,
                                     "%s" % viewfile, "%s" % latfile,
                                     "%s" % lonfile, "%s" % view_sum_fn, "%s" % view_count_fn])
    
            view_agg = os.path.join(agg_I5_path,"view_%s_%03d_%s.dat" % (date,tile,time))
            trad_agg_day = os.path.join(agg_I5_path,"day_bt11_%s_%03d_%s.dat" % (date,tile,time))
            if os.path.exists(trad_count_fn):            
                subprocess.check_output(["%s" % agg4,"%s" % trad_sum_fn,"%s" % trad_count_fn, "%s" % trad_agg_day ])
                subprocess.check_output(["%s" % agg_view,"%s" % view_sum_fn,"%s" % view_count_fn, "%s" % view_agg ])
                
        else:
            #grid night I5 data
            trad_sum_fn_night = os.path.join(grid_I5_path,'night_bt11_sum1_%03d_%s.dat' % (tile,time))
            trad_count_fn_night = os.path.join(grid_I5_path,'night_bt11_count1_%03d_%s.dat' % (tile,time))
            subprocess.check_output(["%s" % grid_I5_SDR_night, "%d" % lat, "%d" %  lon,
                                     "%s" % i5_data, "%s" % latfile,
                                     "%s" % lonfile, "%s" % trad_sum_fn_night, "%s" % trad_count_fn_night])
    
            #grid night view data
            view_sum_fn_night = os.path.join(grid_I5_path,'view_sum1_%03d_%s.dat' % (tile,time))
            view_count_fn_night = os.path.join(grid_I5_path,'view_count1_%03d_%s.dat' % (tile,time))
            subprocess.check_output(["%s" % grid_I5_SDR_night, "%d" % lat, "%d" % lon,
                                     "%s" % viewfile, "%s" % latfile,
                                     "%s" % lonfile, "%s" % view_sum_fn_night, "%s" % view_count_fn_night])
        
            view_agg = os.path.join(agg_I5_path,"view_%s_%03d_%s.dat" % (date,tile,time))
            trad_agg_night = os.path.join(agg_I5_path,"night_bt11_%s_%03d_%s.dat" % (date,tile,time))
            
            if os.path.exists(trad_count_fn_night):
                subprocess.check_output(["%s" % agg4,"%s" % trad_sum_fn_night,"%s" % trad_count_fn_night, "%s" % trad_agg_night ])
                subprocess.check_output(["%s" % agg_view,"%s" % view_sum_fn_night,"%s" % view_count_fn_night, "%s" % view_agg ])


def regrid_cloud(tile,year,doy):
    grid_cloud_day = "grid_cloud_day"
    grid_cloud_night = "grid_cloud_night"
    agg_cloud = "agg_cloud"
    
    #find LL corner based on 15x15 deg tile numbers
    lat,lon = tile2latlon(tile)      
    print lat, lon
    # convert all available data from HDF to binary
    read_cloud(tile,year,doy)

    filelist = glob.glob(os.path.join(grid_I5_temp_path,'bt11_lat_%03d*' % tile))
    for i in range(len(filelist)):
        fn = filelist[i]
        time = fn.split(os.sep)[-1].split("_")[3]
        night_flag = fn.split(os.sep)[-1].split("_")[4].split(".")[0]
        date = "%d%03d" % (year,doy)
        i5_data = os.path.join(cloud_temp_path,"cloud_%03d_%s_%s.dat" % (tile,time,night_flag))
        latfile = os.path.join(cloud_temp_path,"cloud_lat_%03d_%s_%s.dat" % (tile,time,night_flag))
        lonfile = os.path.join(cloud_temp_path,"cloud_lon_%03d_%s_%s.dat" % (tile,time,night_flag))

        #grid day I5 data
        if night_flag == "Day":
            trad_sum_fn = os.path.join(cloud_grid,'cloud_sum1_%03d_%s.dat' % (tile,time))
            trad_count_fn = os.path.join(cloud_grid,'cloud_count1_%03d_%s.dat' % (tile,time))
            subprocess.check_output(["%s" % grid_cloud_day, "%d" % lat, "%d" %  lon,
                                     "%s" % i5_data, "%s" % latfile,
                                     "%s" % lonfile, "%s" % trad_sum_fn, "%s" % trad_count_fn])

            trad_agg_day = os.path.join(agg_cloud_path,"cloud_%s_%03d_%s.dat" % (date,tile,time))
            if os.path.exists(trad_sum_fn):
                subprocess.check_output(["%s" % agg_cloud,"%s" % trad_sum_fn,"%s" % trad_count_fn, "%s" % trad_agg_day ])

        else:
            #grid night I5 data
            trad_sum_fn = os.path.join(cloud_grid,'cloud_sum1_%03d_%s.dat' % (tile,time))
            trad_count_fn = os.path.join(cloud_grid,'cloud_count1_%03d_%s.dat' % (tile,time))
            subprocess.check_output(["%s" % grid_cloud_night, "%d" % lat, "%d" %  lon,
                                     "%s" % i5_data, "%s" % latfile,
                                     "%s" % lonfile, "%s" % trad_sum_fn, "%s" % trad_count_fn])

            trad_agg_night = os.path.join(agg_cloud_path,"cloud_%s_%03d_%s.dat" % (date,tile,time))
            if os.path.exists(trad_sum_fn):
                subprocess.check_output(["%s" % agg_cloud,"%s" % trad_sum_fn,"%s" % trad_count_fn, "%s" % trad_agg_night ])
            
            
def Apply_mask(tile,year,doy):
    mask = "mask_cloud_water"
    tile_path = os.path.join(tile_base_path,"T%03d" % tile)
    water_data = os.path.join(static_path,"WATER_MASK","WATER_T%03d.dat" % tile)
    if not os.path.exists(tile_path):
        os.makedirs(tile_path)
    filelist = glob.glob(os.path.join(agg_I5_path,'*_bt11_*'))
    for i in range(len(filelist)):
            fn = filelist[i]
            time = fn.split(os.sep)[-1].split("_")[4].split(".")[0]
            night_flag = fn.split(os.sep)[-1].split("_")[0]
            date = "%d%03d" % (year,doy)
            cloud_data = os.path.join(agg_cloud_path,"cloud_%s_%03d_%s.dat" % (date,tile,time))        
            i5_data = os.path.join(agg_I5_path,"%s_bt11_%s_%03d_%s.dat" % (night_flag,date,tile,time))
            view_data = os.path.join(agg_I5_path,"view_%s_%03d_%s.dat" % (date,tile,time))
            out_bt_fn = os.path.join(tile_path,"%s_bt_flag_%s_T%03d_%s.dat" % (night_flag,date,tile,time))
            if os.path.exists(i5_data):
                subprocess.check_output("%s %s %s %s %s" % (mask,i5_data,cloud_data,water_data, out_bt_fn), shell=True)
                subprocess.check_output(["%s" % mask, "%s" % i5_data, "%s" %  cloud_data,
                                         "%s" % water_data, "%s" % out_bt_fn])
                gzipped(out_bt_fn)
                out_view_fn = os.path.join(tile_path,"view_angle_%s_T%03d_%s.dat" % (date,tile,time))
                subprocess.check_output(["%s" % mask, "%s" % view_data, "%s" %  cloud_data,
                                         "%s" % water_data, "%s" % out_view_fn])
                gzipped(out_view_fn)
    
def getIJcoords(tile):
    coords = "generate_lookup"
    lat,lon = tile2latlon(tile)
    tilestr = "T%03d" % tile
    tile_lut_path = os.path.join(processing_path,"CFSR","viirs_tile_lookup_tables")
    if not os.path.exists(tile_lut_path):
        os.makedirs(tile_lut_path) 
    icoordpath = os.path.join(tile_lut_path,"CFSR_T%03d_lookup_icoord.dat" % tile)
    jcoordpath = os.path.join(tile_lut_path,"CFSR_T%03d_lookup_jcoord.dat" % tile)
    if not os.path.exists(icoordpath):
        print("generating i and j coords...")
        subprocess.check_output(["%s" % coords, "%d" % lat, "%d" % lon, "%s" % tilestr])
        shutil.move(os.path.join(base,"CFSR_T%03d_lookup_icoord.dat" % tile), icoordpath)
        shutil.move(os.path.join(base,"CFSR_T%03d_lookup_jcoord.dat" % tile), jcoordpath)

ncdcURL = 'https://nomads.ncdc.noaa.gov/modeldata/cfsv2_analysis_pgbh/'

class earthDataHTTPRedirectHandler(urllib2.HTTPRedirectHandler):
    def http_error_302(self, req, fp, code, msg, headers):
        return urllib2.HTTPRedirectHandler.http_error_302(self, req, fp, code, msg, headers)
    

def getHTTPdata(url,outFN,auth=None):
    request = urllib2.Request(url) 
    if not (auth == None):
        username = auth[0]
        password = auth[1]
        base64string = base64.encodestring('%s:%s' % (username, password)).replace('\n', '')
        request.add_header("Authorization", "Basic %s" % base64string) 
    
    cookieprocessor = urllib2.HTTPCookieProcessor()
    opener = urllib2.build_opener(earthDataHTTPRedirectHandler, cookieprocessor)
    urllib2.install_opener(opener) 
    r = opener.open(request)
    result = r.read()
    
    with open(outFN, 'wb') as f:
        f.write(result)

def moveFiles(sourcepath,destinationpath,date,forcastHR,hr,ext):
    hr = hr+forcastHR
    source = os.listdir(sourcepath)
    for files in source:
        if files.endswith('.%s' % ext):
            shutil.move(os.path.join(sourcepath,files), os.path.join(destinationpath,files[:-4]+'_%s_%02d00.dat') % (date,hr))   

def write_gen_sfc_prof():
    fn = os.path.join('./gen_sfc_prof_fields.gs')
    file = open(fn, "w")
    file.write("'open current.ctl'\n")
    file.write("\n")
    file.write("'set lon -180 180'\n")
    file.write("'set lat -89.875 89.875' \n")
    file.write("\n")
    file.write("'set gxout fwrite'\n")
    file.write("'set fwrite sfc_temp.dat'\n")
    file.write("'d re(smth9(tmpsig995),0.25,0.25)'\n")
    file.write("'disable fwrite'\n")
    file.write("\n")
    file.write("'set gxout fwrite'\n")
    file.write("'set fwrite sfc_pres.dat'\n")
    file.write("'d re(smth9(pressfc),0.25,0.25)'\n")
    file.write("'disable fwrite'\n")
    file.write("\n")
    file.write("'set gxout fwrite'\n")
    file.write("'set fwrite sfc_spfh.dat'\n")
    file.write("'d re(smth9(spfhhy1),0.25,0.25)'\n")
    file.write("'disable fwrite'\n")
    file.write("\n")
    file.write("\n")
    file.write("'set gxout fwrite'\n")
    file.write("'set fwrite temp_profile.dat'\n")
    file.write("z=1\n")
    file.write("while (z <= 21)\n")
    file.write(" 'set z 'z\n")
    file.write(" 'd re(smth9(tmpprs),0.25,0.25)'\n")
    file.write(" z=z+1\n")
    file.write("endwhile\n")
    file.write("'disable fwrite'\n")
    file.write("\n")
    file.write("'set gxout fwrite'\n")
    file.write("'set fwrite spfh_profile.dat'\n")
    file.write("z=1\n")
    file.write("while (z <= 21)\n")
    file.write(" 'set z 'z\n")
    file.write(" 'd re(smth9(spfhprs),0.25,0.25)'\n")
    file.write(" z=z+1\n")
    file.write("endwhile\n")
    file.write("'disable fwrite'\n")
    file.close()


def runGrads(fn):
    g2ctl = 'g2ctl'
    gribmap = 'gribmap'
    opengrads = 'opengrads'
    gen_sfc_prof = './gen_sfc_prof_fields.gs'
    subprocess.check_output("%s %s > ./current.ctl" % (g2ctl,fn), shell=True)
    subprocess.check_output("%s -i ./current.ctl -0" % gribmap, shell=True)
    out = subprocess.check_output("%s -blxc 'run %s '" % (opengrads,gen_sfc_prof), shell=True)
    
    print out
  
def getCFSRdata(year,doy):  
    write_gen_sfc_prof()     
    levs="(100|150|200|250|300|350|400|450|500|550|600|650|700|750|800|850|900|925|950|975|1000) mb"

    
    s1="(HGT):%s" % levs
    s2="(TMP):%s" % levs
    s3="(SPFH):%s" % levs
    s4="DLWRF:surface"
    s5="HGT:surface"
    s6="PRES:surface"
    s7="SPFH:1 hybrid level"
    s8="TMP:0.995 sigma level"
    s9="UGRD:0.995 sigma level"
    s10="VGRD:0.995 sigma level"
    wgrib = "wgrib2"

#    for year in range(iyear,eyear):
#        for doy in range(iday,eday):
    dd = datetime.datetime(year, 1, 1) + datetime.timedelta(doy - 1)
    date = "%d%03d" %(year,doy)
    print "date:%s" % date
    print "============================================================"
    for i in range(0,8):
        hrs = [0,0,6,6,12,12,18,18]
        hr = hrs[i]
        forcastHRs = [0,3,0,3,0,3,0,3]
        forcastHR = forcastHRs[i]
        hr1file = 'cdas1.t%02dz.pgrbh%02d.grib2' % (hr,forcastHR)
        print "processing file...%s" % hr1file
    
        #------download file                
        pydapURL = os.path.join(ncdcURL,"%s" % year,"%d%02d" % (year,dd.month),
                                "%d%02d%02d" % (year,dd.month,dd.day),hr1file)
        outFN = os.path.join(os.getcwd(),hr1file)
        getHTTPdata(pydapURL,outFN)
        
        #------extract data
        cfsr_out = os.path.join(os.getcwd(),"CFSR_%d%02d%02d_%02d00_00%d.grib2" % (year,dd.month,dd.day,hr,forcastHR))
        
        subprocess.check_output(["%s" % wgrib, "%s" % outFN, "-match",
                                 "\"%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\"" % (s1,s2,s3,s4,s5,s6,s7,s8,s9,s10),
                                 "-grib", "%s" % cfsr_out])
        #-------process using grads------
        runGrads(cfsr_out)
        srcpath = os.getcwd()
        dstpath =  os.path.join(CFSR_path,"%d" % year)
        if not os.path.exists(dstpath):
            os.makedirs(dstpath) 
        moveFiles(srcpath,dstpath,date,forcastHR,hr,"dat")
    print "finished processing!"
        
def atmosCorrection(tile,year,doy):
    #====get week date=====
    nweek=(doy-1)/7
    cday=nweek*7
    offset=(doy-cday)/7
    rday=((offset+nweek)*7)+1
    avgddd=2014*1000+rday 
    date = "%d%03d" % (year,doy)
    #=========================
    offset = "calc_offset_correction"
    run_correction = "run_correction"
    overpass_corr_cache = os.path.join(static_path,"nominal_overpass_tiles")
#    overpass_corr_path = os.path.join(processing_path,"overpass_corr")
    ztime_fn = os.path.join(overpass_corr_path,"CURRENT_DAY_ZTIME_T%03d.dat" % tile)
    gunzip(os.path.join(overpass_corr_cache,"DAY_ZTIME_T%03d.dat.gz" % tile),
       out_fn=ztime_fn)
    dtrad_cache = os.path.join(static_path,"dtrad_avg")
    dtrad_fn =os.path.join(overpass_corr_path,"CURRENT_DTRAD_AVG_T%03d.dat" % tile)
    gunzip(os.path.join(dtrad_cache,"DTRAD_T%03d_%d.dat.gz" % (tile,avgddd)),
       out_fn=dtrad_fn)
    tile_path = os.path.join(tile_base_path,"T%03d" % tile) 
    filelist = glob.glob(os.path.join(tile_path,'day_bt_flag_%s*T%03d*.gz' % (date,tile)))
    tile_lut_path = os.path.join(CFSR_path,"viirs_tile_lookup_tables")
    cfsr_tile_path = os.path.join(CFSR_path,"%d" % year)
    for i in range(len(filelist)):
        fn = filelist[i]
        time_str = fn.split(os.sep)[-1].split("_")[5].split(".")[0]
        time=((int(time_str)/300)+1)*300
        if (int(time_str)==2400):
            time=2100

        #======io filenames============================================
        tprof = os.path.join(cfsr_tile_path,"temp_profile_%s_%04d.dat" % (date,time))
        qprof = os.path.join(cfsr_tile_path,"spfh_profile_%s_%04d.dat" % (date,time))
        tsfcfile = os.path.join(cfsr_tile_path,"sfc_temp_%s_%04d.dat" % (date,time))
        presfile = os.path.join(cfsr_tile_path,"sfc_pres_%s_%04d.dat" % (date,time))
        qsfcfile = os.path.join(cfsr_tile_path,"sfc_spfh_%s_%04d.dat" % (date,time))
        icoordpath = os.path.join(tile_lut_path,"CFSR_T%03d_lookup_icoord.dat" % tile)
        jcoordpath = os.path.join(tile_lut_path,"CFSR_T%03d_lookup_jcoord.dat" % tile)
        view_fn = os.path.join(tile_path,"view_angle_%s_T%03d_%s.dat.gz" % (date,tile,time_str))
        raw_trad_fn = os.path.join(overpass_corr_path,"RAW_TRAD1_T%03d.dat" % tile)
        trad_fn = os.path.join(overpass_corr_path,"TRAD1_T%03d.dat" % tile)
        out_view_fn = os.path.join(overpass_corr_path,"VIEW_ANGLE_T%03d.dat" % tile)
        #=============================================================
#        out_trad_fn = os.path.join(overpass_corr_path,"TRAD1_T%03d" % tile)
        gunzip(fn,out_fn=raw_trad_fn)
        gunzip(view_fn,out_fn= out_view_fn)
        subprocess.check_output(["%s" % offset, "%d" % year, "%03d" %  doy, "%s" % time_str,
                                     "T%03d" % tile, "%s" % ztime_fn, "%s" % raw_trad_fn,
                                     "%s" % dtrad_fn, "%s" % trad_fn])

        outfn = os.path.join(tile_path,"lst_%s_T%03d_%s.dat" % (date,tile,time_str))
        cmd = "%s %s %s %s %s %s %s %s %s %s %s"% (run_correction,tprof,qprof,
                                                    tsfcfile,presfile,qsfcfile,
                                                    icoordpath,jcoordpath,trad_fn,
                                                    out_view_fn,outfn)
        print cmd
        out = subprocess.check_output(["%s" % run_correction,"%s" % tprof, 
                                       "%s" % qprof,"%s" % tsfcfile,
                                       "%s" % presfile, "%s" % qsfcfile,
                                       "%s" % icoordpath, "%s" % jcoordpath,
                                       "%s" % trad_fn,"%s" % out_view_fn, "%s" % outfn])
        print out
        
        gzipped(outfn)
#        os.remove(raw_trad_fn)
#        os.remove(out_trad_fn)
#        os.remove(out_view_fn)
        
    filelist = glob.glob(os.path.join(tile_path,'night_bt_flag_%s*T%03d*.gz' % (date,tile)))
    for i in range(len(filelist)):
        fn = filelist[i]
        time_str = fn.split(os.sep)[-1].split("_")[5].split(".")[0]
        time=((int(time_str)/300)+1)*300
        if (time==2400):
            time=2100
        #======io filenames============================================
        tprof = os.path.join(cfsr_tile_path,"temp_profile_%s_%04d.dat" % (date,time))
        qprof = os.path.join(cfsr_tile_path,"spfh_profile_%s_%04d.dat" % (date,time))
        tsfcfile = os.path.join(cfsr_tile_path,"sfc_temp_%s_%04d.dat" % (date,time))
        presfile = os.path.join(cfsr_tile_path,"sfc_pres_%s_%04d.dat" % (date,time))
        qsfcfile = os.path.join(cfsr_tile_path,"sfc_spfh_%s_%04d.dat" % (date,time))
        icoordpath = os.path.join(tile_lut_path,"CFSR_T%03d_lookup_icoord.dat" % tile)
        jcoordpath = os.path.join(tile_lut_path,"CFSR_T%03d_lookup_jcoord.dat" % tile)
        view_fn = os.path.join(tile_path,"view_angle_%s_T%03d_%s.dat.gz" % (date,tile,time_str))
        raw_trad_fn = os.path.join(overpass_corr_path,"RAW_TRAD1_T%03d.dat" % tile)
        trad_fn = os.path.join(overpass_corr_path,"TRAD1_T%03d.dat" % tile)
        out_view_fn = os.path.join(overpass_corr_path,"VIEW_ANGLE_T%03d.dat" % tile)
        #=============================================================
#        out_trad_fn = os.path.join(overpass_corr_path,"TRAD1_T%03d" % tile)
        gunzip(fn,out_fn=raw_trad_fn)
        gunzip(view_fn,out_fn= out_view_fn)
        outfn = os.path.join(tile_path,"lst_%s_T%03d_%s.dat" % (date,tile,time_str))
        out = subprocess.check_output(["%s" % run_correction,"%s" % tprof, 
                                       "%s" % qprof,"%s" % tsfcfile,
                                       "%s" % presfile, "%s" % qsfcfile,
                                       "%s" % icoordpath, "%s" % jcoordpath,
                                       "%s" % trad_fn,"%s" % out_view_fn, "%s" % outfn])
        
        gzipped(outfn)
#        os.remove(out_trad_fn)
#        os.remove(out_view_fn)
        
    
def merge_lst(tile,year,doy):
    merge = "merge_overpass"
    date = "%d%03d" % (year,doy)
    #=====georeference information=====
    row = tile/24
    col = tile-(row*24)
    ULlat= (75.-(row)*15.)
    ULlon=(-180.+(col-1.)*15.)      
    inUL = [ULlon,ULlat] 
    ALEXIshape = [3750,3750]
    ALEXIres = [0.004,0.004]
    #=====create times text file=======
    merge_path = os.path.join(processing_path,"MERGE_DAY_NIGHT")
    if not os.path.exists(merge_path):
        os.makedirs(merge_path)
    tile_path = os.path.join(tile_base_path,"T%03d" % tile) 

    #======day=============================
#    lst_list_fn = os.path.join(merge_path,"lst_files_T%03d.txt" % tile)
#    view_list_fn = os.path.join(merge_path,"view_files_T%03d.txt" % tile)
    filelist = glob.glob(os.path.join(tile_path,'day_bt_flag_%s*T%03d*.gz' % (date,tile)))
#    lstfiles = []
#    viewfiles = []
    nfiles = len(filelist)
    viewout = np.empty([3750,3750,nfiles])    
    lstout = np.empty([3750,3750,nfiles]) 
    if nfiles > 0:
        for i in range(len(filelist)):
            fn = filelist[i]
            time = fn.split(os.sep)[-1].split("_")[5].split(".")[0]
#            times.append(time)
            view_fn = os.path.join(tile_path,"view_angle_%s_T%03d_%s.dat.gz" % (date,tile,time))
            gunzip(view_fn)
            view_stack = os.path.join(tile_path,"view_stack.dat")
            read_data = np.fromfile(view_fn[:-3], dtype=np.float32)
            viewout[:,:,i]= np.flipud(read_data.reshape([3750,3750]))
        
#            view.tofile(view_stack)
#            with open(view_stack, 'wb') as fout:
#                fout.write(viewout.tostring())
#            viewfiles.append(view_fn[:-3])
            
            lst_fn = os.path.join(tile_path,"lst_%s_T%03d_%s.dat.gz" % (date,tile,time))
            gunzip(lst_fn)
            lst_stack = os.path.join(tile_path,"lst_stack.dat")
            read_data = np.fromfile(lst_fn[:-3], dtype=np.float32)
            lstout[:,:,i]= np.flipud(read_data.reshape([3750,3750]))
        
        aa = np.reshape(viewout,[3750*3750,nfiles])
        aa[aa==-9999.]=9999.
        view = aa.min(axis=1)
        indcol = np.argmin(aa,axis=1)
        indrow = range(0,len(indcol))
        viewmin = np.reshape(view,[3750,3750])
        viewmin[viewmin==9999.]=-9999.
        view = np.array(viewmin,dtype='Float32')
        
        bb = np.reshape(lstout,[3750*3750,nfiles])
        lst = bb[indrow,indcol]
        lst = np.reshape(lst,[3750,3750])
        lst = np.array(lst,dtype='Float32')
        out_lst_fn = os.path.join(tile_path,"FINAL_DAY_LST_%s_T%03d.dat" % (date,tile))
        out_view_fn = os.path.join(tile_path,"FINAL_DAY_VIEW_%s_T%03d.dat" % (date,tile))
#        subprocess.check_output(["%s" % merge,"%s" % lst_list_fn, "%s" % view_list_fn, 
#                                 "%d" % nfiles,"%s" % out_lst_fn, "%s" % out_view_fn])
#        subprocess.check_output(["%s" % merge,"%s" % lst_stack, "%s" % view_stack, 
#                                 "%d" % nfiles,"%s" % out_lst_fn, "%s" % out_view_fn])
        view[view>60]= -9999.
        lst[view>60] = -9999.
        view.tofile(out_view_fn)
        lst.tofile(out_lst_fn)
        convertBin2tif(out_view_fn,inUL,ALEXIshape,ALEXIres)
        convertBin2tif(out_lst_fn,inUL,ALEXIshape,ALEXIres)
        gzipped(out_lst_fn)
        gzipped(out_view_fn)
    
    #===night=======
#    lst_list_fn = os.path.join(merge_path,"lst_files_T%03d.txt" % tile)
    filelist = glob.glob(os.path.join(tile_path,'night_bt_flag_%s*T%03d*.gz' % (date,tile)))
    nfiles = len(filelist)
    viewout = np.empty([3750,3750,nfiles])    
    lstout = np.empty([3750,3750,nfiles])
    if nfiles > 0:
        for i in range(len(filelist)):
            fn = filelist[i]
            time = fn.split(os.sep)[-1].split("_")[5].split(".")[0]
#            times.append(time)
            view_fn = os.path.join(tile_path,"view_angle_%s_T%03d_%s.dat.gz" % (date,tile,time))
            gunzip(view_fn)
            view_stack = os.path.join(tile_path,"view_stack.dat")
            read_data = np.fromfile(view_fn[:-3], dtype=np.float32)
            viewout[:,:,i]= np.flipud(read_data.reshape([3750,3750]))
        
#            view.tofile(view_stack)
#            with open(view_stack, 'wb') as fout:
#                fout.write(viewout.tostring())
#            viewfiles.append(view_fn[:-3])
            
            lst_fn = os.path.join(tile_path,"lst_%s_T%03d_%s.dat.gz" % (date,tile,time))
            gunzip(lst_fn)
            lst_stack = os.path.join(tile_path,"lst_stack.dat")
            read_data = np.fromfile(lst_fn[:-3], dtype=np.float32)
            lstout[:,:,i]= np.flipud(read_data.reshape([3750,3750]))
        aa = np.reshape(viewout,[3750*3750,nfiles])
        aa[aa==-9999.]=9999.
        view = aa.min(axis=1)
        indcol = np.argmin(aa,axis=1)
        indrow = range(0,len(indcol))
        viewmin = np.reshape(view,[3750,3750])
        viewmin[viewmin==9999.]=-9999.
        view = np.array(viewmin,dtype='Float32')
        
        bb = np.reshape(lstout,[3750*3750,nfiles])
        lst = bb[indrow,indcol]
        lst = np.reshape(lst,[3750,3750])
        lst = np.array(lst,dtype='Float32')
        out_lst_fn = os.path.join(tile_path,"FINAL_DAY_LST_%s_T%03d.dat" % (date,tile))
        out_view_fn = os.path.join(tile_path,"FINAL_DAY_VIEW_%s_T%03d.dat" % (date,tile))
#        subprocess.check_output(["%s" % merge,"%s" % lst_list_fn, "%s" % view_list_fn, 
#                                 "%d" % nfiles,"%s" % out_lst_fn, "%s" % out_view_fn])
#        subprocess.check_output(["%s" % merge,"%s" % lst_stack, "%s" % view_stack, 
#                                 "%d" % nfiles,"%s" % out_lst_fn, "%s" % out_view_fn])
        lst[view>60] = -9999.
        view[view>60]= -9999.
        view.tofile(out_view_fn)
        lst.tofile(out_lst_fn)
        convertBin2tif(out_view_fn,inUL,ALEXIshape,ALEXIres)
        convertBin2tif(out_lst_fn,inUL,ALEXIshape,ALEXIres)
        gzipped(out_lst_fn)
        gzipped(out_view_fn)
    
    #========clean up======
    files2remove = glob.glob(os.path.join(tile_path,"*.dat"))
    for fn in files2remove:
        os.remove(fn)

def pred_dtrad(tile,year,doy):
    tile_path = os.path.join(tile_base_path,"T%03d" % tile)
    final_dtrad_p250_fmax0 = 'final_dtrad_p250_fmax0'
    final_dtrad_p250_fmax20 = 'final_dtrad_p250_fmax20'
    final_dtrad_p500 = 'final_dtrad_p500'
    final_dtrad_p750 = 'final_dtrad_p750'
    final_dtrad_p1000 = 'final_dtrad_p1000'
    final_dtrad_p2000 = 'final_dtrad_p2000'
    merge = 'merge'
    calc_predicted_trad2= 'calc_predicted_trad2'
    #====create processing folder========
    dtrad_path = os.path.join(processing_path,'DTRAD_PREDICTION')
    if not os.path.exists(dtrad_path):
        os.makedirs(dtrad_path) 
    
    date = "%d%03d" % (year,doy)
    files2unzip = glob.glob(os.path.join(tile_path,"*LST_%s*.gz" % date))
    for fn in files2unzip:
        gunzip(fn)

    subprocess.check_output(["%s" % final_dtrad_p250_fmax0,"%d" % year, 
                         "%03d" % doy, "T%03d" % tile, "%s" % base ])
    subprocess.check_output(["%s" % final_dtrad_p250_fmax20,"%d" % year, 
                         "%03d" % doy, "T%03d" % tile, "%s" % base ])
    subprocess.check_output(["%s" % final_dtrad_p500,"%d" % year, 
                         "%03d" % doy, "T%03d" % tile, "%s" % base ])
    subprocess.check_output(["%s" % final_dtrad_p750,"%d" % year, 
                         "%03d" % doy, "T%03d" % tile, "%s" % base ])
    subprocess.check_output(["%s" % final_dtrad_p1000,"%d" % year, 
                         "%03d" % doy, "T%03d" % tile, "%s" % base ])
    subprocess.check_output(["%s" % final_dtrad_p2000,"%d" % year, 
                         "%03d" % doy, "T%03d" % tile, "%s" % base ])    
    subprocess.check_output(["%s" % merge,"%d" % year, "%03d" % doy,
                             "T%03d" % tile, "%s" % base])
    subprocess.check_output(["%s" % calc_predicted_trad2,"%d" % year, 
                         "%03d" % doy, "T%03d" % tile, "%s" % base ])
    lst_path = os.path.join(tile_path,
                            "FINAL_DAY_LST_TIME2_%s_T%03d.dat" % ( date, tile))
    dtrad_path = os.path.join(tile_path,
                            "FINAL_DTRAD_%s_T%03d.dat" % ( date, tile))
    gzipped(lst_path)
    gzipped(dtrad_path)
      
def main():
    # Get time and location from user
    parser = argparse.ArgumentParser()
    parser.add_argument("tile", type=int, help="15x15 deg tile number")
    parser.add_argument("year", type=int, help="year of data")
    parser.add_argument("doy", type=int, help="day of year of data")
    args = parser.parse_args()
      
    tile = args.tile
    year = args.year
    doy = args.doy
#######this should be run when downloading data########
#data_cache = os.path.join(data_path,"2016","12")
#ff = glob.glob(os.path.join(data_cache,"SVI05*"))
#for fn in ff:
#    aa = get_VIIRS_bounds(fn)
#year = 2016
#tile = 87
#dd = datetime.datetime(2016,6,1)
#doy = (dd-datetime.datetime(2016,1,1)).days
    regrid_I5(tile,year,doy)
    regrid_cloud(tile,year,doy)
    Apply_mask(tile,year,doy)
    getIJcoords(tile)
    getCFSRdata(year,doy)
    atmosCorrection(tile,year,doy)
    merge_lst(tile,year,doy)
#    pred_dtrad(tile,year,doy)

#=====convert to geotiff=================
#
#ALEXIshape = [3750,3750]
ALEXIshape = [2880,1200]
ALEXIshape = [720,1440]
#ALEXIres = [0.004,0.004]
ALEXIres = [0.25,0.25]
#row = tile/24
#col = tile-(row*24)
#ULlat= (75.-(row)*15.)
#ULlon=(-180.+(col-1.)*15.)      
#inUL = [ULlon,ULlat]  
inUL = [-180., 90.0]
#tile_path = os.path.join(base,"TILES","T%03d" % tile) 
#tile_path = os.path.join(base,"overpass_corr")
#tile_path = os.path.join(base,"CFSR","output","2016")
#files2convert = glob.glob(os.path.join(tile_path,"FINAL*"))
#files2convert = glob.glob(os.path.join(tile_path,"sfc*"))
#for fn in files2convert:
#    if fn.endswith(".gz"):
#        gunzip(fn)
#        inFile = fn[:-3]
#    else:
#        inFile = fn
#    convertBin2tif(inFile,inUL,ALEXIshape,ALEXIres)
                 