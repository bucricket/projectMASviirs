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
    out = {'grid_I5_path':grid_I5_path,'grid_I5_temp_path':grid_I5_temp_path,
           'agg_I5_path':agg_I5_path,'data_path':data_path,
           'cloud_grid': cloud_grid,'temp_cloud_data':temp_cloud_data,
           'agg_cloud_path':agg_cloud_path,'processing_path':processing_path,
           'static_path':static_path,'tile_base_path':tile_base_path}
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
            subprocess.check_output(["%s" % grid_I5_SDR, "%d" % lat, "%d" %  lon,
                                     "%s" % i5_data, "%s" % latfile,
                                     "%s" % lonfile, "%s" % trad_sum_fn, "%s" % trad_count_fn])
            #grid day view data
            view_sum_fn = os.path.join(grid_I5_path,'view_sum1_%03d_%s.dat' % (tile,time))
            view_count_fn = os.path.join(grid_I5_path,'view_count1_%03d_%s.dat' % (tile,time))
            subprocess.check_output(["%s" % grid_I5_SDR, "%d" % lat, "%d" %  lon,
                                     "%s" % viewfile, "%s" % latfile,
                                     "%s" % lonfile, "%s" % view_sum_fn, "%s" % view_count_fn])
    
            view_agg = os.path.join(agg_I5_path,"view_%s_%03d_%s.dat" % (date,tile,time))
            trad_agg_day = os.path.join(agg_I5_path,"day_bt11_%s_%03d_%s.dat" % (date,tile,time))
        
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
        
            subprocess.check_output(["%s" % agg_cloud,"%s" % trad_sum_fn,"%s" % trad_count_fn, "%s" % trad_agg_day ])

        else:
            #grid night I5 data
            trad_sum_fn = os.path.join(cloud_grid,'cloud_sum1_%03d_%s.dat' % (tile,time))
            trad_count_fn = os.path.join(cloud_grid,'cloud_count1_%03d_%s.dat' % (tile,time))
            subprocess.check_output(["%s" % grid_cloud_night, "%d" % lat, "%d" %  lon,
                                     "%s" % i5_data, "%s" % latfile,
                                     "%s" % lonfile, "%s" % trad_sum_fn, "%s" % trad_count_fn])

            trad_agg_night = os.path.join(agg_cloud_path,"cloud_%s_%03d_%s.dat" % (date,tile,time))
        
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
    icoordpath = os.path.join(base,"CFSR","viirs_tile_lookup_tables","CFSR_T%03d_lookup_icoord.dat" % tile)
    jcoordpath = os.path.join(base,"CFSR","viirs_tile_lookup_tables","CFSR_T%03d_lookup_jcoord.dat" % tile)
    if not os.path.exists(icoordpath):
        print("generating i and j coords...")
        subprocess.check_output(["%s" % coords, "%d" % lat, "%d" % lon, "%s" % tilestr])
        shutil.move(os.path.join(base,"CFSR_T%03d_lookup_icoord.dat" % tile), icoordpath)
        shutil.move(os.path.join(base,"CFSR_T%03d_lookup_jcoord.dat" % tile), jcoordpath)
        
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
    overpass_corr_cache = os.path.join(base,"nominal_overpass_time_tiles")
    overpass_corr_path = os.path.join(base,"overpass_corr")
    gunzip(os.path.join(overpass_corr_cache,"DAY_ZTIME_T%03d.dat.gz" % tile),
       out_fn=os.path.join(overpass_corr_path,"CURRENT_DAY_ZTIME_T%03d.dat" % tile))
    dtrad_cache = os.path.join(base,"dtrad_avg")
    gunzip(os.path.join(dtrad_cache,"DTRAD_T%03d_%d.dat.gz" % (tile,avgddd)),
       out_fn=os.path.join(overpass_corr_path,"CURRENT_DTRAD_AVG_T%03d.dat" % tile))
    tile_path = os.path.join(tile_base_path,"T%03d" % tile) 
    filelist = glob.glob(os.path.join(tile_path,'day_bt_flag_%s*T%03d*.gz' % (date,tile)))
    for i in range(len(filelist)):
        fn = filelist[i]
        time = fn.split(os.sep)[-1].split("_")[5].split(".")[0]
        view_fn = os.path.join(tile_path,"view_angle_%s_T%03d_%s.dat.gz" % (date,tile,time))
        raw_trad_fn = os.path.join(overpass_corr_path,"RAW_TRAD1_T%03d" % tile)
        out_view_fn = os.path.join(overpass_corr_path,"VIEW_ANGLE_T%03d" % tile)
        out_trad_fn = os.path.join(overpass_corr_path,"TRAD1_T%03d" % tile)
        gunzip(fn,out_fn=raw_trad_fn)
        gunzip(view_fn,out_fn= out_view_fn)
        subprocess.check_output(["%s" % offset, "%d" % year, "%03d" %  doy, "%s" % time,
                                     "T%03d" % tile, "%s" % base])
        outfn = os.path.join(tile_path,"lst_%s_T%03d_%s.dat" % (date,tile,time))
        out = subprocess.check_output(["%s" % run_correction,"T%03d" % tile, "%s" % time,
                                 "%s" % date, "%d" % year,"%s" % base, "%s" % outfn])
#        print out
        
        gzipped(outfn)
#        os.remove(raw_trad_fn)
#        os.remove(out_trad_fn)
#        os.remove(out_view_fn)
        
    filelist = glob.glob(os.path.join(tile_path,'night_bt_flag_%s*T%03d*.gz' % (date,tile)))
    for i in range(len(filelist)):
        fn = filelist[i]
        time = fn.split(os.sep)[-1].split("_")[5].split(".")[0]
        view_fn = os.path.join(tile_path,"view_angle_%s_T%03d_%s.dat.gz" % (date,tile,time))
        out_view_fn = os.path.join(overpass_corr_path,"VIEW_ANGLE_T%03d" % tile)
        out_trad_fn = os.path.join(overpass_corr_path,"TRAD1_T%03d" % tile)
        gunzip(fn,out_fn=out_trad_fn)
        gunzip(view_fn,out_fn= out_view_fn)
        outfn = os.path.join(tile_path,"lst_%s_T%03d_%s.dat" % (date,tile,time))
        subprocess.check_output(["%s" % run_correction,"T%03d" % tile, "%s" % time,
                                 "%s" % date, "%d" % year,"%s" % base, "%s" % outfn])
        
        gzipped(outfn)
        os.remove(out_trad_fn)
        os.remove(out_view_fn)
        
    
def merge_lst(tile,year,doy):
    merge = "merge_overpass"
    date = "%d%03d" % (year,doy)
    #=====create times text file=======
    merge_path = os.path.join(base,"MERGE_DAY_NIGHT")
    if not os.path.exists(merge_path):
        os.makedirs(merge_path)
    tile_path = os.path.join(tile_base_path,"T%03d" % tile) 

    #======day=============================
    lst_list_fn = os.path.join(merge_path,"lst_files_T%03d.txt" % tile)
    filelist = glob.glob(os.path.join(tile_path,'day_bt_flag_%s*T%03d*.gz' % (date,tile)))
    times = []
    nfiles = len(filelist)
    if nfiles > 0:
        for i in range(len(filelist)):
            fn = filelist[i]
            time = fn.split(os.sep)[-1].split("_")[5].split(".")[0]
            times.append(time)
            view_fn = os.path.join(tile_path,"view_angle_%s_T%03d_%s.dat.gz" % (date,tile,time))
            gunzip(view_fn)
            lst_fn = os.path.join(tile_path,"lst_%s_T%03d_%s.dat.gz" % (date,tile,time))
            gunzip(lst_fn)
        np.savetxt(lst_list_fn,times,fmt="%s" )
        out_lst_fn = os.path.join(tile_path,"FINAL_DAY_LST_%s_T%03d.dat" % (date,tile))
        out_view_fn = os.path.join(tile_path,"FINAL_DAY_VIEW_%s_T%03d.dat" % (date,tile))
        subprocess.check_output(["%s" % merge,"%s" % lst_list_fn, "%d" % year, 
                                 "%03d" % doy, "T%03d" % tile, "%d" % nfiles,
                                 "%s%s" % (tile_path, os.sep), "%s" % out_lst_fn, "%s" % out_view_fn])
        gzipped(out_lst_fn)
        gzipped(out_view_fn)
    
    #===night=======
    lst_list_fn = os.path.join(merge_path,"lst_files_T%03d.txt" % tile)
    filelist = glob.glob(os.path.join(tile_path,'night_bt_flag_%s*T%03d*.gz' % (date,tile)))
    times = []
    nfiles = len(filelist)
    if nfiles > 0:
        for i in range(len(filelist)):
            fn = filelist[i]
            time = fn.split(os.sep)[-1].split("_")[5].split(".")[0]
            times.append(time)
            view_fn = os.path.join(tile_path,"view_angle_%s_T%03d_%s.dat.gz" % (date,tile,time))
            gunzip(view_fn)
            lst_fn = os.path.join(tile_path,"lst_%s_T%03d_%s.dat.gz" % (date,tile,time))
            gunzip(lst_fn)
        np.savetxt(lst_list_fn,times,fmt="%s" )
        out_lst_fn = os.path.join(tile_path,"FINAL_NIGHT_LST_%s_T%03d.dat" % (date,tile))
        out_view_fn = os.path.join(tile_path,"FINAL_NIGHT_VIEW_%s_T%03d.dat" % (date,tile))
        subprocess.check_output(["%s" % merge,"%s" % lst_list_fn, "%d" % year, 
                                 "%03d" % doy, "T%03d" % tile, "%d" % nfiles,
                                 "%s%s" % (tile_base_path, os.sep), "%s" % out_lst_fn, "%s" % out_view_fn])
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
    parser.add_argument("tile", type=float, help="15x15 deg tile number")
    parser.add_argument("year", type=float, help="year of data")
    parser.add_argument("doy", type=str, help="day of year of data")
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
    atmosCorrection(tile,year,doy)
    merge_lst(tile,year,doy)
    pred_dtrad(tile,year,doy)

#=====convert to geotiff=================
#
#ALEXIshape = [3750,3750]
#ALEXIshape = [2880,1200]
#ALEXIres = [0.004,0.004]
#ALEXIres = [0.125,0.125]
#row = tile/24
#col = tile-(row*24)
#ULlat= (75.-(row)*15.)
#ULlon=(-180.+(col-1.)*15.)      
#inUL = [ULlon,ULlat]  
#inUL = [-180., 59.95]
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
                 