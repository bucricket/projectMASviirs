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
from osgeo.gdalconst import GA_ReadOnly
import argparse
import time as timer
import ephem
from joblib import Parallel, delayed
from pyresample import image, kd_tree, geometry
from pyresample import utils
import numpy.ma as ma




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
    calc_rnet_path = os.path.join(processing_path,'CALC_RNET')
    if not os.path.exists(calc_rnet_path):
        os.makedirs(calc_rnet_path)
    overpass_correction_path = os.path.join(processing_path,"overpass_corr")   
    if not os.path.exists(overpass_correction_path):
        os.makedirs(overpass_correction_path)
    CFSR_path = os.path.join(static_path,"CFSR")   
    if not os.path.exists(CFSR_path):
        os.makedirs(CFSR_path)
    fsun_trees_path = os.path.join(processing_path,'FSUN_TREES')
    if not os.path.exists(fsun_trees_path):
        os.makedirs(fsun_trees_path)
    out = {'grid_I5_path':grid_I5_path,'grid_I5_temp_path':grid_I5_temp_path,
           'agg_I5_path':agg_I5_path,'data_path':data_path,
           'cloud_grid': cloud_grid,'temp_cloud_data':temp_cloud_data,
           'agg_cloud_path':agg_cloud_path,'processing_path':processing_path,
           'static_path':static_path,'tile_base_path':tile_base_path,
           'overpass_correction_path':overpass_correction_path,
           'CFSR_path':CFSR_path,'calc_rnet_path':calc_rnet_path,
           'fsun_trees_path':fsun_trees_path}
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
calc_rnet_path = Folders['calc_rnet_path']
fsun_trees_path = Folders['fsun_trees_path']


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
    
def convertBin2tif(inFile,inUL,shape,res,informat,outFormat):
    inProj4 = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'
#    outFormat = gdal.GDT_UInt16
#    outFormat = gdal.GDT_Float32
    read_data = np.fromfile(inFile, dtype=informat)
#    dataset = np.flipud(read_data.reshape([shape[0],shape[1]]))
    dataset = read_data.reshape([shape[0],shape[1]])
#    dataset = np.array(dataset*1000,dtype='uint16')
    dataset = np.array(dataset,dtype=informat)
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

def gridMergePython(tile,year,doy):
    tile_path = os.path.join(tile_base_path,"T%03d" % tile) 
    if not os.path.exists(tile_path):
        os.makedirs(tile_path)
    dd = datetime.datetime(year, 1, 1) + datetime.timedelta(doy - 1)
    date = "%d%03d" % (year,doy)
    LLlat,LLlon = tile2latlon(tile)
    URlat = LLlat+15.
    URlon = LLlon+15.
    inUL = [LLlon,URlat]
    ALEXIshape = [3750,3750]
    ALEXIres = [0.004,0.004]

    latmid = LLlat+7.5
    lonmid = LLlon+7.5
    db = pd.read_csv(os.path.join(data_path,'I5_database.csv'))
    db = pd.DataFrame.drop_duplicates(db)
    
    
    ###Day 
    files = db[(db['south']-5 <= latmid) & (db['north']+5 >= latmid) & 
               (db['west']-5 <= lonmid) & (db['east']+5 >= lonmid) & 
               (db['year'] == year) & (db['doy'] == doy) & (db['N_Day_Night_Flag'] == 'Day')]
    filenames = files['filename']
#    night_flags = files['N_Day_Night_Flag']
    
    mergedata =np.array([])
    mergelat = np.array([])
    mergelon = np.array([])
    mergeview = np.array([])
    mergecloudlat = np.array([])
    mergecloudlon = np.array([])
    mergecloud = np.array([])
    for i in range(len(filenames)):    
        filename = filenames.iloc[i]
#        night_flag = night_flags.iloc[i]
        folder = os.sep.join(filename.split(os.sep)[:-1])
        parts = filename.split(os.sep)[-1].split('_')
        search_geofile = os.path.join(folder,"*"+"_".join(("GITCO",parts[1],parts[2],parts[3],parts[4])))
#        search_geofile = os.path.join(folder,'GITCO'+filename.split(os.sep)[-1][5:-34])
        geofile = glob.glob(search_geofile+'*')[0]
#        print(filename)
#        print(geofile)
        search_geofile = os.path.join(folder,"*"+"_".join(("GMTCO",parts[1],parts[2],parts[3],parts[4])))
        datet = datetime.datetime(year,dd.month, dd.day,0,0,0)
        if datet > datetime.datetime(2017,3,8,0,0,0):
           search_cloudfile = os.path.join(folder,"*"+"_".join(("VICMO",parts[1],parts[2],parts[3],parts[4]))) 
        else:
           search_cloudfile = os.path.join(folder,"*"+"_".join(("IICMO",parts[1],parts[2],parts[3],parts[4])))
        cloudgeofile = glob.glob(search_geofile+'*')[0]
        cloudfile = glob.glob(search_cloudfile+'*')[0]
        
        f=h5py.File(cloudfile,'r')
        g=h5py.File(cloudgeofile,'r')
        data_array = f['/All_Data/VIIRS-CM-IP_All/QF1_VIIRSCMIP'][()]
        lat_array = g['/All_Data/VIIRS-MOD-GEO-TC_All/Latitude'][()]
        lon_array = g['/All_Data/VIIRS-MOD-GEO-TC_All/Longitude'][()]
        
        latcloud=np.array(lat_array,'float32')
        loncloud=np.array(lon_array,'float32')
        cloud=np.array(data_array,'uint8')
        
        start=filename.find('_t')
        out_time=filename[start+2:start+6]
        
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
        vals = data[np.where((lat>LLlat) & (lat <=URlat) & (lon>LLlon) & (lon<=URlon)
        & (abs(view)<60.0))]
        lats = lat[np.where((lat>LLlat) & (lat <=URlat) & (lon>LLlon) & (lon<=URlon)
        & (abs(view)<60.0))]
        lons = lon[np.where((lat>LLlat) & (lat <=URlat) & (lon>LLlon) & (lon<=URlon)
        & (abs(view)<60.0))]
        views = view[np.where((lat>LLlat) & (lat <=URlat) & (lon>LLlon) & (lon<=URlon)
        & (abs(view)<60.0))]
        cloudlats = latcloud[np.where((latcloud>LLlat) & (latcloud <=URlat) & (loncloud>LLlon) & (loncloud<=URlon))]
        cloudlons = loncloud[np.where((latcloud>LLlat) & (latcloud <=URlat) & (loncloud>LLlon) & (loncloud<=URlon))]
        clouds = cloud[np.where((latcloud>LLlat) & (latcloud <=URlat) & (loncloud>LLlon) & (loncloud<=URlon))]
#        print(len(vals))
#        print(len(lats))
#        print(len(lons))
#        print(len(views))
        mergedata = np.append(mergedata,vals)
        mergelat = np.append(mergelat,lats)
        mergelon = np.append(mergelon,lons)
        mergeview = np.append(mergeview,views)
        mergecloudlat = np.append(mergecloudlat,cloudlats)
        mergecloudlon = np.append(mergecloudlon,cloudlons)
        mergecloud = np.append(mergecloud,clouds)
    res=0
    if mergelat.any():
        #get 2-3 bits
        mergecloud=np.array(mergecloud,'uint8')
        mergecloud = np.reshape(mergecloud,[mergecloud.size, 1])
        b = np.unpackbits(mergecloud, axis=1)
        mergecloud = np.sum(b[:,4:6],axis=1)
        mergelat = ma.array(mergelat, mask = (mergedata==65533.))
        mergelon = ma.array(mergelon, mask = (mergedata==65533.))
#        mergedata = ma.array(mergedata, mask = (mergedata==65533.))
        mergecloudlat = ma.array(mergecloudlat, mask = (mergecloud==0.))
        mergecloudlon = ma.array(mergecloudlon, mask = (mergecloud==0.))
#        mergecloud = ma.array(mergecloud, mask = (mergecloud==0.))
        
        projection = '+proj=longlat +ellps=WGS84 +datum=WGS84'
        area_id ='tile'
        proj_id = 'latlon'
        description = 'lat lon grid'
        x_size = 3750
        y_size = 3750
        area_extent = (LLlon,LLlat,URlon,URlat)
        area_def = utils.get_area_def(area_id, description, proj_id, projection,
                                               x_size, y_size, area_extent)

        swath_def = geometry.SwathDefinition(lons=mergecloudlon, lats=mergecloudlat)
#        swath_points_in_grid, cols, rows = ll2cr(swath_def, area_def)
#        num_valid_points, gridded_data = fornav(cols, rows, area_def, mergedata, rows_per_scan=32)
        swath_con = image.ImageContainerNearest(mergecloud, swath_def, radius_of_influence=5000)
        area_con = swath_con.resample(area_def)
        cloud = area_con.image_data

        
        swath_def = geometry.SwathDefinition(lons=mergelon, lats=mergelat)
#        swath_points_in_grid, cols, rows = ll2cr(swath_def, area_def)
#        num_valid_points, gridded_data = fornav(cols, rows, area_def, mergedata, rows_per_scan=32)
        swath_con = image.ImageContainerNearest(mergedata, swath_def, radius_of_influence=5000)
        area_con = swath_con.resample(area_def)
        lst = area_con.image_data*0.00351+150.0
#        gridded_data = kd_tree.resample_gauss(swath_def, mergedata,area_def, radius_of_influence=50000, sigmas=25000)
#        lst = gridded_data*0.00351+150.0
        lst[lst==150]=-9999.
        lst[cloud>1]=-9999.
        lst=np.array(lst,'float32')
        out_bt_fn = os.path.join(tile_path,"merged_day_bt_%s_T%03d_%s.dat" % (date,tile,out_time))
        lst.tofile(out_bt_fn)
        convertBin2tif(out_bt_fn,inUL,ALEXIshape,ALEXIres,'float32',gdal.GDT_Float32) 
        
        
#        swath_points_in_grid, cols, rows = ll2cr(swath_def, area_def)
#        num_valid_points, gridded_data = fornav(cols, rows, area_def, mergeview, rows_per_scan=32)
        swath_con = image.ImageContainerNearest(mergeview, swath_def, radius_of_influence=5000)
        area_con = swath_con.resample(area_def)
        view = area_con.image_data
#        gridded_data = kd_tree.resample_gauss(swath_def, mergeview,area_def, radius_of_influence=50000, sigmas=25000)
#        view = gridded_data
        out_view_fn =os.path.join(tile_path,"merged_day_view_%s_T%03d_%s.dat" % (date,tile,out_time))
        view[view==0]=-9999.
        view[cloud>1]=-9999.
        view=np.array(view,'float32')
        view.tofile(out_view_fn)
    else:
        res+=1

    ###Night 
    files = db[(db['south']-5 <= latmid) & (db['north']+5 >= latmid) & 
               (db['west']-5 <= lonmid) & (db['east']+5 >= lonmid) & 
               (db['year'] == year) & (db['doy'] == doy) & (db['N_Day_Night_Flag'] == 'Night')]
    filenames = files['filename']
#    night_flags = files['N_Day_Night_Flag']
    
    mergedata =np.array([])
    mergelat = np.array([])
    mergelon = np.array([])
    mergeview = np.array([])
    mergecloudlat = np.array([])
    mergecloudlon = np.array([])
    mergecloud = np.array([])
    for i in range(len(filenames)):    
        filename = filenames.iloc[i]
#        night_flag = night_flags.iloc[i]
        folder = os.sep.join(filename.split(os.sep)[:-1])
        parts = filename.split(os.sep)[-1].split('_')
        search_geofile = os.path.join(folder,"*"+"_".join(("GITCO",parts[1],parts[2],parts[3],parts[4])))
#        search_geofile = os.path.join(folder,'GITCO'+filename.split(os.sep)[-1][5:-34])
        geofile = glob.glob(search_geofile+'*')[0]
#        print(filename)
#        print(geofile)
        search_geofile = os.path.join(folder,"*"+"_".join(("GMTCO",parts[1],parts[2],parts[3],parts[4])))
        datet = datetime.datetime(year,dd.month, dd.day,0,0,0)
        if datet > datetime.datetime(2017,3,8,0,0,0):
           search_cloudfile = os.path.join(folder,"*"+"_".join(("VICMO",parts[1],parts[2],parts[3],parts[4]))) 
        else:
           search_cloudfile = os.path.join(folder,"*"+"_".join(("IICMO",parts[1],parts[2],parts[3],parts[4])))
        cloudgeofile = glob.glob(search_geofile+'*')[0]
        cloudfile = glob.glob(search_cloudfile+'*')[0]
        
        f=h5py.File(cloudfile,'r')
        g=h5py.File(cloudgeofile,'r')
        data_array = f['/All_Data/VIIRS-CM-IP_All/QF1_VIIRSCMIP'][()]
        lat_array = g['/All_Data/VIIRS-MOD-GEO-TC_All/Latitude'][()]
        lon_array = g['/All_Data/VIIRS-MOD-GEO-TC_All/Longitude'][()]
        
        latcloud=np.array(lat_array,'float32')
        loncloud=np.array(lon_array,'float32')
        cloud=np.array(data_array,'uint8')
        
        start=filename.find('_t')
        out_time=filename[start+2:start+6]
        
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
        vals = data[np.where((lat>LLlat) & (lat <=URlat) & (lon>LLlon) & (lon<=URlon)
        & (abs(view)<60.0))]
        lats = lat[np.where((lat>LLlat) & (lat <=URlat) & (lon>LLlon) & (lon<=URlon)
        & (abs(view)<60.0))]
        lons = lon[np.where((lat>LLlat) & (lat <=URlat) & (lon>LLlon) & (lon<=URlon)
        & (abs(view)<60.0))]
        views = view[np.where((lat>LLlat) & (lat <=URlat) & (lon>LLlon) & (lon<=URlon)
        & (abs(view)<60.0))]
        cloudlats = latcloud[np.where((latcloud>LLlat) & (latcloud <=URlat) & (loncloud>LLlon) & (loncloud<=URlon))]
        cloudlons = loncloud[np.where((latcloud>LLlat) & (latcloud <=URlat) & (loncloud>LLlon) & (loncloud<=URlon))]
        clouds = cloud[np.where((latcloud>LLlat) & (latcloud <=URlat) & (loncloud>LLlon) & (loncloud<=URlon))]
        
        mergedata = np.append(mergedata,vals)
        mergelat = np.append(mergelat,lats)
        mergelon = np.append(mergelon,lons)
        mergeview = np.append(mergeview,views)
        mergecloudlat = np.append(mergecloudlat,cloudlats)
        mergecloudlon = np.append(mergecloudlon,cloudlons)
        mergecloud = np.append(mergecloud,clouds)

    if mergelat.any():  
        #get 2-3 bits
        mergecloud=np.array(mergecloud,'uint8')
        mergecloud = np.reshape(mergecloud,[mergecloud.size, 1])
        b = np.unpackbits(mergecloud, axis=1)
        mergecloud = np.sum(b[:,4:6],axis=1)
        mergelat = ma.array(mergelat, mask = (mergedata==65533.))
        mergelon = ma.array(mergelon, mask = (mergedata==65533.))
#        mergedata = ma.array(mergedata, mask = (mergedata==65533.))
        mergecloudlat = ma.array(mergecloudlat, mask = (mergecloud==0.))
        mergecloudlon = ma.array(mergecloudlon, mask = (mergecloud==0.))
#        mergecloud = ma.array(mergecloud, mask = (mergecloud==0.))
        
        projection = '+proj=longlat +ellps=WGS84 +datum=WGS84'
        area_id ='tile'
        proj_id = 'latlon'
        description = 'lat lon grid'
        x_size = 3750
        y_size = 3750
        area_extent = (LLlon,LLlat,URlon,URlat)
        area_def = utils.get_area_def(area_id, description, proj_id, projection,
                                               x_size, y_size, area_extent)

        swath_def = geometry.SwathDefinition(lons=mergecloudlon, lats=mergecloudlat)
#        swath_points_in_grid, cols, rows = ll2cr(swath_def, area_def)
#        num_valid_points, gridded_data = fornav(cols, rows, area_def, mergedata, rows_per_scan=32)
        swath_con = image.ImageContainerNearest(mergecloud, swath_def, radius_of_influence=5000)
        area_con = swath_con.resample(area_def)
        cloud = area_con.image_data
        
        
        swath_def = geometry.SwathDefinition(lons=mergelon, lats=mergelat)
#        swath_points_in_grid, cols, rows = ll2cr(swath_def, area_def)
#        num_valid_points, gridded_data = fornav(cols, rows, area_def, mergedata, rows_per_scan=32)
        swath_con = image.ImageContainerNearest(mergedata, swath_def, radius_of_influence=5000)
        area_con = swath_con.resample(area_def)
        lst = area_con.image_data*0.00351+150.0
#        gridded_data = kd_tree.resample_gauss(swath_def, mergedata,area_def, radius_of_influence=50000, sigmas=25000)
#        lst = gridded_data*0.00351+150.0
        lst[lst==150]=-9999.
        lst[cloud>1]=-9999.
        lst=np.array(lst,'float32')
        out_bt_fn = os.path.join(tile_path,"merged_night_bt_%s_T%03d_%s.dat" % (date,tile,out_time))
        lst.tofile(out_bt_fn)
        convertBin2tif(out_bt_fn,inUL,ALEXIshape,ALEXIres,'float32',gdal.GDT_Float32) 
        
        swath_def = geometry.SwathDefinition(lons=mergelon, lats=mergelat)
#        swath_points_in_grid, cols, rows = ll2cr(swath_def, area_def)
#        num_valid_points, gridded_data = fornav(cols, rows, area_def, mergeview, rows_per_scan=32)
        swath_con = image.ImageContainerNearest(mergeview, swath_def, radius_of_influence=5000)
        area_con = swath_con.resample(area_def)
        view = area_con.image_data
#        gridded_data = kd_tree.resample_gauss(swath_def, mergeview,area_def, radius_of_influence=50000, sigmas=25000)
#        view = gridded_data
        out_view_fn =os.path.join(tile_path,"merged_night_view_%s_T%03d_%s.dat" % (date,tile,out_time))
        view[view==0]=-9999.
        view[cloud>1]=-9999.
        view=np.array(view,'float32')
        view.tofile(out_view_fn)
    else:
        res+=1
        
    return res

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
        parts = filename.split(os.sep)[-1].split('_')
        search_geofile = os.path.join(folder,"*"+"_".join(("GITCO",parts[1],parts[2],parts[3],parts[4])))
#        search_geofile = os.path.join(folder,'GITCO'+filename.split(os.sep)[-1][5:-34])
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
    dd = datetime.datetime(year, 1, 1) + datetime.timedelta(doy - 1)
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
        parts = filename.split(os.sep)[-1].split('_')
        search_geofile = os.path.join(folder,"*"+"_".join(("GMTCO",parts[1],parts[2],parts[3],parts[4])))
        date = datetime.datetime(year,dd.month, dd.day,0,0,0)
        if date > datetime.datetime(2017,3,8,0,0,0):
           search_cloudfile = os.path.join(folder,"*"+"_".join(("VICMO",parts[1],parts[2],parts[3],parts[4]))) 
        else:
           search_cloudfile = os.path.join(folder,"*"+"_".join(("IICMO",parts[1],parts[2],parts[3],parts[4])))
#        search_geofile = os.path.join(folder,'GMTCO'+filename.split(os.sep)[-1][5:-34])
#        search_cloudfile = os.path.join(folder,'IICMO'+filename.split(os.sep)[-1][5:-34])
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

        #grid day I5 data
        if night_flag == "Day":
            start = timer.time()
            trad_sum_fn = os.path.join(grid_I5_path,'bt11_sum1_%03d_%s.dat' % (tile,time))
            trad_count_fn = os.path.join(grid_I5_path,'bt11_count1_%03d_%s.dat' % (tile,time))
            out = subprocess.check_call(["%s" % grid_I5_SDR, "%d" % lat, "%d" %  lon,
                                     "%s" % i5_data, "%s" % latfile,
                                     "%s" % lonfile, "%s" % trad_sum_fn, "%s" % trad_count_fn])
            end = timer.time()
            print(end - start)
            #grid day view data
            start = timer.time()
            view_sum_fn = os.path.join(grid_I5_path,'view_sum1_%03d_%s.dat' % (tile,time))
            view_count_fn = os.path.join(grid_I5_path,'view_count1_%03d_%s.dat' % (tile,time))
            subprocess.check_output(["%s" % grid_I5_SDR, "%d" % lat, "%d" %  lon,
                                     "%s" % viewfile, "%s" % latfile,
                                     "%s" % lonfile, "%s" % view_sum_fn, "%s" % view_count_fn])
            end = timer.time()
            print(end - start)
    
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
    tile_lut_path = os.path.join(static_path,"CFSR","viirs_tile_lookup_tables")
    if not os.path.exists(tile_lut_path):
        os.makedirs(tile_lut_path) 
    icoordpath = os.path.join(tile_lut_path,"CFSR_T%03d_lookup_icoord.dat" % tile)
    jcoordpath = os.path.join(tile_lut_path,"CFSR_T%03d_lookup_jcoord.dat" % tile)
    if not os.path.exists(icoordpath):
        print("generating i and j coords...")
        subprocess.check_output(["%s" % coords, "%d" % lat, "%d" % lon, "%s" % tilestr])
        shutil.move(os.path.join(base,"CFSR_T%03d_lookup_icoord.dat" % tile), icoordpath)
        shutil.move(os.path.join(base,"CFSR_T%03d_lookup_jcoord.dat" % tile), jcoordpath)

def getIJcoordsPython(tile):
    lat,lon = tile2latlon(tile)
#    lat = lat+15.
    tile_lut_path = os.path.join(static_path,"CFSR","viirs_tile_lookup_tables")
    if not os.path.exists(tile_lut_path):
        os.makedirs(tile_lut_path) 
    icoordpath = os.path.join(tile_lut_path,"CFSR_T%03d_lookup_icoord.dat" % tile)
    jcoordpath = os.path.join(tile_lut_path,"CFSR_T%03d_lookup_jcoord.dat" % tile)
    
    istart = (180+lon)*4
#    icor = np.floor((istart+0.252+(0.004*np.array(range(3750))))/0.5)+1
    addArray = np.floor(np.array(range(3750))*0.004/0.25)
    icor = istart+addArray
#    icor = np.floor((istart+(0.004*np.array(range(3750))))/0.5)+1
    icormat = np.repeat(np.reshape(icor,[icor.size,1]),3750,axis=1)
    icormat = icormat.T
    icormat = np.array(icormat,dtype='int32')
#    icormat = np.flipud(icormat)
    icormat.tofile(icoordpath) 
    
#    jstart = 90+lat
    jstart = (89.875+lat)*4
    jcor = jstart+addArray
#    jcor = np.floor((jstart+0.252-(0.004*np.array(range(3750))))/0.5)+1
#    jcor = np.floor((jstart-(0.004*np.array(range(3750))))/0.5)+1
    jcormat = np.repeat(np.reshape(jcor,[jcor.size,1]),3750,axis=1)
#    jcormat = jcormat.T
    jcormat = np.array(jcormat,dtype='int32')
#    jcormat = np.flipud(jcormat)
    jcormat.tofile(jcoordpath)

def get_rise55(year,doy,tile):
    dd=datetime.datetime(year,1,1)+datetime.timedelta(days=doy-1)
    o = ephem.Observer()
    lat,lon = tile2latlon(tile)
    o.lat, o.long = '%3.2f' % (lat+7.5), '%3.2f' % (lon+7.5)
    sun = ephem.Sun()
    sunrise = o.previous_rising(sun, start=dd)
    noon = o.next_transit(sun, start=sunrise)
    hr = noon.datetime().hour
    minute = noon.datetime().minute
    minfraction = minute/60.
    return (hr+minfraction)-1.5

def is_odd(num):
   return num % 2 != 0

def getGrabTime(time):    
    return int(((time/600)+1)*600)


def getGrabTimeInv(grab_time,doy):
    if is_odd(grab_time):
        hr = grab_time-3
        forecastHR = 3
    else:
        hr = grab_time
        forecastHR = 0
    if hr == 24:
        hr = 0
        doy+=1
    return hr, forecastHR,doy 

        
def atmosCorrection(tile,year,doy):
    LLlat,LLlon = tile2latlon(tile)
    URlat = LLlat+15.
    inUL = [LLlon,URlat]
    ALEXIshape = [3750,3750]
    ALEXIres = [0.004,0.004]
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
    overpass_corr_cache = os.path.join(static_path,"OVERPASS_OFFSET_CORRECTION")
#    overpass_corr_path = os.path.join(processing_path,"overpass_corr")
    ztime_fn = os.path.join(overpass_corr_path,"CURRENT_DAY_ZTIME_T%03d.dat" % tile)
    gunzip(os.path.join(overpass_corr_cache,"DAY_ZTIME_T%03d.dat.gz" % tile),
       out_fn=ztime_fn)
    convertBin2tif(ztime_fn,inUL,ALEXIshape,ALEXIres,'float32',gdal.GDT_Float32)
    dtrad_cache = os.path.join(static_path,"dtrad_avg")
    dtrad_fn =os.path.join(overpass_corr_path,"CURRENT_DTRAD_AVG_T%03d.dat" % tile)
    gunzip(os.path.join(dtrad_cache,"DTRAD_T%03d_%d.dat.gz" % (tile,avgddd)),
       out_fn=dtrad_fn)
#    dtrad = np.fromfile(dtrad_fn, dtype=np.float32)
#    dtrad = np.flipud(dtrad.reshape([3750,3750]))
#    dtrad = dtrad.reshape([3750,3750])
#    dtrad.tofile(dtrad_fn)
    convertBin2tif(dtrad_fn,inUL,ALEXIshape,ALEXIres,'float32',gdal.GDT_Float32)
    tile_path = os.path.join(tile_base_path,"T%03d" % tile) 
#    filelist = glob.glob(os.path.join(tile_path,'day_bt_flag_%s*T%03d*.gz' % (date,tile)))
    tile_lut_path = os.path.join(CFSR_path,"viirs_tile_lookup_tables")
    out_bt_fn = glob.glob(os.path.join(tile_path,"merged_day_bt_%s_T%03d*.dat" % (date,tile)))[0]
    out_view_fn1 = glob.glob(os.path.join(tile_path,"merged_day_view_%s_T%03d*.dat" % (date,tile)))[0]

#    time_str = fn.split(os.sep)[-1].split("_")[5].split(".")[0]
    time_str = out_bt_fn.split(os.sep)[-1].split("_")[5].split(".")[0]
#        time=((int(time_str)/300)+1)*300
#        if (time==2400):
#            time=2100

    grab_time = getGrabTime(int(time_str))
    # use forecast hour
    if (grab_time)==2400:
        time = 0000
    else:
        time = grab_time
    hr,forcastHR,cfsr_doy = getGrabTimeInv(grab_time/100,doy)
    cfsr_date = "%d%03d" % (year,cfsr_doy)
    cfsr_tile_path = os.path.join(CFSR_path,"%d" % year,"%03d" % cfsr_doy)

    #======io filenames============================================
    tprof = os.path.join(cfsr_tile_path,"temp_profile_%s_%04d.dat" % (cfsr_date,time))
    qprof = os.path.join(cfsr_tile_path,"spfh_profile_%s_%04d.dat" % (cfsr_date,time))
    tsfcfile = os.path.join(cfsr_tile_path,"sfc_temp_%s_%04d.dat" % (cfsr_date,time))
    presfile = os.path.join(cfsr_tile_path,"sfc_pres_%s_%04d.dat" % (cfsr_date,time))
    qsfcfile = os.path.join(cfsr_tile_path,"sfc_spfh_%s_%04d.dat" % (cfsr_date,time))
    icoordpath = os.path.join(tile_lut_path,"CFSR_T%03d_lookup_icoord.dat" % tile)
    jcoordpath = os.path.join(tile_lut_path,"CFSR_T%03d_lookup_jcoord.dat" % tile)
#    view_fn = os.path.join(tile_path,"view_angle_%s_T%03d_%s.dat.gz" % (date,tile,time_str))
    raw_trad_fn = os.path.join(overpass_corr_path,"RAW_TRAD1_T%03d.dat" % tile)
    trad_fn = os.path.join(overpass_corr_path,"TRAD1_T%03d.dat" % tile)
    out_view_fn = os.path.join(overpass_corr_path,"VIEW_ANGLE_T%03d.dat" % tile)
    #=============================================================
#        out_trad_fn = os.path.join(overpass_corr_path,"TRAD1_T%03d" % tile)
#    gunzip(fn,out_fn=raw_trad_fn)
#    gunzip(view_fn,out_fn= out_view_fn)
    shutil.copyfile(out_bt_fn,raw_trad_fn)
    shutil.copyfile(out_view_fn1,out_view_fn)
    bt = np.fromfile(raw_trad_fn, dtype=np.float32)
    bt= np.flipud(bt.reshape([3750,3750]))
    bt.tofile(raw_trad_fn)
    view = np.fromfile(out_view_fn, dtype=np.float32)
    view= np.flipud(view.reshape([3750,3750]))
    view.tofile(out_view_fn)
    subprocess.check_output(["%s" % offset, "%d" % year, "%03d" %  doy, "%s" % time_str,
                                 "T%03d" % tile, "%s" % ztime_fn, "%s" % raw_trad_fn,
                                 "%s" % dtrad_fn, "%s" % trad_fn])
    convertBin2tif(trad_fn,inUL,ALEXIshape,ALEXIres,'float32',gdal.GDT_Float32)
#    read_data = np.fromfile(trad_fn, dtype=np.float32)
#    tradflipped= np.flipud(read_data.reshape([3750,3750]))
#    tradflipped.tofile(trad_fn)

    outfn = os.path.join(tile_path,"lst_%s_T%03d_%s.dat" % (date,tile,time_str))
    outfn = os.path.join(tile_path,"FINAL_DAY_LST_%s_T%03d.dat" % (date,tile))
    out = subprocess.check_output(["%s" % run_correction,"%s" % tprof, 
                                   "%s" % qprof,"%s" % tsfcfile,
                                   "%s" % presfile, "%s" % qsfcfile,
                                   "%s" % icoordpath, "%s" % jcoordpath,
                                   "%s" % trad_fn,"%s" % out_view_fn, "%s" % outfn])
#    print out
    convertBin2tif(outfn,inUL,ALEXIshape,ALEXIres,'float32',gdal.GDT_Float32)
    
    
    #======Night=============================================================
    out_bt_fn = glob.glob(os.path.join(tile_path,"merged_night_bt_%s_T%03d*.dat" % (date,tile)))[0]
    out_view_fn1 = glob.glob(os.path.join(tile_path,"merged_night_view_%s_T%03d*.dat" % (date,tile)))[0]
    time_str = out_bt_fn.split(os.sep)[-1].split("_")[5].split(".")[0]
    grab_time = getGrabTime(int(time_str))
    # use forecast hour
    if (grab_time)==2400:
        time = 0
    else:
        time = grab_time
    hr,forcastHR,cfsr_doy = getGrabTimeInv(grab_time/100,doy)
    cfsr_date = "%d%03d" % (year,cfsr_doy)
    cfsr_tile_path = os.path.join(CFSR_path,"%d" % year,"%03d" % cfsr_doy)
    

    #======io filenames============================================
    tprof = os.path.join(cfsr_tile_path,"temp_profile_%s_%04d.dat" % (cfsr_date,time))
    qprof = os.path.join(cfsr_tile_path,"spfh_profile_%s_%04d.dat" % (cfsr_date,time))
    tsfcfile = os.path.join(cfsr_tile_path,"sfc_temp_%s_%04d.dat" % (cfsr_date,time))
    presfile = os.path.join(cfsr_tile_path,"sfc_pres_%s_%04d.dat" % (cfsr_date,time))
    qsfcfile = os.path.join(cfsr_tile_path,"sfc_spfh_%s_%04d.dat" % (cfsr_date,time))
    icoordpath = os.path.join(tile_lut_path,"CFSR_T%03d_lookup_icoord.dat" % tile)
    jcoordpath = os.path.join(tile_lut_path,"CFSR_T%03d_lookup_jcoord.dat" % tile)
#    view_fn = os.path.join(tile_path,"view_angle_%s_T%03d_%s.dat.gz" % (date,tile,time_str))
#    raw_trad_fn = os.path.join(overpass_corr_path,"RAW_TRAD1_T%03d.dat" % tile)
    trad_fn = os.path.join(overpass_corr_path,"TRAD1_T%03d.dat" % tile)
    out_view_fn = os.path.join(overpass_corr_path,"VIEW_ANGLE_T%03d.dat" % tile)
    #=============================================================
#        out_trad_fn = os.path.join(overpass_corr_path,"TRAD1_T%03d" % tile)
#    gunzip(fn,out_fn=raw_trad_fn)
#    gunzip(view_fn,out_fn= out_view_fn)
    shutil.copyfile(out_bt_fn,trad_fn)
    shutil.copyfile(out_view_fn1,out_view_fn)
    outfn = os.path.join(tile_path,"lst_%s_T%03d_%s.dat" % (date,tile,time_str))
    outfn = os.path.join(tile_path,"FINAL_NIGHT_LST_%s_T%03d.dat" % (date,tile))
    out = subprocess.check_output(["%s" % run_correction,"%s" % tprof, 
                                   "%s" % qprof,"%s" % tsfcfile,
                                   "%s" % presfile, "%s" % qsfcfile,
                                   "%s" % icoordpath, "%s" % jcoordpath,
                                   "%s" % trad_fn,"%s" % out_view_fn, "%s" % outfn])

    convertBin2tif(outfn,inUL,ALEXIshape,ALEXIres,'float32',gdal.GDT_Float32)


def merge_bt(tile,year,doy):
#    merge = "merge_overpass"
    date = "%d%03d" % (year,doy)
    #=====georeference information=====
    LLlat,LLlon = tile2latlon(tile)
    URlat = LLlat+15.
#    URlon = LLlon+15.
    inUL = [LLlon,URlat]
    ALEXIshape = [3750,3750]
    ALEXIres = [0.004,0.004]
    #=====create times text file=======
    merge_path = os.path.join(processing_path,"MERGE_DAY_NIGHT")
    if not os.path.exists(merge_path):
        os.makedirs(merge_path)
    tile_path = os.path.join(tile_base_path,"T%03d" % tile) 
    if not os.path.exists(tile_path):
        os.makedirs(tile_path)

    #======day=============================
#    lst_list_fn = os.path.join(merge_path,"lst_files_T%03d.txt" % tile)
#    view_list_fn = os.path.join(merge_path,"view_files_T%03d.txt" % tile)
    filelist = glob.glob(os.path.join(agg_I5_path,"day_bt11_%s_%03d*" % (date,tile)))
#    lstfiles = []
#    viewfiles = []
    nfiles = len(filelist)
    viewout = np.empty([3750,3750,nfiles])    
    lstout = np.empty([3750,3750,nfiles]) 
    if nfiles > 0:
        times = []
        for i in range(len(filelist)):
            fn = filelist[i]
            times.append(int(fn.split(os.sep)[-1].split("_")[4].split(".")[0]))
        out_time = "%04d" % np.array(times).mean()
        for i in range(len(filelist)):
            fn = filelist[i]
            time = fn.split(os.sep)[-1].split("_")[4].split(".")[0]
#            times.append(time)
#            view_fn = os.path.join(tile_path,"view_angle_%s_T%03d_%s.dat.gz" % (date,tile,time))
            view_fn = os.path.join(agg_I5_path,"view_%s_%03d_%s.dat" % (date,tile,time))
#            gunzip(view_fn)
            read_data = np.fromfile(view_fn, dtype=np.float32)
            viewout[:,:,i]= np.flipud(read_data.reshape([3750,3750]))
            bt_fn = os.path.join(agg_I5_path,"day_bt11_%s_%03d_%s.dat" % (date,tile,time))
            read_data = np.fromfile(bt_fn, dtype=np.float32)
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
        out_bt_fn = os.path.join(tile_path,"merged_day_bt_%s_T%03d_%s.dat" % (date,tile,out_time))
        out_view_fn = os.path.join(tile_path,"merged_day_view_%s_T%03d_%s.dat" % (date,tile,out_time))
        view[view>60]= -9999.
        lst[view>60] = -9999.
        view = np.flipud(view)
        view.tofile(out_view_fn)
        lst= np.flipud(lst)
        lst.tofile(out_bt_fn)
        convertBin2tif(out_view_fn,inUL,ALEXIshape,ALEXIres,'float32',gdal.GDT_Float32)
        convertBin2tif(out_bt_fn,inUL,ALEXIshape,ALEXIres,'float32',gdal.GDT_Float32)
    
    #===night=======
    filelist = glob.glob(os.path.join(agg_I5_path,"night_bt11_%s_%03d*" % (date,tile)))
    nfiles = len(filelist)
    viewout = np.empty([3750,3750,nfiles])    
    lstout = np.empty([3750,3750,nfiles])
    if nfiles > 0:
        times = []
        for i in range(len(filelist)):
            fn = filelist[i]
            times.append(int(fn.split(os.sep)[-1].split("_")[4].split(".")[0]))
        out_time = "%04d" % np.array(times).mean()
        for i in range(len(filelist)):
            fn = filelist[i]
            time = fn.split(os.sep)[-1].split("_")[4].split(".")[0]
            view_fn = os.path.join(agg_I5_path,"view_%s_%03d_%s.dat" % (date,tile,time))
            read_data = np.fromfile(view_fn, dtype=np.float32)
            viewout[:,:,i]= np.flipud(read_data.reshape([3750,3750]))
            bt_fn = os.path.join(agg_I5_path,"night_bt11_%s_%03d_%s.dat" % (date,tile,time))
            read_data = np.fromfile(bt_fn, dtype=np.float32)
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
        out_bt_fn = os.path.join(tile_path,"merged_night_bt_%s_T%03d_%s.dat" % (date,tile,out_time))
        out_view_fn = os.path.join(tile_path,"merged_night_view_%s_T%03d_%s.dat" % (date,tile,out_time))
        view[view>60]= -9999.
        lst[view>60] = -9999.
        view = np.flipud(view)
        view.tofile(out_view_fn)
        lst= np.flipud(lst)
        lst.tofile(out_bt_fn)
        convertBin2tif(out_view_fn,inUL,ALEXIshape,ALEXIres,'float32',gdal.GDT_Float32)
        convertBin2tif(out_bt_fn,inUL,ALEXIshape,ALEXIres,'float32',gdal.GDT_Float32)       
    
def merge_lst(tile,year,doy):
    date = "%d%03d" % (year,doy)
    #=====georeference information=====
    LLlat,LLlon = tile2latlon(tile)
    URlat = LLlat+15.
    inUL = [LLlon,URlat]
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
            read_data = np.fromfile(view_fn[:-3], dtype=np.float32)
            viewout[:,:,i]= np.flipud(read_data.reshape([3750,3750]))
            lst_fn = os.path.join(tile_path,"lst_%s_T%03d_%s.dat.gz" % (date,tile,time))
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
        view[view>60]= -9999.
        lst[view>60] = -9999.
        view.tofile(out_view_fn)
        lst.tofile(out_lst_fn)
        convertBin2tif(out_view_fn,inUL,ALEXIshape,ALEXIres,'float32',gdal.GDT_Float32)
        convertBin2tif(out_lst_fn,inUL,ALEXIshape,ALEXIres,'float32',gdal.GDT_Float32)
    
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
            read_data = np.fromfile(view_fn[:-3], dtype=np.float32)
            viewout[:,:,i]= np.flipud(read_data.reshape([3750,3750]))
            
            lst_fn = os.path.join(tile_path,"lst_%s_T%03d_%s.dat.gz" % (date,tile,time))
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
        out_lst_fn = os.path.join(tile_path,"FINAL_NIGHT_LST_%s_T%03d.dat" % (date,tile))
        out_view_fn = os.path.join(tile_path,"FINAL_NIGHT_VIEW_%s_T%03d.dat" % (date,tile))
        lst[view>60] = -9999.
        view[view>60]= -9999.
        view.tofile(out_view_fn)
        lst.tofile(out_lst_fn)
        convertBin2tif(out_view_fn,inUL,ALEXIshape,ALEXIres,'float32',gdal.GDT_Float32)
        convertBin2tif(out_lst_fn,inUL,ALEXIshape,ALEXIres,'float32',gdal.GDT_Float32)
    #========clean up======
#    files2remove = glob.glob(os.path.join(tile_path,"*.dat"))
#    for fn in files2remove:
#        os.remove(fn)

def pred_dtrad(tile,year,doy):
    LLlat,LLlon = tile2latlon(tile)
    URlat = LLlat+15.
    inUL = [LLlon,URlat]
    ALEXIshape = [3750,3750]
    ALEXIres = [0.004,0.004]
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
#    files2unzip = glob.glob(os.path.join(tile_path,"*LST_%s*.gz" % date))
#    for fn in files2unzip:
#        gunzip(fn)
            
    dtimedates = np.array(range(1,366,7))
    rday = dtimedates[dtimedates>=doy][0]
    riseddd="%d%03d" %(year,rday)
    risedoy = rday

    laidates = np.array(range(1,366,4))
    rday = laidates[laidates>=doy][0]
    laiddd="%d%03d" %(year,rday)

    precip_fn = os.path.join(base,'STATIC','PRECIP','PRECIP_T%03d.dat' % tile)
    fmax_fn = os.path.join(base,'STATIC','FMAX','FMAX_T%03d.dat' % tile)
    terrain_fn = os.path.join(base,'STATIC','TERRAIN_SD','TERRAIN_T%03d.dat' % tile)
    daylst_fn = os.path.join(base,'TILES','T%03d' % tile,'FINAL_DAY_LST_%s_T%03d.dat' % (date,tile))
    nightlst_fn = os.path.join(base,'TILES','T%03d' % tile,'FINAL_NIGHT_LST_%s_T%03d.dat' % (date,tile))
    lai_fn = os.path.join(base,'STATIC','LAI','MLAI_%s_T%03d.dat' % (laiddd,tile))
    dtime_fn = os.path.join(base,'STATIC','DTIME','DTIME_2014%03d_T%03d.dat' % (risedoy,tile))
    lst_day = np.fromfile(daylst_fn, dtype=np.float32)
    lst_day= np.flipud(lst_day.reshape([3750,3750]))
#    lst_day= lst_day.reshape([3750,3750])
#    plt.imshow(lst_day)
    lst_day.tofile(daylst_fn)
    convertBin2tif(daylst_fn,inUL,ALEXIshape,ALEXIres,'float32',gdal.GDT_Float32)
#    lst_night = np.fromfile(nightlst_fn, dtype=np.float32)
#    lst_night= np.flipud(lst_night.reshape([3750,3750]))
#    lst_night= lst_night.reshape([3750,3750])
#    plt.imshow(lst_night,vmin=280,vmax=310)
#    lst_night.tofile(nightlst_fn)
    convertBin2tif(nightlst_fn,inUL,ALEXIshape,ALEXIres,'float32',gdal.GDT_Float32)
#    precip = np.fromfile(precip_fn, dtype=np.float32)
#    precip= np.flipud(precip.reshape([3750,3750]))
#    plt.imshow(precip)
#    precip.tofile(precip_fn)
    convertBin2tif(precip_fn,inUL,ALEXIshape,ALEXIres,'float32',gdal.GDT_Float32)
#    fmax = np.fromfile(fmax_fn, dtype=np.float32)
#    fmax= np.flipud(fmax.reshape([3750,3750]))
#    plt.imshow(fmax, vmin=0, vmax=0.8)
#    fmax.tofile(fmax_fn)
    convertBin2tif(fmax_fn,inUL,ALEXIshape,ALEXIres,'float32',gdal.GDT_Float32)
#    terrain = np.fromfile(terrain_fn, dtype=np.float32)
#    terrain= np.flipud(terrain.reshape([3750,3750]))
#    plt.imshow(terrain)
#    terrain.tofile(terrain_fn)
    convertBin2tif(terrain_fn,inUL,ALEXIshape,ALEXIres,'float32',gdal.GDT_Float32)
#    lai = np.fromfile(lai_fn, dtype=np.float32)
#    lai= np.flipud(lai.reshape([3750,3750]))
#    plt.imshow(lai,vmin=0,vmax = 3)
#    lai.tofile(lai_fn)
    convertBin2tif(lai_fn,inUL,ALEXIshape,ALEXIres,'float32',gdal.GDT_Float32)
#    dtime = np.fromfile(dtime_fn, dtype=np.float32)
#    dtime= np.flipud(dtime.reshape([3750,3750]))
#    plt.imshow(dtime,vmin=3,vmax=4)
#    dtime.tofile(dtime_fn)
    convertBin2tif(dtime_fn,inUL,ALEXIshape,ALEXIres,'float32',gdal.GDT_Float32)
    fn1 = os.path.join(base,'PROCESSING','DTRAD_PREDICTION','comp1_T%03d.dat' % tile)
    fn2 = os.path.join(base,'PROCESSING','DTRAD_PREDICTION','comp2_T%03d.dat' % tile)
    fn3 = os.path.join(base,'PROCESSING','DTRAD_PREDICTION','comp3_T%03d.dat' % tile)
    fn4 = os.path.join(base,'PROCESSING','DTRAD_PREDICTION','comp4_T%03d.dat' % tile)
    fn5 = os.path.join(base,'PROCESSING','DTRAD_PREDICTION','comp5_T%03d.dat' % tile)
    fn6 = os.path.join(base,'PROCESSING','DTRAD_PREDICTION','comp6_T%03d.dat' % tile)


    subprocess.check_output(["%s" % final_dtrad_p250_fmax0,"%s" % precip_fn, 
                         "%s" % fmax_fn, "%s" % terrain_fn, "%s" % daylst_fn,
                         "%s" % nightlst_fn, "%s" % lai_fn, "%s" % dtime_fn,
                         "%s" % fn1])
    subprocess.check_output(["%s" % final_dtrad_p250_fmax20,"%s" % precip_fn, 
                         "%s" % fmax_fn, "%s" % terrain_fn, "%s" % daylst_fn,
                         "%s" % nightlst_fn, "%s" % lai_fn, "%s" % dtime_fn,
                         "%s" % fn2])
    subprocess.check_output(["%s" % final_dtrad_p500,"%s" % precip_fn, 
                         "%s" % fmax_fn, "%s" % terrain_fn, "%s" % daylst_fn,
                         "%s" % nightlst_fn, "%s" % lai_fn, "%s" % dtime_fn,
                         "%s" % fn3])
    subprocess.check_output(["%s" % final_dtrad_p750,"%s" % precip_fn, 
                         "%s" % fmax_fn, "%s" % terrain_fn, "%s" % daylst_fn,
                         "%s" % nightlst_fn, "%s" % lai_fn, "%s" % dtime_fn,
                         "%s" % fn4])
    subprocess.check_output(["%s" % final_dtrad_p1000,"%s" % precip_fn, 
                         "%s" % fmax_fn, "%s" % terrain_fn, "%s" % daylst_fn,
                         "%s" % nightlst_fn, "%s" % lai_fn, "%s" % dtime_fn,
                         "%s" % fn5])
    subprocess.check_output(["%s" % final_dtrad_p2000,"%s" % precip_fn, 
                         "%s" % fmax_fn, "%s" % terrain_fn, "%s" % daylst_fn,
                         "%s" % nightlst_fn, "%s" % lai_fn, "%s" % dtime_fn,
                         "%s" % fn6])
    dtrad_path = os.path.join(tile_path,
                            "FINAL_DTRAD_%s_T%03d.dat" % ( date, tile))
    subprocess.check_output(["%s" % merge,"%s" % fn1, "%s" % fn2,"%s" % fn3,
                             "%s" % fn4, "%s" % fn5, "%s" % fn6, "%s" % dtrad_path])
    lst_path = os.path.join(tile_path,
                            "FINAL_DAY_LST_TIME2_%s_T%03d.dat" % ( date, tile))
    subprocess.check_output(["%s" % calc_predicted_trad2,"%s" % nightlst_fn, 
                         "%s" % daylst_fn, "%s" % lai_fn, "%s" % lst_path ])
#    read_data = np.fromfile(lst_path, dtype=np.float32)
#    tradflipped= np.flipud(read_data.reshape([3750,3750]))
#    tradflipped.tofile(lst_path)
    convertBin2tif(dtrad_path,inUL,ALEXIshape,ALEXIres,'float32',gdal.GDT_Float32)
    convertBin2tif(lst_path,inUL,ALEXIshape,ALEXIres,'float32',gdal.GDT_Float32)

   
def writeCTL(tile,year,doy):
    LLlat,LLlon = tile2latlon(tile)
    #====create insol.ctl====================================
    date = "%d%03d" % (year,doy)
    date_tile_str = "T%03d_%s" % (tile,date)
    
    srcfn = os.path.join(static_path,'INSOL','deg125','insol55_2011%03d.dat' % doy)
    
    dtimedates = np.array(range(1,366,7))
    rday = dtimedates[dtimedates>=doy][0]
    riseddd="%d%03d" %(year,rday)
    data = './%s_insol.dat' % date_tile_str
    shutil.copyfile(srcfn,data)
    fn = os.path.join('./%s_insol.ctl'% date_tile_str)
    
    file = open(fn, "w")
    file.write("dset ^%s\n" % data)
    file.write("options template\n")
    file.write("title soil moisture\n")
    file.write("undef -9999.\n")
    file.write("xdef 2880 linear -180.0 .125\n")
    file.write("ydef 1200 linear -60.0 .125\n")
    file.write("zdef 1 levels 1 1\n")
    file.write("tdef 365 linear 0z01jan2002 1dy\n")
    file.write("vars 1\n")
    file.write("soil 0 0 soil\n")
    file.write("endvars\n")
    file.close()
    
    #====create rnet.ctl======================================
    srcfn = os.path.join(static_path,'5KM','RNET','RNET%s.dat.gz' % riseddd)
    data = os.path.join('./%s_rnet.dat' % date_tile_str)
    gunzip(srcfn,out_fn=data)
    rnet05 = np.fromfile(data, dtype=np.float32)
    rnet05 = np.flipud(rnet05.reshape([3000,7200]))
    rnet05.tofile(data)
    fn = os.path.join('./%s_rnet.ctl' % date_tile_str)
    
    file = open(fn, "w")
    file.write("dset ^%s\n" % data)
    file.write("options yrev template\n")
    file.write("title soil moisture\n")
    file.write("undef -9999.\n")
    file.write("xdef 7200 linear -180.0 .05\n")
    file.write("ydef 3000 linear -60.0 .05\n")
    file.write("zdef 1 levels 1 1\n")
    file.write("tdef 365 linear 0z01jan2002 1dy\n")
    file.write("vars 1\n")
    file.write("soil 0 0 soil\n")
    file.write("endvars\n")
    file.close()
    
    #====create albedo.ctl======================================
    srcfn = os.path.join(static_path,'ALBEDO','ALBEDO_T%03d.dat' % tile)
    data = os.path.join('./%s_albedo.dat' % date_tile_str)
    shutil.copyfile(srcfn,data)
    fn = os.path.join('./%s_albedo.ctl' % date_tile_str)
    
    file = open(fn, "w")
    file.write("dset ^%s\n" % data)
    file.write("options template\n")
    file.write("title soil moisture\n")
    file.write("undef -9999.\n")
    file.write("xdef 3750 linear %3.2f .004\n" % LLlon)
    file.write("ydef 3750 linear %3.2f .004\n" % LLlat)
    file.write("zdef 1 levels 1 1\n")
    file.write("tdef 365 linear 0z01jan2002 1dy\n")
    file.write("vars 1\n")
    file.write("soil 0 0 soil\n")
    file.write("endvars\n")
    file.close()
    
    #====create lst2.ctl======================================
    srcfn = os.path.join(tile_base_path,'T%03d' % tile,'FINAL_DAY_LST_TIME2_%s_T%03d.dat' % (date,tile))
    data = os.path.join('./%s_lst2.dat' % date_tile_str)
    shutil.copyfile(srcfn,data)
    fn = os.path.join('./%s_lst2.ctl' % date_tile_str)
    
    file = open(fn, "w")
    file.write("dset ^%s\n" % data)
    file.write("options template\n")
    file.write("title soil moisture\n")
    file.write("undef -9999.\n")
    file.write("xdef 3750 linear %3.2f .004\n" % LLlon)
    file.write("ydef 3750 linear %3.2f .004\n" % LLlat)
    file.write("zdef 1 levels 1 1\n")
    file.write("tdef 365 linear 0z01jan2002 1dy\n")
    file.write("vars 1\n")
    file.write("soil 0 0 soil\n")
    file.write("endvars\n")
    file.close()
    
    #====create lwdn.ctl======================================
    time = get_rise55(year,doy,tile)
    grab_time = getGrabTime(int(time)*100)
    hr,forecastHR,cfsr_doy = getGrabTimeInv(grab_time/100,doy)
    cfsr_date = "%d%03d" % (year,cfsr_doy)
    if (grab_time)==2400:
        grab_time = 0000
    srcfn = os.path.join(static_path,'CFSR','%d' % year,'%03d' % cfsr_doy,'sfc_lwdn_%s_%02d00.dat' % (cfsr_date,grab_time/100))
    data = os.path.join('./%s_lwdn.dat' % date_tile_str)
    shutil.copyfile(srcfn,data)
    fn = os.path.join('./%s_lwdn.ctl' % date_tile_str)
    
    file = open(fn, "w")
    file.write("dset ^%s\n" % data)
    file.write("options template\n")
    file.write("title soil moisture\n")
    file.write("undef -9999.\n")
    file.write("xdef 1440 linear -180.0 .25\n")
    file.write("ydef 720 linear -90.0 .25\n")
    file.write("zdef 1 levels 1 1\n")
    file.write("tdef 365 linear 0z01jan2002 1dy\n")
    file.write("vars 1\n")
    file.write("soil 0 0 soil\n")
    file.write("endvars\n")
    file.close()


def write_insol_viirs(outfn,date_tile_str):
    fn = os.path.join('./%s_agg_insol.gs' % date_tile_str)
    file = open(fn, "w")
    file.write("function main(args)\n")
    file.write("lat1=subwrd(args,1);if(lat1='');lat1"";endif\n")
    file.write("lat2=subwrd(args,2);if(lat2='');lat2"";endif\n")
    file.write("lon1=subwrd(args,3);if(lon1='');lon1"";endif\n")
    file.write("lon2=subwrd(args,4);if(lon2='');lon2"";endif\n")
    file.write("\n")
    file.write("say lat1\n")
    file.write("say lat2\n")
    file.write("say lon1\n")
    file.write("say lon2\n")
    file.write("\n")
    file.write("'reinit'\n")
    file.write("'open ./%s_insol.ctl'\n" % date_tile_str)
    file.write("'set lat ' lat1+0.025 ' ' lat2-0.025\n")
    file.write("'set lon ' lon1+0.025 ' ' lon2-0.025\n")
    file.write("'set undef -9999.'\n")
    file.write("'define test=re(soil.1,0.05,0.05)'\n")
    file.write("'set gxout fwrite'\n")
    file.write("'set fwrite %s'\n" % outfn)
    file.write("'d test'\n")
    file.write("'disable fwrite'\n")
    file.close()

def write_agg_insol_viirs(outfn,date_tile_str):
    fn = os.path.join('./%s_agg_insol_viirs.gs' % date_tile_str)
    file = open(fn, "w")
    file.write("function main(args)\n")
    file.write("lat1=subwrd(args,1);if(lat1='');lat1"";endif\n")
    file.write("lat2=subwrd(args,2);if(lat2='');lat2"";endif\n")
    file.write("lon1=subwrd(args,3);if(lon1='');lon1"";endif\n")
    file.write("lon2=subwrd(args,4);if(lon2='');lon2"";endif\n")
    file.write("\n")
    file.write("say lat1\n")
    file.write("say lat2\n")
    file.write("say lon1\n")
    file.write("say lon2\n")
    file.write("\n")
    file.write("'reinit'\n")
    file.write("'open ./%s_insol.ctl'\n" % date_tile_str)
    file.write("'set lat ' lat1+0.002 ' ' lat2-0.002\n")
    file.write("'set lon ' lon1+0.002 ' ' lon2-0.002\n")
    file.write("'set undef -9999.'\n")
    file.write("'define test=re(soil.1,0.004,0.004)'\n")
    file.write("'set gxout fwrite'\n")
    file.write("'set fwrite %s'\n" % outfn)
    file.write("'d test'\n")
    file.write("'disable fwrite'\n")
    file.close()

def write_agg_rnet(outfn,date_tile_str):
    fn = os.path.join('./%s_agg_rnet.gs' % date_tile_str)
    file = open(fn, "w")
    file.write("function main(args)\n")
    file.write("lat1=subwrd(args,1);if(lat1='');lat1"";endif\n")
    file.write("lat2=subwrd(args,2);if(lat2='');lat2"";endif\n")
    file.write("lon1=subwrd(args,3);if(lon1='');lon1"";endif\n")
    file.write("lon2=subwrd(args,4);if(lon2='');lon2"";endif\n")
    file.write("\n")
    file.write("say lat1\n")
    file.write("say lat2\n")
    file.write("say lon1\n")
    file.write("say lon2\n")
    file.write("\n")
    file.write("'reinit'\n")
    file.write("'open ./%s_rnet.ctl'\n" % date_tile_str)
    file.write("'set lat ' lat1+0.025 ' ' lat2-0.025\n")
    file.write("'set lon ' lon1+0.025 ' ' lon2-0.025\n")
    file.write("'set undef -9999.'\n")
    file.write("'define test=re(soil.1,0.05,0.05)'\n")
    file.write("'set gxout fwrite'\n")
    file.write("'set fwrite %s'\n" % outfn)
    file.write("'d test'\n")
    file.write("'disable fwrite'\n")
    file.close()

def write_agg_albedo(outfn,date_tile_str):
    fn = os.path.join('./%s_agg_albedo.gs' % date_tile_str)
    file = open(fn, "w")
    file.write("function main(args)\n")
    file.write("lat1=subwrd(args,1);if(lat1='');lat1"";endif\n")
    file.write("lat2=subwrd(args,2);if(lat2='');lat2"";endif\n")
    file.write("lon1=subwrd(args,3);if(lon1='');lon1"";endif\n")
    file.write("lon2=subwrd(args,4);if(lon2='');lon2"";endif\n")
    file.write("\n")
    file.write("say lat1\n")
    file.write("say lat2\n")
    file.write("say lon1\n")
    file.write("say lon2\n")
    file.write("\n")
    file.write("'reinit'\n")
    file.write("'open ./%s_albedo.ctl'\n" % date_tile_str)
    file.write("'set lat ' lat1+0.025 ' ' lat2-0.025\n")
    file.write("'set lon ' lon1+0.025 ' ' lon2-0.025\n")
    file.write("'set undef -9999.'\n")
    file.write("'define test=re(soil.1,0.05,0.05)'\n")
    file.write("'set gxout fwrite'\n")
    file.write("'set fwrite %s'\n" % outfn)
    file.write("'d test'\n")
    file.write("'disable fwrite'\n")
    file.close()
    
    
def write_agg_lst2(outfn,date_tile_str):
    fn = os.path.join('./%s_agg_lst2.gs' % date_tile_str)
    file = open(fn, "w")
    file.write("function main(args)\n")
    file.write("lat1=subwrd(args,1);if(lat1='');lat1"";endif\n")
    file.write("lat2=subwrd(args,2);if(lat2='');lat2"";endif\n")
    file.write("lon1=subwrd(args,3);if(lon1='');lon1"";endif\n")
    file.write("lon2=subwrd(args,4);if(lon2='');lon2"";endif\n")
    file.write("\n")
    file.write("say lat1\n")
    file.write("say lat2\n")
    file.write("say lon1\n")
    file.write("say lon2\n")
    file.write("\n")
    file.write("'reinit'\n")
    file.write("'open ./%s_lst2.ctl'\n" % date_tile_str)
    file.write("'set lat ' lat1+0.025 ' ' lat2-0.025\n")
    file.write("'set lon ' lon1+0.025 ' ' lon2-0.025\n")
    file.write("'set undef -9999.'\n")
    file.write("'define test=re(soil.1,0.05,0.05)'\n")
    file.write("'set gxout fwrite'\n")
    file.write("'set fwrite %s'\n" % outfn)
    file.write("'d test'\n")
    file.write("'disable fwrite'\n")
    file.close()
    
def write_agg_lwdn(outfn,date_tile_str):
    fn = os.path.join('./%s_agg_lwdn.gs' % date_tile_str)
    file = open(fn, "w")
    file.write("function main(args)\n")
    file.write("lat1=subwrd(args,1);if(lat1='');lat1"";endif\n")
    file.write("lat2=subwrd(args,2);if(lat2='');lat2"";endif\n")
    file.write("lon1=subwrd(args,3);if(lon1='');lon1"";endif\n")
    file.write("lon2=subwrd(args,4);if(lon2='');lon2"";endif\n")
    file.write("\n")
    file.write("say lat1\n")
    file.write("say lat2\n")
    file.write("say lon1\n")
    file.write("say lon2\n")
    file.write("\n")
    file.write("'reinit'\n")
    file.write("'open ./%s_lwdn.ctl'\n" % date_tile_str)
    file.write("'set lat ' lat1+0.025 ' ' lat2-0.025\n")
    file.write("'set lon ' lon1+0.025 ' ' lon2-0.025\n")
    file.write("'set undef -9999.'\n")
    file.write("'define test=re(soil.1,0.05,0.05)'\n")
    file.write("'set gxout fwrite'\n")
    file.write("'set fwrite %s'\n" % outfn)
    file.write("'d test'\n")
    file.write("'disable fwrite'\n")
    file.close()

def write_agg_lwdn_viirs(outfn,date_tile_str):
    fn = os.path.join('./%s_agg_lwdn_viirs.gs'% date_tile_str)
    file = open(fn, "w")
    file.write("function main(args)\n")
    file.write("lat1=subwrd(args,1);if(lat1='');lat1"";endif\n")
    file.write("lat2=subwrd(args,2);if(lat2='');lat2"";endif\n")
    file.write("lon1=subwrd(args,3);if(lon1='');lon1"";endif\n")
    file.write("lon2=subwrd(args,4);if(lon2='');lon2"";endif\n")
    file.write("\n")
    file.write("say lat1\n")
    file.write("say lat2\n")
    file.write("say lon1\n")
    file.write("say lon2\n")
    file.write("\n")
    file.write("'reinit'\n")
    file.write("'open ./%s_lwdn.ctl'\n" % date_tile_str)
    file.write("'set lat ' lat1+0.002 ' ' lat2-0.002\n")
    file.write("'set lon ' lon1+0.002 ' ' lon2-0.002\n")
    file.write("'set undef -9999.'\n")
    file.write("'define test=re(soil.1,0.004,0.004)'\n")
    file.write("'set gxout fwrite'\n")
    file.write("'set fwrite %s'\n" % outfn)
    file.write("'d test'\n")
    file.write("'disable fwrite'\n")
    file.close()    
    
#    outfn = os.path.join(base,'CALC_RNET','insol_T%03d.dat' % tile)
#    outfn = os.path.join(base,'insol_T%03d.dat' % tile)



def get_tiles_fstem_names(namefn):
    of = open(namefn,'w')
    of.write("rnet.\n")
    of.write("\n")
    of.write("rnet:		continuous.\n")
    of.write("albedo:         continuous.\n")
    of.write("insol:          continuous.\n")
    of.write("lwdn:           continuous.\n")
    of.write("lst2:	  	continuous.\n") 
    of.close()

def get_trees_fstem_names(namefn):
    of = open(namefn,'w')
    of.write("fsun.\n")
    of.write("\n")
    of.write("fsun:		continuous.\n")
    of.write("dthr:           continuous.\n")
    of.write("rnet_dthr:      continuous.\n")
    of.write("rnet:           continuous.\n")
    of.write("trad2:          continuous.\n")
    of.close()
    
def readCubistOut(input,outDF):
    outAltSplit = input.split('\n')
    b = np.char.strip(outAltSplit)
    ifIndex = np.argwhere(np.array(b) == "if")
    thenIndex = np.argwhere(np.array(b) == "then")
    lstOut = np.zeros([outDF.shape[0]])
    count = 0
    mask_formula = ''
    for name in list(outDF):
        count+=1
        if count < len(list(outDF)):
            mask_formula = mask_formula + '(outDF["%s"] < 0.0) | ' % name
        else:
            mask_formula = mask_formula + '(outDF["%s"] < 0.0) ' % name
    mask = eval('(%s)' % mask_formula)
    if len(ifIndex)<1: # Theres only 1 rule
        modelIndex = np.argwhere(np.array(b) == "Model:")
        formulaSplit = b[modelIndex+2][0].split(' ')
        for k in xrange(len(formulaSplit)/3):
            if k == 0:
                formula = '%s' % formulaSplit[2]
            else:
                formSub = '%s %s*outDF.%s' % (formulaSplit[(k*3)],formulaSplit[(k*3)+1],formulaSplit[(k*3)+2])
                formula = '%s%s' % (formula,formSub)
        #===Check for another line of equation
        if (modelIndex+3 < len(b)):
            formulaSplit = b[modelIndex+3][0].split(' ')
            k = 0
            if len(formulaSplit) > 1:
                formSub = '%s %s*outDF[rule2use].%s' % (formulaSplit[(k*3)],formulaSplit[(k*3)+1],formulaSplit[(k*3)+2])
                formula = '%s%s' % (formula,formSub)
#        print 'formula:lai = %s' % formula
        lstOut = eval('(%s)' % formula)
        lstOut[np.where(mask)] = -9999.
    else:
        for i in xrange(len(ifIndex)): # rules
            for j in xrange((thenIndex[i][0]-ifIndex[i][0])-1): #rule branches (i.e. if x>y)
                treeRule = b[ifIndex[i][0]+1:thenIndex[i][0]]
                treeRuleSplit = treeRule[j].split(' ')
                treeRuleSplit[0] = 'outDF.%s' % treeRule[j].split(' ')[0]
                if j < 1:
                    rule = '(%s)' % ' '.join(treeRuleSplit)
                else:
                    rule = '%s & (%s)' % (rule, ' '.join(treeRuleSplit))
#            print 'rule2use:%s' % rule
            # run the rule on the dataset
            rule2use=eval('(%s)'% rule)
        
            formulaSplit = b[thenIndex[i]+1][0].split(' ')
            for k in xrange(len(formulaSplit)/3):
                if k == 0:
                    formula = '%s' % formulaSplit[2]
                else:
                    formSub = '%s %s*outDF[rule2use].%s' % (formulaSplit[(k*3)],formulaSplit[(k*3)+1],formulaSplit[(k*3)+2])
                    formula = '%s%s' % (formula,formSub)
            #===Check for another line of equation
            if (thenIndex[i]+2 < len(b)):
                formulaSplit = b[thenIndex[i]+2][0].split(' ')
                k = 0
                if len(formulaSplit) > 1:
                    formSub = '%s %s*outDF[rule2use].%s' % (formulaSplit[(k*3)],formulaSplit[(k*3)+1],formulaSplit[(k*3)+2])
                    formula = '%s%s' % (formula,formSub)
                
#            print 'formula:rnet = %s' % formula
            lstOut[np.where(rule2use)] = eval('(%s)' % formula)
            lstOut[np.where(mask)] = -9999.
    return lstOut    
    
def processTiles(year,doy,tile):
    LLlat,LLlon = tile2latlon(tile)
    URlat = LLlat+15.
    URlon = LLlon+15.
    inUL = [LLlon,URlat]
    halfdeg_shape = [300,300]
    halfdeg_sizeArr = 300*300
    ALEXI_shape = [3750,3750]
    ALEXI_res = [0.004,0.004]
#    ALEXI_sizeArr = 3750*3750
    date = '%d%03d' % (year,doy)
    date_tile_str = "T%03d_%s" % (tile,date)
    writeCTL(tile,year,doy)
    #========process insol==================================================
    tiles_path = os.path.join(calc_rnet_path,'tiles')
    if not os.path.exists(tiles_path):
        os.makedirs(tiles_path) 
        
    insol_fn = os.path.join(tiles_path,'INSOL_%03d_%s.dat' % (tile,date))
    write_insol_viirs(insol_fn,date_tile_str)
    out = subprocess.check_output("opengrads -blxc 'run ./%s_agg_insol.gs %3.2f %3.2f  %3.2f  %3.2f'" 
                                  % (date_tile_str,LLlat, URlat, LLlon, URlon), shell=True)

    read_data = np.fromfile(insol_fn, dtype=np.float32)
    insol = np.flipud(read_data.reshape([halfdeg_shape[0],halfdeg_shape[1]]))
    insol = np.reshape(insol,[halfdeg_sizeArr])
    
    viirs_tile_path = os.path.join(calc_rnet_path,'viirs','T%03d' % tile)
    if not os.path.exists(viirs_tile_path):
        os.makedirs(viirs_tile_path) 
        
    insol_viirs_fn = os.path.join(viirs_tile_path,'INSOL_%03d_%s.dat' % (tile,date))
    write_agg_insol_viirs(insol_viirs_fn,date_tile_str)
#    print("opengrads -blxc 'run ./%s_agg_insol_viirs.gs %3.2f %3.2f  %3.2f  %3.2f'" 
#                                  % (date_tile_str,LLlat, URlat, LLlon, URlon))
    out = subprocess.check_output("opengrads -blxc 'run ./%s_agg_insol_viirs.gs %3.2f %3.2f  %3.2f  %3.2f'" 
                                  % (date_tile_str,LLlat, URlat, LLlon, URlon), shell=True)
    insol_viirs = np.fromfile(insol_viirs_fn, dtype=np.float32)
    insol_viirs = np.flipud(insol_viirs.reshape([ALEXI_shape[0],ALEXI_shape[1]]))
    insol_viirs = np.reshape(insol_viirs,[3750*3750])
    convertBin2tif(insol_viirs_fn,inUL,ALEXI_shape,ALEXI_res,'float32',gdal.GDT_Float32)
    #======process RNET======================================================
    rnet_fn = os.path.join(tiles_path,'RNET_%03d_%s.dat' % (tile,date))
    write_agg_rnet(rnet_fn,date_tile_str)
    
    out = subprocess.check_output("opengrads -blxc 'run ./%s_agg_rnet.gs %3.2f %3.2f  %3.2f  %3.2f'" 
                                  % (date_tile_str,LLlat, URlat, LLlon, URlon), shell=True)
    read_data = np.fromfile(rnet_fn, dtype=np.float32)
    rnet = np.flipud(read_data.reshape([halfdeg_shape[0],halfdeg_shape[1]]))
    rnet = np.reshape(rnet,[halfdeg_sizeArr])    
    #======process albedo====================================================
    albedo_fn = os.path.join(tiles_path,'ALBEDO_%03d_%s.dat' % (tile,date))
    write_agg_albedo(albedo_fn,date_tile_str)
    
    out = subprocess.check_output("opengrads -blxc 'run ./%s_agg_albedo.gs %3.2f %3.2f  %3.2f  %3.2f'" 
                                  % (date_tile_str,LLlat, URlat, LLlon, URlon), shell=True)
    albedo = np.fromfile(albedo_fn, dtype=np.float32)
#    albedo = np.flipud(read_data.reshape([halfdeg_shape[0],halfdeg_shape[1]]))
    albedo = albedo.reshape([halfdeg_shape[0],halfdeg_shape[1]])
    albedo = np.reshape(albedo,[halfdeg_sizeArr])
    #=====process LST2=======================================================
    lst_fn = os.path.join(tiles_path,'LST2_%03d_%s.dat' % (tile,date))
    write_agg_lst2(lst_fn,date_tile_str)
    
    out = subprocess.check_output("opengrads -blxc 'run ./%s_agg_lst2.gs %3.2f %3.2f  %3.2f  %3.2f'" 
                                  % (date_tile_str,LLlat, URlat, LLlon, URlon), shell=True)
    lst = np.fromfile(lst_fn, dtype=np.float32)
#    lst = np.flipud(read_data.reshape([halfdeg_shape[0],halfdeg_shape[1]]))
#    lst = np.reshape(lst,[halfdeg_sizeArr])
    #====process LWDN========================================================
    lwdn_fn = os.path.join(tiles_path,'LWDN_%03d_%s.dat' % (tile,date))
    write_agg_lwdn(lwdn_fn,date_tile_str)
    out = subprocess.check_output("opengrads -blxc 'run ./%s_agg_lwdn.gs %3.2f %3.2f  %3.2f  %3.2f'" 
                                  % (date_tile_str,LLlat, URlat, LLlon, URlon), shell=True)
    
    read_data = np.fromfile(lwdn_fn, dtype=np.float32)
    lwdn = np.flipud(read_data.reshape([halfdeg_shape[0],halfdeg_shape[1]]))
    lwdn = np.reshape(lwdn,[halfdeg_sizeArr])

    lwdn_viirs_fn = os.path.join(viirs_tile_path,'LWDN_%03d_%s.dat' % (tile,date))
    write_agg_lwdn_viirs(lwdn_viirs_fn,date_tile_str)
    out = subprocess.check_output("opengrads -blxc 'run ./%s_agg_lwdn_viirs.gs %3.2f %3.2f  %3.2f  %3.2f'" 
                                  % (date_tile_str,LLlat, URlat, LLlon, URlon), shell=True)
    
    lwdn_viirs = np.fromfile(lwdn_viirs_fn, dtype=np.float32)
    lwdn_viirs = np.flipud(lwdn_viirs.reshape([ALEXI_shape[0],ALEXI_shape[1]]))
    lwdn_viirs = np.reshape(lwdn_viirs,[3750*3750])
    convertBin2tif(lwdn_viirs_fn,inUL,ALEXI_shape,ALEXI_res,'float32',gdal.GDT_Float32)
    outDict = {'rnet': rnet, 'albedo':albedo, 'insol':insol, 'lwdn': lwdn, 'lst2':lst}
    inDF = pd.DataFrame.from_dict(outDict)
    outDF = inDF.loc[(inDF["rnet"] > 0.0) & (inDF["albedo"] > 0.0) & 
                (inDF["insol"] > 0.0) & (inDF["lwdn"] > 0.0) &
                (inDF["lst2"] > 0.0), ["rnet","albedo","insol","lwdn","lst2"]]

    #
    ####=========create fstem.data for cubist====================================

    ##===========create dictionary and convert to csv=======
    #
    calc_rnet_tile_ctl = os.path.join(calc_rnet_path,'tiles_ctl','T%03d' % tile )
    if not os.path.exists(calc_rnet_tile_ctl):
        os.makedirs(calc_rnet_tile_ctl) 
    file_data = os.path.join(calc_rnet_tile_ctl,'rnet.data')
    outDF.to_csv(file_data , header=True, index=False,columns=["rnet",
                                        "albedo","insol","lwdn","lst2"])
    file_names = os.path.join(calc_rnet_tile_ctl,'rnet.names')
    get_tiles_fstem_names(file_names)
    
    #====run cubist======================================
#    print("running cubist...")
    cubist_name = os.path.join(calc_rnet_tile_ctl,'rnet')
    out = subprocess.check_output("cubist -f %s -u -a -r 20" % cubist_name, shell=True)
    lst2 = np.fromfile('./%s_lst2.dat' % date_tile_str, dtype=np.float32)
#    lst2 = np.flipud(lst2.reshape([3750,3750]))
#    lst2 = np.reshape(lst2,[3750*3750])
    albedo = np.fromfile('./%s_albedo.dat' % date_tile_str, dtype=np.float32)
#    albedo = np.flipud(albedo.reshape([3750,3750]))
#    albedo = np.reshape(albedo,[3750*3750])
    convertBin2tif('./%s_albedo.dat' % date_tile_str,inUL,ALEXI_shape,ALEXI_res,'float32',gdal.GDT_Float32)
    cubDict = {'albedo':albedo, 'insol':insol_viirs, 'lwdn': lwdn_viirs, 'lst2':lst2}
    cubDF = pd.DataFrame.from_dict(cubDict)
    rnet_out = readCubistOut(out,cubDF)
    rnet_out = np.reshape(rnet_out, [3750,3750])

    #=======run final_rnet===================
    rnet_tile = os.path.join(tile_base_path,'T%03d' % tile)
    if not os.path.exists(rnet_tile):
        os.makedirs(rnet_tile)
    finalrnet_fn = os.path.join(rnet_tile,'FINAL_RNET_%s_T%03d.dat' % (date,tile))

    rnet_out = np.array(rnet_out,dtype='Float32')
    rnet_out.tofile(finalrnet_fn)
    convertBin2tif(finalrnet_fn,inUL,ALEXI_shape,ALEXI_res,'float32',gdal.GDT_Float32)

#def processTrees(year,doy,tile):
#    LLlat,LLlon = tile2latlon(tile)
#    URlat = LLlat+15.
#    inUL = [LLlon,URlat]
#    ALEXI_shape = [3750,3750]
#    ALEXI_res = [0.004,0.004]
#    date = '%d%03d' % (year,doy)
#    dtimedates = np.array(range(1,366,7))
#    r7day = dtimedates[dtimedates>=doy][0]
#    riseddd="%d%03d" %(year,r7day)
#    fsun_trees_tile_ctl = os.path.join(fsun_trees_path,'tiles_ctl','T%03d' % tile )
#    if not os.path.exists(fsun_trees_tile_ctl):
#        os.makedirs(fsun_trees_tile_ctl) 
#    ##===========create dictionary and convert to csv=======
#    #======load 5 km data and subset it========================================  
#    dthr_zip_fn = os.path.join(static_path,"5KM","DTHR","DTHR%s.dat.gz" % riseddd)  
#    dthr_fn = os.path.join("./DTHR%s.dat" % date)  
#    gunzip(dthr_zip_fn,out_fn=dthr_fn)
#    dthr = np.fromfile(dthr_fn, dtype=np.float32)
#    dthr = np.flipud(dthr.reshape([3000,7200]))
#    dthr_sub = dthr[901:1801,3201:4801]
##    plt.imshow(dthr_sub)
#    dthr = np.reshape(dthr_sub,[dthr_sub.size])
#    
#    rnet_zip_fn = os.path.join(static_path,"5KM","RNET","RNET%s.dat.gz" % riseddd)  
#    rnet_fn = os.path.join("./RNET%s.dat" % date)  
#    gunzip(rnet_zip_fn,out_fn=rnet_fn)
#    rnet = np.fromfile(rnet_fn, dtype=np.float32)
#    rnet = np.flipud(rnet.reshape([3000,7200]))
#    rnet_sub = rnet[901:1801,3201:4801]
##    plt.imshow(rnet_sub)
#    rnet = np.reshape(rnet_sub,[rnet_sub.size])
#    
#    fsun_src_fn = os.path.join(static_path,"5KM","FSUN","FSUN%s.dat" % riseddd)  
#    fsun_fn = os.path.join("./FSUN%s.dat" % date)  
#    shutil.copyfile(fsun_src_fn,fsun_fn)
#    fsun = np.fromfile(fsun_fn, dtype=np.float32)
#    fsun = np.flipud(fsun.reshape([3000,7200]))
#    fsun_sub = fsun[901:1801,3201:4801]
##    plt.imshow(fsun_sub[100:400,1000:1300],vmin=0, vmax=0.5)
#    fsun  = np.reshape(fsun_sub,[fsun_sub.size])
#    
#    rnet_dthr = rnet/dthr    
#    # note* FMAX is actually max LAI here
#    fmax_src_fn = os.path.join(static_path,"5KM","FMAX","FMAX.dat")  
#    fmax_fn = os.path.join("./FMAX.dat")  
#    shutil.copyfile(fmax_src_fn,fmax_fn)
#    fmax = np.fromfile(fmax_fn, dtype=np.float32)
#    fmax = 1-np.exp(-0.5*fmax)
#    fmax_sub = np.flipud(fmax.reshape([900,1600]))
##    plt.imshow(fmax_sub, vmin=0, vmax=0.8)
#    fmax  = np.reshape(fmax_sub,[fmax_sub.size])
#    
#    precip_src_fn = os.path.join(static_path,"5KM","PRECIP","PRECIP.dat")  
#    precip_fn = os.path.join("./PRECIP.dat")  
#    shutil.copyfile(precip_src_fn,precip_fn)
#    precip = np.fromfile(precip_fn, dtype=np.float32)
#    precip_sub = np.flipud(precip.reshape([900,1600]))
##    plt.imshow(precip_sub)
#    precip  = np.reshape(precip_sub,[precip_sub.size])
#    
#    trad2_src_fn = os.path.join(static_path,"5KM","TRAD2","TRD2%s.dat.gz" % riseddd)  
#    trad2_fn = os.path.join("./TRD2%s.dat" % date)  
#    gunzip(trad2_src_fn,out_fn=trad2_fn)
#    trad2 = np.fromfile(trad2_fn, dtype=np.float32)
#    trad2 = np.flipud(trad2.reshape([3000,7200]))
#    trad2_sub = trad2[901:1801,3201:4801]
##    plt.imshow(trad2_sub,vmin=280, vmax=320)
#    trad2 = np.reshape(trad2_sub,[trad2_sub.size])
#    
#    lai_src_fn = os.path.join(static_path,"5KM","LAI","MLAI2014%03d.dat" % r7day)  
#    lai_fn = os.path.join("./LAI2014%03d.dat" % doy)  
#    shutil.copyfile(lai_src_fn,lai_fn)
#    lai = np.fromfile(lai_fn, dtype=np.float32)
#    lai = lai.reshape([3000,7200])
#    lai_sub = lai[901:1801,3201:4801]
##    plt.imshow(lai_sub,vmin=0, vmax=2)
#    lai = np.reshape(lai_sub,[lai_sub.size])
#    
#    outDict = {'fsun':fsun, 'dthr':dthr, 'rnet_dthr':rnet_dthr, 
#               'rnet': rnet, 'fmax':fmax, 'precip':precip,
#               'lai':lai,'trad2':trad2}
#    outDF = pd.DataFrame.from_dict(outDict)
#    #=======ALEXI resolution inputs===============================================
#    laidates = np.array(range(1,366,4))
#    r4day = laidates[laidates>=doy][0]
#    laiddd="%d%03d" %(year,r4day)
#    dthr_fn = os.path.join(tile_base_path,'T%03d' % tile, 'FINAL_DTRAD_%s_T%03d.dat' % (date,tile))
#    trad2_fn = os.path.join(tile_base_path,'T%03d' % tile, 'FINAL_DAY_LST_TIME2_%s_T%03d.dat' % (date,tile))
#    rnet_fn = os.path.join(tile_base_path,'T%03d' % tile, 'FINAL_RNET_%s_T%03d.dat' % (date,tile))
#    lai_fn = os.path.join(static_path,'LAI','MLAI_%s_T%03d.dat' % (laiddd,tile)) # only have 2015 so far
#    dthr_corr_fn = os.path.join(static_path,'DTHR_CORR','DTHR_CORR_2010%03d_T%03d.dat' % (r7day,tile))
#    dtime_fn = os.path.join(static_path,'DTIME','DTIME_2014%03d_T%03d.dat' % (r7day,tile))
#    fmax_fn = os.path.join(static_path,'FMAX','FMAX_T%03d.dat' % (tile))
#    precip_fn = os.path.join(static_path,'PRECIP','PRECIP_T%03d.dat' % (tile))
#    
#    dthr = np.fromfile(dthr_fn, dtype=np.float32)
#    dthr = np.flipud(dthr.reshape([3750,3750]))
##    plt.imshow(dthr)
#    dthr = np.reshape(dthr,[3750*3750])
#    trad2 = np.fromfile(trad2_fn, dtype=np.float32)
#    trad2 = np.flipud(trad2.reshape([3750,3750]))
##    plt.imshow(trad2)
#    trad2 = np.reshape(trad2,[3750*3750])
#    rnet = np.fromfile(rnet_fn, dtype=np.float32)
#    rnet = np.flipud(rnet.reshape([3750,3750]))
##    plt.imshow(rnet)
#    rnet = np.reshape(rnet,[3750*3750])    
#    lai = np.fromfile(lai_fn, dtype=np.float32)
#    lai = np.flipud(lai.reshape([3750,3750]))
##    plt.imshow(lai, vmin=0, vmax=2)
#    lai = np.reshape(lai,[3750*3750])
#    dthr_corr = np.fromfile(dthr_corr_fn, dtype=np.float32)
#    dthr_corr = np.flipud(dthr_corr.reshape([3750,3750]))
##    plt.imshow(dthr_corr, vmin=0,vmax=3)
#    dthr_corr = np.reshape(dthr_corr,[3750*3750])    
#    dtime = np.fromfile(dtime_fn, dtype=np.float32)
#    dtime = np.flipud(dtime.reshape([3750,3750]))
##    plt.imshow(dtime)
#    dtime = np.reshape(dtime,[3750*3750])
#    fmax = np.fromfile(fmax_fn, dtype=np.float32)
#    fmax = np.flipud(fmax.reshape([3750,3750]))
##    plt.imshow(fmax, vmin=0, vmax=0.3)
#    fmax = np.reshape(fmax,[3750*3750])    
#    precip = np.fromfile(precip_fn, dtype=np.float32)
#    precip = np.flipud(precip.reshape([3750,3750]))
##    plt.imshow(precip)
#    precip = np.reshape(precip,[3750*3750])    
#    dthr = (dthr/dtime)*dthr_corr
#    
#    rnet_dthr = rnet/dthr
#
#    
#    predDict = {'dthr':dthr,'rnet_dthr':rnet_dthr,'rnet': rnet,'trad2':trad2,
#                'fmax':fmax, 'precip':precip, 'lai':lai}
#    predDF = pd.DataFrame.from_dict(predDict)
#
#    
#    #============create final_p250_fmax0.f90======================================
#    #========create fsun.data=====================================================
#    p1 = 0
#    p2 = 250
#    f1 = 0
#    f2 = 0.2
#    out = outDF.loc[(outDF["fsun"] > 0.0) & (outDF["rnet"] > 0.0) & 
#                    (outDF["lai"] > 0.0) & (outDF["trad2"] > 0.0) &
#                    (outDF["dthr"] > 0.0) & (outDF["precip"] >= p1) &
#                    (outDF["precip"] < p2) & (outDF["fmax"] >= f1) &
#                    (outDF["fmax"] < f2), ["fsun","dthr","rnet_dthr","rnet","trad2"]]
#
#    file_data = os.path.join(fsun_trees_tile_ctl,'fsun.data')
#    out.to_csv(file_data , header=True, index=False,columns=["fsun",
#                                    "dthr","rnet_dthr","rnet","trad2"])
#        
#    file_names = os.path.join(fsun_trees_tile_ctl,'fsun.names')
#    get_trees_fstem_names(file_names)
#    
#    #====run cubist======================================
#    cubist_name = os.path.join(fsun_trees_tile_ctl,'fsun')
#    out = subprocess.check_output("cubist -f %s -u -a -i -r 8 -S 75" % cubist_name, shell=True)
##    print out
#    pred1 = readCubistOut(out,predDF)
#    mask = ((predDF["rnet"] < 0.0) | 
#            (predDF["lai"] < 0.0) | (predDF["trad2"] < 0.0) |
#            (predDF["precip"] < p1) | (predDF["precip"] >= p2) | 
#            (predDF["fmax"] < f1) | (predDF["fmax"] >= f2))
#    pred1[mask]=-9999.
#    pred1[pred1==-9999.]=np.nan
#    #===============================================================================
#    
#    #============create final_p250_fmax20.f90======================================
#    #========create fsun.data=====================================================
#    p1 = 0
#    p2 = 250
#    f1 = 0.2
#    f2 = 1.0
#    
#    out = outDF.loc[(outDF["fsun"] > 0.0) & (outDF["rnet"] > 0.0) & 
#                    (outDF["lai"] > 0.0) & (outDF["trad2"] > 0.0) &
#                    (outDF["dthr"] > 0.0) & (outDF["precip"] >= p1) &
#                    (outDF["precip"] < p2) & (outDF["fmax"] >= f1) &
#                    (outDF["fmax"] < f2), ["fsun","dthr","rnet_dthr","rnet","trad2"]]
#
#    file_data = os.path.join(fsun_trees_tile_ctl,'fsun.data')
#    out.to_csv(file_data , header=True, index=False,columns=["fsun",
#                                    "dthr","rnet_dthr","rnet","trad2"])
#        
#    file_names = os.path.join(fsun_trees_tile_ctl,'fsun.names')
#    get_trees_fstem_names(file_names)
#    
#    #====run cubist======================================
#    cubist_name = os.path.join(fsun_trees_tile_ctl,'fsun')
#    out = subprocess.check_output("cubist -f %s -u -a -i -r 8 -S 75" % cubist_name, shell=True)
##    print out
#    pred2 = readCubistOut(out,predDF)
#    mask = ((predDF["rnet"] < 0.0) | 
#            (predDF["lai"] < 0.0) | (predDF["trad2"] < 0.0) |
#            (predDF["precip"] < p1) | (predDF["precip"] >= p2) | 
#            (predDF["fmax"] < f1) | (predDF["fmax"] >= f2))
#    pred2[mask]=-9999.
#    pred2[pred2==-9999.]=np.nan
#    #===============================================================================
#    
#    #============create final_p500.f90======================================
#    #========create fsun.data=====================================================
#    p1 = 250
#    p2 = 500
#    f1 = 0.0
#    f2 = 1.0
#    
#    out = outDF.loc[(outDF["fsun"] > 0.0) & (outDF["rnet"] > 0.0) & 
#                    (outDF["lai"] > 0.0) & (outDF["trad2"] > 0.0) &
#                    (outDF["dthr"] > 0.0) & (outDF["precip"] >= p1) &
#                    (outDF["precip"] < p2) & (outDF["fmax"] >= f1) &
#                    (outDF["fmax"] < f2), ["fsun","dthr","rnet_dthr","rnet","trad2"]]
#
#    file_data = os.path.join(fsun_trees_tile_ctl,'fsun.data')
#    out.to_csv(file_data , header=True, index=False,columns=["fsun",
#                                    "dthr","rnet_dthr","rnet","trad2"])
#        
#    file_names = os.path.join(fsun_trees_tile_ctl,'fsun.names')
#    get_trees_fstem_names(file_names)
#    
#    #====run cubist======================================
#    cubist_name = os.path.join(fsun_trees_tile_ctl,'fsun')
#    out = subprocess.check_output("cubist -f %s -u -a -i -r 8 -S 50" % cubist_name, shell=True)
##    print out
#    pred3 = readCubistOut(out,predDF)
#    mask = ((predDF["rnet"] < 0.0) | 
#            (predDF["lai"] < 0.0) | (predDF["trad2"] < 0.0) |
#            (predDF["precip"] < p1) | (predDF["precip"] >= p2) | 
#            (predDF["fmax"] < f1) | (predDF["fmax"] >= f2))
#    pred3[mask]=-9999.
#    pred3[pred3==-9999.]=np.nan
#    #===============================================================================
#    
#    #============create final_p1000.f90======================================
#    #========create fsun.data=====================================================
#    p1 = 500
#    p2 = 1000
#    f1 = 0.0
#    f2 = 1.0
#    
#    out = outDF.loc[(outDF["fsun"] > 0.0) & (outDF["rnet"] > 0.0) & 
#                    (outDF["lai"] > 0.0) & (outDF["trad2"] > 0.0) &
#                    (outDF["dthr"] > 0.0) & (outDF["precip"] >= p1) &
#                    (outDF["precip"] < p2) & (outDF["fmax"] >= f1) &
#                    (outDF["fmax"] < f2), ["fsun","dthr","rnet_dthr","rnet","trad2"]]
#
#    file_data = os.path.join(fsun_trees_tile_ctl,'fsun.data')
#    out.to_csv(file_data , header=True, index=False,columns=["fsun",
#                                    "dthr","rnet_dthr","rnet","trad2"])
#        
#    file_names = os.path.join(fsun_trees_tile_ctl,'fsun.names')
#    get_trees_fstem_names(file_names)
#    
#    #====run cubist======================================
#    cubist_name = os.path.join(fsun_trees_tile_ctl,'fsun')
#    out = subprocess.check_output("cubist -f %s -u -a -i -r 8 -S 50" % cubist_name, shell=True)
##    print out
#    pred4 = readCubistOut(out,predDF)
#    mask = ((predDF["rnet"] < 0.0) | 
#            (predDF["lai"] < 0.0) | (predDF["trad2"] < 0.0) |
#            (predDF["precip"] < p1) | (predDF["precip"] >= p2) | 
#            (predDF["fmax"] < f1) | (predDF["fmax"] >= f2))
#    pred4[mask]=-9999.
#    pred4[pred4==-9999.]=np.nan
#    #===============================================================================
#    
#    #============create final_p2000.f90======================================
#    #========create fsun.data=====================================================
#    p1 = 1000
#    p2 = 9999
#    f1 = 0.0
#    f2 = 1.0
#    
#    out = outDF.loc[(outDF["fsun"] > 0.0) & (outDF["rnet"] > 0.0) & 
#                    (outDF["lai"] > 0.0) & (outDF["trad2"] > 0.0) &
#                    (outDF["dthr"] > 0.0) & (outDF["precip"] >= p1) &
#                    (outDF["precip"] < p2) & (outDF["fmax"] >= f1) &
#                    (outDF["fmax"] < f2), ["fsun","dthr","rnet_dthr","rnet","trad2"]]
#
#    file_data = os.path.join(fsun_trees_tile_ctl,'fsun.data')
#    out.to_csv(file_data , header=True, index=False,columns=["fsun",
#                                    "dthr","rnet_dthr","rnet","trad2"])
#        
#    file_names = os.path.join(fsun_trees_tile_ctl,'fsun.names')
#    get_trees_fstem_names(file_names)
#    
#    #====run cubist======================================
#    cubist_name = os.path.join(fsun_trees_tile_ctl,'fsun')
#    out = subprocess.check_output("cubist -f %s -u -a -i -r 8 -S 50" % cubist_name, shell=True)
##    print out
#    pred5 = readCubistOut(out,predDF)
#    mask = ((predDF["rnet"] < 0.0) | 
#            (predDF["lai"] < 0.0) | (predDF["trad2"] < 0.0) |
#            (predDF["precip"] < p1) | (predDF["precip"] >= p2) | 
#            (predDF["fmax"] < f1) | (predDF["fmax"] >= f2))
#    pred5[mask]=-9999.
#    pred5[pred5==-9999.]=np.nan
#    #===============================================================================
#    
#    #=====use the trees to estimate fsun===========================================
#    
#    aa = np.array([pred1,pred2,pred3,pred4,pred5])
#    a_nans = np.sum(np.isnan(aa),axis=0)
#    a_nans = a_nans.reshape([3750,3750])
#    a_nans = np.array(a_nans,dtype='Float32')
#    out_nancount_fn = os.path.join(tile_base_path,'T%03d' % tile, 'FINAL_NAN_COUNT_%s_T%03d.dat' % (date,tile))
#    a = np.nansum(aa,axis=0)
#    final_pred = a.reshape([3750,3750])
#    final_pred = np.array(final_pred,dtype='Float32')
#    out_fsun_fn = os.path.join(tile_base_path,'T%03d' % tile, 'FINAL_FSUN_%s_T%03d.dat' % (date,tile))
## =============================================================================
#    final_pred.tofile(out_fsun_fn)
#    convertBin2tif(out_fsun_fn,inUL,ALEXI_shape,ALEXI_res,'float32',gdal.GDT_Float32)
#    a_nans.tofile(out_nancount_fn)
#    convertBin2tif(out_nancount_fn,inUL,ALEXI_shape,ALEXI_res,'float32',gdal.GDT_Float32)
## =============================================================================

def processTrees(year,doy):
    date = '%d%03d' % (year,doy)
    dtimedates = np.array(range(1,366,7))
    r7day = dtimedates[dtimedates>=doy][0]
    riseddd="%d%03d" %(year,r7day)
    fsun_trees_tile_ctl = os.path.join(fsun_trees_path,'tiles_ctl')
    if not os.path.exists(fsun_trees_tile_ctl):
        os.makedirs(fsun_trees_tile_ctl) 
    ##===========create dictionary and convert to csv=======
    #======load 5 km data and subset it========================================  
    dthr_zip_fn = os.path.join(static_path,"5KM","DTHR","DTHR%s.dat.gz" % riseddd)  
#    dthr_fn = os.path.join("./DTHR%s.dat" % date)  
    gunzip(dthr_zip_fn)
    dthr = np.fromfile(dthr_zip_fn[:-3], dtype=np.float32)
    dthr = np.flipud(dthr.reshape([3000,7200]))
    dthr_sub = dthr[901:1801,3201:4801]
#    plt.imshow(dthr_sub)
    dthr = np.reshape(dthr_sub,[dthr_sub.size])
    
    rnet_zip_fn = os.path.join(static_path,"5KM","RNET","RNET%s.dat.gz" % riseddd)  
#    rnet_fn = os.path.join("./RNET%s.dat" % date)  
    gunzip(rnet_zip_fn)
    rnet = np.fromfile(rnet_zip_fn[:-3], dtype=np.float32)
    rnet = np.flipud(rnet.reshape([3000,7200]))
    rnet_sub = rnet[901:1801,3201:4801]
#    plt.imshow(rnet_sub)
    rnet = np.reshape(rnet_sub,[rnet_sub.size])
    
    fsun_src_fn = os.path.join(static_path,"5KM","FSUN","FSUN%s.dat" % riseddd)  
#    fsun_fn = os.path.join("./FSUN%s.dat" % date)  
#    shutil.copyfile(fsun_src_fn)
    fsun = np.fromfile(fsun_src_fn, dtype=np.float32)
    fsun = np.flipud(fsun.reshape([3000,7200]))
    fsun_sub = fsun[901:1801,3201:4801]
#    plt.imshow(fsun_sub[100:400,1000:1300],vmin=0, vmax=0.5)
    fsun  = np.reshape(fsun_sub,[fsun_sub.size])
    
    rnet_dthr = rnet/dthr    
    # note* FMAX is actually max LAI here
    fmax_src_fn = os.path.join(static_path,"5KM","FMAX","FMAX.dat")  
#    fmax_fn = os.path.join("./FMAX.dat")  
#    shutil.copyfile(fmax_src_fn,fmax_fn)
    fmax = np.fromfile(fmax_src_fn, dtype=np.float32)
    fmax = 1-np.exp(-0.5*fmax)
    fmax_sub = np.flipud(fmax.reshape([900,1600]))
#    plt.imshow(fmax_sub, vmin=0, vmax=0.8)
    fmax  = np.reshape(fmax_sub,[fmax_sub.size])
    
    precip_src_fn = os.path.join(static_path,"5KM","PRECIP","PRECIP.dat")  
#    precip_fn = os.path.join("./PRECIP.dat")  
#    shutil.copyfile(precip_src_fn,precip_fn)
    precip = np.fromfile(precip_src_fn, dtype=np.float32)
    precip_sub = np.flipud(precip.reshape([900,1600]))
#    plt.imshow(precip_sub)
    precip  = np.reshape(precip_sub,[precip_sub.size])
    
    trad2_src_fn = os.path.join(static_path,"5KM","TRAD2","TRD2%s.dat.gz" % riseddd)  
#    trad2_fn = os.path.join("./TRD2%s.dat" % date)  
    gunzip(trad2_src_fn)
    trad2 = np.fromfile(trad2_src_fn[:-3], dtype=np.float32)
    trad2 = np.flipud(trad2.reshape([3000,7200]))
    trad2_sub = trad2[901:1801,3201:4801]
#    plt.imshow(trad2_sub,vmin=280, vmax=320)
    trad2 = np.reshape(trad2_sub,[trad2_sub.size])
    
    lai_src_fn = os.path.join(static_path,"5KM","LAI","MLAI2014%03d.dat" % r7day)  
#    lai_fn = os.path.join("./LAI2014%03d.dat" % doy)  
#    shutil.copyfile(lai_src_fn,lai_fn)
    lai = np.fromfile(lai_src_fn, dtype=np.float32)
    lai = lai.reshape([3000,7200])
    lai_sub = lai[901:1801,3201:4801]
#    plt.imshow(lai_sub,vmin=0, vmax=2)
    lai = np.reshape(lai_sub,[lai_sub.size])
    
    outDict = {'fsun':fsun, 'dthr':dthr, 'rnet_dthr':rnet_dthr, 
               'rnet': rnet, 'fmax':fmax, 'precip':precip,
               'lai':lai,'trad2':trad2}
    outDF = pd.DataFrame.from_dict(outDict)
    

    
    #============create final_p250_fmax0.f90======================================
    #========create fsun.data=====================================================
    p1 = 0
    p2 = 250
    f1 = 0
    f2 = 0.2
    out = outDF.loc[(outDF["fsun"] > 0.0) & (outDF["rnet"] > 0.0) & 
                    (outDF["lai"] > 0.0) & (outDF["trad2"] > 0.0) &
                    (outDF["dthr"] > 0.0) & (outDF["precip"] >= p1) &
                    (outDF["precip"] < p2) & (outDF["fmax"] >= f1) &
                    (outDF["fmax"] < f2), ["fsun","dthr","rnet_dthr","rnet","trad2"]]

    file_data = os.path.join(fsun_trees_tile_ctl,'fsun.data')
    out.to_csv(file_data , header=True, index=False,columns=["fsun",
                                    "dthr","rnet_dthr","rnet","trad2"])
        
    file_names = os.path.join(fsun_trees_tile_ctl,'fsun.names')
    get_trees_fstem_names(file_names)
    
    #====run cubist======================================
    cubist_name = os.path.join(fsun_trees_tile_ctl,'fsun')
    out1 = subprocess.check_output("cubist -f %s -u -a -i -r 8 -S 75" % cubist_name, shell=True)

    #============create final_p250_fmax20.f90======================================
    #========create fsun.data=====================================================
    p1 = 0
    p2 = 250
    f1 = 0.2
    f2 = 1.0
    
    out = outDF.loc[(outDF["fsun"] > 0.0) & (outDF["rnet"] > 0.0) & 
                    (outDF["lai"] > 0.0) & (outDF["trad2"] > 0.0) &
                    (outDF["dthr"] > 0.0) & (outDF["precip"] >= p1) &
                    (outDF["precip"] < p2) & (outDF["fmax"] >= f1) &
                    (outDF["fmax"] < f2), ["fsun","dthr","rnet_dthr","rnet","trad2"]]

    file_data = os.path.join(fsun_trees_tile_ctl,'fsun.data')
    out.to_csv(file_data , header=True, index=False,columns=["fsun",
                                    "dthr","rnet_dthr","rnet","trad2"])
        
    file_names = os.path.join(fsun_trees_tile_ctl,'fsun.names')
    get_trees_fstem_names(file_names)
    
    #====run cubist======================================
    cubist_name = os.path.join(fsun_trees_tile_ctl,'fsun')
    out2 = subprocess.check_output("cubist -f %s -u -a -i -r 8 -S 75" % cubist_name, shell=True)
    #============create final_p500.f90======================================
    #========create fsun.data=====================================================
    p1 = 250
    p2 = 500
    f1 = 0.0
    f2 = 1.0
    
    out = outDF.loc[(outDF["fsun"] > 0.0) & (outDF["rnet"] > 0.0) & 
                    (outDF["lai"] > 0.0) & (outDF["trad2"] > 0.0) &
                    (outDF["dthr"] > 0.0) & (outDF["precip"] >= p1) &
                    (outDF["precip"] < p2) & (outDF["fmax"] >= f1) &
                    (outDF["fmax"] < f2), ["fsun","dthr","rnet_dthr","rnet","trad2"]]

    file_data = os.path.join(fsun_trees_tile_ctl,'fsun.data')
    out.to_csv(file_data , header=True, index=False,columns=["fsun",
                                    "dthr","rnet_dthr","rnet","trad2"])
        
    file_names = os.path.join(fsun_trees_tile_ctl,'fsun.names')
    get_trees_fstem_names(file_names)
    
    #====run cubist======================================
    cubist_name = os.path.join(fsun_trees_tile_ctl,'fsun')
    out3 = subprocess.check_output("cubist -f %s -u -a -i -r 8 -S 50" % cubist_name, shell=True)

    #============create final_p1000.f90======================================
    #========create fsun.data=====================================================
    p1 = 500
    p2 = 1000
    f1 = 0.0
    f2 = 1.0
    
    out = outDF.loc[(outDF["fsun"] > 0.0) & (outDF["rnet"] > 0.0) & 
                    (outDF["lai"] > 0.0) & (outDF["trad2"] > 0.0) &
                    (outDF["dthr"] > 0.0) & (outDF["precip"] >= p1) &
                    (outDF["precip"] < p2) & (outDF["fmax"] >= f1) &
                    (outDF["fmax"] < f2), ["fsun","dthr","rnet_dthr","rnet","trad2"]]

    file_data = os.path.join(fsun_trees_tile_ctl,'fsun.data')
    out.to_csv(file_data , header=True, index=False,columns=["fsun",
                                    "dthr","rnet_dthr","rnet","trad2"])
        
    file_names = os.path.join(fsun_trees_tile_ctl,'fsun.names')
    get_trees_fstem_names(file_names)
    
    #====run cubist======================================
    cubist_name = os.path.join(fsun_trees_tile_ctl,'fsun')
    out4 = subprocess.check_output("cubist -f %s -u -a -i -r 8 -S 50" % cubist_name, shell=True)

    #============create final_p2000.f90======================================
    #========create fsun.data=====================================================
    p1 = 1000
    p2 = 9999
    f1 = 0.0
    f2 = 1.0
    
    out = outDF.loc[(outDF["fsun"] > 0.0) & (outDF["rnet"] > 0.0) & 
                    (outDF["lai"] > 0.0) & (outDF["trad2"] > 0.0) &
                    (outDF["dthr"] > 0.0) & (outDF["precip"] >= p1) &
                    (outDF["precip"] < p2) & (outDF["fmax"] >= f1) &
                    (outDF["fmax"] < f2), ["fsun","dthr","rnet_dthr","rnet","trad2"]]

    file_data = os.path.join(fsun_trees_tile_ctl,'fsun.data')
    out.to_csv(file_data , header=True, index=False,columns=["fsun",
                                    "dthr","rnet_dthr","rnet","trad2"])
        
    file_names = os.path.join(fsun_trees_tile_ctl,'fsun.names')
    get_trees_fstem_names(file_names)
    
    #====run cubist======================================
    cubist_name = os.path.join(fsun_trees_tile_ctl,'fsun')
    out5 = subprocess.check_output("cubist -f %s -u -a -i -r 8 -S 50" % cubist_name, shell=True)
    
    return [out1,out2,out3,out4,out5]

    
def useTrees(year,doy,tile,trees):
    LLlat,LLlon = tile2latlon(tile)
    URlat = LLlat+15.
    inUL = [LLlon,URlat]
    ALEXI_shape = [3750,3750]
    ALEXI_res = [0.004,0.004]
    dtimedates = np.array(range(1,366,7))
    r7day = dtimedates[dtimedates>=doy][0]
    date = '%d%03d' % (year,doy)
    p1 = [0,0,250,500,1000]
    p2 = [250,250,500,1000,9999]
    f1 = [0,0.2,0.0,0.0,0.0]
    f2 = [0.2,1.0,1.0,1.0,1.0]
    #=======ALEXI resolution inputs===============================================
    laidates = np.array(range(1,366,4))
    r4day = laidates[laidates>=doy][0]
    laiddd="%d%03d" %(year,r4day)
    dthr_fn = os.path.join(tile_base_path,'T%03d' % tile, 'FINAL_DTRAD_%s_T%03d.dat' % (date,tile))
    trad2_fn = os.path.join(tile_base_path,'T%03d' % tile, 'FINAL_DAY_LST_TIME2_%s_T%03d.dat' % (date,tile))
    rnet_fn = os.path.join(tile_base_path,'T%03d' % tile, 'FINAL_RNET_%s_T%03d.dat' % (date,tile))
    lai_fn = os.path.join(static_path,'LAI','MLAI_%s_T%03d.dat' % (laiddd,tile)) # only have 2015 so far
    dthr_corr_fn = os.path.join(static_path,'DTHR_CORR','DTHR_CORR_2010%03d_T%03d.dat' % (r7day,tile))
    dtime_fn = os.path.join(static_path,'DTIME','DTIME_2014%03d_T%03d.dat' % (r7day,tile))
    fmax_fn = os.path.join(static_path,'FMAX','FMAX_T%03d.dat' % (tile))
    precip_fn = os.path.join(static_path,'PRECIP','PRECIP_T%03d.dat' % (tile))
    
    dthr = np.fromfile(dthr_fn, dtype=np.float32)
#    dthr = np.flipud(dthr.reshape([3750,3750]))
#    plt.imshow(dthr)
#    dthr = np.reshape(dthr,[3750*3750])
    trad2 = np.fromfile(trad2_fn, dtype=np.float32)
#    trad2 = np.flipud(trad2.reshape([3750,3750]))
#    plt.imshow(trad2)
#    trad2 = np.reshape(trad2,[3750*3750])
    rnet = np.fromfile(rnet_fn, dtype=np.float32)
#    rnet = np.flipud(rnet.reshape([3750,3750]))
#    plt.imshow(rnet)
#    rnet = np.reshape(rnet,[3750*3750])    
    lai = np.fromfile(lai_fn, dtype=np.float32)
#    lai = np.flipud(lai.reshape([3750,3750]))
#    plt.imshow(lai, vmin=0, vmax=2)
#    lai = np.reshape(lai,[3750*3750])
    dthr_corr = np.fromfile(dthr_corr_fn, dtype=np.float32)
#    dthr_corr = np.flipud(dthr_corr.reshape([3750,3750]))
#    plt.imshow(dthr_corr, vmin=0,vmax=3)
#    dthr_corr = np.reshape(dthr_corr,[3750*3750])    
    dtime = np.fromfile(dtime_fn, dtype=np.float32)
#    dtime = np.flipud(dtime.reshape([3750,3750]))
#    plt.imshow(dtime)
#    dtime = np.reshape(dtime,[3750*3750])
    fmax = np.fromfile(fmax_fn, dtype=np.float32)
#    fmax = np.flipud(fmax.reshape([3750,3750]))
#    plt.imshow(fmax, vmin=0, vmax=0.3)
#    fmax = np.reshape(fmax,[3750*3750])    
    precip = np.fromfile(precip_fn, dtype=np.float32)
#    precip = np.flipud(precip.reshape([3750,3750]))
#    plt.imshow(precip)
#    precip = np.reshape(precip,[3750*3750])    
    dthr = (dthr/dtime)*dthr_corr
    
    rnet_dthr = rnet/dthr

    
    predDict = {'dthr':dthr,'rnet_dthr':rnet_dthr,'rnet': rnet,'trad2':trad2,
                'fmax':fmax, 'precip':precip, 'lai':lai}
    predDF = pd.DataFrame.from_dict(predDict)
    
    outDF = []
    for i in range(len(trees)):
        mask = ((predDF["rnet"] < 0.0) | 
                (predDF["lai"] < 0.0) | (predDF["trad2"] < 0.0) |
                (predDF["precip"] < p1[i]) | (predDF["precip"] >= p2[i]) | 
                (predDF["fmax"] < f1[i]) | (predDF["fmax"] >= f2[i]))
        out = readCubistOut(trees[i],predDF)
        out[mask]=np.nan
        outDF.append(out)
        
        #=====use the trees to estimate fsun===========================================
    
    aa = np.array(outDF)
    a_nans = np.sum(np.isnan(aa),axis=0)
    a_nans = a_nans.reshape([3750,3750])
    a_nans = np.array(a_nans,dtype='Float32')
    out_nancount_fn = os.path.join(tile_base_path,'T%03d' % tile, 'FINAL_NAN_COUNT_%s_T%03d.dat' % (date,tile))
    a = np.nansum(aa,axis=0)
    final_pred = a.reshape([3750,3750])
    final_pred = np.array(final_pred,dtype='Float32')
    out_fsun_fn = os.path.join(tile_base_path,'T%03d' % tile, 'FINAL_FSUN_%s_T%03d.dat' % (date,tile))
# =============================================================================
    final_pred.tofile(out_fsun_fn)
    convertBin2tif(out_fsun_fn,inUL,ALEXI_shape,ALEXI_res,'float32',gdal.GDT_Float32)
    a_nans.tofile(out_nancount_fn)
    convertBin2tif(out_nancount_fn,inUL,ALEXI_shape,ALEXI_res,'float32',gdal.GDT_Float32)
# =============================================================================

#    gzipped(lst_path)
#    gzipped(dtrad_path)
def getDailyET(tile,year,doy):
    LLlat,LLlon = tile2latlon(tile)
    URlat = LLlat+15.
    inUL = [LLlon,URlat]
    ALEXI_shape = [3750,3750]
    ALEXI_res = [0.004,0.004]
    date = '%d%03d' % (year,doy)
    insol24_fn = os.path.join(static_path,'INSOL24', 'RS24_%s_T%03d.tif' % (date,tile))
    g = gdal.Open(insol24_fn,GA_ReadOnly)
    Rs24= g.ReadAsArray()
    Rs24=(Rs24*0.0864)/24.0 
    fsun_fn = os.path.join(tile_base_path,'T%03d' % tile, 'FINAL_FSUN_%s_T%03d.dat' % (date,tile))
#    Rs24 = np.fromfile(insol24_fn, dtype=np.float32)
#    Rs24 = np.flipud(Rs24.reshape([3750,3750]))
    Fsun = np.fromfile(fsun_fn, dtype=np.float32)
    Fsun = Fsun.reshape([3750,3750])
    EFeq=Fsun*(Rs24)
    ET_24 = EFeq/2.45
    ET_24[ET_24<0.01]=0.01
    ET_24 = np.array(ET_24,dtype='Float32')
    out_ET_fn = os.path.join(tile_base_path,'T%03d' % tile, 'FINAL_EDAY_%s_T%03d.dat' % (date,tile))
#    ET_24 = np.reshape()
    ET_24.tofile(out_ET_fn)
    convertBin2tif(out_ET_fn,inUL,ALEXI_shape,ALEXI_res,np.float32,gdal.GDT_Float32)
      
#def main():
#    # Get time and location from user
#    parser = argparse.ArgumentParser()
#    parser.add_argument("tile", type=int, help="15x15 deg tile number")
#    parser.add_argument("year", type=int, help="year of data")
#    parser.add_argument("doy", type=int, help="day of year of data")
#    args = parser.parse_args()
#      
#    tile = args.tile
#    year = args.year
#    doy = args.doy
#######this should be run when downloading data########
#data_cache = os.path.join(data_path,"2016","12")
#ff = glob.glob(os.path.join(data_cache,"SVI05*"))
#for fn in ff:
#    aa = get_VIIRS_bounds(fn)
#year = 2016
#tile = 87
#dd = datetime.datetime(2016,6,1)
#doy = (dd-datetime.datetime(2016,1,1)).days
#tile = 63

def runSteps(tile,trees,year=None,doy=None):
    if year==None:
        dd = datetime.date.today()+datetime.timedelta(days=-1)
        year = dd.year
        
    if doy==None:
        doy = (datetime.date.today()-datetime.date(year,1,1)).days-1


#    
#    print("gridding I5---------------------------------->")
#    regrid_I5(tile,year,doy)
#    print("gridding cloud------------------------------->")
#    regrid_cloud(tile,year,doy)
#    #print("Applying Mask--------------->")
#    #Apply_mask(tile,year,doy)
#    #getIJcoords(tile)
#    print("merging BT--------------->")
#    merge_bt(tile,year,doy)
#    end = timer.time()
#    print("old way time: %f" % (end - start))
    print("building VIIRS coordinates LUT--------------->")
    getIJcoordsPython(tile)
#    #getCFSRdata(year,doy)
#    startmerge = timer.time()
    print("gridding VIIRS data-------------------------->")
    res = gridMergePython(tile,year,doy)
    if res > 0:
        print("no viirs data")
    else:
    #    end = timer.time()
    #    print("new way time: %f" % (end - startmerge))
        print("running I5 atmosperic correction------------->")
    #    startatmos = timer.time()
        atmosCorrection(tile,year,doy)
    #    end = timer.time()
    #    print("atmoscorr time: %f" % (end - startatmos))
        #print("merging LST--------------->")
        #merge_lst(tile,year,doy)
        print("estimating dtrad and LST2-------------------->")
        pred_dtrad(tile,year,doy)
        print("estimating RNET ----------------------------->")
        processTiles(year,doy,tile)
        print("estimating FSUN------------------------------>")
    #    processTrees(year,doy,tile)
        useTrees(year,doy,tile,trees)
        print("making ET------------------------------------>")
        getDailyET(tile,year,doy)
        print("============FINISHED!=========================")

year = 2015
doy = 221
tiles = [60,61,62,63,64,83,84,85,86,87,88,107,108,109,110,111,112]
#tiles = [64,87,88]
#tiles = [61]
start = timer.time() 
print("building regression trees from 5KM data---------->")
trees = processTrees(year,doy)
#for tile in tiles:
#    print("========processing T%03d======================" % tile)
#    runSteps(tile,trees,year,doy) 
r = Parallel(n_jobs=-1, verbose=5)(delayed(runSteps)(tile,trees,year,doy) for tile in tiles)
end = timer.time()
print(end - start)
