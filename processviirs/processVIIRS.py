#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 08:54:48 2017

@author: mschull
"""
import os
import datetime
import pandas as pd
import numpy as np
import glob
import h5py
from pyresample import kd_tree, geometry
from pyresample import utils
import numpy.ma as ma
from osgeo import gdal,osr
import shutil
import gzip
import ephem
import subprocess
from osgeo.gdalconst import GA_ReadOnly
from joblib import Parallel, delayed
import time as timer
from pyresample.ewa import ll2cr, fornav
import argparse
import warnings
warnings.filterwarnings("ignore",category =RuntimeWarning)

#========utility functions=====================================================
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
    rnet_tile_path = os.path.join(calc_rnet_path,'tiles')
    if not os.path.exists(rnet_tile_path):
        os.makedirs(rnet_tile_path) 
    dtrad_path = os.path.join(processing_path,'DTRAD_PREDICTION')
    if not os.path.exists(dtrad_path):
        os.makedirs(dtrad_path)
    out = {'grid_I5_path':grid_I5_path,'grid_I5_temp_path':grid_I5_temp_path,
           'agg_I5_path':agg_I5_path,'data_path':data_path,
           'cloud_grid': cloud_grid,'temp_cloud_data':temp_cloud_data,
           'agg_cloud_path':agg_cloud_path,'processing_path':processing_path,
           'static_path':static_path,'tile_base_path':tile_base_path,
           'overpass_correction_path':overpass_correction_path,
           'CFSR_path':CFSR_path,'calc_rnet_path':calc_rnet_path,
           'fsun_trees_path':fsun_trees_path,'rnet_tile_path':rnet_tile_path}
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
rnet_tile_path = Folders['rnet_tile_path']


def tile2latlon(tile):
    row = tile/24
    col = tile-(row*24)
    # find lower left corner
    lat= (75.-row*15.)-15.
    lon=(col*15.-180.)-15. 
    return [lat,lon]

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
    ds.FlushCache()  
    
def convertBin2tif(inFile,inUL,shape,res,informat,outFormat):
    inProj4 = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'
    read_data = np.fromfile(inFile, dtype=informat)
    dataset = read_data.reshape([shape[0],shape[1]])
    dataset = np.array(dataset,dtype=informat)
    outTif = inFile[:-4]+".tif"
    writeArray2Tiff(dataset,res,inUL,inProj4,outTif,outFormat) 
    
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

def writeCTL(tile,year,doy):
    LLlat,LLlon = tile2latlon(tile)
    #====create insol.ctl====================================
    date = "%d%03d" % (year,doy)
    date_tile_str = "T%03d_%s" % (tile,date)
    
    srcfn = os.path.join(static_path,'INSOL','deg125','insol55_2011%03d.dat' % doy)
    
    dtimedates = np.array(range(1,366,7))
    rday = dtimedates[dtimedates>=doy][0]
#    riseddd="%d%03d" %(year,rday)
    riseddd="%d%03d" %(2015,rday) # FOR RT UNTIL I GET RT DATA FROM CHRIS
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


def write_agg_insol(outfn,date_tile_str):
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
        print(b)
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

            lstOut[np.where(rule2use)] = eval('(%s)' % formula)
            lstOut[np.where(mask)] = -9999.
    return lstOut  

def planck(X,ANV):
    C1=1.191E-9
    C2=1.439
    C1P = C1*ANV**3		# different wavelength##
    C2P = C2*ANV
    return C1P/(np.exp(C2P/X)-1.)

def invplanck(X,ANV):
    C1=1.191E-9
    C2=1.439
    C1P = C1*ANV**3
    C2P = C2*ANV
    return C2P/np.log(1.+C1P/X)

#  This function contains an empirical correction for absorption
#  by non-water-vapor constituents.  It was developed in comparison
#  with a series of MODTRAN experiments.     
# FUNCTION DTAUDP(WAVENUMBER,TEMPERATURE,VAPOR PRESSURE,PRESSURE)
def dtaudp(W,X,Y,Z):
    GO=9.78
    DTAUDP1=(4.18+5578.*np.exp((-7.87E-3)*W))*np.exp(1800.*
            (1.0/X-1.0/296.))*(Y/Z+.002)*(.622/(101.3*GO))*Y
    return DTAUDP1+(0.00004+Y/60000.)
#=======functions for processing===============================================
#==============================================================================
    

def processTrees(year=None,doy=None):
    inProjection = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'
    if year==None:
        dd = datetime.date.today()+datetime.timedelta(days=-1)
        year = dd.year 
    if doy==None:
        doy = (datetime.date.today()-datetime.date(year,1,1)).days-1
    year = 2015 # TEMP FOR RT PROCESSING  
        
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
    writeArray2Tiff(fsun,[0.05,0.05],[-180.,90],inProjection,fsun_src_fn[:-4]+'.tif',gdal.GDT_Float32)
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
    

    
    #============create final_p250_fmax0.f90===================================
    #========create fsun.data==================================================
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

    #============create final_p250_fmax20.f90==================================
    #========create fsun.data==================================================
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
    #============create final_p500.f90=========================================
    #========create fsun.data==================================================
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

    #============create final_p1000.f90========================================
    #========create fsun.data==================================================
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

    #============create final_p2000.f90========================================
    #========create fsun.data==================================================
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

def getIJcoordsPython(tile):
    lat,lon = tile2latlon(tile)
#    lat = lat
    tile_lut_path = os.path.join(static_path,"CFSR","viirs_tile_lookup_tables")
    if not os.path.exists(tile_lut_path):
        os.makedirs(tile_lut_path) 
    icoordpath = os.path.join(tile_lut_path,"CFSR_T%03d_lookup_icoord.dat" % tile)
    jcoordpath = os.path.join(tile_lut_path,"CFSR_T%03d_lookup_jcoord.dat" % tile)
    
    istart = (180+lon)*4
    addArray = np.floor(np.array(range(3750))*0.004/0.25)
    icor = istart+addArray
    icormat = np.repeat(np.reshape(icor,[icor.size,1]),3750,axis=1)
    icormat = icormat.T
    icormat = np.array(icormat,dtype='int32')
    icormat.tofile(icoordpath) 
    
    jstart = (89.875+lat)*4
    jcor = jstart+addArray
    jcormat = np.repeat(np.reshape(jcor,[jcor.size,1]),3750,axis=1)
    jcormat = np.array(jcormat,dtype='int32')
    jcormat.tofile(jcoordpath)
    
def gridMergePythonEWA(tile,year,doy):
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
    
    
    #=====================Day==================================================
    #==========================================================================
    files = db[(db['south']-5 <= latmid) & (db['north']+5 >= latmid) & 
               (db['west']-5 <= lonmid) & (db['east']+5 >= lonmid) & 
               (db['year'] == year) & (db['doy'] == doy) & (db['N_Day_Night_Flag'] == 'Day')]
    filenames = files['filename']

    orbits = []
    for fn in filenames:
        parts = fn.split(os.sep)[-1].split('_')
        orbits.append(parts[5])
    orbits = list(set(orbits)) 
    orbitcount = 0
    for orbit in orbits:   
        fns = [s for s in filenames if orbit in s.lower()]
        count = 0
        for filename in fns:
            count+=1
            folder = os.sep.join(filename.split(os.sep)[:-1])
            parts = filename.split(os.sep)[-1].split('_')
            search_geofile = os.path.join(folder,"*"+"_".join(("GITCO",parts[1],parts[2],parts[3],parts[4])))
            geofile = glob.glob(search_geofile+'*')[0]
            search_geofile = os.path.join(folder,"*"+"_".join(("GMTCO",parts[1],parts[2],parts[3],parts[4])))
            datet = datetime.datetime(year,dd.month, dd.day,0,0,0)
            if datet > datetime.datetime(2017,3,8,0,0,0):
               search_cloudfile = os.path.join(folder,"*"+"_".join(("VICMO",parts[1],parts[2],parts[3],parts[4]))) 
            else:
               search_cloudfile = os.path.join(folder,"*"+"_".join(("IICMO",parts[1],parts[2],parts[3],parts[4])))
            cloudgeofile = glob.glob(search_geofile+'*')[0]
            cloudfile = glob.glob(search_cloudfile+'*')[0]
            
            # load cloud data==========
            f=h5py.File(cloudfile,'r')
            g=h5py.File(cloudgeofile,'r')
            if datet > datetime.datetime(2017,3,8,0,0,0):
                data_array = f['/All_Data/VIIRS-CM-EDR_All/QF1_VIIRSCMEDR'][()]
            else:
                data_array = f['/All_Data/VIIRS-CM-IP_All/QF1_VIIRSCMIP'][()]
            lat_array = g['/All_Data/VIIRS-MOD-GEO-TC_All/Latitude'][()]
            lon_array = g['/All_Data/VIIRS-MOD-GEO-TC_All/Longitude'][()]
            view_array = g['/All_Data/VIIRS-MOD-GEO-TC_All/SatelliteZenithAngle'][()]
            
            start=filename.find('_t')
            out_time=filename[start+2:start+6]
            
            if count ==1:
                latcloud = np.array(lat_array,'float32')
                loncloud=np.array(lon_array,'float32')
                cloud=np.array(data_array,'float32')
                viewcloud=np.array(view_array,'float32')
            else:
                latcloud = np.vstack((latcloud,np.array(lat_array,'float32')))
                loncloud = np.vstack((loncloud,np.array(lon_array,'float32')))
                cloud = np.vstack((cloud,np.array(data_array,'float32')))
                viewcloud = np.vstack((viewcloud,np.array(view_array,'float32')))
                
            # load water mask===========
            f=h5py.File(cloudfile,'r')
            g=h5py.File(cloudgeofile,'r')
            if datet > datetime.datetime(2017,3,8,0,0,0):
                data_array = f['/All_Data/VIIRS-CM-EDR_All/QF2_VIIRSCMEDR'][()]
            else:
                data_array = f['/All_Data/VIIRS-CM-IP_All/QF2_VIIRSCMIP'][()]
            lat_array = g['/All_Data/VIIRS-MOD-GEO-TC_All/Latitude'][()]
            lon_array = g['/All_Data/VIIRS-MOD-GEO-TC_All/Longitude'][()]
            view_array = g['/All_Data/VIIRS-MOD-GEO-TC_All/SatelliteZenithAngle'][()]
            
            start=filename.find('_t')
            out_time=filename[start+2:start+6]
            
            if count ==1:
                watermask=np.array(data_array,'float32')
            else:
                watermask = np.vstack((watermask,np.array(data_array,'float32')))
                
            #  load BT data============
            f=h5py.File(filename,'r')
            g=h5py.File(geofile,'r')
            data_array = f['/All_Data/VIIRS-I5-SDR_All/BrightnessTemperature'][()]
            lat_array = g['/All_Data/VIIRS-IMG-GEO-TC_All/Latitude'][()]
            lon_array = g['/All_Data/VIIRS-IMG-GEO-TC_All/Longitude'][()]
            view_array = g['/All_Data/VIIRS-IMG-GEO-TC_All/SatelliteZenithAngle'][()]
            if count ==1:
                lat=np.array(lat_array,'float32')
                lon=np.array(lon_array,'float32')
                data=np.array(data_array,'float32')
                view=np.array(view_array,'float32')
            else:
                lat = np.vstack((lat,np.array(lat_array,'float32')))
                lon = np.vstack((lon,np.array(lon_array,'float32')))
                data = np.vstack((data,np.array(data_array,'float32')))
                view = np.vstack((view,np.array(view_array,'float32')))
        #====cloud gridding=====================
        cloudOrig=cloud.copy()
        #get 2-3 bits
        cloud=np.array(cloud,'uint8')
        cloud = np.reshape(cloud,[cloud.size, 1])
        b = np.unpackbits(cloud, axis=1)
        cloud = np.sum(b[:,4:6],axis=1)
        cloud = np.reshape(cloud,[cloudOrig.shape[0],cloudOrig.shape[1]])
        cloud = np.array(cloud, dtype='float32')
        
        #====get water mask from bits===========
        
        watermask=np.array(watermask,'uint8')
        watermask = np.reshape(watermask,[watermask.size, 1])
        b = np.unpackbits(watermask, axis=1)
        watermask = np.sum(b[:,5:7],axis=1)
        watermask = np.reshape(watermask,[cloudOrig.shape[0],cloudOrig.shape[1]])
        watermask = np.array(watermask, dtype='float32')
        
        mask = (cloudOrig==0.)
        cloud[mask]=np.nan
        viewcloud[mask]=np.nan
        watermask[mask]=np.nan
        #=====check if data is in range========================================
        rangeIndex = ((latcloud<-90.) | (latcloud > 90.) | (loncloud < -180.) | (loncloud > 180.))
        latcloud[rangeIndex] = np.nan
        loncloud[rangeIndex] = np.nan
        cloud[rangeIndex] = np.nan
        viewcloud[rangeIndex] = np.nan
        watermask[rangeIndex] = np.nan
        if np.nansum(cloud)==0: # check if there is any data
            continue

        projection = '+proj=longlat +ellps=WGS84 +datum=WGS84'
        area_id ='tile'
        proj_id = 'latlon'
        description = 'lat lon grid'

        swath_def = geometry.SwathDefinition(lons=loncloud, lats=latcloud)
        x_size = 3750
        y_size = 3750
        area_extent = (LLlon,LLlat,URlon,URlat)
        area_def = utils.get_area_def(area_id, description, proj_id, projection,
                                                   x_size, y_size, area_extent)
        swath_points_in_grid, cols, rows = ll2cr(swath_def, area_def, copy=False)
        rows_per_scan = 16
        try: # if there are no valid pixels in the region move on
            num_valid_points, gridded_cloud = fornav(cols, rows, area_def, cloud, rows_per_scan=rows_per_scan)
            orbitcount+=1
        except:
            continue
        try:
            num_valid_points, gridded_cloudview = fornav(cols, rows, area_def, viewcloud, rows_per_scan=rows_per_scan)
        except:
            continue
        
        try:
            num_valid_points, gridded_watermask = fornav(cols, rows, area_def, watermask, rows_per_scan=rows_per_scan)
        except:
            continue
        
        gridded_cloud[gridded_cloudview>60.0]=np.nan
        gridded_watermask[gridded_cloudview>60.0]=np.nan
        #stack data
        if orbitcount==1:
            cloud_stack = gridded_cloud
            cloudview_stack = gridded_cloudview
            watermask_stack = gridded_watermask
        else:
            cloud_stack = np.dstack((cloud_stack,gridded_cloud))
            cloudview_stack = np.dstack((cloudview_stack,gridded_cloudview))
            watermask_stack = np.dstack((watermask_stack,gridded_watermask))
            
        #==LST gridding===========================
        mask = (data>65527.)
        data[mask]=np.nan
        view[mask]=np.nan
        #=====check if data is in range========================================
        rangeIndex = ((lat<-90.) | (lat > 90.) | (lon < -180.) | (lon > 180.))
        lat[rangeIndex] = np.nan
        lon[rangeIndex] = np.nan
        data[rangeIndex] = np.nan
        view[rangeIndex] = np.nan
        
        if np.nansum(data)==0: # check if there is any data
            continue
        projection = '+proj=longlat +ellps=WGS84 +datum=WGS84'
        area_id ='tile'
        proj_id = 'latlon'
        description = 'lat lon grid'

        swath_def = geometry.SwathDefinition(lons=lon, lats=lat)
        x_size = 3750
        y_size = 3750
        area_extent = (LLlon,LLlat,URlon,URlat)
        area_def = utils.get_area_def(area_id, description, proj_id, projection,
                                                   x_size, y_size, area_extent)
        swath_points_in_grid, cols, rows = ll2cr(swath_def, area_def, copy=False)
        rows_per_scan = 32
        try: # if there are no valid pixels in the region move on
            num_valid_points, gridded_data = fornav(cols, rows, area_def, data, rows_per_scan=rows_per_scan)
        except:
            continue
        try:
            num_valid_points, gridded_view = fornav(cols, rows, area_def, view, rows_per_scan=rows_per_scan)
        except:
            continue
        
        lst = gridded_data*0.00351+150.0
        lst[gridded_view>60.0]=-9999.
        #stack data
        if orbitcount==1:
            lst_stack = lst
            view_stack = gridded_view
        else:
            lst_stack = np.dstack((lst_stack,lst))
            view_stack = np.dstack((view_stack,gridded_view))

    #=========CLOUD:doing angle clearing======================================
    if cloudview_stack.ndim == 2:  
        dims = [cloudview_stack.shape[0],cloudview_stack.shape[0],1]
    else:
        dims = cloudview_stack.shape     
    aa = np.reshape(cloudview_stack,[dims[0]*dims[1],dims[2]])
    aa[np.isnan(aa)]=9999.
    indcol = np.argmin(aa,axis=1)
    indrow = range(0,len(indcol))
    bb = np.reshape(cloud_stack,[dims[0]*dims[1],dims[2]])
    cloud = bb[indrow,indcol]
    cloud = np.reshape(cloud,[3750,3750])
    #=========WATERMASK:doing angle clearing======================================
    if watermask_stack.ndim == 2:  
        dims = [watermask_stack.shape[0],watermask_stack.shape[0],1]
    else:
        dims = watermask_stack.shape     
    aa = np.reshape(watermask_stack,[dims[0]*dims[1],dims[2]])
    aa[np.isnan(aa)]=9999.
    indcol = np.argmin(aa,axis=1)
    indrow = range(0,len(indcol))
    bb = np.reshape(watermask_stack,[dims[0]*dims[1],dims[2]])
    watermask = bb[indrow,indcol]
    watermask = np.reshape(watermask,[3750,3750])
    #=========BT:doing angle and cloud clearing================================     
    aa = np.reshape(view_stack,[dims[0]*dims[1],dims[2]])
    aa[np.isnan(aa)]=9999.
    indcol = np.argmin(aa,axis=1)
    indrow = range(0,len(indcol))
    bb = np.reshape(lst_stack,[dims[0]*dims[1],dims[2]])
    lst = bb[indrow,indcol]
    lst = np.reshape(lst,[3750,3750])
    lst = np.array(lst,dtype='Float32')
#    out_bt_fn = os.path.join(tile_base_path,"bt.dat" )
    out_bt_fn = os.path.join(tile_path,"merged_day_bt_%s_T%03d_%s.dat" % (date,tile,out_time))
    lst[cloud>1]=-9999.
    lst[(watermask==1) | (watermask==2)]=np.nan
    lst.tofile(out_bt_fn)
    convertBin2tif(out_bt_fn,inUL,ALEXIshape,ALEXIres,'float32',gdal.GDT_Float32) 
    
    #=========VIEW:doing angle and cloud clearing================================     
    aa = np.reshape(view_stack,[dims[0]*dims[1],dims[2]])
    aa[np.isnan(aa)]=9999.
    indcol = np.argmin(aa,axis=1)
    indrow = range(0,len(indcol))
    bb = np.reshape(view_stack,[dims[0]*dims[1],dims[2]])
    view = bb[indrow,indcol]
    view = np.reshape(view,[3750,3750])
    view = np.array(view,dtype='Float32')
#    out_bt_fn = os.path.join(tile_base_path,"bt.dat" )
    out_view_fn = os.path.join(tile_path,"merged_day_view_%s_T%03d_%s.dat" % (date,tile,out_time))
    view[cloud>1]=-9999.
    view[(watermask==1) | (watermask==2)]=np.nan
    view.tofile(out_view_fn)
    convertBin2tif(out_view_fn,inUL,ALEXIshape,ALEXIres,'float32',gdal.GDT_Float32)
    
    #=====================Night================================================
    #==========================================================================
    files = db[(db['south']-5 <= latmid) & (db['north']+5 >= latmid) & 
               (db['west']-5 <= lonmid) & (db['east']+5 >= lonmid) & 
               (db['year'] == year) & (db['doy'] == doy) & (db['N_Day_Night_Flag'] == 'Night')]
    filenames = files['filename']

    orbits = []
    for fn in filenames:
        parts = fn.split(os.sep)[-1].split('_')
        orbits.append(parts[5])
    orbits = list(set(orbits)) 
    orbitcount = 0
    for orbit in orbits:           
        fns = [s for s in filenames if orbit in s.lower()]
        count = 0
        for filename in fns:
            count+=1
            folder = os.sep.join(filename.split(os.sep)[:-1])
            parts = filename.split(os.sep)[-1].split('_')
            search_geofile = os.path.join(folder,"*"+"_".join(("GITCO",parts[1],parts[2],parts[3],parts[4])))
            geofile = glob.glob(search_geofile+'*')[0]
            search_geofile = os.path.join(folder,"*"+"_".join(("GMTCO",parts[1],parts[2],parts[3],parts[4])))
            datet = datetime.datetime(year,dd.month, dd.day,0,0,0)
            if datet > datetime.datetime(2017,3,8,0,0,0):
               search_cloudfile = os.path.join(folder,"*"+"_".join(("VICMO",parts[1],parts[2],parts[3],parts[4]))) 
            else:
               search_cloudfile = os.path.join(folder,"*"+"_".join(("IICMO",parts[1],parts[2],parts[3],parts[4])))
            cloudgeofile = glob.glob(search_geofile+'*')[0]
            cloudfile = glob.glob(search_cloudfile+'*')[0]
            
            # load cloud data==========
            f=h5py.File(cloudfile,'r')
            g=h5py.File(cloudgeofile,'r')
            if datet > datetime.datetime(2017,3,8,0,0,0):
                data_array = f['/All_Data/VIIRS-CM-EDR_All/QF1_VIIRSCMEDR'][()]
            else:
                data_array = f['/All_Data/VIIRS-CM-IP_All/QF1_VIIRSCMIP'][()]
            lat_array = g['/All_Data/VIIRS-MOD-GEO-TC_All/Latitude'][()]
            lon_array = g['/All_Data/VIIRS-MOD-GEO-TC_All/Longitude'][()]
            view_array = g['/All_Data/VIIRS-MOD-GEO-TC_All/SatelliteZenithAngle'][()]
            
            start=filename.find('_t')
            out_time=filename[start+2:start+6]
            
            if count ==1:
                latcloud = np.array(lat_array,'float32')
                loncloud=np.array(lon_array,'float32')
                cloud=np.array(data_array,'float32')
                viewcloud=np.array(view_array,'float32')
            else:
                latcloud = np.vstack((latcloud,np.array(lat_array,'float32')))
                loncloud = np.vstack((loncloud,np.array(lon_array,'float32')))
                cloud = np.vstack((cloud,np.array(data_array,'float32')))
                viewcloud = np.vstack((viewcloud,np.array(view_array,'float32')))
                
            # load water mask===========
            f=h5py.File(cloudfile,'r')
            g=h5py.File(cloudgeofile,'r')
            if datet > datetime.datetime(2017,3,8,0,0,0):
                data_array = f['/All_Data/VIIRS-CM-EDR_All/QF2_VIIRSCMEDR'][()]
            else:
                data_array = f['/All_Data/VIIRS-CM-IP_All/QF2_VIIRSCMIP'][()]
            lat_array = g['/All_Data/VIIRS-MOD-GEO-TC_All/Latitude'][()]
            lon_array = g['/All_Data/VIIRS-MOD-GEO-TC_All/Longitude'][()]
            view_array = g['/All_Data/VIIRS-MOD-GEO-TC_All/SatelliteZenithAngle'][()]
            
            start=filename.find('_t')
            out_time=filename[start+2:start+6]
            
            if count ==1:
                watermask=np.array(data_array,'float32')
            else:
                watermask = np.vstack((watermask,np.array(data_array,'float32')))
                
            #  Load BT data=============
            f=h5py.File(filename,'r')
            g=h5py.File(geofile,'r')
            data_array = f['/All_Data/VIIRS-I5-SDR_All/BrightnessTemperature'][()]
            lat_array = g['/All_Data/VIIRS-IMG-GEO-TC_All/Latitude'][()]
            lon_array = g['/All_Data/VIIRS-IMG-GEO-TC_All/Longitude'][()]
            view_array = g['/All_Data/VIIRS-IMG-GEO-TC_All/SatelliteZenithAngle'][()]
            if count ==1:
                lat=np.array(lat_array,'float32')
                lon=np.array(lon_array,'float32')
                data=np.array(data_array,'float32')
                view=np.array(view_array,'float32')
            else:
                lat = np.vstack((lat,np.array(lat_array,'float32')))
                lon = np.vstack((lon,np.array(lon_array,'float32')))
                data = np.vstack((data,np.array(data_array,'float32')))
                view = np.vstack((view,np.array(view_array,'float32')))
        #====cloud gridding=====================
        cloudOrig=cloud.copy()
        #get 2-3 bits
        cloud=np.array(cloud,'uint8')
        cloud = np.reshape(cloud,[cloud.size, 1])
        b = np.unpackbits(cloud, axis=1)
        cloud = np.sum(b[:,4:6],axis=1)
        cloud = np.reshape(cloud,[cloudOrig.shape[0],cloudOrig.shape[1]])
        cloud = np.array(cloud, dtype='float32')
        mask = (cloudOrig==0.)
        cloud[mask]=np.nan
        viewcloud[mask]=np.nan
        #====get water mask from bits===========
        
        watermask=np.array(watermask,'uint8')
        watermask = np.reshape(watermask,[watermask.size, 1])
        b = np.unpackbits(watermask, axis=1)
        watermask = np.sum(b[:,5:7],axis=1)
        watermask = np.reshape(watermask,[cloudOrig.shape[0],cloudOrig.shape[1]])
        watermask = np.array(watermask, dtype='float32')
        
        #=====check if data is in range========================================
        rangeIndex = ((latcloud<-90.) | (latcloud > 90.) | (loncloud < -180.) | (loncloud > 180.))
        latcloud[rangeIndex] = np.nan
        loncloud[rangeIndex] = np.nan
        cloud[rangeIndex] = np.nan
        viewcloud[rangeIndex] = np.nan
        watermask[rangeIndex] = np.nan
        if np.nansum(cloud)==0: # check if there is any data
            continue

        projection = '+proj=longlat +ellps=WGS84 +datum=WGS84'
        area_id ='tile'
        proj_id = 'latlon'
        description = 'lat lon grid'

        swath_def = geometry.SwathDefinition(lons=loncloud, lats=latcloud)
        x_size = 3750
        y_size = 3750
        area_extent = (LLlon,LLlat,URlon,URlat)
        area_def = utils.get_area_def(area_id, description, proj_id, projection,
                                                   x_size, y_size, area_extent)
        swath_points_in_grid, cols, rows = ll2cr(swath_def, area_def, copy=False)
        rows_per_scan = 16
        try: # if there are no valid pixels in the region move on
            num_valid_points, gridded_cloud = fornav(cols, rows, area_def, cloud, rows_per_scan=rows_per_scan)
            orbitcount+=1
        except:
            continue
        try:
            num_valid_points, gridded_cloudview = fornav(cols, rows, area_def, viewcloud, rows_per_scan=rows_per_scan)
        except:
            continue
        
        try:
            num_valid_points, gridded_watermask = fornav(cols, rows, area_def, watermask, rows_per_scan=rows_per_scan)
        except:
            continue
        
        gridded_cloud[gridded_cloudview>60.0]=np.nan
        gridded_watermask[gridded_cloudview>60.0]=np.nan
        #stack data
        if orbitcount==1:
            cloud_stack = gridded_cloud
            cloudview_stack = gridded_cloudview
            watermask_stack = gridded_watermask
        else:
            cloud_stack = np.dstack((cloud_stack,gridded_cloud))
            cloudview_stack = np.dstack((cloudview_stack,gridded_cloudview))
            watermask_stack = np.dstack((watermask_stack,gridded_watermask))
            
        #==LST gridding===========================
        mask = (data>65527.)
        data[mask]=np.nan
        view[mask]=np.nan
        #=====check if data is in range========================================
        rangeIndex = ((lat<-90.) | (lat > 90.) | (lon < -180.) | (lon > 180.))
        lat[rangeIndex] = np.nan
        lon[rangeIndex] = np.nan
        data[rangeIndex] = np.nan
        view[rangeIndex] = np.nan
        if np.nansum(data)==0: # check if there is any data
            continue

        projection = '+proj=longlat +ellps=WGS84 +datum=WGS84'
        area_id ='tile'
        proj_id = 'latlon'
        description = 'lat lon grid'

        swath_def = geometry.SwathDefinition(lons=lon, lats=lat)
        x_size = 3750
        y_size = 3750
        area_extent = (LLlon,LLlat,URlon,URlat)
        area_def = utils.get_area_def(area_id, description, proj_id, projection,
                                                   x_size, y_size, area_extent)
        swath_points_in_grid, cols, rows = ll2cr(swath_def, area_def, copy=False)
        rows_per_scan = 32
        try: # if there are no valid pixels in the region move on
            num_valid_points, gridded_data = fornav(cols, rows, area_def, data, rows_per_scan=rows_per_scan)
        except:
            continue
        try:
            num_valid_points, gridded_view = fornav(cols, rows, area_def, view, rows_per_scan=rows_per_scan)
        except:
            continue
        
        lst = gridded_data*0.00351+150.0
        lst[gridded_view>60.0]=-9999.
        #stack data
        if orbitcount==1:
            lst_stack = lst
            view_stack = gridded_view
        else:
            lst_stack = np.dstack((lst_stack,lst))
            view_stack = np.dstack((view_stack,gridded_view))

    #=========CLOUD:doing angle clearing====================================== 
    if cloudview_stack.ndim == 2:  
        dims = [cloudview_stack.shape[0],cloudview_stack.shape[0],1]
    else:
        dims = cloudview_stack.shape
        
    aa = np.reshape(cloudview_stack,[dims[0]*dims[1],dims[2]])
    aa[np.isnan(aa)]=9999.
    indcol = np.argmin(aa,axis=1)
    indrow = range(0,len(indcol))
    bb = np.reshape(cloud_stack,[dims[0]*dims[1],dims[2]])
    cloud = bb[indrow,indcol]
    cloud = np.reshape(cloud,[3750,3750])
    #=========WATERMASK:doing angle clearing======================================
    if watermask_stack.ndim == 2:  
        dims = [watermask_stack.shape[0],watermask_stack.shape[0],1]
    else:
        dims = watermask_stack.shape     
    aa = np.reshape(watermask_stack,[dims[0]*dims[1],dims[2]])
    aa[np.isnan(aa)]=9999.
    indcol = np.argmin(aa,axis=1)
    indrow = range(0,len(indcol))
    bb = np.reshape(watermask_stack,[dims[0]*dims[1],dims[2]])
    watermask = bb[indrow,indcol]
    watermask = np.reshape(watermask,[3750,3750])
    #=========BT:doing angle and cloud clearing================================     
    aa = np.reshape(view_stack,[dims[0]*dims[1],dims[2]])
    aa[np.isnan(aa)]=9999.
    indcol = np.argmin(aa,axis=1)
    indrow = range(0,len(indcol))
    bb = np.reshape(lst_stack,[dims[0]*dims[1],dims[2]])
    lst = bb[indrow,indcol]
    lst = np.reshape(lst,[3750,3750])
    lst = np.array(lst,dtype='Float32')
#    out_bt_fn = os.path.join(tile_base_path,"bt.dat" )
    out_bt_fn = os.path.join(tile_path,"merged_night_bt_%s_T%03d_%s.dat" % (date,tile,out_time))
    lst[cloud>1]=-9999.
    lst[(watermask==1) | (watermask==2)]=-9999.
    lst.tofile(out_bt_fn)
    convertBin2tif(out_bt_fn,inUL,ALEXIshape,ALEXIres,'float32',gdal.GDT_Float32)
    
    #=========VIEW:doing angle and cloud clearing================================     
    aa = np.reshape(view_stack,[dims[0]*dims[1],dims[2]])
    aa[np.isnan(aa)]=9999.
    indcol = np.argmin(aa,axis=1)
    indrow = range(0,len(indcol))
    bb = np.reshape(view_stack,[dims[0]*dims[1],dims[2]])
    view = bb[indrow,indcol]
    view = np.reshape(view,[3750,3750])
    view = np.array(view,dtype='Float32')
#    out_bt_fn = os.path.join(tile_base_path,"bt.dat" )
    out_view_fn = os.path.join(tile_path,"merged_night_view_%s_T%03d_%s.dat" % (date,tile,out_time))
    view[cloud>1]=np.nan
    view[(watermask==1) | (watermask==2)]=-9999.
    view.tofile(out_view_fn)
    convertBin2tif(out_view_fn,inUL,ALEXIshape,ALEXIres,'float32',gdal.GDT_Float32)
    
    
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
    
    
    #=====================Day==================================================
    #==========================================================================
    files = db[(db['south']-5 <= latmid) & (db['north']+5 >= latmid) & 
               (db['west']-5 <= lonmid) & (db['east']+5 >= lonmid) & 
               (db['year'] == year) & (db['doy'] == doy) & (db['N_Day_Night_Flag'] == 'Day')]
    filenames = files['filename']
    
    mergedata =np.array([])
    mergelat = np.array([])
    mergelon = np.array([])
    mergeview = np.array([])
    mergecloudlat = np.array([])
    mergecloudlon = np.array([])
    mergecloud = np.array([])
    for i in range(len(filenames)):    
        filename = filenames.iloc[i]
        folder = os.sep.join(filename.split(os.sep)[:-1])
        parts = filename.split(os.sep)[-1].split('_')
        search_geofile = os.path.join(folder,"*"+"_".join(("GITCO",parts[1],parts[2],parts[3],parts[4])))
        geofile = glob.glob(search_geofile+'*')[0]
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
        view_array = g['/All_Data/VIIRS-MOD-GEO-TC_All/SatelliteZenithAngle'][()]
        
        latcloud=np.array(lat_array,'float32')
        loncloud=np.array(lon_array,'float32')
        cloud=np.array(data_array,'uint8')
        viewcloud=np.array(view_array,'float32')
        
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
        cloudlats = latcloud[np.where((latcloud>LLlat) & (latcloud <=URlat) & (loncloud>LLlon) & (loncloud<=URlon)
        & (abs(viewcloud)<60.0))]
        cloudlons = loncloud[np.where((latcloud>LLlat) & (latcloud <=URlat) & (loncloud>LLlon) & (loncloud<=URlon)
        & (abs(viewcloud)<60.0))]
        clouds = cloud[np.where((latcloud>LLlat) & (latcloud <=URlat) & (loncloud>LLlon) & (loncloud<=URlon)
        & (abs(viewcloud)<60.0))]
        mergedata = np.append(mergedata,vals)
        mergelat = np.append(mergelat,lats)
        mergelon = np.append(mergelon,lons)
        mergeview = np.append(mergeview,views)
        mergecloudlat = np.append(mergecloudlat,cloudlats)
        mergecloudlon = np.append(mergecloudlon,cloudlons)
        mergecloud = np.append(mergecloud,clouds)
    res=0
    if mergelat.any():
        mergecloudOrig=mergecloud.copy()
        #get 2-3 bits
        mergecloud=np.array(mergecloud,'uint8')
        mergecloud = np.reshape(mergecloud,[mergecloud.size, 1])
        b = np.unpackbits(mergecloud, axis=1)
        mergecloud = np.sum(b[:,4:6],axis=1)
        
        mergelat = ma.array(mergelat, mask = (mergedata>65527.),copy=False)
        mergelon = ma.array(mergelon, mask = (mergedata>65527.),copy=False)
        mergedata = ma.array(mergedata, mask = (mergedata>65527.),copy=False)
        mergecloudlat = ma.array(mergecloudlat, mask = (mergecloudOrig==0.),copy=False)
        mergecloudlon = ma.array(mergecloudlon, mask = (mergecloudOrig==0.),copy=False)
        mergecloud = ma.array(mergecloud, mask = (mergecloudOrig==0.),copy=False)

        
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
#        swath_con = image.ImageContainerNearest(mergecloud, swath_def, radius_of_influence=5000)
#        area_con = swath_con.resample(area_def)
#        cloud = area_con.image_data
        cloud = kd_tree.resample_nearest(swath_def, mergecloud.ravel(),area_def, radius_of_influence=500, fill_value=None)
        inProjection = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'
        outfn = os.path.join(tile_path,"cloud_day.tif")
        writeArray2Tiff(cloud,ALEXIres,inUL,inProjection,outfn,gdal.GDT_Float32)
        swath_def = geometry.SwathDefinition(lons=mergelon, lats=mergelat)
#        swath_con = image.ImageContainerNearest(mergedata, swath_def, radius_of_influence=5000)
#        area_con = swath_con.resample(area_def)
#        lst = area_con.image_data*0.00351+150.0
        gridded = kd_tree.resample_nearest(swath_def, mergedata.ravel(),area_def, radius_of_influence=500, fill_value=None)
        lst = gridded*0.00351+150.0
        lst[lst==150]=-9999.
        lst[cloud>1]=-9999.
        lst=np.array(lst,'float32')
        out_bt_fn = os.path.join(tile_path,"merged_day_bt_%s_T%03d_%s.dat" % (date,tile,out_time))
        lst.tofile(out_bt_fn)
        convertBin2tif(out_bt_fn,inUL,ALEXIshape,ALEXIres,'float32',gdal.GDT_Float32) 
        
#        swath_con = image.ImageContainerNearest(mergeview, swath_def, radius_of_influence=5000)
#        area_con = swath_con.resample(area_def)
#        view = area_con.image_data
        view = kd_tree.resample_nearest(swath_def, mergeview.ravel(),area_def, radius_of_influence=500, fill_value=None)
        out_view_fn =os.path.join(tile_path,"merged_day_view_%s_T%03d_%s.dat" % (date,tile,out_time))
        view[view==0]=-9999.
        view[cloud>1]=-9999.
        view=np.array(view,'float32')
        view.tofile(out_view_fn)
    else:
        res+=1

    #================Night=====================================================
    #==========================================================================
    files = db[(db['south']-5 <= latmid) & (db['north']+5 >= latmid) & 
               (db['west']-5 <= lonmid) & (db['east']+5 >= lonmid) & 
               (db['year'] == year) & (db['doy'] == doy) & (db['N_Day_Night_Flag'] == 'Night')]
    filenames = files['filename']
    
    mergedata =np.array([])
    mergelat = np.array([])
    mergelon = np.array([])
    mergeview = np.array([])
    mergecloudlat = np.array([])
    mergecloudlon = np.array([])
    mergecloud = np.array([])
    for i in range(len(filenames)):    
        filename = filenames.iloc[i]
        folder = os.sep.join(filename.split(os.sep)[:-1])
        parts = filename.split(os.sep)[-1].split('_')
        search_geofile = os.path.join(folder,"*"+"_".join(("GITCO",parts[1],parts[2],parts[3],parts[4])))
        geofile = glob.glob(search_geofile+'*')[0]
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
        view_array = g['/All_Data/VIIRS-MOD-GEO-TC_All/SatelliteZenithAngle'][()]
        
        latcloud=np.array(lat_array,'float32')
        loncloud=np.array(lon_array,'float32')
        cloud=np.array(data_array,'uint8')
        viewcloud=np.array(view_array,'float32')
        
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
        cloudlats = latcloud[np.where((latcloud>LLlat) & (latcloud <=URlat) & (loncloud>LLlon) & (loncloud<=URlon)
        & (abs(viewcloud)<60.0))]
        cloudlons = loncloud[np.where((latcloud>LLlat) & (latcloud <=URlat) & (loncloud>LLlon) & (loncloud<=URlon)
        & (abs(viewcloud)<60.0))]
        clouds = cloud[np.where((latcloud>LLlat) & (latcloud <=URlat) & (loncloud>LLlon) & (loncloud<=URlon)
        & (abs(viewcloud)<60.0))]
        
        mergedata = np.append(mergedata,vals)
        mergelat = np.append(mergelat,lats)
        mergelon = np.append(mergelon,lons)
        mergeview = np.append(mergeview,views)
        mergecloudlat = np.append(mergecloudlat,cloudlats)
        mergecloudlon = np.append(mergecloudlon,cloudlons)
        mergecloud = np.append(mergecloud,clouds)

    if mergelat.any(): 
        mergecloudOrig=mergecloud.copy()
        #get 2-3 bits
        mergecloud=np.array(mergecloud,'uint8')
        mergecloud = np.reshape(mergecloud,[mergecloud.size, 1])
        b = np.unpackbits(mergecloud, axis=1)
        mergecloud = np.sum(b[:,4:6],axis=1)

        mergelat = ma.array(mergelat, mask = (mergedata>65527.),copy=False)
        mergelon = ma.array(mergelon, mask = (mergedata>65527.),copy=False)
        mergedata = ma.array(mergedata, mask = (mergedata>65527.),copy=False)
        mergecloudlat = ma.array(mergecloudlat, mask = (mergecloudOrig==0.),copy=False)
        mergecloudlon = ma.array(mergecloudlon, mask = (mergecloudOrig==0.),copy=False)
        mergecloud = ma.array(mergecloud, mask = (mergecloudOrig==0.),copy=False)

        
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
#        swath_con = image.ImageContainerNearest(mergecloud, swath_def, radius_of_influence=5000)
#        area_con = swath_con.resample(area_def)
#        cloud = area_con.image_data
        cloud = kd_tree.resample_nearest(swath_def, mergecloud.ravel(),area_def, radius_of_influence=500, fill_value=None)
        
        
        swath_def = geometry.SwathDefinition(lons=mergelon, lats=mergelat)
#        swath_con = image.ImageContainerNearest(mergedata, swath_def, radius_of_influence=5000)
#        area_con = swath_con.resample(area_def)
#        lst = area_con.image_data*0.00351+150.0
        gridded = kd_tree.resample_nearest(swath_def, mergedata.ravel(),area_def, radius_of_influence=500, fill_value=None)
        lst = gridded*0.00351+150.0
        lst[lst==150]=-9999.
        lst[cloud>1]=-9999.
        lst=np.array(lst,'float32')
        out_bt_fn = os.path.join(tile_path,"merged_night_bt_%s_T%03d_%s.dat" % (date,tile,out_time))
        lst.tofile(out_bt_fn)
        convertBin2tif(out_bt_fn,inUL,ALEXIshape,ALEXIres,'float32',gdal.GDT_Float32) 
        
        swath_def = geometry.SwathDefinition(lons=mergelon, lats=mergelat)
#        swath_con = image.ImageContainerNearest(mergeview, swath_def, radius_of_influence=5000)
#        area_con = swath_con.resample(area_def)
#        view = area_con.image_data
        
        view = kd_tree.resample_nearest(swath_def, mergeview.ravel(),area_def, radius_of_influence=500, fill_value=None)

        out_view_fn =os.path.join(tile_path,"merged_night_view_%s_T%03d_%s.dat" % (date,tile,out_time))
        view[view==0]=-9999.
        view[cloud>1]=-9999.
        view=np.array(view,'float32')
        view.tofile(out_view_fn)
    else:
        res+=1
        
    return res
        
def atmosCorrection(tile,year,doy):
    LLlat,LLlon = tile2latlon(tile)
    URlat = LLlat+15.
    inUL = [LLlon,URlat]
    ALEXIshape = [3750,3750]
    ALEXIres = [0.004,0.004]
    day_minus_coeff = [0.0504,1.384,2.415,3.586,4.475,4.455]
    day_minus_b=[-0.023,0.003,0.088,0.221,0.397,0.606]

    
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
    ztime_fn = os.path.join(overpass_corr_path,"CURRENT_DAY_ZTIME_T%03d.dat" % tile)
    gunzip(os.path.join(overpass_corr_cache,"DAY_ZTIME_T%03d.dat.gz" % tile),
       out_fn=ztime_fn)
    ztime= np.fromfile(ztime_fn, dtype=np.float32)
    dztime= ztime.reshape([3750,3750])
    convertBin2tif(ztime_fn,inUL,ALEXIshape,ALEXIres,'float32',gdal.GDT_Float32)
    dtrad_cache = os.path.join(static_path,"dtrad_avg")
    dtrad_fn =os.path.join(overpass_corr_path,"CURRENT_DTRAD_AVG_T%03d.dat" % tile)
    gunzip(os.path.join(dtrad_cache,"DTRAD_T%03d_%d.dat.gz" % (tile,avgddd)),
       out_fn=dtrad_fn)
    dtrad= np.fromfile(dtrad_fn, dtype=np.float32)
    dtrad= np.flipud(dtrad.reshape([3750,3750]))
#    dtrad.tofile(dtrad_fn)
    convertBin2tif(dtrad_fn,inUL,ALEXIshape,ALEXIres,'float32',gdal.GDT_Float32)
    tile_path = os.path.join(tile_base_path,"T%03d" % tile) 
    tile_lut_path = os.path.join(CFSR_path,"viirs_tile_lookup_tables")
    #=========================Day==============================================
    #==========================================================================
    out_bt_fn = glob.glob(os.path.join(tile_path,"merged_day_bt_%s_T%03d*.dat" % (date,tile)))[0]
    out_view_fn1 = glob.glob(os.path.join(tile_path,"merged_day_view_%s_T%03d*.dat" % (date,tile)))[0]
    time_str = out_bt_fn.split(os.sep)[-1].split("_")[5].split(".")[0]
    grab_time = getGrabTime(int(time_str))
    #===========use forecast hour==============================================
    if (grab_time)==2400:
        time = 0000
    else:
        time = grab_time
    hr,forcastHR,cfsr_doy = getGrabTimeInv(grab_time/100,doy)
    cfsr_date = "%d%03d" % (year,cfsr_doy)
    cfsr_tile_path = os.path.join(CFSR_path,"%d" % year,"%03d" % cfsr_doy)

    #==================io filenames============================================
    tprof = os.path.join(cfsr_tile_path,"temp_profile_%s_%04d.dat" % (cfsr_date,time))
    qprof = os.path.join(cfsr_tile_path,"spfh_profile_%s_%04d.dat" % (cfsr_date,time))
    tsfcfile = os.path.join(cfsr_tile_path,"sfc_temp_%s_%04d.dat" % (cfsr_date,time))
    presfile = os.path.join(cfsr_tile_path,"sfc_pres_%s_%04d.dat" % (cfsr_date,time))
    qsfcfile = os.path.join(cfsr_tile_path,"sfc_spfh_%s_%04d.dat" % (cfsr_date,time))
    icoordpath = os.path.join(tile_lut_path,"CFSR_T%03d_lookup_icoord.dat" % tile)
    jcoordpath = os.path.join(tile_lut_path,"CFSR_T%03d_lookup_jcoord.dat" % tile)
    raw_trad_fn = os.path.join(overpass_corr_path,"RAW_TRAD1_T%03d.dat" % tile)
    trad_fn = os.path.join(overpass_corr_path,"TRAD1_T%03d.dat" % tile)
    out_view_fn = os.path.join(overpass_corr_path,"VIEW_ANGLE_T%03d.dat" % tile)
    #==============preparing data==============================================
    shutil.copyfile(out_bt_fn,raw_trad_fn)
    shutil.copyfile(out_view_fn1,out_view_fn)
    day_lst = np.fromfile(raw_trad_fn, dtype=np.float32)
    day_lst= day_lst.reshape([3750,3750])
    view = np.fromfile(out_view_fn1, dtype=np.float32)
    view = np.flipud(view.reshape([3750,3750]))
    view.tofile(out_view_fn)
#    bt.tofile(raw_trad_fn)
#    view = np.fromfile(out_view_fn, dtype=np.float32)
#    view= np.flipud(view.reshape([3750,3750]))
#    view.tofile(out_view_fn)
    #=====get the offset for day FORTRAN VERSION===============================
#    subprocess.check_output(["%s" % offset, "%d" % year, "%03d" %  doy, "%s" % time_str,
#                                 "T%03d" % tile, "%s" % ztime_fn, "%s" % raw_trad_fn,
#                                 "%s" % dtrad_fn, "%s" % trad_fn])
    
    ###=====python version=====================================================
    ctime = grab_time/100.
    tdiff_day=abs(ctime-dztime)
    tindex1=np.array(tdiff_day, dtype=int)
    tindex2=tindex1+1
    tindex1[np.where((day_lst==-9999.) | (dtrad==-9999.))]=0
    tindex2[np.where((day_lst==-9999.) | (dtrad==-9999.))]=0
    w2=(tdiff_day-tindex1)
    w1=(1.0-w2)
    c1 = np.empty([3750,3750])
    c2 = np.empty([3750,3750])
    day_corr = np.empty([3750,3750])
    for i in range(len(day_minus_coeff)-1):
        c1[np.where(tindex1==i)]=day_minus_coeff[i]+(day_minus_b[i]*dtrad[np.where(tindex1==i)])
        c2[np.where(tindex2==i+1)]=day_minus_coeff[i+1]+(day_minus_b[i+1]*dtrad[np.where(tindex2==i+1)])
        day_corr[np.where(tindex1==i)] = day_lst[np.where(tindex1==i)]+(c1[np.where(tindex1==i)]*w1[np.where(tindex1==i)]+c2[np.where(tindex1==i)]*w2[np.where(tindex1==i)])
    day_corr[np.where(dtrad==-9999.)]=-9999.
    day_corr = np.array(np.flipud(day_corr),dtype='Float32')
    day_corr.tofile(trad_fn)
    convertBin2tif(trad_fn,inUL,ALEXIshape,ALEXIres,'float32',gdal.GDT_Float32)
    
    #=======run atmospheric correction=========================================
    outfn = os.path.join(tile_path,"FINAL_DAY_LST_%s_T%03d.dat" % (date,tile))
    out = subprocess.check_output(["%s" % run_correction,"%s" % tprof, 
                                   "%s" % qprof,"%s" % tsfcfile,
                                   "%s" % presfile, "%s" % qsfcfile,
                                   "%s" % icoordpath, "%s" % jcoordpath,
                                   "%s" % trad_fn,"%s" % out_view_fn, "%s" % outfn])
#    print out
    convertBin2tif(outfn,inUL,ALEXIshape,ALEXIres,'float32',gdal.GDT_Float32)
    
    
    #======Night===============================================================
    #==========================================================================
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
    
    #======io filenames========================================================
    tprof = os.path.join(cfsr_tile_path,"temp_profile_%s_%04d.dat" % (cfsr_date,time))
    qprof = os.path.join(cfsr_tile_path,"spfh_profile_%s_%04d.dat" % (cfsr_date,time))
    tsfcfile = os.path.join(cfsr_tile_path,"sfc_temp_%s_%04d.dat" % (cfsr_date,time))
    presfile = os.path.join(cfsr_tile_path,"sfc_pres_%s_%04d.dat" % (cfsr_date,time))
    qsfcfile = os.path.join(cfsr_tile_path,"sfc_spfh_%s_%04d.dat" % (cfsr_date,time))
    icoordpath = os.path.join(tile_lut_path,"CFSR_T%03d_lookup_icoord.dat" % tile)
    jcoordpath = os.path.join(tile_lut_path,"CFSR_T%03d_lookup_jcoord.dat" % tile)
    trad_fn = os.path.join(overpass_corr_path,"TRAD1_T%03d.dat" % tile)
    out_view_fn = os.path.join(overpass_corr_path,"VIEW_ANGLE_T%03d.dat" % tile)
    
    #=======run atmospheric correction=========================================
#    shutil.copyfile(out_bt_fn,trad_fn)
    bt = np.fromfile(out_bt_fn, dtype=np.float32)
    bt= np.flipud(bt.reshape([3750,3750]))
    bt.tofile(trad_fn)
#    shutil.copyfile(out_view_fn1,out_view_fn)
    view = np.fromfile(out_view_fn1, dtype=np.float32)
    view = np.flipud(view.reshape([3750,3750]))
    view.tofile(out_view_fn)
    outfn = os.path.join(tile_path,"lst_%s_T%03d_%s.dat" % (date,tile,time_str))
    outfn = os.path.join(tile_path,"FINAL_NIGHT_LST_%s_T%03d.dat" % (date,tile))
    out = subprocess.check_output(["%s" % run_correction,"%s" % tprof, 
                                   "%s" % qprof,"%s" % tsfcfile,
                                   "%s" % presfile, "%s" % qsfcfile,
                                   "%s" % icoordpath, "%s" % jcoordpath,
                                   "%s" % trad_fn,"%s" % out_view_fn, "%s" % outfn])

    convertBin2tif(outfn,inUL,ALEXIshape,ALEXIres,'float32',gdal.GDT_Float32)
    
def atmosCorrectPython(tile,year,doy):
    tile_path = os.path.join(tile_base_path,"T%03d" % tile) 
    LLlat,LLlon = tile2latlon(tile)
    URlat = LLlat+15.
    inUL = [LLlon,URlat]
    ALEXIshape = [3750,3750]
    ALEXIres = [0.004,0.004]
    day_minus_coeff = [0.0504,1.384,2.415,3.586,4.475,4.455]
    day_minus_b=[-0.023,0.003,0.088,0.221,0.397,0.606]

    
    #====get week date=====
    nweek=(doy-1)/7
    cday=nweek*7
    offset=(doy-cday)/7
    rday=((offset+nweek)*7)+1
    avgddd=2014*1000+rday 
    date = "%d%03d" % (year,doy)
    #=========================
    istart = abs(-89.875+URlat)*4
    addArray = np.floor(np.array(range(3750))*0.004/0.25)
    icor = istart+addArray
    icormat = np.repeat(np.reshape(icor,[icor.size,1]),3750,axis=1)
    icormat = icormat.T
    icormat = np.array(icormat,dtype='int32')
    icormat = np.reshape(icormat,[3750*3750,1])
     
    
    jstart = (180+LLlon)*4
    jcor = jstart+addArray
    jcormat = np.repeat(np.reshape(jcor,[jcor.size,1]),3750,axis=1)
    jcormat = np.array(jcormat,dtype='int32')
    jcormat = np.reshape(jcormat,[3750*3750,1])
    
    #=========================Day==============================================
    #==========================================================================
    out_bt_fn = glob.glob(os.path.join(tile_path,"merged_day_bt_%s_T%03d*.dat" % (date,tile)))[0]
    out_view_fn1 = glob.glob(os.path.join(tile_path,"merged_day_view_%s_T%03d*.dat" % (date,tile)))[0]
    time_str = out_bt_fn.split(os.sep)[-1].split("_")[5].split(".")[0]
    grab_time = getGrabTime(int(time_str))
    #===========use forecast hour==============================================
    if (grab_time)==2400:
        time = 0000
    else:
        time = grab_time
    hr,forcastHR,cfsr_doy = getGrabTimeInv(grab_time/100,doy)
    cfsr_date = "%d%03d" % (year,cfsr_doy)
    cfsr_tile_path = os.path.join(CFSR_path,"%d" % year,"%03d" % cfsr_doy)
    
    temp_prof_fn = os.path.join(cfsr_tile_path,"temp_profile_%s_%04d.dat" % (cfsr_date,time))
    spfh_prof_fn = os.path.join(cfsr_tile_path,"spfh_profile_%s_%04d.dat" % (cfsr_date,time))
    sfc_temp_fn  = os.path.join(cfsr_tile_path,"sfc_temp_%s_%04d.dat" % (cfsr_date,time))
    sfc_pres_fn = os.path.join(cfsr_tile_path,"sfc_pres_%s_%04d.dat" % (cfsr_date,time))
    sfc_spfh_fn = os.path.join(cfsr_tile_path,"sfc_spfh_%s_%04d.dat" % (cfsr_date,time))
    overpass_corr_cache = os.path.join(static_path,"OVERPASS_OFFSET_CORRECTION")
    ztime_fn = os.path.join(overpass_corr_path,"CURRENT_DAY_ZTIME_T%03d.dat" % tile)
    gunzip(os.path.join(overpass_corr_cache,"DAY_ZTIME_T%03d.dat.gz" % tile),
       out_fn=ztime_fn)
    ztime= np.fromfile(ztime_fn, dtype=np.float32)
    dztime= ztime.reshape([3750,3750])
    dtrad_cache = os.path.join(static_path,"dtrad_avg")
    dtrad_fn =os.path.join(overpass_corr_path,"CURRENT_DTRAD_AVG_T%03d.dat" % tile)
    gunzip(os.path.join(dtrad_cache,"DTRAD_T%03d_%d.dat.gz" % (tile,avgddd)),
       out_fn=dtrad_fn)
    dtrad= np.fromfile(dtrad_fn, dtype=np.float32)
    dtrad= np.flipud(dtrad.reshape([3750,3750]))
    convertBin2tif(ztime_fn,inUL,ALEXIshape,ALEXIres,'float32',gdal.GDT_Float32)
    convertBin2tif(dtrad_fn,inUL,ALEXIshape,ALEXIres,'float32',gdal.GDT_Float32)

    #==============preparing data==============================================
    day_lst = np.fromfile(out_bt_fn, dtype=np.float32)
    day_lst= day_lst.reshape([3750,3750])
    
    ###=====python version=====================================================
    ctime = int(time_str)/100.
    tdiff_day=abs(ctime-dztime)
    tindex1=np.array(tdiff_day, dtype=int)
    tindex2=tindex1+1
    tindex1[np.where((day_lst==-9999.) | (dtrad==-9999.))]=0
    tindex2[np.where((day_lst==-9999.) | (dtrad==-9999.))]=0
    w2=(tdiff_day-tindex1)
    w1=(1.0-w2)
    c1 = np.empty([3750,3750])
    c2 = np.empty([3750,3750])
    day_corr = np.empty([3750,3750])
    for i in range(1,len(day_minus_coeff)-1):
        c1[np.where(tindex1==i)]=day_minus_coeff[i]+(day_minus_b[i]*dtrad[np.where(tindex1==i)])
        c2[np.where(tindex2==i+1)]=day_minus_coeff[i+1]+(day_minus_b[i+1]*dtrad[np.where(tindex2==i+1)])
        day_corr[np.where(tindex1==i)] = day_lst[np.where(tindex1==i)]+(c1[np.where(tindex1==i)]*w1[np.where(tindex1==i)]+c2[np.where(tindex1==i)]*w2[np.where(tindex1==i)])
    day_corr[np.where(tindex1<1)] = day_lst[np.where(tindex1<1)]+c2[np.where(tindex1<1)]*w2[np.where(tindex1<1)]
    day_corr[np.where(dtrad==-9999.)]=-9999.
    day_corr = np.array(np.flipud(day_corr),dtype='Float32')
    inProjection = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'
    outfn = os.path.join(tile_path,"tindex1.tif")
    writeArray2Tiff(tindex1,ALEXIres,inUL,inProjection,outfn,gdal.GDT_Float32)

    
    #=======run atmospheric correction=========================================
    
    view = np.fromfile(out_view_fn1, dtype=np.float32)
    view = view.reshape([3750,3750])
    
    spfh_prof = np.fromfile(spfh_prof_fn, dtype=np.float32)
    spfh_prof = spfh_prof.reshape([21,720,1440])
    
    temp_prof = np.fromfile(temp_prof_fn, dtype=np.float32)
    temp_prof = temp_prof.reshape([21,720,1440])
    
    temp_prof1 = np.empty([21,720,1440])
    for i in range(21):
        temp_prof1[i,:,:] = np.flipud(np.squeeze(temp_prof[i,:,:]))
        
    spfh_prof1 = np.empty([21,720,1440])
    for i in range(21):
        spfh_prof1[i,:,:] = np.flipud(np.squeeze(spfh_prof[i,:,:]))
    trad = day_corr
    trad = day_lst
    trad = np.reshape(trad,[3750*3750,1])
    
    sfc_temp = np.fromfile(sfc_temp_fn, dtype=np.float32)
    sfc_temp = np.flipud(sfc_temp.reshape([720,1440]))   
    sfc_pres = np.fromfile(sfc_pres_fn, dtype=np.float32)
    sfc_pres = np.flipud(sfc_pres.reshape([720,1440]))    
    sfc_spfh = np.fromfile(sfc_spfh_fn, dtype=np.float32)
    sfc_spfh = np.flipud(sfc_spfh.reshape([720,1440]))
    sfc_temp = np.reshape(sfc_temp,[720*1440,1])
    sfc_pres = np.reshape(sfc_pres,[720*1440,1])
    sfc_spfh = np.reshape(sfc_spfh,[720*1440,1])    
    temp_prof = np.reshape(temp_prof1,[21,720*1440]).T
    spfh_prof = np.reshape(spfh_prof1,[21,720*1440]).T    
    view1 = np.reshape(view,[3750*3750,1])
    
    pres = np.array([1000,975,950,925,900,850,800,750,700,650,600,550,500,450,400,350,300,250,200,150,100])
    ta=temp_prof/(1000/pres)**0.286
    ei=spfh_prof*temp_prof/(.378*spfh_prof+.622)
    anv=873.6
    epsln=0.98
    emis = np.empty([720*1440,21])
    tau = np.empty([720*1440,21])
    for i in range(20):
        emis[:,i]=0.5*(planck(ta[:,i],anv)+planck(ta[:,i+1],anv))
        tau[:,i]=(pres[i]-pres[i+1])*(dtaudp(anv,ta[:,i],emis[:,i],pres[i])+
           dtaudp(anv,ta[:,i+1],emis[:,i+1],pres[i+1]) +4.*
                  dtaudp(anv,(ta[:,i]+ta[:,i+1])/2.,(emis[:,i]+emis[:,i+1])/2.,
                                    (pres[i]-pres[i+1])/2.))/6
    
    optd = np.sum(tau,axis=1)
    cs=np.cos(np.deg2rad(view1)/np.deg2rad(57.29))
    optd = np.array(optd,dtype=np.float32)
    optd = np.reshape(optd,[720,1440])
    optd = optd[icormat,jcormat]
    a = -optd/cs
    trans=np.exp(a)
        
    #=========angular invariant sky================
    cs=np.cos(np.deg2rad(0.)/np.deg2rad(57.29))
    a = -tau[:,0]/cs
    sky1 = np.empty([720*1440,21])
    sky1[:,1]=emis[:,0]*(1.0-np.exp(a))
    
    for i in range(20):
        sky1[:,i]=emis[:,i-1]+(sky1[:,i-1]-emis[:,i-1])*np.exp(-tau[:,i-1]/cs)
    
    sky1 = np.reshape(sky1,[21,720,1440])
    sky1 = np.squeeze(sky1[:,icormat,jcormat]).T
    sky1 = np.reshape(sky1,[3750*3750,21])
    #====final results=============================
    grndrad1=(planck(trad[:,0],anv)-sky1[:,20]*(1.0+trans[:,0]*(1.0-epsln)))
    
    trad11=invplanck(grndrad1/epsln,anv)
    trad11 = np.reshape(trad11,[3750,3750])
    trad11[trad11<0]=-9999.
    
    outfn = os.path.join(tile_path,"FINAL_DAY_LST_%s_T%03d.dat" % (date,tile))
    trad11 = np.array(trad11, dtype=np.float32)
    trad11.tofile(outfn)
    
    #======Night===============================================================
    #==========================================================================
    out_bt_fn = glob.glob(os.path.join(tile_path,"merged_night_bt_%s_T%03d*.dat" % (date,tile)))[0]
    out_view_fn1 = glob.glob(os.path.join(tile_path,"merged_night_view_%s_T%03d*.dat" % (date,tile)))[0]
  
    #=======run atmospheric correction=========================================
#    shutil.copyfile(out_bt_fn,trad_fn)
    bt = np.fromfile(out_bt_fn, dtype=np.float32)
    trad = bt.reshape([3750,3750])
    trad = np.reshape(trad,[3750*3750,1])


#=======run atmospheric correction=========================================
    
    view = np.fromfile(out_view_fn1, dtype=np.float32)
    view = view.reshape([3750,3750])
    
    spfh_prof = np.fromfile(spfh_prof_fn, dtype=np.float32)
    spfh_prof = spfh_prof.reshape([21,720,1440])
    
    temp_prof = np.fromfile(temp_prof_fn, dtype=np.float32)
    temp_prof = temp_prof.reshape([21,720,1440])
    
    temp_prof1 = np.empty([21,720,1440])
    for i in range(21):
        temp_prof1[i,:,:] = np.flipud(np.squeeze(temp_prof[i,:,:]))
        
    spfh_prof1 = np.empty([21,720,1440])
    for i in range(21):
        spfh_prof1[i,:,:] = np.flipud(np.squeeze(spfh_prof[i,:,:]))
    
    sfc_temp = np.fromfile(sfc_temp_fn, dtype=np.float32)
    sfc_temp = np.flipud(sfc_temp.reshape([720,1440]))   
    sfc_pres = np.fromfile(sfc_pres_fn, dtype=np.float32)
    sfc_pres = np.flipud(sfc_pres.reshape([720,1440]))    
    sfc_spfh = np.fromfile(sfc_spfh_fn, dtype=np.float32)
    sfc_spfh = np.flipud(sfc_spfh.reshape([720,1440]))
    sfc_temp = np.reshape(sfc_temp,[720*1440,1])
    sfc_pres = np.reshape(sfc_pres,[720*1440,1])
    sfc_spfh = np.reshape(sfc_spfh,[720*1440,1])    
    temp_prof = np.reshape(temp_prof1,[21,720*1440]).T
    spfh_prof = np.reshape(spfh_prof1,[21,720*1440]).T    
    view1 = np.reshape(view,[3750*3750,1])
    
    pres = np.array([1000,975,950,925,900,850,800,750,700,650,600,550,500,450,400,350,300,250,200,150,100])
    ta=temp_prof/(1000/pres)**0.286
    ei=spfh_prof*temp_prof/(.378*spfh_prof+.622)
    anv=873.6
    epsln=0.98
    emis = np.empty([720*1440,21])
    tau = np.empty([720*1440,21])
    for i in range(20):
        emis[:,i]=0.5*(planck(ta[:,i],anv)+planck(ta[:,i+1],anv))
        tau[:,i]=(pres[i]-pres[i+1])*(dtaudp(anv,ta[:,i],emis[:,i],pres[i])+
           dtaudp(anv,ta[:,i+1],emis[:,i+1],pres[i+1]) +4.*
                  dtaudp(anv,(ta[:,i]+ta[:,i+1])/2.,(emis[:,i]+emis[:,i+1])/2.,
                                    (pres[i]-pres[i+1])/2.))/6
    
    optd = np.sum(tau,axis=1)
    cs=np.cos(np.deg2rad(view1)/np.deg2rad(57.29))
    optd = np.array(optd,dtype=np.float32)
    optd = np.reshape(optd,[720,1440])
    optd = optd[icormat,jcormat]
    a = -optd/cs
    trans=np.exp(a)
        
    #=========angular invariant sky================
    cs=np.cos(np.deg2rad(0.)/np.deg2rad(57.29))
    a = -tau[:,0]/cs
    sky1 = np.empty([720*1440,21])
    sky1[:,1]=emis[:,0]*(1.0-np.exp(a))
    
    for i in range(20):
        sky1[:,i]=emis[:,i-1]+(sky1[:,i-1]-emis[:,i-1])*np.exp(-tau[:,i-1]/cs)
    
    sky1 = np.reshape(sky1,[21,720,1440])
    sky1 = np.squeeze(sky1[:,icormat,jcormat]).T
    sky1 = np.reshape(sky1,[3750*3750,21])
    #====final results=============================
    grndrad1=(planck(trad[:,0],anv)-sky1[:,20]*(1.0+trans[:,0]*(1.0-epsln)))
    
    trad11=invplanck(grndrad1/epsln,anv)
    trad11 = np.reshape(trad11,[3750,3750])
    trad11[trad11<0]=-9999.
    
    outfn = os.path.join(tile_path,"FINAL_NIGHT_LST_%s_T%03d.dat" % (date,tile))
    trad11 = np.array(trad11, dtype=np.float32)
    trad11.tofile(outfn)

    convertBin2tif(outfn,inUL,ALEXIshape,ALEXIres,'float32',gdal.GDT_Float32)
    
    return trad11

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
#    dtrad_path = os.path.join(processing_path,'DTRAD_PREDICTION')
#    if not os.path.exists(dtrad_path):
#        os.makedirs(dtrad_path) 
    
    date = "%d%03d" % (year,doy)
            
    dtimedates = np.array(range(1,366,7))
    rday = dtimedates[dtimedates>=doy][0]

    risedoy = rday

    laidates = np.array(range(1,366,4))
    rday = laidates[laidates>=doy][0]
    laiddd="%d%03d" %(year,rday)

    precip_fn = os.path.join(base,'STATIC','PRECIP','PRECIP_T%03d.dat' % tile)
    fmax_fn = os.path.join(base,'STATIC','FMAX','FMAX_T%03d.dat' % tile)
    terrain_fn = os.path.join(base,'STATIC','TERRAIN_SD','TERRAIN_T%03d.dat' % tile)
    daylst_fn = os.path.join(base,'TILES','T%03d' % tile,'FINAL_DAY_LST_%s_T%03d.dat' % (date,tile))
    nightlst_fn = os.path.join(base,'TILES','T%03d' % tile,'FINAL_NIGHT_LST_%s_T%03d.dat' % (date,tile))
#    lai_fn = os.path.join(base,'STATIC','LAI','MLAI_%s_T%03d.dat' % (laiddd,tile))
    lai_fn = os.path.join(base,'STATIC','LAI','MLAI_2015%03d_T%03d.dat' % (rday,tile))
    dtime_fn = os.path.join(base,'STATIC','DTIME','DTIME_2014%03d_T%03d.dat' % (risedoy,tile))
#    lst_day = np.fromfile(daylst_fn, dtype=np.float32)
#    lst_day= np.flipud(lst_day.reshape([3750,3750]))
#    lst_day.tofile(daylst_fn)
    convertBin2tif(daylst_fn,inUL,ALEXIshape,ALEXIres,'float32',gdal.GDT_Float32)
#    lst_night = np.fromfile(nightlst_fn, dtype=np.float32)
#    lst_night= np.flipud(lst_night.reshape([3750,3750]))
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
    dtrad_fn = os.path.join(tile_path,
                            "FINAL_DTRAD_%s_T%03d.dat" % ( date, tile))
    subprocess.check_output(["%s" % merge,"%s" % fn1, "%s" % fn2,"%s" % fn3,
                             "%s" % fn4, "%s" % fn5, "%s" % fn6, "%s" % dtrad_fn])
    lst_fn = os.path.join(tile_path,
                            "FINAL_DAY_LST_TIME2_%s_T%03d.dat" % ( date, tile))
    subprocess.check_output(["%s" % calc_predicted_trad2,"%s" % nightlst_fn, 
                         "%s" % daylst_fn, "%s" % lai_fn, "%s" % lst_fn ])
    #================+TESTING==================================================
    
    testing_path = os.path.join(tile_base_path,'DTRAD','%03d' % doy)
    if not os.path.exists(testing_path):
        os.makedirs(testing_path)
    testing_fn = os.path.join(testing_path,'FINAL_DTRAD_%s_T%03d.dat' % (date,tile))
#    dtime = np.fromfile(dtrad_fn, dtype=np.float32)
#    dtime= np.flipud(dtime.reshape([3750,3750]))
#    dtime.tofile(dtrad_fn)
    shutil.copyfile(dtrad_fn,testing_fn)
    convertBin2tif(testing_fn,inUL,ALEXIshape,ALEXIres,np.float32,gdal.GDT_Float32)
#    dtime = np.fromfile(lst_fn, dtype=np.float32)
#    dtime= np.flipud(dtime.reshape([3750,3750]))
#    dtime.tofile(lst_fn)
    testing_path = os.path.join(tile_base_path,'LST2','%03d' % doy)
    if not os.path.exists(testing_path):
        os.makedirs(testing_path)
    testing_fn = os.path.join(testing_path,'FINAL_DAY_LST_TIME2_%s_T%03d.dat' % (date,tile))
    
    shutil.copyfile(lst_fn,testing_fn)
    convertBin2tif(testing_fn,inUL,ALEXIshape,ALEXIres,np.float32,gdal.GDT_Float32)

def buildRNETtrees(year,doy):
    dtimedates = np.array(range(1,366,7))
    r7day = dtimedates[dtimedates>=doy][0]
    riseddd="%d%03d" %(year,r7day)
    halfdeg_sizeArr = 900*1800

    #========process insol====================================================
    srcfn = os.path.join(static_path,'INSOL','deg05','insol55_2011%03d.tif' % doy)
    g = gdal.Open(srcfn,GA_ReadOnly)
    insol= g.ReadAsArray()
    insol = np.reshape(insol,[halfdeg_sizeArr])

    #======process RNET========================================================
#    srcfn = os.path.join(static_path,'5KM','RNET','RNET%s.dat' % riseddd)
    srcfn = os.path.join(static_path,'5KM','RNET','RNET2015%03d.dat' % r7day)
    rnet = np.fromfile(srcfn, dtype=np.float32)
    rnet = np.flipud(rnet.reshape([3000,7200]))
    inProjection = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'
    tif_fn = srcfn[:-4]+'.tif'
    if not os.path.exists(tif_fn):
        writeArray2Tiff(rnet,[0.05,0.05],[-180.,90.],inProjection,tif_fn,gdal.GDT_Float32)
    outfn = tif_fn[:-4]+'subset.tif'
    out = subprocess.check_output('gdal_translate -of GTiff -projwin -30 45 60 0 -tr 0.05 0.05 %s %s' % (tif_fn,outfn), shell=True)
    g = gdal.Open(outfn,GA_ReadOnly)
    rnet= g.ReadAsArray()
    rnet = np.reshape(rnet,[halfdeg_sizeArr])
    #======process albedo======================================================
    srcfn = os.path.join(static_path,'ALBEDO','ALBEDO.tif')
    g = gdal.Open(srcfn,GA_ReadOnly)
    albedo = g.ReadAsArray()
    albedo = np.reshape(albedo,[halfdeg_sizeArr])
    
    #=====process LST2=========================================================
    srcPath = os.path.join(tile_base_path,'LST2','%03d' % doy)
    searchPath = os.path.join(srcPath,'FINAL_DAY_LST_TIME2*.tif')
    outfn = os.path.join(srcPath,'LST2.vrt')
    outfn05 = outfn[:-4]+'05.tif'
    subprocess.check_output('gdalbuildvrt %s %s' % (outfn, searchPath), shell=True)
    out = subprocess.check_output('gdal_translate -of GTiff -tr 0.05 0.05 %s %s' % (outfn,outfn05), shell=True)
    g = gdal.Open(outfn05,GA_ReadOnly)
    lst2 = g.ReadAsArray()
    lst2 = np.reshape(lst2,[halfdeg_sizeArr])
    #====process LWDN==========================================================
    time = get_rise55(year,doy,86)
    grab_time = getGrabTime(int(time)*100)
    hr,forecastHR,cfsr_doy = getGrabTimeInv(grab_time/100,doy)
    cfsr_date = "%d%03d" % (year,cfsr_doy)
    if (grab_time)==2400:
        grab_time = 0000
    srcfn = os.path.join(static_path,'CFSR','%d' % year,'%03d' % cfsr_doy,'sfc_lwdn_%s_%02d00.dat' % (cfsr_date,grab_time/100))
    lwdn25 = np.fromfile(srcfn, dtype=np.float32)
    lwdn25 = np.flipud(lwdn25.reshape([720, 1440]))
    inProjection = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'
    tif_fn = srcfn[:-4]+'.tif'
    if not os.path.exists(tif_fn):
        writeArray2Tiff(lwdn25,[0.25,0.25],[-180.,90.],inProjection,tif_fn,gdal.GDT_Float32)
    outfn = tif_fn[:-4]+'05.tif'
#    outfn = os.path.join(outPath,tif_fn.split(os.sep)[-1])
    out = subprocess.check_output('gdal_translate -of GTiff -projwin -30 45 60 0 -tr 0.05 0.05 %s %s' % (tif_fn,outfn), shell=True)
    g = gdal.Open(outfn,GA_ReadOnly)
    lwdn = g.ReadAsArray()
    lwdn = np.reshape(lwdn,[halfdeg_sizeArr])
    
    #==========create fstem.data for cubist====================================
    outDict = {'rnet': rnet, 'albedo':albedo, 'insol':insol, 'lwdn': lwdn, 'lst2':lst2}
    inDF = pd.DataFrame.from_dict(outDict)
    outDF = inDF.loc[(inDF["rnet"] > 0.0) & (inDF["albedo"] > 0.0) & 
                (inDF["insol"] > 0.0) & (inDF["lwdn"] > 0.0) &
                (inDF["lst2"] > 0.0), ["rnet","albedo","insol","lwdn","lst2"]]
    calc_rnet_tile_ctl = os.path.join(calc_rnet_path,'tiles_ctl')
    if not os.path.exists(calc_rnet_tile_ctl):
        os.makedirs(calc_rnet_tile_ctl) 
    file_data = os.path.join(calc_rnet_tile_ctl,'rnet.data')
    outDF.to_csv(file_data , header=True, index=False,columns=["rnet",
                                        "albedo","insol","lwdn","lst2"])
    file_names = os.path.join(calc_rnet_tile_ctl,'rnet.names')
    get_tiles_fstem_names(file_names)
    
    #====run cubist============================================================
#    print("running cubist...")
    cubist_name = os.path.join(calc_rnet_tile_ctl,'rnet')
    rnet_cub_out = subprocess.check_output("cubist -f %s -u -a -r 20" % cubist_name, shell=True)
    return rnet_cub_out

def getRNETfromTrees(tile,year,doy,rnet_cub_out):
    LLlat,LLlon = tile2latlon(tile)
    URlat = LLlat+15.
    URlon = LLlon+15.
    inUL = [LLlon,URlat]
    ALEXI_shape = [3750,3750]
    ALEXI_res = [0.004,0.004]
    date = '%d%03d' % (year,doy)
    tile_path = os.path.join(tile_base_path,"T%03d" % tile)
    #====open INSOL =============================================
    srcfn = os.path.join(static_path,'INSOL','deg004','insol55_2011%03d.tif' % doy)
    outfn = srcfn[:-4]+'_T%03d.tif' % tile
    out = subprocess.check_output('gdalwarp -overwrite -of GTiff -te %f %f %f %f -tr 0.004 0.004 %s %s' % (LLlon,LLlat,URlon,URlat,srcfn,outfn), shell=True)
    g = gdal.Open(outfn,GA_ReadOnly)
    insol_viirs= g.ReadAsArray()
    insol_viirs = np.reshape(insol_viirs,[3750*3750])
    
    #======process albedo======================================================
    albedo_fn = os.path.join(static_path,'ALBEDO','ALBEDO_T%03d.dat' % tile)
    albedo = np.fromfile(albedo_fn, dtype=np.float32)
    albedo = np.reshape(albedo,[3750*3750])
    
    #=====process LST2=========================================================
#    lst_fn = os.path.join(rnet_tile_path,'LST2_%03d_%s.dat' % (tile,date))
    lst_fn = os.path.join(tile_path,
                            "FINAL_DAY_LST_TIME2_%s_T%03d.dat" % ( date, tile))
    lst = np.fromfile(lst_fn, dtype=np.float32)
    lst2 = np.reshape(lst,[3750*3750])
    
    #====process LWDN==========================================================
    time = get_rise55(year,doy,tile)
    grab_time = getGrabTime(int(time)*100)
    hr,forecastHR,cfsr_doy = getGrabTimeInv(grab_time/100,doy)
    cfsr_date = "%d%03d" % (year,cfsr_doy)
    if (grab_time)==2400:
        grab_time = 0000
    srcfn = os.path.join(static_path,'CFSR','%d' % year,'%03d' % cfsr_doy,'sfc_lwdn_%s_%02d00.dat' % (cfsr_date,grab_time/100))
    lwdn25 = np.fromfile(srcfn, dtype=np.float32)
    lwdn25 = np.flipud(lwdn25.reshape([720, 1440]))
    inProjection = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'
    tif_fn = srcfn[:-4]+'.tif'
    if not os.path.exists(tif_fn):
        writeArray2Tiff(lwdn25,[0.25,0.25],[-180.,90.],inProjection,tif_fn,gdal.GDT_Float32)
    outfn = tif_fn[:-4]+'_T%03d.tif' % tile
    out = subprocess.check_output('gdal_translate -of GTiff -projwin %f %f %f %f -tr 0.004 0.004 %s %s' % (LLlon,URlat,URlon,LLlat,tif_fn,outfn), shell=True)
    g = gdal.Open(outfn,GA_ReadOnly)
    lwdn = g.ReadAsArray()
    lwdn_viirs = np.reshape(lwdn,[3750*3750])
    #=======get the final_rnet=================================================
    cubDict = {'albedo':albedo, 'insol':insol_viirs, 'lwdn': lwdn_viirs, 'lst2':lst2}
    cubDF = pd.DataFrame.from_dict(cubDict)
    rnet_out = readCubistOut(rnet_cub_out,cubDF)
    rnet_out = np.reshape(rnet_out, [3750,3750])
    rnet_tile = os.path.join(tile_base_path,'T%03d' % tile)
    if not os.path.exists(rnet_tile):
        os.makedirs(rnet_tile)
    finalrnet_fn = os.path.join(rnet_tile,'FINAL_RNET_%s_T%03d.dat' % (date,tile))
    rnet_out = np.array(rnet_out,dtype='Float32')
    rnet_out.tofile(finalrnet_fn)
    convertBin2tif(finalrnet_fn,inUL,ALEXI_shape,ALEXI_res,'float32',gdal.GDT_Float32)
    
    #======TESTING=============================================================
    testing_path = os.path.join(tile_base_path,'RNET','%03d' % doy)
    if not os.path.exists(testing_path):
        os.makedirs(testing_path)
    testing_fn = os.path.join(testing_path,'FINAL_RNET_%s_T%03d.dat' % (date,tile))
    shutil.copyfile(finalrnet_fn,testing_fn)
    convertBin2tif(testing_fn,inUL,ALEXI_shape,ALEXI_res,np.float32,gdal.GDT_Float32)  
    
def processTiles(tile,year,doy):
    LLlat,LLlon = tile2latlon(tile)
    URlat = LLlat+15.
    URlon = LLlon+15.
    inUL = [LLlon,URlat]
    halfdeg_shape = [300,300]
    halfdeg_sizeArr = 300*300
    ALEXI_shape = [3750,3750]
    ALEXI_res = [0.004,0.004]
    date = '%d%03d' % (year,doy)
    date_tile_str = "T%03d_%s" % (tile,date)
    writeCTL(tile,year,doy)
    
    #========process insol====================================================
        
    insol_fn = os.path.join(rnet_tile_path,'INSOL_%03d_%s.dat' % (tile,date))
    write_agg_insol(insol_fn,date_tile_str)
    out = subprocess.check_output("opengrads -blxc 'run ./%s_agg_insol.gs %3.2f %3.2f  %3.2f  %3.2f'" 
                                  % (date_tile_str,LLlat, URlat, LLlon, URlon), shell=True)

    read_data = np.fromfile(insol_fn, dtype=np.float32)
    insol = np.flipud(read_data.reshape([halfdeg_shape[0],halfdeg_shape[1]]))
    insol = np.reshape(insol,[halfdeg_sizeArr])
    
    viirs_tile_path = os.path.join(calc_rnet_path,'viirs','T%03d' % tile)
    if not os.path.exists(viirs_tile_path):
        os.makedirs(viirs_tile_path) 
    #====process VIIRS resolution =============================================   
    insol_viirs_fn = os.path.join(viirs_tile_path,'INSOL_%03d_%s.dat' % (tile,date))
    write_agg_insol_viirs(insol_viirs_fn,date_tile_str)
    out = subprocess.check_output("opengrads -blxc 'run ./%s_agg_insol_viirs.gs %3.2f %3.2f  %3.2f  %3.2f'" 
                                  % (date_tile_str,LLlat, URlat, LLlon, URlon), shell=True)
    insol_viirs = np.fromfile(insol_viirs_fn, dtype=np.float32)
    insol_viirs = np.flipud(insol_viirs.reshape([ALEXI_shape[0],ALEXI_shape[1]]))
    insol_viirs = np.reshape(insol_viirs,[3750*3750])
    convertBin2tif(insol_viirs_fn,inUL,ALEXI_shape,ALEXI_res,'float32',gdal.GDT_Float32)
    
    #======process RNET=======================================================
    rnet_fn = os.path.join(rnet_tile_path,'RNET_%03d_%s.dat' % (tile,date))
    write_agg_rnet(rnet_fn,date_tile_str)
    
    out = subprocess.check_output("opengrads -blxc 'run ./%s_agg_rnet.gs %3.2f %3.2f  %3.2f  %3.2f'" 
                                  % (date_tile_str,LLlat, URlat, LLlon, URlon), shell=True)
    read_data = np.fromfile(rnet_fn, dtype=np.float32)
    rnet = np.flipud(read_data.reshape([halfdeg_shape[0],halfdeg_shape[1]]))
    rnet = np.reshape(rnet,[halfdeg_sizeArr])  
    
    #======process albedo======================================================
    albedo_fn = os.path.join(rnet_tile_path,'ALBEDO_%03d_%s.dat' % (tile,date))
    write_agg_albedo(albedo_fn,date_tile_str)
    
    out = subprocess.check_output("opengrads -blxc 'run ./%s_agg_albedo.gs %3.2f %3.2f  %3.2f  %3.2f'" 
                                  % (date_tile_str,LLlat, URlat, LLlon, URlon), shell=True)
    albedo = np.fromfile(albedo_fn, dtype=np.float32)
    albedo = albedo.reshape([halfdeg_shape[0],halfdeg_shape[1]])
    albedo = np.reshape(albedo,[halfdeg_sizeArr])
    
    #=====process LST2=========================================================
    lst_fn = os.path.join(rnet_tile_path,'LST2_%03d_%s.dat' % (tile,date))
    write_agg_lst2(lst_fn,date_tile_str)
    
    out = subprocess.check_output("opengrads -blxc 'run ./%s_agg_lst2.gs %3.2f %3.2f  %3.2f  %3.2f'" 
                                  % (date_tile_str,LLlat, URlat, LLlon, URlon), shell=True)
    lst = np.fromfile(lst_fn, dtype=np.float32)

    #====process LWDN==========================================================
    lwdn_fn = os.path.join(rnet_tile_path,'LWDN_%03d_%s.dat' % (tile,date))
    write_agg_lwdn(lwdn_fn,date_tile_str)
    out = subprocess.check_output("opengrads -blxc 'run ./%s_agg_lwdn.gs %3.2f %3.2f  %3.2f  %3.2f'" 
                                  % (date_tile_str,LLlat, URlat, LLlon, URlon), shell=True)
    
    read_data = np.fromfile(lwdn_fn, dtype=np.float32)
    lwdn = np.flipud(read_data.reshape([halfdeg_shape[0],halfdeg_shape[1]]))
    lwdn = np.reshape(lwdn,[halfdeg_sizeArr])
    
    #====process VIIRS resolution =============================================
    lwdn_viirs_fn = os.path.join(viirs_tile_path,'LWDN_%03d_%s.dat' % (tile,date))
    write_agg_lwdn_viirs(lwdn_viirs_fn,date_tile_str)
    out = subprocess.check_output("opengrads -blxc 'run ./%s_agg_lwdn_viirs.gs %3.2f %3.2f  %3.2f  %3.2f'" 
                                  % (date_tile_str,LLlat, URlat, LLlon, URlon), shell=True)
    
    lwdn_viirs = np.fromfile(lwdn_viirs_fn, dtype=np.float32)
    lwdn_viirs = np.flipud(lwdn_viirs.reshape([ALEXI_shape[0],ALEXI_shape[1]]))
    lwdn_viirs = np.reshape(lwdn_viirs,[3750*3750])
    convertBin2tif(lwdn_viirs_fn,inUL,ALEXI_shape,ALEXI_res,'float32',gdal.GDT_Float32)

    #==========create fstem.data for cubist====================================
    outDict = {'rnet': rnet, 'albedo':albedo, 'insol':insol, 'lwdn': lwdn, 'lst2':lst}
    inDF = pd.DataFrame.from_dict(outDict)
    outDF = inDF.loc[(inDF["rnet"] > 0.0) & (inDF["albedo"] > 0.0) & 
                (inDF["insol"] > 0.0) & (inDF["lwdn"] > 0.0) &
                (inDF["lst2"] > 0.0), ["rnet","albedo","insol","lwdn","lst2"]]
    calc_rnet_tile_ctl = os.path.join(calc_rnet_path,'tiles_ctl','T%03d' % tile )
    if not os.path.exists(calc_rnet_tile_ctl):
        os.makedirs(calc_rnet_tile_ctl) 
    file_data = os.path.join(calc_rnet_tile_ctl,'rnet.data')
    outDF.to_csv(file_data , header=True, index=False,columns=["rnet",
                                        "albedo","insol","lwdn","lst2"])
    file_names = os.path.join(calc_rnet_tile_ctl,'rnet.names')
    get_tiles_fstem_names(file_names)
    
    #====run cubist============================================================
#    print("running cubist...")
    cubist_name = os.path.join(calc_rnet_tile_ctl,'rnet')
    rnet_cub_out = subprocess.check_output("cubist -f %s -u -a -r 20" % cubist_name, shell=True)

    #=======get the final_rnet=================================================
    lst2 = np.fromfile('./%s_lst2.dat' % date_tile_str, dtype=np.float32)
    albedo = np.fromfile('./%s_albedo.dat' % date_tile_str, dtype=np.float32)
    convertBin2tif('./%s_albedo.dat' % date_tile_str,inUL,ALEXI_shape,ALEXI_res,'float32',gdal.GDT_Float32)
    cubDict = {'albedo':albedo, 'insol':insol_viirs, 'lwdn': lwdn_viirs, 'lst2':lst2}
    cubDF = pd.DataFrame.from_dict(cubDict)
    rnet_out = readCubistOut(rnet_cub_out,cubDF)
    rnet_out = np.reshape(rnet_out, [3750,3750])
    rnet_tile = os.path.join(tile_base_path,'T%03d' % tile)
    if not os.path.exists(rnet_tile):
        os.makedirs(rnet_tile)
    finalrnet_fn = os.path.join(rnet_tile,'FINAL_RNET_%s_T%03d.dat' % (date,tile))
    rnet_out = np.array(rnet_out,dtype='Float32')
    rnet_out.tofile(finalrnet_fn)
    convertBin2tif(finalrnet_fn,inUL,ALEXI_shape,ALEXI_res,'float32',gdal.GDT_Float32)
    
    #======TESTING=============================================================
    testing_path = os.path.join(tile_base_path,'RNET','%03d' % doy)
    if not os.path.exists(testing_path):
        os.makedirs(testing_path)
    testing_fn = os.path.join(testing_path,'FINAL_RNET_%s_T%03d.dat' % (date,tile))
    shutil.copyfile(finalrnet_fn,testing_fn)
    convertBin2tif(testing_fn,inUL,ALEXI_shape,ALEXI_res,np.float32,gdal.GDT_Float32)
    
def useTrees(tile,year,doy,trees):
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
#    lai_fn = os.path.join(static_path,'LAI','MLAI_%s_T%03d.dat' % (laiddd,tile)) # only have 2015 so far
    lai_fn = os.path.join(static_path,'LAI','MLAI_2015%03d_T%03d.dat' % (r4day,tile)) # TEMPORARY FOR RT PROCESSING
    dthr_corr_fn = os.path.join(static_path,'DTHR_CORR','DTHR_CORR_2010%03d_T%03d.dat' % (r7day,tile))
    dtime_fn = os.path.join(static_path,'DTIME','DTIME_2014%03d_T%03d.dat' % (r7day,tile))
    fmax_fn = os.path.join(static_path,'FMAX','FMAX_T%03d.dat' % (tile))
    precip_fn = os.path.join(static_path,'PRECIP','PRECIP_T%03d.dat' % (tile))
    
    dthr = np.fromfile(dthr_fn, dtype=np.float32)
#    dthr = dthr.reshape([3750,3750])
#    plt.imshow(dthr)
#    dthr = np.reshape(dthr,[3750*3750])
    trad2 = np.fromfile(trad2_fn, dtype=np.float32)
#    trad2 = trad2.reshape([3750,3750])
#    plt.imshow(trad2)
#    trad2 = np.reshape(trad2,[3750*3750])
    rnet = np.fromfile(rnet_fn, dtype=np.float32)
#    rnet = rnet.reshape([3750,3750])
#    plt.imshow(rnet)
#    rnet = np.reshape(rnet,[3750*3750])    
    lai = np.fromfile(lai_fn, dtype=np.float32)
#    lai = lai.reshape([3750,3750])
#    plt.imshow(lai, vmin=0, vmax=2)
#    lai = np.reshape(lai,[3750*3750])
    dthr_corr = np.fromfile(dthr_corr_fn, dtype=np.float32)
    dthr_corr = np.flipud(dthr_corr.reshape([3750,3750]))
#    plt.imshow(dthr_corr, vmin=0,vmax=3)
    dthr_corr = np.reshape(dthr_corr,[3750*3750])    
    dtime = np.fromfile(dtime_fn, dtype=np.float32)
#    dtime = dtime.reshape([3750,3750])
#    plt.imshow(dtime)
#    dtime = np.reshape(dtime,[3750*3750])
    fmax = np.fromfile(fmax_fn, dtype=np.float32)
#    fmax = fmax.reshape([3750,3750])
#    plt.imshow(fmax, vmin=0, vmax=0.3)
#    fmax = np.reshape(fmax,[3750*3750])    
    precip = np.fromfile(precip_fn, dtype=np.float32)
#    precip = precip.reshape([3750,3750])
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
#    a_nans.tofile(out_nancount_fn)
#    convertBin2tif(out_nancount_fn,inUL,ALEXI_shape,ALEXI_res,'float32',gdal.GDT_Float32)
# =============================================================================
    testing_path = os.path.join(tile_base_path,'FSUN','%03d' % doy)
    if not os.path.exists(testing_path):
        os.makedirs(testing_path)
    testing_fn = os.path.join(testing_path,'FINAL_FSUN_%s_T%03d.dat' % (date,tile))

    shutil.copyfile(out_fsun_fn,testing_fn)
    convertBin2tif(testing_fn,inUL,ALEXI_shape,ALEXI_res,np.float32,gdal.GDT_Float32)

def getDailyET(tile,year,doy):
    inProjection = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'
    LLlat,LLlon = tile2latlon(tile)
    URlat = LLlat+15.
    inUL = [LLlon,URlat]
    ALEXI_shape = [3750,3750]
    ALEXI_res = [0.004,0.004]
    date = '%d%03d' % (year,doy)
    insol24_fn = os.path.join(static_path,'INSOL24', 'RS24_%s_T%03d.tif' % (date,tile))
    g = gdal.Open(insol24_fn,GA_ReadOnly)
    Rs24= g.ReadAsArray()
#    Rs24=(Rs24*0.0864)/24.0 
    Rs24=(Rs24/8.)*0.0864 # there are 8 measurements of 3 hour averages from CFSR NOT 24!
    fsun_fn = os.path.join(tile_base_path,'T%03d' % tile, 'FINAL_FSUN_%s_T%03d.dat' % (date,tile))

    Fsun = np.fromfile(fsun_fn, dtype=np.float32)
    Fsun = Fsun.reshape([3750,3750])
    EFeq=Fsun*(Rs24)
    ET_24 = EFeq*0.408
    ET_24[ET_24<0.01]=0.01
    ET_24 = np.array(ET_24,dtype='Float32')
    
    testing_path = os.path.join(tile_base_path,'ET','%03d' % doy)
    if not os.path.exists(testing_path):
        os.makedirs(testing_path)
    testing_fn = os.path.join(testing_path,'FINAL_EDAY_%s_T%03d.dat' % (date,tile))
    
#    ET_24.tofile(testing_fn)
#    convertBin2tif(testing_fn,inUL,ALEXI_shape,ALEXI_res,np.float32,gdal.GDT_Float32)
#    searchPath = os.path.join(testing_path,'*.tif')
#    outfn = os.path.join(testing_path,'ET_%03d.vrt' % doy)
    out_tif_fn = os.path.join(testing_path,'ET_%03d.tif' % doy)
#    subprocess.check_output('gdalbuildvrt %s %s' % (outfn, searchPath), shell=True)
#    out = subprocess.check_output('gdal_translate -of GTiff %s %s' % (outfn,out_tif_fn), shell=True)

    writeArray2Tiff(ET_24,ALEXI_res,inUL,inProjection,out_tif_fn,gdal.GDT_Float32)
    
def runSteps(par,trees,tile=None,year=None,doy=None):
    if year==None:
        dd = datetime.date.today()+datetime.timedelta(days=-1)
        year = dd.year
        
    if doy==None:
        doy = (datetime.date.today()-datetime.date(year,1,1)).days-1

    # ============process one tile at a time ==================================
    if par==0:
#        print("building VIIRS coordinates LUT--------------->")
#        getIJcoordsPython(tile)
        print("gridding VIIRS data-------------------------->")
#        res = gridMergePython(tile,year,doy)
        gridMergePythonEWA(tile,year,doy)
#        if res > 0:
#            print("no viirs data")
#        else:
        print("running I5 atmosperic correction------------->")
    #    startatmos = timer.time()
#            atmosCorrection(tile,year,doy)
        atmosCorrectPython(tile,year,doy)
    #    end = timer.time()
    #    print("atmoscorr time: %f" % (end - startatmos))
        print("estimating dtrad and LST2-------------------->")
        pred_dtrad(tile,year,doy)
        print("estimating RNET ----------------------------->")
        processTiles(tile,year,doy)
        print("estimating FSUN------------------------------>")
        useTrees(tile,year,doy,trees)
        print("making ET------------------------------------>")
        getDailyET(tile,year,doy)
        print("============FINISHED!=========================")
    else:
        # ===========for processing all tiles in parallel======================
        tiles = [60,61,62,63,64,83,84,85,86,87,88,107,108,109,110,111,112]
        print("gridding VIIRS data-------------------------->")
#        r = Parallel(n_jobs=-1, verbose=5)(delayed(gridMergePython)(tile,year,doy) for tile in tiles)
        r = Parallel(n_jobs=-1, verbose=5)(delayed(gridMergePythonEWA)(tile,year,doy) for tile in tiles)
        print("running I5 atmosperic correction------------->")
#        r = Parallel(n_jobs=-1, verbose=5)(delayed(atmosCorrection)(tile,year,doy) for tile in tiles)
        r = Parallel(n_jobs=-1, verbose=5)(delayed(atmosCorrectPython)(tile,year,doy) for tile in tiles)
        print("estimating dtrad and LST2-------------------->")
        r = Parallel(n_jobs=-1, verbose=5)(delayed(pred_dtrad)(tile,year,doy) for tile in tiles)
        print("build RNET trees----------------------------->") # Using MENA region for building trees
        tree = buildRNETtrees(year,doy)
        print("estimating RNET ----------------------------->")
#        r = Parallel(n_jobs=-1, verbose=5)(delayed(processTiles)(tile,year,doy) for tile in tiles) # using Tiles to build RNET trees 
        r = Parallel(n_jobs=-1, verbose=5)(delayed(getRNETfromTrees)(tile,year,doy,tree) for tile in tiles)
#        getRNETfromTrees(tile,year,doy,rnet_cub_out)
        print("estimating FSUN------------------------------>")
        r = Parallel(n_jobs=-1, verbose=5)(delayed(useTrees)(tile,year,doy,trees) for tile in tiles)
        print("making ET------------------------------------>")
        r = Parallel(n_jobs=-1, verbose=5)(delayed(getDailyET)(tile,year,doy) for tile in tiles)
        print("============FINISHED!=========================")


def main():
    # Get time and location from user
    parser = argparse.ArgumentParser()
    parser.add_argument("year", type=float, default=None, help="year of data")
    parser.add_argument("start_doy", type=int, default=-1, help="start day of processing. *Note: leave blank for Real-time")
    parser.add_argument("end_doy", type=int, default=-1, help="end day of processing. *Note: leave blank for Real-time")
    args = parser.parse_args()
    year= args.year
    start_doy = args.start_doy
    end_doy= args.end_doy
 
    if start_doy == None:
        start = timer.time()
        trees = processTrees() # until we have data for other years only use 2015
        #    runSteps(1,trees,None,year,doy)
        runSteps(1,trees)   
        end = timer.time()
        print("program duration: %f minutes" % ((end - start)/60.))
    else:
        days = range(start_doy,end_doy)
        start = timer.time()
        for doy in days:
            print("processing day:%d of year:%d" % (doy,year))
            print("building regression trees from 5KM data---------->")
            trees = processTrees(year,doy) # until we have data for other years only use 2015
            runSteps(1,trees,None,year,doy)
        end = timer.time()
        print("program duration: %f minutes" % ((end - start)/60.))
    
#year = 2015
#doy = 221
#days = range(225,228)