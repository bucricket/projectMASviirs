#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 09:02:05 2017

@author: mschull
"""
import os
import subprocess
import gzip
import shutil
import numpy as np
import ephem
import datetime
import pandas as pd
from osgeo import gdal, osr

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
    calc_rnet_path = os.path.join(processing_path,'CALC_RNET')
    if not os.path.exists(calc_rnet_path):
        os.makedirs(calc_rnet_path)
    out = {'processing_path':processing_path,
   'static_path':static_path,'tile_base_path':tile_base_path,
   'calc_rnet_path':calc_rnet_path}
    return out

base = os.getcwd()
Folders = folders(base) 
processing_path = Folders['processing_path']
static_path = Folders['static_path']
tile_base_path = Folders['tile_base_path']
calc_rnet_path = Folders['calc_rnet_path']

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
def tile2latlon(tile):
    row = tile/24
    col = tile-(row*24)
    # find lower left corner
    lat= (75.-row*15.)-15.
    lon=(col*15.-180.)-15. 
    return [lat,lon]

def getTile(LLlat,LLlon):
    URlat = LLlat+15
    URlon = LLlon+15
    return np.abs(((75)-URlat)/15)*24+np.abs(((-180)-(URlon))/15)

def get_rise55(year,doy,tile):
    dd=datetime.datetime(year,1,1)+datetime.timedelta(days=doy)
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

def writeCTL(tile,year,doy):
    LLlat,LLlon = tile2latlon(tile)
    #====create insol.ctl====================================
    time = get_rise55(year,doy,tile)
    time_int=time/3.
    rem=time_int-int(time_int)
    if (rem < 0.50):
        grab_time=int(time_int)*3
    if (rem >= 0.50):
        grab_time=int(time_int)*3+3
    
    date = "%d%03d" % (year,doy)
    if (grab_time == 24): 
        grab_time=0
        cfsdoy=doy+1
        date = "%d%03d" % (year,cfsdoy)
    
    srcfn = os.path.join(static_path,'INSOL','deg125','insol55_2011%03d.dat' % doy)
    
    dtimedates = np.array(range(1,366,7))
    rday = dtimedates[dtimedates>=doy][0]
    riseddd="%d%03d" %(year,rday)
    shutil.copyfile(srcfn,'./current_insol.dat')
    fn = os.path.join('./insol.ctl')
    data = 'current_insol.dat'
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
    gunzip(srcfn,out_fn='./current_rnet.dat')
    rnet05 = np.fromfile('./current_rnet.dat', dtype=np.float32)
    rnet05 = np.flipud(rnet05.reshape([3000,7200]))
    rnet05.tofile('./current_rnet.dat')
    fn = os.path.join('./rnet.ctl')
    data = os.path.join('current_rnet.dat')
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
    shutil.copyfile(srcfn,'./current_albedo.dat')
    fn = os.path.join('./albedo.ctl')
    data = os.path.join('current_albedo.dat')
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
    #FOR TESTING
#    testing_path = '/data/VIIRS_GLOBAL_PROCESS/tiles/'
#    srcfn = os.path.join(testing_path,'T%03d' % tile,'FINAL_DAY_LST_TIME2_%s_T%03d.dat.gz' % (date,tile))
    shutil.copyfile(srcfn,'./current_lst2.dat')
    fn = os.path.join('./lst2.ctl')
    data = os.path.join('current_lst2.dat')
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
        
    srcfn = os.path.join(processing_path,'CFSR','%d' % year,'sfc_lwdn_%s_%02d00.dat' % (date,grab_time))
    shutil.copyfile(srcfn,'./current_lwdn.dat')
    fn = os.path.join('./lwdn.ctl')
    data = os.path.join('current_lwdn.dat')
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


def write_insol_viirs(outfn):
    fn = os.path.join('./agg_insol.gs')
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
    file.write("'open ./insol.ctl'\n")
    file.write("'set lat ' lat1+0.025 ' ' lat2-0.025\n")
    file.write("'set lon ' lon1+0.025 ' ' lon2-0.025\n")
    file.write("'set undef -9999.'\n")
    file.write("'define test=re(soil.1,0.05,0.05)'\n")
    file.write("'set gxout fwrite'\n")
    file.write("'set fwrite %s'\n" % outfn)
    file.write("'d test'\n")
    file.write("'disable fwrite'\n")
    file.close()

def write_agg_insol_viirs(outfn):
    fn = os.path.join('./agg_insol_viirs.gs')
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
    file.write("'open ./insol.ctl'\n")
    file.write("'set lat ' lat1+0.002 ' ' lat2-0.002\n")
    file.write("'set lon ' lon1+0.002 ' ' lon2-0.002\n")
    file.write("'set undef -9999.'\n")
    file.write("'define test=re(soil.1,0.004,0.004)'\n")
    file.write("'set gxout fwrite'\n")
    file.write("'set fwrite %s'\n" % outfn)
    file.write("'d test'\n")
    file.write("'disable fwrite'\n")
    file.close()

def write_agg_rnet(outfn):
    fn = os.path.join('./agg_rnet.gs')
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
    file.write("'open ./rnet.ctl'\n")
    file.write("'set lat ' lat1+0.025 ' ' lat2-0.025\n")
    file.write("'set lon ' lon1+0.025 ' ' lon2-0.025\n")
    file.write("'set undef -9999.'\n")
    file.write("'define test=re(soil.1,0.05,0.05)'\n")
    file.write("'set gxout fwrite'\n")
    file.write("'set fwrite %s'\n" % outfn)
    file.write("'d test'\n")
    file.write("'disable fwrite'\n")
    file.close()

def write_agg_albedo(outfn):
    fn = os.path.join('./agg_albedo.gs')
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
    file.write("'open ./albedo.ctl'\n")
    file.write("'set lat ' lat1+0.025 ' ' lat2-0.025\n")
    file.write("'set lon ' lon1+0.025 ' ' lon2-0.025\n")
    file.write("'set undef -9999.'\n")
    file.write("'define test=re(soil.1,0.05,0.05)'\n")
    file.write("'set gxout fwrite'\n")
    file.write("'set fwrite %s'\n" % outfn)
    file.write("'d test'\n")
    file.write("'disable fwrite'\n")
    file.close()
    
    
def write_agg_lst2(outfn):
    fn = os.path.join('./agg_lst2.gs')
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
    file.write("'open ./lst2.ctl'\n")
    file.write("'set lat ' lat1+0.025 ' ' lat2-0.025\n")
    file.write("'set lon ' lon1+0.025 ' ' lon2-0.025\n")
    file.write("'set undef -9999.'\n")
    file.write("'define test=re(soil.1,0.05,0.05)'\n")
    file.write("'set gxout fwrite'\n")
    file.write("'set fwrite %s'\n" % outfn)
    file.write("'d test'\n")
    file.write("'disable fwrite'\n")
    file.close()
    
def write_agg_lwdn(outfn):
    fn = os.path.join('./agg_lwdn.gs')
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
    file.write("'open ./lwdn.ctl'\n")
    file.write("'set lat ' lat1+0.025 ' ' lat2-0.025\n")
    file.write("'set lon ' lon1+0.025 ' ' lon2-0.025\n")
    file.write("'set undef -9999.'\n")
    file.write("'define test=re(soil.1,0.05,0.05)'\n")
    file.write("'set gxout fwrite'\n")
    file.write("'set fwrite %s'\n" % outfn)
    file.write("'d test'\n")
    file.write("'disable fwrite'\n")
    file.close()

def write_agg_lwdn_viirs(outfn):
    fn = os.path.join('./agg_lwdn_viirs.gs')
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
    file.write("'open ./lwdn.ctl'\n")
    file.write("'set lat ' lat1+0.002 ' ' lat2-0.002\n")
    file.write("'set lon ' lon1+0.002 ' ' lon2-0.002\n")
    file.write("'set undef -9999.'\n")
    file.write("'define test=re(soil.1,0.004,0.004)'\n")
    file.write("'set gxout fwrite'\n")
    file.write("'set fwrite %s'\n" % outfn)
    file.write("'d test'\n")
    file.write("'disable fwrite'\n")
    file.close()    
    
    outfn = os.path.join(base,'CALC_RNET','insol_T%03d.dat' % tile)
    outfn = os.path.join(base,'insol_T%03d.dat' % tile)

def create_scons():
    of = open("./SConstruct",'w')    
    of.write("#!/usr/bin/env python2\n")
    of.write("\n")
    of.write("env = Environment(LINK='gfortran')\n")
    of.write("# The next line of code is an array of the source files names used in the program.\n")
    of.write("# The next line is the actual code that links the executable. env.Program is generates an executable.\n")
    of.write("final_rnet = env.Program(target='final_rnet', source= ['final_rnet.f90'])\n")
    of.close()
    
def gen_fortran_cubist_model(ifile,ofile): 
    f = open(ifile,'r')
    all_lines = f.readlines()
    f.close()
    of = open(ofile,'w')

    of.write("program compute\n")
    of.write("\n")
    of.write("integer, parameter :: dx=3750, dy=3750\n")
    of.write("real :: rnet(dx,dy), lst2(dx,dy)\n")
    of.write("real :: lwdn(dx,dy), insol(dx,dy), albedo(dx,dy)\n")
    of.write("real :: latn(dx,dy)\n")
    of.write("integer :: iyear\n")
    of.write("integer :: p1, p2, f1, f2\n")
    of.write("real :: pva1, pval2, fval1, fval2\n")
    of.write("character(len=400) :: lst2_fn, lwdn_fn, albedo_fn, insol_fn, out_fn\n")
    of.write("integer :: corrddd, insolddd, iday, yyyyddd, riseddd\n")
    of.write("real :: fc(dx,dy)\n")
    of.write("real :: gsol\n")
    of.write("real :: corr(dx,dy)\n")
    of.write("\n")
    of.write("call getarg(1,lst2_fn)\n")
    of.write("call getarg(2,lwdn_fn)\n")
    of.write("call getarg(3,albedo_fn)\n")
    of.write("call getarg(4,insol_fn)\n")
    of.write("call getarg(5,out_fn)\n")
    of.write("\n")
    of.write("open(11,file=lst2_fn,form='unformatted',access='direct',recl=dx*dy*4)\n")
    of.write("open(12,file=lwdn_fn,form='unformatted',access='direct',recl=dx*dy*4)\n")
    of.write("open(13,file=albedo_fn,form='unformatted',access='direct',recl=dx*dy*4)\n")
    of.write("open(14,file=insol_fn,form='unformatted',access='direct',recl=dx*dy*4)\n")
    of.write("read(11,rec=1) lst2\n")
    of.write("read(12,rec=1) lwdn\n")
    of.write("read(13,rec=1) albedo\n")
    of.write("read(14,rec=1) insol\n")
    of.write("close(11)\n")
    of.write("close(12)\n")
    of.write("close(13)\n")
    of.write("close(14)\n")
    of.write("\n")
    of.write("latn(:,:) = -9999.\n")
    of.write("do j = 1, dy\n")
    of.write("do i = 1, dx\n")
    of.write("\n")
    of.write(" if (lwdn(i,j).ne.-9999.and.albedo(i,j).ne.-9999..and.lst2(i,j).ne.-9999..and.insol(i,j).ne.-9999.) then\n")
    count=0
    for line in all_lines:
        chars = line.split()
        condition = chars[0].split('=')
        count=count+1
        if condition[0] == 'conds':
            var1 = condition[1].split('"')
            nconds = var1[1]
            for x in range(0,int(nconds)+1):
                c1 = all_lines[count+x].split()
                if x < int(nconds):
                    cvar = c1[1].split('"')
                    cval = c1[2].split('"')
                    cond = c1[3].split('"')
                    str1='   if ('+str(cvar[1])+'(i,j) '+str(cond[1])+' '+str(cval[1])+') then\n'
                    of.write(str1)
                if x == int(nconds):
                    print c1
                    a0=c1[0].split('"')
                    print str(a0[1])
                    str2='    latn(i,j) = '+str(a0[1])
                    for y in range(1,len(c1),2):
                        print y, len(c1)
                        a1=c1[y].split('"') 
                        a2=c1[y+1].split('"')
                        str2=str2+'+('+str(a2[1])+'*'+str(a1[1])+'(i,j)'+')'
                        print str2
                    str2=str2+'\n'
                    of.write(str2)
                    for z in range(0,int(nconds)):
                        of.write('   endif\n')

    of.write(" endif\n")
#    of.write(" if (latn(i,j).lt.0.0.and.latn(i,j).ne.-9999.) then\n") 
#    of.write("  latn(i,j) = 0.0\n")
#    of.write(" endif\n")
    of.write("enddo\n")
    of.write("enddo\n")
    of.write("\n")
    of.write("open(10,file=out_fn,form='unformatted',access='direct',recl=dx*dy*4)\n")
    of.write("write(10,rec=1) latn\n")
    of.write("close(10)\n")
    of.write("\n")
    of.write("end program\n")
    of.close()

def get_fstem_names(namefn):
    of = open(namefn,'w')
    of.write("rnet.\n")
    of.write("\n")
    of.write("rnet:		continuous.\n")
    of.write("albedo:         continuous.\n")
    of.write("insol:          continuous.\n")
    of.write("lwdn:           continuous.\n")
    of.write("lst2:	  	continuous.\n") 
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
            print 'rule2use:%s' % rule
            # run the rule on the dataset
            rule2use=eval('(%s)'% rule)
        
            formulaSplit = b[thenIndex[i]+1][0].split(' ')
            for k in xrange(len(formulaSplit)/3):
                if k == 0:
                    formula = '%s' % formulaSplit[2]
                else:
                    formSub = '%s %s*outDF[rule2use].%s' % (formulaSplit[(k*3)],formulaSplit[(k*3)+1],formulaSplit[(k*3)+2])
                    formula = '%s%s' % (formula,formSub)
            print 'formula:rnet = %s' % formula
            lstOut[np.where(rule2use)] = eval('(%s)' % formula)
            lstOut[np.where(mask)] = -9999.
    return lstOut    
    
def processData(year,doy,tile):
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
    writeCTL(tile,year,doy)
    #========process insol==================================================
    tiles_path = os.path.join(calc_rnet_path,'tiles')
    if not os.path.exists(tiles_path):
        os.makedirs(tiles_path) 
        
    insol_fn = os.path.join(tiles_path,'INSOL_%03d_%s.dat' % (tile,date))
    write_insol_viirs(insol_fn)
    out = subprocess.check_output("opengrads -blxc 'run ./agg_insol.gs %3.2f %3.2f  %3.2f  %3.2f'" 
                                  % (LLlat, URlat, LLlon, URlon), shell=True)

    read_data = np.fromfile(insol_fn, dtype=np.float32)
    insol = np.flipud(read_data.reshape([halfdeg_shape[0],halfdeg_shape[1]]))
    insol = np.reshape(insol,[halfdeg_sizeArr])
    
    viirs_tile_path = os.path.join(calc_rnet_path,'viirs','T%03d' % tile)
    if not os.path.exists(viirs_tile_path):
        os.makedirs(viirs_tile_path) 
        
    insol_viirs_fn = os.path.join(viirs_tile_path,'INSOL_%03d_%s.dat' % (tile,date))
    write_agg_insol_viirs(insol_viirs_fn)
    out = subprocess.check_output("opengrads -blxc 'run ./agg_insol_viirs.gs %3.2f %3.2f  %3.2f  %3.2f'" 
                                  % (LLlat, URlat, LLlon, URlon), shell=True)
    insol_viirs = np.fromfile(insol_viirs_fn, dtype=np.float32)
#    insol_viirs = np.flipud(insol_viirs.reshape([ALEXI_shape[0],ALEXI_shape[1]]))
#    insol_viirs = np.reshape(insol_viirs,[ALEXI_sizeArr])
    
    #======process RNET======================================================
    rnet_fn = os.path.join(tiles_path,'RNET_%03d_%s.dat' % (tile,date))
    write_agg_rnet(rnet_fn)
    
    out = subprocess.check_output("opengrads -blxc 'run ./agg_rnet.gs %3.2f %3.2f  %3.2f  %3.2f'" 
                                  % (LLlat, URlat, LLlon, URlon), shell=True)
    read_data = np.fromfile(rnet_fn, dtype=np.float32)
    rnet = np.flipud(read_data.reshape([halfdeg_shape[0],halfdeg_shape[1]]))
    rnet = np.reshape(rnet,[halfdeg_sizeArr])    
    #======process albedo====================================================
    albedo_fn = os.path.join(tiles_path,'ALBEDO_%03d_%s.dat' % (tile,date))
    write_agg_albedo(albedo_fn)
    
    out = subprocess.check_output("opengrads -blxc 'run ./agg_albedo.gs %3.2f %3.2f  %3.2f  %3.2f'" 
                                  % (LLlat, URlat, LLlon, URlon), shell=True)
    read_data = np.fromfile(albedo_fn, dtype=np.float32)
    albedo = np.flipud(read_data.reshape([halfdeg_shape[0],halfdeg_shape[1]]))
    albedo = np.reshape(albedo,[halfdeg_sizeArr])
    #=====process LST2=======================================================
    lst_fn = os.path.join(tiles_path,'LST2_%03d_%s.dat' % (tile,date))
    write_agg_lst2(lst_fn)
    
    out = subprocess.check_output("opengrads -blxc 'run ./agg_lst2.gs %3.2f %3.2f  %3.2f  %3.2f'" 
                                  % (LLlat, URlat, LLlon, URlon), shell=True)
    read_data = np.fromfile(lst_fn, dtype=np.float32)
    lst = np.flipud(read_data.reshape([halfdeg_shape[0],halfdeg_shape[1]]))
    lst = np.reshape(lst,[halfdeg_sizeArr])
    #====process LWDN========================================================
    lwdn_fn = os.path.join(tiles_path,'LWDN_%03d_%s.dat' % (tile,date))
    write_agg_lwdn(lwdn_fn)
    out = subprocess.check_output("opengrads -blxc 'run ./agg_lwdn.gs %3.2f %3.2f  %3.2f  %3.2f'" 
                                  % (LLlat, URlat, LLlon, URlon), shell=True)
    
    read_data = np.fromfile(lwdn_fn, dtype=np.float32)
    lwdn = np.flipud(read_data.reshape([halfdeg_shape[0],halfdeg_shape[1]]))
    lwdn = np.reshape(lwdn,[halfdeg_sizeArr])

    lwdn_viirs_fn = os.path.join(viirs_tile_path,'LWDN_%03d_%s.dat' % (tile,date))
    write_agg_lwdn_viirs(lwdn_viirs_fn)
    out = subprocess.check_output("opengrads -blxc 'run ./agg_lwdn_viirs.gs %3.2f %3.2f  %3.2f  %3.2f'" 
                                  % (LLlat, URlat, LLlon, URlon), shell=True)
    
    lwdn_viirs = np.fromfile(lwdn_viirs_fn, dtype=np.float32)
#    lwdn_viirs = np.flipud(lwdn_viirs.reshape([ALEXI_shape[0],ALEXI_shape[1]]))
#    lwdn_viirs = np.reshape(lwdn_viirs,[ALEXI_sizeArr])
    
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
    get_fstem_names(file_names)
    
    #====run cubist======================================
    print("running cubist...")
    cubist_name = os.path.join(calc_rnet_tile_ctl,'rnet')
    out = subprocess.check_output("cubist -f %s -u -a -r 20" % cubist_name, shell=True)
    lst2 = np.fromfile('./current_lst2.dat', dtype=np.float32)
    albedo = np.fromfile('./current_albedo.dat', dtype=np.float32)
    cubDict = {'albedo':albedo, 'insol':insol_viirs, 'lwdn': lwdn_viirs, 'lst2':lst2}
    cubDF = pd.DataFrame.from_dict(cubDict)
    rnet_out = readCubistOut(out,cubDF)
    rnet_out = np.reshape(rnet_out, [3750,3750])
    #=====generate and compile fortran code with results from cubist==========
#    file_model = os.path.join(calc_rnet_tile_ctl,'rnet.model')
#    final_rnet_fortran = './final_rnet.f90'
#    gen_fortran_cubist_model(file_model,final_rnet_fortran)
#    create_scons()
#    subprocess.check_output("scons",shell=True)
    #=======run final_rnet===================
    rnet_tile = os.path.join(tile_base_path,'T%03d' % tile)
    if not os.path.exists(rnet_tile):
        os.makedirs(rnet_tile)
    finalrnet_fn = os.path.join(rnet_tile,'FINAL_RNET_%s_T%03d.dat' % (date,tile))
#    cmd = "./final_rnet %s %s %s %s %s" % (lst_fn,lwdn_fn,
#                                albedo_fn,insol_fn,finalrnet_fn)
#    print cmd
#    subprocess.check_output("./final_rnet %s %s %s %s %s" % ('./current_lst2.dat',lwdn_viirs_fn,
#                                './current_albedo.dat',insol_viirs_fn,finalrnet_fn), shell=True)
    rnet_out = np.array(rnet_out,dtype='Float32')
    rnet_out.tofile(finalrnet_fn)
    convertBin2tif(finalrnet_fn,inUL,ALEXI_shape,ALEXI_res)
#    gzipped(finalrnet_fn)

tile = 63
year = 2015
doy = 221
processData(year,doy,tile)