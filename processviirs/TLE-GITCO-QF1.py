#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 14:18:20 2016

@author: mschull
"""
import sys
sys.path.insert(0,"/data/smcd/mschull/code/python-rep") # import user defined modules
import numpy as np
import os
import h5py
import subprocess
from datetime import datetime,timedelta
import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEBase import MIMEBase
from email import Encoders
from email.mime.text import MIMEText



dataBase = os.path.join(os.sep,'data','smcd','mschull') #smcd
sdrGeoDataBase = os.path.join(dataBase,'data','S-NPP','VIIRS','GEO')

def getBits(num,offset,numBits):
    if offset == 0:
        bitsVal = int('{0:08b}'.format(num)[-numBits:],2)
    else:
        bitsVal =int('{0:08b}'.format(num)[-offset-numBits:-offset],2)
    return bitsVal

def findTLEuse(fn):
    fh5  = h5py.File(fn, 'r')
    nscans = fh5["/All_Data/VIIRS-IMG-GEO-TC_All/NumberOfScans"][0]
    QF1 = fh5["/All_Data/VIIRS-IMG-GEO-TC_All/QF1_SCAN_VIIRSSDRGEO"]
    sTime = fh5["/All_Data/VIIRS-IMG-GEO-TC_All/StartTime"]
    
    TLEflag = []
    scanTime =[]
    for i in xrange(nscans):
        TLEflag.append(getBits(QF1[i],0,2)>1)
        scanTime.append(sTime[i])
    
    if np.sum(TLEflag)>0:
        textOut =  "%s   possible TLE use:   Y    %d scans     <====Missing_Ephemperis  ?" % (fn.split('/')[-1],np.sum(TLEflag))
    else:
        textOut = "%s   possible TLE use:   N    %d scans" % (fn.split('/')[-1],np.sum(TLEflag))
    
    return textOut, scanTime,np.sum(TLEflag)

    
def writeTLEresults(date):
# SCDR-FILES
    output = subprocess.check_output('/data/starfs1/bin/scdr-files -t GITCO %s' % date,shell=True)
    batchFiles = output.split()
    results = []
    sTime = []
    numScans = []
    for i in xrange(len(batchFiles)):   
        text,scanTime,scans = findTLEuse(batchFiles[i])
        results.append(text)
        sTime.append(scanTime[0])
        numScans.append(scans)
    
    resultFile = open(os.path.join(sdrGeoDataBase,'TLE_usage%s.csv' % date),'wb')
    for item in results:
        resultFile.write("%s\n" % item)
    resultFile.close()
    
    return sTime,numScans

def emailTLEdata(fn,numScans):
    date = ((fn.split('/')[-1]).split('.')[0])[-10:]
    SUBJECT = "TLE usage for %s" % date
    
    msg = MIMEMultipart()
    msg['Subject'] = SUBJECT 
    msg['From'] = 'mitch.schull@noaa.gov'
    msg['To'] = 'mitch.schull@noaa.gov'
    
    text = " Hi Gary,\n\nHere is the TLE usage log from yesterday.  %d scans are using TLE data.\n\n" % numScans
    part1 = MIMEText(text,'plain')
    msg.attach(part1)
    part2 = MIMEBase('application', "octet-stream")
    part2.set_payload(open("%s" % fn, "rb").read())
    Encoders.encode_base64(part2)
    
    part2.add_header('Content-Disposition', 'attachment; filename="%s"' % fn.split('/')[-1])
    
    msg.attach(part2)
    signText = "\n\nPS I'm setting up a cron job that will send you this report everyday at 6 am.  Let me know if this irritates you.\n\nThanks!\nMitch"
    signedText = MIMEText(signText,'plain')
    msg.attach(signedText)

    server = smtplib.SMTP('localhost')
    server.sendmail(msg['From'], [msg['To']] + [msg['CC']], msg.as_string())

#year = 2016
#month = 3
#startDay = 11
#numDays = 3
#startTime = datetime(year,month,startDay)
#sTime = []
#for i in xrange(numDays):
#    nextDay= startTime+timedelta(days = i)
#    year = nextDay.year
#    month = nextDay.month
#    day = nextDay.day
#    date = '%d-%02d-%02d' % (year,month,day)
#    sTime,numScans = writeTLEresults(date)
#    filename = os.path.join(sdrGeoDataBase,'TLE_usage%s.csv' % date)
#    emailTLEdata(filename)

today = datetime.today()
nextDay= today+timedelta(days = -1.)
year = nextDay.year
month = nextDay.month
day = nextDay.day
date = '%d-%02d-%02d' % (year,month,day)
sTime,numScans = writeTLEresults(date)
filename = os.path.join(sdrGeoDataBase,'TLE_usage%s.csv' % date)
emailTLEdata(filename,np.sum(numScans))
