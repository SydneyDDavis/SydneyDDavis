# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 20:58:41 2020

@author: sdavis
"""
import h5py
import csv
import numpy as np
import os
import gdal, osr
import matplotlib.pyplot as plt
import sys
from math import floor
import time
import warnings
import pandas as pd
from multiprocessing import Pool
import functools
import seaborn as sns
from functools import partial 

def array2raster(newRasterfn,rasterOrigin,pixelWidth,pixelHeight,array,epsg):
    cols = array.shape[1]
    rows = array.shape[0]
    originX = rasterOrigin[0]
    originY = rasterOrigin[1]
    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(epsg)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()

def stack_images(tif_files): 
    #takes a list of tif files and stacks them all up 
    #Get extents of all images
    min_x,min_y,max_x,max_y,bands = get_map_extents(tif_files)
    #Creates an empty image
    full_extent = np.zeros((int(np.ceil(max_y-min_y)),int(np.ceil(max_x-min_x)),int(bands)),dtype=np.float)
    file_counter=0
    for tif_file in tif_files:
               
        data_layer, metadata = raster2array(tif_file)

        ul_x = metadata['ext_dict']['xMin']
        lr_x = metadata['ext_dict']['xMax']
        ul_y = metadata['ext_dict']['yMax']
        lr_y = metadata['ext_dict']['yMin']
        bands = metadata['bands']
        
        rows = int(ul_y-lr_y);
        columns = int(lr_x-ul_x);
        start_index_x = int(np.ceil(abs(max_y - ul_y)));
        start_index_y = int(np.ceil(abs(min_x - ul_x)));
        full_extent[start_index_x:start_index_x+rows,start_index_y:start_index_y+columns,file_counter:file_counter+bands] = data_layer
        #np.reshape(data_layer, (data_layer_shape[0], data_layer_shape[1]))
        file_counter += bands
    
    return full_extent, min_x,min_y,max_x,max_y

def raster2array(geotif_file):
    metadata = {}
    dataset = gdal.Open(geotif_file)
    metadata['array_rows'] = dataset.RasterYSize
    metadata['array_cols'] = dataset.RasterXSize
    metadata['bands'] = dataset.RasterCount
    metadata['driver'] = dataset.GetDriver().LongName
    metadata['projection'] = dataset.GetProjection()
    metadata['geotransform'] = dataset.GetGeoTransform()

    mapinfo = dataset.GetGeoTransform()
    metadata['pixelWidth'] = mapinfo[1]
    metadata['pixelHeight'] = mapinfo[5]

    metadata['ext_dict'] = {}
    metadata['ext_dict']['xMin'] = mapinfo[0]
    metadata['ext_dict']['xMax'] = mapinfo[0] + dataset.RasterXSize/mapinfo[1]
    metadata['ext_dict']['yMin'] = mapinfo[3] + dataset.RasterYSize/mapinfo[5]
    metadata['ext_dict']['yMax'] = mapinfo[3]

    metadata['extent'] = (metadata['ext_dict']['xMin'],metadata['ext_dict']['xMax'],
                          metadata['ext_dict']['yMin'],metadata['ext_dict']['yMax'])

    raster = dataset.GetRasterBand(1)
    metadata['noDataValue'] = raster.GetNoDataValue()
    metadata['scaleFactor'] = raster.GetScale()

    # band statistics
    metadata['bandstats'] = {} #make a nested dictionary to store band stats in same 
    stats = raster.GetStatistics(True,True)
    metadata['bandstats']['min'] = round(stats[0],2)
    metadata['bandstats']['max'] = round(stats[1],2)
    metadata['bandstats']['mean'] = round(stats[2],2)
    metadata['bandstats']['stdev'] = round(stats[3],2)

    array = np.empty((metadata['array_rows'],metadata['array_cols'],metadata['bands']), dtype=np.float)
    for band_counter in range(1,metadata['bands']+1):
        array[:,:,band_counter-1] = dataset.GetRasterBand(band_counter).ReadAsArray(0,0,metadata['array_cols'],metadata['array_rows']).astype(np.float)
    
    array[array==int(metadata['noDataValue'])]=np.nan
    array = array/metadata['scaleFactor']
    return array, metadata

def get_map_extents(files):
    
    max_x=0;
    min_x=1000000;
    max_y=0;
    min_y=9000000;
    band_counter = 0
    filename, file_extension = os.path.splitext(files[0])
    
    for file in files:
        
        if file_extension == '.h5':
            ul_x,ul_y,lr_x,lr_y = get_image_extent_h5(file)
        elif file_extension == '.tif':
            array, metadata = raster2array(file)
            ul_x = metadata['ext_dict']['xMin']
            lr_x = metadata['ext_dict']['xMax']
            ul_y = metadata['ext_dict']['yMax']
            lr_y = metadata['ext_dict']['yMin']
            num_bands = metadata['bands']

        if ul_x < min_x:
            min_x = ul_x
        if ul_y > max_y:
            max_y = ul_y
        if lr_x > max_x:
            max_x = lr_x
        if lr_y < min_y:
            min_y = lr_y   
        band_counter += num_bands

    return min_x,min_y,max_x,max_y,band_counter
            
def matchingFiles(coverage_directory, data_product_directory, data_product_file): 
    tif_files_list = []
    CoverageFiles = os.listdir(coverage_directory)
    CoverageTifFiles = [i for i in CoverageFiles if i.endswith('.tif')]
    for CoverageTifFile in CoverageTifFiles:
        coverage_tif_file_split = CoverageTifFile.split('_')
        site_coverage = coverage_tif_file_split[1]
        for root, dirs, files in os.walk(data_product_directory):
            for fname in files:  
                root = root.replace('\\', '/')
                root_split = root.split('/')
                site_data_product = root_split[4].split('_')[1]
                if site_coverage == site_data_product and fname == data_product_file:
                    tif_files_list.append([os.path.join(coverage_directory,CoverageTifFile), os.path.join(root,fname)])
    return tif_files_list


def percent_covered(matchingFiles):
    firstFile= matchingFiles[0].split('/')[3]
    site_name = firstFile.split('_')[1]
    print("Site name:", site_name)
    full_extent, min_x,min_y,max_x,max_y = stack_images(matchingFiles) 
    full_extent[np.isnan(full_extent)] = 0 
    reference_area = np.zeros_like(full_extent[:,:,0])
    reference_area[full_extent[:,:,1]!=0]=1
    np.sum(reference_area)
    reference_coverage_area = np.multiply(reference_area, full_extent[:,:,0])
    AreaCoveredDataProduct_Site = np.sum(reference_coverage_area) / np.sum(full_extent[:,:,0])
    return site_name, AreaCoveredDataProduct_Site

def weather_percent_covered(matchingFiles):
      firstFile= matchingFiles[0].split('/')[3]
      siteName = firstFile.split('_')[1]
      full_extent, min_x,min_y,max_x,max_y = stack_images(matchingFiles) 
      full_extent[np.isnan(full_extent)] = 0 
      referenceAreaRed = np.zeros_like(full_extent[:,:,0])
      referenceAreaYellow = np.zeros_like(full_extent[:,:,0])
      referenceAreaGreen = np.zeros_like(full_extent[:,:,0])
      referenceAreaRed[full_extent[:,:,2]==23]=1
      referenceAreaYellow[full_extent[:,:,3]==6]=1
      referenceAreaGreen[full_extent[:,:,1]==23]=1
      np.sum(referenceAreaRed) 
      np.sum(referenceAreaYellow) 
      np.sum(referenceAreaGreen)
      referenceAreaRed = np.multiply(referenceAreaRed, full_extent[:,:,0])
      referenceAreaYellow = np.multiply(referenceAreaYellow, full_extent[:,:,0])
      referenceAreaGreen = np.multiply(referenceAreaGreen, full_extent[:,:,0])
      pixelsRed = np.sum(referenceAreaRed)
      pixelsYellow = np.sum(referenceAreaYellow)
      pixelsGreen = np.sum(referenceAreaGreen)
      pixelsSummed = np.sum([pixelsYellow,pixelsGreen,pixelsRed],
                             dtype='float64')
      totalPixels = np.sum(full_extent[:,:,0])
      pixelsMissed = (totalPixels - pixelsSummed)
      areaGreen = np.sum(referenceAreaGreen) / np.sum(full_extent[:,:,0])
      areaRed = np.sum(referenceAreaRed) / np.sum(full_extent[:,:,0])
      areaYellow = np.sum(referenceAreaYellow) / np.sum(full_extent[:,:,0])
      return [siteName, areaGreen, areaRed, areaYellow, pixelsGreen, pixelsRed, 
              pixelsYellow, totalPixels, pixelsSummed, pixelsMissed] 
  
def lidarUncertainty(matchingFiles, threshold):
      firstFile = matchingFiles[0]
      firstPath = firstFile.split('/')[3]
      siteName = firstPath.split('_')[1]
      full_extent, min_x,min_y,max_x,max_y = stack_images(matchingFiles)
      full_extent[np.isnan(full_extent)] = 0
      uncertaintyReferenceArea = np.zeros_like(full_extent[:,:,0])
      uncertaintyReferenceArea[full_extent[:,:,1]<threshold]=1
      thresholdLayer = np.multiply(uncertaintyReferenceArea, full_extent[:,:,0])
      sumThreshLayer = np.sum(thresholdLayer)
      sumFullExtent = np.sum(full_extent[:,:,0])
      percentBelow = sumThreshLayer/sumFullExtent
      return [siteName, percentBelow, sumThreshLayer, sumFullExtent] 
    
if __name__ == '__main__': 
    print('Starting program')
    print('Before the P1FB pool')
    weatherCoverageFilesP1FB = matchingFiles('D:/Coverage/P1FB/', 'D:/2019/FullSite/', 'Weather_Quality_Indicator.tif')
    pool = Pool(processes = 30)
    weatherOutputsP1FB = pool.map(weather_percent_covered, weatherCoverageFilesP1FB)    
    print('Done')
    
    weatherCoverageFilesTOS = matchingFiles('D:/Coverage/TOS/', 'D:/2019/FullSite/', 'Weather_Quality_Indicator.tif')
    print("Before the TOS pool")
    pool = Pool(processes = 30)
    weatherOutputsTOS = pool.map(weather_percent_covered, weatherCoverageFilesTOS)    
    print("Done")
    
    weatherCoverageFilesAirsheds = matchingFiles('D:/Coverage/Airsheds/', 'D:/2019/FullSite/', 'Weather_Quality_Indicator.tif')
    print("Before the Airsheds pool")
    pool = Pool(processes =30)
    weatherOutputsAirsheds = pool.map(weather_percent_covered, weatherCoverageFilesAirsheds)    
    print("Done")
    
    #P1FB lists for df
    siteP1FB = []
    areaGreenP1FB = []
    areaRedP1FB = []
    areaYellowP1FB = []
    boundaryP1FB = []
    yearP1FB = []
    for lst in weatherOutputsP1FB: 
        siteP1FB.append(lst[0]) 
        areaGreenP1FB.append(lst[1])
        areaRedP1FB.append(lst[2])
        areaYellowP1FB.append(lst[3])
        boundaryP1FB.append('P1FB')
        yearP1FB.append(2019)
        
    #TOS lists for df
    siteTOS = []
    areaGreenTOS = []
    areaRedTOS = []
    areaYellowTOS = []
    boundaryTOS = []
    yearTOS = []
    for lst in weatherOutputsTOS: 
        siteTOS.append(lst[0]) 
        areaGreenTOS.append(lst[1])
        areaRedTOS.append(lst[2])
        areaYellowTOS.append(lst[3])
        boundaryTOS.append('TOS')
        yearTOS.append(2019)
        
    #Airsheds list for df
    siteAirsheds = []
    areaGreenAirsheds = []
    areaRedAirsheds = []
    areaYellowAirsheds = []
    boundaryAirsheds = []
    yearAirsheds = []
    for lst in weatherOutputsAirsheds: 
        siteAirsheds.append(lst[0]) 
        areaGreenAirsheds.append(lst[1])
        areaRedAirsheds.append(lst[2])
        areaYellowAirsheds.append(lst[3])
        boundaryAirsheds.append('Airsheds')
        yearAirsheds.append(2019)    
        
    dfP1FB = pd.DataFrame(columns = ['Sites', 'Boundary', 'Year', 'Green Percent Covered', 
                                      'Red Percent Covered', 'Yellow Percent Covered'])     
    dfP1FB['Sites'] = siteP1FB 
    dfP1FB['Green Percent Covered'] = areaGreenP1FB
    dfP1FB['Red Percent Covered'] = areaRedP1FB 
    dfP1FB['Yellow Percent Covered'] = areaYellowP1FB
    dfP1FB['Year'] = yearP1FB 
    dfP1FB['Boundary'] = boundaryP1FB
    dfP1FB.set_index('Sites', inplace=True, drop=True)

    dfTOS = pd.DataFrame(columns = ['Sites', 'Boundary', 'Year', 'Green Percent Covered', 
                                      'Red Percent Covered', 'Yellow Percent Covered'])
    dfTOS['Sites'] = siteTOS
    dfTOS['Green Percent Covered'] = areaGreenTOS
    dfTOS['Red Percent Covered'] = areaRedTOS
    dfTOS['Yellow Percent Covered'] = areaYellowTOS
    dfTOS['Year'] = yearTOS
    dfTOS['Boundary'] = boundaryTOS
    dfTOS.set_index('Sites', inplace=True, drop=True)

    dfAirsheds = pd.DataFrame(columns = ['Sites', 'Boundary', 'Year', 'Green Percent Covered', 
                                      'Red Percent Covered', 'Yellow Percent Covered'])
    
    dfAirsheds['Sites'] = siteAirsheds 
    dfAirsheds['Green Percent Covered'] = areaGreenAirsheds
    dfAirsheds['Red Percent Covered'] = areaRedAirsheds
    dfAirsheds['Yellow Percent Covered'] = areaYellowAirsheds
    dfAirsheds['Boundary'] = boundaryAirsheds
    dfAirsheds['Year'] = yearAirsheds
    dfAirsheds.set_index('Sites', inplace=True, drop=True)

    dfWeather = pd.concat([dfP1FB, dfTOS, dfAirsheds])
    
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(20,10))
    for i, (name, group) in enumerate(dfWeather.groupby('Boundary')):
        axes[i].set_title(name)
        group.plot(kind="bar", y=['Green Percent Covered', 
                                  'Red Percent Covered', 
                                  'Yellow Percent Covered'], 
                                        ax=axes[i],
                                        legend=True, stacked=True, color=['g', 'r', 'y'])
    axes[i].set_xlabel("")
    axes[i].set_ylabel("Percent Covered")
    axes[i].set_xlabel("Sites")
    plt.tight_layout()

    plotP1FB = dfP1FB.loc[:, ['Green Percent Covered',
                              'Red Percent Covered', 
                              'Yellow Percent Covered']].plot(kind='bar', stacked=True, color=['g', 'r', 'y'])
    plt.title('Weather Percent Covered for P1FB Boundary') 
    
    plotAirsheds = dfAirsheds.loc[:, ['Green Percent Covered',
                                      'Red Percent Covered', 
                                      'Yellow Percent Covered']].plot(kind='bar', stacked=True, color=['g', 'r', 'y'])
    plt.title('Weather Percent Covered for Airsheds Boundary') 

    
    plotTOS = dfTOS.loc[:, ['Green Percent Covered',
                            'Red Percent Covered', 
                            'Yellow Percent Covered']].plot(kind='bar', stacked=True, color=['g', 'r', 'y'])
    plt.title('Weather Percent Covered for TOS Boundary') 


    #P1FB lists for pixel counts
    p1fbPixelSites = []
    p1fbPixelsGreen = []
    p1fbPixelsRed = []
    p1fbPixelsYellow = []
    p1fbPixelsTotal = []
    p1fbPixelsMissed = []
    p1fbPixelYear = []
    p1fbBoundaryName = []
    for lst in weatherOutputsP1FB: 
        p1fbPixelSites.append(lst[0])
        p1fbPixelsGreen.append(lst[4])
        p1fbPixelsRed.append(lst[5])
        p1fbPixelsYellow.append(lst[6])
        p1fbPixelsMissed.append(lst[9])
        p1fbPixelsTotal.append(lst[7])
        p1fbPixelYear.append(2019)
        p1fbBoundaryName.append('P1FB')
        
    dfPixelP1FB = pd.DataFrame(columns = ['Sites', 'Boundary', 'Year', 'Green Pixels', 
                                      'Red Pixels', 'Yellow Pixels', 'Total Pixels',
                                      'Pixels Missed'])
    
    dfPixelP1FB['Sites'] = p1fbPixelSites
    dfPixelP1FB['Boundary'] = p1fbBoundaryName
    dfPixelP1FB['Year'] = p1fbPixelYear
    dfPixelP1FB['Green Pixels'] = p1fbPixelsGreen
    dfPixelP1FB['Red Pixels'] = p1fbPixelsRed
    dfPixelP1FB['Yellow Pixels'] = p1fbPixelsYellow 
    dfPixelP1FB['Total Pixels'] = p1fbPixelsTotal 
    dfPixelP1FB['Pixels Missed'] = p1fbPixelsMissed
    
    p1fbSummedPixels = dfPixelP1FB.sum(axis=0)
    
    p1fbTotalGreenPix = p1fbSummedPixels['Green Pixels']
    p1fbTotalRedPix = p1fbSummedPixels['Red Pixels']
    p1fbTotalYellowPix = p1fbSummedPixels['Yellow Pixels']
    p1fbTotalMissedPix = p1fbSummedPixels['Pixels Missed']
    p1fbTotalPixs = p1fbSummedPixels['Total Pixels']
    
    p1fbPercentPixGreen = p1fbTotalGreenPix/p1fbTotalPixs
    p1fbPercentPixRed = p1fbTotalRedPix/p1fbTotalPixs
    p1fbPercentPixYellow = p1fbTotalYellowPix/p1fbTotalPixs
    p1fbPercentMissedPix = p1fbTotalMissedPix/p1fbTotalPixs
    
    p1fbTotalPixelData = {'Boundary': ['P1FB'], 
                              'Year': [2019],
                              'Percent Green Pixels': p1fbPercentPixGreen, 
                              'Percent Red Pixels': p1fbPercentPixRed,
                              'Percent Yellow Pixels': p1fbPercentPixYellow,
                              'Percent Missed Pixels': p1fbPercentMissedPix}

    dfTotalPixelP1FB = pd.DataFrame(p1fbTotalPixelData,
                                        columns = ['Boundary',
                                                   'Year',
                                                   'Percent Green Pixels',
                                                   'Percent Red Pixels',
                                                   'Percent Yellow Pixels',
                                                   'Percent Missed Pixels'])
    
    #TOS lists pixel counts
    tosPixelSites = []
    tosPixelsGreen = []
    tosPixelsRed = []
    tosPixelsYellow = []
    tosPixelsTotal = []
    tosPixelsMissed = []
    tosPixelYear = []
    tosBoundaryName = []
    for lst in weatherOutputsTOS: 
        tosPixelSites.append(lst[0])
        tosPixelsGreen.append(lst[4])
        tosPixelsRed.append(lst[5])
        tosPixelsYellow.append(lst[6])
        tosPixelsMissed.append(lst[9])
        tosPixelsTotal.append(lst[7])
        tosPixelYear.append(2019)
        tosBoundaryName.append('TOS')
        
    dfPixelTOS = pd.DataFrame(columns = ['Sites', 'Boundary', 'Year', 'Green Pixels', 
                                      'Red Pixels', 'Yellow Pixels', 'Total Pixels',
                                      'Pixels Missed'])
    
    dfPixelTOS['Sites'] = tosPixelSites
    dfPixelTOS['Boundary'] = tosBoundaryName
    dfPixelTOS['Year'] = tosPixelYear
    dfPixelTOS['Green Pixels'] = tosPixelsGreen
    dfPixelTOS['Red Pixels'] = tosPixelsRed
    dfPixelTOS['Yellow Pixels'] = tosPixelsYellow 
    dfPixelTOS['Total Pixels'] = tosPixelsTotal 
    dfPixelTOS['Pixels Missed'] = tosPixelsMissed
    
    tosSummedPixels = dfPixelTOS.sum(axis=0)
    
    tosTotalGreenPix = tosSummedPixels['Green Pixels']
    tosTotalRedPix = tosSummedPixels['Red Pixels']
    tosTotalYellowPix = tosSummedPixels['Yellow Pixels']
    tosTotalMissedPix = tosSummedPixels['Pixels Missed']
    tosTotalPixs = tosSummedPixels['Total Pixels']
    
    tosPercentPixGreen = tosTotalGreenPix/tosTotalPixs
    tosPercentPixRed = tosTotalRedPix/tosTotalPixs
    tosPercentPixYellow = tosTotalYellowPix/tosTotalPixs
    tosPercentMissedPix = tosTotalMissedPix/tosTotalPixs
    
    tosTotalPixelData = {'Boundary': ['TOS'], 
                              'Year': [2019],
                              'Percent Green Pixels': tosPercentPixGreen, 
                              'Percent Red Pixels': tosPercentPixRed,
                              'Percent Yellow Pixels': tosPercentPixYellow,
                              'Percent Missed Pixels': tosPercentMissedPix}

    dfTotalPixelTos = pd.DataFrame(tosTotalPixelData,
                                        columns = ['Boundary',
                                                   'Year',
                                                   'Percent Green Pixels',
                                                   'Percent Red Pixels',
                                                   'Percent Yellow Pixels',
                                                   'Percent Missed Pixels']) 
    
    #Airsheds list pixel counts
    airshedsPixelSites = []
    airshedsPixelsGreen = []
    airshedsPixelsRed = []
    airshedsPixelsYellow = []
    airshedsPixelsTotal = []
    airshedsPixelsMissed = []
    airshedsPixelYear = []
    airshedsBoundaryName = []
    for lst in weatherOutputsAirsheds: 
        airshedsPixelSites.append(lst[0])
        airshedsPixelsGreen.append(lst[4])
        airshedsPixelsRed.append(lst[5])
        airshedsPixelsYellow.append(lst[6])
        airshedsPixelsMissed.append(lst[9])
        airshedsPixelsTotal.append(lst[7])
        airshedsPixelYear.append(2019)
        airshedsBoundaryName.append('Airsheds')
        
    dfPixelAirsheds = pd.DataFrame(columns = ['Sites', 'Boundary', 'Year', 'Green Pixels', 
                                      'Red Pixels', 'Yellow Pixels', 'Total Pixels',
                                      'Pixels Missed'])
    
    dfPixelAirsheds['Sites'] = airshedsPixelSites
    dfPixelAirsheds['Boundary'] = airshedsBoundaryName
    dfPixelAirsheds['Year'] = airshedsPixelYear
    dfPixelAirsheds['Green Pixels'] = airshedsPixelsGreen
    dfPixelAirsheds['Red Pixels'] = airshedsPixelsRed
    dfPixelAirsheds['Yellow Pixels'] = airshedsPixelsYellow 
    dfPixelAirsheds['Total Pixels'] = airshedsPixelsTotal 
    dfPixelAirsheds['Pixels Missed'] = airshedsPixelsMissed
    
    airshedsSummedPixels = dfPixelAirsheds.sum(axis=0)

    airshedsTotalGreenPix = airshedsSummedPixels['Green Pixels']
    airshedsTotalRedPix = airshedsSummedPixels['Red Pixels']
    airshedsTotalYellowPix = airshedsSummedPixels['Yellow Pixels']
    airshedsTotalMissedPix = airshedsSummedPixels['Pixels Missed']
    airshedsTotalPixs = airshedsSummedPixels['Total Pixels']
    
    airshedsPercentPixGreen = airshedsTotalGreenPix/airshedsTotalPixs
    airshedsPercentPixRed = airshedsTotalRedPix/airshedsTotalPixs
    airshedsPercentPixYellow = airshedsTotalYellowPix/airshedsTotalPixs
    airshedsPercentMissedPix = airshedsTotalMissedPix/airshedsTotalPixs
    
    airshedsTotalPixelData = {'Boundary': ['Airsheds'], 
                              'Year': [2019],
                              'Percent Green Pixels': airshedsPercentPixGreen, 
                              'Percent Red Pixels': airshedsPercentPixRed,
                              'Percent Yellow Pixels': airshedsPercentPixYellow,
                              'Percent Missed Pixels': airshedsPercentMissedPix}

    dfTotalPixelAirsheds = pd.DataFrame(airshedsTotalPixelData,
                                         columns = ['Boundary',
                                                    'Year',
                                                    'Percent Green Pixels',
                                                    'Percent Red Pixels',
                                                    'Percent Yellow Pixels',
                                                    'Percent Missed Pixels'])
   
    dfTotalPixelBoundaries = pd.concat([dfTotalPixelAirsheds,
                                        dfTotalPixelP1FB,
                                        dfTotalPixelTos])
    dfTotalPixelBoundaries.set_index('Boundary', inplace=True, drop=True)
    
    plotTotalPixels = dfTotalPixelBoundaries.loc[:, ['Percent Green Pixels',
                                                    'Percent Red Pixels',
                                                    'Percent Yellow Pixels',
                                                    'Percent Missed Pixels']].plot(kind='bar', stacked=True, color=['g', 'r', 'y', 'b'])


    #Lidar products 
    #Horizontal Uncertainty P1FB
    print("Matching P1FB with Horizontal Uncertainty")
    lidarFilesHorzP1FB = matchingFiles('D:/Coverage/P1FB/', 'D:/2019/FullSite/', 'HorzUncertainty.tif')
    print("Before the pool")
    pool = Pool(processes = 30)
    func = partial(lidarUncertainty, threshold = 0.58)
    lidarOutputsHorzP1FB = pool.map(func, lidarFilesHorzP1FB)  
    
    #Vertical Uncertainty P1FB
    print("Matching P1FB with Vertical Uncertainty")
    lidarFilesVertP1FB = matchingFiles('D:/Coverage/P1FB/', 'D:/2019/FullSite/', 'VertUncertainty.tif')
    print("Before the pool")
    pool = Pool(processes = 30)
    func = partial(lidarUncertainty, threshold = 0.15)
    lidarOutputsVertP1FB = pool.map(func, lidarFilesVertP1FB)  
    
    #Horizontal Uncertainty TOS
    print("Matching TOS with Horizontal Uncertainty")
    lidarFilesHorzTOS = matchingFiles('D:/Coverage/TOS/', 'D:/2019/FullSite/', 'HorzUncertainty.tif')
    print("Before the pool")
    pool = Pool(processes = 30)
    func = partial(lidarUncertainty, threshold = 0.58)
    lidarOutputsHorzTOS = pool.map(func, lidarFilesHorzTOS)  
    
    #Vertical Uncertainty TOS
    print("Matching TOS with Vertical Uncertainty")
    lidarFilesVertTOS = matchingFiles('D:/Coverage/TOS/', 'D:/2019/FullSite/', 'VertUncertainty.tif')
    print("Before the pool")
    pool = Pool(processes = 30)
    func = partial(lidarUncertainty, threshold = 0.15)
    lidarOutputsVertTOS = pool.map(func, lidarFilesVertTOS) 
    
    #Horizontal Uncertainty Airsheds
    print("Matching Airsheds with Horizontal Uncertainty")
    lidarFilesHorzAirsheds = matchingFiles('D:/Coverage/Airsheds/', 'D:/2019/FullSite/', 'HorzUncertainty.tif')
    print("Before the pool")
    pool = Pool(processes = 30)
    func = partial(lidarUncertainty, threshold = 0.58)
    lidarOutputsHorzAirsheds = pool.map(func, lidarFilesHorzAirsheds) 
    
    #Vertical Uncertainty Airsheds
    print("Matching Airsheds with Vertical Uncertainty")
    lidarFilesVertAirsheds = matchingFiles('D:/Coverage/Airsheds/', 'D:/2019/FullSite/', 'VertUncertainty.tif')
    print("Before the pool")
    pool = Pool(processes = 30)
    func = partial(lidarUncertainty, threshold = 0.15)
    lidarOutputsVertAirsheds = pool.map(func, lidarFilesVertAirsheds) 
    
    #P1FB Longest_Triangular_Edge_All_Points
    print("Running P1FB with Longest_Triangular_Edge_All_Points.tif")
    lidarFilesTriAllP1FB = matchingFiles('D:/Coverage/P1FB/', 'D:/2019/FullSite/', 'Longest_Triangular_Edge_All_Points.tif')
    print("Before the pool")
    pool = Pool(processes = 30)
    func = partial(lidarUncertainty, threshold = 1.0)
    lidarOutputsTriAllP1FB = pool.map(func, lidarFilesTriAllP1FB)
    
    #P1FB Longest_Triangular_Edge_Ground_Points
    print("Running P1FB with Longest_Triangular_Edge_Ground_Points.tif")
    lidarFilesTriGroundP1FB = matchingFiles('D:/Coverage/P1FB/', 'D:/2019/FullSite/', 'Longest_Triangular_Edge_Ground_Points.tif')
    print("Before the pool")
    pool = Pool(processes = 30)
    func = partial(lidarUncertainty, threshold = 1.0)
    lidarOutputsTriGroundP1FB = pool.map(func, lidarFilesTriGroundP1FB)
    
    #TOS Longest_Triangular_Edge_All_Points
    print("Running TOS with Longest_Triangular_Edge_All_Points.tif")
    lidarFilesTriAllTOS = matchingFiles('D:/Coverage/TOS/', 'D:/2019/FullSite/', 'Longest_Triangular_Edge_All_Points.tif')
    print("Before the pool")
    pool = Pool(processes = 30)
    func = partial(lidarUncertainty, threshold = 1.0)
    lidarOutputsTriAllTOS = pool.map(func, lidarFilesTriAllTOS)
    
    #TOS Longest_Triangular_Edge_Ground_Points
    print("Running TOS with Longest_Triangular_Edge_Ground_Points.tif")
    lidarFilesTriGroundTOS = matchingFiles('D:/Coverage/TOS/', 'D:/2019/FullSite/', 'Longest_Triangular_Edge_Ground_Points.tif')
    print("Before the pool")
    pool = Pool(processes = 30)
    func = partial(lidarUncertainty, threshold = 1.0)
    lidarOutputsTriGroundTOS = pool.map(func, lidarFilesTriGroundTOS)
    
    #Airsheds Longest_Triangular_Edge_All_Points
    print("Running Airsheds with Longest_Triangular_Edge_All_Points.tif")
    lidarFilesTriAllAirsheds = matchingFiles('D:/Coverage/Airsheds/', 'D:/2019/FullSite/', 'Longest_Triangular_Edge_All_Points.tif')
    print("Before the pool")
    pool = Pool(processes = 30)
    func = partial(lidarUncertainty, threshold = 1.0)
    lidarOutputsTriAllAirsheds = pool.map(func, lidarFilesTriAllAirsheds)
    
    #Airsheds Longest_Triangular_Edge_Ground_Points
    print("Running Airsheds with Longest_Triangular_Edge_Ground_Points.tif")
    lidarFilesTriGroundAirsheds = matchingFiles('D:/Coverage/Airsheds/', 'D:/2019/FullSite/', 'Longest_Triangular_Edge_Ground_Points.tif')
    print("Before the pool")
    pool = Pool(processes = 30)
    func = partial(lidarUncertainty, threshold = 1.0)
    lidarOutputsTriGroundAirsheds = pool.map(func, lidarFilesTriGroundAirsheds)
    
    #P1FB Uncertainty Plot
    site = []
    percentBelow = []
    pixelsBelow = []
    totalPixels = []
    for lst in lidarOutputsHorzP1FB:
        site.append(lst[0])
        percentBelow.append(lst[1])
        pixelsBelow.append(lst[2])
        totalPixels.append(lst[3])
        
    sitesVert = []
    percentBelowVert = []
    pixelsBelowVert = []
    totalPixelsVert = []
    for lst in lidarOutputsVertP1FB:
        percentBelowVert.append(lst[1])
        pixelsBelowVert.append(lst[2])
        totalPixelsVert.append(lst[3])
        
    dfUncertP1FB = pd.DataFrame(columns = ['Sites', 
                                        'Percent Below Horizontal Uncertainty Threshold',
                                        'Percent Below Vertical Uncertainty Threshold',
                                        'Pixels Below Horizontal Uncertainty Threshold',
                                        'Pixels Below Vertical Uncertainty Threshold',
                                        'Total Pixels Horizontal',
                                        'Total Pixels Vertical'])
    
    dfUncertP1FB['Sites'] = site 
    dfUncertP1FB['Percent Below Horizontal Uncertainty Threshold'] = percentBelow
    dfUncertP1FB['Percent Below Vertical Uncertainty Threshold'] = percentBelowVert
    dfUncertP1FB['Pixels Below Horizontal Uncertainty Threshold'] = pixelsBelow
    dfUncertP1FB['Pixels Below Vertical Uncertainty Threshold'] = pixelsBelowVert
    dfUncertP1FB['Total Pixels Horizontal'] = totalPixels
    dfUncertP1FB['Total Pixels Vertical'] = totalPixelsVert
    dfUncertP1FB.set_index('Sites', inplace=True, drop=True)
    

    plotUncertP1FB = dfUncertP1FB.loc[:, ['Percent Below Horizontal Uncertainty Threshold',
                                          'Percent Below Vertical Uncertainty Threshold']].plot(kind='bar', figsize = (20, 10), color=['g', 'r'])
    plt.title('Percentage of Site Uncertainties within P1FB Boundary')
    plt.ylabel('Percent')
    
    #TOS Uncertainty Plot
    siteTOS = []
    percentBelowTOS = []
    pixelsBelowTOS = []
    totalPixelsTOS = []
    for lst in lidarOutputsHorzTOS:
        siteTOS.append(lst[0])
        percentBelowTOS.append(lst[1])
        pixelsBelowTOS.append(lst[2])
        totalPixelsTOS.append(lst[3])
        
    percentBelowVertTOS = []
    pixelsBelowVertTOS = []
    totalPixelsVertTOS = []
    for lst in lidarOutputsVertTOS:
        percentBelowVertTOS.append(lst[1])
        pixelsBelowVertTOS.append(lst[2])
        totalPixelsVertTOS.append(lst[3])
        
    dfUncertTOS = pd.DataFrame(columns = ['Sites', 
                                        'Percent Below Horizontal Uncertainty Threshold',
                                        'Percent Below Vertical Uncertainty Threshold',
                                        'Pixels Below Horizontal Uncertainty Threshold',
                                        'Pixels Below Vertical Uncertainty Threshold',
                                        'Total Pixels Horizontal',
                                        'Total Pixels Vertical'])
    
    dfUncertTOS['Sites'] = siteTOS
    dfUncertTOS['Percent Below Horizontal Uncertainty Threshold'] = percentBelowTOS
    dfUncertTOS['Percent Below Vertical Uncertainty Threshold'] = percentBelowVertTOS
    dfUncertTOS['Pixels Below Horizontal Uncertainty Threshold'] = pixelsBelowTOS
    dfUncertTOS['Pixels Below Vertical Uncertainty Threshold'] = pixelsBelowVertTOS
    dfUncertTOS['Total Pixels Horizontal'] = totalPixelsTOS
    dfUncertTOS['Total Pixels Vertical'] = totalPixelsVertTOS
    dfUncertTOS.set_index('Sites', inplace=True, drop=True)
    

    plotUncertTOS = dfUncertTOS.loc[:, ['Percent Below Horizontal Uncertainty Threshold',
                                        'Percent Below Vertical Uncertainty Threshold']].plot(kind='bar', figsize = (20, 10), color=['g', 'r'])
    plt.title('Percentage of Each Site in TOS Boundary Under Uncertainty Threshold')
    plt.ylabel('Percent')
    
    #Airsheds Uncertainty Plot
    siteAirsheds = []
    percentBelowAirsheds = []
    pixelsBelowAirsheds = []
    totalPixelsAirsheds = []
    for lst in lidarOutputsHorzAirsheds:
        siteAirsheds.append(lst[0])
        percentBelowAirsheds.append(lst[1])
        pixelsBelowAirsheds.append(lst[2])
        totalPixelsAirsheds.append(lst[3])
        
    percentBelowVertAirsheds = []
    pixelsBelowVertAirsheds = []
    totalPixelsVertAirsheds = []
    for lst in lidarOutputsVertAirsheds:
        percentBelowVertAirsheds.append(lst[1])
        pixelsBelowVertAirsheds.append(lst[2])
        totalPixelsVertAirsheds.append(lst[3])
        
    dfUncertAirsheds = pd.DataFrame(columns = ['Sites', 
                                        'Percent Below Horizontal Uncertainty Threshold',
                                        'Percent Below Vertical Uncertainty Threshold',
                                        'Pixels Below Horizontal Uncertainty Threshold',
                                        'Pixels Below Vertical Uncertainty Threshold',
                                        'Total Pixels Horizontal',
                                        'Total Pixels Vertical'])
    
    dfUncertAirsheds['Sites'] = siteAirsheds
    dfUncertAirsheds['Percent Below Horizontal Uncertainty Threshold'] = percentBelowAirsheds
    dfUncertAirsheds['Percent Below Vertical Uncertainty Threshold'] = percentBelowVertAirsheds
    dfUncertAirsheds['Pixels Below Horizontal Uncertainty Threshold'] = pixelsBelowAirsheds
    dfUncertAirsheds['Pixels Below Vertical Uncertainty Threshold'] = pixelsBelowVertAirsheds
    dfUncertAirsheds['Total Pixels Horizontal'] = totalPixelsAirsheds
    dfUncertAirsheds['Total Pixels Vertical'] = totalPixelsVertAirsheds
    dfUncertAirsheds.set_index('Sites', inplace=True, drop=True)
    

    plotUncertAirsheds = dfUncertAirsheds.loc[:, ['Percent Below Horizontal Uncertainty Threshold',
                                                  'Percent Below Vertical Uncertainty Threshold']].plot(kind='bar', figsize = (20, 10), color=['g', 'r'])
    plt.title('Percentage of Each Site in Airsheds Boundary Under Uncertainty Threshold')
    plt.ylabel('Percent')
    
    airshedsUncertSummedPixels = dfUncertAirsheds.sum(axis=0)
    tosUncertSummedPixels = dfUncertTOS.sum(axis=0)
    p1fbUncertSummedPixels = dfUncertP1FB.sum(axis=0)
    
    siteAirsheds = 'Airsheds'
    airshedsTotalUnderHorz = airshedsUncertSummedPixels['Pixels Below Horizontal Uncertainty Threshold']
    airshedsTotalUnderVert = airshedsUncertSummedPixels['Pixels Below Vertical Uncertainty Threshold'] 
    airshedsTotalHorz = airshedsUncertSummedPixels['Total Pixels Horizontal'] 
    airshedsTotalVert = airshedsUncertSummedPixels['Total Pixels Vertical'] 

    siteTOS = 'TOS'
    tosTotalUnderHorz = tosUncertSummedPixels['Pixels Below Horizontal Uncertainty Threshold'] 
    tosTotalUnderVert = tosUncertSummedPixels['Pixels Below Vertical Uncertainty Threshold'] 
    tosTotalHorz = tosUncertSummedPixels['Total Pixels Horizontal'] 
    tosTotalVert = tosUncertSummedPixels['Total Pixels Vertical'] 
    
    sitep1fb = 'P1FB'
    p1fbTotalUnderHorz = p1fbUncertSummedPixels['Pixels Below Horizontal Uncertainty Threshold'] 
    p1fbTotalUnderVert = p1fbUncertSummedPixels['Pixels Below Vertical Uncertainty Threshold'] 
    p1fbTotalHorz = p1fbUncertSummedPixels['Total Pixels Horizontal'] 
    p1fbTotalVert = p1fbUncertSummedPixels['Total Pixels Vertical'] 
    
    threeBoundaries = ['Airsheds', 'TOS', 'P1FB']
    percentUnderHorz = [airshedsTotalUnderHorz/airshedsTotalHorz,
                        tosTotalUnderHorz/tosTotalHorz,
                        p1fbTotalUnderHorz/p1fbTotalHorz]
    percentUnderVert = [airshedsTotalUnderVert/airshedsTotalVert,
                        tosTotalUnderVert/tosTotalVert, 
                        p1fbTotalUnderVert/p1fbTotalVert]
    
    dfTotalUncertPixsBoundaries = pd.DataFrame(columns = ['Boundaries',
                                                          'Percent Under Horizontal Threshold', 
                                                          'Percent Under Vertical Threshold'])
                                               
    dfTotalUncertPixsBoundaries['Boundaries'] = threeBoundaries
    dfTotalUncertPixsBoundaries['Percent Under Horizontal Threshold'] = percentUnderHorz
    dfTotalUncertPixsBoundaries['Percent Under Vertical Threshold'] = percentUnderVert
    dfTotalUncertPixsBoundaries.set_index('Boundaries', inplace=True, drop=True)
    
    plotTotalUncertPixBoundaries = dfTotalUncertPixsBoundaries.loc[:, ['Percent Under Horizontal Threshold',
                                                                       'Percent Under Vertical Threshold']].plot(kind='bar', figsize = (20, 10), 
                                                                                                                 color=['g', 'b']) 
                                                                                                                                                        
                                                                                                                                                        
    plt.title('Percent Pixels Under Horizontal and Vertical Threshold by Boundary in 2019')
    plt.ylabel('Percent')
    
    
    #Triangle Interpolation Plots
    #P1FB Triangle_All Plot
    siteP1FBTriAll = []
    percentBelowTriAll = []
    pixelsBelowTriAll = []
    totalPixelsTriAll = []
    for lst in lidarOutputsTriAllP1FB:
        siteP1FBTriAll.append(lst[0])
        percentBelowTriAll.append(lst[1])
        pixelsBelowTriAll.append(lst[2])
        totalPixelsTriAll.append(lst[3])
        
    percentBelowTriGround = []
    pixelsBelowTriGround = []
    totalPixelsTriGround = []
    for lst in lidarOutputsTriGroundP1FB:
        percentBelowTriGround.append(lst[1])
        pixelsBelowTriGround.append(lst[2])
        totalPixelsTriGround.append(lst[3])
        
    dfP1FBTri = pd.DataFrame(columns = ['Sites', 
                                        'Percent Below Triangle All Points Threshold',
                                        'Percent Below Triangle Ground Points Threshold',
                                        'Pixels Below Triangle All Points Threshold',
                                        'Pixels Below Triangle Ground Points Threshold',
                                        'Total Pixels Triangle All Points',
                                        'Total Pixels Triangle Ground Points'])
    
    dfP1FBTri['Sites'] = siteP1FBTriAll
    dfP1FBTri['Percent Below Triangle All Points Threshold'] = percentBelowTriAll
    dfP1FBTri['Percent Below Triangle Ground Points Threshold'] = percentBelowTriGround
    dfP1FBTri['Pixels Below Triangle All Points Threshold'] = pixelsBelowTriAll
    dfP1FBTri['Pixels Below Triangle Ground Points Threshold'] = pixelsBelowTriGround
    dfP1FBTri['Total Pixels Triangle All Points'] = totalPixelsTriAll
    dfP1FBTri['Total Pixels Triangle Ground Points'] = totalPixelsTriGround
    dfP1FBTri.set_index('Sites', inplace=True, drop=True)
    

    plotTriP1FB = dfP1FBTri.loc[:, ['Percent Below Triangle All Points Threshold',
                                    'Percent Below Triangle Ground Points Threshold']].plot(kind='bar', figsize = (20, 10), color=['g', 'r'])
   
    plt.title('Percent of Triangular Points Under Threshold by Site in P1FB')
    plt.ylabel('Percent')
    
    # TOS Interpolation Plots 
    siteTOSTriAll = []
    percentBelowTOSTriAll = []
    pixelsBelowTOSTriAll = []
    totalPixelsTOSTriAll = []
    for lst in lidarOutputsTriAllTOS:
        siteTOSTriAll.append(lst[0])
        percentBelowTOSTriAll.append(lst[1])
        pixelsBelowTOSTriAll.append(lst[2])
        totalPixelsTOSTriAll.append(lst[3])
        
    percentBelowTOSTriGround = []
    pixelsBelowTOSTriGround = []
    totalPixelsTOSTriGround = []
    for lst in lidarOutputsTriGroundTOS:
        percentBelowTOSTriGround.append(lst[1])
        pixelsBelowTOSTriGround.append(lst[2])
        totalPixelsTOSTriGround.append(lst[3])
        
    dfTOSTri = pd.DataFrame(columns = ['Sites', 
                                        'Percent Below Triangle All Points Threshold',
                                        'Percent Below Triangle Ground Points Threshold',
                                        'Pixels Below Triangle All Points Threshold',
                                        'Pixels Below Triangle Ground Points Threshold',
                                        'Total Pixels Triangle All Points',
                                        'Total Pixels Triangle Ground Points'])
    
    dfTOSTri['Sites'] = siteTOSTriAll
    dfTOSTri['Percent Below Triangle All Points Threshold'] = percentBelowTOSTriAll
    dfTOSTri['Percent Below Triangle Ground Points Threshold'] = percentBelowTOSTriGround
    dfTOSTri['Pixels Below Triangle All Points Threshold'] = pixelsBelowTOSTriAll
    dfTOSTri['Pixels Below Triangle Ground Points Threshold'] = pixelsBelowTOSTriGround
    dfTOSTri['Total Pixels Triangle All Points'] = totalPixelsTOSTriAll
    dfTOSTri['Total Pixels Triangle Ground Points'] = totalPixelsTOSTriGround
    dfTOSTri.set_index('Sites', inplace=True, drop=True)
    

    plotTriTOS = dfTOSTri.loc[:, ['Percent Below Triangle All Points Threshold',
                                  'Percent Below Triangle Ground Points Threshold']].plot(kind='bar', figsize = (20, 10), color=['g', 'r'])
    
    plt.title('Percent of Triangular Points Under Threshold by Site in TOS')
    plt.ylabel('Percent')
    
    #Airsheds Interpolation Plots 
    siteAirshedsTriAll = []
    percentBelowAirshedsTriAll = []
    pixelsBelowAirshedsTriAll = []
    totalPixelsAirshedsTriAll = []
    for lst in lidarOutputsTriAllAirsheds:
        siteAirshedsTriAll.append(lst[0])
        percentBelowAirshedsTriAll.append(lst[1])
        pixelsBelowAirshedsTriAll.append(lst[2])
        totalPixelsAirshedsTriAll.append(lst[3])
        
    percentBelowAirshedsTriGround = []
    pixelsBelowAirshedsTriGround = []
    totalPixelsAirshedsTriGround = []
    for lst in lidarOutputsTriGroundAirsheds:
        percentBelowAirshedsTriGround.append(lst[1])
        pixelsBelowAirshedsTriGround.append(lst[2])
        totalPixelsAirshedsTriGround.append(lst[3])
        
    dfAirshedsTri = pd.DataFrame(columns = ['Sites', 
                                        'Percent Below Triangle All Points Threshold',
                                        'Percent Below Triangle Ground Points Threshold',
                                        'Pixels Below Triangle All Points Threshold',
                                        'Pixels Below Triangle Ground Points Threshold',
                                        'Total Pixels Triangle All Points',
                                        'Total Pixels Triangle Ground Points'])
    
    dfAirshedsTri['Sites'] = siteAirshedsTriAll
    dfAirshedsTri['Percent Below Triangle All Points Threshold'] = percentBelowAirshedsTriAll
    dfAirshedsTri['Percent Below Triangle Ground Points Threshold'] = percentBelowAirshedsTriGround
    dfAirshedsTri['Pixels Below Triangle All Points Threshold'] = pixelsBelowAirshedsTriAll
    dfAirshedsTri['Pixels Below Triangle Ground Points Threshold'] = pixelsBelowAirshedsTriGround
    dfAirshedsTri['Total Pixels Triangle All Points'] = totalPixelsAirshedsTriAll
    dfAirshedsTri['Total Pixels Triangle Ground Points'] = totalPixelsAirshedsTriGround
    dfAirshedsTri.set_index('Sites', inplace=True, drop=True)
    
    plotTriAirsheds = dfAirshedsTri.loc[:, ['Percent Below Triangle All Points Threshold',
                                            'Percent Below Triangle Ground Points Threshold']].plot(kind='bar', figsize = (20, 10), color=['g', 'r'])
    
    plt.title('Percent of Triangular Points Under Threshold by Site in Airsheds')
    plt.ylabel('Percent')
    
    airshedsTriSummed = dfAirshedsTri.sum(axis=0)
    tosTriSummed = dfTOSTri.sum(axis=0)
    p1fbTriSummed= dfP1FBTri.sum(axis=0)
    
    siteAirsheds = 'Airsheds'
    airshedsTotalUnderAll = airshedsTriSummed['Pixels Below Triangle All Points Threshold']
    airshedsTotalUnderGround = airshedsTriSummed['Pixels Below Triangle Ground Points Threshold'] 
    airshedsTotalAll = airshedsTriSummed['Total Pixels Triangle All Points'] 
    airshedsTotalGround = airshedsTriSummed['Total Pixels Triangle Ground Points'] 

    siteTOS = 'TOS'
    tosTotalUnderAll = tosTriSummed['Pixels Below Triangle All Points Threshold'] 
    tosTotalUnderGround = tosTriSummed['Pixels Below Triangle Ground Points Threshold'] 
    tosTotalAll = tosTriSummed['Total Pixels Triangle All Points'] 
    tosTotalGround = tosTriSummed['Total Pixels Triangle Ground Points'] 
    
    sitep1fb = 'P1FB'
    p1fbTotalUnderAll = p1fbTriSummed['Pixels Below Triangle All Points Threshold'] 
    p1fbTotalUnderGround = p1fbTriSummed['Pixels Below Triangle Ground Points Threshold'] 
    p1fbTotalAll = p1fbTriSummed['Total Pixels Triangle All Points'] 
    p1fbTotalGround = p1fbTriSummed['Total Pixels Triangle Ground Points'] 
    
    threeBoundariesTri = ['Airsheds', 'TOS', 'P1FB']
    
    percentUnderAll = [airshedsTotalUnderAll/airshedsTotalAll,
                        tosTotalUnderAll/tosTotalAll,
                        p1fbTotalUnderAll/p1fbTotalAll]
    
    percentUnderGround= [airshedsTotalUnderGround/airshedsTotalGround,
                        tosTotalUnderGround/tosTotalGround, 
                        p1fbTotalUnderGround/p1fbTotalGround]
    
    dfTotalTriBoundary = pd.DataFrame(columns = ['Boundaries',
                                                 'Percent Pixels Under Triangle All Points Threshold', 
                                                 'Percent Pixels Under Triangle Ground Points Threshold'])
                                               
    dfTotalTriBoundary['Boundaries'] = threeBoundariesTri
    dfTotalTriBoundary['Percent Pixels Under Triangle All Points Threshold'] = percentUnderAll
    dfTotalTriBoundary['Percent Pixels Under Triangle Ground Points Threshold'] = percentUnderGround
    dfTotalTriBoundary.set_index('Boundaries', inplace=True, drop=True)
    
    plotTotalTriBoundary = dfTotalTriBoundary.loc[:, ['Percent Pixels Under Triangle All Points Threshold',
                                                      'Percent Pixels Under Triangle Ground Points Threshold']].plot(kind='bar', figsize = (20, 10), 
                                                                                                                     color=['g', 'b']) 
                                                                                                                                                        
                                                                                                                                                        
    plt.title('Percent Pixels Under Triangle Point Threshold by Boundary in 2019')
    plt.ylabel('Percent Pixels')
    
    
        
    
    
    


    
    
    
    

    
