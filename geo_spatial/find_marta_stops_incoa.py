# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 15:53:21 2018

@author: hli378
"""

import pandas as pd
import numpy as np
from geopandas import GeoDataFrame
from shapely.geometry import Point, LineString
import os
import geopandas as gpd
from geopy.distance import vincenty

def make_transit_stops(input_folder,output_name):
    os.chdir(input_folder)
    crs = {'init' :'epsg:4326'} # http://spatialreference.org/ref/epsg/wgs-84/

    df=pd.read_csv('stops.txt',header=0)
    geometry = [Point(xy) for xy in zip(df.stop_lon, df.stop_lat)]
    df_out = GeoDataFrame(df, geometry=geometry)
    df_out.crs=crs
    df_out['stop_id']=df['stop_id']
    df_out['stop_name']=df['stop_name']
    df_out.to_file(output_name, driver='ESRI Shapefile')
    return df_out

os.chdir(r'E:\small projects\gtfs 10312018')
input_folder=r'E:\small projects\gtfs 10312018\gtfs'
stops=make_transit_stops(input_folder,'gtfs_stops.shp')

coa=gpd.read_file(r'E:\small projects\gtfs 10312018\Neighborhood_Planning_Units_2015\Neighborhood_Planning_Units_2015.shp')

stops_in_coa = gpd.sjoin(stops, coa, how="inner", op='intersects')
print(len(stops_in_coa))
print(len(stops))
print(len(stops.drop_duplicates('stop_id')))
print(len(stops_in_coa.drop_duplicates('stop_id')))

'''
Based on the latest MARTA GTFS data (released on Oct 8), 3888 out of 9288 marta stops/stations are in the city of atlanta 
'''