"""
JDPedit:Combo of Sadhana's (shapely function) and Elco's (the rest) script + change in the substation function for a wider net.
Extract specified keys + values from an OSM pbf-file. This script is based on the fetch.py, but contains improvements. 
Note: pick either the retrieve function using Shapely (if geometry needs to be recognized, e.g. for plotting using Matplotlib), or the retrieve function using pygeos (if geometry does not need to be recognized, e.g. if you want to make a geopackage and export)  
@Authors: Sadhana Nirandjan, Elco Koks, Ben Dickens - Institute for Environmental studies, VU University Amsterdam
"""

import os
import pandas as pd
import numpy as np
import geopandas
import geopandas as gpd
import pygeos
from osgeo import ogr,gdal
from tqdm import tqdm
#this file is used to overide the gdal one in conda env
gdal.SetConfigOption("OSM_CONFIG_FILE", os.path.join('..',"osmconf.ini"))
from pygeos import from_wkb
from shapely.wkb import loads

def query_b(geoType,keyCol,**valConstraint):
    """
    This function builds an SQL query from the values passed to the retrieve() function.
    Arguments:
         *geoType* : Type of geometry (osm layer) to search for.
         *keyCol* : A list of keys/columns that should be selected from the layer.
         ***valConstraint* : A dictionary of constraints for the values. e.g. WHERE 'value'>20 or 'value'='constraint'
    Returns:
        *string: : a SQL query string.
    """
    query = "SELECT " + "osm_id"
    for a in keyCol: query+= ","+ a  
    query += " FROM " + geoType + " WHERE "
    # If there are values in the dictionary, add constraint clauses
    if valConstraint: 
        for a in [*valConstraint]:
            # For each value of the key, add the constraint
            for b in valConstraint[a]: query += a + b
        query+= " AND "
    # Always ensures the first key/col provided is not Null.
    query+= ""+str(keyCol[0]) +" IS NOT NULL" 
    return query 

def retrieve(osm_path,geoType,keyCol,**valConstraint):
    """
    Function to extract specified geometry and keys/values from OpenStreetMap
    Arguments:
        *osm_path* (str) : file path to the .osm.pbf file of the region 
        for which we want to do the analysis.     
        *geoType* (str) : Type of Geometry to retrieve. e.g. lines, multipolygons, etc.
        *keyCol* (str or list): These keys will be returned as columns in the dataframe.
        ***valConstraint: A dictionary specifiying the value constraints.  
        A key can have multiple values (as a list) for more than one constraint for key/value.  
    Returns:
        *GeoDataFrame* : a geopandas GeoDataFrame with all columns, geometries, and constraints specified.    
    """
    # Get the OSM driver
    driver=ogr.GetDriverByName('OSM')
    
    # Open the OSM file and execute the SQL query
    data = driver.Open(osm_path)
    query = query_b(geoType,keyCol,**valConstraint)
    sql_lyr = data.ExecuteSQL(query)

    # Create an empty list to store features
    features =[]
    
    #cl = columns #changed by me?
    # Create a list of column names for the GeoDataFrame 
    cl = ['osm_id'] 
    for a in keyCol: 
        cl.append(a)
    
    # Loop through the SQL layer and extract features
    if data is not None:
        print('query is finished, lets start the loop')
        for feature in tqdm(sql_lyr,desc='extract'):
            try:
                # Check if the specified key column has a value
                if feature.GetField(keyCol[0]) is not None:
                    # Convert the geometry to a pygeos object
                    geom = pygeos.from_wkt(feature.geometry().ExportToWkt()) 
                    if geom is None:
                        continue
                    # field is a list to styore the feature, it will be the row in the dataframe.
                    field = []
                    # Loop through the specified columns and append the data to the list
                    for i in cl: field.append(feature.GetField(i))
                    # Append the geometry to the list field
                    field.append(geom)
                    # Append the feature data to the list of features   
                    features.append(field)
            except:
                print("WARNING: skipped OSM feature")   
    else:
        print("ERROR: Nonetype error when requesting SQL. Check required.")

    # Add the 'geometry' column name to the list of column names
    cl.append('geometry')                   
    if len(features) > 0:
        # Create a pandas DataFrame from the list of features and column names, 
        # then convert it to a geopandas GeoDataFrame
        return pd.DataFrame(features,columns=cl)
    else:
        print("WARNING: No features or No Memory. returning empty GeoDataFrame") 
        return pd.DataFrame(columns=['osm_id','geometry'])

def power_subs(osm_path):
    """
    Function to extract substations polygons from OpenStreetMap    
    Arguments:
        *osm_path* : file path to the .osm.pbf file of the region 
        for which we want to do the analysis.        
    Returns:
        *GeoDataFrame* : a geopandas GeoDataFrame with all unique substation polygons.    
    """
    return retrieve(osm_path, 'multipolygons',['power'],**{'power':["='substation'"]})

def power_point(osm_path):
    """
    Function to extract energy points from OpenStreetMap  
    Arguments:
        *osm_path* : file path to the .osm.pbf file of the region 
        for which we want to do the analysis.        
    Returns:
        *GeoDataFrame* : a geopandas GeoDataFrame with specified unique energy linestrings.
    """   
    df = retrieve(osm_path,'points',['other_tags']) 
    
    df = df.loc[(df.other_tags.str.contains('substation'))]  #keep rows containing power data       
    df = df.reset_index(drop=True).rename(columns={'other_tags': 'asset'}) 
            
    return df.reset_index(drop=True)

def retrieve_poly_subs (osm_path, w_list, b_list):
    """
    Function to extract electricity substation polygons from OpenStreetMap  
    Arguments:
        *osm_path* : file path to the .osm.pbf file of the region 
        for which we want to do the analysis.
        *w_list* :  white list of keywords to search in the other_tags columns
        *b_list* :  black list of keywords of rows that should not be selected  
    Returns:s
        *GeoDataFrame* : a geopandas GeoDataFrame with specified unique substation.
    """   
    df = retrieve(osm_path,'multipolygons',['other_tags'])
    df = df[df.other_tags.str.contains('substation', case=False, na=False)]
    df = df[~df.other_tags.str.contains('|'.join(b_list))]
    df['asset']  = 'substation' #specify row
    return df.reset_index(drop=True)

def retrieve_line_subs (osm_path, w_list, b_list):
    """
    Function to extract electricity substation polygons from OpenStreetMap  
    Arguments:
        *osm_path* : file path to the .osm.pbf file of the region 
        for which we want to do the analysis.
        *w_list* :  white list of keywords to search in the other_tags columns
        *b_list* :  black list of keywords of rows that should not be selected  
    Returns:
        *GeoDataFrame* : a geopandas GeoDataFrame with specified unique substation.
    """   
    df = retrieve(osm_path,'lines',['other_tags'])
    df = df[df.other_tags.str.contains('substation', case=False, na=False)]
    df = df[~df.other_tags.str.contains('|'.join(b_list))]
    df['asset']  = 'substation' #specify row
    return df.reset_index(drop=True)

def save_df_to_shp (dataframe, file_name, crs_gdf):
    """A function to save dataframe to shapefile

    Args:
        dataframe (df): a pandas geoserie with the assets
        file_name (str): the file path

    Returns:
        shp: a shapefile of the geoseries
    """    
    gdf = gpd.GeoDataFrame(dataframe)
    gdf.crs = crs_gdf
    return gdf.to_file(os.path.join('..', 'output', file_name))

#Not checked
def substation_transmission_test (osm_path):
    """
    Function to extract substations polygons from OpenStreetMap    
    Arguments:
        *osm_path* : file path to the .osm.pbf file of the region 
        for which we want to do the analysis.        
    Returns:
        *GeoDataFrame* : a geopandas GeoDataFrame with all unique substation polygons.    
    """
    return retrieve(osm_path, 'multipolygons',['substation'],**{'substation':["='transmission' or ", "'='distribution'"]})


#Not checked
def retrieve_point_subs (osm_path, w_list, b_list):
    """
    Function to extract electricity substation polygons from OpenStreetMap  
    Arguments:
        *osm_path* : file path to the .osm.pbf file of the region 
        for which we want to do the analysis.
        *w_list* :  white list of keywords to search in the other_tags columns
        *b_list* :  black list of keywords of rows that should not be selected  
    Returns:
        *GeoDataFrame* : a geopandas GeoDataFrame with specified unique substation.
    """   
    df = retrieve(osm_path,'point',['other_tags'])
    df = df[df.other_tags.str.contains('substation', case=False, na=False)]
    df = df[~df.other_tags.str.contains('|'.join(b_list))]
    df['asset']  = 'substation' #specify row
    return df.reset_index(drop=True)

#Not checked
def power_subs_point_test(osm_path):
    """
    Function to extract substations polygons from OpenStreetMap    
    Arguments:
        *osm_path* : file path to the .osm.pbf file of the region 
        for which we want to do the analysis.        
    Returns:
        *GeoDataFrame* : a geopandas GeoDataFrame with all unique substation polygons.    
    """
    return retrieve(osm_path, 'points',['power'],**{'power':["='substation'"]})

#Not checked
def fast_retrieve_poly_subs(osm_path, w_list):
    df = retrieve(osm_path,'multipolygons',['other_tags'])
    listed = df.other_tags.tolist()
    df["new"] = [w_list in n for n in listed]
    return df.reset_index(drop=True)

# ### Delete me please ###
# def retrieve_shapely(osm_path,geoType,keyCol,**valConstraint):
#     """
#     Function to extract specified geometry and keys/values from OpenStreetMap using Shapely
#     Arguments:
#         *osm_path* : file path to the .osm.pbf file of the region 
#         for which we want to do the analysis.     
#         *geoType* : Type of Geometry to retrieve. e.g. lines, multipolygons, etc.
#         *keyCol* : These keys will be returned as columns in the dataframe.
#         ***valConstraint: A dictionary specifiying the value constraints.  
#         A key can have multiple values (as a list) for more than one constraint for key/value.  
#     Returns:
#         *GeoDataFrame* : a geopandas GeoDataFrame with all columns, geometries, and constraints specified.    
#     """
#     driver=ogr.GetDriverByName('OSM')
#     data = driver.Open(osm_path)
#     query = query_b(geoType,keyCol,**valConstraint)
#     sql_lyr = data.ExecuteSQL(query)
#     features =[]
#     # cl = columns 
#     cl = ['osm_id'] 
#     for a in keyCol: cl.append(a)
#     if data is not None:
#         print('query is finished, lets start the loop')
#         for feature in tqdm(sql_lyr):
#             try:
#                 if feature.GetField(keyCol[0]) is not None:
#                     geom = loads(feature.geometry().ExportToWkb()) 
#                     if geom is None:
#                         continue
#                     # field will become a row in the dataframe.
#                     field = []
#                     for i in cl: field.append(feature.GetField(i))
#                     field.append(geom)   
#                     features.append(field)
#             except:
#                 print("WARNING: skipped OSM feature")   
#     else:
#         print("ERROR: Nonetype error when requesting SQL. Check required.")    
#     cl.append('geometry')                   
#     if len(features) > 0:
#         return geopandas.GeoDataFrame(features,columns=cl,crs={'init': 'epsg:4326'})
#     else:
#         print("WARNING: No features or No Memory. returning empty GeoDataFrame") 
#         return geopandas.GeoDataFrame(columns=['osm_id','geometry'],crs={'init': 'epsg:4326'})