import geopandas as gpd
import pandas as pd
import json
import os
import pygeos
import shapely
import rasterio
import json
import pyproj
import numpy as np
from pathlib import Path
from tqdm import tqdm

def split_geotiff (rasters_path, output_dir, width, height, pixel_size):
    """
    This function splits a given geotiff raster image into multiple tiles with a specified width and height.
    Also skips tiles with more than 10% of black pixels

    Parameters:
    rasters_path (str): File path to the input folder with the rasters.
    output_dir (str): File path to the output folder where the created tiles will be saved.
    width (int): Width of the tiles in pixels.
    height (int): Height of the tiles in pixels.
    pixel_size (float): Pixel size of the raster.

    Returns:
    None
    """

    files = os.listdir(rasters_path)

    # Filter for only the files that end with ".tif"
    tif_files = [file for file in files if file.endswith(".tif")]

    #raster id to be used in the output file name
    raster_id = 0

    # Iterate through all the raster files in the input directory if ends with "tif"
    for tif_file in tqdm(tif_files):
        raster_id+=1
        img_id = tif_file.split(".tif")[0]
        raster = os.path.join(rasters_path, tif_file)
        # Open the input raster file using rasterio.open() and read the raster image and metadata information
        with rasterio.open(raster) as src:
            out_image = src.read()
            out_meta = src.meta
            out_transform = src.transform
            
            # Array for the itermediate heights and widths of the tiles based on the specified width and height parameters
            heights = np.arange(0, out_image.shape[1], height)
            widths = np.arange(0, out_image.shape[2], width)
            
            # Extract the origin x (min) and y(max) of the raster image
            min_x = out_transform[2]
            max_y = out_transform[5]

            # Iterate through all the tiles and create a new tile image
            for x in range(len(heights)):
                for y in range(len(widths)):
                    # Skip the last value of array heights [x] and widths [y] to avoid [IndexError: index 62 is out of bounds for axis 0 with size 62] (if it is smaller than the specified height or width?)
                    if (x + 1 == len(heights)) or (y + 1 == len(widths)):
                        continue
                    else:
                        # Extract the new tile from the input raster using slice operation
                        new_tile = out_image[:, heights[x]:heights[x+1], widths[y]:widths[y+1]]
        
                        # skip tiles if percentage black pixels is higher than 10%
                        if np.count_nonzero((new_tile[0] == 0) & (new_tile[1] == 0) & (new_tile[2] == 0)) > 0.1*new_tile[0].size:
                            continue
                        else:
                        
                            # Calculate the transform for the new tile using rasterio.transform.from_origin() function: origin x (min) of the raster image + widths[y]*pixel_size, and y(max) - heights[x]*pixel_size
                            transform_new = rasterio.transform.from_origin(min_x + widths[y]*pixel_size, max_y - heights[x]*pixel_size, pixel_size, pixel_size)
                            
                            # Update the metadata information for the new tile
                            out_meta.update({"driver": "GTiff",
                                            "height": new_tile.shape[1],
                                            "width": new_tile.shape[2],
                                            "transform": transform_new})
                            
                            # Save the new tile to the output directory using rasterio.open() and dest.write() functions
                            with rasterio.open(os.path.join(output_dir, "{}_{}_{}_{}.tif".format(raster_id, img_id, heights[x], widths[y])),
                                            "w", compress='lzw', **out_meta) as dest:
                                dest.write(new_tile)
    return None

def reproject(df_ds, current_crs, approximate_crs):
    """Reproject a geodataframe of a shapefile from one CRS to raster (target) CRS.

    Args:
        df_ds (DataFrame): A geodataframe of a shapefile layer.
        current_crs (str): The EPSG code of the current CRS of the shapefile.
        approximate_crs (str): The EPSG code of the target CRS.

    Returns:
        DataFrame: A geodataframe of a shapefile layer with the new coordinate reference system.
    """

    # Extract the geometries from the geodataframe
    geometries = df_ds['geometry']
    
    # Extract the coordinates of the geometries
    coords = pygeos.get_coordinates(geometries)
    
    # Set up a coordinate transformer to transform from the current CRS to the target CRS
    transformer = pyproj.Transformer.from_crs(current_crs, approximate_crs, always_xy=True)
    
    # Transform the coordinates to the target CRS
    new_coords = transformer.transform(coords[:, 0], coords[:, 1])
    
    # Update the geometries in the original geodataframe with the transformed coordinates
    return pygeos.set_coordinates(geometries.copy(), np.array(new_coords).T)

def split_annotations_to_geojson(df, tiles_path, geojson_folder):
    """
    Split annotations data into the raster tiles and convert to the format of train-net.py: 
    Pygeos to GPD to geojson.

    Args:
    df (GeoDataFrame): Vector dataset containing features to split.
    tiles_path (str): Path to the directory where the raster tiles are stored
    geojson_folder (str): Path to the directory where the output geojson files will be saved
    
    Returns:
    None
    """
    # Create spatial index for faster query
    #spatial index are the bounding boxes of the asset geometries
    spatial_index = pygeos.STRtree(df.geometry)
    id_obj = 0
    
    # Get a list of all files in the directory
    files = os.listdir(tiles_path)

    # Filter for only the files that end with ".tif"
    tif_tiles = [file for file in files if file.endswith(".tif")]

    # Iterate through each raster tile
    for tile in tqdm(tif_tiles):
        input_file = os.path.join(tiles_path,tile)
        with rasterio.open(input_file) as src:
            out_image = src.read()
            out_meta = src.meta
        # geom is the bounding box of the raster tile
        geom = pygeos.box(*src.bounds)
        
        # Query overlapping geometries from geom (tile) and the asset (from the spatial index)
        check_overlaps = spatial_index.query(geom, predicate='intersects')
        print (check_overlaps)

        # Create new geojson file for each overlapping geometry
        if len(check_overlaps) > 0:            
            get_matches = df.iloc[check_overlaps]
            get_exact_overlap = pygeos.intersection(get_matches['geometry'].values, geom)
            df_objs = pd.DataFrame()
            for polygon in get_exact_overlap:
                #print (polygon)
                #print (type(polygon))
                id_obj += 1
                #convert pygeos geometry to shapely geometry
                shapely_geom = pygeos.to_shapely(polygon)
                #geometries_objects = pygeos.to_wkt(object)
                df_row = pd.DataFrame()
                df_row['properties'] = [{"id":"{}".format(id_obj),"building":"yes"}]
                #convert shapely geometry to geopandas geometry
                df_row['geometry'] = gpd.GeoSeries([shapely_geom])
                #print (df_row)
                df_objs = df_objs.append(df_row)
            df_objs = df_objs.append(df_row)
            gdf_obj = gpd.GeoDataFrame(df_objs, geometry='geometry', crs="epsg:28992")
            gdf_obj.set_geometry(col='geometry', inplace=True)
                
            img_id = tile.split(".tif")[0]
            gdf_obj.to_file(os.path.join(geojson_folder, "{}.geojson".format(img_id)), driver="GeoJSON")
    return None

# Pyhtonic version to check if the geojsons are correct:
def split_annotations_to_geojson_pythonic (df, tiles_path, geojson_folder):
    """
    Split annotations data into the raster tiles and convert to the format of train-net.py: 
    Pygeos to GPD to geojson.

    Args:
    df (GeoDataFrame): Vector dataset containing features to split.
    tiles_path (str): Path to the directory where the raster tiles are stored
    geojson_folder (str): Path to the directory where the output geojson files will be saved
    
    Returns:
    None
    """
    # Create spatial index for faster query
    spatial_index = pygeos.STRtree(df.geometry)
    
    # Iterate through each raster tile
    for tile in os.listdir(tiles_path):
        if not tile.endswith('.tif'):
            continue
        
        input_file = os.path.join(tiles_path, tile)
        with rasterio.open(input_file) as src:
            geom = pygeos.box(*src.bounds)
            overlaps = spatial_index.query(geom, predicate='intersects')
        
        # Create new geojson file for each overlapping geometry
        if len(overlaps) > 0:
            matches = df.iloc[overlaps]
            exact_overlap = pygeos.intersection(matches.geometry.values, geom)
            
            objs = []
            for polygon in exact_overlap:
                shapely_geom = pygeos.to_shapely(polygon)
                objs.append({'properties': {'id': len(objs) + 1, 'building': 'yes'},
                             'geometry': shapely_geom})
            gdf = gpd.GeoDataFrame(objs, geometry='geometry', crs=src.crs)
            img_id = tile.split(".tif")[0]
            gdf.to_file(os.path.join(geojson_folder, f"{img_id}.geojson"), driver="GeoJSON")