import os
import json
import geojson
import pandas as pd
from tqdm import tqdm
from osgeo import gdal
from sklearn.model_selection import train_test_split

def geojson_to_json_pix_coords(dataset_split, small_tiles_path, geojson_path, dataset_path):
    """
    Converts geojson annotations to JSON format with pixel coordinates.

    Args:
        dataset_split (list): List of image files in the dataset split:train,test or val
        small_tiles_path (str): Path to the directory containing the small tiles.tif.
        geojson_path (str): Path to the directory containing the geojson files of the annotations.
        dataset_path (str): Path to the datasets train, val or test.

    Returns:
        None

    Description:
        This function iterates over each image in the dataset split and converts the corresponding geojson
        annotations to JSON format, with pixel coordinates calculated using GDAL. It creates a dictionary
        containing image file information and a regions dictionary storing the asset footprints with their
        respective shape attributes. The resulting JSON file is saved as "nso.json" in the dataset path.Images 
        with no annotation have "regions= {}"
    """
  
    # Create an empty dictionary to store the training/test/val set of annotations and their pixel coordinates
    dataset_dict = {}

    # Loop over each image in the training set
    for file in tqdm(dataset_split, desc=f"Creating JSONs for Detectron2 on {dataset_path}", ncols=150, bar_format="{l_bar}{bar:10}{r_bar}"):
        file_path = os.path.join(small_tiles_path, file)
        img_id = file.split(".tif")[0]
        geojson_image = os.path.join(geojson_path, f"{img_id}.geojson")

        #Not all tiles have annotations, thus:
        if os.path.exists(geojson_image):

            # Load the geojson in gj
            with open(geojson_image) as f:
                gj = geojson.load(f)

            # Create a dictionary to store the regions (annotations spatial features) for the image
            regions = {}
            num_buildings = len(gj["features"])
            #print (num_buildings) 

            # Open the image with gdal to get pixel size and origin if feature exists
            #if num_buildings > 0:
            gdal_image = gdal.Open(file_path)

            # Get the pixel width and height(0.5 for nso) and the origin coordinates
            #https://www.gis.usu.edu/~chrisg/python/2009/lectures/ospy_slides4.pdf
            pixel_width, pixel_height = gdal_image.GetGeoTransform()[1], gdal_image.GetGeoTransform()[5]
            originX, originY = gdal_image.GetGeoTransform()[0], gdal_image.GetGeoTransform()[3]

            # Loop over each building/assets in the image
            for i in range(num_buildings):

                # Get the polygon points for the asset
                #https://stackoverflow.com/questions/23306653/python-accessing-nested-json-data
                points = gj["features"][i]["geometry"]["coordinates"][0]

                # Save asset id in the dictionary
                asset_id = gj["features"][i]["properties"]["id"]

                # If there is only one point, unwarp it=>check
                if len(points) == 1:
                    points = points[0]

                #Empty lists to store pixel coordinates
                all_points_x, all_points_y = [], []

                # Convert the lat/long points to pixel coordinates by substacting origin
                for j in range(len(points)):
                    all_points_x.append(int(round((points[j][0] - originX) / pixel_width)))
                    all_points_y.append(int(round((points[j][1] - originY) / pixel_height)))

                # Create a dictionary to store the asset footprint
                regions[str(i)] = {"shape_attributes":
                                       {"name": "polygon",
                                        "all_points_x": all_points_x,
                                        "all_points_y": all_points_y,
                                        "category": 0
                                       },
                                #save asset id in region attribute for the split function
                                   "region_attributes": {
                                        "asset_id": asset_id
                                   }
                                  }
                #print (regions)
            dictionary = {"file_ref": '',
                          "size": os.path.getsize(file_path),
                          "filename": file.replace(".tif", ".png"),
                          "base64_img_data": '',
                          "file_attributes": {},
                          "regions": regions,
                          "origin_x": originX,
                          "origin_y": originY
                         }
            #print (dictionary)
            dataset_dict[file.replace(".tif", ".png")] = dictionary
        else:
            # region is empty
            
            # still save data dic with empty regions and origins
            gdal_image = gdal.Open(file_path)
            # Get the pixel width and height(0.5 for nso) and the origin coordinates
            #https://www.gis.usu.edu/~chrisg/python/2009/lectures/ospy_slides4.pdf
            pixel_width, pixel_height = gdal_image.GetGeoTransform()[1], gdal_image.GetGeoTransform()[5]
            originX, originY = gdal_image.GetGeoTransform()[0], gdal_image.GetGeoTransform()[3]
            
            dictionary = {"file_ref": '',
                          "size": os.path.getsize(file_path),
                          "filename": file.replace(".tif", ".png"),
                          "base64_img_data": '',
                          "file_attributes": {},
                          "regions": {},
                          "origin_x": originX,
                          "origin_y": originY
                         }
            #print (dictionary)
        dataset_dict[file.replace(".tif", ".png")] = dictionary
            
    jsons_path = os.path.join(dataset_path,"nso_with_empty_annotations.json")
    with open(jsons_path, "w") as f:
        json.dump(dataset_dict, f)
    return None