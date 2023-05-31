import requests
import json
import time
import pandas as pd
import ast
import geojson

username = 'alban'
password = 'fukhez-0sejci-suhZiz'

df=pd.read_csv('datafiles/baseline_input.csv')

def fetch_data(coord,date):
    # Request payload
    payload = {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": coord
        },
        "properties": {
            "fields": {
                "geometry": True
            },
            "filters": {
                "datefilter": {
                    "startdate": date,
                    "enddate": date
                },
                "resolutionfilter": {
                    "maxres": 0.5,
                    "minres": 0.5
                }
            }
        }
    }
    data = json.dumps(payload)

    # Set the authentication
    auth = requests.auth.HTTPBasicAuth(username, password)

    # Send the POST request
    response = requests.post('https://api.satellietdataportaal.nl/v1/search', data=data, headers={"Content-Type": "application/json"}, auth=auth)

    return response

for i,row in df.iterrows():
    date = row['date']
    coord = ast.literal_eval(row['coordinate'])
    response = fetch_data(coord,date,id)
    response_data=response.json()
    geom = response_data['features']['geometry']



# List of multipolygons with corresponding IDs
multipolygons = [
    {"id": 1, "multipolygon": [[[[-77.036, 38.897], [-77.036, 38.898], [-77.035, 38.898], [-77.035, 38.897], [-77.036, 38.897]]]]},
    {"id": 2, "multipolygon": [[[[-77.039, 38.891], [-77.039, 38.892], [-77.038, 38.892], [-77.038, 38.891], [-77.039, 38.891]]]]},
    {"id": 3, "multipolygon": [[[[-77.042, 38.883], [-77.042, 38.884], [-77.041, 38.884], [-77.041, 38.883], [-77.042, 38.883]]]]}
]

# Create a FeatureCollection to store the multipolygons
features = []
for item in multipolygons:
    feature = geojson.Feature(
        geometry=geojson.MultiPolygon(item["multipolygon"]),
        properties={"id": item["id"]}
    )
    features.append(feature)

feature_collection = geojson.FeatureCollection(features)

# Save the FeatureCollection as a GeoJSON file
with open("multipolygons.geojson", "w") as f:
    geojson.dump(feature_collection, f)