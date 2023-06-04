import json

geojson = {
    "type": "FeatureCollection",
    "features": []
}

with open("datafiles/wms_geom.jsonl", "r") as file:
    for line in file:
        data = json.loads(line)
        name = data["name"]
        multipolygon = data["multipolygon"]

        feature = {
            "type": "Feature",
            "properties": {"name": name},
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": multipolygon
            }
        }

        geojson["features"].append(feature)

with open("datafiles/baseline.geojson", "w") as outfile:
    json.dump(geojson, outfile)
