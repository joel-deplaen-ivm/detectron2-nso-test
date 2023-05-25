import requests
import json
import time

username = 'alban'
password = 'fukhez-0sejci-suhZiz'

def fetch_data(coord):
    # Request payload
    payload = {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": coord
        },
        "properties": {
            "fields": {
                "geometry": False
            },
            "filters": {
                # "datefilter": {
                #     "startdate": "2018-06-14",
                #     "enddate": "2019-04-16"
                # },
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
    
    if response.status_code == 200:
        # Save the URL to 'urls_done' file
        with open('datafiles/coords_done.json', 'r+') as file:
            coords_done = json.load(file)
            coords_done.append(coord)

            file.seek(0)
            json.dump(coords_done, file, indent=4)
            file.truncate()
        
        # Process the response data as per your requirement
        response_data = response.json()
        # Do something with the data
        instances = response_data['features']

        print('number of instances found for coordinate: ', len(instances))
        coord_wms_links = []
        for instance in instances:
            cloudcover = instance['properties']['cloudcover']
            wms_link = instance['services']['wms']['link']
            if (cloudcover < 0.05) & ('_SV_RD_8bit_RGB_' in wms_link):
                date = instance['properties']['acquired'].split( )[0]
                wms_name = instance['services']['wms']['layername']
                coord_wms_links.append({'wms_name': wms_name, 'wms_link': wms_link, 'cloudcover': cloudcover, 'coordinate': coord, 'date': date})
        print('number of wms links found for coordinate: ', len(coord_wms_links))
        print(f"Request successful for coordinate: {coord}")
        print("Sleeping for 15 seconds...")
        time.sleep(15)
    else:
        print(f"Request failed for coordinate: {coord}")
        print("Retrying after 1 minute...")
        time.sleep(60)  # Sleep for 1 minute (60 seconds)
        return fetch_data(coord) # yes it can theoretically get stuck in a loop.....

    return coord_wms_links

def process_coords():
    with open('datafiles/coords_done.json', 'r') as file:
        coords_done = json.load(file)
    
    with open('datafiles/centers_4326.geojson', 'r') as file:
        file = json.load(file)
        coords = [feature['geometry']['coordinates'] for feature in file['features']]
    
    for coord in coords:
        if coord not in coords_done:
            coord_wms_links = fetch_data(coord)
            if coord_wms_links is not None:
                with open('datafiles/wms_links.jsonl', 'a') as file:
                    for item in coord_wms_links:
                        json.dump(item, file)
                        file.write('\n')

# Call the main function
process_coords()