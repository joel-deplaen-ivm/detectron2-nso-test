import requests
import json
import time
import pandas as pd
import ast

df=pd.read_csv('datafiles/whole_input.csv')

def fetch_data(coord,date):
    username = 'alban'
    password = 'fukhez-0sejci-suhZiz'
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

def get_geom(response,name):
    response_data=response.json()
    item_list = response_data['features']
    for index in range(0, len(item_list), 2): # this skips the IRG items
        item = item_list[index]
        if item['services']['wms']['layername']==name:
            geom=item['geometry']['coordinates']
            print('geom created')
    dict_={"name": name, "multipolygon": [geom]}

    with open("datafiles/whole_wms_geom.jsonl", "a") as file:
        file.write(json.dumps(dict_) + "\n")
        file.close()
        print('geom dumped')

df_unique = df.drop_duplicates(subset=['wms_name'])
for i,row in df_unique[2:].iterrows():
    date = row['date']
    coord = ast.literal_eval(row['coordinate'])
    name = row['wms_name']
    response = fetch_data(coord,date)

    if response.status_code==200:
        get_geom(response,name)
        print("Sleeping for 15 seconds...")
        time.sleep(15)
    else:
        print('status code', response.status_code)
        print("Retrying after 1 minute...")
        time.sleep(60)  # Sleep for 1 minute (60 seconds)
        response = fetch_data(coord,date)
        if response.status_code==200:
            print('succes after wait')
            get_geom(response,name) # yes it can theoretically get stuck in a loop.....
        else:
            print('no succes after wait')

print('done with it')