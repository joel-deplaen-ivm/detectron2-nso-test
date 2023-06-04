import requests
import json
import pandas as pd

def extract_name(ftp_url):
    return ftp_url.split('/')[8].split('_SV_RD_8bit')[0]

def extract_filename(wms_link):
    return wms_link.strip('https://tiles1.geoserve.eu/SuperView1/tileserver/').strip('/wmts')

wms_data = []
with open('datafiles/wms_links.jsonl', 'r') as file:
    for line in file:
        json_line = json.loads(line)
        wms_data.append(json_line)

df_wms_data = pd.DataFrame(wms_data)
df_ftp = pd.read_csv('ftp_urls.csv').drop_duplicates(subset=['ftp_url']).reset_index(drop=True)
df_ftp['wms_name'] = df_ftp['ftp_url'].apply(extract_name)
df = pd.merge(df_wms_data, df_ftp, left_on='wms_name', right_on='wms_name')
df['coordinate_str']=df['coordinate'].astype(str)
df['file_name'] = df['wms_link'].apply(extract_filename)
# merged_df['wms_name'].value_counts().tolist()
# df_baseline = df.sort_values('date',ascending=False).groupby('coordinate_str', as_index=False).head(1)
df.to_csv('datafiles/whole_baseline_input.csv',index=False)

df_filenames = df['file_name']
df_filenames.to_csv('datafiles/whole_filenames.csv',index=False)
