#!/bin/bash

### This script reads a list of FTP URLs from a CSV file, downloads the corresponding files using curl, ###
### unzips them, and moves the resulting TIFF files to /NSO/NSO_big_tiles . ###


# Read the URLs from the CSV file
urls=()  # Initialize an empty array called 'urls'
while IFS=, read -r url; do  # Loop through each line of the CSV file
    urls+=("$url")  # Add the current URL to the 'urls' array
done < ftp_urls.csv  # Read from the CSV file called 'ftp_urls_test.csv'

# Make and move to the directory where the files will be downloaded
mkdir -p NSO_raw  
cd NSO_raw

# Loop through the list of URLs in array and download each file
for url in "${urls[@]}" 
do
    # Get the filename from the URL
    filename=$(basename "$url")  

    # Download the file using curl
    if curl -u 'joel.deplaen:p2RMhQWRs8P67si' -O --ftp-pasv --ssl --insecure "$url"; then  
        echo "Downloaded $filename successfully"  
    else
        echo "Error downloading $filename"  
    fi
done

# Unzip the files and move them to a new directory
mkdir -p NSO_unzip  

# Loop through the list of zip files in the current directory and unzip them to the new directory
for file in *.zip
do
    unzip "$file" -d NSO_unzip/ 
done

# Move .tif to NSO directory
mkdir -p ../../../NSO/
mkdir -p ../../../NSO/NSO_big_tiles
mv NSO_unzip/*.tif ../../../NSO/NSO_big_tiles  

echo "Done downloading and organzing imagery dataset"
