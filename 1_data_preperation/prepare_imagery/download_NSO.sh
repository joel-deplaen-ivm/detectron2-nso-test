#!/bin/bash

# Read the URLs from the CSV file
urls=()
while IFS=, read -r url; do
    urls+=("$url")
done < ftp_urls_test.csv

# Make and move to the directory where the files will be downloaded
mkdir -p NSO_raw
cd NSO_raw

# Loop through the list of URLs and download each file
for url in "${urls[@]}"
do
    # Get the filename from the URL
    filename=$(basename "$url")

    # Download the file using curl
    if curl -u 'joel.deplaen:WelcomeToTheM4chine!' -O --ftp-pasv --ssl --insecure "$url"; then
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
mkdir -p ../../NSO/ 
mv NSO_unzip/*.tif ../../NSO/

echo "Done downloading and organzing imagery dataset"