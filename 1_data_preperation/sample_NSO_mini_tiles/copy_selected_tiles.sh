file_list="NSO_mini_tiles_list.txt"
source_folder="/scistor/ivm/jpl204/projects/detectron2-nso/NSO/NSO_small_tiles"
destination_folder="/scistor/ivm/jpl204/projects/detectron2-nso/1_data_preperation/sample_NSO_mini_tiles/mini_tiles"

# Check if the file list exists
if [ ! -f "$file_list" ]; then
  echo "File list does not exist: $file_list"
  exit 1
fi

# Read each line from the file list and copy the corresponding file to the destination folder
while IFS= read -r filename; do
  source_file="$source_folder/$filename"
  destination_file="$destination_folder/$filename"
  cp "$source_file" "$destination_file"
done < "$file_list"

# Optionally, you can print a message indicating the completion of the task
echo "Files have been copied to $destination_folder"