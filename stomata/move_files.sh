#!/bin/bash

# Usage: ./move_files.sh <csv_file> <source_dir> <dest_dir>
# Assumes CSV has file names in the first column, no header row

if [ $# -ne 3 ]; then
    echo "Usage: $0 <csv_file> <source_dir> <dest_dir>"
    exit 1
fi

CSV_FILE="$1"
SOURCE_DIR="$2"
DEST_DIR="$3"

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Read file names from CSV into an associative array
declare -A files_from_csv
while IFS= read -r line; do
    # Extract first column (file name)
    filename=$(echo "$line" | awk -F',' '{print $1}' | tr -d '\r')
    if [ -n "$filename" ]; then
        files_from_csv["$filename"]=1
    fi
done < "$CSV_FILE"

# Loop through files in source directory
for file in "$SOURCE_DIR"/*; do
    if [ -f "$file" ]; then
        basename_file=$(basename "$file")
        # If file not in CSV list, move it
        if [[ ! ${files_from_csv["$basename_file"]} ]]; then
            mv "$file" "$DEST_DIR"
            echo "Moved: $basename_file"
        fi
    fi
done

echo "Operation completed."