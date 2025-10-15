#!/bin/bash

# Usage: ./clean_csv.sh <csv_file> <directory>
# Removes CSV entries where the file (first column) does not exist in the directory

if [ $# -ne 2 ]; then
    echo "Usage: $0 <csv_file> <directory>"
    exit 1
fi

CSV_FILE="$1"
DIRECTORY="$2"

# Temporary file for filtered CSV
TEMP_FILE=$(mktemp)

# Read CSV and filter
while IFS= read -r line; do
    # Extract first column (file name)
    filename=$(echo "$line" | awk -F',' '{print $1}' | tr -d '\r')
    if [ -n "$filename" ] && [ -f "$DIRECTORY/$filename" ]; then
        echo "$line" >> "$TEMP_FILE"
    fi
done < "$CSV_FILE"

# Replace original CSV with filtered version
mv "$TEMP_FILE" "$CSV_FILE"

echo "CSV cleaned: entries for non-existent files removed."