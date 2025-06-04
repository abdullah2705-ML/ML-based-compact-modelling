#!/bin/bash

# Input and output file paths
input_file="iv_comparison.csv"
output_file="iv_comparison_sorted.csv"

# Extract header
head -n 1 "$input_file" > "$output_file"

# Sort by Vg (1st column) then Vd (2nd column), numerically and append to output
tail -n +2 "$input_file" | sort -t, -k1,1n -k2,2n >> "$output_file"

echo "Sorted file saved as $output_file"
