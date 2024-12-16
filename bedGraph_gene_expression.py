#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 21:00:31 2024

@author: samir
"""
import pandas as pd
import argparse

def extract_gene_expression(bedgraph_file, output_file):
    # Read the BEDGraph file into a DataFrame
    try:
        data = pd.read_csv(bedgraph_file, sep="\t", header=None, names=["chrom", "start", "end", "value"])
    except Exception as e:
        print(f"Error reading BEDGraph file: {e}")
        return
    
    # Display the first few rows of the file
    print(f"First few rows of the input file:\n{data.head()}")

    # Save the extracted data as a CSV file
    try:
        data.to_csv(output_file, index=False)
        print(f"Gene expression levels saved to {output_file}")
    except Exception as e:
        print(f"Error saving output file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract gene expression levels from a BEDGraph file.")
    parser.add_argument("bedgraph_file", help="Path to the input BEDGraph file.")
    parser.add_argument("--output", default="gene_expression_levels.csv", help="Output file to save gene expression levels.")
    args = parser.parse_args()

    extract_gene_expression(args.bedgraph_file, args.output)
