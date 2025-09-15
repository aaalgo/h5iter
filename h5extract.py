#!/usr/bin/env python3
"""
Extract columns from H5AD files and save as dense numpy arrays.

Usage:
    python h5extract.py -i INPUT.h5ad -o OUTPUT.npy -c columns.json

columns.json format:
{
    "columns": [1, 3, 5, ...]
}
"""

import argparse
import json
import numpy as np
import scanpy as sc
import sys
from pathlib import Path


def load_columns(columns_path):
    """Load column indices from JSON file."""
    try:
        with open(columns_path, 'r') as f:
            data = json.load(f)
        
        if 'columns' not in data:
            raise ValueError("JSON file missing 'columns' field")
        
        columns = data['columns']
        if not columns:
            raise ValueError("No columns specified in JSON file")
        
        return columns
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error loading columns file: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Extract columns from H5AD files")
    parser.add_argument('-i', '--input', required=True, help='Path to input H5AD file')
    parser.add_argument('-o', '--output', required=True, help='Path to output NPY file')
    parser.add_argument('-c', '--columns', required=True, help='Path to columns JSON file')
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    # Load columns
    columns = load_columns(args.columns)
    print(f"Loaded {len(columns)} columns from {args.columns}")
    
    try:
        # Load H5AD file
        print(f"Loading H5AD file: {args.input}")
        adata = sc.read_h5ad(args.input)
        
        # Extract specified columns
        print(f"Extracting columns from {adata.shape[0]} x {adata.shape[1]} matrix")
        
        # Extract specified columns first (keeping sparse if applicable)
        X_selected = adata.X[:, columns]
        
        # Convert to dense only after extraction
        if hasattr(X_selected, 'toarray'):
            X_selected = X_selected.toarray()
        
        # Save as NPY file
        print(f"Saving extracted matrix ({X_selected.shape[0]} x {X_selected.shape[1]}) to: {args.output}")
        np.save(args.output, X_selected)
        
        print("Extraction completed successfully")
        
    except Exception as e:
        print(f"Error processing H5AD file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()