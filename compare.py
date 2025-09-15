#!/usr/bin/env python3
import sys
import numpy as np

if len(sys.argv) != 3:
    print("Usage: python compare.py <file1.npy> <file2.npy>", file=sys.stderr)
    sys.exit(1)

file1 = sys.argv[1]
file2 = sys.argv[2]

try:
    arr1 = np.load(file1)
    arr2 = np.load(file2)
    
    assert np.array_equal(arr1, arr2), f"Arrays are not identical"
    print(f"Files {file1} and {file2} are identical")
    
except FileNotFoundError as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error comparing files: {e}", file=sys.stderr)
    sys.exit(1)