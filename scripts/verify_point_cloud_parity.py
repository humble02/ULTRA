#!/usr/bin/env python3
"""Verify runtime point clouds against the original 4096-env training PCA frames.

The earlier single-env GPU reference was batch-dependent. Keep this entry
point but use the training-reference and multi-pose regression suite.
"""
from verify_training_pca_frames import main

if __name__ == '__main__':
    main()
