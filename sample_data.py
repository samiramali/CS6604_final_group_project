#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 11:14:59 2024

@author: samir
"""

import numpy as np

# Layer 1 adjacency matrix (3 nodes)
layer1 = np.array([[0, 1, 1],
                   [1, 0, 0],
                   [1, 0, 0]])

# Layer 2 adjacency matrix (3 nodes)
layer2 = np.array([[0, 0, 1],
                   [0, 0, 1],
                   [1, 1, 0]])

# Cross-layer dependency matrix (3x3)
D = [np.array([[0, 1, 0],
                [0, 0, 1],
                [1, 0, 0]]),  # Dependency from Layer 1 to Layer 2
     np.array([[0, 1, 0],
                [0, 0, 1],
                [1, 0, 0]])]  # Dependency from Layer 2 to Layer 1

# Example usage of the MANE class with the synthetic dataset
if __name__ == "__main__":
    layers = [layer1, layer2]
    
    mane = MANE(layers, D)
    mane.fit()
    embeddings = mane.get_embeddings()
    print("Layer 1 Embeddings:\n", embeddings[0])
    print("Layer 2 Embeddings:\n", embeddings[1])