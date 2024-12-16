#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 14:00:01 2024

@author: samir
"""

import pandas as pd
import numpy as np

# Load BioGRID data
data = pd.read_csv('BIOGRID-Interactions.tab', sep='\t')

# Extract interacting nodes
nodes = pd.unique(data[['Interactor A', 'Interactor B']].values.ravel('K'))
node_index = {node: i for i, node in enumerate(nodes)}

# Initialize adjacency matrix
adj_matrix = np.zeros((len(nodes), len(nodes)))

# Populate the adjacency matrix
for _, row in data.iterrows():
    i, j = node_index[row['Interactor A']], node_index[row['Interactor B']]
    adj_matrix[i, j] = 1
    adj_matrix[j, i] = 1  # Assuming undirected interactions

# Convert to DataFrame for easier visualization
adj_matrix_df = pd.DataFrame(adj_matrix, index=nodes, columns=nodes)
