#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:44:10 2024

@author: samira
"""

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.sparse import random as sparse_random


class MANE:
    def __init__(self, layers, D, alpha=0.1, embedding_dim=100):
        self.layers = layers
        self.D = D
        self.alpha = alpha
        self.embedding_dim = embedding_dim
        self.embeddings = [None] * len(layers)

    def _compute_normalized_laplacian(self, A):
        D = np.diag(np.sum(A, axis=1))
        L = D - A
        epsilon = 1e-6
        D_inv_sqrt = np.linalg.inv(np.sqrt(D + epsilon * np.eye(D.shape[0])))
        L_norm = D_inv_sqrt @ L @ D_inv_sqrt
        return L_norm

    def _optimize_within_layer(self, layer_index):
        A = self.layers[layer_index]
        L_norm = self._compute_normalized_laplacian(A)
        eigenvalues, eigenvectors = eigh(L_norm, eigvals=(0, self.embedding_dim - 1))
        self.embeddings[layer_index] = eigenvectors

    def _cross_layer_loss(self):
        loss = 0
        for i in range(len(self.layers)):
            for j in range(len(self.layers)):
                if i != j:
                    F_i = self.embeddings[i]
                    F_j = self.embeddings[j]
                    D_ij = self.D[i][j]
                    
                    if F_i.shape[0] != D_ij.shape[0] or F_j.shape[0] != D_ij.shape[1]:
                        raise ValueError(f"Shape mismatch: F_i {F_i.shape}, F_j {F_j.shape}, D_ij {D_ij.shape}")
                    
                    loss += np.linalg.norm(D_ij - F_i @ F_j.T, ord='fro') ** 2
        return loss

    def _optimize_cross_layer(self):
        initial_embeddings = np.concatenate([F.flatten() for F in self.embeddings])

        def objective_function(flat_embeddings):
            start = 0
            for i in range(len(self.layers)):
                end = start + self.embeddings[i].size
                self.embeddings[i] = flat_embeddings[start:end].reshape(-1, self.embedding_dim)
                start = end
            return self._cross_layer_loss()

        result = minimize(objective_function, initial_embeddings, method='L-BFGS-B')

        start = 0
        for i in range(len(self.layers)):
            end = start + self.embeddings[i].size
            self.embeddings[i] = result.x[start:end].reshape(-1, self.embedding_dim)
            start = end

    def fit(self):
        for i in range(len(self.layers)):
            self._optimize_within_layer(i)

        self._optimize_cross_layer()

    def get_embeddings(self):
        return self.embeddings


if __name__ == "__main__":
    num_nodes = [1162, 27, 100]
    num_layers = 3
    sparsity = 0.01

    layer1 = np.load("layer_1_adjacency_new.npy")
    layer2 = np.load("layer_2_adjacency_new.npy")
    layer3 = np.load("layer_3_adjacency_new.npy")
    layers = [layer1, layer2, layer3]

    for i, layer in enumerate(layers):
        if layer.shape[0] != layer.shape[1]:
            min_dim = min(layer.shape[0], layer.shape[1])
            layer = layer[:min_dim, :min_dim]
        layer = (layer + layer.T) / 2
        np.fill_diagonal(layer, 0)
        layers[i] = layer
        print(f"Fixed layer {i} shape: {layer.shape}")

    predefined_dependency = np.load("predefined_dependency_0_1.npy")
    
    D = []
    for i in range(num_layers):
        D_row = []
        for j in range(num_layers):
            if i == 0 and j == 1:
                rows, cols = layers[i].shape[0], layers[j].shape[0]
                assert predefined_dependency.shape == (rows, cols), f"Predefined matrix shape mismatch for D[0][1]: {predefined_dependency.shape}"
                D_row.append(predefined_dependency)
            elif i != j:
                rows, cols = layers[i].shape[0], layers[j].shape[0]
                cross_layer = sparse_random(rows, cols, density=sparsity, format="coo").toarray()
                D_row.append(cross_layer)
            else:
                rows, cols = layers[i].shape[0], layers[j].shape[0]
                D_row.append(np.zeros((rows, cols)))
        D.append(D_row)

    for i in range(num_layers):
        for j in range(num_layers):
            print(f"D[{i}][{j}] shape: {D[i][j].shape}")

    mane = MANE(layers, D, alpha=0.1, embedding_dim=2)
    mane.fit()

    np.set_printoptions(threshold=np.inf, linewidth=np.inf, suppress=True)
    embeddings = mane.get_embeddings()

    for idx, embedding in enumerate(embeddings):
        np.savetxt(f"embedding_layer_{idx}.txt", embedding, delimiter=',')
        print(f"Embedding for layer {idx} saved to embedding_layer_{idx}.txt")
