#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:44:10 2024

@author: samir
"""

""

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.sparse import random as sparse_random


class MANE:
    def __init__(self, layers, D, alpha=0.1, embedding_dim=100):
        """
        Initialize the MANE class.

        Parameters:
        layers (list of np.ndarray): List of adjacency matrices for each layer.
        D (list of list of np.ndarray): Cross-layer dependency matrices.
        alpha (float): Regularization parameter for cross-layer loss.
        embedding_dim (int): Dimensionality of embeddings.
        """
        self.layers = layers  # List of adjacency matrices for each layer
        self.D = D  # Cross-layer dependency matrices
        self.alpha = alpha
        self.embedding_dim = embedding_dim
        self.embeddings = [None] * len(layers)

    def _compute_normalized_laplacian(self, A):
        """
        Compute the normalized Laplacian of an adjacency matrix.
        """
        D = np.diag(np.sum(A, axis=1))
        L = D - A
        D_inv_sqrt = np.linalg.inv(np.sqrt(D))
        L_norm = D_inv_sqrt @ L @ D_inv_sqrt
        return L_norm

    def _optimize_within_layer(self, layer_index):
        """
        Optimize the embeddings within a single layer using eigen decomposition.
        """
        A = self.layers[layer_index]
        L_norm = self._compute_normalized_laplacian(A)
        eigenvalues, eigenvectors = eigh(L_norm, eigvals=(0, self.embedding_dim - 1))
        self.embeddings[layer_index] = eigenvectors

    def _cross_layer_loss(self):
        """
        Compute the cross-layer loss based on Frobenius norm differences.
        """
        loss = 0
        for i in range(len(self.layers)):
            for j in range(len(self.layers)):
                if i != j:
                    F_i = self.embeddings[i]
                    F_j = self.embeddings[j]
                    D_ij = self.D[i][j]  # Cross-layer dependency matrix
                    loss += np.linalg.norm(D_ij - F_i @ F_j.T, ord='fro') ** 2
        return loss

    def _optimize_cross_layer(self):
        """
        Optimize embeddings jointly across layers to minimize cross-layer loss.
        """
        # Flatten the embeddings for optimization
        initial_embeddings = np.concatenate([F.flatten() for F in self.embeddings])

        def objective_function(flat_embeddings):
            start = 0
            for i in range(len(self.layers)):
                end = start + self.embeddings[i].size
                self.embeddings[i] = flat_embeddings[start:end].reshape(-1, self.embedding_dim)
                start = end
            return self._cross_layer_loss()

        # Optimize the embeddings
        result = minimize(objective_function, initial_embeddings, method='L-BFGS-B')

        # Update embeddings with optimized values
        start = 0
        for i in range(len(self.layers)):
            end = start + self.embeddings[i].size
            self.embeddings[i] = result.x[start:end].reshape(-1, self.embedding_dim)
            start = end

    def fit(self):
        """
        Fit the model by optimizing within-layer and cross-layer dependencies.
        """
        # Optimize within-layer connections
        for i in range(len(self.layers)):
            self._optimize_within_layer(i)

        # Optimize cross-layer dependencies
        self._optimize_cross_layer()

    def get_embeddings(self):
        """
        Retrieve the computed embeddings for each layer.
        """
        return self.embeddings


# Example usage
if __name__ == "__main__":
    num_nodes = 1000  # Number of nodes
    num_layers = 3  # Number of layers
    sparsity = 0.01  # Sparsity of the adjacency matrix

    # Generate random sparse adjacency matrices for each layer
    layers = []
    for _ in range(num_layers):
        A = sparse_random(num_nodes, num_nodes, density=sparsity, format="coo").toarray()
        A = (A + A.T) / 2  # Make symmetric
        np.fill_diagonal(A, 0)  # No self-loops
        layers.append(A)

    # Generate random cross-layer dependency matrices
    D = []
    for i in range(num_layers):
        D_row = []
        for j in range(num_layers):
            if i != j:
                cross_layer = sparse_random(num_nodes, num_nodes, density=sparsity, format="coo").toarray()
                D_row.append(cross_layer)
            else:
                D_row.append(np.zeros((num_nodes, num_nodes)))  # No dependencies within the same layer
        D.append(D_row)

    # Initialize and fit MANE
    mane = MANE(layers, D, alpha=0.1, embedding_dim=50)  # Adjust embedding_dim as needed
    mane.fit()

    # Get embeddings
    embeddings = mane.get_embeddings()

    # Print shape of embeddings for each layer
    for idx, emb in enumerate(embeddings):
        print(f"Embedding for layer {idx}: {emb.shape}")

