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
        # Add a small epsilon to the diagonal to handle singular matrices
        epsilon = 1e-6
        D_inv_sqrt = np.linalg.inv(np.sqrt(D + epsilon * np.eye(D.shape[0])))
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
                    
                    if F_i.shape[0] != D_ij.shape[0] or F_j.shape[0] != D_ij.shape[1]:
                        raise ValueError(f"Shape mismatch: F_i {F_i.shape}, F_j {F_j.shape}, D_ij {D_ij.shape}")
                    
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
    num_nodes = [1162, 27, 100]  # Number of nodes for each layer
    num_layers = 3  # Number of layers
    sparsity = 0.01  # Sparsity of the adjacency matrix

    # Load your adjacency matrices (or generate them dynamically)
    layer1 = np.load("layer_1_adjacency_new.npy") # Replace with np.load if using actual data
    layer2 = np.load("layer_2_adjacency_new.npy")
    layer3 = np.load("layer_3_adjacency_new.npy")
    layers = [layer1, layer2, layer3]

    # Ensure all layers are square
    for i, layer in enumerate(layers):
       if layer.shape[0] != layer.shape[1]:
           min_dim = min(layer.shape[0], layer.shape[1])
           layer = layer[:min_dim, :min_dim]  # Trim to the smallest dimension
       layer = (layer + layer.T) / 2  # Make symmetric
       np.fill_diagonal(layer, 0)  # Remove self-loops
       layers[i] = layer
       print(f"Fixed layer {i} shape: {layer.shape}")

    # Ensure symmetry and no self-loops
    for i, layer in enumerate(layers):
        layer = (layer + layer.T) / 2
        np.fill_diagonal(layer, 0)
        layers[i] = layer
        print(f"Fixed layer {i} shape: {layer.shape}")


   
    # Generate random cross-layer dependency matrices
    D = []
    for i in range(num_layers):
        D_row = []
        for j in range(num_layers):
            if i != j:
                rows, cols = layers[i].shape[0], layers[j].shape[0]  # Match sizes to layers
                cross_layer = sparse_random(rows, cols, density=sparsity, format="coo").toarray()
                D_row.append(cross_layer)
            else:
                rows, cols = layers[i].shape[0], layers[j].shape[0]
                D_row.append(np.zeros((rows, cols)))  # No dependencies within the same layer
        D.append(D_row)

    # Debugging: Print shapes of cross-layer matrices
    for i in range(num_layers):
        for j in range(num_layers):
            print(f"Fixed D[{i}][{j}] shape: {D[i][j].shape}")

    # Initialize and fit MANE
    mane = MANE(layers, D, alpha=0.1, embedding_dim=2)  # Adjust embedding_dim as needed
    mane.fit()
   

    # Ensure the full matrix is displayed
    np.set_printoptions(threshold=np.inf, linewidth=np.inf, suppress=True)
    # Access the embeddings
    embeddings = mane.get_embeddings()

    # Print shape of embeddings for each layer
    for idx, emb in enumerate(embeddings):
        print(f"Embedding for layer {idx}: {emb.shape}")

    # Print shape and embeddings for each layer
    embeddings = mane.get_embeddings()
    # Save embeddings to text files for each layer
    for idx, embedding in enumerate(embeddings):
        np.savetxt(f"embedding_layer_{idx}.txt", embedding, delimiter=',')
        print(f"Embedding for layer {idx} saved to embedding_layer_{idx}.txt")


