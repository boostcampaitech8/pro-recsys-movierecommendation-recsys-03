"""
EASE (Embarrassingly Shallow Autoencoders for Sparse Data) implementation.

Reference: https://arxiv.org/abs/1905.03375

EASE is a linear model that learns item-item similarity through:
    B = (X^T X + λI)^(-1) X^T X
    
with the constraint that diagonal elements of B are zero.
"""

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import inv
import time
from typing import Optional, Tuple
import warnings


class EASE:
    """
    EASE: Embarrassingly Shallow Autoencoders for Sparse Data.
    
    Closed-form solution for item-item collaborative filtering.
    """
    
    def __init__(self, reg_weight: float = 500.0):
        """
        Initialize EASE model.
        
        Args:
            reg_weight: L2 regularization weight (lambda)
        """
        self.reg_weight = reg_weight
        self.B: Optional[np.ndarray] = None  # Item-item weight matrix
        self.n_items: int = 0
    
    def fit(self, X: csr_matrix, verbose: bool = True) -> 'EASE':
        """
        Fit EASE model using closed-form solution.
        
        B = (X^T X + λI)^(-1) X^T X
        diag(B) = 0
        
        Args:
            X: User-item interaction matrix (sparse, shape: n_users x n_items)
            verbose: Print progress
            
        Returns:
            self
        """
        if verbose:
            print(f"Fitting EASE model (λ={self.reg_weight})...")
            start_time = time.time()
        
        self.n_items = X.shape[1]
        
        # Compute X^T X (item-item co-occurrence)
        if verbose:
            print("  Computing X^T X...")
        G = (X.T @ X).toarray()  # Dense matrix for inversion
        
        # Add regularization: G + λI
        G += self.reg_weight * np.eye(self.n_items)
        
        # Compute inverse
        if verbose:
            print("  Computing matrix inverse...")
        G_inv = np.linalg.inv(G)
        
        # Compute B = G_inv @ (X^T X) = G_inv @ (G - λI)
        # But we need to zero out diagonal, so compute directly
        if verbose:
            print("  Computing weight matrix B...")
        
        # B = G_inv @ G - λ * G_inv = I - λ * G_inv
        # Then set diagonal to zero
        self.B = np.eye(self.n_items) - self.reg_weight * G_inv
        
        # Zero out diagonal (constraint)
        np.fill_diagonal(self.B, 0)
        
        if verbose:
            elapsed = time.time() - start_time
            print(f"  Done! Elapsed time: {elapsed:.2f}s")
        
        return self
    
    def predict(
        self, 
        X: csr_matrix, 
        user_indices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict scores for users.
        
        score = X @ B
        
        Args:
            X: User-item interaction matrix (same as training or subset)
            user_indices: Specific user indices to predict (None for all)
            
        Returns:
            Score matrix (n_users x n_items) or (len(user_indices) x n_items)
        """
        if self.B is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        if user_indices is not None:
            X_subset = X[user_indices]
            return X_subset @ self.B
        
        return X @ self.B
    
    def recommend(
        self,
        X: csr_matrix,
        top_k: int = 10,
        filter_already_liked: bool = True,
        user_indices: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate top-K recommendations for users.
        
        Args:
            X: User-item interaction matrix
            top_k: Number of items to recommend
            filter_already_liked: Remove already interacted items from recommendations
            user_indices: Specific user indices (None for all)
            
        Returns:
            Tuple of (item_indices, scores) arrays
            - item_indices: shape (n_users, top_k)
            - scores: shape (n_users, top_k)
        """
        # Get predictions
        scores = self.predict(X, user_indices)
        
        # Get the interaction matrix for filtering
        if user_indices is not None:
            X_filter = X[user_indices].toarray()
        else:
            X_filter = X.toarray()
        
        # Mask already liked items
        if filter_already_liked:
            scores[X_filter > 0] = -np.inf
        
        # Get top-K indices
        if top_k >= scores.shape[1]:
            top_indices = np.argsort(-scores, axis=1)
            top_scores = np.take_along_axis(scores, top_indices, axis=1)
        else:
            # Partial sort for efficiency
            top_indices = np.argpartition(-scores, top_k, axis=1)[:, :top_k]
            top_scores = np.take_along_axis(scores, top_indices, axis=1)
            
            # Sort the top-K by score
            sort_order = np.argsort(-top_scores, axis=1)
            top_indices = np.take_along_axis(top_indices, sort_order, axis=1)
            top_scores = np.take_along_axis(top_scores, sort_order, axis=1)
        
        return top_indices[:, :top_k], top_scores[:, :top_k]


class WeightedEASE(EASE):
    """
    Weighted EASE with time-based co-occurrence weighting.
    
    Instead of X^T X, uses weighted co-occurrence matrix C.
    B = (C + λI)^(-1) C
    """
    
    def __init__(self, reg_weight: float = 500.0):
        super().__init__(reg_weight)
    
    def fit_with_cooccurrence(
        self, 
        X: csr_matrix,
        C: csr_matrix,
        verbose: bool = True
    ) -> 'WeightedEASE':
        """
        Fit Weighted EASE using pre-computed co-occurrence matrix.
        
        B = (C + λI)^(-1) C
        diag(B) = 0
        
        Args:
            X: User-item interaction matrix (for reference/prediction)
            C: Item-item weighted co-occurrence matrix
            verbose: Print progress
            
        Returns:
            self
        """
        if verbose:
            print(f"Fitting Weighted EASE model (λ={self.reg_weight})...")
            start_time = time.time()
        
        self.n_items = X.shape[1]
        
        # Convert C to dense if sparse
        if hasattr(C, 'toarray'):
            C_dense = C.toarray()
        else:
            C_dense = C
        
        # Add regularization
        if verbose:
            print("  Computing (C + λI)^(-1)...")
        C_reg = C_dense + self.reg_weight * np.eye(self.n_items)
        
        # Compute inverse
        C_reg_inv = np.linalg.inv(C_reg)
        
        # Compute B = C_reg_inv @ C
        if verbose:
            print("  Computing weight matrix B...")
        self.B = C_reg_inv @ C_dense
        
        # Zero out diagonal
        np.fill_diagonal(self.B, 0)
        
        if verbose:
            elapsed = time.time() - start_time
            print(f"  Done! Elapsed time: {elapsed:.2f}s")
        
        return self


if __name__ == "__main__":
    # Simple test
    from scipy.sparse import random as sparse_random
    
    print("Testing EASE model...")
    
    # Create dummy data
    n_users, n_items = 100, 50
    X = sparse_random(n_users, n_items, density=0.1, format='csr')
    X.data[:] = 1  # Binary interactions
    
    # Fit model
    model = EASE(reg_weight=100.0)
    model.fit(X)
    
    # Get recommendations
    rec_items, rec_scores = model.recommend(X, top_k=5)
    print(f"Recommendations shape: {rec_items.shape}")
    print(f"Sample recommendations for user 0: {rec_items[0]}")
    print(f"Sample scores: {rec_scores[0]}")
