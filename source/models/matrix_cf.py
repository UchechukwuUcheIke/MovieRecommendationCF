import numpy as np
from scipy.sparse.linalg import svds

class MatrixFactorizationSVD:
    """Matrix Factorization using SVD"""
    
    def __init__(self, n_factors=20):
        self.n_factors = n_factors
        self.user_factors = None
        self.movie_factors = None
        self.global_mean = None
        
    def fit(self, train_matrix):
        matrix = train_matrix.fillna(0).values
        self.global_mean = train_matrix.stack().mean()
        
        user_means = train_matrix.mean(axis=1).values
        matrix_centered = matrix - user_means.reshape(-1, 1)
        matrix_centered[matrix == 0] = 0
        
        U, sigma, Vt = svds(matrix_centered, k=self.n_factors)
        
        self.user_factors = U
        self.sigma = np.diag(sigma)
        self.movie_factors = Vt.T
        self.user_means = user_means
        self.user_index = train_matrix.index
        self.movie_index = train_matrix.columns
        
    def predict(self, user_id, movie_id, train_matrix):
        """Predict using learned factors"""
        try:
            user_idx = self.user_index.get_loc(user_id)
            movie_idx = self.movie_index.get_loc(movie_id)
            
            pred = np.dot(
                np.dot(self.user_factors[user_idx, :], self.sigma),
                self.movie_factors[movie_idx, :]
            ) + self.user_means[user_idx]
            
            return np.clip(pred, 1, 5)
        except:
            return self.global_mean