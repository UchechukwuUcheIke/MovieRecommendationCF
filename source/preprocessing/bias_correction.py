"""
Movie Recommender System - CS 441 Final Project
Complete implementation with annotator quality control, bias correction, and model comparison
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class BiasCorrection:
    """Methods to correct for annotator biases"""
    # Written with help from gemini
    
    @staticmethod
    def remove_bias(ratings_df):
        """
        Remove user bias (harsh vs generous raters)
        Returns debiased ratings and user biases
        """
        global_mean = ratings_df.stack().mean()
        user_biases = ratings_df.mean(axis=1) - global_mean
        
        debiased = ratings_df.copy()
        for user_id in debiased.index:
            debiased.loc[user_id] = debiased.loc[user_id] - user_biases[user_id]
        
        return debiased, user_biases
    
    @staticmethod
    def remove_outliers(ratings_df, threshold=2.5):
        """Remove outlier ratings (z-score method)"""
        outliers_mask = pd.DataFrame(False, index=ratings_df.index, columns=ratings_df.columns)
        
        for movie in ratings_df.columns:
            movie_ratings = ratings_df[movie].dropna()
            if len(movie_ratings) < 3:
                continue
            
            mean = movie_ratings.mean()
            std = movie_ratings.std()
            
            if std > 0:
                for user in movie_ratings.index:
                    rating = ratings_df.loc[user, movie]
                    z_score = abs(rating - mean) / std
                    if z_score > threshold:
                        outliers_mask.loc[user, movie] = True
        
        cleaned = ratings_df.copy()
        cleaned[outliers_mask] = np.nan
        
        n_outliers = outliers_mask.sum().sum()
        return cleaned, n_outliers