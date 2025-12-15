import numpy as np
import pandas as pd


class BiasCorrection:
    # Generated with help from gemini
    """Methods to correct for annotator biases (bad annotator)"""
    
    @staticmethod
    def remove_bias(ratings_df):
        """
        Remove user bias (harsh vs generous raters)
        
        Args:
            ratings_df: Raw ratings DataFrame
            
        Returns:
            debiased_df: Bias-corrected ratings
            user_biases: Series of biases for each user
        """
        global_mean = ratings_df.stack().mean()
        user_biases = ratings_df.mean(axis=1) - global_mean
        
        debiased = ratings_df.copy()
        for user_id in debiased.index:
            debiased.loc[user_id] = debiased.loc[user_id] - user_biases[user_id]
        
        return debiased, user_biases
    
    @staticmethod
    def remove_outliers(ratings_df, threshold=2.5):
        """
        Remove outlier ratings using z-score method
        
        Args:
            ratings_df: Raw ratings DataFrame
            threshold: Z-score threshold for outlier detection
            
        Returns:
            cleaned_df: DataFrame with outliers removed (set to NaN)
            n_outliers: Number of outliers removed
        """
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