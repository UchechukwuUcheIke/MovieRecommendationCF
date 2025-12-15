import pandas as pd
from scipy.spatial.distance import cosine

class UserBasedCF:
    """Find similar users, Look at what those similar users rated for movies you haven't seen
    Weighted average of their ratings"""
    
    def __init__(self, k=5):
        self.k = k
        self.train_matrix = None
        
    def fit(self, train_matrix):
        self.train_matrix = train_matrix.fillna(0)
        
    def compute_similarity(self, user1, user2):
        ratings1 = self.train_matrix.loc[user1].values
        ratings2 = self.train_matrix.loc[user2].values
        
        mask = (ratings1 != 0) & (ratings2 != 0)
        if mask.sum() < 2:
            return 0
        
        return 1 - cosine(ratings1[mask], ratings2[mask])
    
    def predict(self, user_id, movie_id, train_matrix):
        similarities = []
        for other_user in train_matrix.index:
            if other_user == user_id:
                continue
            if pd.notna(train_matrix.loc[other_user, movie_id]) and train_matrix.loc[other_user, movie_id] > 0:
                sim = self.compute_similarity(user_id, other_user)
                if sim > 0:
                    similarities.append((other_user, sim))
        
        if not similarities:
            return train_matrix.stack().mean()
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:self.k]
        
        weighted_sum = sum(sim * train_matrix.loc[other_user, movie_id] 
                          for other_user, sim in top_k)
        sum_weights = sum(sim for _, sim in top_k)
        
        return weighted_sum / sum_weights if sum_weights > 0 else train_matrix.stack().mean()

class ItemBasedCF:
    def __init__(self, k=5):
        self.k = k
        self.train_matrix = None
        
    def fit(self, train_matrix):
        self.train_matrix = train_matrix.fillna(0)
        
    def compute_similarity(self, movie1, movie2):
        ratings1 = self.train_matrix[movie1].values
        ratings2 = self.train_matrix[movie2].values
        
        mask = (ratings1 != 0) & (ratings2 != 0)
        if mask.sum() < 2:
            return 0
        
        return 1 - cosine(ratings1[mask], ratings2[mask])
    
    def predict(self, user_id, movie_id, train_matrix):
        user_ratings = train_matrix.loc[user_id]
        rated_movies = user_ratings[user_ratings > 0]
        
        if len(rated_movies) == 0:
            return train_matrix.stack().mean()
        
        similarities = []
        for other_movie in rated_movies.index:
            if other_movie == movie_id:
                continue
            sim = self.compute_similarity(movie_id, other_movie)
            if sim > 0:
                similarities.append((other_movie, sim))
        
        if not similarities:
            return train_matrix.stack().mean()
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:self.k]
        
        weighted_sum = sum(sim * user_ratings[movie] for movie, sim in top_k)
        sum_weights = sum(sim for _, sim in top_k)
        
        return weighted_sum / sum_weights if sum_weights > 0 else train_matrix.stack().mean()
