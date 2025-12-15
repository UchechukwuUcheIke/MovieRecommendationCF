from scipy.spatial.distance import cosine

class ItemBasedCF:
    """Item-based Collaborative Filtering"""
    
    def __init__(self, k=5):
        self.k = k
        self.train_matrix = None
        
    def fit(self, train_matrix):
        self.train_matrix = train_matrix.fillna(0)
        
    def compute_similarity(self, movie1, movie2):
        """Cosine similarity between movies"""
        ratings1 = self.train_matrix[movie1].values
        ratings2 = self.train_matrix[movie2].values
        
        mask = (ratings1 != 0) & (ratings2 != 0)
        if mask.sum() < 2:
            return 0
        
        return 1 - cosine(ratings1[mask], ratings2[mask])
    
    def predict(self, user_id, movie_id, train_matrix):
        """Predict based on similar movies"""
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
