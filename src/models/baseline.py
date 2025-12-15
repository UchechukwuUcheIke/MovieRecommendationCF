class BaselinePredictor:
    """
    Simple baseline predictions using just statistical heuristics (avging score for movie)
    """
    
    def __init__(self, method='global'):
        self.method = method
        self.global_mean = None
        self.user_means = None
        
    def fit(self, train_matrix):
        self.global_mean = train_matrix.stack().mean()
        self.user_means = train_matrix.mean(axis=1)
        
    def predict(self, user_id, movie_id, train_matrix):
        if self.method == 'global':
            return self.global_mean
        elif self.method == 'user':
            return self.user_means.get(user_id, self.global_mean)
        return self.global_mean