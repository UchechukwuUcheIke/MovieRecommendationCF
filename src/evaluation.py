import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_model(model, train_matrix, test_matrix):
    """
    Evaluate model on test set
    
    Args:
        model: Trained model with predict() method
        train_matrix: Training data
        test_matrix: Test data
        
    Returns:
        rmse, mae, predictions, actuals
    """
    predictions = []
    actuals = []
    
    for user_id in test_matrix.index:
        for movie_id in test_matrix.columns:
            actual = test_matrix.loc[user_id, movie_id]
            if pd.notna(actual):
                pred = model.predict(user_id, movie_id, train_matrix)
                predictions.append(pred)
                actuals.append(actual)
    
    if len(predictions) == 0:
        return np.nan, np.nan, [], []
    
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    
    return rmse, mae, predictions, actuals


def precision_at_k(model, train_matrix, test_matrix, k=10, threshold=4.0):
    """
    Calculate Precision@K
    
    Args:
        model: Trained model
        train_matrix: Training data
        test_matrix: Test data
        k: Number of top recommendations to consider
        threshold: Rating threshold for considering item as "relevant"
        
    Returns:
        Average precision@k across all users
    """
    precisions = []
    
    for user_id in test_matrix.index:
        test_ratings = test_matrix.loc[user_id].dropna()
        if len(test_ratings) == 0:
            continue
        
        train_ratings = train_matrix.loc[user_id]
        unrated_movies = train_ratings[train_ratings.isna()].index
        
        if len(unrated_movies) == 0:
            continue
        
        predictions = []
        for movie_id in unrated_movies:
            pred = model.predict(user_id, movie_id, train_matrix)
            predictions.append((movie_id, pred))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_k_movies = [m for m, _ in predictions[:k]]
        
        relevant = sum(1 for m in top_k_movies 
                      if m in test_ratings.index and test_ratings[m] >= threshold)
        
        precisions.append(relevant / k if k > 0 else 0)
    
    return np.mean(precisions) if precisions else 0


def get_recommendations(model, user_id, train_matrix, k=10):
    """
    Get top-k movie recommendations for a user
    
    Args:
        model: Trained model
        user_id: User to get recommendations for
        train_matrix: Training data
        k: Number of recommendations
        
    Returns:
        List of (movie_id, predicted_rating) tuples
    """
    user_ratings = train_matrix.loc[user_id]
    unrated_movies = user_ratings[user_ratings.isna()].index
    
    predictions = []
    for movie_id in unrated_movies:
        pred = model.predict(user_id, movie_id, train_matrix)
        predictions.append((movie_id, pred))
    
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:k]


def compare_models(train_matrix, test_matrix, data_label="Original"):
    """
    Train and evaluate all models
    
    Args:
        train_matrix: Training data
        test_matrix: Test data
        data_label: Label for this dataset (e.g., "Original" or "Bias-Corrected")
        
    Returns:
        results_df: DataFrame with performance metrics
        best_model: The best performing model
    """
    # Importing here gets rid of import error idk why
    from src.models.baseline import BaselinePredictor
    from src.models.collaborative_filtering import UserBasedCF, ItemBasedCF
    from src.models.matrix_factorization import MatrixFactorizationSVD
    
    results = []
    
    print(f"\n{'='*70}")
    print(f"EVALUATING MODELS - {data_label} Data")
    print(f"{'='*70}")
    
    print("\n1. Baseline (Global Average) Heuristic")
    baseline_global = BaselinePredictor(method='global')
    baseline_global.fit(train_matrix)
    rmse, mae, _, _ = evaluate_model(baseline_global, train_matrix, test_matrix)
    prec = precision_at_k(baseline_global, train_matrix, test_matrix, k=10)
    results.append({
        'Model': 'Baseline (Global)',
        'Type': 'Heuristic',
        'RMSE': rmse,
        'MAE': mae,
        'Precision@10': prec
    })
    print(f"   RMSE: {rmse:.4f}, MAE: {mae:.4f}, P@10: {prec:.4f}")
    
    # Baseline - User
    print("\n2. Baseline (User Average) - Heuristic")
    baseline_user = BaselinePredictor(method='user')
    baseline_user.fit(train_matrix)
    rmse, mae, _, _ = evaluate_model(baseline_user, train_matrix, test_matrix)
    prec = precision_at_k(baseline_user, train_matrix, test_matrix, k=10)
    results.append({
        'Model': 'Baseline (User)',
        'Type': 'Heuristic',
        'RMSE': rmse,
        'MAE': mae,
        'Precision@10': prec
    })
    print(f"   RMSE: {rmse:.4f}, MAE: {mae:.4f}, P@10: {prec:.4f}")
    
    # User-Based CF
    print("\n3. User-Based CF (k-NN) - Heuristic")
    user_cf = UserBasedCF(k=5)
    user_cf.fit(train_matrix)
    rmse, mae, _, _ = evaluate_model(user_cf, train_matrix, test_matrix)
    prec = precision_at_k(user_cf, train_matrix, test_matrix, k=10)
    results.append({
        'Model': 'User-Based CF',
        'Type': 'Heuristic',
        'RMSE': rmse,
        'MAE': mae,
        'Precision@10': prec
    })
    print(f"   RMSE: {rmse:.4f}, MAE: {mae:.4f}, P@10: {prec:.4f}")
    
    # Item-Based CF
    print("\n4. Item-Based CF (k-NN) - Heuristic")
    item_cf = ItemBasedCF(k=5)
    item_cf.fit(train_matrix)
    rmse, mae, _, _ = evaluate_model(item_cf, train_matrix, test_matrix)
    prec = precision_at_k(item_cf, train_matrix, test_matrix, k=10)
    results.append({
        'Model': 'Item-Based CF',
        'Type': 'Heuristic',
        'RMSE': rmse,
        'MAE': mae,
        'Precision@10': prec
    })
    print(f"   RMSE: {rmse:.4f}, MAE: {mae:.4f}, P@10: {prec:.4f}")
    
    # SVD
    print("\n5. Matrix Factorization (SVD) - Machine Learning")
    svd = MatrixFactorizationSVD(n_factors=20)
    svd.fit(train_matrix)
    rmse, mae, _, _ = evaluate_model(svd, train_matrix, test_matrix)
    prec = precision_at_k(svd, train_matrix, test_matrix, k=10)
    results.append({
        'Model': 'SVD',
        'Type': 'ML',
        'RMSE': rmse,
        'MAE': mae,
        'Precision@10': prec
    })
    print(f"   RMSE: {rmse:.4f}, MAE: {mae:.4f}, P@10: {prec:.4f}")
    
    best_model = svd
    
    # Neural Collaborative Filtering
    from src.models.neural_cf import NCFWrapper
        
    print("\n6. Neural Collaborative Filtering - Deep Learning")
    ncf = NCFWrapper(embedding_dim=50, hidden_layers=[64, 32], 
                        learning_rate=0.001, n_epochs=20, batch_size=256)
    ncf.fit(train_matrix)
    rmse, mae, _, _ = evaluate_model(ncf, train_matrix, test_matrix)
    prec = precision_at_k(ncf, train_matrix, test_matrix, k=10)
    results.append({
        'Model': 'Neural CF',
        'Type': 'Deep Learning',
        'RMSE': rmse,
        'MAE': mae,
        'Precision@10': prec
    })
    print(f"   RMSE: {rmse:.4f}, MAE: {mae:.4f}, P@10: {prec:.4f}")
        
    ncf.plot_training_curve()
        
    if rmse < results[-2]['RMSE']:
        best_model = ncf   
    
    return pd.DataFrame(results), best_model