import numpy as np
import pandas as pd


def load_survey_data(filepath):
    df = pd.read_csv(filepath)
    df = df.set_index(df.columns[0])
    df = df.apply(pd.to_numeric, errors='coerce')
    
    return df


def preprocess_data(df, min_ratings_per_user=10, min_ratings_per_movie=5):
    # Remove users with too few ratings
    user_counts = df.notna().sum(axis=1)
    df = df[user_counts >= min_ratings_per_user]
    
    # Remove movies with too few ratings
    movie_counts = df.notna().sum(axis=0)
    df = df.loc[:, movie_counts >= min_ratings_per_movie]
    return df


def train_test_split_users(ratings_df, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    train = ratings_df.copy()
    test = pd.DataFrame(np.nan, index=ratings_df.index, columns=ratings_df.columns)
    
    for idx in ratings_df.index:
        user_ratings = ratings_df.loc[idx]
        rated_movies = user_ratings.dropna()
        
        if len(rated_movies) < 5:
            continue
        
        n_test = max(1, int(len(rated_movies) * test_size))
        test_indices = np.random.choice(rated_movies.index, size=n_test, replace=False)
        
        test.loc[idx, test_indices] = train.loc[idx, test_indices]
        train.loc[idx, test_indices] = np.nan
    
    print(f"\nTrain/Test Split:")
    print(f"Train ratings: {train.notna().sum().sum()}")
    print(f"Test ratings: {test.notna().sum().sum()}")
    
    return train, test