import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

#Created with help from gemini
class RatingDataset(Dataset):
    def __init__(self, ratings_df, user_to_idx, movie_to_idx):
        self.samples = []
            
        for user_id in ratings_df.index:
            for movie_id in ratings_df.columns:
                rating = ratings_df.loc[user_id, movie_id]
                if pd.notna(rating):
                    user_idx = user_to_idx[user_id]
                    movie_idx = movie_to_idx[movie_id]
                    self.samples.append((user_idx, movie_idx, rating))
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        user_idx, movie_idx, rating = self.samples[idx]
        return (torch.LongTensor([user_idx])[0], 
                torch.LongTensor([movie_idx])[0], 
                torch.FloatTensor([rating])[0])
    
    
class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, n_users, n_movies, embedding_dim=50, hidden_layers=[64, 32]):
        super(NeuralCollaborativeFiltering, self).__init__()
            
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)
            
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.movie_embedding.weight, std=0.01)
            
        layers = []
        input_dim = embedding_dim * 2
            
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(input_dim, 1))
            
        self.dense_layers = nn.Sequential(*layers)
        
    def forward(self, user_ids, movie_ids):
        user_embedded = self.user_embedding(user_ids)
        movie_embedded = self.movie_embedding(movie_ids)
            
        x = torch.cat([user_embedded, movie_embedded], dim=1)
            
        output = self.dense_layers(x)
            
        return output.squeeze()
    
    
class NCFWrapper:
    # Made with help  from gemini
        
    def __init__(self, embedding_dim=50, hidden_layers=[64, 32], 
                     learning_rate=0.001, n_epochs=20, batch_size=256):
        self.embedding_dim = embedding_dim
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.model = None
        self.user_to_idx = None
        self.movie_to_idx = None
        self.idx_to_user = None
        self.idx_to_movie = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_losses = []
        self.global_mean = None
            
    def fit(self, train_matrix):
        self.global_mean = train_matrix.stack().mean()
            
        users = train_matrix.index.tolist()
        movies = train_matrix.columns.tolist()
            
        self.user_to_idx = {user: idx for idx, user in enumerate(users)}
        self.movie_to_idx = {movie: idx for idx, movie in enumerate(movies)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.idx_to_movie = {idx: movie for movie, idx in self.movie_to_idx.items()}
            
        dataset = RatingDataset(train_matrix, self.user_to_idx, self.movie_to_idx)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
        n_users = len(users)
        n_movies = len(movies)
        self.model = NeuralCollaborativeFiltering(
                n_users, n_movies, 
                self.embedding_dim, 
                self.hidden_layers
            ).to(self.device)
            
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
        self.model.train()
        for epoch in range(self.n_epochs):
            epoch_loss = 0
            n_batches = 0
                
            for user_ids, movie_ids, ratings in dataloader:
                user_ids = user_ids.to(self.device)
                movie_ids = movie_ids.to(self.device)
                ratings = ratings.to(self.device)
                    
                predictions = self.model(user_ids, movie_ids)
                loss = criterion(predictions, ratings)
                    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                    
                epoch_loss += loss.item()
                n_batches += 1
                
            avg_loss = epoch_loss / n_batches
            self.training_losses.append(avg_loss)
                
            if (epoch + 1) % 5 == 0:
                print(f"   Epoch {epoch+1}/{self.n_epochs}, Loss: {avg_loss:.4f}")
        
    def predict(self, user_id, movie_id, train_matrix):
            
        if self.model is None:
            return self.global_mean
            
        if user_id not in self.user_to_idx or movie_id not in self.movie_to_idx:
            return self.global_mean
            
        self.model.eval()
        with torch.no_grad():
            user_idx = torch.LongTensor([self.user_to_idx[user_id]]).to(self.device)
            movie_idx = torch.LongTensor([self.movie_to_idx[movie_id]]).to(self.device)
                
            prediction = self.model(user_idx, movie_idx).item()
                
            return np.clip(prediction, 1, 5)
        
    def plot_training_curve(self):
        if not self.training_losses:
            return
            
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(self.training_losses) + 1), self.training_losses, 'b-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Neural Collaborative Filtering - Training Loss')
        plt.grid(alpha=0.3)
        plt.savefig('outputs/ncf_training_curve.png', dpi=300, bbox_inches='tight')
        plt.close()