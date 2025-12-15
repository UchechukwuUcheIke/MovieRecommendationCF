import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class AnnotatorQualityAnalyzer:
    """Analyze and identify problematic annotators"""
    # Written with help from gemini
    
    def __init__(self):
        self.user_stats = None
        self.flagged_users = None
        
    def analyze_users(self, ratings_df):
        """Analyze each user's rating patterns"""
        results = []
        
        for user_id in ratings_df.index:
            user_ratings = ratings_df.loc[user_id].dropna()
            
            if len(user_ratings) < 3:
                continue
            
            stats_dict = {
                'user_id': user_id,
                'n_ratings': len(user_ratings),
                'mean_rating': user_ratings.mean(),
                'std_rating': user_ratings.std(),
                'min_rating': user_ratings.min(),
                'max_rating': user_ratings.max(),
                'rating_range': user_ratings.max() - user_ratings.min(),
            }
            
            # Detect patterns i.e. all the same rating or all extreme ratings
            stats_dict['all_same'] = len(user_ratings.unique()) == 1
            stats_dict['only_extremes'] = all(r in [1, 5] for r in user_ratings)
            
            # Agreement with others
            movie_means = ratings_df[user_ratings.index].mean(axis=0)
            if len(movie_means) > 3:
                correlation, p_value = stats.spearmanr(user_ratings, movie_means)
                stats_dict['agreement_with_others'] = correlation
            else:
                stats_dict['agreement_with_others'] = np.nan
            
            results.append(stats_dict)
        
        self.user_stats = pd.DataFrame(results)
        return self.user_stats
    
    def flag_problematic_users(self, ratings_df):
        """Identify users with problematic rating patterns"""
        if self.user_stats is None:
            self.analyze_users(ratings_df)
        
        flagged = []
        
        for _, row in self.user_stats.iterrows():
            reasons = []
            
            if row['mean_rating'] < 2.0:
                reasons.append('too_harsh')
            if row['mean_rating'] > 4.5:
                reasons.append('too_generous')
            if row['std_rating'] < 0.5:
                reasons.append('no_variance')
            if row['all_same']:
                reasons.append('all_same_rating')
            if row['only_extremes']:
                reasons.append('only_extremes')
            if not np.isnan(row['agreement_with_others']) and row['agreement_with_others'] < 0.0:
                reasons.append('low_agreement')
            
            if reasons:
                flagged.append({
                    'user_id': row['user_id'],
                    'reasons': reasons,
                    'mean_rating': row['mean_rating'],
                    'std_rating': row['std_rating'],
                    'n_ratings': row['n_ratings'],
                    'agreement': row['agreement_with_others']
                })
        
        self.flagged_users = pd.DataFrame(flagged)
        return self.flagged_users
    
    def visualize_quality(self):
        """Create visualization of annotator quality"""
        if self.user_stats is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Mean ratings
        axes[0, 0].hist(self.user_stats['mean_rating'], bins=20, edgecolor='black')
        axes[0, 0].axvline(2.0, color='r', linestyle='--', label='Too harsh')
        axes[0, 0].axvline(4.5, color='r', linestyle='--', label='Too generous')
        axes[0, 0].set_xlabel('Mean Rating')
        axes[0, 0].set_ylabel('Number of Users')
        axes[0, 0].set_title('Distribution of User Mean Ratings')
        axes[0, 0].legend()
        
        # Rating variance
        axes[0, 1].hist(self.user_stats['std_rating'], bins=20, edgecolor='black')
        axes[0, 1].axvline(0.5, color='r', linestyle='--', label='Low variance')
        axes[0, 1].set_xlabel('Std Dev of Ratings')
        axes[0, 1].set_ylabel('Number of Users')
        axes[0, 1].set_title('Rating Variance')
        axes[0, 1].legend()
        
        # Agreement
        valid_agreement = self.user_stats['agreement_with_others'].dropna()
        axes[1, 0].hist(valid_agreement, bins=20, edgecolor='black')
        axes[1, 0].axvline(0, color='r', linestyle='--', label='No agreement')
        axes[1, 0].set_xlabel('Correlation with Others')
        axes[1, 0].set_ylabel('Number of Users')
        axes[1, 0].set_title('Agreement with Other Raters')
        axes[1, 0].legend()
        
        # Mean vs Std
        axes[1, 1].scatter(self.user_stats['mean_rating'], 
                          self.user_stats['std_rating'], alpha=0.6)
        axes[1, 1].set_xlabel('Mean Rating')
        axes[1, 1].set_ylabel('Std Dev')
        axes[1, 1].set_title('Rating Mean vs Variance')
        
        plt.tight_layout()
        plt.savefig('annotator_quality.png', dpi=300, bbox_inches='tight')
        plt.close()