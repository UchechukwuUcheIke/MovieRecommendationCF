"""
visualization.py
Plotting and visualization functions
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_comparison(results_original, results_cleaned):
    """
    Compare results with and without bias correction
    
    Args:
        results_original: DataFrame with results on original data
        results_cleaned: DataFrame with results on bias-corrected data
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    models = results_original['Model'].values
    x = np.arange(len(models))
    width = 0.35
    
    # Color code by type
    colors_orig = ['gray' if t == 'Heuristic' else 'blue' if t == 'ML' else 'green' 
                   for t in results_original['Type']]
    colors_clean = ['lightgray' if t == 'Heuristic' else 'lightblue' if t == 'ML' else 'lightgreen' 
                    for t in results_cleaned['Type']]
    
    # RMSE
    bars1 = axes[0].bar(x - width/2, results_original['RMSE'], width, 
                        label='Original', alpha=0.8, color=colors_orig)
    bars2 = axes[0].bar(x + width/2, results_cleaned['RMSE'], width, 
                        label='Bias-Corrected', alpha=0.8, color=colors_clean)
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('Root Mean Squared Error')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # MAE
    axes[1].bar(x - width/2, results_original['MAE'], width, 
                label='Original', alpha=0.8, color=colors_orig)
    axes[1].bar(x + width/2, results_cleaned['MAE'], width, 
                label='Bias-Corrected', alpha=0.8, color=colors_clean)
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Mean Absolute Error')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    # Precision@10
    axes[2].bar(x - width/2, results_original['Precision@10'], width, 
                label='Original', alpha=0.8, color=colors_orig)
    axes[2].bar(x + width/2, results_cleaned['Precision@10'], width, 
                label='Bias-Corrected', alpha=0.8, color=colors_clean)
    axes[2].set_ylabel('Precision@10')
    axes[2].set_title('Precision at K=10')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=45, ha='right')
    axes[2].legend()
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/model_comparison.png', dpi=300, bbox_inches='tight')
    print("   Saved: outputs/model_comparison.png")
    plt.close()