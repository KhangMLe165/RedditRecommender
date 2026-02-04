#!/usr/bin/env python3
"""
Evaluation script for Engage Corpus recommender systems.

Computes two metrics:
1. Hit Rate @ 10 (HR@10): Rate at which the correct subreddit appears in top 10 recommendations
2. NDCG @ 10: Normalized Discounted Cumulative Gain at 10

Usage:
    python evaluate.py --predictions predictions.pkl --ground_truth ground_truth.pkl
    
    Or for direct numpy arrays:
    python evaluate.py --predictions_npy predictions.npy --ground_truth_npy ground_truth.npy
"""

import numpy as np
import pickle
import argparse
from typing import List, Union

def hit_rate_at_k(predictions: np.ndarray, ground_truth: np.ndarray, k: int = 10) -> float:
    """
    Calculate Hit Rate @ K.
    
    HR@K is the rate at which the correct item appears in the top K recommendations.
    
    Args:
        predictions: Array of shape (n_users, n_items) with prediction scores
        ground_truth: Array of shape (n_users,) with ground truth item indices
        k: Number of top recommendations to consider
    
    Returns:
        Hit rate as a float between 0 and 1
    """
    n_users = predictions.shape[0]
    hits = 0
    
    for i in range(n_users):
        # Get top k predictions for this user
        top_k_items = np.argsort(predictions[i])[::-1][:k]
        
        # Check if ground truth is in top k
        if ground_truth[i] in top_k_items:
            hits += 1
    
    return hits / n_users


def ndcg_at_k(predictions: np.ndarray, ground_truth: np.ndarray, k: int = 10) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain @ K.
    
    NDCG@K is a rank-based metric that gives higher scores when the correct
    item appears earlier in the top K recommendations.
    
    For binary relevance (1 correct item), NDCG@K is equivalent to:
    (1/n) * sum_i (1 / log2(rank_i + 1))
    
    where rank_i is the 1-indexed rank of the correct item for user i,
    and rank_i = infinity if the item is not in top K.
    
    Args:
        predictions: Array of shape (n_users, n_items) with prediction scores
        ground_truth: Array of shape (n_users,) with ground truth item indices
        k: Number of top recommendations to consider
    
    Returns:
        NDCG score as a float between 0 and 1
    """
    n_users = predictions.shape[0]
    ndcg_sum = 0.0
    
    for i in range(n_users):
        # Get top k predictions for this user
        top_k_items = np.argsort(predictions[i])[::-1][:k]
        
        # Find rank of ground truth item (1-indexed)
        try:
            rank = np.where(top_k_items == ground_truth[i])[0][0] + 1
            # DCG for single relevant item at this rank
            dcg = 1.0 / np.log2(rank + 1)
            # IDCG (ideal DCG) for single item is always 1 / log2(2) = 1
            idcg = 1.0
            ndcg = dcg / idcg
            ndcg_sum += ndcg
        except IndexError:
            # Ground truth not in top k, contribution is 0
            pass
    
    return ndcg_sum / n_users


def evaluate_from_scores(predictions: np.ndarray, ground_truth: np.ndarray, k: int = 10) -> dict:
    """
    Evaluate predictions using both HR@K and NDCG@K.
    
    Args:
        predictions: Array of shape (n_users, n_items) with prediction scores
        ground_truth: Array of shape (n_users,) with ground truth item indices
        k: Number of top recommendations to consider
    
    Returns:
        Dictionary with 'hr@k' and 'ndcg@k' metrics
    """
    hr = hit_rate_at_k(predictions, ground_truth, k)
    ndcg = ndcg_at_k(predictions, ground_truth, k)
    
    return {
        f'hr@{k}': hr,
        f'ndcg@{k}': ndcg
    }


def evaluate_from_predictions_file(predictions_file: str, ground_truth_file: str, k: int = 10) -> dict:
    """
    Evaluate from pickle files containing predictions and ground truth.
    
    Expected file format:
    - predictions_file: pickle file with dict {'dev': [...], 'test': [...]} where each entry
      is a list of predicted subreddit indices for each user
    - ground_truth_file: pickle file with dict {'dev': [...], 'test': [...]} where each entry
      is the ground truth subreddit index
    
    Args:
        predictions_file: Path to pickle file with predictions
        ground_truth_file: Path to pickle file with ground truth
        k: Number of top recommendations to consider
    
    Returns:
        Dictionary with results for dev and test sets
    """
    # Load predictions and ground truth
    with open(predictions_file, 'rb') as f:
        predictions_dict = pickle.load(f)
    
    with open(ground_truth_file, 'rb') as f:
        ground_truth_dict = pickle.load(f)
    
    results = {}
    
    for split in ['dev', 'test']:
        if split in predictions_dict and split in ground_truth_dict:
            predictions = predictions_dict[split]
            ground_truth = ground_truth_dict[split]
            
            # Convert list of predicted indices to score matrix
            n_users = len(predictions)
            n_items = 5000  # Number of subreddits
            
            # Create score matrix (higher score = more likely to be predicted)
            score_matrix = np.zeros((n_users, n_items), dtype=np.float32)
            
            for i, pred_list in enumerate(predictions):
                if pred_list:
                    # Give score based on reverse order (later predictions get higher scores)
                    # This way argsort will put predicted items first
                    for rank, item_idx in enumerate(pred_list):
                        if isinstance(item_idx, int) and 0 <= item_idx < n_items:
                            score_matrix[i, item_idx] = 1000 - rank
            
            ground_truth_array = np.array(ground_truth, dtype=np.int32)
            
            # Evaluate
            split_results = evaluate_from_scores(score_matrix, ground_truth_array, k)
            results[split] = split_results
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate recommender system predictions'
    )
    parser.add_argument(
        '--predictions',
        type=str,
        help='Path to pickle file with predictions dict'
    )
    parser.add_argument(
        '--ground_truth',
        type=str,
        help='Path to pickle file with ground truth dict'
    )
    parser.add_argument(
        '--predictions_npy',
        type=str,
        help='Path to numpy file with prediction scores (n_users, n_items)'
    )
    parser.add_argument(
        '--ground_truth_npy',
        type=str,
        help='Path to numpy file with ground truth indices (n_users,)'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=10,
        help='Number of top recommendations to consider (default: 10)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['dev', 'test', 'both'],
        help='Which split to evaluate (default: test)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("ENGAGE CORPUS EVALUATION")
    print("="*60)
    
    if args.predictions and args.ground_truth:
        # Evaluate from pickle files
        print(f"Loading predictions from: {args.predictions}")
        print(f"Loading ground truth from: {args.ground_truth}")
        print(f"Metric: HR@{args.k} and NDCG@{args.k}")
        print()
        
        results = evaluate_from_predictions_file(args.predictions, args.ground_truth, args.k)
        
        # Print results
        if args.split == 'both':
            splits_to_show = ['dev', 'test']
        else:
            splits_to_show = [args.split]
        
        for split in splits_to_show:
            if split in results:
                print(f"\n{split.upper()} SET RESULTS:")
                print("-" * 60)
                print(f"  Hit Rate @ {args.k}:  {results[split][f'hr@{args.k}']:.4f}  ({results[split][f'hr@{args.k}']*100:.2f}%)")
                print(f"  NDCG @ {args.k}:      {results[split][f'ndcg@{args.k}']:.4f}")
    
    elif args.predictions_npy and args.ground_truth_npy:
        # Evaluate from numpy files
        print(f"Loading prediction scores from: {args.predictions_npy}")
        print(f"Loading ground truth from: {args.ground_truth_npy}")
        print(f"Metric: HR@{args.k} and NDCG@{args.k}")
        print()
        
        predictions = np.load(args.predictions_npy)
        ground_truth = np.load(args.ground_truth_npy)
        
        print(f"Predictions shape: {predictions.shape}")
        print(f"Ground truth shape: {ground_truth.shape}")
        print()
        
        results = evaluate_from_scores(predictions, ground_truth, args.k)
        
        print("\nRESULTS:")
        print("-" * 60)
        print(f"  Hit Rate @ {args.k}:  {results[f'hr@{args.k}']:.4f}  ({results[f'hr@{args.k}']*100:.2f}%)")
        print(f"  NDCG @ {args.k}:      {results[f'ndcg@{args.k}']:.4f}")
    
    else:
        print("ERROR: Must provide either:")
        print("  --predictions and --ground_truth (for pickle files)")
        print("  OR")
        print("  --predictions_npy and --ground_truth_npy (for numpy arrays)")
        parser.print_help()
        return
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
