#!/usr/bin/env python3
"""
Simple Baseline for Reddit Subreddit Recommendation
Using Vowpal Wabbit with logistic regression on trigrams

This baseline predicts which subreddit a user will interact with next
based on the text content of their previous posts and comments.

Model: Vowpal Wabbit One-Against-All (OAA) logistic regression
Features: Padded word trigrams from user text
Task: Predict first July-Sept interaction using Jan-June text
"""

import re
import numpy as np
import vowpal_wabbit_next as vw
from tqdm import tqdm
import pickle
import argparse


def simple_tokenize(text):
    """
    Simple whitespace tokenizer with lowercasing.
    
    Args:
        text: Input text string
        
    Returns:
        List of lowercase tokens
    """
    if not text:
        return []
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []
    return text.split()


def padded_trigrams(tokens, pad_token="PAD", pad_size=2):
    """
    Create padded word trigrams for VW format.
    
    Example:
        tokens = ["hello", "world"]
        returns: ["PAD_PAD_hello", "PAD_hello_world", "hello_world_PAD", "world_PAD_PAD"]
    
    Args:
        tokens: List of word tokens
        pad_token: Token to use for padding
        pad_size: Number of padding tokens on each side
        
    Returns:
        List of trigram strings
    """
    if not tokens:
        return ["PAD_PAD_PAD"]
    
    padded = [pad_token] * pad_size + tokens + [pad_token] * pad_size
    trigrams = []
    for i in range(len(padded) - 2):
        trigram = "_".join(padded[i:i+3])
        trigrams.append(trigram)
    return trigrams


def train_vw_model(train_file, num_classes, output_model_file=None):
    """
    Train a Vowpal Wabbit OAA logistic regression model.
    
    Args:
        train_file: Path to VW format training file
        num_classes: Number of classes (subreddits)
        output_model_file: Path to save the trained model (optional)
        
    Returns:
        Tuple of (workspace, parser) for making predictions
    """
    print(f"Training VW model with {num_classes} classes...")
    
    # Initialize VW workspace with OAA and logistic loss
    workspace = vw.Workspace([
        f"--oaa={num_classes}",
        "--loss_function=logistic",
        "--probabilities",  # Output probabilities instead of class labels
        "--passes=1",
        "--cache",
        "--kill_cache"
    ])
    parser = vw.TextFormatParser(workspace)
    
    # Train on data
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Training"):
            line = line.strip()
            if line:
                try:
                    example = parser.parse_line(line)
                    workspace.learn_one(example)
                except Exception as e:
                    print(f"Error training on line: {e}")
    
    print("Training complete!")
    
    # Save model if requested
    if output_model_file:
        model_bytes = workspace.serialize()
        with open(output_model_file, 'wb') as f:
            f.write(model_bytes)
        print(f"Model saved to: {output_model_file}")
    
    # Create new workspace for predictions
    model_bytes = workspace.serialize()
    workspace = vw.Workspace(
        args=[
            f"--oaa={num_classes}",
            "--loss_function=logistic",
            "--probabilities",
            "--quiet"
        ],
        model_data=model_bytes
    )
    parser = vw.TextFormatParser(workspace)
    
    return workspace, parser


def get_vw_predictions(workspace, parser, text, num_classes, top_k=10):
    """
    Get top-k predictions from VW model for given text.
    
    Args:
        workspace: VW workspace
        parser: VW parser
        text: Input text string
        num_classes: Total number of classes
        top_k: Number of top predictions to return
        
    Returns:
        Numpy array of top-k class indices (sorted by probability)
    """
    # Tokenize and create features
    tokens = simple_tokenize(text)
    if not tokens:
        # Return random predictions if no text
        return np.random.choice(num_classes, size=top_k, replace=False)
    
    trigrams = padded_trigrams(tokens)
    vw_line = f"1 | {' '.join(trigrams)}"
    
    # Get predictions
    try:
        example = parser.parse_line(vw_line)
        prediction = workspace.predict_one(example)
        
        # Convert to probability array
        if isinstance(prediction, list):
            prob_array = np.array(prediction, dtype=np.float32)
        else:
            # Fallback: one-hot encoding
            prob_array = np.zeros(num_classes, dtype=np.float32)
            prob_array[int(prediction)] = 1.0
        
        # Get top-k indices
        top_k_indices = np.argsort(prob_array)[::-1][:top_k]
        return top_k_indices
        
    except Exception as e:
        print(f"Error getting predictions: {e}")
        return np.random.choice(num_classes, size=top_k, replace=False)


def evaluate_predictions(predictions, ground_truth, k=10):
    """
    Evaluate predictions using Hit Rate and NDCG metrics.
    
    Args:
        predictions: List of arrays, each containing top-k predicted indices
        ground_truth: List of true class indices
        k: Number of top predictions to consider
        
    Returns:
        Dictionary with 'hit_rate' and 'ndcg' metrics
    """
    assert len(predictions) == len(ground_truth), "Predictions and ground truth must have same length"
    
    hits = 0
    ndcg_sum = 0.0
    
    for pred_indices, true_idx in zip(predictions, ground_truth):
        # Hit Rate: 1 if true class in top-k, 0 otherwise
        if true_idx in pred_indices[:k]:
            hits += 1
            # NDCG: 1 / log2(rank + 2) where rank is 0-indexed position
            rank = np.where(pred_indices[:k] == true_idx)[0][0]
            ndcg_sum += 1.0 / np.log2(rank + 2)
    
    n = len(predictions)
    hit_rate = hits / n
    ndcg = ndcg_sum / n
    
    return {
        'hit_rate': hit_rate,
        'ndcg': ndcg,
        'num_samples': n
    }


def main():
    parser = argparse.ArgumentParser(
        description='Train and evaluate simple VW baseline'
    )
    parser.add_argument(
        '--train_file',
        type=str,
        required=True,
        help='Path to VW training file'
    )
    parser.add_argument(
        '--predictions_file',
        type=str,
        required=True,
        help='Path to pickled predictions file'
    )
    parser.add_argument(
        '--ground_truth_file',
        type=str,
        required=True,
        help='Path to pickled ground truth file'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=5000,
        help='Number of subreddit classes'
    )
    parser.add_argument(
        '--model_output',
        type=str,
        default='vw_model.bin',
        help='Path to save trained model'
    )
    
    args = parser.parse_args()
    
    # Train model
    print("="*60)
    print("SIMPLE BASELINE: VW Logistic Regression on Trigrams")
    print("="*60)
    workspace, vw_parser = train_vw_model(
        args.train_file,
        args.num_classes,
        args.model_output
    )
    
    # Load predictions and ground truth
    print("\nLoading evaluation data...")
    with open(args.predictions_file, 'rb') as f:
        predictions = pickle.load(f)
    
    with open(args.ground_truth_file, 'rb') as f:
        ground_truth = pickle.load(f)
    
    # Evaluate on dev and test sets
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    for split in ['dev', 'test']:
        if split in predictions and split in ground_truth:
            print(f"\n{split.upper()} SET:")
            metrics = evaluate_predictions(
                predictions[split],
                ground_truth[split],
                k=10
            )
            print(f"  Hit Rate @ 10:  {metrics['hit_rate']:.4f}  ({metrics['hit_rate']*100:.2f}%)")
            print(f"  NDCG @ 10:      {metrics['ndcg']:.4f}")
            print(f"  Samples:        {metrics['num_samples']}")


if __name__ == "__main__":
    main()
