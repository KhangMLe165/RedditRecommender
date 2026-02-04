#!/usr/bin/env python3
"""
REVISED: Data processing script for Engage Corpus with proper NCF full history.

KEY CHANGES FROM PREVIOUS VERSION:
1. VW Training: Jan-June text → First July-Sept interaction (proxy task)
2. NCF Training: ALL Jan-Sept interactions (full collaborative filtering graph)
3. Context Map: ONE static 5000-dim vector per user (not per interaction)
4. NCF Evaluation: Oct-Dec interactions using learned embeddings + static context

This properly replicates the paper's approach:
- NCF learns from full interaction history (collaborative patterns)
- VW provides static text-based context (one vector per user)
- Evaluation tests on future interactions (Oct-Dec)
"""

import json
import os
import re
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm
import argparse
import random
from datetime import datetime
import pickle
import vowpal_wabbit_next as vw

# Configuration
CHUNK_SIZE = 10000
RANDOM_SEED = 42
NUM_SUBREDDITS = 5000

# Date ranges (Unix timestamps)
START_2019 = 1546300800      # Jan 1, 2019
CONTEXT_END = 1561939199     # June 30, 2019 (VW context window)
VW_TARGET_START = 1561939200 # July 1, 2019 (VW target start)
VW_TARGET_END = 1569887999   # Sept 30, 2019 (NCF training end)
EVAL_START = 1569888000      # Oct 1, 2019 (Evaluation period start)
END_2019 = 1577836799        # Dec 31, 2019

def simple_tokenize(text):
    """Simple whitespace tokenizer with lowercasing."""
    if not text:
        return []
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []
    return text.split()

def padded_trigrams(tokens, pad_token="PAD", pad_size=2):
    """Create padded word trigrams for VW format."""
    if not tokens:
        return ["PAD_PAD_PAD"]
    
    padded = [pad_token] * pad_size + tokens + [pad_token] * pad_size
    trigrams = []
    for i in range(len(padded) - 2):
        trigram = "_".join(padded[i:i+3])
        trigrams.append(trigram)
    return trigrams

def load_all_data_files(data_dir):
    """Load all data files and yield user records."""
    json_files = sorted([f for f in os.listdir(data_dir) 
                        if f.startswith('data') and f.endswith('.json')])
    
    print(f"Found {len(json_files)} data files")
    
    for json_file in tqdm(json_files, desc="Loading data files"):
        json_path = os.path.join(data_dir, json_file)
        with open(json_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        user = json.loads(line)
                        yield user
                    except json.JSONDecodeError:
                        continue

def filter_user_by_periods(user):
    """
    Filter user's posts/comments into periods:
    - context: Jan-June 2019 (for VW features)
    - vw_target: July-Sept 2019 (for VW training target)
    - ncf_train: Jan-Sept 2019 (for NCF collaborative filtering)
    - eval: Oct-Dec 2019 (for final evaluation)
    """
    context_posts = []
    context_comments = []
    vw_target_posts = []
    vw_target_comments = []
    ncf_train_posts = []
    ncf_train_comments = []
    eval_posts = []
    eval_comments = []
    
    for post in user.get('posts', []):
        timestamp = post.get('created_utc', 0)
        if START_2019 <= timestamp <= CONTEXT_END:
            context_posts.append(post)
            ncf_train_posts.append(post)  # Jan-June is part of NCF training
        elif VW_TARGET_START <= timestamp <= VW_TARGET_END:
            vw_target_posts.append(post)
            ncf_train_posts.append(post)  # July-Sept is part of NCF training
        elif EVAL_START <= timestamp <= END_2019:
            eval_posts.append(post)
    
    for comment in user.get('comments', []):
        timestamp = comment.get('created_utc', 0)
        if START_2019 <= timestamp <= CONTEXT_END:
            context_comments.append(comment)
            ncf_train_comments.append(comment)
        elif VW_TARGET_START <= timestamp <= VW_TARGET_END:
            vw_target_comments.append(comment)
            ncf_train_comments.append(comment)
        elif EVAL_START <= timestamp <= END_2019:
            eval_comments.append(comment)
    
    return {
        'user_number': user['user_number'],
        'context': {
            'posts': sorted(context_posts, key=lambda x: x['created_utc']),
            'comments': sorted(context_comments, key=lambda x: x['created_utc'])
        },
        'vw_target': {
            'posts': sorted(vw_target_posts, key=lambda x: x['created_utc']),
            'comments': sorted(vw_target_comments, key=lambda x: x['created_utc'])
        },
        'ncf_train': {
            'posts': sorted(ncf_train_posts, key=lambda x: x['created_utc']),
            'comments': sorted(ncf_train_comments, key=lambda x: x['created_utc'])
        },
        'eval': {
            'posts': sorted(eval_posts, key=lambda x: x['created_utc']),
            'comments': sorted(eval_comments, key=lambda x: x['created_utc'])
        }
    }

def pass1_collect_subreddit_stats(data_dir, output_dir):
    """
    First pass: Collect subreddit statistics to find top 5000.
    Count number of unique users per subreddit.
    Caches result to avoid recomputation.
    """
    cache_file = os.path.join(output_dir, 'cache_subreddit_stats.pkl')
    
    if os.path.exists(cache_file):
        print("\n" + "="*60)
        print("PASS 1: Loading cached subreddit statistics")
        print("="*60)
        with open(cache_file, 'rb') as f:
            result = pickle.load(f)
        print(f"Loaded {len(result[0])} top subreddits from cache")
        return result
    
    print("\n" + "="*60)
    print("PASS 1: Collecting subreddit statistics")
    print("="*60)
    
    subreddit_users = defaultdict(set)  # subreddit -> set of user_numbers
    
    for user in load_all_data_files(data_dir):
        context_user, eval_user = filter_user_by_periods(user)
        
        user_num = user['user_number']
        
        # Count from both context and eval periods
        for post in context_user['posts'] + eval_user['posts']:
            subreddit_users[post['subreddit']].add(user_num)
        
        for comment in context_user['comments'] + eval_user['comments']:
            subreddit_users[comment['subreddit']].add(user_num)
    
    # Convert to counts and sort
    subreddit_counts = [(sub, len(users)) for sub, users in subreddit_users.items()]
    subreddit_counts.sort(key=lambda x: x[1], reverse=True)
    
    # Take top 5000
    top_5000 = [sub for sub, count in subreddit_counts[:5000]]
    top_5000_set = set(top_5000)
    
    print(f"Total subreddits found: {len(subreddit_counts)}")
    print(f"Top 5000 largest subreddits selected")
    print(f"Top 10 subreddits by user count:")
    for i, (sub, count) in enumerate(subreddit_counts[:10], 1):
        print(f"  {i}. r/{sub}: {count} users")
    
    result = (top_5000_set, top_5000)
    
    # Cache result
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)
    print(f"✓ Cached subreddit stats to: {cache_file}")
    
    return result

def pass2_select_users(data_dir, top_5000_subreddits, output_dir):
    """
    Second pass: Select 6% of users who have interactions in top 5000 subreddits.
    Caches result to avoid recomputation.
    Returns set of selected user numbers.
    """
    cache_file = os.path.join(output_dir, 'cache_selected_users.pkl')
    
    if os.path.exists(cache_file):
        print("\n" + "="*60)
        print("PASS 2: Loading cached user selection")
        print("="*60)
        with open(cache_file, 'rb') as f:
            selected_users = pickle.load(f)
        print(f"Loaded {len(selected_users)} selected users from cache")
        return selected_users
    
    print("\n" + "="*60)
    print("PASS 2: Selecting 6% of users")
    print("="*60)
    
    random.seed(RANDOM_SEED)
    
    eligible_users = []
    
    for user in load_all_data_files(data_dir):
        context_user, eval_user = filter_user_by_periods(user)
        
        # Check if user has any interactions in top 5000 subreddits
        has_interaction = False
        
        for post in context_user['posts'] + eval_user['posts']:
            if post['subreddit'] in top_5000_subreddits:
                has_interaction = True
                break
        
        if not has_interaction:
            for comment in context_user['comments'] + eval_user['comments']:
                if comment['subreddit'] in top_5000_subreddits:
                    has_interaction = True
                    break
        
        if has_interaction:
            eligible_users.append(user['user_number'])
    
    print(f"Eligible users: {len(eligible_users)}")
    
    # Sample 6%
    sample_size = int(len(eligible_users) * 0.06)
    selected_users = set(random.sample(eligible_users, sample_size))
    
    print(f"Selected {len(selected_users)} users (6% sample)")
    
    # Cache result
    with open(cache_file, 'wb') as f:
        pickle.dump(selected_users, f)
    print(f"✓ Cached user selection to: {cache_file}")
    
    return selected_users

def pass3_prepare_vw_data(data_dir, top_5000_subreddits, subreddit_list, 
                          selected_users, output_dir):
    """
    REVISED Pass 3: Prepare VW training data.
    
    Task: Predict FIRST July-Sept interaction using Jan-June text.
    This is the proxy task that prevents test leakage.
    """
    cache_file = os.path.join(output_dir, 'vw_data', 'train.vw')
    
    if os.path.exists(cache_file):
        print("\n" + "="*60)
        print("PASS 3: VW training data already exists")
        print("="*60)
        return cache_file
    
    print("\n" + "="*60)
    print("PASS 3: Preparing VW training data (Jan-June → July-Sept)")
    print("="*60)
    
    vw_dir = os.path.join(output_dir, 'vw_data')
    os.makedirs(vw_dir, exist_ok=True)
    
    subreddit2idx = {sub: i for i, sub in enumerate(subreddit_list)}
    
    vw_train_file = open(cache_file, 'w', encoding='utf-8')
    
    examples_written = 0
    skipped_users = 0
    
    for user in tqdm(load_all_data_files(data_dir), desc="Creating VW training data"):
        if user['user_number'] not in selected_users:
            continue
        
        periods = filter_user_by_periods(user)
        
        # Get Jan-June context text
        context_items = periods['context']['posts'] + periods['context']['comments']
        if not context_items:
            skipped_users += 1
            continue
        
        # Collect all text
        all_text = []
        for item in context_items:
            subreddit = item.get('subreddit', '')
            if subreddit in top_5000_subreddits:
                title = item.get('title', '')
                body = item.get('body', '') or item.get('selftext', '')
                all_text.append(title + ' ' + body)
        
        if not all_text:
            skipped_users += 1
            continue
        
        combined_text = ' '.join(all_text)
        tokens = simple_tokenize(combined_text)
        
        if not tokens:
            skipped_users += 1
            continue
        
        trigrams = padded_trigrams(tokens)
        
        # Get FIRST July-Sept interaction as target (proxy task)
        vw_target_items = periods['vw_target']['posts'] + periods['vw_target']['comments']
        if not vw_target_items:
            skipped_users += 1
            continue
        
        # Sort by timestamp and get first
        vw_target_items.sort(key=lambda x: x['created_utc'])
        first_target = vw_target_items[0]
        target_subreddit = first_target.get('subreddit')
        
        if target_subreddit not in subreddit2idx:
            skipped_users += 1
            continue
        
        target_idx = subreddit2idx[target_subreddit]
        
        # Write VW example
        # Format: label | features
        vw_train_file.write(f"{target_idx} | {' '.join(trigrams)}\n")
        examples_written += 1
    
    vw_train_file.close()
    
    print(f"\nVW training statistics:")
    print(f"  Examples written: {examples_written}")
    print(f"  Skipped users: {skipped_users}")
    print(f"  Target period: July-Sept 2019")
    print(f"  Context period: Jan-June 2019")
    
    return cache_file

def pass4_train_vw_model(vw_train_file, output_dir, num_subreddits):
    """Train VW model (same as before)."""
    model_file = os.path.join(output_dir, 'vw_model.bin')
    
    if os.path.exists(model_file):
        print("\n" + "="*60)
        print("PASS 4: Loading cached VW model")
        print("="*60)
        with open(model_file, 'rb') as f:
            model_bytes = f.read()
        workspace = vw.Workspace(
            args=[
                f"--oaa={num_subreddits}",
                "--loss_function=logistic",
                "--probabilities",
                "--quiet"
            ],
            model_data=model_bytes
        )
        parser = vw.TextFormatParser(workspace)
        print("✓ Model loaded successfully")
        return workspace, parser
    
    print("\n" + "="*60)
    print("PASS 4: Training VW model")
    print("="*60)
    
    # Train
    workspace = vw.Workspace([
        f"--oaa={num_subreddits}",
        "--loss_function=logistic",
        "--probabilities",  # Output probabilities
        "--passes=1",  # Multiple passes over data
        "--cache",  # Use cache for faster training
        "--kill_cache"  # Clear cache after training
    ])
    parser = vw.TextFormatParser(workspace)
    
    with open(vw_train_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Training VW"):
            line = line.strip()
            if line:
                try:
                    example = parser.parse_line(line)
                    workspace.learn_one(example)
                except Exception as e:
                    print(f"Error during training: {e}")

    # Serialize and save model
    model_bytes = workspace.serialize()
    with open(model_file, 'wb') as f:
        f.write(model_bytes)
    print(f"✓ Model saved to: {model_file}")

    # Reload model for predictions
    workspace = vw.Workspace(
        args=[
            f"--oaa={num_subreddits}",
            "--loss_function=logistic",
            "--probabilities",
            "--quiet"
        ],
        model_data=model_bytes
    )
    parser = vw.TextFormatParser(workspace)

    return workspace, parser

def get_vw_probability_scores(workspace, parser, text, subreddit2idx):
    """Get VW probability scores for all subreddits."""
    tokens = simple_tokenize(text)
    if not tokens:
        return np.zeros(len(subreddit2idx), dtype=np.float32)

    trigrams = padded_trigrams(tokens)
    vw_line = f"1 | {' '.join(trigrams)}"

    # Parse and predict
    example = parser.parse_line(vw_line)
    prediction = workspace.predict_one(example)

    # With --oaa and --probabilities, prediction is a List[float]
    # with probabilities in order (one per class)
    if isinstance(prediction, list):
        prob_array = np.array(prediction, dtype=np.float32)
    else:
        # Fallback: single prediction (shouldn't happen with --probabilities)
        prob_array = np.zeros(len(subreddit2idx), dtype=np.float32)
        prob_array[int(prediction)] = 1.0

    return prob_array

def pass5_generate_ncf_data_and_context(data_dir, top_5000_subreddits, subreddit_list,
                                       selected_users, workspace, parser, output_dir):
    """
    REVISED Pass 5: Generate NCF training data with full history + static context map.
    
    Outputs:
    1. train.tsv - ALL Jan-Sept interactions (full NCF training graph)
    2. dev.tsv - First Oct-Dec interaction per user
    3. test.tsv - Second Oct-Dec interaction per user
    4. user_context_map.npy - ONE 5000-dim vector per user (static block)
    5. vw_predictions.pkl - VW baseline predictions
    6. ground_truth.pkl - Ground truth for evaluation
    """
    ncf_dir = os.path.join(output_dir, 'ncf_data')
    os.makedirs(ncf_dir, exist_ok=True)
    
    required_files = [
        os.path.join(ncf_dir, 'train.tsv'),
        os.path.join(ncf_dir, 'dev.tsv'),
        os.path.join(ncf_dir, 'test.tsv'),
        os.path.join(ncf_dir, 'user_context_map.npy'),
        os.path.join(output_dir, 'vw_predictions.pkl'),
        os.path.join(output_dir, 'ground_truth.pkl'),
        os.path.join(output_dir, 'user_mapping.json')
    ]
    
    if all(os.path.exists(f) for f in required_files):
        print("\n" + "="*60)
        print("PASS 5: All output files already exist")
        print("="*60)
        return
    
    print("\n" + "="*60)
    print("PASS 5: Generating NCF data + static context map")
    print("="*60)
    
    subreddit2idx = {sub: i for i, sub in enumerate(subreddit_list)}
    
    # Open output files
    train_file = open(os.path.join(ncf_dir, 'train.tsv'), 'w', encoding='utf-8')
    dev_file = open(os.path.join(ncf_dir, 'dev.tsv'), 'w', encoding='utf-8')
    test_file = open(os.path.join(ncf_dir, 'test.tsv'), 'w', encoding='utf-8')
    
    # User mapping
    user_id_counter = 0
    user_num_to_id = {}
    
    # Pre-allocate context map (will fill as we go)
    max_users = len(selected_users)
    user_context_map = np.zeros((max_users, NUM_SUBREDDITS), dtype=np.float32)
    
    # Statistics
    train_interactions = 0
    dev_count = 0
    test_count = 0
    skipped_users = 0
    
    # VW predictions and ground truth
    vw_predictions = {'dev': [], 'test': []}
    ground_truth = {'dev': [], 'test': []}
    
    print("Processing users...")
    for user in tqdm(load_all_data_files(data_dir), desc="Generating NCF data"):
        if user['user_number'] not in selected_users:
            continue
        
        periods = filter_user_by_periods(user)
        user_num = user['user_number']
        
        # Assign user ID
        if user_num not in user_num_to_id:
            user_num_to_id[user_num] = user_id_counter
            user_id_counter += 1
        user_id = user_num_to_id[user_num]
        
        # 1. Generate STATIC context vector (Jan-June text → VW scores)
        context_items = periods['context']['posts'] + periods['context']['comments']
        if context_items:
            all_text = []
            for item in context_items:
                subreddit = item.get('subreddit', '')
                if subreddit in top_5000_subreddits:
                    title = item.get('title', '')
                    body = item.get('body', '') or item.get('selftext', '')
                    all_text.append(title + ' ' + body)
            
            if all_text:
                combined_text = ' '.join(all_text)
                context_vector = get_vw_probability_scores(workspace, parser, combined_text, subreddit2idx)
                user_context_map[user_id] = context_vector
        
        # 2. Write ALL Jan-Sept interactions to train.tsv (NCF full history)
        ncf_train_items = periods['ncf_train']['posts'] + periods['ncf_train']['comments']
        user_train_subreddits = set()  # Avoid duplicates
        
        for item in ncf_train_items:
            subreddit = item.get('subreddit')
            if subreddit in subreddit2idx:
                sub_idx = subreddit2idx[subreddit]
                if sub_idx not in user_train_subreddits:
                    train_file.write(f"{user_id}\t{sub_idx}\n")
                    user_train_subreddits.add(sub_idx)
                    train_interactions += 1
        
        # 3. Get Oct-Dec evaluation interactions
        eval_items = periods['eval']['posts'] + periods['eval']['comments']
        if len(eval_items) < 2:
            skipped_users += 1
            continue
        
        # Sort by timestamp
        eval_items.sort(key=lambda x: x['created_utc'])
        
        first_eval = eval_items[0]
        second_eval = eval_items[1]
        
        first_sub = first_eval.get('subreddit')
        second_sub = second_eval.get('subreddit')
        
        # Get top-10 VW predictions for evaluation
        top_10_indices = np.argsort(user_context_map[user_id])[::-1][:10].tolist()
        
        # Dev: first Oct-Dec interaction
        if first_sub in subreddit2idx:
            sub_idx = subreddit2idx[first_sub]
            dev_file.write(f"{user_id}\t{sub_idx}\n")
            dev_count += 1
            vw_predictions['dev'].append(top_10_indices)
            ground_truth['dev'].append(sub_idx)
        
        # Test: second Oct-Dec interaction
        if second_sub in subreddit2idx:
            sub_idx = subreddit2idx[second_sub]
            test_file.write(f"{user_id}\t{sub_idx}\n")
            test_count += 1
            vw_predictions['test'].append(top_10_indices)
            ground_truth['test'].append(sub_idx)
    
    # Close files
    train_file.close()
    dev_file.close()
    test_file.close()
    
    # Trim context map to actual number of users
    user_context_map = user_context_map[:user_id_counter]
    
    # Save outputs
    np.save(os.path.join(ncf_dir, 'user_context_map.npy'), user_context_map)
    
    with open(os.path.join(output_dir, 'vw_predictions.pkl'), 'wb') as f:
        pickle.dump(vw_predictions, f)
    
    with open(os.path.join(output_dir, 'ground_truth.pkl'), 'wb') as f:
        pickle.dump(ground_truth, f)
    
    with open(os.path.join(output_dir, 'user_mapping.json'), 'w') as f:
        json.dump({
            'user_num_to_id': {str(k): v for k, v in user_num_to_id.items()},
            'num_users': user_id_counter
        }, f, indent=2)
    
    print(f"\n✅ NCF Data Generation Complete!")
    print(f"  NCF training interactions: {train_interactions:,}")
    print(f"  Dev samples: {dev_count}")
    print(f"  Test samples: {test_count}")
    print(f"  Unique users: {user_id_counter}")
    print(f"  Context map shape: ({user_id_counter}, {NUM_SUBREDDITS})")
    print(f"  Skipped users: {skipped_users}")
    
    print(f"\n✅ Files saved:")
    print(f"  {os.path.join(ncf_dir, 'train.tsv')}")
    print(f"  {os.path.join(ncf_dir, 'dev.tsv')}")
    print(f"  {os.path.join(ncf_dir, 'test.tsv')}")
    print(f"  {os.path.join(ncf_dir, 'user_context_map.npy')}")
    print(f"  {os.path.join(output_dir, 'vw_predictions.pkl')}")
    print(f"  {os.path.join(output_dir, 'ground_truth.pkl')}")

def main():
    parser = argparse.ArgumentParser(
        description='Process Reddit data for Engage Corpus replication with Vowpal Wabbit (Baseline 0)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        default='converted_data',
        help='Directory containing data00.json through data23.json'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='engage_corpus_processed_v2',
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--force_recompute',
        action='store_true',
        help='Force recomputation even if cached files exist'
    )
    
    args = parser.parse_args()
    
    # Verify data directory exists
    if not os.path.exists(args.data_dir):
        print(f"ERROR: Data directory not found: {args.data_dir}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Clear cache if force recompute
    if args.force_recompute:
        print("Force recompute enabled - removing cached files...")
        cache_files = [
            os.path.join(args.output_dir, 'cache_subreddit_stats.pkl'),
            os.path.join(args.output_dir, 'cache_selected_users.pkl'),
            os.path.join(args.output_dir, 'vw_data', 'train.vw'),
            os.path.join(args.output_dir, 'vw_model.bin')
        ]
        for cf in cache_files:
            if os.path.exists(cf):
                os.remove(cf)
                print(f"  Removed: {cf}")
    
    print("="*60)
    print("ENGAGE CORPUS DATA PROCESSING WITH VOWPAL WABBIT")
    print("BASELINE 0: Simplified Single-Window Approach")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Context window: Jan-June 2019")
    print(f"Prediction targets: Oct-Dec 2019")
    print("="*60)
    
    # Pass 1: Collect subreddit statistics and select top 5000
    top_5000_subreddits, subreddit_list = pass1_collect_subreddit_stats(args.data_dir, args.output_dir)
    
    # Pass 2: Select 6% of users
    selected_users = pass2_select_users(args.data_dir, top_5000_subreddits, args.output_dir)
    
    # Pass 3: Prepare VW training data
    vw_train_file = pass3_prepare_vw_data(args.data_dir, top_5000_subreddits, 
                                          subreddit_list, selected_users, args.output_dir)
    
    # Pass 4: Train VW model
    workspace, parser = pass4_train_vw_model(vw_train_file, args.output_dir, num_subreddits=NUM_SUBREDDITS)
    
    # Pass 5: Generate predictions and context for NCF
    pass5_generate_ncf_data_and_context(args.data_dir, top_5000_subreddits, subreddit_list,
                                          selected_users, workspace, parser, args.output_dir)
    
    # Save metadata
    metadata = {
        'num_subreddits': len(subreddit_list),
        'num_users': len(selected_users),
        'date_range': '2019-01-01 to 2019-12-31',
        'context_period': '2019-01-01 to 2019-06-30',
        'eval_period': '2019-10-01 to 2019-12-31',
        'model': 'Vowpal Wabbit OAA logistic regression (Baseline 0)',
        'method': 'single_window_jan_june_context',
        'approach': 'simplified',
        'examples_per_user': '1 training, 1 dev, 1 test'
    }
    
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save subreddit mapping
    with open(os.path.join(args.output_dir, 'subreddit_mapping.json'), 'w') as f:
        json.dump({
            'subreddits': subreddit_list,
            'subreddit2idx': {sub: i for i, sub in enumerate(subreddit_list)},
            'num_subreddits': len(subreddit_list)
        }, f, indent=2)
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    print(f"\nOutput structure:")
    print(f"  {args.output_dir}/")
    print(f"    vw_data/")
    print(f"      train.vw           - VW training data (1 example per user)")
    print(f"    ncf_data/")
    print(f"      dev.tsv            - Dev user-subreddit pairs")
    print(f"      test.tsv           - Test user-subreddit pairs")
    print(f"      dev_context.npy    - VW probability context (N, 5000)")
    print(f"      test_context.npy   - VW probability context (N, 5000)")
    print(f"    vw_model.bin         - Trained VW model")
    print(f"    vw_predictions.pkl   - VW baseline predictions (top-10 indices)")
    print(f"    ground_truth.pkl     - Ground truth labels")
    print(f"    metadata.json")
    print(f"    subreddit_mapping.json")
    print(f"    user_mapping.json")
    print(f"    cache_*.pkl          - Cached intermediate results")
    
    print("\nNext steps:")
    print("1. Evaluate VW baseline:")
    print(f"   python evaluate.py --predictions {args.output_dir}/vw_predictions.pkl --ground_truth {args.output_dir}/ground_truth.pkl")
    print("2. Upload the engage_corpus_processed folder to Google Drive")
    print("3. Run train_ncf_models.ipynb to train neural models")
    print("4. Use evaluate.py to compare all models")

if __name__ == "__main__":
    main()
