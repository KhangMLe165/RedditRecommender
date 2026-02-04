#!/usr/bin/env python3
"""
Process Engage Corpus v4 with stricter filtering and TF-IDF.

New requirements:
1. Only include data from 2000 most popular subreddits in 2019
2. Only include users with 50-5000 (inclusive) posts+comments in train period
3. Users must have data in both Jan-June AND July-Sep 2019
4. New splits: Train (Jan-June predicting July-Sept), Dev (Oct & Dec), Test (Nov)
5. TF-IDF calculation for users and subreddits
6. Multi-label training (all positive interactions)

As in the v3 baseline, we process incrementally, avoiding loading all of the source data into
memory at once.

The TF-IDF text filtering is necessary to ensure that we are not exceeding the context window
of the embedding model.
"""

import json
import os
import re
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm
import argparse
import random
import pickle
from math import log

# Configuration
CHUNK_SIZE = 10000
RANDOM_SEED = 42
NUM_SUBREDDITS = 2000  # Top 2000 instead of 5000

# Date ranges (Unix timestamps) - NEW SPLITS
START_2019 = 1546300800      # Jan 1, 2019
TRAIN_INPUT_END = 1561939199     # June 30, 2019 (train input period)
TRAIN_TARGET_START = 1561939200  # July 1, 2019
TRAIN_TARGET_END = 1569887999    # Sept 30, 2019 (train target period)
DEV_START_OCT = 1569888000       # Oct 1, 2019
DEV_END_OCT = 1572566399         # Oct 31, 2019
TEST_START_NOV = 1572566400      # Nov 1, 2019
TEST_END_NOV = 1575158399        # Nov 30, 2019
DEV_START_DEC = 1575158400       # Dec 1, 2019
END_2019 = 1577836799            # Dec 31, 2019

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
    Filter user's posts/comments into NEW periods:
    - train_input: Jan-June 2019 (features for prediction)
    - train_target: July-Sept 2019 (what we predict)
    - dev: Oct & Dec 2019
    - test: Nov 2019
    """
    train_input_posts = []
    train_input_comments = []
    train_target_posts = []
    train_target_comments = []
    dev_posts = []
    dev_comments = []
    test_posts = []
    test_comments = []
    
    for post in user.get('posts', []):
        timestamp = post.get('created_utc', 0)
        if START_2019 <= timestamp <= TRAIN_INPUT_END:
            train_input_posts.append(post)
        elif TRAIN_TARGET_START <= timestamp <= TRAIN_TARGET_END:
            train_target_posts.append(post)
        elif (DEV_START_OCT <= timestamp <= DEV_END_OCT) or (DEV_START_DEC <= timestamp <= END_2019):
            dev_posts.append(post)
        elif TEST_START_NOV <= timestamp <= TEST_END_NOV:
            test_posts.append(post)
    
    for comment in user.get('comments', []):
        timestamp = comment.get('created_utc', 0)
        if START_2019 <= timestamp <= TRAIN_INPUT_END:
            train_input_comments.append(comment)
        elif TRAIN_TARGET_START <= timestamp <= TRAIN_TARGET_END:
            train_target_comments.append(comment)
        elif (DEV_START_OCT <= timestamp <= DEV_END_OCT) or (DEV_START_DEC <= timestamp <= END_2019):
            dev_comments.append(comment)
        elif TEST_START_NOV <= timestamp <= TEST_END_NOV:
            test_comments.append(comment)
    
    return {
        'user_number': user['user_number'],
        'train_input': {
            'posts': sorted(train_input_posts, key=lambda x: x['created_utc']),
            'comments': sorted(train_input_comments, key=lambda x: x['created_utc'])
        },
        'train_target': {
            'posts': sorted(train_target_posts, key=lambda x: x['created_utc']),
            'comments': sorted(train_target_comments, key=lambda x: x['created_utc'])
        },
        'dev': {
            'posts': sorted(dev_posts, key=lambda x: x['created_utc']),
            'comments': sorted(dev_comments, key=lambda x: x['created_utc'])
        },
        'test': {
            'posts': sorted(test_posts, key=lambda x: x['created_utc']),
            'comments': sorted(test_comments, key=lambda x: x['created_utc'])
        }
    }

def pass1_collect_subreddit_stats(data_dir, output_dir):
    """First pass: Collect subreddit statistics to find top 2000."""
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
    print("PASS 1: Collecting subreddit statistics (top 2000)")
    print("="*60)
    
    subreddit_users = defaultdict(set)
    
    for user in load_all_data_files(data_dir):
        periods = filter_user_by_periods(user)
        user_num = user['user_number']
        
        # Count from all periods
        for period_name in ['train_input', 'train_target', 'dev', 'test']:
            period = periods[period_name]
            for post in period['posts']:
                subreddit_users[post['subreddit']].add(user_num)
            for comment in period['comments']:
                subreddit_users[comment['subreddit']].add(user_num)
    
    subreddit_counts = [(sub, len(users)) for sub, users in subreddit_users.items()]
    subreddit_counts.sort(key=lambda x: x[1], reverse=True)
    
    top_2000 = [sub for sub, count in subreddit_counts[:2000]]
    top_2000_set = set(top_2000)
    
    print(f"Total subreddits found: {len(subreddit_counts)}")
    print(f"Top 2000 largest subreddits selected")
    print(f"Top 10 subreddits by user count:")
    for i, (sub, count) in enumerate(subreddit_counts[:10], 1):
        print(f"  {i}. r/{sub}: {count} users")
    
    result = (top_2000_set, top_2000)
    
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)
    print(f"✓ Cached subreddit stats to: {cache_file}")
    
    return result

def pass2_select_users(data_dir, top_2000_subreddits, output_dir):
    """
    Second pass: Select users with stricter filtering.
    
    Requirements:
    1. 50-5000 posts+comments in train period (Jan-Sept)
    2. Activity in BOTH Jan-June AND July-Sept
    3. At least one interaction in top 2000 subreddits
    """
    cache_file = os.path.join(output_dir, 'cache_selected_users.pkl')
    
    if os.path.exists(cache_file):
        print("\n" + "="*60)
        print("PASS 2: Loading cached selected users")
        print("="*60)
        with open(cache_file, 'rb') as f:
            selected_users = pickle.load(f)
        print(f"Loaded {len(selected_users)} selected users from cache")
        return selected_users
    
    print("\n" + "="*60)
    print("PASS 2: Selecting users with strict filtering")
    print("="*60)
    
    selected_users = set()
    eligible_count = 0
    activity_filter_count = 0
    volume_filter_count = 0
    
    for user in tqdm(load_all_data_files(data_dir), desc="Filtering users"):
        user_num = user['user_number']
        periods = filter_user_by_periods(user)
        
        # Check 1: Activity in both Jan-June AND July-Sept
        train_input_items = periods['train_input']['posts'] + periods['train_input']['comments']
        train_target_items = periods['train_target']['posts'] + periods['train_target']['comments']
        
        has_jan_june = len(train_input_items) > 0
        has_july_sept = len(train_target_items) > 0
        
        if not (has_jan_june and has_july_sept):
            activity_filter_count += 1
            continue
        
        # Check 2: 50-5000 posts+comments in train period
        total_train_items = len(train_input_items) + len(train_target_items)
        
        if not (50 <= total_train_items <= 5000):
            volume_filter_count += 1
            continue
        
        # Check 3: At least one interaction in top 2000 subreddits
        has_top_2000_activity = False
        for item in train_input_items + train_target_items:
            if item.get('subreddit') in top_2000_subreddits:
                has_top_2000_activity = True
                break
        
        if not has_top_2000_activity:
            continue
        
        eligible_count += 1
        selected_users.add(user_num)
    
    print(f"Filtered out {activity_filter_count} users without activity in both periods")
    print(f"Filtered out {volume_filter_count} users outside 50-5000 volume range")
    print(f"Eligible users: {eligible_count}")
    print(f"Selected users: {len(selected_users)}")
    
    with open(cache_file, 'wb') as f:
        pickle.dump(selected_users, f)
    print(f"✓ Cached selected users to: {cache_file}")
    
    return selected_users

def pass3_compute_word_stats(data_dir, top_2000_subreddits, selected_users, output_dir):
    """
    Third pass: Compute word statistics for TF-IDF.
    
    Computes:
    - Document frequency (DF) for each word
    - Term frequency (TF) for each user and subreddit
    """
    cache_file = os.path.join(output_dir, 'cache_word_stats.pkl')
    
    if os.path.exists(cache_file):
        print("\n" + "="*60)
        print("PASS 3: Loading cached word statistics")
        print("="*60)
        with open(cache_file, 'rb') as f:
            result = pickle.load(f)
        print(f"Loaded word statistics from cache")
        return result
    
    print("\n" + "="*60)
    print("PASS 3: Computing word statistics for TF-IDF")
    print("="*60)
    
    # Track document frequency
    word_doc_freq = Counter()  # word -> number of documents containing it
    total_docs = 0
    
    # Track term frequency per user/subreddit
    user_word_freq_train = defaultdict(Counter)  # user_id -> {word: count}
    user_word_freq_dev = defaultdict(Counter)
    user_word_freq_test = defaultdict(Counter)
    
    subreddit_word_freq_train = defaultdict(Counter)  # subreddit -> {word: count}
    subreddit_word_freq_dev = defaultdict(Counter)
    subreddit_word_freq_test = defaultdict(Counter)
    
    user_num_to_id = {}
    user_id_counter = 0
    
    for user in tqdm(load_all_data_files(data_dir), desc="Computing word stats"):
        user_num = user['user_number']
        
        if user_num not in selected_users:
            continue
        
        periods = filter_user_by_periods(user)
        
        # Assign user ID
        user_id = user_id_counter
        user_num_to_id[user_num] = user_id
        user_id_counter += 1
        
        # Process train input + target (Jan-Sept)
        train_words = set()
        for post in periods['train_input']['posts'] + periods['train_target']['posts']:
            title_text = post.get('title', '')
            body_text = post.get('selftext', '')
            tokens = simple_tokenize(title_text) + simple_tokenize(body_text)
            
            # Update user TF
            for token in tokens:
                user_word_freq_train[user_id][token] += 1
            
            # Update subreddit TF
            sub = post.get('subreddit')
            if sub in top_2000_subreddits:
                for token in tokens:
                    subreddit_word_freq_train[sub][token] += 1
            
            # Track unique words for DF
            train_words.update(tokens)
        
        for comment in periods['train_input']['comments'] + periods['train_target']['comments']:
            body_text = comment.get('body', '')
            tokens = simple_tokenize(body_text)
            
            # Update user TF
            for token in tokens:
                user_word_freq_train[user_id][token] += 1
            
            # Update subreddit TF
            sub = comment.get('subreddit')
            if sub in top_2000_subreddits:
                for token in tokens:
                    subreddit_word_freq_train[sub][token] += 1
            
            # Track unique words for DF
            train_words.update(tokens)
        
        # Update DF
        for word in train_words:
            word_doc_freq[word] += 1
        total_docs += 1
        
        # Process dev period
        dev_words = set()
        for post in periods['dev']['posts']:
            title_text = post.get('title', '')
            body_text = post.get('selftext', '')
            tokens = simple_tokenize(title_text) + simple_tokenize(body_text)
            
            for token in tokens:
                user_word_freq_dev[user_id][token] += 1
            
            sub = post.get('subreddit')
            if sub in top_2000_subreddits:
                for token in tokens:
                    subreddit_word_freq_dev[sub][token] += 1
            
            dev_words.update(tokens)
        
        for comment in periods['dev']['comments']:
            body_text = comment.get('body', '')
            tokens = simple_tokenize(body_text)
            
            for token in tokens:
                user_word_freq_dev[user_id][token] += 1
            
            sub = comment.get('subreddit')
            if sub in top_2000_subreddits:
                for token in tokens:
                    subreddit_word_freq_dev[sub][token] += 1
            
            dev_words.update(tokens)
        
        # Process test period
        test_words = set()
        for post in periods['test']['posts']:
            title_text = post.get('title', '')
            body_text = post.get('selftext', '')
            tokens = simple_tokenize(title_text) + simple_tokenize(body_text)
            
            for token in tokens:
                user_word_freq_test[user_id][token] += 1
            
            sub = post.get('subreddit')
            if sub in top_2000_subreddits:
                for token in tokens:
                    subreddit_word_freq_test[sub][token] += 1
            
            test_words.update(tokens)
        
        for comment in periods['test']['comments']:
            body_text = comment.get('body', '')
            tokens = simple_tokenize(body_text)
            
            for token in tokens:
                user_word_freq_test[user_id][token] += 1
            
            sub = comment.get('subreddit')
            if sub in top_2000_subreddits:
                for token in tokens:
                    subreddit_word_freq_test[sub][token] += 1
            
            test_words.update(tokens)
    
    result = {
        'word_doc_freq': word_doc_freq,
        'total_docs': total_docs,
        'user_word_freq_train': dict(user_word_freq_train),
        'user_word_freq_dev': dict(user_word_freq_dev),
        'user_word_freq_test': dict(user_word_freq_test),
        'subreddit_word_freq_train': dict(subreddit_word_freq_train),
        'subreddit_word_freq_dev': dict(subreddit_word_freq_dev),
        'subreddit_word_freq_test': dict(subreddit_word_freq_test),
        'user_num_to_id': user_num_to_id,
        'num_users': user_id_counter
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)
    print(f"✓ Cached word statistics to: {cache_file}")
    
    print(f"Total users: {user_id_counter}")
    print(f"Unique words: {len(word_doc_freq)}")
    print(f"Most common words: {word_doc_freq.most_common(10)}")
    
    return result

def compute_tfidf_and_filter(word_freq, word_doc_freq, total_docs, top_k=50):
    """
    Compute TF-IDF scores and return top K words.
    
    TF-IDF = TF * IDF
    where IDF = log(N / DF)
    """
    tfidf_scores = {}
    
    for word, tf in word_freq.items():
        df = word_doc_freq.get(word, 1)
        idf = log(total_docs / df)
        tfidf_scores[word] = tf * idf
    
    # Get top K words
    top_words = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    return [word for word, score in top_words], tfidf_scores

def pass4_generate_outputs(data_dir, top_2000_subreddits, subreddit_list, 
                           selected_users, word_stats, output_dir):
    """
    Fourth pass: Generate all outputs.
    
    Outputs:
    1. Text context (unfiltered) - train/dev/test
    2. Text context (TF-IDF filtered, top 50 words) - train/dev/test
    3. Interaction data (multi-label) - train/dev/test
    4. Summary statistics
    """
    print("\n" + "="*60)
    print("PASS 4: Generating outputs")
    print("="*60)
    
    # Create output directories
    text_dir = os.path.join(output_dir, 'text_context')
    text_filtered_dir = os.path.join(output_dir, 'text_context_filtered')
    ncf_dir = os.path.join(output_dir, 'ncf_data')
    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(text_filtered_dir, exist_ok=True)
    os.makedirs(ncf_dir, exist_ok=True)
    
    # Create subreddit mapping
    subreddit2idx = {sub: i for i, sub in enumerate(subreddit_list)}
    
    # Extract word stats
    word_doc_freq = word_stats['word_doc_freq']
    total_docs = word_stats['total_docs']
    user_word_freq_train = word_stats['user_word_freq_train']
    user_word_freq_dev = word_stats['user_word_freq_dev']
    user_word_freq_test = word_stats['user_word_freq_test']
    subreddit_word_freq_train = word_stats['subreddit_word_freq_train']
    subreddit_word_freq_dev = word_stats['subreddit_word_freq_dev']
    subreddit_word_freq_test = word_stats['subreddit_word_freq_test']
    user_num_to_id = word_stats['user_num_to_id']
    
    # Generate unfiltered text
    user_text_train = {}
    user_text_dev = {}
    user_text_test = {}
    subreddit_text_train = {}
    subreddit_text_dev = {}
    subreddit_text_test = {}
    
    # Generate filtered text (top 50 TF-IDF words)
    user_text_train_filtered = {}
    user_text_dev_filtered = {}
    user_text_test_filtered = {}
    subreddit_text_train_filtered = {}
    subreddit_text_dev_filtered = {}
    subreddit_text_test_filtered = {}
    
    print("Generating user text...")
    for user_id, word_freq in tqdm(user_word_freq_train.items(), desc="User train text"):
        # Unfiltered
        user_text_train[user_id] = ' '.join([word for word in word_freq.keys() for _ in range(word_freq[word])])
        
        # Filtered (top 50 TF-IDF)
        top_words, _ = compute_tfidf_and_filter(word_freq, word_doc_freq, total_docs, top_k=50)
        user_text_train_filtered[user_id] = ' '.join(top_words)
    
    for user_id, word_freq in tqdm(user_word_freq_dev.items(), desc="User dev text"):
        user_text_dev[user_id] = ' '.join([word for word in word_freq.keys() for _ in range(word_freq[word])])
        top_words, _ = compute_tfidf_and_filter(word_freq, word_doc_freq, total_docs, top_k=50)
        user_text_dev_filtered[user_id] = ' '.join(top_words)
    
    for user_id, word_freq in tqdm(user_word_freq_test.items(), desc="User test text"):
        user_text_test[user_id] = ' '.join([word for word in word_freq.keys() for _ in range(word_freq[word])])
        top_words, _ = compute_tfidf_and_filter(word_freq, word_doc_freq, total_docs, top_k=50)
        user_text_test_filtered[user_id] = ' '.join(top_words)
    
    print("Generating subreddit text...")
    for sub, word_freq in tqdm(subreddit_word_freq_train.items(), desc="Subreddit train text"):
        subreddit_text_train[sub] = ' '.join([word for word in word_freq.keys() for _ in range(word_freq[word])])
        top_words, _ = compute_tfidf_and_filter(word_freq, word_doc_freq, total_docs, top_k=50)
        subreddit_text_train_filtered[sub] = ' '.join(top_words)
    
    for sub, word_freq in tqdm(subreddit_word_freq_dev.items(), desc="Subreddit dev text"):
        subreddit_text_dev[sub] = ' '.join([word for word in word_freq.keys() for _ in range(word_freq[word])])
        top_words, _ = compute_tfidf_and_filter(word_freq, word_doc_freq, total_docs, top_k=50)
        subreddit_text_dev_filtered[sub] = ' '.join(top_words)
    
    for sub, word_freq in tqdm(subreddit_word_freq_test.items(), desc="Subreddit test text"):
        subreddit_text_test[sub] = ' '.join([word for word in word_freq.keys() for _ in range(word_freq[word])])
        top_words, _ = compute_tfidf_and_filter(word_freq, word_doc_freq, total_docs, top_k=50)
        subreddit_text_test_filtered[sub] = ' '.join(top_words)
    
    # Save text context (unfiltered)
    with open(os.path.join(text_dir, 'user_text_train.json'), 'w') as f:
        json.dump({str(k): v for k, v in user_text_train.items()}, f)
    with open(os.path.join(text_dir, 'user_text_dev.json'), 'w') as f:
        json.dump({str(k): v for k, v in user_text_dev.items()}, f)
    with open(os.path.join(text_dir, 'user_text_test.json'), 'w') as f:
        json.dump({str(k): v for k, v in user_text_test.items()}, f)
    
    with open(os.path.join(text_dir, 'subreddit_text_train.json'), 'w') as f:
        json.dump(subreddit_text_train, f)
    with open(os.path.join(text_dir, 'subreddit_text_dev.json'), 'w') as f:
        json.dump(subreddit_text_dev, f)
    with open(os.path.join(text_dir, 'subreddit_text_test.json'), 'w') as f:
        json.dump(subreddit_text_test, f)
    
    # Save text context (filtered)
    with open(os.path.join(text_filtered_dir, 'user_text_train.json'), 'w') as f:
        json.dump({str(k): v for k, v in user_text_train_filtered.items()}, f)
    with open(os.path.join(text_filtered_dir, 'user_text_dev.json'), 'w') as f:
        json.dump({str(k): v for k, v in user_text_dev_filtered.items()}, f)
    with open(os.path.join(text_filtered_dir, 'user_text_test.json'), 'w') as f:
        json.dump({str(k): v for k, v in user_text_test_filtered.items()}, f)
    
    with open(os.path.join(text_filtered_dir, 'subreddit_text_train.json'), 'w') as f:
        json.dump(subreddit_text_train_filtered, f)
    with open(os.path.join(text_filtered_dir, 'subreddit_text_dev.json'), 'w') as f:
        json.dump(subreddit_text_dev_filtered, f)
    with open(os.path.join(text_filtered_dir, 'subreddit_text_test.json'), 'w') as f:
        json.dump(subreddit_text_test_filtered, f)
    
    # Generate interaction data (multi-label)
    print("Generating interaction data...")
    
    train_interactions = []
    dev_interactions = []
    test_interactions = []
    
    for user in tqdm(load_all_data_files(data_dir), desc="Processing interactions"):
        user_num = user['user_number']
        
        if user_num not in selected_users:
            continue
        
        user_id = user_num_to_id[user_num]
        periods = filter_user_by_periods(user)
        
        # Train: ALL positive interactions in July-Sept (target period)
        train_subreddits = set()
        for item in periods['train_target']['posts'] + periods['train_target']['comments']:
            sub = item.get('subreddit')
            if sub in subreddit2idx:
                train_subreddits.add(subreddit2idx[sub])
        
        for sub_idx in train_subreddits:
            train_interactions.append((user_id, sub_idx, 1))  # label=1 for positive
        
        # Dev: ALL positive interactions in Oct & Dec
        dev_subreddits = set()
        for item in periods['dev']['posts'] + periods['dev']['comments']:
            sub = item.get('subreddit')
            if sub in subreddit2idx:
                dev_subreddits.add(subreddit2idx[sub])
        
        for sub_idx in dev_subreddits:
            dev_interactions.append((user_id, sub_idx, 1))
        
        # Test: ALL positive interactions in Nov
        test_subreddits = set()
        for item in periods['test']['posts'] + periods['test']['comments']:
            sub = item.get('subreddit')
            if sub in subreddit2idx:
                test_subreddits.add(subreddit2idx[sub])
        
        for sub_idx in test_subreddits:
            test_interactions.append((user_id, sub_idx, 1))
    
    # Save interaction data
    with open(os.path.join(ncf_dir, 'train.tsv'), 'w') as f:
        for user_id, sub_idx, label in train_interactions:
            f.write(f"{user_id}\t{sub_idx}\t{label}\n")
    
    with open(os.path.join(ncf_dir, 'dev.tsv'), 'w') as f:
        for user_id, sub_idx, label in dev_interactions:
            f.write(f"{user_id}\t{sub_idx}\t{label}\n")
    
    with open(os.path.join(ncf_dir, 'test.tsv'), 'w') as f:
        for user_id, sub_idx, label in test_interactions:
            f.write(f"{user_id}\t{sub_idx}\t{label}\n")
    
    # Save mappings
    with open(os.path.join(output_dir, 'user_mapping.json'), 'w') as f:
        json.dump({
            'user_num_to_id': {str(k): v for k, v in user_num_to_id.items()},
            'num_users': len(user_num_to_id)
        }, f, indent=2)
    
    with open(os.path.join(output_dir, 'subreddit_mapping.json'), 'w') as f:
        json.dump({
            'subreddits': subreddit_list,
            'subreddit2idx': subreddit2idx,
            'num_subreddits': len(subreddit_list)
        }, f, indent=2)
    
    # Generate summary statistics
    summary = {
        'num_users': len(user_num_to_id),
        'num_subreddits': len(subreddit_list),
        'num_unique_words': len(word_doc_freq),
        'train_interactions': len(train_interactions),
        'dev_interactions': len(dev_interactions),
        'test_interactions': len(test_interactions),
        'top_10_words': word_doc_freq.most_common(10),
        'subreddit_word_freq': {
            sub: dict(subreddit_word_freq_train[sub].most_common(10))
            for sub in list(subreddit_word_freq_train.keys())[:10]
        }
    }
    
    with open(os.path.join(output_dir, 'summary_stats.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nOutputs generated!")
    print(f"  Users: {len(user_num_to_id)}")
    print(f"  Subreddits: {len(subreddit_list)}")
    print(f"  Unique words: {len(word_doc_freq)}")
    print(f"  Train interactions: {len(train_interactions)}")
    print(f"  Dev interactions: {len(dev_interactions)}")
    print(f"  Test interactions: {len(test_interactions)}")

def main():
    parser = argparse.ArgumentParser(
        description='Process Engage Corpus v4 with TF-IDF'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory containing data00.json through data23.json'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='engage_corpus_processed_v4',
        help='Output directory for processed data'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"ERROR: Data directory not found: {args.data_dir}")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("ENGAGE CORPUS PROCESSING V4")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {RANDOM_SEED}")
    print("="*60)
    
    # Pass 1: Collect subreddit statistics
    top_2000_subreddits, subreddit_list = pass1_collect_subreddit_stats(args.data_dir, args.output_dir)
    
    # Pass 2: Select users with strict filtering
    selected_users = pass2_select_users(args.data_dir, top_2000_subreddits, args.output_dir)
    
    # Pass 3: Compute word statistics
    word_stats = pass3_compute_word_stats(args.data_dir, top_2000_subreddits, selected_users, args.output_dir)
    
    # Pass 4: Generate outputs
    pass4_generate_outputs(args.data_dir, top_2000_subreddits, subreddit_list, 
                          selected_users, word_stats, args.output_dir)
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
