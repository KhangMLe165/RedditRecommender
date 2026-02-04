#!/usr/bin/env python3
"""
Extract text context data for MS2/process_engage_corpus_v3.py observations.

Uses the EXACT same filtering methodology and seed as v3, but outputs:
1. Text context data (bag of words per user and per subreddit)
2. Text context filtered (TF-IDF top words per user and per subreddit)
3. Interaction data for NCF training

This allows us to use text embeddings with the v3 interaction data.

The TF-IDF text filtering is necessary to ensure that we are not exceeding the context window
of the embedding model.

As in the v3 baseline, we process incrementally, avoiding loading all of the source data into
memory at once.
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

# Configuration - matches v3 baseline
RANDOM_SEED = 42
NUM_SUBREDDITS = 5000

# Date ranges (Unix timestamps) - matches v3 baseline
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
    """Filter user's posts/comments into periods (matching v3)."""
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
            ncf_train_posts.append(post)
        elif VW_TARGET_START <= timestamp <= VW_TARGET_END:
            vw_target_posts.append(post)
            ncf_train_posts.append(post)
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
    """First pass: Collect subreddit statistics to find top 5000."""
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
    
    subreddit_users = defaultdict(set)
    
    for user in load_all_data_files(data_dir):
        periods = filter_user_by_periods(user)
        user_num = user['user_number']
        
        # Count from all periods
        for period_name in ['context', 'vw_target', 'eval']:
            period = periods[period_name]
            for post in period['posts']:
                subreddit_users[post['subreddit']].add(user_num)
            for comment in period['comments']:
                subreddit_users[comment['subreddit']].add(user_num)
    
    subreddit_counts = [(sub, len(users)) for sub, users in subreddit_users.items()]
    subreddit_counts.sort(key=lambda x: x[1], reverse=True)
    
    top_5000 = [sub for sub, count in subreddit_counts[:5000]]
    top_5000_set = set(top_5000)
    
    print(f"Total subreddits found: {len(subreddit_counts)}")
    print(f"Top 5000 largest subreddits selected")
    
    result = (top_5000_set, top_5000)
    
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)
    print(f"✓ Cached subreddit stats to: {cache_file}")
    
    return result

def pass2_select_users(data_dir, top_5000_subreddits, output_dir):
    """Second pass: Select 6% of users (matching v3 methodology)."""
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
    print("PASS 2: Selecting 6% of users with top 5000 subreddit interactions")
    print("="*60)
    
    eligible_users = []
    
    for user in load_all_data_files(data_dir):
        periods = filter_user_by_periods(user)
        user_num = user['user_number']
        
        # Check if user has any activity in top 5000 subreddits
        has_top_5000_activity = False
        for period_name in ['context', 'vw_target', 'eval']:
            period = periods[period_name]
            for post in period['posts']:
                if post['subreddit'] in top_5000_subreddits:
                    has_top_5000_activity = True
                    break
            if not has_top_5000_activity:
                for comment in period['comments']:
                    if comment['subreddit'] in top_5000_subreddits:
                        has_top_5000_activity = True
                        break
            if has_top_5000_activity:
                break
        
        if has_top_5000_activity:
            eligible_users.append(user_num)
    
    # Sample 6%
    random.seed(RANDOM_SEED)
    num_to_select = int(len(eligible_users) * 0.06)
    selected_users = set(random.sample(eligible_users, num_to_select))
    
    print(f"Eligible users: {len(eligible_users)}")
    print(f"Selected 6%: {len(selected_users)} users")
    
    with open(cache_file, 'wb') as f:
        pickle.dump(selected_users, f)
    print(f"✓ Cached selected users to: {cache_file}")
    
    return selected_users

def pass3_compute_word_stats(data_dir, top_5000_subreddits, selected_users, output_dir):
    """
    Third pass: Compute word statistics for TF-IDF.
    
    Computes:
    - Document frequency (DF) for each word (for IDF calculation)
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
    
    # Track document frequency (for IDF)
    word_doc_freq = Counter()  # word -> number of documents containing it
    total_docs = 0
    
    # Track term frequency per user
    user_word_freq_train = defaultdict(Counter)  # user_id -> {word: count}
    user_word_freq_dev = defaultdict(Counter)
    user_word_freq_test = defaultdict(Counter)
    
    # Track term frequency per subreddit
    subreddit_word_freq_train = defaultdict(Counter)  # subreddit -> {word: count}
    subreddit_word_freq_dev = defaultdict(Counter)
    subreddit_word_freq_test = defaultdict(Counter)
    
    user_num_to_id = {}
    user_id_counter = 0
    skipped_users = 0
    
    for user in tqdm(load_all_data_files(data_dir), desc="Computing word stats"):
        user_num = user['user_number']
        
        if user_num not in selected_users:
            continue
        
        periods = filter_user_by_periods(user)
        
        # Skip users without eval data (same as original v3.5 logic)
        eval_items = periods['eval']['posts'] + periods['eval']['comments']
        if len(eval_items) < 2:
            skipped_users += 1
            continue
        
        # Assign user ID
        user_id = user_id_counter
        user_num_to_id[user_num] = user_id
        user_id_counter += 1
        
        # Process training text (context + vw_target = Jan-Sept)
        train_words = set()
        
        for post in periods['context']['posts'] + periods['vw_target']['posts']:
            title_text = post.get('title', '')
            body_text = post.get('selftext', '')
            tokens = simple_tokenize(title_text) + simple_tokenize(body_text)
            
            # Update user TF
            for token in tokens:
                user_word_freq_train[user_id][token] += 1
            
            # Update subreddit TF
            sub = post.get('subreddit')
            if sub in top_5000_subreddits:
                for token in tokens:
                    subreddit_word_freq_train[sub][token] += 1
            
            # Track unique words for DF
            train_words.update(tokens)
        
        for comment in periods['context']['comments'] + periods['vw_target']['comments']:
            body_text = comment.get('body', '')
            tokens = simple_tokenize(body_text)
            
            # Update user TF
            for token in tokens:
                user_word_freq_train[user_id][token] += 1
            
            # Update subreddit TF
            sub = comment.get('subreddit')
            if sub in top_5000_subreddits:
                for token in tokens:
                    subreddit_word_freq_train[sub][token] += 1
            
            # Track unique words for DF
            train_words.update(tokens)
        
        # Update document frequency (each user is one document)
        for word in train_words:
            word_doc_freq[word] += 1
        total_docs += 1
        
        # Process eval items for dev/test
        eval_items_sorted = sorted(eval_items, key=lambda x: x['created_utc'])
        first_eval = eval_items_sorted[0]
        second_eval = eval_items_sorted[1]
        
        # Dev text (first eval item)
        if 'title' in first_eval:
            dev_tokens = simple_tokenize(first_eval.get('title', '')) + simple_tokenize(first_eval.get('selftext', ''))
        else:
            dev_tokens = simple_tokenize(first_eval.get('body', ''))
        
        for token in dev_tokens:
            user_word_freq_dev[user_id][token] += 1
        
        first_sub = first_eval.get('subreddit')
        if first_sub in top_5000_subreddits:
            for token in dev_tokens:
                subreddit_word_freq_dev[first_sub][token] += 1
        
        # Test text (second eval item)
        if 'title' in second_eval:
            test_tokens = simple_tokenize(second_eval.get('title', '')) + simple_tokenize(second_eval.get('selftext', ''))
        else:
            test_tokens = simple_tokenize(second_eval.get('body', ''))
        
        for token in test_tokens:
            user_word_freq_test[user_id][token] += 1
        
        second_sub = second_eval.get('subreddit')
        if second_sub in top_5000_subreddits:
            for token in test_tokens:
                subreddit_word_freq_test[second_sub][token] += 1
    
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
        'num_users': user_id_counter,
        'skipped_users': skipped_users
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)
    print(f"✓ Cached word statistics to: {cache_file}")
    
    print(f"Total users: {user_id_counter}")
    print(f"Skipped users (insufficient eval data): {skipped_users}")
    print(f"Unique words: {len(word_doc_freq)}")
    print(f"Most common words: {word_doc_freq.most_common(10)}")
    
    return result

def compute_tfidf_and_filter(word_freq, word_doc_freq, total_docs, top_k=50):
    """
    Compute TF-IDF scores and return top K words.
    
    TF-IDF = TF * IDF
    where IDF = log(N / DF)
    
    Args:
        word_freq: Counter of word -> count for this document
        word_doc_freq: Counter of word -> number of documents containing it
        total_docs: Total number of documents in corpus
        top_k: Number of top words to return
    
    Returns:
        Tuple of (list of top_k words, dict of word -> tfidf_score)
    """
    tfidf_scores = {}
    
    for word, tf in word_freq.items():
        df = word_doc_freq.get(word, 1)  # Default to 1 to avoid log(0)
        idf = log(total_docs / df)
        tfidf_scores[word] = tf * idf
    
    # Get top K words by TF-IDF score
    top_words = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    return [word for word, score in top_words], tfidf_scores

def pass4_extract_interactions_and_generate_outputs(data_dir, top_5000_subreddits, subreddit_list, 
                                                     selected_users, word_stats, output_dir):
    """
    Fourth pass: Extract interactions and generate all outputs.
    
    Outputs:
    1. Text context (unfiltered) - bag of words per user/subreddit
    2. Text context filtered (TF-IDF top 50 words) - most important words per user/subreddit
    3. Interaction data for NCF - train/dev/test
    4. Mappings and statistics
    """
    print("\n" + "="*60)
    print("PASS 4: Extracting interactions and generating outputs")
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
    
    # =====================================================================
    # Generate unfiltered text (bag of words)
    # =====================================================================
    user_text_train = {}
    user_text_dev = {}
    user_text_test = {}
    subreddit_text_train = {}
    subreddit_text_dev = {}
    subreddit_text_test = {}
    
    # =====================================================================
    # Generate filtered text (TF-IDF top 50 words)
    # =====================================================================
    user_text_train_filtered = {}
    user_text_dev_filtered = {}
    user_text_test_filtered = {}
    subreddit_text_train_filtered = {}
    subreddit_text_dev_filtered = {}
    subreddit_text_test_filtered = {}
    
    print("Generating user text (unfiltered and TF-IDF filtered)...")
    
    # User train text
    for user_id, word_freq in tqdm(user_word_freq_train.items(), desc="User train text"):
        # Unfiltered: all words with counts
        user_text_train[user_id] = ' '.join([word for word in word_freq.keys() for _ in range(word_freq[word])])
        
        # Filtered: top 50 TF-IDF words
        top_words, _ = compute_tfidf_and_filter(word_freq, word_doc_freq, total_docs, top_k=50)
        user_text_train_filtered[user_id] = ' '.join(top_words)
    
    # User dev text
    for user_id, word_freq in tqdm(user_word_freq_dev.items(), desc="User dev text"):
        user_text_dev[user_id] = ' '.join([word for word in word_freq.keys() for _ in range(word_freq[word])])
        top_words, _ = compute_tfidf_and_filter(word_freq, word_doc_freq, total_docs, top_k=50)
        user_text_dev_filtered[user_id] = ' '.join(top_words)
    
    # User test text
    for user_id, word_freq in tqdm(user_word_freq_test.items(), desc="User test text"):
        user_text_test[user_id] = ' '.join([word for word in word_freq.keys() for _ in range(word_freq[word])])
        top_words, _ = compute_tfidf_and_filter(word_freq, word_doc_freq, total_docs, top_k=50)
        user_text_test_filtered[user_id] = ' '.join(top_words)
    
    print("Generating subreddit text (unfiltered and TF-IDF filtered)...")
    
    # Subreddit train text
    for sub, word_freq in tqdm(subreddit_word_freq_train.items(), desc="Subreddit train text"):
        subreddit_text_train[sub] = ' '.join([word for word in word_freq.keys() for _ in range(word_freq[word])])
        top_words, _ = compute_tfidf_and_filter(word_freq, word_doc_freq, total_docs, top_k=50)
        subreddit_text_train_filtered[sub] = ' '.join(top_words)
    
    # Subreddit dev text
    for sub, word_freq in tqdm(subreddit_word_freq_dev.items(), desc="Subreddit dev text"):
        subreddit_text_dev[sub] = ' '.join([word for word in word_freq.keys() for _ in range(word_freq[word])])
        top_words, _ = compute_tfidf_and_filter(word_freq, word_doc_freq, total_docs, top_k=50)
        subreddit_text_dev_filtered[sub] = ' '.join(top_words)
    
    # Subreddit test text
    for sub, word_freq in tqdm(subreddit_word_freq_test.items(), desc="Subreddit test text"):
        subreddit_text_test[sub] = ' '.join([word for word in word_freq.keys() for _ in range(word_freq[word])])
        top_words, _ = compute_tfidf_and_filter(word_freq, word_doc_freq, total_docs, top_k=50)
        subreddit_text_test_filtered[sub] = ' '.join(top_words)
    
    # =====================================================================
    # Save text context files
    # =====================================================================
    
    # Unfiltered text
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
    
    # Filtered text (TF-IDF top 50)
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
    
    # =====================================================================
    # Generate NCF interaction data
    # =====================================================================
    print("Generating NCF interaction data...")
    
    train_file = open(os.path.join(ncf_dir, 'train.tsv'), 'w')
    dev_file = open(os.path.join(ncf_dir, 'dev.tsv'), 'w')
    test_file = open(os.path.join(ncf_dir, 'test.tsv'), 'w')
    
    train_interactions = 0
    dev_count = 0
    test_count = 0
    
    for user in tqdm(load_all_data_files(data_dir), desc="Processing interactions"):
        user_num = user['user_number']
        
        if user_num not in user_num_to_id:
            continue
        
        user_id = user_num_to_id[user_num]
        periods = filter_user_by_periods(user)
        
        # Extract NCF training interactions (all Jan-Sept interactions)
        user_train_subreddits = set()
        for item in periods['ncf_train']['posts'] + periods['ncf_train']['comments']:
            subreddit = item.get('subreddit')
            if subreddit in subreddit2idx:
                sub_idx = subreddit2idx[subreddit]
                if sub_idx not in user_train_subreddits:
                    train_file.write(f"{user_id}\t{sub_idx}\n")
                    user_train_subreddits.add(sub_idx)
                    train_interactions += 1
        
        # Extract eval interactions (using same logic as original v3.5)
        eval_items = periods['eval']['posts'] + periods['eval']['comments']
        eval_items_sorted = sorted(eval_items, key=lambda x: x['created_utc'])
        first_eval = eval_items_sorted[0]
        second_eval = eval_items_sorted[1]
        
        # Dev (first eval item)
        first_sub = first_eval.get('subreddit')
        if first_sub in subreddit2idx:
            dev_file.write(f"{user_id}\t{subreddit2idx[first_sub]}\n")
            dev_count += 1
        
        # Test (second eval item)
        second_sub = second_eval.get('subreddit')
        if second_sub in subreddit2idx:
            test_file.write(f"{user_id}\t{subreddit2idx[second_sub]}\n")
            test_count += 1
    
    train_file.close()
    dev_file.close()
    test_file.close()
    
    # =====================================================================
    # Save mappings
    # =====================================================================
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
    
    # =====================================================================
    # Save summary statistics
    # =====================================================================
    summary = {
        'num_users': len(user_num_to_id),
        'num_subreddits': len(subreddit_list),
        'num_unique_words': len(word_doc_freq),
        'train_interactions': train_interactions,
        'dev_samples': dev_count,
        'test_samples': test_count,
        'skipped_users': word_stats['skipped_users'],
        'top_10_words_by_doc_freq': word_doc_freq.most_common(10)
    }
    
    with open(os.path.join(output_dir, 'summary_stats.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nData extraction complete!")
    print(f"  Unique users: {len(user_num_to_id)}")
    print(f"  NCF training interactions: {train_interactions:,}")
    print(f"  Dev samples: {dev_count}")
    print(f"  Test samples: {test_count}")
    print(f"  Skipped users (insufficient eval data): {word_stats['skipped_users']}")
    print(f"  Unique words: {len(word_doc_freq)}")
    
    print(f"\nFiles saved:")
    print(f"  {text_dir}/user_text_*.json (unfiltered bag of words)")
    print(f"  {text_dir}/subreddit_text_*.json (unfiltered bag of words)")
    print(f"  {text_filtered_dir}/user_text_*.json (TF-IDF top 50 words)")
    print(f"  {text_filtered_dir}/subreddit_text_*.json (TF-IDF top 50 words)")
    print(f"  {ncf_dir}/train.tsv")
    print(f"  {ncf_dir}/dev.tsv")
    print(f"  {ncf_dir}/test.tsv")

def main():
    parser = argparse.ArgumentParser(
        description='Extract text context for MS2 data (v3.5 with TF-IDF filtering)'
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
        default='engage_corpus_processed_v3_5',
        help='Output directory for processed data'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"ERROR: Data directory not found: {args.data_dir}")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("ENGAGE CORPUS TEXT EXTRACTION (v3.5 with TF-IDF)")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {RANDOM_SEED}")
    print("="*60)
    
    # Pass 1: Collect subreddit statistics
    top_5000_subreddits, subreddit_list = pass1_collect_subreddit_stats(args.data_dir, args.output_dir)
    
    # Pass 2: Select 6% of users
    selected_users = pass2_select_users(args.data_dir, top_5000_subreddits, args.output_dir)
    
    # Pass 3: Compute word statistics for TF-IDF
    word_stats = pass3_compute_word_stats(args.data_dir, top_5000_subreddits, selected_users, args.output_dir)
    
    # Pass 4: Extract interactions and generate outputs
    pass4_extract_interactions_and_generate_outputs(args.data_dir, top_5000_subreddits, 
                                                     subreddit_list, selected_users, 
                                                     word_stats, args.output_dir)
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
