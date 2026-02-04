# RedditRecommender

Subreddit Recommendation with Neural Collaborative Filtering and Semantic Text Embeddings
This repository contains the full implementation, data processing pipeline, and experimental results for Subreddit Recommendation using Neural Collaborative Filtering (NCF) augmented with semantic text embeddings. The project explores whether pretrained large language model embeddings can improve recommendation quality beyond interaction-only collaborative filtering, particularly in sparse and cold-start settings.
This work was completed as part of CIS 5300: Natural Language Processing and evaluates multiple architectural variants on the Engage Corpus Reddit dataset.
Project Overview
Traditional recommender systems rely heavily on user–item interaction histories, which perform poorly for users with limited activity. To address this, we extend Neural Collaborative Filtering (NCF) by incorporating semantic representations of user and subreddit text derived from a large pretrained transformer embedding model.
We investigate:
Whether semantic text embeddings improve recommendation performance
How their impact differs across single-label and multi-label evaluation settings
The interaction between collaborative signals and text-derived features
Key Contributions
Implemented an end-to-end Neural Collaborative Filtering (NCF) pipeline with text augmentation
Integrated Qwen3-Embedding-8B embeddings for users and subreddits
Constructed two dataset variants:
v3.5: Single-label benchmark setting
v4: More realistic multi-label recommendation setting
Performed extensive ablation studies:
TF-IDF filtering
Chi-square feature selection
Gated fusion between collaborative and text signals
Conducted post-hoc analyses to understand why text embeddings help in isolation but not when fused with NCF
Repository Structure
.
├── code/
│   ├── data_processing/     # Dataset preprocessing and filtering
│   ├── embeddings/          # Text embedding generation (Qwen)
│   ├── models/              # NCF and ablation model implementations
│   ├── training/            # Training and evaluation scripts
│   └── utils/               # Metrics, helpers, and utilities
│
├── data/
│   ├── v3_5/                # Single-label dataset split
│   └── v4/                  # Multi-label dataset split
│
├── output/
│   ├── logs/                # Training logs
│   ├── results/             # Metrics and evaluation outputs
│   └── figures/             # Generated plots and visualizations
│
└── CIS_5300_Project_Report.pdf
Dataset
We use the Engage Corpus, a large-scale Reddit dataset containing user–subreddit interactions and associated text.
Dataset Variants
Dataset	Description
v3.5	Single-label evaluation mirroring the original Engage Corpus benchmark
v4	Multi-label evaluation with stricter activity constraints and realistic targets
Key preprocessing steps:
Tokenization, lowercasing, and cleaning
TF-IDF filtering (top 50 tokens per user/subreddit)
Temporal splits to avoid information leakage
Activity-based filtering to ensure meaningful text histories
Models Implemented
Baselines
Cosine Similarity over semantic embeddings (128-dim, 4096-dim)
NCF (No Text) collaborative filtering baseline
Neural Models
NCF + TF-IDF Text
NCF + Semantic Embeddings
NCF + Gated Fusion
NCF + Chi-Square Feature Selection
Combined Chi-Square + Gating
Evaluation Metrics
We evaluate ranking quality using:
HR@10 (Hit Rate@10)
Measures whether at least one relevant subreddit appears in the top 10 recommendations.
NDCG@10 (Normalized Discounted Cumulative Gain)
Rewards higher-ranked correct recommendations.
Both metrics are adapted for multi-label evaluation in the v4 dataset.
Results Summary
Key findings:
Semantic embeddings are highly predictive in isolation, especially for niche and cold-start subreddits.
In the single-label (v3.5) setting, cosine similarity over high-dimensional embeddings outperforms NCF + Text.
In the multi-label (v4) setting, the NCF baseline outperforms all text-augmented variants.
Gated fusion and chi-square filtering generally reduce performance due to over-attenuation of text signals.
Post-hoc analysis shows that NCF systematically down-weights text embeddings during training.
Conclusion:
Text embeddings encode complementary signal but are dominated by strong collaborative signals when fused directly into NCF.
