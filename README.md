# Subreddit Recommendation with Neural Collaborative Filtering and Semantic Text Embeddings

This repository contains the full implementation, data processing pipeline, and experimental results for **Subreddit Recommendation using Neural Collaborative Filtering (NCF) augmented with semantic text embeddings**. The project explores whether pretrained large language model embeddings can improve recommendation quality beyond interaction-only collaborative filtering, particularly in sparse and cold-start settings.

This work was completed as part of **CIS 5300: Natural Language Processing** and evaluates multiple architectural variants on the **Engage Corpus** Reddit dataset.

---

## Project Overview

Traditional recommender systems rely heavily on user–item interaction histories, which perform poorly for users with limited activity. To address this, we extend **Neural Collaborative Filtering (NCF)** by incorporating **semantic representations of user and subreddit text** derived from a large pretrained transformer embedding model.

We investigate:
- Whether semantic text embeddings improve recommendation performance
- How their impact differs across single-label and multi-label evaluation settings
- The interaction between collaborative signals and text-derived features

---

## Key Contributions

- Implemented an end-to-end Neural Collaborative Filtering (NCF) pipeline with text augmentation
- Integrated Qwen3-Embedding-8B embeddings for users and subreddits
- Constructed two dataset variants:
  - v3.5: Single-label benchmark setting
  - v4: Multi-label recommendation setting
- Performed extensive ablation studies:
  - TF-IDF filtering
  - Chi-square feature selection
  - Gated fusion between collaborative and text signals
- Conducted post-hoc analyses explaining why text embeddings help in isolation but not when fused with NCF

---

## Repository Structure

```
.
├── code/
│   ├── data_processing/     # Dataset preprocessing and filtering
│   ├── embeddings/          # Text embedding generation
│   ├── models/              # NCF and ablation model implementations
│   ├── training/            # Training and evaluation scripts
│   └── utils/               # Metrics and helpers
│
├── data/
│   ├── v3_5/                # Single-label dataset
│   └── v4/                  # Multi-label dataset
│
├── output/
│   ├── logs/                # Training logs
│   ├── results/             # Evaluation outputs
│   └── figures/             # Generated plots
│
└── CIS_5300_Project_Report.pdf
```

---

## Dataset

We use the Engage Corpus, a large-scale Reddit dataset containing user–subreddit interactions and associated text.

### Dataset Variants

- **v3.5 (Single-Label)**  
  Mirrors the original Engage Corpus benchmark with one held-out subreddit per user.

- **v4 (Multi-Label)**  
  A stricter and more realistic formulation where users may have multiple relevant subreddits in the evaluation period.

Preprocessing includes tokenization, lowercasing, TF-IDF filtering (top 50 tokens per entity), and temporal splits to prevent information leakage.

---

## Models Implemented

### Baselines
- Cosine similarity over semantic embeddings (128-dim and 4096-dim)
- Neural Collaborative Filtering (NCF) without text

### Neural Models
- NCF + TF-IDF text features
- NCF + semantic text embeddings
- NCF + gated fusion of collaborative and text signals
- NCF + chi-square feature selection
- Combined chi-square + gated fusion variants

---

## Evaluation Metrics

Models are evaluated using standard ranking metrics:

- **Hit Rate@10 (HR@10)**  
  Measures whether at least one relevant subreddit appears in the top 10 recommendations.

- **Normalized Discounted Cumulative Gain@10 (NDCG@10)**  
  Rewards higher-ranked correct recommendations.

Metrics are adapted for multi-label evaluation in the v4 dataset.

---

## Results Summary

Key findings:
- Semantic text embeddings are highly predictive in isolation, especially for niche subreddits.
- In the single-label setting (v3.5), cosine similarity over high-dimensional embeddings outperforms NCF + Text.
- In the multi-label setting (v4), the interaction-only NCF baseline achieves the best overall performance.
- Gated fusion and chi-square filtering generally reduce performance by over-attenuating text signals.
- Post-hoc analysis shows that NCF systematically down-weights text embeddings during training.

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Preprocess Data
```bash
python code/data_processing/preprocess.py
```

### 3. Generate Embeddings
```bash
python code/embeddings/generate_embeddings.py
```

### 4. Train Models
```bash
python code/training/train_ncf.py
```

### 5. Evaluate
```bash
python code/training/evaluate.py
```

---

## Authors

- Ethan Kallett  
- Jack Bader  
- Oscar Wan  
- Khang Le  

---

## References

- Cheng et al., The Engage Corpus, LREC 2022  
- He et al., Neural Collaborative Filtering, WWW 2017  
- Zheng et al., DeepCoNN, WSDM 2017  
- Sun et al., BERT4Rec, CIKM 2019  
