# Subreddit Recommendation System

This README explains in detail all of the code used in this project. Below you will find for each script/notebook a description of 1) how to run the code 2) expected inputs and CLI arguments and 3) expected outputs. We also include the study findings for each model evaluated.

The MS2/ folder contains our attempt to replicate the results in the [ENGAGE Corpus](https://github.com/engage-corpus/dataset) (we failed) and the remaining code was used in the extension, an ablation (Chi-squared) and our analysis of it. 

## Directory Structure

```
code/
├── MS2/                                    # Baseline Replication (Milestone 2)
│   ├── process_engage_corpus_v3.py         # Data processing for paper replication
│   ├── simple-baseline.py                  # VW trigram logistic baseline
│   ├── strong_baseline.ipynb               # NCF collaborative filtering baseline
│   └── score.py                            # Evaluation script
│
├── Data Processing/                        # Data preprocessing pipeline
│   ├── process_engage_corpus_v3_5_tfidf.py # V3.5: Paper methodology + TF-IDF
│   ├── process_engage_corpus_v3_5_chi2.py  # V3.5: Chi-squared feature selection
│   ├── process_engage_corpus_v4.py         # V4: New multi-label methodology
│   ├── process_engage_corpus_v4_chi2.py    # V4: Chi-squared feature selection
│   └── process_engage_corpus_chi2.ipynb    # Notebook version of chi2 processing
│
├── Embedding Generation/                   # Text embedding generation
│   ├── embed_text_vllm_incremental.ipynb   # Qwen3-8B embeddings (TF-IDF data)
│   └── embed_text_vllm_incremental_chi2.ipynb  # Qwen3-8B embeddings (Chi2 data)
│
├── Evaluate/                               # Model evaluation
│   ├── score_v35_single_label_v2.ipynb     # V3.5 single-label evaluation
│   ├── score_v4_multi_label_v2.ipynb       # V4 multi-label evaluation
│   └── score_v4_chisq_multi_label.ipynb    # V4 chi-squared ablation evaluation
│
├── train_ncf_models_v3.ipynb               # NCF model training (Models A-G)
├── generate_predictions_v3.ipynb           # Prediction generation for all models
└── v4_error_analysis.ipynb                 # Post-hoc analysis of predictions
```

---


## MS2/ - Baseline Replication

### `process_engage_corpus_v3.py`
Processes the ENGAGE Corpus following the original paper's methodology for baseline replication.

**Usage:**
```bash
python MS2/process_engage_corpus_v3.py \
    --data_dir /path/to/data \
    --output_dir engage_corpus_processed_v3
```

**CLI Arguments:**
| Argument | Required | Description |
|----------|----------|-------------|
| `--data_dir` | Yes | Directory containing `data00.json` through `data23.json` |
| `--output_dir` | No | Output directory (default: `engage_corpus_processed_v3`) |

**Output:**
```
engage_corpus_processed_v3/
├── vw_data/
│   ├── train.vw           # VW training file
│   └── context_map.pkl    # User context vectors
├── ncf_data/
│   ├── train.tsv          # NCF training interactions
│   ├── dev.tsv            # Development set
│   └── test.tsv           # Test set
├── user_mapping.json      # User ID mappings
└── subreddit_mapping.json # Subreddit ID mappings
```

#### `simple-baseline.py`
Trains a Vowpal Wabbit One-Against-All logistic regression model using trigram features.

**Usage:**
```bash
python MS2/simple-baseline.py \
    --train_file vw_data/train.vw \
    --predictions_file predictions.pkl \
    --ground_truth_file ground_truth.pkl \
    --num_classes 5000 \
    --model_output vw_model.bin
```

**Expected Output:**
```
EVALUATION RESULTS
============================================================
TEST SET:
  Hit Rate @ 10:  0.0175  (1.75%)
  NDCG @ 10:      0.0072
```

### `strong_baseline.ipynb`
Jupyter notebook implementing Neural Collaborative Filtering (NCF) following He et al. (2017).

**How to Run:** Execute cells sequentially in Google Colab with GPU runtime.

**Expected Output:**
| Model | HR@10 | NDCG@10 |
|-------|-------|---------|
| NCF (no context) | 26.12 | 16.46 |
| NCF + text-based context | 26.33 | 16.29 |

### `score.py`
Command-line evaluation script for computing HR@K and NDCG@K metrics.

**Usage:**
```bash
# Using numpy arrays
python MS2/score.py \
    --predictions_npy predictions.npy \
    --ground_truth_npy ground_truth.npy \
    --k 10

# Using pickle files
python MS2/score.py \
    --predictions predictions.pkl \
    --ground_truth ground_truth.pkl \
    --split test
```

---

## Data Processing/

These scripts process the ENGAGE Corpus into formats suitable for model training. There are two main methodologies:

| Version | Subreddits | Users | Evaluation | Feature Selection |
|---------|------------|-------|------------|-------------------|
| **V3.5** | 5,000 | ~33,727 | Single-label | TF-IDF or Chi² |
| **V4** | 2,000 | ~179,936 | Multi-label | TF-IDF or Chi² |

### `process_engage_corpus_v3_5_tfidf.py` / `process_engage_corpus_v3_5_chi2.py`
Processes data using the original paper's methodology (single-label evaluation) with TF-IDF or Chi-squared feature selection.

**Usage:**
```bash
python "Data Processing/process_engage_corpus_v3_5_tfidf.py" \
    --data_dir /path/to/data \
    --output_dir engage_corpus_processed_v3_5tfidf
```

**CLI Arguments:**
| Argument | Required | Description |
|----------|----------|-------------|
| `--data_dir` | Yes | Directory containing raw JSON files |
| `--output_dir` | No | Output directory |

**Output:**
```
engage_corpus_processed_v3_5tfidf/
├── text_context/              # Raw bag-of-words text
├── text_context_filtered/     # TF-IDF filtered text (top 50 words)
│   ├── user_text_train.json
│   ├── user_text_dev.json
│   ├── user_text_test.json
│   ├── subreddit_text_train.json
│   ├── subreddit_text_dev.json
│   └── subreddit_text_test.json
├── ncf_data/
│   ├── train.tsv
│   ├── dev.tsv
│   └── test.tsv
├── user_mapping.json
├── subreddit_mapping.json
└── summary_stats.json
```

### `process_engage_corpus_v4.py` / `process_engage_corpus_v4_chi2.py`
Our extended methodology with stricter filtering and multi-label evaluation.

**Key Differences from V3.5:**
- Top 2,000 subreddits (instead of 5,000)
- Users must have 50-5,000 posts/comments
- Multi-label evaluation (users can have multiple correct subreddits)
- New temporal splits: Train (Jan-June → July-Sept), Dev (Oct & Dec), Test (Nov)

**Usage:**
```bash
python "Data Processing/process_engage_corpus_v4.py" \
    --data_dir /path/to/data \
    --output_dir engage_corpus_processed_v4
```

---

## Embedding Generation/

### `embed_text_vllm_incremental.ipynb` / `embed_text_vllm_incremental_chi2.ipynb`
Generate text embeddings using the Qwen3-Embedding-8B model via vLLM.

**How to Run:** Execute cells sequentially in Google Colab with A100 GPU runtime (16GB+ VRAM required).

**Configuration (in notebook):**
```python
INPUT_DIRS = {
    "v4": "/path/to/engage_corpus_processed_v4/text_context_filtered",
    "v3_5tfidf": "/path/to/engage_corpus_processed_v3_5tfidf/text_context_filtered",
}
OUTPUT_BASE_DIR = "/path/to/embeddings_qwen3_8b"
BATCH_SIZE = 2048  # Reduce if OOM errors
```

**Output:**
```
embeddings_qwen3_8b/
├── v4/
│   └── filtered/
│       ├── train/
│       │   ├── user_embeddings_128.npy    # (N_users, 128) MRL embeddings
│       │   ├── user_embeddings_4096.npy   # (N_users, 4096) full embeddings
│       │   ├── subreddit_embeddings_128.npy
│       │   ├── subreddit_embeddings_4096.npy
│       │   ├── user_ids.json              # Index mapping file
│       │   └── subreddit_names.json       # Index mapping file
│       ├── dev/
│       └── test/
└── v3_5tfidf/
    └── filtered/
        └── ...
```

**Differences between notebooks:**
- `embed_text_vllm_incremental.ipynb`: Processes TF-IDF filtered data
- `embed_text_vllm_incremental_chi2.ipynb`: Processes Chi-squared filtered data

---

## train_ncf_models_v3.ipynb

Trains all NCF model variants for the study.

**Models Trained:**

| Model | Data | Text | Architecture |
|-------|------|------|--------------|
| Model A | V3.5 TF-IDF | 128-dim | NCF + Text (Concat) |
| Model B | V4 | None | NCF Baseline |
| Model C | V4 | 128-dim | NCF + Text (Concat) |
| Model D | V3.5 TF-IDF | 128-dim | NCF + Text (Gated) |
| Model E | V4 | 128-dim | NCF + Text (Gated) |
| Model F | V4 Chi² | 128-dim | NCF + Text (Concat) |
| Model G | V4 Chi² | 128-dim | NCF + Text (Gated) |

**How to Run:** Execute cells sequentially in Google Colab with T4/A100 GPU.

**Hyperparameters:**
```python
EMBEDDING_DIM = 64       # NCF embedding dimension
TEXT_EMB_DIM = 128       # Qwen3 MRL embedding dimension
TEXT_PROJ_DIM = 128      # Projected text embedding dimension
BATCH_SIZE = 2096
LEARNING_RATE = 0.0001
NUM_EPOCHS = 90
NUM_NEGATIVES = 4        # Negative samples per positive
```

**Output:**
```
trained_models/
├── model_A_v35_text_best.pt
├── model_B_v4_baseline_best.pt
├── model_C_v4_text_best.pt
├── model_D_tfidf_gated_best.pt
├── model_E_v4_tfidf_gated_best.pt
├── model_F_chi2_concat_best.pt
├── model_G_chi2_gated_best.pt
├── training_curves.png
└── checkpoints/
```

---

## generate_predictions_v3.ipynb

Generates prediction matrices for all trained models plus cosine similarity baselines.

**How to Run:** Execute cells sequentially in Google Colab.

**Configuration:**
```python
CHECKPOINT_SELECTION = {
    'model_A': 'best',  # Options: 'best', 'final', 'epoch_N'
    'model_B': 'best',
    'model_C': 'best',
    # ...
}
```

**Output:**
```
predictions/
├── cosine_v35_128.npy           # V3.5 cosine similarity (128-dim)
├── cosine_v35_4096.npy          # V3.5 cosine similarity (4096-dim)
├── cosine_v4_128.npy            # V4 cosine similarity (128-dim)
├── cosine_v4_4096.npy           # V4 cosine similarity (4096-dim)
├── cosine_v4_chi2_128.npy       # V4 Chi² cosine similarity
├── predictions_model_A.npy      # NCF + text, V3.5
├── predictions_model_B.npy      # NCF baseline, V4
├── predictions_model_C.npy      # NCF + text, V4
├── predictions_model_D.npy      # NCF + text (gated), V3.5
├── predictions_model_E.npy      # NCF + text (gated), V4
├── predictions_model_F.npy      # NCF + text, V4 Chi²
├── predictions_model_G.npy      # NCF + text (gated), V4 Chi²
├── ground_truth_v35_test.npy    # Single-label ground truth
├── ground_truth_v4_test.npy     # Multi-label ground truth
├── ground_truth_v4_chi2_test.npy
└── prediction_summary.json
```

---

## Evaluate/

### `score_v35_single_label_v2.ipynb`
Evaluates V3.5 predictions using single-label metrics (one correct answer per user).

**How to Run:** Execute cells sequentially in Google Colab.

**Configuration:**
```python
BASE_DIR = Path('/content/drive/MyDrive/CIS 5300/predictions')
PREDICTION_FILES = {
    'Cosine Similarity (v3.5, 128-dim)': BASE_DIR / 'cosine_v35_128.npy',
    'Cosine Similarity (v3.5, 4096-dim)': BASE_DIR / 'cosine_v35_4096.npy',
    'NCF Model A (v3.5)': BASE_DIR / 'predictions_model_A.npy',
}
```

**Expected Output:**
| Model | HR@10 | NDCG@10 |
|-------|-------|---------|
| Cosine Similarity (128-dim) | 27.33 | 23.36 |
| Cosine Similarity (4096-dim) | 32.66 | 28.23 |
| NCF + Text (Model A) | 15.57 | 9.93 |

### `score_v4_multi_label_v2.ipynb`
Evaluates V4 predictions using multi-label metrics (multiple correct answers per user).

**Expected Output:**
| Model | HR@10 | NDCG@10 |
|-------|-------|---------|
| Cosine Similarity (128-dim) | 28.65 | 10.52 |
| Cosine Similarity (4096-dim) | 35.47 | 13.98 |
| NCF Baseline (Model B) | 53.21 | 22.38 |
| NCF + Text (Model C) | 52.46 | 22.21 |
| Model E (V4 + Gated) | 48.11 | 18.71 |

### `score_v4_chisq_multi_label.ipynb`
Evaluates Chi-squared ablation models (Models F and G).

**Expected Output:**
| Model | HR@10 | NDCG@10 |
|-------|-------|---------|
| Model F (Chi²) | 44.22 | 16.67 |
| Model G (Chi² + Gated) | 47.88 | 18.57 |

---

### v4_error_analysis.ipynb

Post-hoc analysis notebook investigating why text embeddings don't improve NCF performance.

**Analyses Performed:**
- Model overlap analysis (Venn diagrams)
- Performance by user/subreddit popularity buckets
- Embedding similarity diagnostics
- CKA analysis between text and NCF embeddings
- Fusion architecture diagnostics

**How to Run:** Execute cells sequentially in Google Colab with GPU.

---

## Results Summary

### Baseline (Paper Replication)
| Model | HR@10 | NDCG@10 |
|-------|-------|---------|
| Trigram logistic baseline | 1.75 | 0.72 |
| NCF (no context) | 26.12 | 16.46 |
| NCF + text-based context | 26.33 | 16.29 |

### V3.5 (Single-Label)
| Model | HR@10 | NDCG@10 |
|-------|-------|---------|
| Cosine Similarity (128-dim) | 27.33 | 23.36 |
| Cosine Similarity (4096-dim) | 32.66 | 28.23 |
| NCF + Text (Model A) | 15.57 | 9.93 |

### V4 (Multi-Label)
| Model | HR@10 | NDCG@10 |
|-------|-------|---------|
| Cosine Similarity (128-dim) | 28.65 | 10.52 |
| Cosine Similarity (4096-dim) | 35.47 | 13.98 |
| NCF Baseline (Model B) | 53.21 | 22.38 |
| NCF + Text (Model C) | 52.46 | 22.21 |

### Ablation (Gating & Features)
| Model | HR@10 | NDCG@10 |
|-------|-------|---------|
| Model E (V4 + Gated) | 48.11 | 18.71 |
| Model F (Chi²) | 44.22 | 16.67 |
| Model G (Chi² + Gated) | 47.88 | 18.57 |

---

## Important Notes

1. **Index Mapping**: The pipeline uses complex index mappings between NCF indices and embedding array indices. See `generate_predictions_v3.ipynb` for reindexing logic.

2. **Hardware Requirements**:
    - All jupyter notebooks are recommended to be run with a GPU backend (we used A100 in Google Colab). Python scripts (.py extension) are meant for a CPU backend and were run locally on a Macbook Air.
   - The Embedding Generation and NCF model training are the most computationally expensive parts of this pipeline.

3. **Reproducibility**: All scripts use `RANDOM_SEED = 42` for reproducibility.
