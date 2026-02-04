# Output Directory

This directory contains model predictions and ground truth labels for evaluation. All prediction and ground truth files are stored as NumPy arrays (`.npy`) or pickle files (`.pkl`).

## Directory Structure

```
output/
├── Milestone 2/                        # Baseline replication
│   ├── vw_predictions.pkl              # VW trigram baseline predictions
│   ├── ground_truth.pkl                # Ground truth for VW baseline
│   ├── ncf_baseline_test_predictions.npy
│   ├── ncf_baseline_test_ground_truth.npy
│   ├── ncf_baseline_dev_predictions.npy
│   └── ncf_baseline_dev_ground_truth.npy
│
└── Milestone 3/                        # Extension models
    ├── predictions_model_A.npy         # V3.5 NCF + Text
    ├── predictions_model_B.npy         # V4 NCF Baseline
    ├── predictions_model_C.npy         # V4 NCF + Text
    ├── predictions_model_D.npy         # V3.5 NCF + Text (Gated)
    ├── predictions_model_E.npy         # V4 NCF + Text (Gated)
    ├── predictions_model_F.npy         # V4 Chi² NCF + Text
    ├── predictions_model_G.npy         # V4 Chi² NCF + Text (Gated)
    ├── cosine_v35_128.npy              # V3.5 Cosine similarity (128-dim)
    ├── cosine_v35_4096.npy             # V3.5 Cosine similarity (4096-dim)
    ├── cosine_v4_128.npy               # V4 Cosine similarity (128-dim)
    ├── cosine_v4_4096.npy              # V4 Cosine similarity (4096-dim)
    ├── ground_truth_v35_test.npy       # V3.5 single-label ground truth
    ├── ground_truth_v4_test.npy        # V4 multi-label ground truth
    ├── ground_truth_v4_chi2_test.npy   # V4 Chi² ground truth
    └── prediction_summary.json
```

---

## File Format Details

### Prediction Files

All prediction files are 2D NumPy arrays of shape `(n_users, n_items)` containing model scores:

| File Pattern | Shape | Description |
|--------------|-------|-------------|
| `predictions_model_*.npy` | varies | NCF model output scores (higher = more likely) |
| `cosine_v35_*.npy` | `(33727, 5000)` | Cosine similarity scores for V3.5 |
| `cosine_v4_*.npy` | `(179936, 2000)` | Cosine similarity scores for V4 |


### Ground Truth Files

Ground truth format differs between single-label (V3.5) and multi-label (V4):

| File | Shape | Format | Description |
|------|-------|--------|-------------|
| `ground_truth_v35_test.npy` | `(27896,)` | 1D array of indices | Each entry is the correct subreddit index for that user |
| `ground_truth_v4_test.npy` | `(179936, 2000)` | Binary matrix | `1` if user interacted with subreddit, `0` otherwise |
| `ground_truth_v4_chi2_test.npy` | `(179936, 2000)` | Binary matrix | Same as above, for Chi² data split |

---

## Running Evaluation

### Milestone 2: `code/MS2/score.py`

Command-line evaluation script for computing HR@K and NDCG@K metrics on single-label predictions.

**CLI Arguments:**
| Argument | Type | Description |
|----------|------|-------------|
| `--predictions_npy` | str | Path to prediction matrix `.npy` file |
| `--ground_truth_npy` | str | Path to ground truth indices `.npy` file |
| `--predictions` | str | Alternative: path to pickle predictions |
| `--ground_truth` | str | Alternative: path to pickle ground truth |
| `--k` | int | Top-K for evaluation (default: 10) |
| `--split` | str | Split name when using pickle format |

**Command:**
```bash
python code/MS2/score.py \
    --predictions_npy "output/Milestone 2/ncf_baseline_test_predictions.npy" \
    --ground_truth_npy "output/Milestone 2/ncf_baseline_test_ground_truth.npy" \
    --k 10
```

**Sample Output:**
```
Loading predictions from: output/Milestone 2/ncf_baseline_test_predictions.npy
Loading ground truth from: output/Milestone 2/ncf_baseline_test_ground_truth.npy

Predictions shape: (27896, 5000)
Ground truth shape: (27896,)

RESULTS:
============================================================
  Hit Rate @ 10:  0.2612  (26.12%)
  NDCG @ 10:      0.1646
============================================================
```

---

### Milestone 3: Evaluation Notebooks

All notebooks are in `code/Evaluate/`. Run cells sequentially in Google Colab.

---

#### `score_v35_single_label_v2.ipynb`

Evaluates V3.5 predictions using **single-label** metrics (one correct subreddit per user).

**Important Note:** This notebook handles a dimension mismatch:
- Predictions are generated for ALL users (train + test): 33,727 users
- Ground truth only contains TEST users: 27,896 users
- The notebook extracts test user indices from `test.tsv` to filter predictions

**Input Files:**
| File | Shape | Description |
|------|-------|-------------|
| `cosine_v35_128.npy` | `(33727, 5000)` | 128-dim cosine similarity predictions |
| `cosine_v35_4096.npy` | `(33727, 5000)` | 4096-dim cosine similarity predictions |
| `predictions_model_A.npy` | `(33727, 5000)` | NCF + Text model predictions |
| `predictions_model_D.npy` | `(33727, 5000)` | NCF + Text (Gated) predictions |
| `ground_truth_v35_test.npy` | `(27896,)` | Correct subreddit index per test user |

**Configuration (Cell 6):**
```python
BASE_DIR = Path('/content/drive/MyDrive/CIS 5300/predictions')
DATA_DIR = Path('/content/drive/MyDrive/CIS 5300/engage_corpus_processed_v3_5tfidf')
GROUND_TRUTH_FILE = BASE_DIR / 'ground_truth_v35_test.npy'
TEST_FILE = DATA_DIR / 'ncf_data' / 'test.tsv'  # Used to extract test user indices
K = 10
```

**Sample Console Output:**
```
================================================================================
ENGAGE CORPUS EVALUATION - v3.5 DATA
================================================================================

Evaluating metrics: HR@10 and NDCG@10
Number of test users: 27896

================================================================================
Model: Cosine Similarity (v3.5, 4096-dim)
File: cosine_v35_4096.npy
--------------------------------------------------------------------------------
Full predictions shape: (33727, 5000)
Test predictions shape: (27896, 5000)

Evaluating...

✓ Results:
  Hit Rate @ 10:  0.3266  (32.66%)
  NDCG @ 10:      0.2823
================================================================================

SUMMARY - v3.5 DATA
================================================================================
                              Model     HR@10   NDCG@10
  Cosine Similarity (v3.5, 128-dim)   0.2733    0.2336
 Cosine Similarity (v3.5, 4096-dim)   0.3266    0.2823
               NCF Model A (v3.5)     0.1557    0.0993
================================================================================

BEST MODELS:

Best HR@10: Cosine Similarity (v3.5, 4096-dim)
  Score: 0.3266 (32.66%)

Best NDCG@10: Cosine Similarity (v3.5, 4096-dim)
  Score: 0.2823
```

---

#### `score_v4_multi_label_v2.ipynb`

Evaluates V4 predictions using **multi-label** metrics (multiple correct subreddits per user).

**Multi-Label Metrics:**
- **HR@K**: Fraction of users where at least one relevant item appears in top K
- **NDCG@K**: Normalized DCG accounting for multiple relevant items at each rank

**Input Files:**
| File | Shape | Description |
|------|-------|-------------|
| `cosine_v4_128.npy` | `(179936, 2000)` | 128-dim cosine similarity predictions |
| `cosine_v4_4096.npy` | `(179936, 2000)` | 4096-dim cosine similarity predictions |
| `predictions_model_B.npy` | `(179936, 2000)` | NCF Baseline (no text) |
| `predictions_model_C.npy` | `(179936, 2000)` | NCF + Text |
| `predictions_model_E.npy` | `(179936, 2000)` | NCF + Text (Gated) |
| `ground_truth_v4_test.npy` | `(179936, 2000)` | Binary interaction matrix |

**Configuration (Cell 6):**
```python
BASE_DIR = Path('/content/drive/MyDrive/CIS 5300/predictions')
GROUND_TRUTH_FILE = BASE_DIR / 'ground_truth_v4_test.npy'
K = 10
```

**Sample Console Output:**
```
Loading ground truth from: /content/drive/MyDrive/CIS 5300/predictions/ground_truth_v4_test.npy
Note: This is a large file (1.4 GB) and may take a moment...

Ground truth shape: (179936, 2000)
Ground truth dtype: float32
Number of users: 179936
Number of items: 2000

Ground Truth Statistics:
  Total interactions: 179,936
  Average interactions per user: 1.00
  Users with at least 1 interaction: 179,936 (100.00%)
  Sparsity: 99.95%

================================================================================
ENGAGE CORPUS EVALUATION - v4 DATA (MULTI-LABEL)
================================================================================

Evaluating metrics: HR@10 and NDCG@10
Number of users: 179,936
Number of items: 2,000

================================================================================
Model: NCF Model B (v4)
File: predictions_model_B.npy
--------------------------------------------------------------------------------
Loading predictions (this may take a moment for large files)...
✓ Loaded successfully
Predictions shape: (179936, 2000)
Predictions dtype: float32

Evaluating (processing 179,936 users)...

✓ Results:
  Hit Rate @ 10:  0.5321  (53.21%)
  NDCG @ 10:      0.2238
================================================================================

SUMMARY - v4 DATA (MULTI-LABEL)
================================================================================
                              Model       HR@10   NDCG@10
   Cosine Similarity (v4, 128-dim)      0.2865    0.1052
  Cosine Similarity (v4, 4096-dim)      0.3547    0.1398
                  NCF Model B (v4)      0.5321    0.2238
                  NCF Model C (v4)      0.5246    0.2221
            NCF Model E (v4 Gated)      0.4811    0.1871
================================================================================

BEST MODELS:

Best HR@10: NCF Model B (v4)
  Score: 0.5321 (53.21%)

Best NDCG@10: NCF Model B (v4)
  Score: 0.2238
```

---

#### `score_v4_chisq_multi_label.ipynb`

Evaluates Chi² ablation models (Models F and G) using **multi-label** metrics.

**Input Files:**
| File | Shape | Description |
|------|-------|-------------|
| `cosine_v4_chi2_128.npy` | `(179936, 2000)` | Chi² 128-dim cosine similarity |
| `predictions_model_F.npy` | `(179936, 2000)` | NCF + Text (Chi² features) |
| `predictions_model_G.npy` | `(179936, 2000)` | NCF + Text (Chi² + Gated) |
| `ground_truth_v4_chi2_test.npy` | `(179936, 2000)` | Binary interaction matrix |

**Configuration (Cell 6):**
```python
BASE_DIR = Path('/content/drive/MyDrive/CIS 5300/predictions')
GROUND_TRUTH_FILE = BASE_DIR / 'ground_truth_v4_chi2_test.npy'
K = 10
```

**Sample Console Output:**
```
================================================================================
ENGAGE CORPUS EVALUATION - v4 Chi² DATA (MULTI-LABEL)
================================================================================

Evaluating metrics: HR@10 and NDCG@10
Number of users: 179,936
Number of items: 2,000

================================================================================
Model: NCF Model F (Chi²)
File: predictions_model_F.npy
--------------------------------------------------------------------------------
✓ Loaded successfully
Predictions shape: (179936, 2000)

Evaluating (processing 179,936 users)...

✓ Results:
  Hit Rate @ 10:  0.4422  (44.22%)
  NDCG @ 10:      0.1667
================================================================================

SUMMARY - v4 Chi² DATA (MULTI-LABEL)
================================================================================
                              Model       HR@10   NDCG@10
  Cosine Similarity (Chi2, 128-dim)     0.2865    0.1052
                NCF Model F (Chi²)      0.4422    0.1667
          NCF Model G (Chi² Gated)      0.4788    0.1857
================================================================================

BEST MODELS:

Best HR@10: NCF Model G (Chi² Gated)
  Score: 0.4788 (47.88%)

Best NDCG@10: NCF Model G (Chi² Gated)
  Score: 0.1857
```

---

## Expected Scores

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