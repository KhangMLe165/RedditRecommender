# ENGAGE Corpus Data Processing

This document describes how to download, extract, and process the ENGAGE Corpus for subreddit recommendation experiments.

## Data Source

This dataset is derived from the [ENGAGE Corpus](https://github.com/engage-corpus/dataset), which contains the posts and comments of Reddit users from 2017-2020.

## Download and Extraction

1. Download the two-part archive from the ENGAGE Corpus repository (via the link above)
2. Combine and extract:

```bash
cat reddit-data-01-part-01.tar.gz reddit-data-01-part-02.tar.gz > reddit.tar.gz
tar -xzf reddit.tar.gz
```

This produces 24 JSON-lines files (`data00.json` through `data23.json`) in a `converted_data/` directory.

## Schema

Each line in the data files is a JSON object representing one user:

| Field | Type | Description |
|-------|------|-------------|
| `user_number` | int | Unique integer ID for each user |
| `posts` | array | List of post objects |
| `comments` | array | List of comment objects |

**Post object:**
| Field | Type | Description |
|-------|------|-------------|
| `created_utc` | int | Unix timestamp (seconds) |
| `title` | string | Post title |
| `selftext` | string | Post body text |
| `id` | string | Unique post ID |
| `score` | int | Net upvotes |
| `subreddit` | string | Subreddit name |

**Comment object:**
| Field | Type | Description |
|-------|------|-------------|
| `created_utc` | int | Unix timestamp (seconds) |
| `body` | string | Comment text |
| `id` | string | Unique comment ID |
| `link_id` | string | Parent post ID |
| `score` | int | Net upvotes |
| `subreddit` | string | Subreddit name |

## Example Record

```json
{
  "user_number": 2059,
  "posts": [
    {
      "created_utc": 1574505378,
      "title": "AGM 28 hound dog, American cruse missile",
      "selftext": "",
      "id": "e0ga72",
      "score": 76,
      "subreddit": "WeirdWings"
    }
  ],
  "comments": [
    {
      "created_utc": 1573296855,
      "body": "It could also be using differential thrust...",
      "id": "f6ytnt9",
      "link_id": "t3_ds9zis",
      "score": 3,
      "subreddit": "WeirdWings"
    }
  ]
}
```

---

## Processing Versions

### v3.5 (TF-IDF)

**Subreddit Selection:** Top 5,000 subreddits by unique user count across all 2019 activity.

**User Filtering:**
1. Must have at least one interaction in a top-5000 subreddit
2. Randomly sample 6% of eligible users (seed=42)
3. Must have ≥2 items in the evaluation period (Oct–Dec)

**Temporal Splits:**

| Period | Date Range | Unix Timestamps |
|--------|------------|-----------------|
| Context | Jan 1 – June 30, 2019 | 1546300800 – 1561939199 |
| VW Target | July 1 – Sept 30, 2019 | 1561939200 – 1569887999 |
| Eval | Oct 1 – Dec 31, 2019 | 1569888000 – 1577836799 |

**Train/Dev/Test Assignment:**
- **Train:** All unique user-subreddit pairs from Jan–Sept (context + vw_target)
- **Dev:** First chronological item from eval period
- **Test:** Second chronological item from eval period

**Text Filtering:** TF-IDF computed per user (each user = one document). Top 50 words by TF-IDF score retained per entity.

**Run:**
```bash
python process_engage_corpus_v3_5_tfidf.py \
    --data_dir ./converted_data \
    --output_dir engage_corpus_processed_v3_5
```

---

### v4 (TF-IDF)

**Subreddit Selection:** Top 2,000 subreddits by unique user count.

**User Filtering (stricter):**
1. Must have activity in **both** Jan–June AND July–Sept
2. Total posts+comments in train period (Jan–Sept) must be 50–5,000 inclusive
3. At least one interaction in a top-2000 subreddit
4. **No random downsampling** — all qualifying users included

**Temporal Splits:**

| Period | Date Range | Unix Timestamps |
|--------|------------|-----------------|
| Train Input | Jan 1 – June 30, 2019 | 1546300800 – 1561939199 |
| Train Target | July 1 – Sept 30, 2019 | 1561939200 – 1569887999 |
| Dev | Oct 1–31 **and** Dec 1–31, 2019 | 1569888000–1572566399, 1575158400–1577836799 |
| Test | Nov 1–30, 2019 | 1572566400 – 1575158399 |

**Train/Dev/Test Assignment:**
- **Train:** All unique user-subreddit pairs from train target period (July–Sept)
- **Dev:** All unique user-subreddit pairs from Oct + Dec
- **Test:** All unique user-subreddit pairs from Nov

This is **multi-label**: each user can have multiple positive subreddit interactions per split.

**Text Filtering:** Same TF-IDF approach as v3.5, top 50 words retained.

**Run:**
```bash
python process_engage_corpus_v4.py \
    --data_dir ./converted_data \
    --output_dir engage_corpus_processed_v4
```

---

### Chi-Squared Ablation

Replaces TF-IDF feature selection with Chi-squared (χ²) feature selection. Available for both v3.5 and v4 filtering schemes.

**Chi-Squared Computation:**
1. Build term-frequency vectors per subreddit (training text only)
2. Vectorize using `sklearn.DictVectorizer` (sparse)
3. Treat each subreddit as its own class label
4. Compute χ² scores in batches of 5,000 features
5. Normalize all scores by the maximum score
6. Select top 50 words per entity by χ² score

**Key difference from TF-IDF:** χ² measures feature discriminativeness across subreddits globally, while TF-IDF is computed per-document.

**Run v3.5 with Chi²:**
```bash
python process_engage_corpus_v3_5_chi2.py \
    --data_dir ./converted_data \
    --output_dir engage_corpus_processed_v3_5_chi2
```

**Run v4 with Chi²:**
```bash
python process_engage_corpus_v4_chi2.py \
    --data_dir ./converted_data \
    --output_dir engage_corpus_processed_v4_chi2
```

---

## Output Structure

All versions produce the same directory structure:

```
output_dir/
├── ncf_data/
│   ├── train.tsv          # user_id \t subreddit_idx [\t label]
│   ├── dev.tsv
│   └── test.tsv
├── text_context/
│   ├── user_text_train.json
│   ├── user_text_dev.json
│   ├── user_text_test.json
│   ├── subreddit_text_train.json
│   ├── subreddit_text_dev.json
│   └── subreddit_text_test.json
├── text_context_filtered/
│   └── ... (same structure, with filtered text)
├── user_mapping.json
├── subreddit_mapping.json
└── summary_stats.json
```

---

## Summary Comparison

| Aspect | v3.5 | v4 |
|--------|------|-----|
| Subreddits | Top 5,000 | Top 2,000 |
| User sampling | 6% random | All qualifying |
| Min train items | None | 50 |
| Max train items | None | 5,000 |
| Require both periods | No | Yes (Jan–June AND July–Sept) |
| Dev period | 1st eval item | Oct + Dec |
| Test period | 2nd eval item | Nov |
| Labels per user | Single | Multi-label |
