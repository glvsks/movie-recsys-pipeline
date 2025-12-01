# MovieLens Recommender System (ALS + TF-IDF)

## TL;DR

This project implements a clean, modular recommendation pipeline comparing **Content-Based** (TF-IDF) and **Collaborative Filtering** (ALS) methods on the MovieLens 100k dataset.

It shows a maintainable codebase with OOP structure, sparse matrix optimizations, and custom ranking metrics (NDCG@K, Precision@K).

**Goal**: demonstrate a clean, production-oriented recommender pipeline using both Content-Based and Collaborative Filtering approaches.

**Key result:** ALS clearly outperforms the Content-Based baseline
(NDCG@10: **0.30** vs Precision@10: **0.006** for the baseline).

## Tech Stack

* **Python** 3.10+
* `implicit` (ALS for implicit feedback), `scikit-learn` (TF-IDF, cosine similarity)
* `pandas`, `numpy`, `scipy` (sparse matrices)

## Dataset
The project uses the MovieLens 100k dataset (User × Movie explicit ratings).  
It contains ~100k interactions, ~1k users, and ~1.7k movies with simple metadata (genres).  
The dataset is downloaded automatically on the first run.
 

## Approach & Architecture

### 1. Content-Based Filtering (Baseline)

* **Features:** TF-IDF vectors built from movie genres.
* **Similarity:** Cosine similarity between user profile vectors and items.
* **Notes:** Handles new items well but tends to produce obvious, narrow recommendations.

### 2. Collaborative Filtering (ALS)

* **Algorithm:** Alternating Least Squares from the `implicit` library.
* **Why ALS:** Scales well and handles sparse implicit feedback matrices efficiently.
* **Setup:**

  * `factors=50`
  * `regularization=0.01`
  * Implicit feedback using explicit ratings as confidence weights.

## Project Structure

```text
movie-recsys-pipeline/
├── data/                  # Auto-downloaded (ignored by Git)
├── src/
│   ├── __init__.py
│   ├── config.py          # Hyperparameters and paths
│   ├── data_loader.py     # ETL logic
│   ├── preprocessor.py    # Sparse matrix & TF-IDF builder
│   ├── models.py          # ALS + Content-Based model wrappers
│   └── metrics.py         # Precision@K, NDCG implementations
├── main.py                # Entry point
├── requirements.txt
└── README.md
```

## How to Run

The pipeline automatically downloads and prepares the data on the first run.


   ```bash
   # clone the repository
   git clone https://github.com/glvsks/movie-recsys-pipeline.git
   cd movie-recsys-pipeline

   # create a virtual environment & install dependencies
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt

   # run the pipeline
   python main.py
   ```

## What You'll See

The script executes the full workflow:

1. **ETL:** Downloads `ml-latest-small.zip`, extracts it into `data/raw`, loads into Pandas.
2. **Preprocessing:** Converts explicit ratings into a **CSR sparse matrix** (≈98% sparsity).
3. **Training:**

   * ALS model for implicit feedback (matrix factorization).
   * Content-Based model using TF-IDF on movie genres.
4. **Evaluation:** Precision@10 and NDCG@10 on a 20% hold-out test split.

**Example output:**

```text
==================================================
RUNNING MOVIELENS RECSYS PIPELINE
==================================================
[INFO] Loading data into memory...

[INFO] Creating matrix...
[INFO] Building TF-IDF matrix based on genres...

--------------------
MODEL 1: ALS (Collaborative Filtering)
--------------------
100%|██████████████████████████████████████████████████████████| 20/20 [00:01<00:00, 16.08it/s]
ALS Precision@10: 0.2618

--------------------
MODEL 2: Content-Based (TF-IDF)
--------------------
Computing metrics for the Content-Based...
Content-Based Precision@10: 0.0059

==================================================
RESULT: ALS (0.2618) vs Content-Based (0.0059)
==================================================
```

## What I Practiced

* **Recommender system fundamentals:**
  Built both Content-Based and Collaborative Filtering models and compared their behavior on real sparse data.

* **ALS for implicit feedback:**
  Worked with Alternating Least Squares (`implicit`), tuning factors and regularization, understanding how latent representations capture user–item preferences.

* **Sparse matrix pipelines:**
  Converted ratings into CSR matrices and optimized preprocessing steps for >98% sparsity.

* **Ranking metrics:**
  Implemented Precision@K and NDCG@K manually to evaluate top-K recommendation quality.

* **Content-based modeling:**
  Used TF-IDF on movie metadata (genres) and explored typical limitations such as overspecialization and weak cold-start handling.


## Future Work

* **Hybrid models:** Combine ALS and Content-Based scores (e.g., LightFM-style).
* **Hyperparameter tuning:** Grid Search / Optuna.
* **Time-based evaluation:** More realistic than random train/test split.
* **Feature engineering:** Incorporate movie tags, release years, and additional metadata.


## Author
Sofia Gulevskaia

https://github.com/glvsks
