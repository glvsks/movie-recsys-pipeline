# MovieLens Hybrid Recommender Pipeline

## 📌 TL;DR
This project implements a modular recommendation pipeline comparing **Content-Based** (TF-IDF) and **Collaborative Filtering** (ALS) approaches on the MovieLens 100k dataset.

It demonstrates a transition from a research notebook to production-ready code with **OOP structure**, **sparse matrix optimizations**, and custom **ranking metrics** (NDCG@K, Precision@K).

**Key Result:** The ALS model significantly outperformed the baseline Content-Based approach (NDCG@10: **0.30** vs Precision@10: **0.006**).

---

## 🛠 Tech Stack
*   **Core:** Python 3, NumPy, Pandas
*   **ML & RecSys:** `implicit` (ALS), `scikit-learn` (TF-IDF, Cosine Similarity), `scipy` (Sparse Matrices)
*   **Engineering:** Modular architecture, Type hinting, Git

---

## 🚀 How to Run

The project automatically downloads and processes the data upon the first run.

1.  **Clone the repository**
    ```bash
    git clone https://github.com/YOUR_USERNAME/movie-recsys-pipeline.git
    cd movie-recsys-pipeline
    ```

2.  **Create virtual environment & install dependencies**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Run the pipeline**
    ```bash
    python main.py
    ```

---

## 👁 What You'll See

The script will execute the full pipeline:
1.  **ETL:** Downloads `ml-latest-small.zip`, extracts it to `data/raw`, and loads it into Pandas.
2.  **Preprocessing:** Converts explicit ratings into a CSR Sparse Matrix (Sparsity ~98.3%).
3.  **Training:**
    *   Trains an **ALS** model (Implicit feedback) using Matrix Factorization.
    *   Builds a **Content-Based** model using TF-IDF on movie genres.
4.  **Evaluation:** Calculates Precision@10 and NDCG@10 on a hold-out test set (20%).

**Sample Output:**
```text
[INFO] Training ALS model (factors=50, iter=20)...
[INFO] Calculating metrics...

------------------------------
RESULTS (Top-10)
------------------------------
ALS Precision@10:       0.2615
ALS NDCG@10:            0.3078
------------------------------
Content-Based Precision@10: 0.0060
------------------------------
```

---

## 🧠 Approach & Architecture

### 1. Content-Based Filtering (Baseline)
*   **Feature Engineering:** Constructed item vectors using **TF-IDF** on movie genres.
*   **Similarity:** Used Cosine Similarity to find items closest to the user's profile (mean vector of watched movies).
*   **Pros/Cons:** Solves the "cold start" problem for new items but suffers from over-specialization (recommendations are too obvious).

### 2. Collaborative Filtering (ALS)
*   **Algorithm:** Alternating Least Squares (ALS) from the `implicit` library.
*   **Why ALS?** It efficiently handles large, sparse matrices and scales well compared to KNN-based approaches.
*   **Configuration:**
    *   `factors=50`: Latent space dimension.
    *   `regularization=0.01`: To prevent overfitting.
    *   **Implicit Feedback:** Explicit ratings (1-5) were treated as confidence weights for the interaction.

---

## 📂 Project Structure

```text
movie-recsys-pipeline/
├── data/                  # Ignored by Git (auto-downloaded)
├── src/
│   ├── __init__.py
│   ├── config.py          # Hyperparameters and paths
│   ├── data_loader.py     # ETL logic
│   ├── preprocessor.py    # Sparse matrix & TF-IDF creation
│   ├── models.py          # OOP wrappers for ALS and Content-Based models
│   └── metrics.py         # Custom implementation of Precision@K and NDCG
├── main.py                # Entry point
├── requirements.txt
└── README.md
```

---

## 🎓 What I Practiced
*   **Refactoring:** Moving from a monolithic Jupyter Notebook to a structured Python package.
*   **Sparse Data Handling:** Working with `scipy.sparse.csr_matrix` to handle high sparsity (98.3%).
*   **Metric Implementation:** Manually implementing **NDCG** (Normalized Discounted Cumulative Gain) to understand ranking quality deeply.
*   **Library Management:** Resolving version conflicts and adapting to the `implicit` library API.

---

## 🔮 Future Work
*   **Hybridization:** Combine ALS and Content-Based scores (e.g., using LightFM) to handle the cold start problem better.
*   **Hyperparameter Tuning:** Implement Grid Search or Optuna to optimize ALS factors and regularization.
*   **Time-based Split:** Replace random train/test split with a time-based split to simulate real-world production scenarios.
*   **Feature Engineering:** Incorporate movie `tags` and `year` into the TF-IDF model to improve the baseline.
```

---
