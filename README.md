# movie-recsys-pipeline
A complete recommendation pipeline built on MovieLens: content-based model, ALS collaborative filtering, evaluation, and examples.

# MovieLens Recommender Pipeline

A modular recommendation system pipeline comparing **Content-Based** and **Collaborative Filtering (ALS)** approaches on the MovieLens 100k dataset.

The project implements a reproducible end-to-end workflow: from automatic data ingestion and sparse matrix construction to model training and evaluation using custom ranking metrics (Precision@K, NDCG@K).

## 🚀 Quick Start

The pipeline automatically downloads the dataset if not present.

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/movie-recsys-pipeline.git
cd movie-recsys-pipeline

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline
python main.py
