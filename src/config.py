import os

# data paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

# URL 
DOWNLOAD_URL = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'

# ALS params
ALS_PARAMS = {
    'factors': 50,
    'regularization': 0.01,
    'iterations': 20,
    'random_state': 42
}

TOP_K = 10