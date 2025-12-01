import os
import requests
import zipfile
import io
import pandas as pd
from src.config import DATA_DIR, DOWNLOAD_URL

def download_and_load_data():
    """
    Dowloads MovieLens dataset, unpacks it, and loads it into a pandas DataFrame.
    """
    ratings_path = os.path.join(DATA_DIR, 'ml-latest-small', 'ratings.csv')
    movies_path = os.path.join(DATA_DIR, 'ml-latest-small', 'movies.csv')

    # Проверяем, существуют ли файлы
    if not os.path.exists(ratings_path) or not os.path.exists(movies_path):
        print(f"[INFO] Data not found. Download from {DOWNLOAD_URL}...")
        os.makedirs(DATA_DIR, exist_ok=True)
        
        try:
            r = requests.get(DOWNLOAD_URL)
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(DATA_DIR)
            print("[INFO] Data successfully downloaded and unpacked.")
        except Exception as e:
            print(f"[ERROR] Error while downloading data: {e}")
            raise

    print("[INFO] Loading data into memory...")
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    
    return ratings, movies