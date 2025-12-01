import sys
from implicit.evaluation import train_test_split
from src.data_loader import download_and_load_data
from src.preprocessor import create_interaction_matrix, prepare_tfidf_matrix
from src.models import ALSRecommender, ContentBasedRecommender
from src.metrics import precision_at_k
from src.config import ALS_PARAMS, TOP_K

def main():
    print("="*50)
    print("RUNNING MOVIELENS RECSYS PIPELINE")
    print("="*50)

    # 1. Загрузка
    try:
        ratings, movies = download_and_load_data()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    # 2. Препроцессинг
    print("\n[INFO] Creating matrix...")
    user_item_matrix, user_map, movie_map, index_to_movie = create_interaction_matrix(ratings)
    
    # !!! НОВОЕ: Готовим TF-IDF для Content-Based модели
    print("[INFO] Building TF-IDF matrix based on genres...")
    tfidf_matrix = prepare_tfidf_matrix(movies)

    # 3. Сплит
    train_csr, test_csr = train_test_split(user_item_matrix, train_percentage=0.8, random_state=42)

    # --- МОДЕЛЬ 1: ALS ---
    print("\n" + "-"*20)
    print("MODEL 1: ALS (Collaborative Filtering)")
    print("-"*20)
    als_model = ALSRecommender(ALS_PARAMS)
    als_model.fit(train_csr)
    
    als_precision = precision_at_k(als_model, train_csr, test_csr, k=TOP_K)
    print(f"ALS Precision@{TOP_K}: {als_precision:.4f}")

    # --- МОДЕЛЬ 2: Content-Based ---
    print("\n" + "-"*20)
    print("MODEL 2 2: Content-Based (TF-IDF)")
    print("-"*20)
    # Передаем tfidf матрицу и маппинг
    cb_model = ContentBasedRecommender(tfidf_matrix, index_to_movie)
    cb_model.fit(train_csr) # Просто запоминаем историю
    
    # Внимание: Content-Based работает медленнее, так как считает косинус для каждого юзера в цикле
    print("Computing metrics for the Content-Based...")
    cb_precision = precision_at_k(cb_model, train_csr, test_csr, k=TOP_K)
    print(f"Content-Based Precision@{TOP_K}: {cb_precision:.4f}")

    # Итог
    print("\n" + "="*50)
    print(f"RESULT: ALS ({als_precision:.4f}) vs Content-Based ({cb_precision:.4f})")
    print("="*50)

if __name__ == "__main__":
    main()