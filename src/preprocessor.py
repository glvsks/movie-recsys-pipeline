import numpy as np
import scipy.sparse as sparse
from sklearn.feature_extraction.text import TfidfVectorizer

def create_interaction_matrix(ratings_df):
    """
    Creates a sparse user-term interaction matrix.
    returns the matrix and the mapping dictionaries.
    """
    # unique users and films
    users = list(np.sort(ratings_df['userId'].unique()))
    movies = list(np.sort(ratings_df['movieId'].unique()))

    # mapping (Real ID -> Internal Index)
    user_map = {u: i for i, u in enumerate(users)}
    movie_map = {m: i for i, m in enumerate(movies)}

    # reverse mapping (Internal Index -> Real ID)
    index_to_user = {i: u for u, i in user_map.items()}
    index_to_movie = {i: m for m, i in movie_map.items()}

    ratings_df['user_idx'] = ratings_df['userId'].map(user_map)
    ratings_df['movie_idx'] = ratings_df['movieId'].map(movie_map)

    # sparse matrix creation (CSR)
    user_item_matrix = sparse.csr_matrix(
        (ratings_df['rating'], (ratings_df['user_idx'], ratings_df['movie_idx'])),
        shape=(len(users), len(movies))
    )

    return user_item_matrix, user_map, movie_map, index_to_movie

def prepare_tfidf_matrix(movies_df):
    """
    Creates a TF-IDF matrix based on movie genres.
    """
    movies_df['genres_str'] = movies_df['genres'].str.replace('|', ' ', regex=False)
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['genres_str'])
    
    return tfidf_matrix