import implicit
import os
from sklearn.metrics.pairwise import linear_kernel
import numpy as np


# limit the number of threads for OpenBLAS 
os.environ['OPENBLAS_NUM_THREADS'] = '1'

class ALSRecommender:
    """
    Wrapper around implicit.als.AlternatingLeastSquares.
    """
    def __init__(self, params):
        self.model = implicit.als.AlternatingLeastSquares(**params)
        self.train_matrix = None

    def fit(self, train_matrix):
        """
        Trains the model on a sparse (User x Item) matrix.
        """
        self.train_matrix = train_matrix
        self.model.fit(train_matrix)

    def recommend(self, user_id, n=10, filter_liked=True):
        """
        Returns recommendations for a user.
        """
        if self.train_matrix is None:
            raise ValueError("The model is not trained! Call .fit() first")
            
        ids, scores = self.model.recommend(
            user_id, 
            self.train_matrix[user_id], 
            N=n, 
            filter_already_liked_items=filter_liked
        )
        return ids, scores
    

class ContentBasedRecommender:
    """
    Genre-based recommendations (TF-IDF).
    Logic:
    1. Build a user profile as the average of vectors of the movies they’ve watched.
    2. Find movies closest to the user profile using cosine similarity.
    """
    def __init__(self, tfidf_matrix, index_to_movie_map):
        self.tfidf_matrix = tfidf_matrix
        self.index_to_movie = index_to_movie_map
        self.train_matrix = None

    def fit(self, train_matrix):
        """
        There is no actual “training” for Content-Based,
        but we do need to store the user’s watch history (train_matrix).
        """
        self.train_matrix = train_matrix

    def recommend(self, user_id, n=10, filter_liked=True):
        """
        Generates recommendations.
        """
        user_history_indices = self.train_matrix[user_id].indices
        
        if len(user_history_indices) == 0:
            return [], []
        
        user_profile = np.asarray(self.tfidf_matrix[user_history_indices].mean(axis=0))

        cosine_sim = linear_kernel(user_profile, self.tfidf_matrix).flatten()

        if filter_liked:
            cosine_sim[user_history_indices] = -1

        similar_indices = cosine_sim.argsort()[::-1][:n]
        similar_scores = cosine_sim[similar_indices]

        return similar_indices, similar_scores