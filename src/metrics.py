import numpy as np

def precision_at_k(model, train_matrix, test_matrix, k=10):
    """
    Считает средний Precision@K по всем пользователям в тесте.
    """
    test_users = np.unique(test_matrix.nonzero()[0])
    precisions = []
    
    for user_idx in test_users:
        actual_items = test_matrix[user_idx].indices
        
        if len(actual_items) == 0:
            continue
            
        recommended_ids, _ = model.recommend(user_idx, n=k, filter_liked=True)
        
        hits = np.isin(recommended_ids, actual_items).sum()
        precisions.append(hits / k)
        
    return np.mean(precisions)

def ndcg_at_k(model, train_matrix, test_matrix, k=10):
    """
    Counts mean NDCG@K.
    """
    test_users = np.unique(test_matrix.nonzero()[0])
    ndcg_scores = []
    
    for user_idx in test_users:
        actual_items = test_matrix[user_idx].indices
        if len(actual_items) == 0:
            continue
            
        recommended_ids, _ = model.recommend(user_idx, n=k, filter_liked=True)
        
        dcg = 0.0
        for i, item_id in enumerate(recommended_ids):
            if item_id in actual_items:
                dcg += 1.0 / np.log2(i + 2)
                
        idcg = 0.0
        num_possible_hits = min(len(actual_items), k)
        for i in range(num_possible_hits):
            idcg += 1.0 / np.log2(i + 2)
            
        if idcg > 0:
            ndcg_scores.append(dcg / idcg)
        else:
            ndcg_scores.append(0.0)
            
    return np.mean(ndcg_scores)