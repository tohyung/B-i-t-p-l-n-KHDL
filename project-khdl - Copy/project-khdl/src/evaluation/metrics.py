import numpy as np

class Evaluator:
    @staticmethod
    def rmse(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def precision_at_k(recommended_items, relevant_items, k=10):
        if not relevant_items:
            return 0.0
        rec_k = set(recommended_items[:k])
        rel = set(relevant_items)
        return len(rec_k.intersection(rel)) / float(k)

    @staticmethod
    def recall_at_k(recommended_items, relevant_items, k=10):
        if not relevant_items:
            return 0.0
        rec_k = set(recommended_items[:k])
        rel = set(relevant_items)
        return len(rec_k.intersection(rel)) / float(len(rel))

    @staticmethod
    def ndcg_at_k(recommended_items, relevant_items, k=10):
        if not relevant_items:
            return 0.0
        
        dcg = 0.0
        for i, item in enumerate(recommended_items[:k]):
            if item in relevant_items:
                dcg += 1.0 / np.log2(i + 2)
                
        idcg = 0.0
        for i in range(min(len(relevant_items), k)):
            idcg += 1.0 / np.log2(i + 2)
            
        return dcg / idcg if idcg > 0 else 0.0
