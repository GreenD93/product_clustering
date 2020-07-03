import numpy as np

class Filter():

    def __init__():
        pass

    def _filter_weight(self, query_weight, arr_sim_weight, sim_idx):
        
        equal_arr_weight = np.where(arr_sim_weight == query_weight)[0]
        sim_array = sim_idx[equal_arr_weight]
        
        return sim_array

    def _find_idx_vector(self, query, arr_query, arr_query_vector):
        # find index
        idx = np.isin(arr_query, query)
        query_idx = np.where(idx)[0][0]

        # query vector (cleaned_title이 vector가 된 곳에서 query index로 찾기)
        # 실제 서비스에서 db로 대체 가능
        query_vector = arr_query_vector[query_idx]

        return query_idx, query_vector

    def _find_range_sim_query(self, query_vector, index, arr_query, threshold):
        
        _, arr_sim_score, arr_sim_idx = index.range_search(query_vector, threshold)

        arr_sim_idx = arr_sim_idx.reshape(-1)
        arr_sim_score = arr_sim_score.reshape(-1)

        # 실제 서비스에서는 db로 교체
        scores = []
        sim_titles = []
        
        for idx, score in zip(arr_sim_idx, arr_sim_score):
            sim_titles.append(arr_query[idx])
            scores.append(score)

        return arr_sim_idx, sim_titles, scores

    def _search_range_idx(self, query, arr_query, index, arr_query_vector, threshold):

        vector_dim = arr_query_vector[0].shape[0]

        # title index, vector
        query_idx, query_vector = self._find_idx_vector(query, arr_query, arr_query_vector)

        query_vector = query_vector.reshape(-1, vector_dim)
        arr_sim_idx, sim_titles, scores = self._find_range_sim_query(query_vector, index, arr_query, threshold)

        return arr_sim_idx, sim_titles, scores

    def _cal_cosine_similarity(self, query_vector, sim_query_vector):
        arr_similarity = np.ones(len(sim_query_vector))

        for num, sim_vector in enumerate(sim_query_vector):
            similarity = np.inner(query_vector, sim_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(sim_vector))
            arr_similarity[num] = similarity

        return arr_similarity