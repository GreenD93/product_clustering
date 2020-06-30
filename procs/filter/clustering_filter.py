import numpy as np

class Filter():

    def __init__():
        pass

    def _weight_filter(self, find_weight, sim_weight_array, sim_idx):
        equal_weight_array = np.where(sim_weight_array == find_weight)[0]
        final_sim_array = sim_idx[equal_weight_array]
        return final_sim_array

    def _find_idx_vector(self, find_title, title_array, word_vectors):
        # find index
        idx = np.isin(title_array, find_title)
        find_idx = np.where(idx)[0][0]

        # find vector (cleaned_title이 vector가 된 곳에서 title index로 찾기)
        # 실제 서비스에서 db로 대체 가능
        find_vector = word_vectors[find_idx]

        return find_idx, find_vector

    def _find_range_sim_title(self, title_vector, index, title_array, threshold):
        scores = []
        sim_titles = []

        _, sim_score_array, sim_index_array = index.range_search(title_vector, threshold)

        sim_index_array = sim_index_array.reshape(-1)
        sim_score_array = sim_score_array.reshape(-1)

        # 실제 서비스에서는 db로 교체
        for idx, score in zip(sim_index_array, sim_score_array):
            sim_titles.append(title_array[idx])
            scores.append(score)

        return sim_index_array, sim_titles, scores

    def _search_range_index(self, find_title, title_array, index, vector, threshold):

        vector_dim = vector[0].shape[0]

        # title index, vector
        find_idx, find_vector = self._find_idx_vector(find_title, title_array, vector)

        find_vector = find_vector.reshape(-1, vector_dim)
        sim_index_array, sim_titles, scores = self._find_range_sim_title(find_vector, index, title_array, threshold)

        return sim_index_array, sim_titles, scores

    def _cal_cosine_similarity(self, find_vector, sim_title_vector):
        similarity_array = np.ones(len(sim_title_vector))

        for num, sim_vector in enumerate(sim_title_vector):
            similarity = np.inner(find_vector, sim_vector) / (np.linalg.norm(find_vector) * np.linalg.norm(sim_vector))
            similarity_array[num] = similarity

        return similarity_array