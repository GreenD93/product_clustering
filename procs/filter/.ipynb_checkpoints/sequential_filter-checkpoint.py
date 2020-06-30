from procs.filter.clustering_filter import Filter
import numpy as np

class SequentialFilter(Filter):

    def __init__(self, title_array, weight_array, title_index, title_vector, img_index, img_vector):
        self.title_array = title_array
        
        self.weight_array = weight_array
        
        self.title_index = title_index
        self.title_vector = title_vector
        
        self.img_index = img_index
        self.img_vector = img_vector

    def do(self, find_title, filter_type, thrshold=0.1, cos_threshold=0.8, weight_check=True):

        find_index = np.where(self.title_array == find_title)[0].item(0)

        if filter_type == 'img':
            search_index = self.img_index
            search_vector = self.img_vector
            second_filter_vector = self.title_vector

        else:
            search_index = self.title_index
            search_vector = self.title_vector
            second_filter_vector = self.img_vector

        # idx, title, distance
        sim_index_array, sim_titles, _ = self._search_range_index(find_title, self.title_array, \
                                                                  search_index, search_vector, thrshold)

        ## 2차 필터링용
        find_vector = second_filter_vector[find_index]
        sim_search_vector = second_filter_vector[sim_index_array]

        similarity_array = self._cal_cosine_similarity(find_vector, sim_search_vector)
        sim_title_idx = np.where(similarity_array > cos_threshold)[0]

        ## 3차 필터링 (중량)
        if weight_check:
            find_weight = self.weight_array[find_index].item()
            if str(find_weight) == 'nan':
                weight_check = False
                
        # 유사 title이 없는경우 조건 ..?
        if len(sim_title_idx) == 0:
            if weight_check:
                sim_weight_array = self.weight_array[sim_index_array]
                sim_index_array = self._weight_filter(find_weight, sim_weight_array, sim_index_array)

            get = self.title_array[sim_index_array]
        
        else:

            find_sim_title_idx = sim_index_array[sim_title_idx]

            if weight_check:
                sim_weight_array = self.weight_array[find_sim_title_idx]
                find_sim_title_idx = self._weight_filter(find_weight, sim_weight_array, find_sim_title_idx)

            get = self.title_array[find_sim_title_idx]

        return get