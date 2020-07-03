from procs.filter.clustering_filter import Filter
import numpy as np

class SequentialFilter(Filter):

    def __init__(self, arr_title, arr_weight, title_index, arr_title_vector, img_index, arr_img_vector):
        self.arr_title = arr_title
        
        self.arr_weight = arr_weight
        
        self.title_index = title_index
        self.arr_title_vector = arr_title_vector
        
        self.img_index = img_index
        self.arr_img_vector = arr_img_vector

    def do(self, query_title, filter_type, thrshold=0.1, cos_threshold=0.8, weight_check=True):

        find_idx = np.where(self.arr_title == query_title)[0].item(0)

        if filter_type == 'img':
            search_index = self.img_index
            search_vector = self.arr_img_vector
            second_filter_vector = self.arr_title_vector

        else:
            search_index = self.title_index
            search_vector = self.arr_title_vector
            second_filter_vector = self.arr_img_vector

        # idx, title, distance
        arr_sim_idx, sim_titles, _ = self._search_range_idx(query_title, self.arr_title, \
                                                                  search_index, search_vector, thrshold)

        ## 2차 필터링용
        find_vector = second_filter_vector[find_idx]
        sim_search_vector = second_filter_vector[arr_sim_idx]

        similarity_array = self._cal_cosine_similarity(find_vector, sim_search_vector)
        sim_title_idx = np.where(similarity_array > cos_threshold)[0]

        ## 3차 필터링 (중량)
        if weight_check:
            find_weight = self.arr_weight[find_idx].item()
            if str(find_weight) == 'nan':
                weight_check = False
                
        # 유사 title이 없는경우 조건 ..?
        if len(sim_title_idx) == 0:
            if weight_check:
                sim_arr_weight = self.arr_weight[arr_sim_idx]
                arr_sim_idx = self._filter_weight(find_weight, sim_arr_weight, arr_sim_idx)

            arr_sim_title = self.arr_title[arr_sim_idx]
        
        else:

            find_sim_title_idx = arr_sim_idx[sim_title_idx]

            if weight_check:
                sim_arr_weight = self.arr_weight[find_sim_title_idx]
                find_sim_title_idx = self._filter_weight(find_weight, sim_arr_weight, find_sim_title_idx)

            arr_sim_title = self.arr_title[find_sim_title_idx]

        return arr_sim_title