from procs.filter.clustering_filter import Filter
import numpy as np

class ConcatFilter(Filter):

    def __init__(self, arr_title, arr_weight, concat_index, arr_concat_vector):

        self.arr_title = arr_title
        self.arr_weight = arr_weight

        self.concat_index = concat_index
        self.arr_concat_vector = arr_concat_vector

    def do(self, query_title, threshold=0.1, weight_check=True):

        # 예측 array, idx, title, distance
        sim_arr_idx, sim_titles, _ = self._search_range_idx(query_title, self.arr_title, \
                                                                  self.concat_index, self.arr_concat_vector, threshold)

        find_idx = np.where(self.arr_title == query_title)[0].item(0)

        ## 2차 필터링 (중량)
        if weight_check:
            find_weight = self.arr_weight[find_idx].item()
            if str(find_weight) == 'nan':
                weight_check = False

        if weight_check:
            sim_arr_weight = self.arr_weight[sim_arr_idx]
            sim_arr_idx = self._filter_weight(find_weight, sim_arr_weight, sim_arr_idx)

        arr_sim_title = self.arr_title[sim_arr_idx]
        arr_sim_title = arr_sim_title.flatten()

        return arr_sim_title
