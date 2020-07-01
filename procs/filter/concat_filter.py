from procs.filter.clustering_filter import Filter
import numpy as np

class ConcatFilter(Filter):

    def __init__(self, arr_title, arr_weight, concat_index, concat_vector):

        self.arr_title = arr_title
        self.arr_weight = arr_weight

        self.concat_index = concat_index
        self.concat_vector = concat_vector

    def do(self, query_title, threshold=0.1, weight_check=True):

        # 예측 array, idx, title, distance
        sim_arr_index, sim_titles, _ = self._search_range_index(query_title, self.arr_title, \
                                                                  self.concat_index, self.concat_vector, threshold)

        find_index = np.where(self.arr_title == query_title)[0].item(0)

        ## 2차 필터링 (중량)
        if weight_check:
            find_weight = self.arr_weight[find_index].item()
            if str(find_weight) == 'nan':
                weight_check = False

        if weight_check:
            sim_arr_weight = self.arr_weight[sim_arr_index]
            sim_arr_index = self._filter_weight(find_weight, sim_arr_weight, sim_arr_index)

        get = self.arr_title[sim_arr_index]

        return get
