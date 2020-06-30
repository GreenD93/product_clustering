from procs.filter.clustering_filter import Filter
import numpy as np

class ConcatFilter(Filter):

    def __init__(self, title_array, weight_array, concat_index, concat_vector):
        
        self.title_array = title_array
        self.weight_array = weight_array
        
        self.concat_index = concat_index
        self.concat_vector = concat_vector

    def do(self, find_title, threshold=0.1, weight_check=True):

        # 예측 array, idx, title, distance
        sim_index_array, sim_titles, _ = self._search_range_index(find_title, self.title_array, \
                                                                  self.concat_index, self.concat_vector, threshold)

        find_index = np.where(self.title_array == find_title)[0].item(0)

        ## 2차 필터링 (중량)
        if weight_check:
            find_weight = self.weight_array[find_index].item()
            if str(find_weight) == 'nan':
                weight_check = False

        if weight_check:
            sim_weight_array = self.weight_array[sim_index_array]
            sim_index_array = self._weight_filter(find_weight, sim_weight_array, sim_index_array)

        get = self.title_array[sim_index_array]

        return get