from procs.filter.clustering_filter import Filter
import numpy as np

class IntersectionFilter(Filter):

    def __init__(self, title_array, weight_array, title_index, title_vector, img_index, img_vector):
        self.title_array = title_array
        self.weight_array = weight_array
        
        self.title_index = title_index
        self.title_vector = title_vector
        
        self.img_index = img_index
        self.img_vector = img_vector

    def do(self, find_title, img_thrshold=0.3, title_threshold=0.3, weight_check=True):

        # 예측 array
        sim_index_array, get = self._find_intersection(find_title, img_thrshold, title_threshold)

        find_index = np.where(self.title_array == find_title)[0].item(0)

        ## 3차 필터링 (중량)
        if weight_check:
            find_weight = self.weight_array[find_index].item()
            if str(find_weight) == 'nan':
                weight_check = False

        if weight_check:
            sim_weight_array = self.weight_array[sim_index_array]
            sim_index_array = self._weight_filter(find_weight, sim_weight_array, sim_index_array)

        get = self.title_array[sim_index_array]

        return get

    def _find_intersection(self, find_title, img_thrshold, title_threshold):

        # 1차 img, idx, title, distance
        img_sim_idx, img_sim_titles, _ = self._search_range_index(find_title, self.title_array,
                                                                  self.img_index, self.img_vector, img_thrshold)

        # 2차 title, idx, title, distance
        title_sim_idx, title_sim_titles, _ = self._search_range_index(find_title, self.title_array,
                                                                      self.title_index, self.title_vector, title_threshold)

        intersect_idx = np.intersect1d(img_sim_idx, title_sim_idx)

        get = self.title_array[intersect_idx]
        
        return intersect_idx, get