from procs.filter.clustering_filter import Filter
import numpy as np

class IntersectionFilter(Filter):

    def __init__(self, arr_title, arr_weight, title_index, arr_title_vector, img_index, arr_img_vector):
        self.arr_title = arr_title
        self.arr_weight = arr_weight
        
        self.title_index = title_index
        self.arr_title_vector = arr_title_vector
        
        self.img_index = img_index
        self.arr_img_vector = arr_img_vector

    def do(self, query_title, img_thrshold=0.3, title_threshold=0.3, weight_check=True):

        # 예측 array
        arr_sim_idx, _ = self._find_intersection(query_title, img_thrshold, title_threshold)

        find_idx = np.where(self.arr_title == query_title)[0].item(0)

        ## 3차 필터링 (중량)
        if weight_check:
            find_weight = self.arr_weight[find_idx].item()
            if str(find_weight) == 'nan':
                weight_check = False

        if weight_check:
            sim_arr_weight = self.arr_weight[arr_sim_idx]
            arr_sim_idx = self._filter_weight(find_weight, sim_arr_weight, arr_sim_idx)

        arr_sim_title = self.arr_title[arr_sim_idx]
        arr_sim_title = arr_sim_title.flatten()

        return arr_sim_title

    def _find_intersection(self, query_title, img_thrshold, title_threshold):

        # 1차 img, idx, title, distance
        img_sim_idx, img_sim_titles, _ = self._search_range_idx(query_title, self.arr_title,
                                                                  self.img_index, self.arr_img_vector, img_thrshold)

        # 2차 title, idx, title, distance
        title_sim_idx, title_sim_titles, _ = self._search_range_idx(query_title, self.arr_title,
                                                                      self.title_index, self.arr_title_vector, title_threshold)

        arr_sim_idx = np.intersect1d(img_sim_idx, title_sim_idx)

        arr_sim_title = self.arr_title[arr_sim_idx]
        
        return arr_sim_idx, arr_sim_title