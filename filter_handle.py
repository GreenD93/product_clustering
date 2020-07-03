from procs.filter.intersection_filter import IntersectionFilter
from procs.filter.concat_filter import ConcatFilter
from procs.filter.sequential_filter import SequentialFilter

import numpy as np

class FilterHandler:

    def __init__(self, arr_title, arr_weight, title_index=None, arr_title_vector=None, img_index=None, arr_img_vector=None, concat_index=None, arr_concat_vector=None):

        self.arr_title = arr_title
        self.arr_weight = arr_weight

        self.title_index = title_index
        self.arr_title_vector = np.ascontiguousarray(arr_title_vector, dtype=np.float32)

        self.img_index = img_index
        self.arr_img_vector = np.ascontiguousarray(arr_img_vector, dtype=np.float32)

        self.concat_index = concat_index
        self.arr_concat_vector = np.ascontiguousarray(arr_concat_vector, dtype=np.float32)

    def get_filter(self, filter_type):

        if filter_type == 'intersection':
            clustering_filter = IntersectionFilter(self.arr_title, self.arr_weight, self.title_index, self.arr_title_vector, self.img_index, self.arr_img_vector)

        elif filter_type == 'concat':
            clustering_filter = ConcatFilter(self.arr_title, self.arr_weight, self.concat_index, self.arr_concat_vector)

        else:
            clustering_filter = SequentialFilter(self.arr_title, self.arr_weight, self.title_index, self.arr_title_vector, self.img_index, self.arr_img_vector)

        return clustering_filter
