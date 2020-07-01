from procs.filter.intersection_filter import IntersectionFilter
from procs.filter.concat_filter import ConcatFilter
from procs.filter.sequential_filter import SequentialFilter

import numpy as np

class FilterHandler:

    def __init__(self, arr_title, arr_weight, title_index=None, query_vector=None, img_index=None, img_vector=None, concat_index=None, concat_vector=None):
        
        self.arr_title = arr_title
        self.arr_weight = arr_weight

        self.title_index = title_index
        self.query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)
        
        self.img_index = img_index
        self.img_vector = np.ascontiguousarray(img_vector, dtype=np.float32)
        
        self.concat_index = concat_index
        self.concat_vector = np.ascontiguousarray(concat_vector, dtype=np.float32)

    def get_filter(self, filter_type):

        if filter_type == 'intersection':
            clustering_filter = IntersectionFilter(self.arr_title, self.arr_weight, self.title_index, self.query_vector, self.img_index, self.img_vector)

        elif filter_type == 'concat':
            clustering_filter = ConcatFilter(self.arr_title, self.arr_weight, self.concat_index, self.concat_vector)

        else:
            clustering_filter = SequentialFilter(self.arr_title, self.arr_weight, self.title_index, self.query_vector, self.img_index, self.img_vector)

        return clustering_filter