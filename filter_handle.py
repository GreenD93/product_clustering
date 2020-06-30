from procs.filter.intersection_filter import IntersectionFilter
from procs.filter.concat_filter import ConcatFilter
from procs.filter.sequential_filter import SequentialFilter

import numpy as np

class FilterHandler:

    def __init__(self, title_array, weight_array, title_index=None, title_vector=None, img_index=None, img_vector=None, concat_index=None, concat_vector=None):
        
        self.title_array = title_array
        self.weight_array = weight_array

        self.title_index = title_index
        self.title_vector = np.ascontiguousarray(title_vector, dtype=np.float32)
        
        self.img_index = img_index
        self.img_vector = np.ascontiguousarray(img_vector, dtype=np.float32)
        
        self.concat_index = concat_index
        self.concat_vector = np.ascontiguousarray(concat_vector, dtype=np.float32)

    def get_filter(self, filter_type):

        if filter_type == 'intersection':
            clustering_filter = IntersectionFilter(self.title_array, self.weight_array, self.title_index, self.title_vector, self.img_index, self.img_vector)

        elif filter_type == 'concat':
            clustering_filter = ConcatFilter(self.title_array, self.weight_array, self.concat_index, self.concat_vector)

        else:
            clustering_filter = SequentialFilter(self.title_array, self.weight_array, self.title_index, self.title_vector, self.img_index, self.img_vector)

        return clustering_filter