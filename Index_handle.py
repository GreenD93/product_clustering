import numpy as np
import faiss

SIMPLE_IVF_COUNT = 4
SIMPLE_METHOD = 'IVF{},Flat'.format(SIMPLE_IVF_COUNT)

class IndexHandler():
    def __init__(self):
        self.index = None
                
    def get_index(self):
        return self.index

    def save_index(self, save_path):
        faiss.write_index(self.index, save_path)
        
    def make_index(self, vector):
        # transform contigousarray vector
        contigousarray_vector = np.ascontiguousarray(vector, dtype=np.float32)
        self._make_index(contigousarray_vector)
        
    def load_index(self, index_path):
        return faiss.read_index(index_path)
        
    def _make_index(self, vector):
        # index
        np_ids = np.arange(len(vector))

        # faiss setting
        dim_count = vector[0].shape[0]
        method, train_count = self._suggest_method_by_size(dim_count)

        index = faiss.index_factory(dim_count, method)
        index.train(vector)

        # reconstruct : index -> vector return () : update 할 때
        index.add_with_ids(vector, np_ids)
        self.index = index
        print(index.is_trained)

    def _suggest_method_by_size(self, n):

        train_count = 0

        # n < 200 -> 'IVF4,Flat'
        if 0 <= n and n < 200:
            train_count = SIMPLE_IVF_COUNT * 40
            method = SIMPLE_METHOD

        # 500 <= n < 1000 -> 'IVF25,Flat'
        elif 200 <= n and n < 1000:
            train_count = 40 * 5 # 200
            method = 'IVF5,Flat'

        # 1000 <= n < 2000 -> 'IVF25,Flat'
        elif 1000 <= n and n < 2000:
            train_count = 40 * 25 # 1000
            method = 'IVF25,Flat'

        # 2000 <= n < 4000 -> 'IVF50,Flat'
        elif 2000 <= n and n < 4000:
            train_count = 40 * 50
            method = 'IVF50,Flat' # 2000

        # 4000 <= n < 8000 -> 'IVF50,Flat'
        elif 4000 <= n and n < 8000:
            train_count = 40 * 100 # 4000
            method = 'IVF100,Flat'

        # 4000 <= n < 40960 -> 'IVF100,Flat'
        elif 8000 <= n and n < 40960:
            train_count = 40 * 200 # 8000
            method = 'IVF200,Flat'

        # 40960 <= n < 163840 -> 'IVF1024,Flat' 임시
        elif 40960 <= n and n < 163840:
            train_count = 40 * 1024 # 40,960
            method = 'IVF1024,Flat'

        # 163840 <= n < 10M -> 'IVF4096,Flat' 임시
        elif 163840 <= n and n < 10000000:
            train_count = 40 * 4096 # 163,840
            method = 'IVF4096,Flat'

        # 10M <= n < ... -> 'IVF16384,Flat'
        else:
            train_count = 40 * 16384 # 638,976
            method = 'IVF16384,Flat'

        # 이것이상이 c5.9xlarge에서도 그닥 성능이 안나옴
        return method, train_count