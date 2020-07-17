import multiprocessing
from collections import Counter

from sklearn.preprocessing import MinMaxScaler
from scipy import linalg
import numpy as np

from tqdm import tqdm

#https://bab2min.tistory.com/631
class SIFGenerator():
    
    def __init__(self, arr_title, w2v_model, sif_parm=0.001, vector_size=100):
        
        self.arr_title = arr_title
        self.w2v_model = w2v_model
        self.scaler = MinMaxScaler()
        
        self.sif_parm = sif_parm
        self.vector_size = vector_size
        
        self.word_freq_dict = self._make_word_freq_dict()
        self.total_word_freq = self._get_total_word_freq()
        
        self.total_sif_arr_wvs = None

    def get_total_wvs(self):
        return self.total_sif_arr_wvs
        
    def save_total_wvs(self, save_path):
        np.savez(save_path, self.total_sif_arr_wvs)
        
    def _make_word_freq_dict(self):
        
        arr_token_list = list(map(lambda x : x.split(' '), self.arr_title))
        word_freq_dict = Counter([ token for token_list in arr_token_list for token in token_list])
        
        return word_freq_dict
    
    def _get_total_word_freq(self):
        return sum(self.word_freq_dict.values())
    
    def _get_word_weight(self, word):
        
        prob_word = self.word_freq_dict[word] / self.total_word_freq
        word_weight = self.sif_parm / (self.sif_parm + prob_word)
        
        return word_weight
    
    def _get_word_vector(self, word):
        return self.w2v_model.wv[word]
    
    def _normalize(self, vector):

        vector = vector.reshape(-1, 1)
        normalized_vector = self.scaler.fit_transform(vector)
        normalized_vector = normalized_vector.reshape(1, -1)
        
        return normalized_vector

    def _process_sif_wvs(self, arr_title, process_num=0):
        
        num = 0
        total_count = len(arr_title)
        
        arr_sif_wvs = np.zeros(shape=(total_count, self.vector_size))
        
        for query_title in tqdm(arr_title):
            sif_wvs = self.make_sif_wvs_vector(query_title)
            arr_sif_wvs[num] = sif_wvs
            num += 1
        
        return arr_sif_wvs
        
    def _get_first_singular_value_idx(self, arr_vector):
        
        U, S, VT = linalg.svd(arr_vector)
        fisrt_singular_value_idx = np.argmax(S)
        
        return fisrt_singular_value_idx
    
    def _delete_first_singular_value_col(self, total_arr_wvs):
        
        fisrt_singular_value_idx = self._get_first_singular_value_idx(total_arr_wvs)
        total_arr_wvs = np.delete(total_arr_wvs, [fisrt_singular_value_idx], axis=1)
        
        return total_arr_wvs
        
    # make feature vector
    def make_sif_wvs_vector(self, query_title, avg=True, normalize=True):
        
        arr_token = query_title.split(' ')
        sif_wvs_vector = np.zeros(shape=self.vector_size)
        
        for token in arr_token:        
            
            word_weight = self._get_word_weight(token)
            word_vector = self._get_word_vector(token)
            
            sif_wvs_vector += word_weight * word_vector
            
        if avg:
            sif_wvs_vector = sif_wvs_vector / len(arr_token)
        
        if normalize:
            sif_wvs_vector = self._normalize(sif_wvs_vector)
        
        return sif_wvs_vector

    def do(self, n_worker):
        
        if n_worker > 1:
            
            # split arr with n chunks
            arr_process_num = [i for i in range(0, n_worker)]
            arr_chunks = np.array_split(self.arr_title, n_worker)

            # multi-processing
            print('{0} process running...'.format(n_worker))
            pool = multiprocessing.Pool(processes = n_worker)
            results = pool.starmap(self._process_sif_wvs, zip(arr_chunks, arr_process_num))
            pool.close()
            pool.join()

            total_sif_arr_wvs = np.vstack(results)
        
        else:
            total_sif_arr_wvs = self._process_sif_wvs(self.arr_title)
            
        total_sif_wvs = np.ascontiguousarray(total_sif_arr_wvs).astype('float32')
        total_sif_arr_wvs = self._delete_first_singular_value_col(total_sif_arr_wvs)
        self.total_sif_arr_wvs = total_sif_arr_wvs
        
        return total_sif_arr_wvs