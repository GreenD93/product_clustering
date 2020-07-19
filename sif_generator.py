import multiprocessing
from collections import Counter

from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler

import numpy as np
from numpy.linalg import svd

from tqdm import tqdm

#https://bab2min.tistory.com/631
class SIFGenerator():
    
    def __init__(self, w2v_model=None, word_freq_dict=None, sif_parm=0.001, vector_size=100):
        
        self.scaler = MinMaxScaler()
        
        self.sif_parm = sif_parm
        self.vector_size = vector_size
        
        self.total_sif_wvs = None
        
        self.w2v_model = w2v_model
        self.word_freq_dict = word_freq_dict
        self.total_word_freq = None

    def get_total_wvs(self):
        return self.total_sif_wvs
        
    def save_total_wvs(self, save_path):
        print('=> save sif_arr_wvs to {0}'.format(save_path))
        np.savez(save_path, self.total_sif_wvs)
        
    def get_w2v_model(self):
        return self.w2v_model
    
    def save_w2v_model(self, save_path):
        print('=> save w2v_model to {0}'.format(save_path))
        self.w2v_model.save(model_path)

    def _normalize(self, vector):

        vector = vector.reshape(-1, 1)
        normalized_vector = self.scaler.fit_transform(vector)
        normalized_vector = normalized_vector.reshape(1, -1)
        
        return normalized_vector
    
    def _get_word_vector(self, word):
        
        try:
            word_vector = self.w2v_model.wv[word]
            
        except KeyError:
            word_vector = np.zeros(self.vector_size)
            
        return word_vector
        
    def _make_word_freq_dict(self, arr_title):
        
        arr_token_list = list(map(lambda x : x.split(' '), arr_title))
        word_freq_dict = Counter([ token for token_list in arr_token_list for token in token_list ])
        
        self.word_freq_dict = dict(word_freq_dict)
    
    def _get_total_word_freq(self):
        return sum(self.word_freq_dict.values())
    
    def _get_word_weight(self, word):
        
        prob_word = self.word_freq_dict[word] / self.total_word_freq
        word_weight = self.sif_parm / (self.sif_parm + prob_word)
        
        return word_weight
    
    def _delete_first_singular_value_col(self, total_arr_wvs):
        
        print('=> process singular vetcor decomposition....')
        U, Sigma, Vt = svd(total_arr_wvs)
        
        print('=> delete first singular value columns....')
        # delete data regarding first sigular value
        U_ = U[:, 1:]
        Sigma_ = Sigma[1:]
        Vt_ = Vt[1:, :]

        # transform singular value matrix for inner production
        num_rank = len(Sigma_)
        Sigma_mat_ = np.diag(Sigma_)
        row, col = U_.shape[1], Vt_.shape[0]
        expand_Sigma_mat_ = np.zeros((row, col))
        expand_Sigma_mat_[:num_rank, :num_rank] = Sigma_mat_

        # restore embedding matrix
        sif_total_arr_wvs = np.dot(np.dot(U_, expand_Sigma_mat_), Vt_)
        
        return sif_total_arr_wvs
    
    def _process_sif_wvs(self, arr_title, process_num=0):
        
        num = 0
        total_count = len(arr_title)
        
        arr_sif_wvs = np.zeros(shape=(total_count, self.vector_size))
        
        for query_title in tqdm(arr_title):
            sif_wvs = self.make_sif_wvs_vector(query_title)
            arr_sif_wvs[num] = sif_wvs
            num += 1
        
        return arr_sif_wvs
    
    def _make_w2v_model(self, arr_title, n_worker, window_size=5, min_count=1):
        print('=> start training w2v model....')
        splited_token_list = [str_token.split(' ') for str_token in arr_title]
        w2v_model = Word2Vec(splited_token_list, size=100, window=window_size, min_count=min_count, workers=n_worker)
        print('=> end training w2v model....')
        self.w2v_model = w2v_model
        
    # make feature vector
    def make_sif_wvs_vector(self, query_title, avg=True, normalize=True):
        
        arr_token = query_title.split(' ')
        sif_wvs_vector = np.zeros(shape=self.vector_size)
        
        for token in arr_token:        
            
            word_weight = self._get_word_weight(token)
            word_vector = self._get_word_vector(token)
            
            sif_wvs_vector += word_weight * word_vector
            
        if len(arr_token):
            denominator = len(arr_token)
        else:
            denominator = 1e-13
            
        if avg:
            sif_wvs_vector = sif_wvs_vector / denominator
        
        if normalize:
            sif_wvs_vector = self._normalize(sif_wvs_vector)
        
        return sif_wvs_vector

    def do(self, arr_title, n_worker, svd=True):
        
        # make w2v model
        if self.w2v_model is None:
            self._make_w2v_model(arr_title, n_worker=n_worker)
        
        # make word freq dict
        if self.word_freq_dict is None:
            self._make_word_freq_dict(arr_title)
        
        # get total word freq
        self.total_word_freq = self._get_total_word_freq()
        
        if n_worker > 1:
            
            # split arr with n chunks
            arr_process_num = [i for i in range(0, n_worker)]
            arr_chunks = np.array_split(arr_title, n_worker)
            
            # multi-processing
            print('=> {0} process running...'.format(n_worker))
            pool = multiprocessing.Pool(processes = n_worker)
            results = pool.starmap(self._process_sif_wvs, zip(arr_chunks, arr_process_num))
            pool.close()
            pool.join()
            
            total_sif_wvs = np.vstack(results)
        
        else:
            total_sif_wvs = self._process_sif_wvs(arr_title)
            
        print('=> make_sif_wvs process is done....')
        
        if svd:
            total_sif_wvs = self._delete_first_singular_value_col(total_sif_wvs)
            
        self.total_sif_wvs = total_sif_wvs
        
        return total_sif_wvs
    
if __name__ == '__main__':
    
    # test training sif generator
    str_token_list = ['아버지 가방에 들어가신다', '이 문장은 예시 입니다.']
    
    sif_generator = SIFGenerator()
    total_sif_wvs = sif_generator.do(str_token_list, n_worker=3, svd=True)
    
    # make title vector
    test_title = '아버지 가방에 들어가신다'
    title_vector = sif_generator.make_sif_wvs_vector(test_title)
    
    print(title_vector)

# output (vector_size):
# [[0.74431957 0.61360116 0.44281559 0.43043511 0.4612954  0.72085784
#   0.68738114 0.45810331 0.47558207 0.71674105 0.56816636 0.48479882
#   0.68711964 0.48050592 0.69165045 0.39880248 0.25367223 0.69421241
#   0.36063353 0.85520228 0.55787935 0.78199095 0.42107628 0.69898046
#   0.84119596 0.34219257 0.76368784 0.66074067 1.         0.43344344
#   0.47772297 0.39621169 0.72176505 0.58528988 0.57658471 0.48885141
#   0.50563908 0.57258996 0.77255789 0.27598233 0.55486711 0.30793861
#   0.91078327 0.61397585 0.48667944 0.52302082 0.41112175 0.40965649
#   0.61981058 0.37589228 0.56541857 0.29830507 0.42407235 0.57635543
#   0.7769593  0.44169613 0.37599896 0.60667691 0.19914987 0.74202171
#   0.         0.54440061 0.31168192 0.51809926 0.45741705 0.48204998
#   0.60039486 0.61200517 0.38666381 0.3809717  0.52692156 0.71834969
#   0.6082092  0.55477031 0.59747782 0.61403237 0.38915718 0.45059436
#   0.91059626 0.72801478 0.97281345 0.46371083 0.10496224 0.89260069
#   0.71558505 0.53834446 0.87105485 0.56732024 0.81332379 0.36481179
#   0.58243603 0.96643991 0.5012223  0.26631688 0.57651223 0.5364224
#   0.69327588 0.20283512 0.41555806 0.57057105]]