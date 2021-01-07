import logging
import multiprocessing
import json
from collections import Counter

from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler

import numpy as np
from numpy.linalg import svd

from tqdm import tqdm

def init_logger():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s]%(asctime)s| %(message)s', datefmt='%m-%d %H:%M:%S')
    pass

init_logger()

def log_info(s):

    s = str(s)
    logger = logging.getLogger("filelog")
    logger.info(s)

    pass

def file_to_json(path):
    result = {}
    with open(path, encoding='utf-8') as data_file:
        result = json.load(data_file)
    return result

def json_to_file(path, json_data):
    with open(path, 'w', encoding='utf-8') as data_file:
        json.dump(json_data, data_file, indent=2, ensure_ascii=False)
    pass

#https://bab2min.tistory.com/631
class SIFGenerator():

    def __init__(self, sif_parm=0.001, vector_size=200, avg=True, norm=True):

        self.scaler = MinMaxScaler()

        self.sif_parm = sif_parm
        self.vector_size = vector_size
        self.avg = avg
        self.norm = norm

        self.total_sif_wvs = None

        self.w2v_model = None
        self.word_freq_dict = None
        self.total_word_freq = None

    def load_w2v_model(self, model_path):
        self.w2v_model = Word2Vec.load(model_path)
        pass
    
    def load_word_freq_dict(self, dict_path):

        self.word_freq_dict = file_to_json(dict_path)
        # get total word freq
        self.total_word_freq = self._get_total_word_freq()
        pass
        
    def get_total_wvs(self):
        return self.total_sif_wvs

    def save_total_wvs(self, save_path):
        log_info('=> save sif_arr_wvs to {0}'.format(save_path))
        np.savez(save_path, self.total_sif_wvs)

    def get_w2v_model(self):
        return self.w2v_model

    def save_w2v_model(self, save_path):
        log_info('=> save w2v_model to {0}'.format(save_path))
        self.w2v_model.save(save_path)

    def save_word_freq_dict(self, save_path):
        log_info('=> save word_freq_dict to {0}'.format(save_path))
        json_to_file(save_path, self.word_freq_dict)

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

    def _get_word_weight(self, word):

        try:
            prob_word = self.word_freq_dict[word] / self.total_word_freq
            word_weight = self.sif_parm / (self.sif_parm + prob_word)

        except KeyError:
            word_weight = 0

        return word_weight

    def _make_word_freq_dict(self, arr_title):

        arr_token_list = list(map(lambda x : x.split(' '), arr_title))
        word_freq_dict = Counter([ token for token_list in arr_token_list for token in token_list ])

        self.word_freq_dict = dict(word_freq_dict)

    def _get_total_word_freq(self):
        return sum(self.word_freq_dict.values())

    def _delete_first_singular_value_col(self, total_arr_wvs):

        log_info('=> process singular vetcor decomposition....')
        U, Sigma, Vt = svd(total_arr_wvs, full_matrices=False)

        log_info('=> delete first singular value columns....')
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

    def _make_w2v_model(self, arr_title, n_worker, window_size=5, min_count=5):
        log_info('=> start training w2v model....')
        splitted_token_list = [str_token.split(' ') for str_token in arr_title]
        w2v_model = Word2Vec(splitted_token_list, size=200, window=window_size, min_count=min_count, workers=n_worker)
        log_info('=> end training w2v model....')
        self.w2v_model = w2v_model

    # make feature vector
    def make_sif_wvs_vector(self, query_title):

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

        if self.avg:
            sif_wvs_vector = sif_wvs_vector / denominator

        if self.norm:
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

        # multi-processing
        if n_worker > 1:

            # split arr with n chunks
            arr_process_num = [i for i in range(0, n_worker)]
            arr_chunks = np.array_split(arr_title, n_worker)

            log_info('=> {0} process running...'.format(n_worker))
            pool = multiprocessing.Pool(processes = n_worker)
            results = pool.starmap(self._process_sif_wvs, zip(arr_chunks, arr_process_num))
            pool.close()
            pool.join()

            total_sif_wvs = np.vstack(results)

        else:
            total_sif_wvs = self._process_sif_wvs(arr_title)

        log_info('=> make_sif_wvs process is done....')

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
# [[0.16842484 0.48478295 0.43938006 0.64202566 0.28542784 0.56557855
#   0.47317747 0.33322558 0.68239569 0.73471632 0.55114937 0.73462688
#   0.84373943 0.69627755 0.91051939 0.32563735 0.11714658 0.68905419
#   0.79906503 0.4961091  0.53404992 0.33500758 0.51618716 0.57469531
#   0.57729803 0.52641246 0.57141878 0.43489542 1.         0.15463219
#   0.43976343 0.61487265 0.60051449 0.06916656 0.46234    0.
#   0.31019905 0.56210279 0.30504903 0.91139461 0.3442568  0.1671662
#   0.33107325 0.75796447 0.39080268 0.15377928 0.33692595 0.39431252
#   0.58055712 0.90648451 0.56469326 0.60821947 0.08587211 0.54724426
#   0.50695179 0.3369221  0.35853938 0.70090864 0.46866781 0.4976714
#   0.18987964 0.15521283 0.29582289 0.55932603 0.3758403  0.62136278
#   0.77802565 0.6379045  0.15382176 0.50174201 0.60644083 0.505396
#   0.43673226 0.46951301 0.36089233 0.15408622 0.85700328 0.67429633
#   0.6369357  0.33882911 0.34891795 0.45791658 0.69341886 0.40371798
#   0.67302749 0.43779688 0.54103091 0.74556469 0.85005601 0.25462639
#   0.26442314 0.65093263 0.38783726 0.83316057 0.45326058 0.81966835
#   0.58285181 0.56774854 0.10183781 0.87630289]]
