from __future__ import print_function
import os
import logging
import pandas as pd
import numpy as np
import gzip
import gensim


class TrainWord2Vec:
    """
    This class has two purpose:
    1. create clean data-set, will use next as input to word2vec algorithm
    2. create word vector (word2vec)

    purpose (1/2) defined using flags
        create_data_set
        create_word_embedding

    :argument
        embedding_size: word vector dimensions
        window: pair selection into word2vec network
        epoch: number of epochs to train

    :return
    1. save clean data using pickle in data/word2vec_input_data/
    2. save word2vec model (using gensim library) in data/wor2vec_pretrained/

    """

    def __init__(self, input_data_file, vertical_type, output_results_folder, word2vec_parameters_dict):

        # file arguments
        self.input_data_file = input_data_file                      # csv input file
        self.vertical_type = vertical_type                          # 'fashion'/'motors'
        self.output_results_folder = output_results_folder          # output folder to store word2vec parameters
        self.word2vec_parameters_dict = word2vec_parameters_dict    #
        self.verbose_flag = True

        from time import gmtime, strftime
        self.cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

        # define data frame needed for analyzing data
        self.df = pd.DataFrame()
        self.train_df = pd.DataFrame()
        self.test_df = pd.DataFrame()

    # init log file
    def init_debug_log(self):
        import logging

        lod_dir = './log/train_word2vec/'
        log_file_name = str(self.cur_time) + '.log'
        import os
        if not os.path.exists(lod_dir):
            os.makedirs(lod_dir)

        logging.basicConfig(filename=lod_dir + log_file_name,
                            format='%(asctime)s, %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

        # print result in addition to log file
        if self.verbose_flag:
            stderrLogger = logging.StreamHandler()
            stderrLogger.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
            logging.getLogger().addHandler(stderrLogger)

        logging.info("")
        return

    def check_input(self):
        if self.word2vec_parameters_dict['embedding_size'] not in [50, 100, 200, 300]:
            raise ValueError('unknown word2vec embedding size: ' + str(self.word2vec_parameters_dict['embedding_size']))
        pass

    # iterate over all configuration, build model for each
    def create_review_data_set(self):
        """
        1. for each vertical
        2. load all relevant reviews
        3. create list of list
        """
        vertical_list = ['fashion', 'motors']
        for vertical in vertical_list:
            review_series = None
            if vertical == 'fashion':

                input_1 = '../data/word2vec_input_data/fashion/amazon-crawl-output.csv'
                df_1 = pd.read_csv(input_1)
                input_2 = '../data/word2vec_input_data/fashion/amazon-crawl-output-2.csv'
                df_2 = pd.read_csv(input_2)

                review_series_1 = df_1['REVIEWS']
                review_series_2 = df_2['REVIEWS'][:343789]

                print('review_series_1 shape: ' + str(review_series_1.shape))
                print('review_series_2 shape: ' + str(review_series_2.shape))
                review_series = review_series_1.append(review_series_2, ignore_index=True)
                review_series = review_series
                print('review_series total shape: ' + str(review_series.shape))

            elif vertical == 'motors':

                input_3 = '../data/word2vec_input_data/fashion/amazon-crawl-output-3.csv'
                df_3 = pd.read_csv(input_3)
                input_2 = '../data/word2vec_input_data/fashion/amazon-crawl-output-2.csv'
                df_2 = pd.read_csv(input_2)

                review_series_3 = df_3['REVIEWS']
                review_series_2 = df_2['REVIEWS'][343790:]

                print('review_series_3 shape: ' + str(review_series_3.shape))
                print('review_series_2 shape: ' + str(review_series_2.shape))
                review_series = review_series_3.append(review_series_2, ignore_index=True)
                review_series = review_series
                print('review_series total shape: ' + str(review_series.shape))
            else:
                raise ValueError('unknown vertical')

            review_series_list = self._prepare_date_to_word_2_vec(review_series)

            json_dir_name = '../data/word2vec_input_data/' + str(vertical) + '/'
            json_f_name = json_dir_name + str(len(review_series_list)) + '.txt'

            import os
            if not os.path.exists(json_dir_name):
                os.makedirs(json_dir_name)

            import pickle
            with open(json_f_name, "wb") as fp:  # Pickling
                pickle.dump(review_series_list, fp)
            print('save json to file: ' + str(json_f_name))

    def _prepare_date_to_word_2_vec(self, review_series):

        review_series_list = list()
        for i, v in review_series.items():

            if i % 1000 == 0:
                print('parse review number: ' + str(i))
            try:
                review_series[i] = review_series[i].replace("{", "")
                review_series[i] = review_series[i].replace("}", "")
                list_str = review_series[i].split('=')
                review_series[i] = ''.join(list_str[1:])
                review_series[i] = review_series[i][1:-1]
                review_series_list.append(review_series[i])     # .decode('utf-8'))
            except:
                print('except review: ' + str(review_series[i]))

        print('valid reviews: ' + str(len(review_series_list)))
        return review_series_list

    def create_word2vec(self):
        review_series_list = self._load_data()
        self.run_word2vec(review_series_list)

    def _load_data(self):
        """ load list of str (review) using pickle"""
        import pickle
        with open(self.input_data_file, "rb") as fp:
            review_series_list = pickle.load(fp)
        return review_series_list

    def run_word2vec(self, review_series):
        """
        run word2vec on vertical reviews
        """

        logging.info('run word2vec for vertical: ' + str(vertical_type) + ', amount: ' + str(len(review_series)))
        documents = list()
        for i, line in enumerate(review_series):
            if i % 1000 == 0:
                print('pre-process line: ' + str(i))

            line = gensim.utils.simple_preprocess(line)
            documents.append(line)

        # to calculate histogram of review length please un-comment the function below
        # self._build_histogram_review_length(documents)

        print('')
        print('start training word2vec model')

        # build vocabulary and train model
        model = gensim.models.Word2Vec(
            documents,
            size=self.word2vec_parameters_dict['embedding_size'],
            window=self.word2vec_parameters_dict['window'],
            min_count=1,
            workers=10)

        model.train(documents,
                    total_examples=model.corpus_count,
                    # total_examples=len(documents),
                    epochs=self.word2vec_parameters_dict['epoch'])

        word2vec_dir = self.output_results_folder + str(vertical_type) + '/'
        word2vec_path = word2vec_dir + \
                        'd_' + str(self.word2vec_parameters_dict['embedding_size']) + \
                        '_k_' + str(len(documents)) + \
                        '_w_' + str(self.word2vec_parameters_dict['window']) + \
                        '_e_' + str(self.word2vec_parameters_dict['epoch']) + \
                        '_v_' + str(self.vertical_type)
        import os
        if not os.path.exists(word2vec_dir):
            os.makedirs(word2vec_dir)

        model.save(word2vec_path)

        print('save word2vec model' + str(word2vec_path))
        print(model.wv.most_similar(positive='the', topn=6))
        print(model.wv.most_similar(positive='shirt', topn=6))
        return

    def _build_histogram_review_length(self, documents):

        import numpy as np
        import matplotlib.pyplot as plt

        hist_list = [len(x) for x in documents]
        a = np.array(hist_list)
        print('0.05: ' + str(np.percentile(a, 5)))
        print('q1: ' + str(np.percentile(a, 25)))
        print('median: ' + str(np.percentile(a, 50)))
        print('q3: ' + str(np.percentile(a, 75)))
        print('0.95: ' + str(np.percentile(a, 95)))

        for i, x in enumerate(hist_list):
            if x > 500:
                hist_list[i] = 500

        plt.hist(hist_list, bins=100)
        plt.title("Review length histogram")
        plt.xlabel("Review length")
        plt.ylabel("Amount")
        # plt.show()
        word2vec_dir = self.output_results_folder + str(vertical_type) + '/'
        hist_path = word2vec_dir + str(len(documents)) + '_histogram.png'
        plt.savefig(hist_path)


    def _load_glove(self):
        """
        load pre-trained glove with dim 50/100/200/300
        transform to word2vec format
        """

        from gensim.scripts.glove2word2vec import glove2word2vec

        GLOVE_DIR = '../data/golve_pretrained/glove.6B'

        glove_suffix_name = 'glove.6B.' + str(self.word2vec_parameters_dict['embedding_size']) + 'd.txt'
        word2vec_suffix_name = 'glove.6B.' + str(self.word2vec_parameters_dict['embedding_size']) + 'd.txt.word2vec'

        glove_input_file = os.path.join(GLOVE_DIR, glove_suffix_name)
        word2vec_output_file = os.path.join(GLOVE_DIR, word2vec_suffix_name)

        # transform glove to word2vec
        glove2word2vec(glove_input_file, word2vec_output_file)
        return word2vec_output_file


def main(input_data_file, vertical_type, output_results_folder, word2vec_parameters_dict, create_data_set, create_word_embedding):

    train_obj = TrainWord2Vec(input_data_file, vertical_type, output_results_folder, word2vec_parameters_dict)

    train_obj.init_debug_log()                  # init log file
    train_obj.check_input()

    if create_data_set and create_word_embedding:
        raise ValueError('the two value cannot be true')
    if create_data_set:
        train_obj.create_review_data_set()          # create review data set

    if create_word_embedding:
        train_obj.create_word2vec()
    # train_obj.run_word2vec()


if __name__ == '__main__':

    # input file name
    vertical_type = 'fashion'                    # 'fashion'/'motors'
    # input_data_file = '../data/word2vec_input_data/motors/712904.txt'     # 1341062
    input_data_file = '../data/word2vec_input_data/fashion/1341062.txt'  # 1341062

    output_results_folder = '../data/word2vec_pretrained/'
    word2vec_parameters_dict = {
        'embedding_size': 100,
        'window': 10,
        'epoch': 60
    }
    create_data_set = False
    create_word_embedding = True

    main(input_data_file, vertical_type, output_results_folder, word2vec_parameters_dict, create_data_set, create_word_embedding)