from __future__ import print_function
import pandas as pd
import logging
import gzip
import gensim


class TrainWord2Vec:
    '''
    this class run different configurations.
    input:
        1. configuration properties to check
    output:
        1. hyper-parameters tuning using grid search
        2. call to inner class (train) to check every configuration
    '''

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

        lod_dir = '/Users/sguyelad/PycharmProjects/reviews_classifier/log/train_word2vec/'
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
        pass

    # iterate over all configuration, build model for each
    def create_review_data_set(self):
        logging.info('create reviews data-set')

        k = 1000000

        df = pd.read_csv(self.input_data_file)
        k = min(k, df.shape[0])
        print('k=' + str(k))
        review_series = df['REVIEWS'][:k]
        review_series_list = list()
        for i in range(0, k):
            try:
                review_series[i] = review_series[i].replace("{", "")
                review_series[i] = review_series[i].replace("}", "")
                list_str = review_series[i].split('=')
                review_series[i] = ''.join(list_str[1:])
                review_series[i] = review_series[i][1:-1]
                review_series_list.append(review_series[i])
            except:
                print('except ' + str(review_series[i]))
        # print(review_series)
        print(len(review_series_list))
        self.run_word2vec(review_series_list)
        return

    def run_word2vec(self, review_series):
        """
        run word2vec on vertical reviews
        """
        logging.info('run word2vec')
        documents = list()
        for i, line in enumerate(review_series):
            # print('')
            # print(line)
            line = gensim.utils.simple_preprocess(line)
            # print(line)
            documents.append(line)
        window = 9
        dim = 300

        print('')
        print('start training word2vec model')
        # build vocabulary and train model
        model = gensim.models.Word2Vec(
            documents,
            size=dim,
            window=window,
            min_count=2,
            workers=10)
        model.train(documents,
                    total_examples=model.corpus_count,
                    # total_examples=len(documents),
                    epochs=100)

        word2vec_dir = self.output_results_folder + str(vertical_type) + '/'
        word2vec_path = word2vec_dir + 'd_' + str(dim) + '_k_' + str(len(documents)) + '_w_' + str(window)
        import os
        if not os.path.exists(word2vec_dir):
            os.makedirs(word2vec_dir)

        model.save(word2vec_path)

        print('save word2vec model' + str(word2vec_path))
        print(model.wv.most_similar(positive='shirt', topn=6))
        return


def main(input_data_file, vertical_type, output_results_folder, word2vec_parameters_dict):

    train_obj = TrainWord2Vec(input_data_file, vertical_type, output_results_folder, word2vec_parameters_dict)

    train_obj.init_debug_log()                  # init log file
    train_obj.check_input()
    train_obj.create_review_data_set()          # create review data set
    # train_obj.run_word2vec()


if __name__ == '__main__':

    # input file name
    vertical_type = 'fashion'                    # 'fashion'/'motors'
    output_results_folder = '../data/word2vec_pretrained/'
    word2vec_parameters_dict = {
        'embedding_size': 300,
    }

    if vertical_type == 'fashion':
        input_data_file = '../data/word2vec_input_data/fashion/amazon-crawl-output.csv'

    elif vertical_type == 'motors':
        input_data_file = '../data/clean/clean_data_multi_new_motors.csv'

    else:
        raise ValueError('unknown vertical name')

    main(input_data_file, vertical_type, output_results_folder, word2vec_parameters_dict)