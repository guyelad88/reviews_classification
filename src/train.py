from __future__ import print_function
import pandas as pd
import logging
import numpy as np


class TrainModel:
    '''
    this class target:
        1. prepare model data (load and split)
        2. determine lstm parameters
        3. call to lstm class
    input: (mainly)
        1. model parameters
        2. data file path
    '''

    def __init__(self, input_data_file, vertical_type, output_results_folder, tensor_board_dir, lstm_parameters_dict,
                 df_configuration_dict, multi_class_configuration_dict, attention_configuration_dict,
                 cv_configuration, test_size, embedding_pre_trained, logging=None):

        # file arguments
        self.input_data_file = input_data_file              # csv input file
        self.vertical_type = vertical_type                  # 'fashion'/'motors'
        self.output_results_folder = output_results_folder  # output folder to store results
        self.tensor_board_dir = tensor_board_dir

        self.test_size = test_size  # 0.2
        self.embedding_pre_trained = embedding_pre_trained
        self.lstm_parameters_dict = lstm_parameters_dict
        self.df_configuration_dict = df_configuration_dict  # df configuration for pre processing
        self.cv_configuration = cv_configuration            # cross-validation configuration
        self.multi_class_configuration_dict = multi_class_configuration_dict       # multi-class bool
        self.attention_configuration_dict = attention_configuration_dict           # attention keys

        self.verbose_flag = True
        self.logging = logging

        from time import gmtime, strftime
        self.cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

        # define data frame needed for analyzing data
        self.df = pd.DataFrame()
        self.train_df = pd.DataFrame()
        self.test_df = pd.DataFrame()

        self.roc_result_dict_all_folds = dict()         # contain all stats for all folds and epochs
        self.roc_max_result_auc_epoch_dict = dict()     # mapping of max auc -> epoch

        self.plt_list_colors = [
            'darkkhaki',
            'blue',
            'red',
            'green',
            'orange',
            'pink',
            'darkmagenta',
            'darkolivegreen',
            'darkorange',
            'darkorchid',
            'darksalmon',
            'darkseagreen'
        ]

    # init log file
    def init_debug_log(self):
        import logging

        lod_file_name = '../log/' + 'train_' + str(self.cur_time) + '.log'

        logging.basicConfig(filename=lod_file_name,
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

    ########################################## load data and prepare it ##########################################

    # load csv into df
    def load_clean_csv_results(self):

        self.df = pd.read_csv(self.input_data_file)

        return

    # df pre-processing
    # change target column to 1-0 (target columns is defined from wrapper_train)
    # after current function data is ready to split and to build lstm network
    def df_pre_processing(self):

        self.logging.info('')
        self.logging.info('pre-processing df to fit models')
        self.logging.info('target column name: ' + str(self.df_configuration_dict['y_column']))
        self.logging.info('positive group value: ' + str(self.df_configuration_dict['y_positive']))

        # one vs. all method
        # change positive group to 1, otherwise to 0
        self.df[self.df_configuration_dict['y_column']] = np.where(
            self.df[self.df_configuration_dict['y_column']] == self.df_configuration_dict['y_positive'], 1, 0)

        self.logging.info('')
        self.logging.info('Tagging analysis (Y)')
        y_group = self.df.groupby([self.df_configuration_dict['y_column']])
        for group_type, tag_group in y_group:
            group_percentage = float(tag_group.shape[0]) / float(self.df.shape[0])
            logging.info('Tag: ' + str(group_type) + ', amount: ' + str(tag_group.shape[0]) + ', percentage: ' + str(
                round(group_percentage, 3)))

        self.logging.info('')
        return

    ########################################## run lstm for all folds ##########################################

    # navigate by whether using cv or test-train split
    # a. split data
    # b. run lstm model
    def run_experiment(self):

        # cross validation mode
        if self.cv_configuration['use_cv_bool']:
            self._lstm_model_cv()
            self._calculate_average_auc()

        # split into test-train
        elif not self.cv_configuration['use_cv_bool']:
            self._lstm_model_regular()
            self._calculate_average_auc()
        else:
            raise('unknown split method')

        return

    # lstm using cross validation
    def _lstm_model_cv(self):

        from sklearn.model_selection import StratifiedKFold

        stratified_kfold = StratifiedKFold(n_splits=self.cv_configuration['num_fold'], shuffle=True)
        fold_counter = 1
        for train, test in stratified_kfold.split(
                self.df[self.df_configuration_dict['x_column']],
                self.df[self.df_configuration_dict['y_column']]
        ):
            logging.info('')
            logging.info('split CV: ' + str(fold_counter))
            logging.info('')
            logging.info('test indices: ' + str(test[:10]))
            logging.info('train size=' + str(self.df[self.df_configuration_dict['y_column']][train].shape[0]) +
                         ', ratio_good=' + str(
                round(self.df[self.df_configuration_dict['y_column']][train].mean(), 3)) +
                         ', majority=' + str(
                1 - round(self.df[self.df_configuration_dict['y_column']][train].mean(), 3)))

            logging.info('test size=' + str(self.df[self.df_configuration_dict['y_column']][test].shape[0]) +
                         ', ratio_good=' + str(
                round(self.df[self.df_configuration_dict['y_column']][test].mean(), 3)) +
                         ', majority=' + str(
                1 - round(self.df[self.df_configuration_dict['y_column']][test].mean(), 3)))

            x_train = self.df[self.df_configuration_dict['x_column']][train]
            if self.multi_class_configuration_dict['multi_class_bool']:
                y_train_df = \
                self.df[self.multi_class_configuration_dict['multi_class_label']].iloc[train]
                y_train = []
                for class_name in self.multi_class_configuration_dict['multi_class_label']:
                    y_train.append(y_train_df[class_name])

            else:
                y_train = self.df[self.df_configuration_dict['y_column']][train]

            x_test = self.df[self.df_configuration_dict['x_column']][test]
            if self.multi_class_configuration_dict['multi_class_bool']:
                y_test_df = self.df[self.multi_class_configuration_dict['multi_class_label']].iloc[test]
                y_test = []
                for class_name in self.multi_class_configuration_dict['multi_class_label']:
                    y_test.append(y_test_df[class_name])
            else:
                y_test = self.df[self.df_configuration_dict['y_column']][test]

            train_reason = self.df['Reason'][train]
            test_reason = self.df['Reason'][test]

            logging.info('')
            logging.info('')
            logging.info('start lstm keras model, fold #' + str(fold_counter) + '/' +
                         str(self.cv_configuration['num_fold']))

            # run lstm model
            self._run_model_lstm_keras(x_train, y_train, x_test, y_test, train_reason, test_reason, fold_counter)

            fold_counter += 1
        return

    # lstm using regular test-train split
    def _lstm_model_regular(self):

        from sklearn.model_selection import train_test_split

        fold_counter = 1    # only one fold

        self.train_df, self.test_df = train_test_split(
            self.df,
            stratify=self.df[self.df_configuration_dict['y_column']],  # 'review_tag
            test_size=self.test_size
        )

        logging.info('')
        logging.info('split to test-train')
        logging.info('data size=' + str(self.df.shape[0]))
        logging.info('train size=' + str(self.train_df.shape[0]) + ', fraction=' + str(1 - self.test_size)
                     + ', ratio_good=' + str(round(self.train_df[self.df_configuration_dict['y_column']].mean(), 3))
                     + ', majority=' + str(
            1 - round(self.train_df[self.df_configuration_dict['y_column']].mean(), 3)))

        logging.info('test size=' + str(self.test_df.shape[0]) + ', fraction=' + str(self.test_size)
                     + ', ratio_good=' + str(round(self.test_df[self.df_configuration_dict['y_column']].mean(), 3))
                     + ', majority=' + str(
            1 - round(self.test_df[self.df_configuration_dict['y_column']].mean(), 3)))

        x_train = self.train_df[self.df_configuration_dict['x_column']]
        y_train = self.train_df[self.df_configuration_dict['y_column']]
        x_test = self.test_df[self.df_configuration_dict['x_column']]
        y_test = self.test_df[self.df_configuration_dict['y_column']]

        logging.info('')
        logging.info('')
        logging.info('start lstm keras model')

        # run lstm model
        self._run_model_lstm_keras(x_train, y_train, x_test, y_test, fold_counter)
        return

    # run lstm model with embedding using Keras platform
    def _run_model_lstm_keras(self, x_train, y_train, x_test, y_test, train_reason, test_reason, fold_counter):

        from classifier_lstm import PredictDescriptionModelLSTM

        logging.info('')
        logging.info('Run LSTM on Keras')

        lstm_obj = PredictDescriptionModelLSTM(

            '',             # self.file_directory,
            logging,
            self.cur_time,

            x_train,        # self.train_df[self.df_configuration_dict['x_column']],  # ['Review'],
            y_train,        # self.train_df[self.df_configuration_dict['y_column']],  # ['review_tag'],
            x_test,         # self.test_df[self.df_configuration_dict['x_column']],   # e.g ['Review'],
            y_test,         # self.test_df[self.df_configuration_dict['y_column']],   # e.g. ['review_tag'],
            train_reason,    # self.train_df['Reason']
            test_reason,    # self.test_df['Reason']
            fold_counter,   # fold counter for building dir

            self.lstm_parameters_dict,              # self.lstm_parameters
            self.df_configuration_dict,             # df column configuration
            self.multi_class_configuration_dict,    # configuration for using multi-class classification
            self.attention_configuration_dict,      # configuration fot attention
            self.tensor_board_dir,
            self.embedding_pre_trained,     # Boolean value
            self.vertical_type              # vertical fashion/motors
        )

        roc_statistic_results_dict, global_max_auc_epoch_dict = lstm_obj.run_experiment()

        import copy
        cur_dict = copy.deepcopy(roc_statistic_results_dict)
        cur_max_dict = copy.deepcopy(global_max_auc_epoch_dict)     # map max auc -> epoch

        self.roc_result_dict_all_folds[fold_counter] = cur_dict
        self.roc_max_result_auc_epoch_dict[fold_counter] = cur_max_dict

        logging.info('finish LSTM model')

        return

    ########################################## analyze lstm results ##########################################

    # after finished to build models and to calculate auc results to all folds
    # calculate avg auc
    # build ROC for all the plots together
    def _calculate_average_auc(self):

        logging.info('')
        logging.info('*********************************** ROC global dict ****************************************')
        # logging.info(self.roc_result_dict_all_folds)
        max_auc_val = 0
        for cur_epoch in xrange(self.lstm_parameters_dict['num_epoch']):
            # cur_epoch + 1
            cur_auc = self._plot_multi_roc_curve(cur_epoch + 1, 'regular')
            if cur_auc > max_auc_val:
                max_auc_val = cur_auc

        self._plot_multi_roc_curve('best', 'max')
        # save statistic using pickle into
        self._save_roc_statistic_to_pickle_file()

        # should be last (change file suffix directory)
        self._change_dir_name(max_auc_val)

        return

    # change dir name to prefix with max auc
    def _change_dir_name(self, max_auc):

        import os

        file_suffix = self._get_file_suffix()

        #  new directory name with AUC score
        new_dir = '../results/ROC/' + \
                  str(self.vertical_type) + '_' + str(self.df_configuration_dict['y_positive_name']) + '/' + \
                  str(round(max_auc, 3)) + '_' + file_suffix + '/'

        os.rename('../results/ROC/' +
                  str(self.vertical_type) + '_' + str(self.df_configuration_dict['y_positive_name']) + '/' +
                  file_suffix + '/',
                  new_dir)

        self.logging.info('')
        self.logging.info('change dir name: ' + new_dir)

        return

    # TODO save using file_suffix generic
    def _save_roc_statistic_to_pickle_file(self):

        file_suffix = self._get_file_suffix()

        plot_dir = '../results/ROC/' + \
                   str(self.vertical_type) + '_' + str(self.df_configuration_dict['y_positive_name']) + '/' \
                   + str(file_suffix) + '/'

        import pickle

        with open(plot_dir + 'ROC_statistic_pickle.txt', 'w') as file:
            file.write(pickle.dumps(self.roc_result_dict_all_folds))

        with open(plot_dir + 'max_ROC_statistic_pickle.txt', 'w') as file:
            file.write(pickle.dumps(self.roc_max_result_auc_epoch_dict))

        return

    # plot multi auc plot for a specific epoch
    def _plot_multi_roc_curve(self, epoch, type):

        logging.info('*****************************  multi auc plot for epoch  *************************************')
        logging.info('current epoch: ' + str(epoch))

        import matplotlib.pyplot as plt
        plt.figure()

        lw = 2

        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.01])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        num_auc = 0
        total_auc = 0.0
        if type == 'max':
            for fold_num, auc_epoch_dict in self.roc_max_result_auc_epoch_dict.iteritems():     # fold -> auc, epoch
                max_epoch_dict = self.roc_result_dict_all_folds[fold_num][auc_epoch_dict['epoch']]
                num_auc += 1
                total_auc += max_epoch_dict['auc']
                plt.plot(
                    max_epoch_dict['fpr'],
                    max_epoch_dict['tpr'],
                    color=self.plt_list_colors[fold_num],
                    lw=lw,
                    label='Fold: ' + str(fold_num) + ', epoch:' + str(auc_epoch_dict['epoch']) + ' - (AUC = %0.3f)' % max_epoch_dict['auc'])
        else:
            for fold_num, epoch_dict in self.roc_result_dict_all_folds.iteritems():
                if epoch in epoch_dict:
                    num_auc += 1
                    total_auc += epoch_dict[epoch]['auc']
                    plt.plot(
                        epoch_dict[epoch]['fpr'],
                        epoch_dict[epoch]['tpr'],
                        color=self.plt_list_colors[fold_num],
                        lw=lw,
                        label='Fold: ' + str(fold_num) + ' - (AUC = %0.3f)' % epoch_dict[epoch]['auc'])

        if num_auc > 0:
            mean_auc = float(total_auc) / float(num_auc)
        else:       # epoch without any result in all folds (due to early stopping)
            plt.close()
            return

        plt.title('ROC - epoch number: ' + str(epoch) + ', ' + str(round(mean_auc, 3)))
        plt.legend(loc="lower right")

        file_suffix = self._get_file_suffix()

        import os
        plot_dir = '../results/ROC/' + \
                   str(self.vertical_type) + '_' + str(self.df_configuration_dict['y_positive_name']) + '/' \
                   + str(file_suffix) + '/' + 'auc_cv' + '/'

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        plot_path = plot_dir \
                    + str(round(mean_auc, 3)) + \
                    '_epoch=' + str(epoch)

        plt.savefig(plot_path + '.png')
        plt.close()
        logging.info('save ROC plot: ' + str(plot_path))

        return mean_auc

    # TODO merge with same function from classifier_lstm.py
    # TODO maybe, calculate this and pass it inside the inner class
    def _get_file_suffix(self):

        file_suffix = 'sen_len=' + str(self.lstm_parameters_dict['maxlen']) + \
                      '_batch=' + str(self.lstm_parameters_dict['batch_size']) + \
                      '_optimizer=' + str(self.lstm_parameters_dict['optimizer']) + \
                      '_embedding=' + str(self.lstm_parameters_dict['embedding_size']) + \
                      '_lstm_hidden=' + str(self.lstm_parameters_dict['lstm_hidden_layer']) + \
                      '_pre_trained=' + str(self.embedding_pre_trained) + \
                      '_epoch=' + str(self.lstm_parameters_dict['num_epoch']) + \
                      '_dropout=' + str(self.lstm_parameters_dict['dropout']) + \
                      '_multi=' + str(self.multi_class_configuration_dict['multi_class_bool']) + \
                      '_attention=' + str(self.attention_configuration_dict['use_attention_bool']) + \
                      '_time=' + str(self.cur_time)

        return file_suffix

    # run lstm model with embedding using Keras platform
    def run_model_lstm_keras_old(self):

        from classifier_lstm import PredictDescriptionModelLSTM

        logging.info('')
        logging.info('Run LSTM on Keras')

        lstm_obj = PredictDescriptionModelLSTM(
            '',     # self.file_directory,
            logging,
            self.cur_time,
            self.train_df[self.df_configuration_dict['x_column']],  # ['Review'],
            self.train_df[self.df_configuration_dict['y_column']],  # ['review_tag'],
            self.test_df[self.df_configuration_dict['x_column']],   # e.g ['Review'],
            self.test_df[self.df_configuration_dict['y_column']],   # e.g. ['review_tag'],
            self.lstm_parameters_dict,      # self.lstm_parameters
            self.df_configuration_dict,     # df column configuration
            self.tensor_board_dir,
            self.embedding_pre_trained,     # Boolean value
            self.vertical_type              # vertical fashion/motors

        )

        test_loss, test_accuracy = lstm_obj.run_experiment()

        logging.info('finish LSTM model')

        return


def main():
    raise()


if __name__ == '__main__':

    raise()