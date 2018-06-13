from __future__ import print_function
import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt


class CleanData:
    '''
    remain only valid users
    check duplication
    remove according to threshold defined
    '''

    def __init__(self, input_data_file, vertical_type, output_clean_folder):

        # file arguments
        self.input_data_file = input_data_file          # csv input file
        self.vertical_type = vertical_type              # 'fashion'/'motors'
        self.output_clean_folder = output_clean_folder  # output folder to store clean data

        self.verbose_flag = True

        from time import gmtime, strftime
        self.cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

        # define data frame needed for analyzing data
        self.data_set_df = pd.DataFrame()

    # init log file
    def init_debug_log(self):
        import logging

        lod_file_name = './log/' + 'clean_data_' + str(self.cur_time) + '.log'

        logging.basicConfig(filename=lod_file_name,
                            format='%(asctime)s, %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
        import os
        if not os.path.exists(lod_file_name):
            os.makedirs(lod_file_name)

        # print result in addition to log file
        if self.verbose_flag:
            stderrLogger = logging.StreamHandler()
            stderrLogger.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
            logging.getLogger().addHandler(stderrLogger)

        logging.info("")
        logging.info("")
        logging.info("start log program")
        return

    # load csv into df
    def load_clean_csv_results(self):

        self.data_set_df = pd.read_csv(self.input_data_file)

        return

    # statistical analysis
    def statistical_data(self):

        logging.info('Data vertical type: ' + str(self.vertical_type))
        logging.info('start statistical meta analysis')
        logging.info('')
        logging.info('data size: ' + str(self.data_set_df.shape[0]))

        self._target_column_analysis()
        self._review_analysis()
        self._reason_failure_analysis()

        return

    def _target_column_analysis(self):

        logging.info('')
        logging.info('Tagging analysis (Y)')
        y_group = self.data_set_df.groupby(['Tagging'])
        for group_type, tag_group in y_group:
            group_percentage = float(tag_group.shape[0]) / float(self.data_set_df.shape[0])
            logging.info('Tag: ' + str(group_type) + ', amount: ' + str(tag_group.shape[0]) + ', percentage: ' + str(
                round(group_percentage, 2)))

        return

    def _review_analysis(self):

        logging.info('')
        logging.info('Reviews analysis')

        # add review length column
        self.data_set_df['review_length'] = np.nan

        # calculate review length
        for index, row in self.data_set_df.iterrows():
            if isinstance(row['Review'], basestring):
                review_length = len(row['Review'].split(' ')) - 1.0
            else:
                review_length = 0
            self.data_set_df.at[index, 'review_length'] = review_length

        # delete review with 0 length - empty
        init_data_size = self.data_set_df.shape[0]
        self.data_set_df = self.data_set_df[self.data_set_df['review_length'] != 0]
        logging.info('Reviews deleted because size=0: ' + str(init_data_size - self.data_set_df.shape[0]))
        logging.info('')

        # calculate review statistic
        percentile_series = self.data_set_df['review_length'].quantile([.25, .5, .75])
        review_q1 = percentile_series[.25]
        review_median = percentile_series[.5]
        review_q3 = percentile_series[.75]
        review_std = self.data_set_df['review_length'].std()
        review_max = self.data_set_df['review_length'].max()
        review_min = self.data_set_df['review_length'].min()

        logging.info('Reviews min: ' + str(round(review_min, 2)))
        logging.info('Reviews Q1: ' + str(round(review_q1, 2)))
        logging.info('Reviews median: ' + str(round(review_median, 2)))
        logging.info('Reviews Q3: ' + str(round(review_q3, 2)))
        logging.info('Reviews max: ' + str(round(review_max, 2)))
        logging.info('Reviews std: ' + str(round(review_std, 2)))

        # histogram of reviews length
        import matplotlib.pyplot as plt
        plt.figure()
        plt.title('Histogram of review length')
        plt.xlabel('Review length')
        plt.ylabel('Amount')
        plt.hist(self.data_set_df['review_length'], bins=int(review_max - 0), range=(0, review_max))
        plt.savefig(
            '/Users/sguyelad/PycharmProjects/reviews_classifier/results/statistics/' + 'histogram_' + str(self.vertical_type) + '_review_length')
        plt.close()


        return

    def _reason_failure_analysis(self):

        logging.info('')
        logging.info('Reason failure analysis')

        # remain only reviews which do not fit into descriptions
        failure_df = self.data_set_df[self.data_set_df['Tagging'] == 'Bad']
        logging.info('Bad reviews amount: ' + str(failure_df.shape[0]))

        failure_type_group = failure_df.groupby(['Reason'])

        # sort failure amount into dictionary
        failure_type_amount_dict = {}
        for group_name, group_df in failure_type_group:
            failure_type_amount_dict[group_name] = group_df.shape[0]

        # sort occurrences by failure reason amount
        import operator
        sorted_by_amount = sorted(failure_type_amount_dict.items(), key=operator.itemgetter(1))
        sorted_by_amount.reverse()

        # log failure type
        logging.info('')
        logging.info('Failure type amount:')
        for cur_tuple in sorted_by_amount:
            failure_type = cur_tuple[0]
            amount = cur_tuple[1]
            perncetile_group = float(amount)/float(failure_df.shape[0])
            aligned_type = '{:<43}'.format(failure_type)
            aligned_amount = '{:<6}'.format(amount)
            logging.info('Type: ' + str(aligned_type) + 'Amount: ' + str(aligned_amount) + 'percentile: ' + str(round(perncetile_group, 3)))

        # create bar plot
        failure_type_amount_dict_new = {}
        for key, val in failure_type_amount_dict.iteritems():
            if len(key) > 20:
                failure_type_amount_dict_new[key[0:20]] = val
            else:
                failure_type_amount_dict_new[key] = val

        import operator
        sorted_by_amount = sorted(failure_type_amount_dict_new.items(), key=operator.itemgetter(1))
        sorted_by_amount.reverse()

        list_failure_type = tuple(i[0] for i in sorted_by_amount)
        list_failure_amount = tuple(i[1] for i in sorted_by_amount)
        y_pos = np.arange(len(list_failure_type))

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.title('Failure type vs. amount')
        plt.ylabel('Amount')
        plt.bar(y_pos, list_failure_amount)
        plt.xticks(y_pos, list_failure_type, rotation=25)
        plt.tight_layout()
        plt.savefig(
            '/Users/sguyelad/PycharmProjects/reviews_classifier/results/statistics/' + 'histogram_' + str(self.vertical_type) + '_failure_type')
        plt.close()
        return

    # a. delete reviews with 0-length
    # b. change prediction to binary Bad-good to 0:1
    # c. change reason prediction to multi-class
    def clean_df(self):

        logging.info('')
        logging.info('clean data before start model:')

        # a. delete reviews with 0-length
        self.data_set_df = self.data_set_df[self.data_set_df['review_length'] != 0]

        # b. change prediction to binary Bad-good to 0:1
        self.data_set_df['review_tag'] = np.nan
        self.data_set_df['review_tag'] = np.where(self.data_set_df['Tagging'] == 'Bad', 0, 1)   # good - 1, bad - 0

        # b. change subjective to binary 0:1
        self.data_set_df['subjective_sentence'] = np.nan
        self.data_set_df['subjective_sentence'] = np.where(self.data_set_df['Reason'] == 'Subjective sentence', 1, 0)

        # c. change missing context to Binary 0:1
        self.data_set_df['missing_context'] = np.nan
        self.data_set_df['missing_context'] = np.where(self.data_set_df['Reason'] == 'Missing context', 1, 0)

        # c. change missing context to Binary 0:1
        self.data_set_df['Refers to a specific listing aspect'] = np.nan
        self.data_set_df['Refers to a specific listing aspect'] = np.where(self.data_set_df['Reason'] == 'Refers to a specific listing aspect', 1, 0)

        # c. change missing context to Binary 0:1
        self.data_set_df['Non-informative sentence'] = np.nan
        self.data_set_df['Non-informative sentence'] = np.where(
            self.data_set_df['Reason'].isin(['Non-informative sentence ', 'Non-informative sentence']), 1, 0)

        # c. change reason prediction to multi-class
        self.data_set_df['failure_reason'] = np.nan

        # calculate review length
        for index, row in self.data_set_df.iterrows():
            # print(row['Reason'])
            if isinstance(row['Reason'], basestring):
                self.data_set_df.at[index, 'failure_reason'] = self._reason_class_mapping(row['Reason'])
            else:   # nan - good review
                self.data_set_df.at[index, 'failure_reason'] = 12   # nan

        # save clean df
        file_path = self.output_clean_folder + 'clean_data_multi_new_' + str(self.vertical_type) + '.csv'
        self.data_set_df.to_csv(file_path)

        logging.info('')
        logging.info('save clean df: ' + file_path)
        return

    @staticmethod
    def _reason_class_mapping(reason):

        def find_class(reason):        # relative to -7 (server in USA)
            return {
                'Subjective sentence': 0,
                'Missing context': 1,
                'Refers to a specific listing aspect': 2,
                'Non-informative sentence ': 3,
                'Non-informative sentence': 3,
                'Poor language (spelling mistakes)': 4,
                'poor language (spelling mistakes)': 4,
                'Purely negative sentence ': 5,
                'Purely negative sentence': 5,
                'Expresses explicit doubt': 6,
                'Refers to the description': 7,
                'refers to the description': 7,
                'Too detailed': 8,
                'too detailed': 8,
                'Other (please explain in comments column)': 9,
                'Offensive language': 10,
                'Too specific/narrow': 11,
                'too specific/narrow': 11
                # nan - 12 - good review
            }[reason]

        cur_class = find_class(reason=reason)
        return cur_class


def main(input_data_file, vertical_type, output_clean_folder):

    clean_data_obj = CleanData(input_data_file, vertical_type, output_clean_folder)

    clean_data_obj.init_debug_log()                         # init log file
    clean_data_obj.load_clean_csv_results()                 # load data set
    clean_data_obj.statistical_data()                       # analyze data and save statistical data
    clean_data_obj.clean_df()                               # clean df - e.g. remain valid users only


if __name__ == '__main__':

    # input file name
    vertical_type = 'motors'   # 'fashion'/'motors'
    output_clean_folder = '../data/clean/'
    if vertical_type == 'fashion':
        input_data_file = '../data/25K_amazon_fashion.csv'
    elif vertical_type == 'motors':
        input_data_file = '../data/25K_amazon_motors.csv'
    else:
        raise()

    main(input_data_file, vertical_type, output_clean_folder)