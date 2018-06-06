from __future__ import print_function
from scipy.stats import ttest_ind


class StatisticalSignificance:

    def __init__(self):

        a= 5

    # statistical analysis
    def check_statistical_data(self):

        group_a_result = [0.92, 0.916, 0.916, 0.918, 0.921]     # LSTM +
        group_b_result = [0.91, 0.908, 0.909, 0.913, 0.914]

        t_stat, p_value = ttest_ind(group_a_result, group_b_result, equal_var = False)

        print('group a: ' + str(group_a_result))
        print('group b: ' + str(group_b_result))
        print('t statistic: ' + str(round(t_stat, 4)))
        print('p value: ' + str(round(p_value, 4)))

        print('')
        print('check extend groups')

        group_a_result = [0.92, 0.916, 0.916, 0.918, 0.921, 0.909, 0.909, 0.918, 0.922, 0.931]
        group_b_result = [0.91, 0.908, 0.909, 0.913, 0.914, 0.906, 0.907, 0.911, 0.918, 0.919]

        t_stat, p_value = ttest_ind(group_a_result, group_b_result, equal_var=True)

        print('group a extend: ' + str(group_a_result))
        print('group b extend: ' + str(group_b_result))
        print('t statistic: ' + str(round(t_stat, 4)))
        print('p value: ' + str(round(p_value, 4)))
        return


def main():

    stat_obj = StatisticalSignificance()

    stat_obj.check_statistical_data()                         # init log file


if __name__ == '__main__':
    main()