"""
.. module:: validation
   :synopsis: Validation Modules

Lib: validation

This module is to the collection of functions and classes to help generate validation report and perform
analysis.

Todo:
    * Docstring fro all the utils
"""

# region imports and set up
from arch.unitroot import ADF
from arch import arch_model
from arch.bootstrap import StationaryBootstrap
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
# import pyflux as pf
import warnings
import math
import multiprocessing as mp

from scipy.stats import ttest_ind

#from post_modelling_analysis import logger

warnings.filterwarnings("ignore") 


# endregion

# region GLOBALS

def return_percent_func(x):
    return np.sum(x)


def annualised_sharpe_func(df):
    return (np.mean(df) / np.std(df)) * (252 ** 0.5)


DICT_PERFORMANCE_METRIC_FUNCTIONS = {'annualised_sharpe': annualised_sharpe_func,
                                     'return': np.sum,
                                     'return_percent': return_percent_func}


DICT_STRAT_NAME = {'BETA' : 'BETA',
                   'BAYES' : 'BAYES',
                   'RSEC' : 'RSEC'}

# endregion

# region helper functions

def ecdf(ser, thresh):
    """
    compute the empirical cumulative distribution function of the series ser, with cutoff array thresh

    """

    l = len(ser) 

    return pd.Series(data=[(float((ser <= x).sum())) / l for x in thresh], index=thresh)


def l_infty(cdf1, cdf2):
    hulp = pd.DataFrame()
    hulp['cdf1'] = (cdf1.reset_index()[0])
    hulp['cdf2'] = (cdf2.reset_index()[0])
    hulp['absdiff'] = (hulp['cdf1'] - hulp['cdf2']).abs()

    return hulp['absdiff'].max()


def c(alpha):
    return np.sqrt(-0.5 * np.log(alpha / 2))


def ks(n, m, alpha):
    return c(alpha) * np.sqrt((float(n + m)) / (n * m))


def get_CI(series, ci):
    """
    calculate lower and upper bound at confidence level ci

    """

    lower_bound = np.percentile(series, (100 - ci) / 2)
    upper_bound = np.percentile(series, ci + (100 - ci) / 2)

    return lower_bound, upper_bound

def get_pct(series, value):
    """
    calculate the percentile of a value
    """
    percentile = stats.percentileofscore(series, value)
    return percentile

def merge_partitions(partitions, list_num_par, length_out):
    '''

    merge dataframe partitions

    Args:

        partitions (dict): dict of dataframes
        list_num_par (list of int): list of dataframe indexes to merge

    Returns:

        df (pandas.DataFrame): merged df

    '''

    df = partitions[0]

    for num_par in list_num_par[1:]:
        df = df.append(partitions[num_par])

    return df.iloc[:length_out]

# region multiprocessing

def get_samples_CB(series_value, partition_size, n_val, statistic_func, num_samples):
    blocks_per_sample = math.ceil(n_val / partition_size)
    num_blocks = blocks_per_sample * num_samples
    starting_points = np.random.choice(np.arange(len(series_value)), num_blocks)
    indicies = np.repeat(starting_points, partition_size) + np.tile(np.arange(partition_size), len(starting_points))
    indicies_wrapped = np.mod(indicies, len(series_value))
    long_sample = series_value[indicies_wrapped][:((partition_size * blocks_per_sample * num_samples))]
    sample_list_cb = list(np.array_split(long_sample, num_samples))
    #statistic_list = [statistic_func(x) for x in sample_list_cb]
    return sample_list_cb

def bootstrap_adf(i, sample_list_cb):
    df_simulated_valid_garch = pd.DataFrame(sample_list_cb[i])
    adf_garch_pvalue = ADF(df_simulated_valid_garch).pvalue
    return adf_garch_pvalue

def bootstrap_garch(i, sample_list_cb):
    df_simulated_valid_garch = pd.DataFrame(sample_list_cb[i])
    am_garch = arch_model(df_simulated_valid_garch.iloc[:, 0], p = 1)
    res_garch_pvalue = am_garch.fit(update_freq = 5, disp = 'off', show_warning = False).pvalues[2]
    return res_garch_pvalue

def bootstrap_levene_valid(i, sample_list_cb, valid_series):
    df_simulated_valid_garch = pd.DataFrame(sample_list_cb[i])
    levene_valid_pvalue = stats.levene(df_simulated_valid_garch.iloc[:, 0], valid_series).pvalue
    return levene_valid_pvalue

def bootstrap_levene_train(i, sample_list_cb, train_series):
    df_simulated_valid_garch = pd.DataFrame(sample_list_cb[i])
    levene_train_pvalue = stats.levene(df_simulated_valid_garch.iloc[:, 0], train_series).pvalue
    return levene_train_pvalue

def bootstrap_kst_approx_train(i, sample_list_cb, train_series):
    df_simulated_valid_garch = pd.DataFrame(sample_list_cb[i])
    rvs_boot = df_simulated_valid_garch.values
    rvs_train = train_series.values
    endpoint_train = max([np.max(np.abs(rvs_boot)), np.max(np.abs(rvs_train))])
    thresh_train = np.linspace(-endpoint_train, endpoint_train, 10000)
    cdf_boot = ecdf(rvs_boot, thresh_train)
    cdf_train = ecdf(rvs_train, thresh_train)
    maxDif_train = l_infty(cdf_boot, cdf_train)
    critical_Val_train = ks(len(rvs_boot), len(rvs_train), 0.05)
    kst_train_approx = (maxDif_train - critical_Val_train)
    return kst_train_approx

def bootstrap_kst_twoside_train(sample_list_cb, train_series, kst_train_twoside):
    for i in range(len(sample_list_cb)):
        df_simulated_valid_garch = pd.DataFrame(sample_list_cb[i])
        rvs_boot = df_simulated_valid_garch.values
        rvs_train = train_series.values
        KS_train_pValue = stats.ks_2samp(rvs_boot, rvs_train).pvalue
        kst_train_twoside.append(KS_train_pValue)
    return kst_train_twoside

def bootstrap_kst_approx_valid(i, sample_list_cb, valid_series):
    df_simulated_valid_garch = pd.DataFrame(sample_list_cb[i])
    rvs_boot = df_simulated_valid_garch.values
    rvs_valid = valid_series.values
    endpoint_valid = max([np.max(np.abs(rvs_boot)), np.max(np.abs(rvs_valid))])
    thresh_valid = np.linspace(-endpoint_valid, endpoint_valid, 10000)
    cdf_boot = ecdf(rvs_boot, thresh_valid)
    cdf_valid = ecdf(rvs_valid, thresh_valid)
    maxDif_valid = l_infty(cdf_boot, cdf_valid)
    critical_Val_valid = ks(len(rvs_boot), len(rvs_valid), 0.05)
    kst_valid_approx = (maxDif_valid - critical_Val_valid)
    return kst_valid_approx

def bootstrap_kst_twoside_valid(sample_list_cb, valid_series, kst_valid_twoside):
    for i in range(len(sample_list_cb)):
        df_simulated_valid_garch = pd.DataFrame(sample_list_cb[i])
        rvs_boot = df_simulated_valid_garch.values
        rvs_valid = valid_series.values
        KS_valid_pValue = stats.ks_2samp(rvs_boot, rvs_valid).pvalue
        kst_valid_twoside.append(KS_valid_pValue)
    return kst_valid_twoside

# endregion


# region main classes

class ttest:

    """
    A class to perform t-test

    Args:
        data_Dir: Location of data
        fileName: file name
        performanceMeasure: Performance Measure
        Training_Period_Start: Start date of training period
        Training_Period_End: End data of training period
        Validation_Period_Start_1: validation period-1 start date
        Validation_Period_Start_2: validation period-2 start date
        Validation_Period_End_1: validation period-1 end date
        Validation_Period_End_2: validation period-2 end date

    """


    def __init__(self, data_Dir , fileName, performanceMeasure, Training_Period_Start = '2011-01-01',
                             Training_Period_End = '2016-12-31',
                             Validation_Period_Start_1 = '2010-01-01',
                             Validation_Period_Start_2 = '2017-01-01',
                             Validation_Period_End_1 = '2010-12-31',
                             Validation_Period_End_2 = '2019-12-31' ):
        """

        Args:
            data_Dir:
            fileName:
            performanceMeasure:
            Training_Period_Start:
            Training_Period_End:
            Validation_Period_Start_1:
            Validation_Period_Start_2:
            Validation_Period_End_1:
            Validation_Period_End_2:
        """
        self.data_Dir = data_Dir
        self.fileName = fileName
        self.performanceMeasure = performanceMeasure
        self.Training_Period_Start = Training_Period_Start
        self.Training_Period_End = Training_Period_End
        self.Validation_Period_Start_1 = Validation_Period_Start_1
        self.Validation_Period_End_1 = Validation_Period_End_1
        self.Validation_Period_Start_2 = Validation_Period_Start_2
        self.Validation_Period_End_2 = Validation_Period_End_2


    def _create_train_valid_data(self, df):
        """
        Create train and validation splits
        Args:
            df: pandas data frame

        Returns: train and valid time series

        """

        df_train = df.loc[self.Training_Period_Start:self.Training_Period_End]

        df_valid = (df.loc[self.Validation_Period_Start_1:self.Validation_Period_End_1]).append(
            df.loc[self.Validation_Period_Start_2:self.Validation_Period_End_2])

        return df_train, df_valid

    def perform_test(self):
        """
        Run the t-test

        Returns: valstats and one sided p-value

        """
        df_return_series = pd.read_csv(self.data_Dir + self.fileName, index_col=0)[self.performanceMeasure]
        train_series, valid_series = self._create_train_valid_data(df_return_series)
        # train_series = df_return_series.loc[Training_Period_Start:Training_Period_End]
        # valid_series = df_return_series.loc[Validation_Period_Start:Validation_Period_End]
        self.pval = None

        def t_test(y_train, y_val):
            tstat, pval = ttest_ind(y_train, y_val, equal_var=False)
            # logger.info('t-test: test statistic is {} with one sided p-value of {}'.format(tstat, pval / 2))
            print('t-test: test statistic is {} with one sided p-value of {}'.format(str(round(tstat,3)), str(round((pval / 2),3))))
            if pval/2 >= 0.05:
                print('Not reject null at 5%. Train and valid populations are not different.')
            elif 0.01 <= pval/2 < 0.05:
                if y_train > y_val:
                    print('Reject null at 5%. Train population is greater than valid population')
                else:
                    print('Reject null at 5%. Train population is smaller than valid population')
            else:
                if y_train > y_val:
                    print('Reject null at 1%. Train population is greater than valid population')
                else:
                    print('Reject null at 1%. Train population is smaller than valid population')
            return tstat, pval / 2

        self.valstat, self.pval = t_test(train_series, valid_series)

        return self.valstat, self.pval


class KSTest:
    """
    perform ks test on strategy performance series

    CI (float): confidence level

    """

    def __init__(self,
                 CI):

        self.maxDif = None
        self.KS_stat = None
        self.pValue = None
        self.cdf1 = None
        self.cdf2 = None
        self.CI = CI
        self.train_period = None
        self.valid_period = None
        self.performance_measure = None
        self.maxDif = None
        self.critical_Val = None

        return

    def perform_test(self,
                     data_Dir,
                     fileName,
                     Training_Period_Start,
                     Training_Period_End,
                     Validation_Period_Start,
                     Validation_Period_End,
                     performanceMeasure,
                     strat_name=None):

        self.train_period = '{} to {}'.format(Training_Period_Start, Training_Period_End)
        self.valid_period = '{} to {}'.format(Validation_Period_Start, Validation_Period_End)
        self.performance_measure = performanceMeasure
        if strat_name is None: strat_name = ''
        c2cData = pd.read_csv(data_Dir + fileName)
        dailyReturnDF = c2cData[['Date', performanceMeasure]]

        train_ret = dailyReturnDF[
            (dailyReturnDF['Date'] >= Training_Period_Start) & (dailyReturnDF['Date'] <= Training_Period_End)]

        valid_ret = dailyReturnDF[
            (dailyReturnDF['Date'] >= Validation_Period_Start) & (dailyReturnDF['Date'] <= Validation_Period_End)]

        rvs1 = train_ret[performanceMeasure].values
        rvs2 = valid_ret[performanceMeasure].values

        endpoint = max([np.max(np.abs(rvs1)), np.max(np.abs(rvs2))])
        thresh = np.linspace(-endpoint, endpoint, 10000)
        self.cdf1 = ecdf(rvs1, thresh)
        self.cdf2 = ecdf(rvs2, thresh)

        self.maxDif = l_infty(self.cdf1, self.cdf2)
        self.critical_Val = ks(len(rvs1), len(rvs2), 1 - self.CI)

        print('########################################')
        print('{} Approximate KS-test Values: left_hand_{}_RightHand_{}'.format(strat_name, self.maxDif,
                                                                                self.critical_Val))

        if self.maxDif > self.critical_Val:
            print('Null Hypothesis is rejected and thus validation is not the same as training')
        else:
            print('Null Hypothesis is accepted and thus validation is the same as training')

        print('########################################')

        print('                                        ')
        print('########################################')

        print('SCIPY ks two-sided test {}'.format(strat_name))
        self.KS_stat, self.pValue = stats.ks_2samp(rvs1, rvs2)
        print('KS-test: KS_stat_{}_pValue_{}\n \n'.format(self.KS_stat, self.pValue))
        print('########################################')
        return

    def plot_result(self, strat_name=None):

        from textwrap import wrap
        if strat_name is None: strat_name = ''
        plt.figure()
        plt.plot(self.cdf1)
        plt.plot(self.cdf2)
        plt.gca().set_title("\n".join(wrap('{} KS Test, P-Value={} \
                            train_period: {} \
                            valid_period: {} \
                            performance_measure: {}' \
                                           .format(strat_name, round(self.pValue, 2),
                                                   self.train_period, self.valid_period, self.performance_measure),
                                           60)),
                            fontsize='large')

        plt.gca().legend(('train', 'valid'), fontsize='large')

        return

class BootstrapValid:
    """
    consistency check train data set and validation data set

    """

    def __init__(self):

        self.train_period = None
        self.valid_period = None
        self.performance_measure = None
        self.partition_size = None
        self.num_sample = None
        self.perf_metric = None
        self.df_simulaiton = None
        self.train_series = None
        self.valid_series = None
        self.valid_result = None
        self.train_result = None
        self.replacement = None
        self.df_summary = pd.DataFrame(index = [0], columns = ['strategy', 'num_sample', 'partition_size', 'bootstrap_mean',
                                                  'bootstrap_std', 'valid_pct_dev'])
        #self.df_sample_series = pd.DataFrame(index=[0], columns=['one', 'two', 'three', 'four'])

        return

    def simulate(self,
                 data_Dir,
                 fileName,
                 Training_Period_Start,
                 Training_Period_End,
                 Validation_Period_Start,
                 Validation_Period_End,
                 performanceMeasure,
                 partition_size=3,
                 num_sample=1000,
                 perf_metric='annualised_sharpe',
                 replacement=True):

        self.train_period = '{} to {}'.format(Training_Period_Start, Training_Period_End)
        self.valid_period = '{} to {}'.format(Validation_Period_Start, Validation_Period_End)
        self.performance_measure = performanceMeasure
        self.partition_size = partition_size
        self.num_sample = num_sample
        self.perf_metric = perf_metric
        perf_metric_func = DICT_PERFORMANCE_METRIC_FUNCTIONS[self.perf_metric]
        df_return_series = pd.read_csv(data_Dir + fileName, index_col=0)[performanceMeasure]
        self.train_series = df_return_series.loc[Training_Period_Start:Training_Period_End]
        self.valid_series = df_return_series.loc[Validation_Period_Start:Validation_Period_End]
        self.replacement = replacement
        self.valid_result = perf_metric_func(self.valid_series)
        self.train_result = perf_metric_func(self.train_series)

        train_num_partition = int(self.train_series.shape[0] / partition_size)
        valid_num_partition = int(self.valid_series.shape[0] / partition_size) + 1

        partitions = np.array_split(self.train_series, train_num_partition)

        list_partitions = list(range(train_num_partition))

        self.df_simulaiton = pd.DataFrame(index=range(num_sample), columns=['measurement'])

#        for i in range(num_sample):
#            simulated_valid_partition_num = np.random.choice(list_partitions, valid_num_partition,
#                                                             replace=self.replacement)

#            df_simulated_valid = merge_partitions(partitions, simulated_valid_partition_num,
#                                                  length_out=self.valid_series.shape[0])

#            self.df_simulaiton.loc[i] = perf_metric_func(df_simulated_valid)
        self.sample_list = []
        self.adf_garch_list = []
        self.res_beta_garch = []

        for i in range(num_sample):
            simulated_valid_partition_num = np.random.choice(list_partitions, valid_num_partition,
                                                             replace=self.replacement)

            df_simulated_valid = merge_partitions(partitions, simulated_valid_partition_num,
                                                  length_out=self.valid_series.shape[0])
            # if i % 10 == 0:
            #    self.sample_list.append(df_simulated_valid.to_frame(i))
            if i in [len(range(num_sample)) // 80, len(range(num_sample)) // 60, len(range(num_sample)) // 50,
                     len(range(num_sample)) // 40, len(range(num_sample)) // 20]:
                self.sample_list.append(df_simulated_valid.to_frame(i))

            self.df_simulaiton.loc[i] = perf_metric_func(df_simulated_valid)

            self.df_simulated_valid_garch = df_simulated_valid.to_frame()
            adf_garch = ADF(self.df_simulated_valid_garch)
            self.adf_garch_list.append(adf_garch.pvalue)

            am_garch = arch_model(self.df_simulated_valid_garch.iloc[:, 0], p=1)
            self.res_garch = am_garch.fit(update_freq=5, disp='off', show_warning=False)
            self.res_beta_garch.append(self.res_garch.pvalues[3])

        self.unit_root_count = sum(map(lambda x: x >= 0.05, self.adf_garch_list))

        self.beta_garch_count = sum(map(lambda x: x >= 0.05, self.res_beta_garch))

        self.sample_list_reset = [x.reset_index().drop('Date', axis=1) for x in self.sample_list]
        self.df_sample = pd.concat(self.sample_list_reset, axis=1)

        for i in range(4):
            simulated_valid_partition_num = np.random.choice(list_partitions, valid_num_partition,
                                                             replace=self.replacement)
            df_simulated_valid = merge_partitions(partitions, simulated_valid_partition_num,
                                                           length_out=self.valid_series.shape[0])
            self.df_sample_series.iloc[:, i] = df_simulated_valid
        return

    def plot_result(self, strat_name=None):

        """
        plot simulation result
        """
        if strat_name is None: strat_name = ''
        lower_bound_95, upper_bound_95 = get_CI(self.df_simulaiton['measurement'], 95)
        lower_bound_90, upper_bound_90 = get_CI(self.df_simulaiton['measurement'], 90)

        self.df_simulaiton['measurement'] = self.df_simulaiton['measurement'].astype(float)
        sns.kdeplot(self.df_simulaiton['measurement'], shade=True);
        plt.axvline(lower_bound_95, 0, 1, color='Red', linestyle='--', label='95% CI', linewidth=3)
        plt.axvline(upper_bound_95, 0, 1, color='Red', linestyle='--', linewidth=3)

        plt.axvline(lower_bound_90, 0, 1, color='Green', linestyle='--', label='90% CI', linewidth=3)
        plt.axvline(upper_bound_90, 0, 1, color='Green', linestyle='--', linewidth=3)

        plt.axvline(self.valid_result, 0, 1, color='Black', linestyle='--', label='Valid', linewidth=5)

        plt.title('%s num_sample = %s, %s, replacement=%s' % (strat_name, self.num_sample, self.perf_metric, str(self.replacement)), size='large')
        plt.margins(0.02)
        #plt.xlabel('perf_metric')
        plt.xlabel('%s' % (self.perf_metric))
        plt.ylabel('pdf')

        plt.legend()
        plt.show()
        return

    def plot_result_all(self, strat_name=None):

        """
        plot simulation result of bootstrap sample distributions
        """
        if strat_name is None: strat_name = ''
        self.df_simulaiton['measurement'] = self.df_simulaiton['measurement'].astype(float)
        # sns.kdeplot(self.df_simulaiton['measurement'], label = ('block size = %s; test stat = %s; one-sided p-value = %s' %(self.partition_size, str(round(self.valstat*100, 3)), str(round(self.pval, 3)))), shade=True);
        if self.pval >= 0.05:
            sns.kdeplot(self.df_simulaiton['measurement'], label = ('block size = %s; test stat = %s; one-sided p-value = %s. Not reject null at 5%%.' %(self.partition_size, str(round(self.valstat, 3)), str(round(self.pval, 3)))), shade=True);
        if 0.01 <= self.pval < 0.05:
            sns.kdeplot(self.df_simulaiton['measurement'], label = ('block size = %s; test stat = %s; one-sided p-value = %s. Reject null at 5%%.' % (self.partition_size, str(round(self.valstat, 3)), str(round(self.pval, 3)))), shade=True);
        if self.pval < 0.01:
            sns.kdeplot(self.df_simulaiton['measurement'], label = ('block size = %s; test stat = %s; one-sided p-value = %s. Reject null at 1%%.' % (self.partition_size, str(round(self.valstat, 3)), str(round(self.pval, 3)))), shade=True);
        plt.axvline(self.valid_result, 0, 1, color='Black', linestyle='--', linewidth=5)
        plt.title('%s, replacement=%s' % (strat_name, str(self.replacement)), size='large')
        plt.margins(0.02)
        plt.xlabel('%s' % (self.perf_metric))
        plt.ylabel('pdf')
        plt.legend()
        return

    def plot_sample_series(self, strat_name = None):
        """
        plot a sample of bootstrap train series
        """
        if strat_name is None: strat_name = ''
        plt.figure(figsize=(20, 80))
        for i in range(1, len(self.df_sample.columns)):
            plt.subplot(len(self.df_sample), 1, i)
            plt.plot(self.df_sample.iloc[:, i])
        return


    def df_summary_func(self, fileName, strat_name = None):
        """
        extract summary of the simulated data frames
        """
        if strat_name is None: strat_name = ''
        self.fileName = fileName
        self.df_summary['strategy'] = self.fileName
        self.df_summary['num_sample'] = self.num_sample
        self.df_summary['partition_size'] = self.partition_size
        self.df_summary['bootstrap_mean'] = self.df_simulaiton['measurement'].mean()
        self.df_summary['bootstrap_std'] = self.df_simulaiton['measurement'].std()
        self.df_summary['valid_pct_dev'] = get_pct(self.df_simulaiton['measurement'],self.valid_result) - 50
        return self.df_summary


class BootstrapAdvanced(BootstrapValid):
    """
    A Class to perform various Bootstrap techniques including Stationary Bootstrap (STB)
    """

    def __init__(self):

        super().__init__()
        self.boot_method = None

    def _create_train_valid_data(self, df, Training_Period_Start = '2011-01-01',
                             Training_Period_End = '2016-12-31',
                             Validation_Period_Start_1 = '2010-01-01',
                             Validation_Period_Start_2 = '2017-01-01',
                             Validation_Period_End_1 = '2010-12-31',
                             Validation_Period_End_2 = '2019-12-31'):
        """
        Create train and validation periods

        Args:
            df:
            Training_Period_Start:
            Training_Period_End:
            Validation_Period_Start_1:
            Validation_Period_Start_2:
            Validation_Period_End_1:
            Validation_Period_End_2:

        Returns:

        """

        df_train = df.loc[Training_Period_Start:Training_Period_End]

        df_valid = (df.loc[Validation_Period_Start_1:Validation_Period_End_1]).append(
            df.loc[Validation_Period_Start_2:Validation_Period_End_2])

        return df_train, df_valid

    def simulate_advanced(self, boot_method, para, num_cores, data_Dir, fileName,
                          Training_Period_Start, Training_Period_End, Validation_Period_Start_1,
                          Validation_Period_End_1,Validation_Period_Start_2,
                          Validation_Period_End_2, performanceMeasure, partition_size=3, num_sample=1000,
                          perf_metric='annualised_sharpe', replacement=True):
        """
        Simulate Advanced
        Args:
            boot_method:
            para:
            num_cores:
            data_Dir:
            fileName:
            Training_Period_Start:
            Training_Period_End:
            Validation_Period_Start_1:
            Validation_Period_End_1:
            Validation_Period_Start_2:
            Validation_Period_End_2:
            performanceMeasure:
            partition_size:
            num_sample:
            perf_metric:
            replacement:

        Returns:

        """

        self.boot_method = boot_method
        self.num_cores = num_cores
        self.train_period = '{} to {}'.format(Training_Period_Start, Training_Period_End)
        self.valid_period = '{} to {} & {} to {}'.format(Validation_Period_Start_1, Validation_Period_End_1,
                                                         Validation_Period_Start_2, Validation_Period_End_2)
        self.performance_measure = performanceMeasure
        self.partition_size = partition_size
        self.num_sample = num_sample
        self.perf_metric = perf_metric
        perf_metric_func = DICT_PERFORMANCE_METRIC_FUNCTIONS[self.perf_metric]
        df_return_series = pd.read_csv(data_Dir + fileName, index_col=0)[performanceMeasure]

        self.train_series, self.valid_series = self._create_train_valid_data(df_return_series,
                                                                             Training_Period_Start=Training_Period_Start,
                                                                             Training_Period_End=Training_Period_End,
                                                                             Validation_Period_Start_1=Training_Period_End,
                                                                             Validation_Period_Start_2=Validation_Period_Start_2,
                                                                             Validation_Period_End_1=Validation_Period_End_1,
                                                                             Validation_Period_End_2=Validation_Period_End_2)

        # self.train_series = df_return_series.loc[Training_Period_Start:Training_Period_End]
        # self.valid_series = df_return_series.loc[Validation_Period_Start:Validation_Period_End]


        self.replacement = replacement
        self.valid_result = perf_metric_func(self.valid_series)
        self.train_result = perf_metric_func(self.train_series)

        if self.boot_method == 'SBB':
            train_series = self.train_series.copy()
            valid_series = self.valid_series.copy()
            train_num_partition = int(self.train_series.shape[0] / partition_size)
            valid_num_partition = int(self.valid_series.shape[0] / partition_size) + 1
            partitions = np.array_split(self.train_series, train_num_partition)
            list_partitions = list(range(train_num_partition))

            self.df_simulaiton = pd.DataFrame(index=range(num_sample), columns=['measurement'])

            self.sample_list = []
            self.sample_list_sbb = []
            self.adf_garch_list = []
            self.res_alpha_garch = []
            self.levene_train_pvalue = []
            self.levene_valid_pvalue = []
            self.kst_train_approx = []
#            self.kst_train_twoside = []
            self.kst_valid_approx = []
#            self.kst_valid_twoside = []

            for i in range(num_sample):
                simulated_valid_partition_num = np.random.choice(list_partitions, valid_num_partition,
                                                                 replace=self.replacement)

                df_simulated_valid = merge_partitions(partitions, simulated_valid_partition_num,
                                                      length_out=self.valid_series.shape[0])

                self.df_simulaiton.loc[i] = perf_metric_func(df_simulated_valid)
                self.sample_list_sbb.append(df_simulated_valid)
                # Select some random bootstrap samples for plotting:
#                if i in [len(range(num_sample)) // 80, len(range(num_sample)) // 60, len(range(num_sample)) // 50,
#                         len(range(num_sample)) // 40, len(range(num_sample)) // 20]:
#                    self.sample_list.append(df_simulated_valid.to_frame(i))
            with mp.Pool(num_cores) as pool:
                res_adf = [pool.apply_async(bootstrap_adf,
                                            args=(i, self.sample_list_sbb))
                           for i in range(len(self.sample_list_sbb))]
                for r in res_adf:
                    self.adf_garch_list.append(r.get())

            with mp.Pool(num_cores) as pool:
                res_garch_tmp = [pool.apply_async(bootstrap_garch,
                                                  args=(i, self.sample_list_sbb))
                                 for i in range(len(self.sample_list_sbb))]
                for r in res_garch_tmp:
                    self.res_alpha_garch.append(r.get())

            with mp.Pool(num_cores) as pool:
                res_levene_valid = [pool.apply_async(bootstrap_levene_valid,
                                                     args=(i, self.sample_list_sbb, valid_series))
                                    for i in range(len(self.sample_list_sbb))]
                for r in res_levene_valid:
                    self.levene_valid_pvalue.append(r.get())
            #
            with mp.Pool(num_cores) as pool:
                res_levene_train = [pool.apply_async(bootstrap_levene_train,
                                                     args=(i, self.sample_list_sbb, train_series))
                                    for i in range(len(self.sample_list_sbb))]
                for r in res_levene_train:
                    self.levene_train_pvalue.append(r.get())
            #
            with mp.Pool(num_cores) as pool:
                res_kst_approx_train = [pool.apply_async(bootstrap_kst_approx_train,
                                                         args=(i, self.sample_list_sbb, train_series))
                                        for i in range(len(self.sample_list_sbb))]
                for r in res_kst_approx_train:
                    self.kst_train_approx.append(r.get())

            #                with mp.Pool(num_cores) as pool:
            #                    res_kst_twoside_train = [pool.apply_async(bootstrap_kst_twoside_train,
            #                                                              args = (self.sample_list_cb, train_series, kst_train_twoside_emp))
            #                                             for i in range(num_cores)]
            #                    for r in res_kst_twoside_train:
            #                        self.kst_train_twoside.extend(r.get())

            with mp.Pool(num_cores) as pool:
                res_kst_approx_valid = [pool.apply_async(bootstrap_kst_approx_valid,
                                                         args=(i, self.sample_list_sbb, valid_series))
                                        for i in range(len(self.sample_list_sbb))]
                for r in res_kst_approx_valid:
                    self.kst_valid_approx.append(r.get())

            # Extract p-values of those tests
            self.unit_root_count = sum(map(lambda x: x >= 0.05, self.adf_garch_list))
            self.alpha_garch_count = sum(map(lambda x: x >= 0.05, self.res_alpha_garch))
            self.levene_valid_pvalue_count = sum(map(lambda x: x < 0.05, self.levene_valid_pvalue))
            self.levene_train_pvalue_count = sum(map(lambda x: x < 0.05, self.levene_train_pvalue))
            self.kst_train_approx_count = sum(map(lambda x: x > 0, self.kst_train_approx))
#            self.kst_train_twoside_count = sum(map(lambda x: x > 0, self.kst_train_twoside))
            self.kst_valid_approx_count = sum(map(lambda x: x > 0, self.kst_valid_approx))
#            self.kst_valid_twoside_count = sum(map(lambda x: x > 0, self.kst_valid_twoside))


            # Create a data frame for the randomly selected samples above for plotting
#            self.sample_list_reset = [x.reset_index().drop('Date', axis = 1) for x in self.sample_list]
#            self.df_sample = pd.concat(self.sample_list_reset, axis = 1)

            return self.df_summary

        elif self.boot_method == 'CB':
            self.df_sample = pd.DataFrame()
            self.adf_garch_list = []
            self.res_alpha_garch = []
            self.some_new_results =[]
            self.levene_train_pvalue = []
            self.levene_valid_pvalue = []
            self.kst_train_approx = []
            self.kst_train_twoside = []
            self.kst_valid_approx = []
            self.kst_valid_twoside = []

            n_val = len(self.valid_series)

            if para:
                train_series = self.train_series.copy()
                valid_series = self.valid_series.copy()
                samples_per_core = math.ceil(self.num_sample / self.num_cores)
                self.results = []
                self.sample_list_cb = []

                with mp.Pool(num_cores) as pool:
                    res = [pool.apply_async(get_samples_CB,
                                            args=(train_series.values,
                                                  partition_size,
                                                  n_val,
                                                  perf_metric_func,
                                                  samples_per_core))
                           for i in range(num_cores)]
                    for r in res:
                        self.sample_list_cb.extend(r.get())

                self.results = [perf_metric_func(x) for x in self.sample_list_cb]

                with mp.Pool(num_cores) as pool:
                    res_adf = [pool.apply_async(bootstrap_adf,
                                                args = (i, self.sample_list_cb))
                               for i in range(len(self.sample_list_cb))]
                    for r in res_adf:
                        self.adf_garch_list.append(r.get())

                with mp.Pool(num_cores) as pool:
                    res_garch_tmp = [pool.apply_async(bootstrap_garch,
                                                  args = (i, self.sample_list_cb))
                                 for i in range(len(self.sample_list_cb))]
                    for r in res_garch_tmp:
                        self.res_alpha_garch.append(r.get())

                with mp.Pool(num_cores) as pool:
                    res_levene_valid = [pool.apply_async(bootstrap_levene_valid,
                                                         args = (i, self.sample_list_cb, valid_series))
                                        for i in range(len(self.sample_list_cb))]
                    for r in res_levene_valid:
                        self.levene_valid_pvalue.append(r.get())
                #
                with mp.Pool(num_cores) as pool:
                    res_levene_train = [pool.apply_async(bootstrap_levene_train,
                                                         args = (i, self.sample_list_cb, train_series))
                                        for i in range(len(self.sample_list_cb))]
                    for r in res_levene_train:
                        self.levene_train_pvalue.append(r.get())
                #
                with mp.Pool(num_cores) as pool:
                    res_kst_approx_train = [pool.apply_async(bootstrap_kst_approx_train,
                                                             args = (i, self.sample_list_cb, train_series))
                                            for i in range(len(self.sample_list_cb))]
                    for r in res_kst_approx_train:
                        self.kst_train_approx.append(r.get())

#                with mp.Pool(num_cores) as pool:
#                    res_kst_twoside_train = [pool.apply_async(bootstrap_kst_twoside_train,
#                                                              args = (self.sample_list_cb, train_series, kst_train_twoside_emp))
#                                             for i in range(num_cores)]
#                    for r in res_kst_twoside_train:
#                        self.kst_train_twoside.extend(r.get())

                with mp.Pool(num_cores) as pool:
                    res_kst_approx_valid = [pool.apply_async(bootstrap_kst_approx_valid,
                                                             args=(i, self.sample_list_cb, valid_series))
                                            for i in range(len(self.sample_list_cb))]
                    for r in res_kst_approx_valid:
                        self.kst_valid_approx.append(r.get())

#                with mp.Pool(num_cores) as pool:
#                    res_kst_twoside_valid = [pool.apply_async(bootstrap_kst_twoside_valid,
#                                                              args=(
#                                                              self.sample_list_cb, valid_series, kst_valid_twoside_emp))
#                                             for i in range(num_cores)]
#                    for r in res_kst_twoside_valid:
#                        self.kst_valid_twoside.extend(r.get())

                # Select some random bootstrap samples for plotting:
                for i in [num_sample//80, num_sample//60, num_sample//50, num_sample//40, num_sample//20]:
                    self.sample_list_tmp = pd.DataFrame(self.sample_list_cb[i])
                    self.df_sample = pd.concat([self.sample_list_tmp, self.df_sample], axis = 1)
                cols = pd.Series(self.df_sample.columns)
                for dup in cols[cols.duplicated()].unique():
                    cols[cols[cols == dup].index.values.tolist()] = ['bootstrap_' + str(j) for j in range(sum(cols == dup))]
                self.df_sample.columns = cols

                # Extract p-values of those tests
                self.unit_root_count = sum(map(lambda x: x >= 0.05, self.adf_garch_list))
                self.alpha_garch_count = sum(map(lambda x: x >= 0.05, self.res_alpha_garch))
                self.levene_valid_pvalue_count = sum(map(lambda x: x < 0.05, self.levene_valid_pvalue))
                self.levene_train_pvalue_count = sum(map(lambda x: x < 0.05, self.levene_train_pvalue))
                self.kst_train_approx_count = sum(map(lambda x: x > 0, self.kst_train_approx))
#                self.kst_train_twoside_count = sum(map(lambda x: x > 0, self.kst_train_twoside))
                self.kst_valid_approx_count = sum(map(lambda x: x > 0, self.kst_valid_approx))
#                self.kst_valid_twoside_count = sum(map(lambda x: x > 0, self.kst_valid_twoside))

            else:
                self.results, self.sample_list_cb = get_samples_CB(self.train_series.values, self.partition_size,
                                                                   n_val, perf_metric_func, num_sample)

                # Select some random bootstrap samples for plotting:
                for i in [num_sample//80, num_sample//60, num_sample//50, num_sample//40, num_sample//20]:
                    self.sample_list_tmp = pd.DataFrame(self.sample_list_cb[i])
                    self.df_sample = pd.concat([self.sample_list_tmp, self.df_sample], axis = 1)
                cols = pd.Series(self.df_sample.columns)
                for dup in cols[cols.duplicated()].unique():
                    cols[cols[cols == dup].index.values.tolist()] = ['bootstrap_' + str(j) for j in range(sum(cols == dup))]
                self.df_sample.columns = cols

                # Augmented Dickey-Fuller test for unit root of bootstrap samples
                for i in range(len(self.sample_list_cb)):
                    self.df_simulated_valid_garch = pd.DataFrame(self.sample_list_cb[i])
                    adf_garch = ADF(self.df_simulated_valid_garch)
                    self.adf_garch_list.append(adf_garch.pvalue)

                # Conditional heteroskedasticity test of bootstrap samples
                am_garch = arch_model(self.df_simulated_valid_garch.iloc[:, 0], p=1)
                self.res_garch = am_garch.fit(update_freq=5, disp='off', show_warning=False)
                self.res_beta_garch.append(self.res_garch.pvalues[3])

                # Extract p-values of those tests
                self.unit_root_count = sum(map(lambda x: x >= 0.05, self.adf_garch_list))
                self.beta_garch_count = sum(map(lambda x: x >= 0.05, self.res_beta_garch))

            self.df_simulaiton = pd.DataFrame(index=range(len(self.results)), data=self.results, columns=['measurement']) #range(num_sample)

            # Create a data frame for the randomly selected samples above for plotting
            # self.sample_list_reset = [x.reset_index().drop('Date', axis = 1) for x in self.sample_list_cb]
            # self.df_sample = pd.concat(self.sample_list_reset, axis = 1)
            return self.results

        elif  self.boot_method == 'STB':
            train_series = self.train_series.copy()
            valid_series = self.valid_series.copy()
            self.valstat = valid_series.sum()
            self.pval = None
            self.samples = pd.DataFrame()
            self.df_simulaiton = pd.DataFrame(index = range(num_sample), columns = ['measurement'])

            bs = StationaryBootstrap(partition_size, train_series)

            def sum_boot(x, val):
                return sum(x[:len(val)])  # make it the size of the validation series

            results = bs.apply(lambda x: sum_boot(x, val=valid_series), num_sample)
            self.samples = pd.DataFrame(results).iloc[:, 0]
            self.pval = sum(self.samples < self.valstat) / len(self.samples)

            for i in range(num_sample):
                self.df_simulaiton.loc[i] = perf_metric_func(results[i])
        else:
            raise(NotImplementedError('Invalid Booststrap method {}'.format(self.boot_method)))


class validation_test_overview:
    """
    overview of the series

    """

    def __init__(self,
                 data_Dir,
                 fileName,
                 Training_Period_Start,
                 Training_Period_End,
                 Validation_Period_Start,
                 Validation_Period_End,
                 performanceMeasure):

        self.data_Dir = data_Dir,
        self.fileName = fileName,
        self.Training_Period_Start = Training_Period_Start,
        self.Training_Period_End = Training_Period_End,
        self.Validation_Period_Start = Validation_Period_Start,
        self.Validation_Period_End = Validation_Period_End,
        self.performance_measure = performanceMeasure
        self.train_period = '{} to {}'.format(Training_Period_Start, Training_Period_End)
        self.valid_period = '{} to {}'.format(Validation_Period_Start, Validation_Period_End)
        self.df_return_series = pd.read_csv(data_Dir + fileName, index_col=0)[performanceMeasure]
        self.train_series = self.df_return_series.loc[Training_Period_Start:Training_Period_End]
        self.valid_series = self.df_return_series.loc[Validation_Period_Start:Validation_Period_End]

    def plot_series(self, title=None):

        """
        plot overview of the series
        """

        # self.df_return_series.index=pd.to_datetime(self.df_return_series.index)
        # self.df_return_series.cumsum().plot()
        to_plot_data = np.block([self.train_series.values, self.valid_series.values])
        print(to_plot_data.shape)
        to_plot = pd.Series(data=to_plot_data)
        # print(self.train_series.shape, self.valid_series.shape, to_plot.shape)
        to_plot.index = range(to_plot.index.shape[0])
        # print(to_plot)
        to_plot.cumsum().plot()
        plt.axvline(self.train_series.shape[0], 0, 1, color='Red', linestyle='--', label='train_valid_split',
                    linewidth=3)
        if title is None:
            plt.title('cumulative %s' % self.performance_measure)
        else:
            plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()

        return

    # def plot_acf(self):
    #
    #     """
    #     check serial correlation
    #     """
    #
    #     pf.acf_plot(self.train_series.dropna().values)
    #
    #     return

    # endregion


# endregion

# region function to generate reports

def generate_validation_report(dir_out, valid_test, ks_test, bootstrap_valid):
    """
    save report

    dir_out (path)
    valid_test (instance)
    ks_test (instance)
    bootstrap_valid (instance)

    """

    # save all plots

    dir_plots = os.path.dirname(dir_out) + '/plot/'
    if not os.path.exists(dir_plots): os.makedirs(dir_plots)

    plt.close("all")
    valid_test.plot_series()
    plt.savefig(dir_plots + 'series.png', bbox_inches='tight')
    plt.close("all")

    # plt.close("all")
    # valid_test.plot_acf()
    # plt.savefig(dir_plots + 'acf.png', bbox_inches='tight')
    # plt.close("all")

    plt.close("all")
    bootstrap_valid.plot_result()
    plt.savefig(dir_plots + 'bootstrap_valid.png', bbox_inches='tight')
    plt.close("all")

    # summary

    writer = pd.ExcelWriter(dir_out, engine='xlsxwriter')

    # overview
    df_overview_param = pd.DataFrame(
        data=[str(valid_test.data_Dir), str(valid_test.fileName), valid_test.train_period, valid_test.valid_period,
              valid_test.performance_measure],
        index=['data_dir', 'file_name', 'train_period', 'valid_period', 'performance_measure'],
        columns=['value'])

    df_overview_param.index.names = ['parameter']
    df_overview_param.to_excel(writer, sheet_name='parameters')

    worksheet = writer.sheets['parameters']
    worksheet.insert_image('H2', dir_plots + 'series.png')
    worksheet.insert_image('H23', dir_plots + 'acf.png')

    # ks test

    description = "Computes the Kolmogorov-Smirnov statistic on 2 samples. \
    This is a two-sided test for the null hypothesis that 2 independent samples are drawn from the same continuous distribution."

    if ks_test.pValue < 1 - ks_test.CI:

        conclusion = 'Reject null hypothesis given p-value < 1 - CI'

    else:

        conclusion = 'Can not reject null hypothesis given p-value >= 1 - CI'

    df_ks_param = pd.DataFrame(
        [ks_test.CI, ks_test.critical_Val, ks_test.maxDif, ks_test.pValue, description, conclusion],
        index=['CI', 'critical_val', 'maxDif', 'pValue', 'description', 'conclusion'],
        columns=['value'])

    df_ks_param.index.names = ['parameter']

    df_ks_param.to_excel(writer, sheet_name='ks_test_parameters')

    df_cdf1 = pd.DataFrame(ks_test.cdf1)
    df_cdf1.columns = ['cdf_train']
    df_cdf1.to_excel(writer, sheet_name='ks_test_cdf_train')

    df_cdf2 = pd.DataFrame(ks_test.cdf2)
    df_cdf2.columns = ['cdf_valid']
    df_cdf2.to_excel(writer, sheet_name='ks_test_cdf_valid')

    # insert chart
    chart = writer.book.add_chart({'type': 'line'})
    _shape = df_cdf1.shape

    chart.add_series({
        'name': '=ks_test_cdf_train!$B$1',
        'categories': '=ks_test_cdf_train!$A$1:$A$%s' % _shape[0],
        'values': '=ks_test_cdf_train!$B$1:$B$%s' % (_shape[0]),
    })

    chart.add_series({
        'name': '=ks_test_cdf_valid!$B$1',
        'categories': '=ks_test_cdf_valid!$A$1:$A$%s' % _shape[0],
        'values': '=ks_test_cdf_valid!$B$1:$B$%s' % (_shape[0]),
    })

    chart.set_title({'name': 'KS Test'})
    chart.set_x_axis({'name': ''})
    chart.set_y_axis({'name': 'cdf'})
    chart.set_style(10)

    writer.sheets['ks_test_parameters'].insert_chart('H2', chart)

    # boostrap simulation test

    df_simula_param = pd.DataFrame([bootstrap_valid.perf_metric, bootstrap_valid.partition_size,
                                    bootstrap_valid.num_sample, bootstrap_valid.replacement],
                                   index=['perf_metric', 'partition_size', 'num_sample', 'replacement'],
                                   columns=['value'])

    df_simula_param.index.names = ['parameter']
    df_simula_param.to_excel(writer, sheet_name='bootstrap_test')

    worksheet = writer.sheets['bootstrap_test']
    worksheet.insert_image('H2', dir_plots + 'bootstrap_valid.png')

    writer.save()

    return


# endregion

# region Test function delete late

def test_mp(para):
    num_sample = 100
    num_cores = 2
    train_series = pd.Series(np.random.randn(100))
    partition_size = 10
    n_val = 20
    perf_metric_func = np.mean
    samples_per_core = math.ceil(num_sample / num_cores)

    results = []
    if para:
        with mp.Pool(num_cores) as pool:
            res = [pool.apply_async(get_samples_CB,
                                    args=(
                                    train_series.values, partition_size, n_val, perf_metric_func, samples_per_core))
                   for i in range(num_cores)]
            results = []
            for r in res:
                results.extend(r.get())
    else:
        [results.extend(get_samples_CB(train_series.values, partition_size, n_val, perf_metric_func, samples_per_core))
         for i in range(num_cores)]

    return results


# endregion

if __name__ == '__main__':
    print('Starting test...')
    # para_results = test_mp(True)
    # normal_results = test_mp(False)
    # print(len(para_results))
    # print(len(normal_results))
    # print(normal_results)

    data_dir = '/media/farmshare2/Research/kieran/01 live_strategies/06 review Jan 2020/01 full return series/'
    beta_dir = 'BETA.csv'
    bayes_dir = 'BAYES.csv'
    rsec_dir = 'RSEC.csv'
    strat_dict = {'BETA': beta_dir, 'BAYES': bayes_dir, 'RSEC': rsec_dir}
    Training_Period_Start = '2010-01-01'
    Training_Period_End = '2019-10-07'
    Validation_Period_Start = '2019-10-08'
    Validation_Period_End = '2020-01-14'
    performanceMeasure = 'Return WITH_COST'

    abs = BootstrapAdvanced()
    results_normal = abs.simulate_advanced("CB", False, 4, data_dir, 'BETA.csv', Training_Period_Start,
                                           Training_Period_End, Validation_Period_Start,
                                           Validation_Period_End, performanceMeasure, partition_size=5,
                                           # size of each partition
                                           num_sample=500, perf_metric='return_percent', replacement=True)
    results_para = abs.simulate_advanced("CB", True, 4, data_dir, 'BETA.csv', Training_Period_Start,
                                         Training_Period_End, Validation_Period_Start, Validation_Period_End,
                                         performanceMeasure, partition_size=5,  # size of each partition
                                         num_sample=500, perf_metric='return_percent', replacement=True)
    print(len(results_normal))
    print(len(results_para))
