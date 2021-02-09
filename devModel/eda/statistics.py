from statsmodels.stats.diagnostic import lilliefors
from scipy.stats import shapiro
from scipy.stats import kstest
from scipy.stats import normaltest
from scipy.stats import anderson
from scipy.stats import norm
from scipy.stats import f_oneway
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
from scipy.stats import kruskal 
from scipy.stats import mannwhitneyu 
from scipy.stats import wilcoxon 
from scipy.stats import friedmanchisquare

class statistics:

    def __init__(self):
        pass

    def get_test(self, df, test='normality', alpha=0.01):
        """
        """
        # check type 

        if test == 'normality':
            df = df.select_dtypes('number')
            features = df.columns.tolist()

            for feature in features:
                norm_test = ''
    
    @staticmethod
    def normality_test(df, alpha=0.01):
        """
        """
        normality = dict()
        mean, std = norm.fit(df)
        stat_kstest, pvalue_kstest = kstest(df, "norm", args=(mean, std))
        normality = {'Kolmogorov-Smirnov': {'Distribution':  'Can be fitted to a normal dist.' if pvalue_kstest > alpha \
                                                                                    else 'Does not fit a normal dist.',
                                            'P values': pvalue_kstest}}

        stat_lilli, pvalue_lilli = lilliefors(df, "norm", pvalmethod='table')
        normality = {'Lilliefors': {'Distribution': 'Can be fitted to a normal dist.' if pvalue_lilli > alpha \
                                                                            else 'Does not fit a normal dist.',
                                    'P values': pvalue_lilli}}

        stat_dagpearson, pvalue_dagpearson = normaltest(df)
        normality = {'D’Agostino and Pearson’s': {'Distribution': 'Can be fitted to a normal dist.' if pvalue_dagpearson > alpha \
                                                                                            else 'Does not fit a normal dist.',
                                                  'P values': pvalue_dagpearson}}

        stat_sha, pvalue_sha = shapiro(df)
        normality = {'Shapiro-Wilk': {'Distribution': 'Can be fitted to a normal dist.' if pvalue_sha > alpha \
                                                            else 'Does not fit a normal dist.',
                                      'P values': pvalue_sha}}
    
        stat_result = anderson(df)
        ande_value = {0.01: 4, 0.025: 3, 0.05: 2, 0.1: 1, 0.15: 0}
        normality = {'Anderson-Darling': {'Distribution': 'Can be fitted to a normal dist.' if stat_result.statistic < stat_result.critical_values[ande_value[alpha]] \
                                                        else 'Does not fit a normal dist.',
                                         'P values': stat_result.critical_values[ande_value[alpha]]}}

        