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
import pandas as pd

class Statistics:

    def __init__(self):
        pass

    def get_test(self, df, test='normality', alpha=0.01, cols=None, analysis_indv=False):
        """
        """
        # check type 

        if cols is not None:
            df = df[cols]

        df = df.select_dtypes('number').dropna().reset_index(drop=True)
        features = df.columns.tolist()

        if test == 'normality':

            tests = dict()
            analysis = dict()

            for feature in features:
                
                if analysis_indv is True:
                    tests[feature] = self.normality_test(df[feature], alpha)

                else:
                    norm_test = pd.DataFrame.from_dict(self.normality_test(df[feature], alpha), orient='index')
                    tests[feature] = {'Distribution': norm_test['Distribution'].value_counts().index[0],
                                    '# Number of tests / 5': norm_test['Distribution'].value_counts()[0],
                                    'Actions': 'Use Parametric Statistical Methods' if norm_test['Distribution'].value_counts().index[0] == \
                                    'Can be fitted to a normal dist.' else 'Use Nonparametric Statistical Methods'}

            return tests

    @staticmethod
    def normality_test(Serie, alpha=0.01):
        """
        """
        normality = dict()
        mean, std = norm.fit(Serie)
        stat_kstest, pvalue_kstest = kstest(Serie, "norm", args=(mean, std))
        normality['Kolmogorov-Smirnov'] = {'Distribution':  'Can be fitted to a normal dist.' if pvalue_kstest > alpha \
                                                                                    else 'Does not fit a normal dist.',
                                            'Critical and p-values': pvalue_kstest}

        stat_lilli, pvalue_lilli = lilliefors(Serie, "norm", pvalmethod='table')
        normality['Lilliefors'] = {'Distribution': 'Can be fitted to a normal dist.' if pvalue_lilli > alpha \
                                                                            else 'Does not fit a normal dist.',
                                    'Critical and p-values': pvalue_lilli}

        stat_dagpearson, pvalue_dagpearson = normaltest(Serie)
        normality['D’Agostino and Pearson’s'] =  {'Distribution': 'Can be fitted to a normal dist.' if pvalue_dagpearson > alpha \
                                                                                            else 'Does not fit a normal dist.',
                                                  'Critical and p-values': pvalue_dagpearson}

        stat_sha, pvalue_sha = shapiro(Serie)
        normality['Shapiro-Wilk'] = {'Distribution': 'Can be fitted to a normal dist.' if pvalue_sha > alpha \
                                                            else 'Does not fit a normal dist.',
                                    'Critical and p-values': pvalue_sha}
    
        stat_result = anderson(Serie)
        ande_value = {0.01: 4, 0.025: 3, 0.05: 2, 0.1: 1, 0.15: 0}
        normality['Anderson-Darling'] = {'Distribution': 'Can be fitted to a normal dist.' if stat_result.statistic < stat_result.critical_values[ande_value[alpha]] \
                                                        else 'Does not fit a normal dist.',
                                         'Critical and p-values': stat_result.critical_values[ande_value[alpha]]}        


        return normality

