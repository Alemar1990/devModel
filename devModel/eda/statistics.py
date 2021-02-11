import pandas as pd
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

class Statistics:

    def __init__(self):
        pass

    def get_test(self, df, test='normality', alpha=0.01, cols=None, analysis_indv=False, group=None, subgroups=None, feature_interest=None):
        """
        """
        # check type 

        if cols is not None:
            df = df[cols]
        
        if group is None:

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

            elif test in ['anova', 'ttest', 'p-ttest', 'kruskal-wallis', 'mann-whitney', 'wilcoxon', 'friedman']:
                pass
        
        #group is not None        
        else:
            df = df.groupby(group)
            features = df.groups.keys()     

            if (subgroups is not None) and (set(subgroups).issubset(set(features))):
                features = subgroups

            if test == 'normality':

                tests = dict()
                analysis = dict()

                for feature in features:
                
                    if analysis_indv is True:
                        tests[feature] = self.normality_test(df.get_gropup[feature][feature_interest], alpha)

                    else:
                        norm_test = pd.DataFrame.from_dict(self.normality_test(df.get_gropup[feature][feature_interest], alpha), orient='index')
                        tests[feature] = {'Distribution': norm_test['Distribution'].value_counts().index[0],
                                        '# Number of tests / 5': norm_test['Distribution'].value_counts()[0],
                                        'Actions': 'Use Parametric Statistical Methods' if norm_test['Distribution'].value_counts().index[0] == \
                                        'Can be fitted to a normal dist.' else 'Use Nonparametric Statistical Methods'}

            elif test in ['anova', 'ttest', 'p-ttest', 'kruskal-wallis', 'mann-whitney', 'wilcoxon', 'friedman']:
                
                args = []
                for feature in features:
                    args.append(df.get_group(feature)[feature_interest])

                tests = self.distribution_comparison_test(test, alpha, *args)        

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

    @staticmethod
    def distribution_comparison_test(test='anova', alpha=0.01, *args):
        """
        """
        test_result=dict()
        if test == 'anova':
            stat_val, p_val = f_oneway(*args)
            result = ['Same distribution' if p_val > alpha else 'Different distribution']
            key = "ANOVA test results"

        elif test == 'ttest':
            stat_val, p_val = ttest_ind(*args)
            result = ["Same distribution" if p_val > alpha else "Different distribution"]
            key = "Student's t-Test results"

        elif test == 'p-ttest':
            stat_val, p_val = ttest_rel(*args)
            result = ['Same distribution' if p_val > alpha else 'Different distribution']
            key = "Paired Student’s t-Test results"

        elif test == 'kruskal-wallis':
            stat_val, p_val = kruskal(*args)
            result = ['Same distribution' if p_val > alpha else 'Different distribution']
            key = "Kruskal-Wallis test results"

        elif test == 'mann-whitney':
            stat_val, p_val = mannwhitneyu(*args)
            result = ['Same distribution' if p_val > alpha else 'Differente distribution']
            key = "Mann-Whitney test results"

        elif test == 'wilcoxon':
            stat_val, p_val = wilcoxon(*args)
            result = ['Same distribution' if p_val > alpha else 'Different distribution']
            key = "Mann-Whitney test results"

        elif test == 'friedman':
            stat_val, p_val = friedmanchisquare(*args)
            result = ['Same distribution' if p_val > alpha else 'Different distribution']
            key = "Friedman Test results"             
        
        test_result[key] = {"Stat": stat_val, "P": p_val, "Result": result}

        return   test_result

if __name__ == '__main__':
    path='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
    df = pd.read_csv(path).dropna()    
    a = Statistics()
    a.get_test(df, test='normality', alpha=0.01, cols=['drive-wheels', 'price'], analysis_indv=False, group='drive-wheels', subgroups=None, feature_interest='Price')