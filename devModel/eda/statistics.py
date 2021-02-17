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
from devModel.utilities import Utilities

class Statistics:

    def __init__(self):
        pass

    def get_test(self, df, test='normality', alpha=0.01, cols=None, **kwargs):
        """
        The statistical tests provide a mechanism for making quantitative decisions about a process or processes. 
        The intent is to determine whether there is enough evidence to "reject" a conjecture or hypothesis about 
        the process. The conjecture is called the null hypothesis. Not rejecting may be a good result if we want 
        to continue to act as if we "believe" the null hypothesis is true. Or it may be a disappointing result, 
        possibly indicating we may not yet have enough data to "prove" something by rejecting the null hypothesis.
        
        Statistics is a collection of tools that you can use to get answers to important questions about data. 
        args:
        -----------
            test (srt): statistical tests to apply to data. These are:
                        - normality -> test to check if the data is normal. The tests applied are: 
                                       Kolmogorov-Smirnov, Lilliefors, 'D’Agostino and Pearson’s, Shapiro-Wilk and 
                                       Anderson-Darling
                        - anova -> test to check if distributions are the same (Parametric Statistical Method)
                        - ttest -> test to check if distributions are the same (Parametric Statistical Method)
                        - p-ttest -> test to check if distributions are the same (Parametric Statistical Method)
                        - kruskal-wallis -> test to check if distributions are the same (Nonparametric Statistical Method)
                        - mann-whitney -> test to check if distributions are the same (Nonparametric Statistical Method)
                        - wilcoxon -> test to check if distributions are the same (Nonparametric Statistical Method)
                        - friedman -> test to check if distributions are the same (Nonparametric Statistical Method)
            alpha (float): significance level to be established as limit
            cols (list): list of features to use from the dataset
            analysis_indv (bool): specifies whether you want the output of all normality tests for each feature
            group (str): specify under which feature you want to group for the distribution comparison test (categorical feature)
            subgroups (list): specifies which subgroups of the selected group are desired for the analysis
            feature_interest (str): feature to use for the analysis of comparison of distributions under the selected group

        return:
        -----------
            (dict): selected test result        
        """

        kwargs_default = {"analysis_indv": False,
                          "group": None,
                          "subgroups":  None,
                          "feature_interest": None} 

        options = Utilities.check_default_kwargs(kwargs_default, kwargs)

        if cols is not None:
            df = df[cols]
        
        if options["group"] is None:

            df = df.select_dtypes('number').dropna().reset_index(drop=True)
            features = df.columns.tolist()

            if test == 'normality':

                tests = dict()
                analysis = dict()

                for feature in features:
                
                    if options["analysis_indv"] is True:
                        tests[feature] = self.normality_test(df[feature], alpha)

                    else:
                        norm_test = pd.DataFrame.from_dict(self.normality_test(df[feature], alpha), orient='index')
                        tests[feature] = {'Distribution': norm_test['Distribution'].value_counts().index[0],
                                        '# Number of tests / 5': norm_test['Distribution'].value_counts()[0],
                                        'Actions': 'Use Parametric Statistical Methods' if norm_test['Distribution'].value_counts().index[0] == \
                                        'Can be fitted to a normal dist.' else 'Use Nonparametric Statistical Methods'}

            elif test in ['anova', 'ttest', 'p-ttest', 'kruskal-wallis', 'mann-whitney', 'wilcoxon', 'friedman']:
                
                args = []
                for feature in features:
                    args.append(df[feature])

                tests = self.distribution_comparison_test(test, alpha, *args)   
        
        #group is not None        
        else:
            df = df.groupby(options["group"])
            features = df.groups.keys()     

            if (options["subgroups"] is not None) and (set(options["subgroups"]).issubset(set(features))):
                features = options["subgroups"]

            if test == 'normality':

                tests = dict()
                analysis = dict()

                for feature in features:
                
                    if options["analysis_indv"] is True:
                        tests[feature] = self.normality_test(df.get_gropup[feature][options["feature_interest"]], alpha)

                    else:
                        norm_test = pd.DataFrame.from_dict(self.normality_test(df.get_gropup[feature][options["feature_interest"]], alpha), orient='index')
                        tests[feature] = {'Distribution': norm_test['Distribution'].value_counts().index[0],
                                        '# Number of tests / 5': norm_test['Distribution'].value_counts()[0],
                                        'Actions': 'Use Parametric Statistical Methods' if norm_test['Distribution'].value_counts().index[0] == \
                                        'Can be fitted to a normal dist.' else 'Use Nonparametric Statistical Methods'}

            elif test in ['anova', 'ttest', 'p-ttest', 'kruskal-wallis', 'mann-whitney', 'wilcoxon', 'friedman']:
                
                args = []
                for feature in features:
                    args.append(df.get_group(feature)[options["feature_interest"]])

                tests = self.distribution_comparison_test(test, alpha, *args)        

        return tests

    @staticmethod
    def normality_test(serie, alpha=0.01):
        """
        There are a large number of statistical tests that can be used to quantify whether a data set appears to have a 
        normal (Gaussian) distribution. Each test makes different assumptions and considers different aspects of the data.

        Each test returns at least two parameters which are:
            - A statistic: a value calculated by the test that is interpreted in the context of the test by comparing it to 
                           the critical values of the distribution of the test statistic. 
            - p-value: used to interpret the test, in this case if the sample was extracted from a Gaussian distribution.

        To interpret the p-value, the following hypothesis must be established:
        Hypothesis:
            H0: the data can be fitted to a normal distribution.
            H1: the data are not normally distributed
        
        Significance Level (alpha) = 1 - Confidence Level/100 thus for 99% is 0.01    

        P-value:
            If p-value is > at the significance level, null hypothesis is not rejected (Normal).
            If the p-value is <= at the significance level, the null hypothesis is rejected (Non-Normal).
        
        Critical Value:
            If Test Statistic < Critical Value, null hypothesis is not rejected (Normal).
            If Test Statistic >= Critical Value, null hypothesis is rejected (Not normal).

        Important:
            A result above the significance level does not mean that the null hypothesis is true. It means that it is very likely 
            to be true given the available evidence. The p-value is not the probability that the data will fit a Gaussian distribution. 
            It can be considered as a value that helps us to interpret the statistical test.

        Tests used:
             - Kolmogorov-Smirnov test
             - Lilliefors' test
             - D’Agostino and Pearson’s test
             - Shapiro-Wilk test
             - Anderson-Darling test -> This differs a bit, as it checks with a critical value and statistics    

        args:
        -----------
            serie (Series): data series to which test will be applied
            alpha (float): significance level to be established as limit

        return:
        -----------
            (dict): test result              
        """
        normality = dict()
        mean, std = norm.fit(serie)
        stat_kstest, pvalue_kstest = kstest(serie, "norm", args=(mean, std))
        normality['Kolmogorov-Smirnov'] = {'Distribution':  'Can be fitted to a normal dist.' if pvalue_kstest > alpha \
                                                                                    else 'Does not fit a normal dist.',
                                            'Critical and p-values': pvalue_kstest}

        stat_lilli, pvalue_lilli = lilliefors(serie, "norm", pvalmethod='table')
        normality['Lilliefors'] = {'Distribution': 'Can be fitted to a normal dist.' if pvalue_lilli > alpha \
                                                                            else 'Does not fit a normal dist.',
                                    'Critical and p-values': pvalue_lilli}

        stat_dagpearson, pvalue_dagpearson = normaltest(serie)
        normality['D’Agostino and Pearson’s'] =  {'Distribution': 'Can be fitted to a normal dist.' if pvalue_dagpearson > alpha \
                                                                                            else 'Does not fit a normal dist.',
                                                  'Critical and p-values': pvalue_dagpearson}

        stat_sha, pvalue_sha = shapiro(serie)
        normality['Shapiro-Wilk'] = {'Distribution': 'Can be fitted to a normal dist.' if pvalue_sha > alpha \
                                                            else 'Does not fit a normal dist.',
                                    'Critical and p-values': pvalue_sha}
    
        stat_result = anderson(serie)
        ande_value = {0.01: 4, 0.025: 3, 0.05: 2, 0.1: 1, 0.15: 0}
        normality['Anderson-Darling'] = {'Distribution': 'Can be fitted to a normal dist.' if stat_result.statistic < stat_result.critical_values[ande_value[alpha]] \
                                                        else 'Does not fit a normal dist.',
                                         'Critical and p-values': stat_result.critical_values[ande_value[alpha]]}        


        return normality

    @staticmethod
    def distribution_comparison_test(test='anova', alpha=0.01, *args):
        """
        Statistical tests to check whether the distribution of one set of data is different from another set of data, 
        the following tests are performed.

        Parametric Test (Data for which the distribution is known, usually a Gaussian distribution)  
            Student's t-Test: Compares the mean of two independent data sets.
            Paired Student's t-Test: Compares the mean of two sets of data that are not independent.
            ANOVA test: Compares the mean of more than two independent data sets. 

        Non-Parametric Tests (Data that do not fit a known or well-understood distribution) 
            Mann-Whitney U Test: Compares independent data samples. It is the non-parametric version of Student's t-test.
            Kruskal-Wallis H and Friedman tests: Compares more than two data samples. It is the non-parametric version of ANOVA.
            Wilcoxon signed-rank test: Compares dependent data samples. It is the nonparametric version of the paired Student's t-test.
            Friedman Test: Compares more than two dependent data samples. It is the nonparametric version of the Repeated Measures ANOVA.

        Hypothesis:
            H0: The distributions of the data sets are equal.
            H1: The means of the data sets are different and possibly their distributions are not equal.

        Significance Level (alpha) = 1 - Confidence Level/100 thus for 99% is 0.01

        If the p-value is > at the significance level, the null hypothesis is not rejected. 
        If the p-value is <= at the significance level, the null hypothesis is rejected.  

        args:
        -----------
            test (srt): statistical tests to apply to data. 
                        - anova -> test to check if distributions are the same (Parametric Statistical Method)
                        - ttest -> test to check if distributions are the same (Parametric Statistical Method)
                        - p-ttest -> test to check if distributions are the same (Parametric Statistical Method)
                        - kruskal-wallis -> test to check if distributions are the same (Nonparametric Statistical Method)
                        - mann-whitney -> test to check if distributions are the same (Nonparametric Statistical Method)
                        - wilcoxon -> test to check if distributions are the same (Nonparametric Statistical Method)
                        - friedman -> test to check if distributions are the same (Nonparametric Statistical Method)
            alpha (float): significance level to be established as limit
            args (Serie): data series to which the statistical tests will be applied  
        return:
        -----------
            (dict): test result  

        """
        test_result=dict()
        # Parametric Statistical Methods
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
        
        # Nonparametric Statistical Methods
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