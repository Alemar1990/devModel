import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from scipy.stats import chi2_contingency
from scipy.stats import entropy
from sklearn import preprocessing
from collections import Counter
import math
import phik
from devModel.utilities import Utilities

class Correlation:
    """
    """

    def __init__(self):
        pass

    def get_correlation(self, df, method="Pearson", cols=None, **kwargs):
        """
        It calculates the correlation between the different variables in the dataset. Correlation shows the 
        strength of a relationship between two variables and is expressed numerically by the correlation coefficient. 
        Two variables are associated when one variable gives us information about the other. On the other hand, when 
        there is no association, the increase or decrease of one variable tells us nothing about the behavior of the 
        other variable. Two variables are correlated when they show an increasing or decreasing trend.       
        args:
        -----------
            method (str): the method by which the correlation is calculated. These are:
                            - Pearson  -> continuous - continuous
                            - Kendall  -> continuous - continuous
                            - Spearman -> continuous - continuous
                            - CramerV  -> categorical - categorical
                            - TheilsU  -> categorical - categorical
                            - CRatio   -> categorical - continuous
                            - Phi      -> categorical - continuous
            cols (list): list of features to use from the dataset
            correction (bool): specifies whether to use skew correction in the CramerV method
            matrix (str): specifies what type of correlation matrix to obtain from the Phi method. These are:
                            - phik_matrix
                            - global_phik
                            - significance_matrix
            p_value (bool): Instead of calculating the correlation, calculate the P-value for the pearson, 
                            kendall, and spearman correlations

        return:
        -----------
            corr (DataFrame): Returns a dataframe with the correlation coefficients between the variables

        """

        kwargs_default = {"p_value": None,
                          "correction": True,
                          "matrix": 'local'}       
        options = Utilities.check_default_kwargs(kwargs_default, kwargs)

        function_call = "_Correlation__{}".format(method.lower())
        if hasattr(self, function_call):
            function_corr = getattr(self, function_call)
        
        if (cols is not None) and (len(cols) > 1):
            df = df[cols]
        elif (cols is not None) and (len(cols) == 1):
            raise ValueError("cols has to be greater than one")
        
        #df = df.dropna().reset_index(drop=True)
        return function_corr(df, **options)
    
    def __pearson(self, df, **kwargs):
        """
        Pearson's correlation coefficient is a measure of linear dependence between two quantitative 
        random variables. If the association between the items is not linear, then the coefficient is 
        not adequately represented. The correlation coefficient can take a range of values from +1 to -1. 
        A value of 0 indicates that there is no association between the two variables. A value greater 
        than 0 indicates a positive association. That is, as the value of one variable increases, so does 
        the value of the other. A value less than 0 indicates a negative association. That is, as the 
        value of one variable increases, the value of the other variable decreases.
        args:
        -----------
            df (DataFrame): dataset to which correlation will be calculated
            p_value (bool): instead of calculating the correlation, it calculates the P-value which is the 
                            probability that the correlation between these two variables is statistically significant. 
                            Normally, a significance level of 0.05 is selected, which means 95% confidence that the 
                            correlation between the variables is significant.
                            By convention when:
                            the p-value is < 0.001: there is strong evidence that the correlation is significant 
                            the p-value is < 0.05: there is moderate evidence that the correlation is significant 
                            the p-value is < 0.1: there is weak evidence that the correlation is significant 
                            p-value is > 0.1: there is no evidence that the correlation is significant
        
        return:
        -----------
            corr (DataFrame): returns a dataframe with the correlations of the variables between [-1 to 1]
        """
        features =  df.select_dtypes(include=['number']).columns.tolist()
        df = df[features].dropna().reset_index(drop=True)

        if kwargs['p_value']:
            rows =[]
            for feature1 in features:
                col = []
                for feature2 in features:
                    _, p_value = pearsonr(df[feature1], df[feature2])
                    col.append(p_value)
                rows.append(col) 

            p_values = np.array(rows)
            corr = pd.DataFrame(p_values, columns=features, index=features)
        else:
            corr = df.corr('pearson') 

        return corr

    def __spearman(self, df, **kwargs):
        """
        In statistics, Spearman's rank correlation coefficient, or Spearman's ρ, is a nonparametric 
        measure of rank correlation (statistical dependence between the rankings of two variables). 
        It evaluates how well the relationship between two variables can be described using a monotonic 
        function.
        args:
        -----------
            df (DataFrame): dataset to which correlation will be calculated
            p_value (bool): instead of calculating the correlation, it calculates the P-value which is the 
                            probability that the correlation between these two variables is statistically significant. 
                            Normally, a significance level of 0.05 is selected, which means 95% confidence that the 
                            correlation between the variables is significant.
                            By convention when:
                            the p-value is < 0.001: there is strong evidence that the correlation is significant 
                            the p-value is < 0.05: there is moderate evidence that the correlation is significant 
                            the p-value is < 0.1: there is weak evidence that the correlation is significant 
                            p-value is > 0.1: there is no evidence that the correlation is significant
        
        return:
        -----------
            corr (DataFrame): returns a dataframe with the correlations of the variables between [-1 to 1]        
        """

        features =  df.select_dtypes(include=['number']).columns.tolist()
        df = df[features].dropna().reset_index(drop=True)

        if kwargs['p_value']:
            rows =[]
            for feature1 in features:
                col = []
                for feature2 in features:
                    _, p_value = spearmanr(df[feature1], df[feature2])
                    col.append(p_value)
                rows.append(col) 

            p_values = np.array(rows)
            corr = pd.DataFrame(p_values, columns=features, index=features)
        else:
            corr = df.corr('spearman') 

        return corr
            
    def __kendall(self, df, **kwargs):
        """
        Kendall's rank correlation coefficient , commonly known as Kendall's τ coefficient        
        (after the Greek letter τ , tau), is a statistic used to measure the ordinal association 
        between two measured quantities.
        More information: https://es.qaz.wiki/wiki/Kendall_rank_correlation_coefficient
                          https://economipedia.com/definiciones/tau-de-kendall-i.html
                          https://economipedia.com/definiciones/tau-de-kendall-ii.html
                          https://towardsdatascience.com/kendall-rank-correlation-explained-dee01d99c535
        args:
        -----------
            df (DataFrame): dataset to which correlation will be calculated
            p_value (bool): instead of calculating the correlation, it calculates the P-value which is the 
                            probability that the correlation between these two variables is statistically significant. 
                            Normally, a significance level of 0.05 is selected, which means 95% confidence that the 
                            correlation between the variables is significant.
                            By convention when:
                            the p-value is < 0.001: there is strong evidence that the correlation is significant 
                            the p-value is < 0.05: there is moderate evidence that the correlation is significant 
                            the p-value is < 0.1: there is weak evidence that the correlation is significant 
                            p-value is > 0.1: there is no evidence that the correlation is significant
        
        return:
        -----------
            corr (DataFrame): returns a dataframe with the correlations of the variables between [-1 to 1]             
        """

        features =  df.select_dtypes(include=['number']).columns.tolist()
        df = df[features].dropna().reset_index(drop=True)

        if kwargs['p_value']:
            rows =[]
            for feature1 in features:
                col = []
                for feature2 in features:
                    _, p_value = kendalltau(df[feature1], df[feature2])
                    col.append(p_value)
                rows.append(col) 

            p_values = np.array(rows)
            corr = pd.DataFrame(p_values, columns=features, index=features)
        else:
            corr = df.corr('kendall') 

        return corr
    
    def __cramerv(self, df, **kwargs):
        """
        Calculates the Cramer's V statistical test for the categorical-categorical association 
        (specifically nominal variables) for all the features, obtaining a value between 0-1. Where 0 indicates 
        independence and 1 indicates perfect association.
        args:
        -----------
            df (DataFrame): dataset to which correlation will be calculated
            correction (bool): Uses Bergsma and Wicher bias correction - default True  
                               (Journal of the Korean Statistical Society 42 (2013): 323-328)  
        return:
        -----------
            corr (DataFrame): returns a dataframe with the correlations of the variables between [0 to 1]                            
        """

        label = preprocessing.LabelEncoder()
        data_encoded = pd.DataFrame()
        features = df.select_dtypes(include=['category', 'object']).columns.tolist()
        df = df[features].dropna().reset_index(drop=True)
        #check if all values of a feature are differents (High Cardinality), that way is eliminated from features******

        for feature in features:
            data_encoded[feature] = label.fit_transform(df[feature])

        rows = []
        for feature1 in data_encoded:
            col = []
            for feature2 in data_encoded:
                cramers = self.cramersv_correlation(data_encoded[feature1], 
                                                    data_encoded[feature2], correction=kwargs['correction'])
                col.append(round(cramers,2))
            rows.append(col)

        cramersV_results = np.array(rows)
        corr = pd.DataFrame(cramersV_results, columns=features, index=features)

        return corr

    @staticmethod
    def cramersv_correlation(feature1, feature2, correction=True):
        """
        Calculates the Cramer's V statistical test for the categorical-categorical association 
        (specifically nominal variables) for the two specified features, obtaining a value between 0-1. Where 0 indicates 
        independence and 1 indicates perfect association.        
        args:
        -----------
            feature1 (Series): data series - categorical 
            feature2 (Series): data series - categorical
            correction (bool): uses Bergsma and Wicher bias correction - default True  
                               (Journal of the Korean Statistical Society 42 (2013): 323-328)  
        return:
        -----------
            corr (float): correlation between the two specified features, between [0-1].   
        """

        crosstab = np.array(pd.crosstab(feature1, feature2, rownames=None, colnames=None))
        chi2 = chi2_contingency(crosstab)[0] 
        n = crosstab.sum().sum()
        phi2 = chi2 / n
        r, k = crosstab.shape

        if correction:
            with np.errstate(divide="warn", invalid="warn"):
                phi2corr = max(0.0, phi2 - ((k - 1.0) * (r - 1.0)) / (n - 1.0))
                rcorr = r - ((r - 1.0) ** 2.0) / (n - 1.0)
                kcorr = k - ((k - 1.0) ** 2.0) / (n - 1.0)
                corr = np.sqrt(phi2corr / min((kcorr - 1.0), (rcorr - 1.0)))  
        else:
            corr = np.sqrt(phi2 / min((k - 1.0), (r - 1.0)))

        return corr              

    def __theilsu(self, df, **kwargs):
        """
        Calculates Theil's U statistic, also known as the uncertainty coefficient for categorical 
        variables in a dataset. Formally labeled U (x | y), this coefficient provides a value in 
        the range [0,1], where 0 means that variable "Y" provides no information about variable "x", 
        and 1 means that characteristic "Y" provides complete information about variable "X". In this 
        case var1 = "X" and var2 ="Y". This is an asymmetric coefficient U(x,y) != U(y,x).
        More information: https://en.wikipedia.org/wiki/Uncertainty_coefficient
        args:
        -----------
            df (DataFrame): dataset to which correlation will be calculated
        return:
        -----------
            corr (DataFrame): returns a dataframe with the correlations of the variables between [0 to 1]                   
        """

        rows = []
        features = df.select_dtypes(include=['category', 'object']).columns.tolist()
        df = df[features].dropna().reset_index(drop=True)

        for feature1 in features:
            col = []
            for feature2 in features:
                theilsu = self.theilsu_correlation(df[feature1], df[feature2])
                theilsu = theilsu if not np.isnan(theilsu) and abs(theilsu) < np.inf else 0.0
                col.append(round(theilsu, 2))
            rows.append(col)
        
        theilsu_results = np.array(rows)
        corr = pd.DataFrame(theilsu_results, columns=features, index=features)

        return corr

    def theilsu_correlation(self, feature1, feature2):
        """
        Calculates Theil's U statistic, also known as the uncertainty coefficient for the two categorical 
        variables specified. Formally labeled U (x | y), this coefficient provides a value in the range [0,1], 
        where 0 means that variable "Y" provides no information about variable "x", and 1 means that 
        characteristic "Y" provides complete information about variable "X". In this case var1 = "X" 
        and var2 ="Y". This is an asymmetric coefficient U(x,y) != U(y,x)      
        args:
        -----------
            df (DataFrame): dataset to which correlation will be calculated
        return:
        -----------
            corr (float): correlation between the two specified features between the range of [0-1].                 
        """ 

        s_feature_1_2 = self.conditional_entropy(feature1, feature2)
        feature1_counter = Counter(feature1)
        total_occurences = sum(feature1_counter.values())
        p_feature1 = list(map(lambda x: x / total_occurences, feature1_counter.values()))
        s_feature1 = entropy(p_feature1)
        theils_u = 1.0 if s_feature1 == 0 else (s_feature1 - s_feature_1_2) / s_feature1

        return theils_u

    @staticmethod
    def conditional_entropy(x, y):
        """
        Conditional entropy quantifies the amount of information needed to describe the outcome 
        of a random variable "Y" given that the value of another random variable "X" is known.
        In this case the entropy of X conditional on Y is calculated as H(X|Y).
        More information: https://en.wikipedia.org/wiki/Conditional_entropy
        args:
        -----------
            x (Series): data series - categorical 
            Y (Series): data series - categorical   
        return:
        -----------   
            entropy (float): the entropy of X conditioned by Y                       
        """

        y_counter = Counter(y)
        xy_counter = Counter(list(zip(x, y)))
        total_occurrences = sum(y_counter.values())
        entropy = 0
        for xy in xy_counter.keys():
            p_xy = xy_counter[xy] / total_occurrences
            p_y = y_counter[xy[1]] / total_occurrences
            entropy += p_xy * math.log(p_y / p_xy)

        return entropy

    def __cratio(self, df):
        """
        Calculates the correlation ratio for a dataset, which is a measure of the relationship 
        between the statistical dispersion within individual categories and the dispersion in the 
        entire population or sample. In other words, it calculates the correlation ratio for a 
        categorical-continuous association. This coefficient provides a value in the range of [0,1], 
        where 0 means that a category cannot be determined by a continuous measurement.  

        The correlation ratio answers the following question: Given a continuous number, how well 
        can you know to which category it belongs?
        More information: https://en.wikipedia.org/wiki/Correlation_ratio
        args:
        -----------   
            df (DataFrame): dataset to which correlation will be calculated
        return:
        ----------- 
            corr (DataFrame): returns a dataframe with the correlations of the variables between [0 to 1]      
        """

        categorical = df.select_dtypes(include=["category", "object"]).dropna().reset_index(drop=True)
        numerical = df.select_dtypes(include="number").dropna().reset_index(drop=True)

        if categorical.empty:
            raise ValueError("There is not category or object features")
        if numerical.empty:
            raise ValueError("There is not numeric features")

        rows = []
        for feature1 in categorical.columns:
            col = []
            for feature2 in numerical.columns: 
                cratio = self.correlation_ratio(df[feature1], df[feature2])
                col.append(round(cratio, 2))
            rows.append(col)
        
        cratio_results = np.array(rows)
        corr = pd.DataFrame(cratio_results, columns= numerical.columns, index=categorical.columns)
        
        return corr

    @staticmethod
    def correlation_ratio(categories, measurements):
        """
        Calculates the correlation ratio between two features, which is a measure of the relationship 
        between the statistical dispersion within individual categories and the dispersion in the 
        entire population or sample. In other words, it calculates the correlation ratio for a 
        categorical-continuous association. This coefficient provides a value in the range of [0,1], 
        where 0 means that a category cannot be determined by a continuous measurement.  

        The correlation ratio answers the following question: Given a continuous number, how well 
        can you know to which category it belongs?
        More information: https://en.wikipedia.org/wiki/Correlation_ratio      
        args:
        -----------   
            categories (Series): data series - categorical 
            measurements (Series): data series - continuous
        return:
        ----------- 
            corr (float): coeficiente de correlación entre el rango de [0-1]         
        """
        fcat, _ = pd.factorize(categories)
        cat_num = np.max(fcat) + 1
        y_avg_array = np.zeros(cat_num)
        n_array = np.zeros(cat_num)
        for i in range(0, cat_num):
            cat_measures = measurements[np.argwhere(fcat == i).flatten()]
            n_array[i] = len(cat_measures)
            y_avg_array[i] = np.average(cat_measures)
        y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
        numerator = np.sum(np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2)))
        denominator = np.sum(np.power(np.subtract(measurements, y_total_avg), 2))
        corr = 0.0 if numerator == 0 else np.sqrt(numerator / denominator)

        return corr

    def __phik(self, df, **kwargs):
        """
        The Phi_K correlation is based on several refinements of Pearson's hypothesis test of 
        independence of two variables. It works between categorical and continuous variables 
        (interval, ordinal and categorical). It also captures non-linear dependence. This coefficient 
        provides a value in the range of [0,1], where 0 means no association and 1 means complete association.
        args:
        -----------  
            matrix (str): specifies which type of matrix of interest to request. These matrices are:
                            - phik_matrix:
                              Calculates the phi correlation coefficients between all variables.   
                            - global_phik:
                              The global correlation coefficient is a measure of the total correlation of a 
                              variable with all other variables in the data set. They give an indication of 
                              how well a variable can be modeled in terms of the other variables.    
                            - significance_matrix:
                              Evaluates the significance level of the correlation by Z-score. A large correlation 
                              may be statistically insignificant and vice versa, a small correlation may be highly 
                              significant. Low values turn out to be statistically insignificant, but higher values 
                              are highly significant.  
                              z-score (Standard Deviations)	| p-value (Probability) | Confidence level
                                    < -1.65 or > +1.65             < 0.10                    90%
                                    < -1.96 or > +1.96             < 0.05                    95%
                                    < -2.58 or > +2.58             < 0.01                    99%

                              More information: https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-statistics/what-is-a-z-score-what-is-a-p-value.htm
        """
        df = df.dropna().reset_index(drop=True)
        interval_cols = dict()
        for feature in df.select_dtypes(include='number').columns.tolist():
            interval_cols[feature] = interval_cols.get(feature, 'interval')  

        if kwargs['matrix']=='local': 
            corr = df.phik_matrix(interval_cols=interval_cols)
        elif kwargs['matrix'] == 'global':
            corr = df.global_phik(interval_cols=interval_cols)
        elif kwargs['matrix'] == 'z score':
            corr = df.significance_matrix(interval_cols=interval_cols)    
        
        return corr
        


