import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import warnings
from .correlation import Correlation
from devModel.utilities import Utilities

class Eda():

    def __init__(self, data):
        self.data = data    
        self.config = {'threshold': {
                                    'skewness': 20,
                                    'cardinality': 50,
                                    'correlation': 0.9,
                                    'missing':0,
                                    'zeros':10,
                                    'constant_basic':100,
                                    'constant_large':0.9,
                                    'high_corr':0.5}
                        }

        self.data_description = None 
        self.features_description = None
        self.features_stats = None
        self.__correlation = Correlation()

    def summary(self):
        """
        Calculates the general characteristics of the dataset

        return:
        -----------
            self.description (Series): pd.Series with an overview of the dataset with the following fields
                                        - Number of Observations
                                         - Unique Values
                                         - Unique Values (%)
                                         - Missing Values (Num)
                                         - Missing Values (%)
                                         - Zeros Values (Num)
                                         - Zeros Values (%)
                                         - Most frequent value
                                         - Most Frequency
                                         - Less Frequent Value
                                         - Less Frequency
                                         - Memory Size
        """

        options = {"data":True,
                   "empty": True,
                   "instance": pd.DataFrame}
        
        Utilities._check(self.data, None, **options)

        summary = {"Number of features": self.data.shape[1],
                   "Number of observations": self.data.shape[0],
                   "Missing Values (Num)": self.get_missing_values(self.data),
                   "Missing Values (%)": round((self.get_missing_values(self.data) \
                                                / self.data.size)*100, 2),
                   "Duplicate rows (Num)": self.get_duplicates(self.data).shape[0],
                   "Duplicate rows (%)": round((self.get_duplicates(self.data)['count'].sum() \
                                                / self.data.size)*100, 2),
                   "Zeros values (Num)": self.get_zeros_values(self.data),
                   "Zeros values (%)": round((self.get_zeros_values(self.data) / self.data.size) * 100, 2), 
                   "Total size in memory (Kb)": self.data.memory_usage().sum()/1000,
                   "Average observations size in memory (B)": self.data.memory_usage().sum()/self.data.shape[0],
                   "Features numerical": self.data.select_dtypes(include=['number']).shape[1],
                   "Features Categorical": self.data.select_dtypes(include=["object", "category"]).shape[1],
                   "Features datetimes": self.data.select_dtypes(include=["datetime"]).shape[1]}

        self.data_description = pd.Series(summary)

        return self.data_description

    @staticmethod
    def get_duplicates(df: pd.DataFrame, columns = None):
        """
        Calculate the number of rows that are duplicates within the dataset

        args:
        -----------
            df (DataFrame) : dataset to be checked for duplicates
            columns (list) : List of features to consider when checking for duplicates
        
        return:
        -----------
            duplicates (DataFrame) : dataframe with the duplicate rows in the dataset    
        """
        duplicates = df[df.duplicated(subset=columns, keep=False)].groupby(
                        df.columns.values.tolist()).size().reset_index(name="count"). \
                        sort_values("count", ascending=False)
        
        return duplicates

    @staticmethod
    def get_missing_values(df: pd.DataFrame):
        """
        Calculate the number of missing values

        args:
        -----------
            df (DataFrame or Series): dataset that will be checked for missing values

        return:
        -----------
            (int): integer value with the number of missing values
        """
        if type(df) == type(pd.DataFrame()):
            return df.isnull().sum().sum()
        
        elif type(df) == type(pd.Series()):
            return df.isnull().sum()

    @staticmethod
    def get_zeros_values(df: pd.DataFrame):
        """
        Calculate the number of values equal to zero

        args:
        -----------
            df (DataFrame or Series): dataset that will be checked for values equal to zero

        return:
        -----------
            (int): integer value with the number of values equal to 0
        """
        return  df.size - np.count_nonzero(df.values)

    def statistics(self):
        """
        Calculates the statistics of each of the numerical features within the dataset.

        return:
        -----------
            (DataFrame): a dataframe with a statistical summary of each of the features with 
                         the following fields:
                        - Mean
                        - Standard Deviation
                        - Variance
                        - Coefficient of variation (CV)
                        - Sum
                        - Minimum Value
                        - Quantile - 0.25
                        - Median
                        - Quantile - 0.75
                        - Maximum Value
                        - Median Absolute Deviation (MAD)
                        - Interquartile range (IQR)
                        - Range
                        - Kurtosis
                        - skewness                         
        """
        features = self.data.select_dtypes(include="number").columns.values.tolist()
        statistics = {key: self.get_numeric_stats(self.data[key]) for key in features}    
        self.features_stats = pd.DataFrame(statistics)
        
        return self.features_stats

    def get_numeric_stats(self, serie):
        """
        Calculates the basic statistics of the specified feature

        args:
        -----------
            series (Series): data series to which the statistics will be calculated

        return:
        -----------
            numeric_stats (dict): statistics of the selected feature 
        """

        options = {"data":True,
                   "empty": True,
                   "instance": pd.Series,
                   "type": "numeric"}
        
        Utilities._check(serie, None, **options)

        numeric_stats = {'Mean': serie.mean(),
                         'Standard Deviation': serie.std(),
                         'Variance': serie.var(),
                         'Coefficient of variation (CV)': self.coeffv(serie),
                         'Sum': serie.sum(),
                         'Minimum Value': serie.min(),
                         'Quantile - 0.25': serie.quantile(q=0.25),
                         'Median': serie.median(),
                         'Quantile - 0.75': serie.quantile(q=0.75),
                         'Maximum Value': serie.max(),
                         'Median Absolute Deviation (MAD)': self.mad(serie),
                         'Interquartile range (IQR)': self.iqr(serie),
                         'Range': self.range(serie),
                         'Kurtosis': serie.kurt(),
                         'skewness': serie.skew()}
        
        return numeric_stats

    @staticmethod
    def coeffv(serie):
        """
        The coefficient of variation is a measure of the variability of the data around
        the mean. It is the relationship between the standard deviation and the mean of
        the data series. It is useful for comparing the degree of variation from one 
        data series to another, even if the means are very different from each other.
        
        args:
        -----------
            series (Series): data series to which the coefficient of variation will be calculated

        return:
        -----------
            (float): coefficient of variation   
        """
        return serie.std() / serie.mean()
    
    @staticmethod
    def mad(serie):
        """
        Finds the mean absolute deviation, which is a robust measure of variability. 
        Variance and standard deviation are also measures of dispersion, but are 
        more affected by extremely high or low values and non-normality. If the 
        data are normal, the standard deviation is usually the best option to 
        evaluate the differential. However, if your data is not normal, MAD is a 
        better option.

        args:
        -----------
            series (Series): data series to which the mad will be calculated

        return:
        -----------
            (float): median absolute deviation      
        """
        return np.median(np.abs(serie - np.median(serie)))

    @staticmethod
    def iqr(serie):
        """
        Calculates the interquartile range (IQR), which is a measure of the spread of
        the data distribution. It consists of the difference between the third and 
        first quartiles.

        args:
        -----------
            series (Series): data series to which the iqr will be calculated

        return:
        -----------
            (float): interquartile range            
        """
        return serie.quantile(q=0.75) - serie.quantile(q=0.25)
    
    @staticmethod
    def range(serie):
        """
        Calculates the range of the data distribution

        args:
        -----------
            series (Series): data series to which the range will be calculated

        return:
        -----------
            (float): data range            
        """
        return serie.max() - serie.min()

    def features_summary(self):
        """
        Calculates the general characteristics of the dataset for each feature

        return:
        -----------
            (DataFrame): characteristics of the features
        """   

        options = {"data": True,
                   "empty": True,
                   "instance": pd.DataFrame}
        
        Utilities._check(self.data, None, **options)

        features = self.data.columns.tolist()
        summary = dict()
        summary = {key: self.get_feature_summary(self.data[key]) for key in features}
        self.features_description = pd.DataFrame(summary)

        return self.features_description

    def get_feature_summary(self, serie):
        """
        Calculates the general characteristics of the data serie

        args:
        -----------
            series (Series): data series to which the characteristics will be calculated

        return:
        -----------
            summary (dict): characteristics of the selected feature
        """        

        options = {"data":True,
                   "empty": True,
                   "instance": pd.Series}

        Utilities._check(serie, None, **options)

        summary = {'Number of Observations': serie.count(),

                   'Missing Values (Num)': self.get_missing_values(serie),
                   'Missing Values (%)': round((self.get_missing_values(serie) / serie.size) * 100, 2),

                   'Unique Values': serie.unique().size,
                   'Unique Values (%)': (serie.unique().size / serie.count()) * 100,
                   
                   'Zeros Values (Num)': self.get_zeros_values(serie),
                   'Zeros Values (%)': round((self.get_zeros_values(serie) / serie.size) * 100, 2),

                   'Most Frequent Value': self.get_most_frequent_value(serie)[0],
                   'Most Frequency (Num)': self.get_most_frequent_value(serie)[1],
                   
                   'Less Frequent Value': self.get_less_frequent_value(serie)[0],
                   'Less Frequency (Num)': self.get_less_frequent_value(serie)[1],
                   'Memory Size (Kb)': serie.memory_usage() / 1000}
      
        return summary

    @staticmethod
    def get_most_frequent_value(serie):
        """
        Get the most frequent value and its frequency
        args:
        -----------
            series (Series): data series to which the most frequent value will be found

        return:
        -----------
            (tuple): most frequent value and frequency    
        """
        frequent_value = (serie.value_counts().reset_index().iloc[0, 0], \
                        serie.value_counts().reset_index().iloc[0, 1])
       
        return frequent_value
    
    @staticmethod
    def get_less_frequent_value(serie):
        """
        Get the less frequent value and its frequency
        args:
        -----------
            series (Series): data series to which the less frequent value will be found

        return:
        -----------
            (tuple): less frequent value and frequency    
        """
        frequent_value = (serie.value_counts().reset_index().iloc[-1, 0], \
                        serie.value_counts().reset_index().iloc[-1, 1])
        
        return frequent_value

    def warnings(self):
        """
        Get the issues that the data has globally and by feature.

        return:
        -----------
            (dict): all the issues of the data          
        """

        options = {"data":True,
                   "empty": True,
                   "attribute": {"features_description": "features_summary",
                                 "data_description": "summary", 
                                 "features_stats": "statistics"}}
        
        Utilities._check(self.data, self, **options)    
        
        alarms = dict() 
        features = self.features_description.columns.values.tolist()
        # duplicates rows
        if self.data_description['Duplicate rows (Num)'] > 0: 
            alarms['Duplicates'] = 'Dataset has {} ({}) duplicated rows'.format(self.data_description['Duplicate rows (Num)'],
                                                               self.data_description['Duplicate rows (%)'])
        # missing values - data
        if self.data_description['Missing Values (Num)'] > self.config['threshold']['missing']:
            alarms['Missing Values'] = 'Dataset has {} ({}) missing values'.format(self.data_description['Missing Values (Num)'],
                                                               self.data_description['Missing Values (%)'])
        
        correlation = self.features_correlations(method="pearson")

        for feature in features:
            # missing values
            if self.features_description.loc['Missing Values (%)', feature] > self.config['threshold']['missing']:
                alarms['Missing Values - '+ feature] = '{} has {} ({}) missing values'.format(feature,
                                                             self.features_description.loc['Missing Values (Num)', feature],
                                                             self.features_description.loc['Missing Values (%)', feature])
            # high cardinality
            if pd.api.types.is_categorical_dtype(self.features_description[feature]) or \
                pd.api.types.is_object_dtype(self.features_description[feature]):
                if self.features_description.loc['Unique Values', feature] > self.config['threshold']['cardinality']:
                    alarms['Cardinality - '+ feature] = '{} has a high cardinality: {}  distinct values'.format(feature,
                                                                self.features_description.loc['Unique Values', feature])

            if pd.api.types.is_numeric_dtype(self.features_description[feature]):            
                # zeros
                if self.features_description.loc['Zeros Values (%)', feature] > self.config['threshold']['zeros']:
                    alarms['Zeros - '+ feature] = '{} has {} ({}) zeros values'.format(feature,
                                                               self.features_description.loc['Zeros Values (Num)', feature],
                                                               self.features_description.loc['Zeros Values (%)', feature]) 
                # constant
                value = self.features_description[feature].value_counts().iloc[0]
                if value > self.config['threshold']['constant_basic']:
                    alarms['Constant - '+ feature] ='{} has a constant value {}'.format(feature, value)
                
                # high constant
                if value >= (self.features_description[feature].shape[0] * self.config['threshold']['constant_large']):
                    alarms['High constant - '+ feature] = '{} has more thant 90% as {}'.format(feature, value)

                # skewed 
                value = self.features_stats.loc['skewness', feature]
                if value > self.config['threshold']['skewness']:
                    alarms['Skewed - '+ feature] = '{} is highly skewed {( )}'.format(feature, value)

            # high correlation
                row = correlation.loc[feature].drop(feature)
                value = row[abs(row) > self.config['threshold']['high_corr']].index.tolist()
                if value:
                    alarms['High correlation - ' + feature] = '{} has a high correlation with  {}'.format(feature, value)

            # uniform 
        return alarms

    def features_correlations(self, method="pearson", cols=None, **kwargs):
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
        options = {"empty": True,
                   "instance": pd.DataFrame,
                   "Nan": True}
        
        Utilities._check(self.data, None, **options)

        return self.__correlation.get_correlation(self.data, method, cols, **kwargs)

