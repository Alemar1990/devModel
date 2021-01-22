import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import warnings

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
                                    'constant_large':0.9
                                    }
                        }
        self.features = None
        self.description = None

    def _check(self, data, **kwargs):
        """
        Check the input conditions for functions
        
        args:
        -----------
            data: dataset to check 
            kwargs: conditions for the functions 
        
        return:
        -----------
        """
        kwargs_default = {"empty": None,
                          "instance": None,
                          "type": None}

        options = {key: kwargs[key] if key in kwargs.keys() else kwargs_default[key] for key in kwargs_default.keys()}
        
        if options["empty"]:
            self._check_empty(data)

        if options["instance"]:
            self._check_instance(data, inst=options["instance"])
        
        if options["type"]:
            self._check_type(data, type=options["type"])

    @staticmethod
    def _check_empty(data):
        """
        Check if the data variable is empty
        
        args:
        -----------
            data (DataFrame):  dataset to check
        return:
        -----------
            (boolen): boolean variable specifying if not empty
        """
        if data.empty:
            raise ValueError("data can not be empty")
        
        return False

    @staticmethod
    def _check_instance(data, inst=pd.DataFrame):
        """
        Check if the input instance is correct
        
        args:
        -----------
            data (DataFrame):  dataset to check
            isnt (instance): specify the instance
        return:
        -----------     
            (boolen): boolean variable specifying if it is the same instance  
        """
        if not isinstance(data, inst):
            warning.warn("data is not of type pandas.DatFrame")

        return True
    
    @staticmethod
    def _check_type(data, type="numeric"):
        """
        Check if the input type is correct

         args:
        -----------
            data (DataFrame or Series):  data to check 
            type (str): specify the type

        return:
        -----------     
            ctype (boolen): boolean variable specifying if it is the right type       
        """
        if type == "numeric":
            ctype = is_numeric_dtype(data)
        
        if ctype:
            return ctype
        else:
            raise ValueError("data is not the right type") 

    def summary(self):
        """
        Calculate the general characteristics of the dataset

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
        if self.data is None:
            raise ValueError("there is not any data")

        options = {"empty": True,
                   "instance": pd.DataFrame}
        
        self._check(self.data, **options)

        summary = {"Number of features": self.data.shape[1],
                   "Number of observations": self.data.shape[0],
                   "Missing Values (Num)": self.get_missingValues(self.data),
                   "Missing Values (%)": round((self.get_missingValues(self.data) \
                                                / self.data.size)*100, 2),
                   "Duplicate rows (Num)": self.get_duplicates(self.data).shape[0],
                   "Duplicate rows (%)": round((self.get_duplicates(self.data)['count'].sum() \
                                                / self.data.size)*100, 2),
                   "Zeros values (Num)": self.get_zerosValues(self.data),
                   "Zeros values (%)": round((self.get_zerosValues(self.data) / self.data.size) * 100, 2), 
                   "Total size in memory (Kb)": self.data.memory_usage().sum()/1000,
                   "Average observations size in memory (B)": self.data.memory_usage().sum()/self.data.shape[0],
                   "Features numerica": self.data.select_dtypes(include=['number']).shape[1],
                   "Features Categorical": self.data.select_dtypes(include=["object", "category"]).shape[1],
                   "Features datetimes": self.data.select_dtypes(include=["datetime"]).shape[1]}

        self.description = pd.Series(summary)

        return self.description

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
    def get_missingValues(df):
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
    def get_zerosValues(df):
        """
        Calculate the number of values equal to zero

        args:
        -----------
            df (DataFrame or Series): dataset that will be checked for values equal to zero

        return:
        -----------
            (int): integer value with the number of values equal to 0
        """
        return df.size - np.count_nonzero(df.values)

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
        statistics = {key: self.get_numericStats(self.data[key]) for key in features}    

        return pd.DataFrame(statistics)

    def get_numericStats(self, serie):
        """
        Calculates the basic statistics of the specified feature

        args:
        -----------
            series (Series): data series to which the statistics will be calculated

        return:
        -----------
            numeric_stats (dict): statistics of the selected feature 
        """

        options = {"empty": True,
                   "instance": pd.Series,
                   "type": "numeric"}
        
        self._check(serie, **options)

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