import pandas as pd
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

        """
        kwargs_default = {"empty": None,
                          "isntance": None}

        options = {key: kwargs[key] if key in kwargs.keys() else kwargs_default[key] for key in kwargs_default.keys()}
        
        if options["empty"]:
            self._check_empty(data)

        if options["instance"]:
            self._check_instance(data, inst=options["instance"])

    @staticmethod
    def _check_empty(data):
        """

        """
        if data.empty:
            raise ValueError("data can not be empty")
        
        return True

    @staticmethod
    def _check_instance(data, inst=pd.DataFrame):
        """
        """
        if not isinstance(data, inst):
            warning.warn("data is not of type pandas.DatFrame")

        return True

    def describe(self):
        """
        args:

        return:

        """

        if self.data is None:
            raise ValueError("there is not any data")

        options = {"empty": True,
                   "instance": pd.DataFrame}

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
        args:

        return:
        """
        duplicates = df[df.duplicated(subset=columns, keep=False)].groupby(
                        df.columns.values.tolist()).size().reset_index(name="count"). \
                        sort_values("count", ascending=False)
        
        return duplicates

    @staticmethod
    def get_missingValues(df):
        """
        args:

        return:
        """
        if type(df) == type(pd.DataFrame()):
            return df.isnull().sum().sum()
        
        elif type(df) == type(pd.Series()):
            return df.isnull().sum()

    @staticmethod
    def get_zerosValues(df):
        """
        args:

        return:
        """
        return df.size - np.count_nonzero(df.values)

        
