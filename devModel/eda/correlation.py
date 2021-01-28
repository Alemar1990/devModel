import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from devModel.utilities import Utilities

class Correlation():
    """
    """

    def __init__(self):
        pass

    #if __name__ == "__main__":
    #    df = df.dropna(inplace=True)

    def get_correlation(self, df, method="Pearson", cols=None, **kwargs):
        """
        """
        kwargs_default = {"p_value": None}       
        options = Utilities.check_default_kwargs(kwargs_default, kwargs)

        function_call = "_Correlation__{}".format(method.lower())
        if hasattr(self, function_call):
            function_corr = getattr(self, function_call)
        
        if (cols is not None) and (len(cols) > 1):
            df = df[cols]
        elif (cols is not None) and (len(cols) == 1):
            raise ValueError("cols has to be greater than one")
        
        df.dropna(inplace=True)
        return function_corr(df, **options)
    
    def __pearson(self, df, **kwargs):
        """
        """
        print(kwargs['p_value'])
        if kwargs['p_value']:
            features =  df.select_dtypes(include=['number']).columns.tolist()
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

    def __spearman(self, df, cols=None, **kwargs):
        """
        """
        df.dropna(inplace=True)
        if kwargs['p_value']:
            features =  df.select_dtypes(include=['number']).columns.tolist()
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
        
    
    def __kendall(self, df, cols=None, **kwargs):
        """
        """
        df.dropna(inplace=True)
        if kwargs['p_value']:
            features =  df.select_dtypes(include=['number']).columns.tolist()
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

        return df.corr('kendall') if cols is not None else df[cols].corr('kendall')
    
    def __cramerv(self, cols=None, **kwargs):
        """
        """
        pass