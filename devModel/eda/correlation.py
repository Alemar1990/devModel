import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from scipy.stats import chi2_contingency
from scipy.stats import entropy
from sklearn import preprocessing
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
        
        df = df.dropna()
        return function_corr(df, **options)
    
    def __pearson(self, df, **kwargs):
        """
        """
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
    
    def __cramerv(self, df, cols=None, **kwargs):
        """
        """
        label = preprocessing.LabelEncoder()
        data_encoded = pd.DataFrame()
        features = df.select_dtypes(include=['category', 'object']).columns.tolist()

        #check if all values of a feature are differentes, that way is eliminated from features

        for feature in features:
            data_encoded[feature] = label.fit_transform(df[feature])

        rows = []
        for feature1 in data_encoded:
            col = []
            for feature2 in data_encoded:
                cramers = self.cramersV_correlation(data_encoded[feature1], 
                                                    data_encoded[feature2], correction=True)
                col.append(round(cramers,2))
            rows.append(col)

        cramersV_results = np.array(rows)
        corr = pd.DataFrame(cramersV_results, columns=data_encoded.columns, index=data_encoded.columns)

        return corr

    @staticmethod
    def cramersV_correlation(feature1, feature2, correction=True):
        """
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

        