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
        kwargs_default = {"p_value": None,
                          "correction": True}       
        options = Utilities.check_default_kwargs(kwargs_default, kwargs)

        function_call = "_Correlation__{}".format(method.lower())
        if hasattr(self, function_call):
            function_corr = getattr(self, function_call)
        
        if (cols is not None) and (len(cols) > 1):
            df = df[cols]
        elif (cols is not None) and (len(cols) == 1):
            raise ValueError("cols has to be greater than one")
        
        df = df.dropna().reset_index(drop=True)
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

    def __spearman(self, df, **kwargs):
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
            
    def __kendall(self, df, **kwargs):
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
    
    def __cramerv(self, df, **kwargs):
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
        """
        rows = []
        features = df.select_dtypes(include=['category', 'object']).columns.tolist()
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

    def __cratio(self, df, **kwargs):
        """
        """

        categorical = df.select_dtypes(include=["category", "object"])
        numerical = df.select_dtypes(include="number")

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
        eta = 0.0 if numerator == 0 else np.sqrt(numerator / denominator)

        return eta