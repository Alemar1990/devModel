class Correlation():
    """
    """

    def __init__(self):
        pass

    def get_correlation(self, df, method="Pearson", cols=None, **kwargs):
        """
        """
        kwargs_default = {"p_value": None}       
        options = {key: kwargs[key] if key in kwargs.keys() else kwargs_default[key] for key in kwargs_default.keys()}

        function_call = "_Correlation__{}".format(method.lower())
        if hasattr(self, function_call):
            function_corr = getattr(self, function_call)

        return function_corr(df, cols=cols, **options)
    
    def __pearson(self, df, cols=None, **kwargs):
        """
        """
        if kwargs['p_value']:
            corr = None
        else:
            corr = df.corr('pearson') if cols is not None else df[cols].corr('pearson')

        return corr

    def __spearman(self, cols=None, **kwargs):
        """
        """
        return df.corr('spearman') if cols is not None else df[cols].corr('spearman')
        
    
    def __kendall(self, cols=None, **kwargs):
        """
        """
        return df.corr('kendall') if cols is not None else df[cols].corr('kendall')
    
    def __cramerv(self, cols=None, **kwargs):
        """
        """
        pass