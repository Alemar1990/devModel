import warnings
from pandas.api.types import is_numeric_dtype
import pandas as pd

class Utilities:

    def __init__(self):
        pass
    
    @staticmethod
    def check_default_kwargs(dict_default, dict_):
        options = {key: dict_[key] if key in dict_.keys() else dict_default[key] for key in dict_default.keys()}
    
        return options

    @classmethod
    def _check(cls, data, inst, **kwargs):
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
                          "type": None,
                          "attribute": None,
                          "Nan": None}

        options = cls.check_default_kwargs(kwargs_default, kwargs)
        
        if options["empty"]:
            cls._check_empty(data)

        if options["instance"]:
            cls._check_instance(data, inst=options["instance"])
        
        if options["type"]:
            cls._check_type(data, type=options["type"])
        
        if options["attribute"]:
            for attribute, method in options["attribute"].items():
                cls._check_attribute(inst, attribute, method)
        
        if options["Nan"]:
            cls._check_nan(data)

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
            warnings.warn("data is not of type {}".format(inst))
            return False
        return True
    
    @staticmethod
    def _check_type(data, type="numeric"):
        """
        Checks if the input type is correct

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

    @staticmethod
    def _check_attribute(instance, attribute, method):
        """
        Checks if the atributte exists and if not it creates it

        args:
        -----------
             atribute (str): specify the attribute

        return:
        -----------     
            (boolen): boolean variable specifying the attribute exists  
        """
        if hasattr(instance, attribute):
            if getattr(instance, attribute) is not None:
                return True
            else:
                function = getattr(instance, method)
                function()
            return True
        else:
            raise ValueError("the attribute does not exist") 

    @staticmethod
    def _check_nan(data):
        """
        Checks if the data has missing values

        args:
        -----------
             data (DataFrame): data to check

        return:
        -----------     
            (boolen): boolean variable specifying if the data has missing values   
        """
        if len(data.columns.tolist()) > 1:
            if data.isna().sum().sum() > 0:
                warnings.warn("data has missing values")
                return True
        if len(data.columns.tolist()) == 1:          
            if (data.isna().sum() > 0):
                warnings.warn("data has missing values")
                return True       
      
        return False 