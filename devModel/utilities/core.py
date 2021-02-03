import warnings

class Utilities:

    def __init__(self):
        pass
    
    @staticmethod
    def check_default_kwargs(dict_default, dict_):
        options = {key: dict_[key] if key in dict_.keys() else dict_default[key] for key in dict_default.keys()}
    
        return options
