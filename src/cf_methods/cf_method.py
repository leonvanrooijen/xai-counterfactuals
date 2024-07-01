from pandas import DataFrame

class CounterfactualMethod:
    def __init__(self, df: DataFrame):
        self.df = df
    
    def set_params(self, **params):
        pass
    
    def generate():
        pass
    
    
    def set_params(self, **params):
        
        # Set all parameters for the model as attributes
        for key, value in params.items():
            setattr(self, key, value)
        
    