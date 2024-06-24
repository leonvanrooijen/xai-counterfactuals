import src.cf_methods.cf_method as boilerplate
import dice_ml

class DICE(boilerplate.CFMethod):
    
    
    def set_params(self, **params):
        
        
    
    def generate():
        
        d = dice_ml.Data(self.dataframe) # Assuming the dataframe is already loaded
        m = dice_ml.Model(model=None)
        
        dice = dice_ml.Dice(d, m, method, continuous_features, outcome_name)
        
    