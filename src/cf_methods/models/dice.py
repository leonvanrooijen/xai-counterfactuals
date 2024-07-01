import src.cf_methods.cf_method as boilerplate
import dice_ml
import pandas as pd
import numpy as np
import src.helpers.helpers as helpers

class DICE(boilerplate.CounterfactualMethod):
    
    model = None
    method = None
    backend = 'sklearn'
    
    diversity = 1
    proximity = 1
    sparsity = 1
    
    def set_data_restrictions(self, continuous_features = []):
        self.continuous_features = continuous_features
    
    
    def generate(self, X_test: pd.DataFrame, y_test: np.array):
        
        factual = X_test
        
        if(self.model is None):
            raise Exception("Classifier not set for DICE")
        
        if self.method is None:
            raise Exception("Method not set for DICE [random, genetic, kdtree]")
        
        # Get col_name of y_train
        y_col_name = self.target
        
        d = dice_ml.Data(dataframe=self.df,
                         continuous_features=[],
                         outcome_name=y_col_name)
        
        m = dice_ml.Model(model=self.model, backend=self.backend)
        
        print(y_col_name)
        print(self.diversity)
        print(self.proximity)
        print(self.sparsity)
        exp = dice_ml.Dice(d, m, method=self.method)

        dice_exp = exp.generate_counterfactuals(factual,
                                                total_CFs=1,
                                                desired_class="opposite",
                                                verbose=True,
                                                diversity_weight=1,
                                                proximity_weight=1,
                                                sparsity_weight=1)
        
        return helpers.visualise_changes(clf, d, encoder, method='DiCE', exp = dice_exp, factual=factual, scaler=scaler, only_changes=False)
        

        
        