import src.cf_methods.cf_method as boilerplate
import dice_ml
import pandas as pd
from pandas import DataFrame

class NICE(boilerplate.CFMethod):
    
    
    def set_data_restrictions(self,
                   real_features = [],
                   binary_features = [],
                   integer_features = [],
                   categorical_encodings = {},
                   only_positive_features = [],
                   only_increasing_features = [],
                   immutable_features = [],q
                   conditionally_mutable_features = []
                   ):
    
        self.real_features = real_features
        self.binary_features = binary_features
        self.integer_features = integer_features
        
        self.categorical_encodings = categorical_encodings
        #e.g. {'gender': ['gender_female', 'gender_male', 'gender_non-binary]}
        
        self.only_positive_features = only_positive_features
        self.only_increasing_featues = only_increasing_features
        
        self.immutable_features = immutable_features
        self.conditionally_mutable_features = conditionally_mutable_features
    
    def generate(factual: DataFrame):
        
        nooice = NICE(
            X_train=X_train,
            predict_fn=predict_fn,
            y_train=y_train,
            cat_feat=self.cateco,
            num_feat=num_feat,
            distance_metric='HEOM',
            num_normalization='minmax',
            optimization='proximity',
            justified_cf=True
        )
        
        #factual = X_test[0:1, :] #THIS IS THE FACTUAL
        
        CF = nooice.explain(factual)
        
        return pd.DataFrame(data=[X_test[0:1, :][0], CF[0]], columns=feature_names)
    