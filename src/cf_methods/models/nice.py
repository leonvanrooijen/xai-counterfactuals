import src.cf_methods.cf_method as boilerplate
import nice as nice_lib
import pandas as pd
from pandas import DataFrame

class NICE(boilerplate.CounterfactualMethod):
    
    reward = None
    normalization = None
    prediction_function = None
    
    
    def set_data_restrictions(self,
                   numerical_features = [],
                   categorical_features = {}
                   ):
    
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
    
    def generate(self, X_test, y_test):
        
        if self.reward is None:
            raise ValueError('Reward function for NICE has not been set.')
        
        if self.normalization is None:
            raise ValueError('Normalization function for NICE has not been set.')
        
        if self.prediction_function is None:
            raise ValueError('Prediction function for NICE has not been set.')
        
        
        
        #convert X_train and y_train to numpy arrays
        #X_train = self.X_train.astype(int).to_numpy(copy=True)
        #y_train = self.y_train.astype(int).to_numpy(copy=True)
        
        y_train = self.df[self.target].to_numpy(copy=True)
        X_train = self.df.drop(self.target, axis=1).to_numpy(copy=True)
        
        print("Checkpoint A")
        # Checkpoint A is reached. Code works until this point
        # https://github.com/DBrughmans/NICE/blob/master/examples/NICE_adult.ipynb
        # According to the author, X_train/y_train should be numpy arrays
        
        print(X_train)
        
        
        
        features = range(0, 9)
        
        nooice = nice_lib.NICE(
            X_train=X_train,
            predict_fn=self.prediction_function,
            y_train=y_train,
            cat_feat=[],
            num_feat=features,
            distance_metric='HEOM', # 'HEOM' is the ONLY distance metric supported
            num_normalization=self.normalization,
            optimization=self.reward,
            justified_cf=True
        )
        
        #factual = X_test[0:1, :] #THIS IS THE FACTUAL
        
        print("checkpoint B")
        X_test_array = X_test[0]
        
        CF = nooice.explain(X_test_array)
        print("checkpoint C")
        print(CF)
        feature_names = self.df.drop(self.target, axis=1).columns
        print("Checkpoint D")
        return CF
       # return pd.DataFrame(data=[X_test, CF[0]], columns=feature_names)
    