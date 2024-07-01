from src.data.api.boilerplate import Dataset
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class Thyroid(Dataset):
    
    """ Dataset for Thyroid """
    
    path = 'src/data/clean/hypothyroid.csv'
    sep = ','
    
    data: pd.DataFrame
    y = 'binaryClass' 
    
    
    numerical_features = [
        'age', 'sex', 'on thyroxine', 'query on thyroxine',
        'on antithyroid medication', 'sick', 'pregnant', 'thyroid surgery',
        'I131 treatment', 'query hypothyroid', 'query hyperthyroid', 'lithium',
        'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured', 'TSH',
       'T3 measured', 'T3', 'TT4 measured', 'TT4', 'T4U measured', 'T4U',
       'FTI measured', 'FTI', 'TBG measured']
       
    categorical_features = []
    
    continuous_features = ['age', 'sex','TSH', 'T3', 'TT4', 'T4U', 'FTI']
    
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        
        # Copied from https://www.kaggle.com/code/yasserhessein/thyroid-disease-detection-using-deep-learning/notebook
        
        # Map target variable to binary
        data["binaryClass"] = data["binaryClass"].map({"P":0,"N":1})
        
        # True = 1, False = 0 mappings
        data = data.map({"t":1,"f":0})
        
        data = data.map({"?":np.NAN})
        
        # Map sex
        data = data.map({"F":0,"M":1})
        
        cols = data.columns[data.dtypes.eq('object')]
        data[cols] = data[cols].apply(pd.to_numeric, errors='coerce')
        
        # imputing missing values
        #data['sex'].fillna(data['sex'].mean(), inplace=True)
        data['T4U measured'].fillna(data['T4U measured'].mean(), inplace=True)
        data['age'].fillna(data['age'].mean(), inplace=True)
        data['sex'].fillna(data['sex'].median(), inplace=True)
        data['TBG'].fillna(data['TBG'].mean(), inplace=True)
    

        imputer = SimpleImputer(strategy='mean')
        for col in ['TSH','T3','TT4','T4U','FTI']:
            data[col] = imputer.fit_transform(data[[col]])
            
        data['referral source'].fillna('unknown', inplace=True)
 
        # Encoding using pandas dummies
        
        data = pd.get_dummies(data, columns=['referral source'])
        
        
    
        
        
        
        data.drop('TBG', axis=1, inplace=True)
        
        return data
    
    def train_test_split(self, x_res, y_res):
        X_train, X_test, y_train, y_test = train_test_split(x_res, y_res, test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test
    