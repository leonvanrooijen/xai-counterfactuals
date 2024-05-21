from src.data.api.boilerplate import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd


class Thyroid(Dataset):
    
    """ Dataset for Thyroid """
    
    path = 'src/data/clean/thyroid.csv'
    sep = ','
    
    data: pd.DataFrame
    y = '' 
    
    def train_test_split(self, x_res, y_res):
        X_train, X_test, y_train, y_test = train_test_split(x_res, y_res, test_size=0.3, random_state=42,stratify=y_res)
        return X_train, X_test, y_train, y_test
    