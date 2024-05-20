from data.boilerplate import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd

class CreditScore(Dataset):
    
    """ Dataset for Credit Score """
    
    path = 'data/clean/credit-score.csv'
    sep = ','

    y = 'Credit_Score' 
    
    def train_test_split(self, x_res, y_res):
        
        X_train, X_test, y_train, y_test = train_test_split(x_res, y_res, test_size=0.3, random_state=42,stratify=y_res)
        return X_train, X_test, y_train, y_test
    


class StudentAddiction(Dataset):
    
    """ Dataset for Student Addiction """
    
    path = 'data/clean/student-addiction-train.csv'
    path_test = 'data/clean/student-addiction-test.csv'
    sep = ','
    
    y = 'Addiction_Class' 
    
    def train_test_split(self, x_res, y_res):
        
        xy_test = pd.read_csv(self.path_test, sep=self.sep)
        
        X_test, y_test = xy_test.drop(columns=[self.y]), xy_test[self.y]
        X_train, y_train = x_res, y_res 
        
        return X_train, X_test, y_train, y_test
    

class Thyroid(Dataset):
    
    """ Dataset for Thyroid """
    
    path = 'data/clean/thyroid.csv'
    sep = ','
    
    data: pd.DataFrame
    y = '' 
    
    def train_test_split(self, x_res, y_res):
        X_train, X_test, y_train, y_test = train_test_split(x_res, y_res, test_size=0.3, random_state=42,stratify=y_res)
        return X_train, X_test, y_train, y_test
    