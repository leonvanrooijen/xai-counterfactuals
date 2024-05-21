from src.data.api.boilerplate import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd


class StudentAddiction(Dataset):
    
    """ Dataset for Student Addiction """
    
    path = 'src/data/clean/student-addiction-train.csv'
    path_test = 'src/data/clean/student-addiction-test.csv'
    sep = ','
    
    y = 'Addiction_Class' 
    
    def train_test_split(self, x_res, y_res):
        
        xy_test = pd.read_csv(self.path_test, sep=self.sep)
        
        X_test, y_test = xy_test.drop(columns=[self.y]), xy_test[self.y]
        X_train, y_train = x_res, y_res 
        
        return X_train, X_test, y_train, y_test
    