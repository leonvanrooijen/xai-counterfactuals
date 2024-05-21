from pandas import DataFrame
from imblearn.over_sampling import SMOTE
import pandas as pd

class Dataset:
    
    """Boilerplate class for datasets
    
    Attributes:
        path: path to the dataset
        sep: separator used in the dataset
        data: the dataset in a pandas DataFrame
        random_state: random state for reproducibility
        y: target variable
        
    Methods:
        get_data: returns the entire dataset
        
        getX: returns the features
        getY: returns the target variable
        __getitem__: returns either the features or the target variable
        
        get_labels: returns the unique values of the target variable
        get_summary: returns a summary of the dataset
    """
    
    path: str
    sep: str
    data: DataFrame
    random_state: int
    y: str
    
    def __init__(self, smote: str | bool = False, random_state: int = 42):
        
        self.data = pd.read_csv(self.path, sep=self.sep)
        self.random_state = random_state
        
        if smote not in [False, None]:
            self.data = self.__smote(smote)
            
    def __smote(self, smote: str) -> DataFrame:
        
        smote = SMOTE(sampling_strategy=smote, random_state=self.random_state)
        
        X, y = smote.fit_resample(self['X'], self['y'])
        return pd.concat([X, y], axis=1)
        
    def get_data(self) -> DataFrame:
        return self.data
    
    def get_summary(self) -> DataFrame:
        return self.data.describe()
    
    def __getitem__(self, key) -> DataFrame:
        
        if key.lower() == 'x':
            return self.getX()
        
        if key.lower() == 'y':
            return self.getY()
        
        raise KeyError("Key not found")
    
    def get_labels(self) -> list:
        return self.getY().unique()
        
    def getX(self) -> DataFrame:
        # Note that .drop() does not modify the original DataFrame
        return self.data.drop(columns=[self.y])
    
    def getY(self) -> DataFrame:
        return self.data[self.y]