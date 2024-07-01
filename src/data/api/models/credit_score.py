from src.data.api.boilerplate import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd


class CreditScore(Dataset):
    
    """ Dataset for Credit Score """
    
    path = 'src/data/clean/credit-score.csv'
    sep = ','

    y = 'Credit_Score'
    
    # Removed 'Unnamed: 0', 'Credit_Mix'
    numerical_features = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary',
       'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
       'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
       'Num_Credit_Inquiries', 'Outstanding_Debt',
       'Credit_Utilization_Ratio', 'Credit_History_Age', 'Total_EMI_per_month',
       'Amount_invested_monthly', 'Monthly_Balance',
       
       'Credit-Builder Loan', #binary
       'Personal Loan', 'Debt Consolidation Loan', 'Student Loan', #binary
       'Payday Loan', 'Mortgage Loan', 'Auto Loan', 'Home Equity Loan' #binary
       ]
    
    categorical_features = ['Payment_Behaviour', 'Occupation', 'Month']
    
    categorical_encodings = {
        'Payment_Behaviour': [
            'Payment_Behaviour_High_spent_Medium_value_payments',
            'Payment_Behaviour_High_spent_Small_value_payments',
            'Payment_Behaviour_Low_spent_Large_value_payments',
            'Payment_Behaviour_Low_spent_Medium_value_payments',
            'Payment_Behaviour_Low_spent_Small_value_payments'
        ],
        'Occupation': [
            'Occupation_Architect',
            'Occupation_Developer', 'Occupation_Doctor', 'Occupation_Engineer',
            'Occupation_Entrepreneur', 'Occupation_Journalist', 'Occupation_Lawyer',
            'Occupation_Manager', 'Occupation_Mechanic', 'Occupation_Media_Manager',
            'Occupation_Musician', 'Occupation_Scientist', 'Occupation_Teacher',
            'Occupation_Writer', 'Payment_of_Min_Amount_Yes'
        ],
        
        'Month': ['Month_August', 'Month_February', 'Month_January', 'Month_July',
       'Month_June', 'Month_March', 'Month_May']
        
    }
    
    continous_features = [ #DICE also counts 'hours per week' as continuous for example
        'Age', 'Annual_Income', 'Monthly_Inhand_Salary',
       'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
       'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
       'Num_Credit_Inquiries', 'Outstanding_Debt',
       'Credit_Utilization_Ratio', 'Credit_History_Age', 'Total_EMI_per_month',
       'Amount_invested_monthly', 'Monthly_Balance'
       ]
       
    
    def train_test_split(self, x_res, y_res):
        
        X_train, X_test, y_train, y_test = train_test_split(x_res, y_res, test_size=0.3, random_state=42,stratify=y_res)
        return X_train, X_test, y_train, y_test
    